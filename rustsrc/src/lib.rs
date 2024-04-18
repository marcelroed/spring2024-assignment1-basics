mod bpe_train;
mod bpe;
mod utils;
use std::collections::HashMap;
use std::collections::HashSet;
use std::ops::Range;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict, PyBytes};
use itertools::Itertools;
use onig::Regex;
use regex::Regex as FastRegex;
use rayon::prelude::*;
use numpy::PyArray1;



/// Formats the sum of two numbers as string.
#[pyfunction]
fn train_bpe<'py>(py: Python<'py>, in_string: Bound<'py, PyBytes>, vocab_size: usize, special_tokens: Vec<String>) -> PyResult<(Bound<'py, PyDict>, Vec<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)>)> {
    println!("Started function");
    assert!(vocab_size <= 2_usize.pow(16), "vocab_size must be less than 2^16");
    let in_string = unsafe{std::str::from_utf8_unchecked(in_string.as_bytes())};

    // Train BPE
    let (vocab, merges) = bpe_train::train_bpe(in_string, vocab_size, special_tokens);

    // Convert vocab to Python
    let vocab_py = vocab.into_iter().map(|(k, v)| (k, PyBytes::new_bound(py, &v))).sorted_by(|e1, e2| Ord::cmp(&e1.0, &e2.0)).into_py_dict_bound(py);

    // Convert merges to Python
    let merges_py: Vec<_> = merges.into_iter().map(|(k, v)| (PyBytes::new_bound(py, &k), PyBytes::new_bound(py, &v))).collect();

    Ok((vocab_py, merges_py))
}


#[pyclass]
struct RustTokenizer {
    vocab: HashMap<u16, Vec<u8>>,
    vocab_inv_bytes: Vec<Option<u16>>,
    merges: HashMap<(u16, u16), u16>,
    special_tokens_inv: HashMap<Vec<u8>, u16>,
    special_regex: Option<FastRegex>,
    re: Regex,
}

#[pymethods]
impl RustTokenizer {
    #[new]
    fn __new__<'py>(vocab: HashMap<u16, Vec<u8>>, merges: Vec<(Vec<u8>, Vec<u8>)>, mut special_tokens: Vec<String>) -> Self {
        special_tokens.sort_by_key(|x| - (x.len() as isize));
        let special_regex = if special_tokens.is_empty() {None} else {
            Some(FastRegex::new(special_tokens.iter().map(|s| regex::escape(s.as_str())).join("|").as_str()).unwrap())
        };
        let mut vocab_inv_bytes = vec![None; 256];
        vocab.iter().for_each(|(&k, v)| {
            if v.len() == 1 {
                vocab_inv_bytes[v[0] as usize] = Some(k);
            }
        });

        let merges: HashMap<(u16, u16), u16> = merges.iter().map(|(e1, e2)|{
            let mut merged = e1.clone();
            merged.append(&mut e2.clone());
            let e1_token = vocab.iter().find(|(_, v)| *v == e1).unwrap().0;
            let e2_token = vocab.iter().find(|(_, v)| *v == e2).unwrap().0;
            let merged_token = vocab.iter().find(|(_, v)| *v == &merged).unwrap().0;
            ((*e1_token, *e2_token), *merged_token)
        }).collect();

        let vocab_inv = vocab.iter().map(|(k, v)| {
            let v = v.as_slice();
            (v.to_owned(), *k)
        }).collect::<HashMap<_, _>>();

        let special_tokens_inv = special_tokens.iter().map(|v| {
            let v = v.as_bytes();
            (v.to_owned(), vocab_inv[v])
        }).collect::<HashMap<_, _>>();

        Self {
            vocab,
            vocab_inv_bytes,
            merges,
            special_regex,
            special_tokens_inv,
            re: Regex::new(r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+").unwrap(),
        }
    }

    fn encode<'py>(&self, py: Python<'py>, text: Bound<'py, PyBytes>) -> PyResult<Bound<'py, PyArray1<u16>>> {
        if text.as_bytes().is_empty() {
            return Ok(PyArray1::from_vec_bound(py, vec![]));
        }
        let text = unsafe{std::str::from_utf8_unchecked(text.as_bytes())};
        let n_threads = if text.len() > 100_000 { rayon::current_num_threads()} else {1};
        let chunk_size = text.len().div_ceil(n_threads);

        let mut boundaries = vec![0];
        for i in 1..n_threads {
            let mut loc = i * chunk_size;
            while let None = &text.get(loc..) {
                loc += 1;
            }
            let loc = self.special_regex.as_ref().and_then(|r| r.find(&text[loc..])).map(|r| r.end() + loc).or_else(|| text[loc..].find(".\n").map(|x| x + 1 + loc));
            if let Some(loc) = loc {
                boundaries.push(loc); // Found a good place to chunk
            }
        }
        boundaries.push(text.len());

        let boundaries = boundaries.into_iter().collect::<HashSet<usize>>().into_iter().sorted().collect::<Vec<usize>>();
        let chunk_ranges: Vec<Range<usize>> = boundaries.into_iter().sorted().tuple_windows().map(|(start, end)| start..end).collect();

        println!("Min chunk size: {}, max chunk size: {}", chunk_ranges.iter().map(|r| r.len()).min().unwrap(), chunk_ranges.iter().map(|r| r.len()).max().unwrap());

        let words_chunks: Vec<_> = chunk_ranges.into_par_iter().map(|range| {
            let chunk = &text[range.clone()];
            let mut offset = 0;
            let mut tokens = vec![];
            if let Some(special_regex) = &self.special_regex {
                for snip in special_regex.find_iter(chunk) {
                    let text = &chunk[offset..snip.start()];
                    let encoding = bpe::encode(&self.re, &self.vocab_inv_bytes, &self.merges, text);
                    tokens.extend(encoding.into_iter());
                    let special_token = *self.special_tokens_inv.get(snip.as_str().as_bytes()).unwrap_or_else(|| panic!("Special token not found: {}", snip.as_str()));
                    tokens.push(special_token);
                    offset = snip.end();
                }
            }
            if offset < chunk.len() {
                let encoding = bpe::encode(&self.re, &self.vocab_inv_bytes, &self.merges, &chunk[offset..]);
                tokens.extend(encoding.into_iter());
            }
            tokens
        }).collect();

        println!("Gathering to a single vector");
        let words = crate::utils::parallel_concat(&words_chunks);

        println!("Assembling to numpy array");
        let words_arr = PyArray1::from_vec_bound(py, words);
        println!("Returning from Rust");

        Ok(words_arr)
    }

    fn decode<'py>(&self, _py: Python<'py>, tokens: Vec<u16>) -> PyResult<String> {
        let tokens = bpe::decode(&tokens, &self.vocab);
        let tokens = String::from_utf8_lossy(&tokens).into_owned();
        Ok(tokens)
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn rustsrc<'py>(_py: Python, m: &Bound<'py, PyModule>) -> PyResult<()> {
    // m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_class::<RustTokenizer>()?;
    m.add_function(wrap_pyfunction!(train_bpe, m)?)?;
    Ok(())
}

