mod bpe;
mod utils;
use std::borrow::Cow;
use std::collections::HashSet;
use std::ops::Range;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict, PyList, PyListMethods, PyString, PyBytes, PyStringData};
use itertools::Itertools;
use onig::Regex;
use rayon::prelude::*;

use crate::utils::parallel_concat;



/// Formats the sum of two numbers as string.
#[pyfunction]
fn train_bpe<'py>(py: Python<'py>, in_string: Bound<'py, PyBytes>, vocab_size: usize, special_tokens: Vec<String>) -> PyResult<(Bound<'py, PyDict>, Vec<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)>)> {
    let n_threads = rayon::current_num_threads();
    println!("Started function");

    // let in_string = unsafe {
    //     match in_string.slice? {
    //         PyStringData::Ucs1(d) => {println!("UCS1"); Cow::Borrowed(std::str::from_utf8_unchecked(d))},
    //         PyStringData::Ucs4(d) => {
    //             println!("UCS4!");
    //             let ptr = d.as_ptr();
    //             let len = d.len();
    //             let slice = &*std::ptr::slice_from_raw_parts(ptr as *const u8, len * 4);
    //             Cow::Borrowed(std::str::from_utf8_unchecked(slice))
    //         },
    //         PyStringData::Ucs2(_d) => {println!("UCS2"); Cow::Owned(in_string.to_string())}
    //     }
    // };
    // let in_string = in_string.to_string();
    let in_string = unsafe{std::str::from_utf8_unchecked(in_string.as_bytes())};

    println!("Starting regex");
    // let words: Vec<&str> = Regex::new(r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+").unwrap().find_iter(in_string).map(|m| m.unwrap().as_str()).collect();

    // Rough chunk size for regex
    let chunk_size = in_string.len().div_ceil(n_threads);

    let mut boundaries = vec![0];

    for i in 1..n_threads {
        let mut loc = i * chunk_size;
        while loc < in_string.len() - 1 {
            if in_string.as_bytes()[loc] == b'.' && in_string.as_bytes()[loc + 1] == b'\n' {
                loc += 1;
                break;
            }
            loc += 1;
        }
        boundaries.push(loc);
    }
    boundaries.push(in_string.len());

    let boundaries = boundaries.into_iter().collect::<HashSet<usize>>().into_iter().sorted().collect::<Vec<usize>>();

    let chunk_ranges: Vec<Range<usize>> = boundaries.into_iter().sorted().tuple_windows().map(|(start, end)| start..end).collect();

    // println!("chunk range length {:?}", chunk_ranges.len());
    // println!("chunk ranges {:?}", chunk_ranges);
    // println!("widths: {:?}", chunk_ranges.iter().map(|r| r.end - r.start).collect::<Vec<usize>>());

    let re = Regex::new(r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+").unwrap();

    println!("Performing regex in parallel");
    let words: Vec<_> = chunk_ranges.into_par_iter().map(|range| {
        let chunk = &in_string[range.clone()];
        let words: Vec<&str> = re.find_iter(chunk).map(|m| &in_string[(&range.start + m.0)..(&range.start + m.1)]).collect();
        words
    }).collect();

    println!("Gathering to a single vector");

    // let words_refs = words.iter().map(|e| e.as_slice()).collect::<Vec<&[&str]>>();
    let words = parallel_concat(&words);

    // let words: Vec<&str> = words.into_par_iter().flatten().collect::<Vec<&str>>();

    // return Err(pyo3::exceptions::PyValueError::new_err("Not implemented"));

    println!("Counting the words");
    let words = bpe::count_words(&words);
    println!("{} words", words.len());

    println!("Starting merging");
    let (vocab, merges) = bpe::train_bpe(words, vocab_size, special_tokens);
    println!("Finished merging");


    println!("Converting vocab to Python");
    let vocab_py = vocab.into_iter().map(|(k, v)| (k, PyBytes::new_bound(py, &v))).sorted_by(|e1, e2| Ord::cmp(&e1.0, &e2.0)).into_py_dict_bound(py);

    println!("Converting merges to Python");
    let merges_py: Vec<_> = merges.into_iter().map(|(k, v)| (PyBytes::new_bound(py, &k), PyBytes::new_bound(py, &v))).collect();

    println!("Returning!");

    Ok((vocab_py, merges_py))
}

/// A Python module implemented in Rust.
#[pymodule]
fn rustsrc<'py>(_py: Python, m: &Bound<'py, PyModule>) -> PyResult<()> {
    // m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(train_bpe, m)?)?;
    Ok(())
}

