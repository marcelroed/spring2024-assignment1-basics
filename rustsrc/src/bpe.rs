use priority_queue::PriorityQueue;
use std::{collections::{HashMap, HashSet}, hint::black_box, sync::{atomic::AtomicU64, Mutex, RwLock}};
use itertools::Itertools;
use rayon::prelude::*;
use indicatif::ProgressBar;
use dashmap::DashMap;
use std::sync::Arc;


#[derive(Clone)]
pub struct Word {
    pub symbols: Vec<u32>,
    pub word_count: isize,
}

type Pair = (u32, u32);


pub fn count_words(words: &[&str]) -> Vec<Word> {
    let num_words = words.len();
    let num_threads = rayon::current_num_threads();
    let chunk_size = num_words.div_ceil(num_threads);

    let all_counts = words.chunks(chunk_size).par_bridge().map(|chunk| {
        let mut thread_word_counts = HashMap::new();
        for &word in chunk {
            *thread_word_counts.entry(word).or_insert(0) += 1;
        }
        thread_word_counts
    }).reduce(|| HashMap::new(), |mut acc, thread_word_counts| {
        for (word, count) in thread_word_counts {
            *acc.entry(word).or_insert(0) += count;
        }
        acc
    });

    all_counts.into_iter().map(|(word, count)| {
        Word {
            symbols: word.as_bytes().into_iter().map(|e| *e as u32).collect(),
            word_count: count,
        }
    }).collect()
}

fn count_pairs(words: &[Word]) -> (HashMap<Pair, isize>, HashMap<Pair, HashSet<u32>>) {
    let mut symbol_counts: HashMap<Pair, isize> = HashMap::new();
    let mut affects_word: HashMap<Pair, HashSet<u32>> = HashMap::new();
    for (word_i, word) in words.iter().enumerate() {
        for i in 0..word.symbols.len() - 1 {
            let pair = (word.symbols[i], word.symbols[i + 1]);
            let count = symbol_counts.entry(pair).or_insert(0);
            *count += word.word_count;

            affects_word.entry(pair).or_insert(HashSet::new()).insert(word_i as u32);
        }
    }
    (symbol_counts, affects_word)
}

fn update_word(w: &mut Word, pair: Pair, new_symbol: u32) -> (Vec<(Pair, isize)>, Vec<Pair>, Vec<Pair>) {
    let mut i = 0;
    let mut count_changes = vec![];
    let old_affects_this_word = w.symbols.iter().tuple_windows().map(|(a, b)| (*a, *b)).collect::<HashSet<Pair>>();
    while i < w.symbols.len() - 1 {
        if w.symbols[i] == pair.0 && w.symbols[i + 1] == pair.1 {
            // Perform the merge
            // count_changes.push((pair, -1)); // This one was removed from the priority queue
            if i >= 1 {
                count_changes.push(((w.symbols[i - 1], pair.0), - w.word_count));
                count_changes.push(((w.symbols[i - 1], new_symbol), w.word_count));
            }
            if w.symbols.len() >= 3 && i <= w.symbols.len() - 3 {
                count_changes.push(((pair.1, w.symbols[i + 2]), -w.word_count));
                count_changes.push(((new_symbol, w.symbols[i + 2]), w.word_count));
            }
            w.symbols[i] = new_symbol;
            w.symbols.remove(i + 1);
        }
        i += 1;
    }
    let new_affects_this_word = w.symbols.iter().tuple_windows().map(|(a, b)| (*a, *b)).collect::<HashSet<Pair>>();
    let added_affects = new_affects_this_word.difference(&old_affects_this_word).copied().collect::<Vec<Pair>>();
    let removed_affects = old_affects_this_word.difference(&new_affects_this_word).copied().collect::<Vec<Pair>>();
    (count_changes, added_affects, removed_affects)
}

fn update_words(words: &mut [Word], affects_word: RwLock<HashMap<Pair, Mutex<HashSet<u32>>>>, pair: Pair, new_symbol: u32) -> Vec<(Pair, isize)> {
    let count_changes: RwLock<HashMap<(u32, u32), AtomicU64>> = RwLock::new(HashMap::new());
    let affects_word_l = RwLock::new(affects_word);

    // let mut affects_changes = HashMap::new();

    let affected_words: Vec<u32> = affects_word.read().unwrap().get(&pair).unwrap().lock().unwrap().iter().copied().collect::<Vec<_>>();
    let chunks = words.chunks_mut(1).enumerate().filter(|(i, chunk)| affected_words.contains(&(*i as u32)));
    let affected_word_refs: Vec<(&mut Word, u32)> = chunks.map(|(word_i, wordrefs)| (&mut wordrefs[0], word_i as u32)).collect();
    let changes: Vec<_> = affected_word_refs.into_par_iter().map(|(word, word_i)| {
        // let word = &mut words[word_i as usize];
        let (count_changes_word, added_affects, removed_affects) = update_word(word, pair, new_symbol);

        for affect in added_affects {
            black_box(dummy)
        }

        // (word_i, added_affects, removed_affects, count_changes_word)
    }).collect();
        // for (word_i, added_affects, removed_affects, count_changes_word) in changes {
        //     for affect in added_affects {
        //         affects_word.entry(affect).or_insert(HashSet::new()).insert(word_i);
        //     }
        //     for affect in removed_affects {
        //         affects_word.entry(affect).or_insert(HashSet::new()).remove(&word_i);
        //     }
        //     count_changes.extend(count_changes_word);
        // }

    count_changes
}

pub fn assemble_token(token: u32, symbols: &Vec<Vec<u8>>) -> String{
    symbols[token as usize].iter().map(|x| *x as char).collect::<String>()
}

pub fn train_bpe(mut words: Vec<Word>, vocab_size: usize, special_tokens: Vec<String>) -> (HashMap<u32, Vec<u8>>, Vec<(Vec<u8>, Vec<u8>)>) {
    // let num_merges = 12;
    // let max_symbols = 256 + num_merges;
    let max_symbols = vocab_size;
    // Will be initialized from outside
    // let words = vec![("low", 5), ("lower", 2), ("widest", 3), ("newest", 6)];
    // let mut words: Vec<Word> = words.into_iter().map(|(word, count)| Word { symbols: word.bytes().map(|x| x as u32).collect(), word_count: count }).collect();

    let (symbol_counts, mut affects_word) = count_pairs(&words);

    // Symbols 0 through 255 are unicode characters
    let mut symbols: Vec<Vec<u8>> = (0..=255).map(|x| vec![x]).collect();
    symbols.extend(special_tokens.into_iter().map(|x| x.bytes().collect::<Vec<u8>>()));

    let mut pq = PriorityQueue::new();
    symbol_counts.into_iter().for_each(|(pair, count)| {
        pq.push(pair, count);
    });

    let mut merges = vec![];

    println!("Starting merges");
    let bar = ProgressBar::new(max_symbols as u64).with_style(indicatif::ProgressStyle::default_bar().template("[{elapsed_precise}] [{bar}] {pos}/{len} ({eta})").unwrap());
    while !pq.is_empty() && symbols.len() < max_symbols {
        bar.set_position(symbols.len() as u64);
        // println!("pair counts: {:?}", pq);
        let pair = {
            let (first_pair, first_count) = pq.pop().unwrap();
            let mut tied_pairs = vec![first_pair];
            while let Some((_next_pair, &next_count)) = pq.peek() {
                if next_count != first_count { break; }
                tied_pairs.push(pq.pop().unwrap().0);
            }
            // Find the greatest pair lexicographically
            let mut greatest_pair = first_pair;
            let assemble_pair = |(p0, p1)| (assemble_token(p0, &symbols), assemble_token(p1, &symbols));

            for pair in tied_pairs.iter().copied() {
                if assemble_pair(pair) > assemble_pair(greatest_pair) {
                    greatest_pair = pair;
                }
            }

            for pair in tied_pairs {
                if pair != greatest_pair {
                    pq.push(pair, first_count);
                }
            }

            greatest_pair
        };

        // Merge the pair
        let new_symbol: Vec<u8> = vec![&symbols[pair.0 as usize], &symbols[pair.1 as usize]].iter().copied().flatten().copied().collect();
        // println!("Adding {:?}", String::from_utf8(new_symbol.clone()).unwrap_or("?".to_string()));
        merges.push((symbols[pair.0 as usize].clone(), symbols[pair.1 as usize].clone()));

        symbols.push(new_symbol);

        let (count_changes_vec, affects_word) = update_words(&mut words, affects_word, pair, symbols.len() as u32 - 1);
        // Group by the changed pair
        let mut count_changes: HashMap<Pair, isize> = HashMap::new();
        for (pair, change) in count_changes_vec {
            *count_changes.entry(pair).or_insert(0) += change as isize;
        }

        for (pair, change) in count_changes {
            let found_item = pq.change_priority_by(&pair, |p| *p += change);
            if !found_item {
                pq.push(pair, change);
            }
        }
    }
    bar.finish();
    // println!("pair counts: {:?}", pq);
    // eprintln!("Finished!");
    // println!("{:?}", symbols.iter().map(|v| String::from_utf8(v.clone()).unwrap_or_else(|_e| "?".to_string())).collect::<Vec<String>>());
    // println!("{:?}", words.iter().map(|w| String::from_utf8(w.symbols.iter().flat_map(|x| &symbols[*x as usize]).cloned().collect()).unwrap_or_else(|_e| "?".to_string())).collect::<Vec<String>>());

    let vocab: HashMap<_, _> = symbols.into_iter().enumerate().map(|(i, v)| (i as u32, v)).collect();

    (vocab, merges)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_train_bpe() {
        train_bpe();
    }
}