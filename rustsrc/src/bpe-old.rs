// use std::{borrow::BorrowMut, collections::{BinaryHeap, HashMap, HashSet}};
// use indicatif::ProgressBar;
// use regex::Regex;
// use rayon::prelude::*;
// use itertools::Itertools;

// const PRE_TOKENIZER: &'static str = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

// struct BPE {
//     vocab: HashMap<usize, String>,
//     merges: Vec<(String, String)>,
// }

// struct Word {
//     ids: Vec<u32>,
// }

// #[derive(Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
// struct Pair(u32, u32);

// #[derive(Debug, Eq)]
// struct Merge {
//     pair: Pair,
//     count: u64,
//     pos: HashSet<usize>,
// }

// impl Word {
//     fn merge(&mut self, pair: Pair, new: u32) -> Vec<(Pair, i32)> {
//         let mut i = 0;
//         let mut count_changes: Vec<(Pair, i32)> = vec![];
//         // let remove_elements = vec![];
//         while i < self.ids.len() {
//             if self.ids[i] == pair.0 && i + 1 < self.ids.len() && self.ids[i + 1] == pair.1 {
//                 // Remember the changes to update counts
//                 // count_changes.push((pair, -1)); // Removed the pair, but this is known to caller
//                 if i > 0 {
//                     count_changes.push((Pair(self.ids[i - 1], pair.0), -1));
//                     count_changes.push((Pair(self.ids[i - 1], new), 1));
//                 }
//                 if i < self.ids.len() - 3 {
//                     count_changes.push((Pair(pair.1, self.ids[i + 2]), -1));
//                     count_changes.push((Pair(new, self.ids[i + 2]), 1));
//                 }

//                 // Perform the changes
//                 self.ids[i] = new;
//                 self.ids.remove(i + 1);

//                 i += 1;
//             }
//         }
//         count_changes
//     }
// }

// fn count_pairs(words: &[Word], counts: &[u64]) -> (HashMap<Pair, u32>, HashMap<Pair, HashSet<usize>>) {
//     const CHUNK_SIZE: usize = 1_000;
//     words.par_chunks(CHUNK_SIZE).enumerate().map(|(chunk_i, words)| {
//         let idx_offset = chunk_i * CHUNK_SIZE;
//         let mut pair_counts = HashMap::new();
//         let mut update_loc = HashMap::new();
//         for (i, word) in words.iter().enumerate() {
//             for pair in word.ids.iter().copied().tuple_windows::<(u32, u32)>() {
//                 let pair = Pair(pair.0, pair.1);
//                 *pair_counts.entry(pair).or_insert(0) += 1;
//                 update_loc.entry(pair).or_insert_with(HashSet::new).insert(i + idx_offset);
//             }
//         }
//         (pair_counts, update_loc)
//     }).reduce(|| (HashMap::new(), HashMap::new()), |(mut pair_counts, mut update_loc), (other_pair_counts, other_update_loc)| {
//         for (pair, count) in other_pair_counts {
//             pair_counts.entry(pair).and_modify(|c| *c += count).or_insert(0);
//         }
//         for (pair, locs) in other_update_loc {
//             update_loc.entry(pair).and_modify(|s| s.extend(locs.into_iter())).or_insert(locs);
//         }
//         (pair_counts, update_loc)
//     })
// }

// pub fn train_bpe(input_string: &str, vocab_size: usize, special_tokens: &[&str]) -> BPE {
//     let re = Regex::new(PRE_TOKENIZER).unwrap();
//     let tokens_counts = re.find_iter(input_string).map(|m| m.as_str()).fold(HashMap::new(), |mut acc, token| {
//         *acc.entry(token).or_insert(0) += 1;
//         acc
//     });

//     let pbar = ProgressBar::new(vocab_size as u64);

//     let mut token_to_id: HashMap<String, u32> = HashMap::with_capacity(vocab_size);

//     let mut id_to_token: Vec<String> = Vec::with_capacity(vocab_size);

//     // Add special tokens
//     for &token in special_tokens {
//         if token_to_id.contains_key(token) {
//             continue;
//         }
//         token_to_id.insert(token.to_string(), token_to_id.len() as u32);
//         id_to_token.push(token.to_string());
//     }

//     // Add other tokens

//     // Count the frequency of each token
//     let (mut pair_counts, mut update_loc) = count_pairs(&words, &counts);

//     let words = todo!();

//     let mut queue: BinaryHeap<Merge> = BinaryHeap::new();

//     let mut merges = Vec::new();

//     loop {
//         if queue.is_empty() || token_to_id.len() >= vocab_size {
//             break;
//         }

//         let mut top: Merge = queue.pop().unwrap();

//         // Check if value is outdated, and if so skip and put the updated value back in the queue
//         if top.count != pair_counts[&top.pair] as u64 {
//             top.count = pair_counts[&top.pair] as u64;
//             queue.push(top);
//             continue;
//         }

//         if top.count < 1 {
//             break;
//         }   

//         let left = &id_to_token[top.pair.0 as usize];
//         let mut right = id_to_token[top.pair.1 as usize].clone();


//         // Create merged token
//         let new_token = format!("{}{}", left, right);

//         let new_token_id = token_to_id.get(&new_token).copied().unwrap_or_else(||{
//             let new_id = token_to_id.len() as u32;
//             token_to_id.insert(new_token.clone(), new_id);
//             id_to_token.push(new_token.clone());
//             new_id
//         });

//         merges.push((top.pair, new_token_id));

//         // Perform the merge in all words
//         let changes = top.pos.par_iter().flat_map(|&i| {
//             let word = &words[i] as *const _ as *mut Word;
//         })
//     }


//     pbar.finish();

//     BPE {
//         vocab: HashMap::new(),
//         merges: Vec::new(),
//     }

// }


// #[cfg(test)]
// mod tests {
//     use super::*;
// }