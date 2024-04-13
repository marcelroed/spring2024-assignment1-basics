from typing import Self
from contextlib import contextmanager
import time
from pathlib import Path
from dataclasses import dataclass
from collections import Counter, defaultdict

from rustsrc import train_bpe

import regex as re

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

import pickle

@contextmanager
def timer(block_name):
    start = time.time()
    yield
    print(f'{block_name} took {time.time() - start} seconds')


@dataclass
class Tokenizer:
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]

    def merge(self):
        pass

    @classmethod
    def from_training_text(cls, text: str, vocab_size: int, special_tokens: list[str]) -> Self:
        # print('Starting regex')
        # pre_tokenized = re.findall(PAT, text)
        # print('Finished regex')

        # vocab, merges = train_bpe(pre_tokenized, vocab_size, special_tokens)
        with timer('Training the bpe model'):
            vocab, merges = train_bpe(text, vocab_size, special_tokens)
        # print('Finished training the bpe model!')


        return cls(vocab, merges)

        exit()

        special_tokens_ids = {tuple(bytes([i])): i for i, token in enumerate(special_tokens)}
        tokens_ids = {(i,): i for i in range(256)}
        tokens_ids.update(special_tokens_ids)

        pre_tokenized = [(special_tokens_ids[token],) if token in special_tokens_ids else tuple(token.encode('utf-8')) for token in pre_tokenized]

        print(pre_tokenized)


        counter = Counter(pre_tokenized)

        pair_counts = defaultdict(int)

        for k, v in counter.items():
            for i in range(len(k) - 1):
                pair = (k[i], k[i + 1])
                pair_counts[pair] += v
            
        while len(tokens_ids) < vocab_size:
            best_pair = max(pair_counts, key=pair_counts.get)
            new_id = len(tokens_ids)
            tokens_ids[best_pair] = new_id

        print(f'{pair_counts=}')
        
        # vocab = {i: bytes([i])  for i in range(256)}
        # vocab.update({i + 256: token.encode('utf-8') for i, token in enumerate(special_tokens)})

        # return cls({}, [])
        return None



if __name__ == '__main__':
    # print(re.findall(PAT, "some text that i'll pre-tokenize"))
    in_path = Path('data/TinyStoriesV2-GPT4-train.txt')
    # in_path = Path('data/owt_train.txt')
    with open(in_path, 'rb') as f:
        text = f.read()
        print('Finished reading')
        tokenizer = Tokenizer.from_training_text(text, 10_000 if 'TinyStories' in str(in_path) else 32_000, ['<|endoftext|>'])
        # print([v.decode('utf-8', errors='replace') for v in sorted(tokenizer.vocab.values(), reverse=True, key=len)])

        # print('Finished training')
        out_path = Path('tokenizers/')
        out_path.mkdir(exist_ok=True)
        with open(out_path / f'{in_path.stem}.pkl', 'wb') as f:
            pickle.dump(tokenizer, f)
        