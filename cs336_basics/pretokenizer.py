import regex as re
from rustsrc import pretokenizer, pretokenized_counts
import time

PATTERN = re.compile(rb"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


with open('/home/marcel/data/TinyStoriesV2-GPT4-train.txt', 'rb') as f:
    bytes = f.read()

# text = bytes.decode('utf-8', errors='replace')

start_time = time.time()
# for token in pretokenizer(bytes):
#     pass
counts = pretokenized_counts(bytes)
print(f'Pretokenization took {time.time() - start_time} seconds')


start_time = time.time()
for match in PATTERN.finditer(bytes):
    pass
print(f'Regex pretokenization took {time.time() - start_time} seconds')

# for i, (token, regex_match) in enumerate(zip(pretokenizer(bytes), PATTERN.finditer(bytes))):
#     regex_token = regex_match.group(0)
#     assert token == regex_token, f'Error at index {i}: {token} != {regex_token}'