import numpy as np

def train_bpe(in_string: bytes, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    ...


class RustTokenizer:
    def __new__(cls, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None) -> 'RustTokenizer':
        ...

    def encode(self, text: bytes) -> np.typing.NDArray[np.uint16]:
        ...
    
    def decode(self, tokens: np.typing.NDArray[np.uint16]) -> str:
        ...



def pretokenize(text: bytes) -> PretokenizerIterator: