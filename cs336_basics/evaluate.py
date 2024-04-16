import torch
import torch.nn as nn

from cs336_basics.training import AdamW
from cs336_basics.transformer import Transformer, softmax
from cs336_basics.bpe import Tokenizer
from cs336_basics.training_loop import load_checkpoint, parse_args


def softmax_with_temperature(dist, temperature: float):
    return softmax(dist / temperature, dim=-1)


def generate(model: Transformer, tokenizer: Tokenizer, prompt: str, max_length: int, temperature: float, p: float = 0.9):
    tokens = tokenizer.encode(prompt)
    print(f'Generating from tokens:')
    decoded = ''
    model.eval()
    with torch.no_grad():
        while len(tokens) < max_length and not decoded.endswith('<|endoftext|>'):
            input_tensor = torch.tensor([tokens], dtype=torch.long, device=model.ln_final.weight.device)
            logits = model(input_tensor)
            logits = logits[0, -1]
            probs = softmax_with_temperature(logits, temperature)
            if p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
                cumulative_sorted_probs = torch.cumsum(sorted_probs, dim=-1)
                nucleus = cumulative_sorted_probs < 0.9
                nucleus[0] = nucleus[0] | (~nucleus.any())
                if not nucleus.any():
                    nucleus[0] = True
                non_nucleus_indices = sorted_indices[~nucleus]
                probs[non_nucleus_indices] = 0.0
                # Renormalize the probabilities
                # print(probs.sum())
                probs /= probs.sum()

            next_token = torch.multinomial(probs, 1).item()
            tokens.append(next_token)
            decoded = tokenizer.decode(tokens)
            print(decoded)
        return decoded


def main():
    args = parse_args()
    args.flash = True
    model = torch.compile(Transformer(vocab_size=args.d_vocab_size, context_length=args.context_length, num_layers=args.num_layer, d_model=args.d_model, num_heads=args.num_heads, d_ff=args.d_ff, device='cuda', use_flash=args.flash), dynamic=True)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    # load_checkpoint('checkpoints/smart-dream-46_2024-04-16_03-09-37_20k', model, optimizer)
    load_checkpoint('checkpoints/reference_2024-04-16_07-10-56_60k', model, optimizer)
    # tokenizer = Tokenizer.from_files('tokenizers/TinyStoriesV2-GPT4-train_vocab.pkl', 'tokenizers/TinyStoriesV2-GPT4-train_merges.pkl', special_tokens=['<|endoftext|>'])
    tokenizer = Tokenizer.from_files('tokenizers/owt_train_vocab.pkl', 'tokenizers/owt_train_merges.pkl', special_tokens=['<|endoftext|>'])
    result = generate(model, tokenizer, 'Her head exploded, and she', 512, 1)


if __name__ == '__main__':
    main()