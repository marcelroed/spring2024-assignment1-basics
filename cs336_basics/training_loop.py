from functools import partial
from typing import BinaryIO, IO, Iterable
from os import PathLike
from itertools import count
import argparse
from pathlib import Path
import time
from math import exp

import numpy as np
import torch
import torch.nn as nn
from numpy import typing as npt
from cs336_basics.transformer import Transformer
from cs336_basics.training import cross_entropy_loss, AdamW, gradient_clipping, cosine_with_warmup_lr_schedule
from cs336_basics.bpe import Tokenizer



def load_data(dataset: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    dataset_length = len(dataset)
    sample_range = dataset_length - context_length
    sample_idcs = np.random.randint(0, sample_range, batch_size)
    samples = np.stack([dataset[idx:idx + context_length] for idx in sample_idcs])
    samples_offset = np.stack([dataset[idx + 1:idx + context_length + 1] for idx in sample_idcs])
    return torch.tensor(samples, dtype=torch.long, device=device), torch.tensor(samples_offset, dtype=torch.long, device=device)

def data_generator(dataset: npt.NDArray, batch_size: int, context_length: int, device: str) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    while True:
        yield load_data(dataset, batch_size, context_length, device)

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | PathLike | BinaryIO | IO[bytes]):
    torch.save({
        "iteration": iteration,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, out)

def load_checkpoint(src: str | PathLike | BinaryIO | IO[bytes], model: nn.Module, optimizer: torch.optim.Optimizer) -> int:
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]


def parse_args():
    parser = argparse.ArgumentParser(description='Train transformer model')
    parser.add_argument('--checkpoint-interval', default=10_000, type=int)
    parser.add_argument('--log-interval', default=100, type=int)
    parser.add_argument('--checkpoint-path', default='checkpoints/', type=str)

    parser.add_argument('--dataset', default='owt', type=str, choices=['owt', 'tinystories'])

    parser.add_argument('--d_ff', default=2048, type=int)
    parser.add_argument('--total_steps', default=70_000, type=int)  # Just an estimate
    parser.add_argument('--d_model', default=512, type=int)
    parser.add_argument('--num_heads', default=16, type=int)
    parser.add_argument('--num_layer', default=10, type=int)
    parser.add_argument('--context_length', default=512, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--parallel_layers', default=False, type=bool)
    parser.add_argument('--post_norm', default=False, type=bool)
    parser.add_argument('--name', default=None)
    parser.add_argument('--rotary', default=True, type=bool)
    parser.add_argument('--activation', default='silu', type=str, choices=['gelu', 'silu'])
    parser.add_argument('--use_gated_mlp', default=True, type=bool)
    parser.add_argument('--tie_embeddings', default=True, type=bool)
    parser.add_argument('--decay', default=1.0, type=float)
    parser.add_argument('--use_sophia', default=False, type=bool)
    parser.add_argument('--flash', default=False, type=bool)
    
    parser.add_argument('--compile', default=True, type=bool)

    return parser.parse_args()


def train_model():
    import wandb
    from tqdm.auto import tqdm
    args = parse_args()

    vocab_size = 32_000 if args.dataset == 'owt' else 10_000
    model = Transformer(tie_embeddings=args.tie_embeddings, use_gated_mlp=args.use_gated_mlp, vocab_size=vocab_size, activation=args.activation, context_length=args.context_length, num_layers=args.num_layer, d_model=args.d_model, num_heads=args.num_heads, d_ff=args.d_ff, parallel_layers=args.parallel_layers, post_norm=args.post_norm, device='cuda', use_flash=args.flash, use_rotary_embeddings=args.rotary)

    if args.compile:
        model = torch.compile(model, fullgraph=True)
    
    if args.use_sophia:
        from sophia import SophiaG
        optimizer = SophiaG(model.parameters(), lr=2e-4, betas=(0.965, 0.99), rho = 0.01, weight_decay=1e-1)
    else:
        optimizer = AdamW(model.parameters(), lr=args.lr)

    loss_func = torch.compile(cross_entropy_loss, fullgraph=True)

    train_dataset = np.memmap(f'data/{args.dataset}_train_tokens.npy', dtype=np.uint16, mode='r')
    valid_dataset = np.memmap(f'data/{args.dataset}_valid_tokens.npy', dtype=np.uint16, mode='r')

    checkpoints_dir = Path(args.checkpoint_path); checkpoints_dir.mkdir(exist_ok=True)
    
    # Format time to be used in checkpoint filenames
    time_str = time.strftime('%Y-%m-%d_%H-%M-%S')

    schedule_end_time = args.total_steps
    warmup_iters = int(schedule_end_time * 0.05)
    lr_schedule = partial(cosine_with_warmup_lr_schedule, max_learning_rate=args.lr, min_learning_rate=args.lr * args.decay, warmup_iters=warmup_iters, cosine_cycle_iters=schedule_end_time)

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        model.train()
        section_training_loss = (0.0, 0)
        try:
            pbar = tqdm(data_generator(train_dataset, batch_size=args.batch_size, context_length=args.context_length, device='cuda'))
            for training_step, (train_x, train_y) in enumerate(pbar):
                # Maybe fix crashes that happen very seldomly
                train_x = torch.minimum(train_x, torch.tensor(vocab_size - 1, device='cuda'))
                train_y = torch.minimum(train_y, torch.tensor(vocab_size - 1, device='cuda'))
                lr_value = lr_schedule(training_step)
                optimizer.zero_grad()
                optimizer.param_groups[0]['lr'] = lr_value
                y_pred = model(train_x)
                training_loss = loss_func(y_pred, train_y)
                training_loss.backward()
                gradient_clipping(model.parameters(), 1.0)
                optimizer.step()
                section_training_loss = (section_training_loss[0] + training_loss.item(), section_training_loss[1] + 1)
                del training_loss
                if training_step % args.checkpoint_interval == 0 and training_step != 0:
                    save_checkpoint(model, optimizer, training_step, checkpoints_dir / f'{wandb.run.name}_{time_str}_{training_step // 1000}k')
                if training_step % args.log_interval == 0:
                    model.eval()
                    # Compute validation loss
                    valid_x, valid_y = load_data(valid_dataset, batch_size=args.batch_size, context_length=args.context_length, device='cuda')
                    with torch.no_grad():
                        valid_loss = loss_func(model(valid_x), valid_y)
                    training_loss_avg = section_training_loss[0] / section_training_loss[1]
                    section_training_loss = (0.0, 0)
                    if training_step == 0:
                        wandb.init(
                            project="cs336-assignment-1", entity="marcelroed", config=vars(args),
                            name=args.name,
                        )
                    wandb.log({'loss/train': training_loss_avg}, step=training_step)
                    wandb.log({'perplexity/train': torch.exp(torch.tensor(training_loss_avg))}, step=training_step)
                    wandb.log({'loss/valid': valid_loss.item()}, step=training_step)
                    wandb.log({'perplexity/valid': torch.exp(valid_loss).item()}, step=training_step)
                    wandb.log({'lr': optimizer.param_groups[0]['lr']}, step=training_step)
                    pbar.set_postfix({'train_loss': training_loss_avg, 'valid_loss': valid_loss.item()})
                    model.train()
        except KeyboardInterrupt:
            pass




if __name__ == '__main__':
    train_model()
