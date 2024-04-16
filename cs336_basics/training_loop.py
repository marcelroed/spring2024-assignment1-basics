from typing import BinaryIO, IO, Iterable
from os import PathLike
from itertools import count
import argparse
from pathlib import Path
import time

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

def load_checkpoint(src: str | PathLike | BinaryIO | IO[bytes], model: nn.Module, optimizer: torch.optim.Optimizer):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]


def parse_args():
    parser = argparse.ArgumentParser(description='Train transformer model')
    parser.add_argument('--checkpoint-interval', default=10_000, type=int)
    parser.add_argument('--log-interval', default=100, type=int)
    parser.add_argument('--checkpoint-path', default='checkpoints/', type=str)

    parser.add_argument('--d_ff', default=2048, type=int)
    parser.add_argument('--d_vocab_size', default=32_000, type=int)
    parser.add_argument('--d_model', default=512, type=int)
    parser.add_argument('--num_heads', default=16, type=int)
    parser.add_argument('--num_layer', default=4, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--context_length', default=512, type=int)
    
    parser.add_argument('--compile', default=True, type=bool)
    parser.add_argument('--flash', default=True, type=bool)

    return parser.parse_args()


def train_model():
    import wandb
    from tqdm.auto import tqdm
    wandb.init(project="cs336-assignment-1", entity="marcelroed")
    args = parse_args()
    model = Transformer(vocab_size=args.d_vocab_size, context_length=args.context_length, num_layers=args.num_layer, d_model=args.d_model, num_heads=args.num_heads, d_ff=args.d_ff, device='cuda', use_flash=args.flash)
    if args.compile:
        model = torch.compile(model)
    
    optimizer = AdamW(model.parameters(), lr=1e-4)

    train_dataset = np.memmap('data/owt_train_tokens.npy', dtype=np.uint16, mode='r')
    valid_dataset = np.memmap('data/owt_valid_tokens.npy', dtype=np.uint16, mode='r')

    checkpoints_dir = Path(args.checkpoint_path); checkpoints_dir.mkdir(exist_ok=True)
    
    # Format time to be used in checkpoint filenames
    time_str = time.strftime('%Y-%m-%d_%H-%M-%S')

    model.train()
    section_training_loss = (0.0, 0)
    try:
        for training_step, (train_x, train_y) in enumerate(tqdm(data_generator(train_dataset, batch_size=128, context_length=512, device='cuda'))):
            optimizer.zero_grad()
            y_pred = model(train_x)
            training_loss = cross_entropy_loss(y_pred, train_y)
            training_loss.backward()
            section_training_loss = (section_training_loss[0] + training_loss.item(), section_training_loss[1] + 1)
            optimizer.step()
            if (training_step + 1) % args.checkpoint_interval == 0:
                save_checkpoint(model, optimizer, training_step, checkpoints_dir / f'{time_str}_{training_step // 1000}k')
            if (training_step + 1) % args.log_interval == 0:
                wandb.log({'loss/train': section_training_loss[0] / section_training_loss[1]}, step=training_step)
                # Compute validation loss
                valid_x, valid_y = load_data(valid_dataset, batch_size=128, context_length=512, device='cuda')
                model.eval()
                with torch.no_grad():
                    valid_loss = cross_entropy_loss(model(valid_x), valid_y)
                wandb.log({'loss/valid': valid_loss}, step=training_step)
                model.train()
    except KeyboardInterrupt:
        pass




if __name__ == '__main__':
    train_model()
