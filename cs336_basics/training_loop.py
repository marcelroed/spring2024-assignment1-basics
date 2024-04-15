import numpy as np
import torch
import torch.nn as nn
from typing import BinaryIO, IO
from os import PathLike
from numpy import typing as npt
from cs336_basics.transformer import Transformer
from cs336_basics.training import cross_entropy_loss, AdamW, gradient_clipping, cosine_with_warmup_lr_schedule
from cs336_basics.bpe import Tokenizer



def load_data(dataset: npt.NDArray, batch_size: int, context_length: int, device: str):
    dataset_length = len(dataset)
    sample_range = dataset_length - context_length
    sample_idcs = np.random.randint(0, sample_range, batch_size)
    samples = np.stack([dataset[idx:idx + context_length] for idx in sample_idcs])
    samples_offset = np.stack([dataset[idx + 1:idx + context_length + 1] for idx in sample_idcs])
    return torch.tensor(samples, device=device), torch.tensor(samples_offset, device=device)


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



def train_model():
    import wandb
    # wandb.init(project="cs336-assignment-1", entity="marcelroed")




if __name__ == '__main__':
    train_model()
