import torch
import torch.nn as nn
import torch.optim as optim

from collections.abc import Callable
from typing import Optional
import math

def cross_entropy_loss(logits, targets):
    # logits: (batch_size, num_classes)
    # targets: (batch_size, )
    logits = logits - torch.max(logits, dim=-1).values.unsqueeze(-1)
    return torch.mean(torch.log(torch.sum(torch.exp(logits), dim=-1, keepdim=True)) - torch.gather(logits, -1, targets.unsqueeze(-1)))
    

class SGD(optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or initial value.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.
        return loss


class AdamW(optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-6, weight_decay: float = 0.0,):
        assert lr >= 0, f"Invalid learning rate: {lr}, must be >= 0"
        assert 0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0, f"Invalid beta values: {betas}, must be in [0, 1)"
        assert eps >= 0, f"Invalid epsilon value: {eps}, must be >= 0"
        assert weight_decay >= 0, f"Invalid weight_decay value: {weight_decay}, must be >= 0"
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                grad = p.grad

                if 't' not in state:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                state["t"] += 1
                t = state["t"]

                m, v = state["m"], state["v"]
                b1, b2 = group["betas"]


                m.mul_(b1).add_(grad, alpha=(1.0 - b1))
                v.mul_(b2).addcmul_(grad, grad, value=1.0 - b2)
                denom = v.sqrt().add_(group["eps"])

                alpha = group["lr"]

                alpha_t = alpha * math.sqrt(1.0 - b2 ** t) / (1.0 - b1 ** t)

                p.addcdiv_(m, denom, value=-alpha_t)

                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-alpha * group["weight_decay"]))

        return loss

def cosine_with_warmup_lr_schedule(it: int, max_learning_rate: float, min_learning_rate: float, warmup_iters: int, cosine_cycle_iters: int):
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    if it < cosine_cycle_iters:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + 0.5 * (1 + math.cos(math.pi * progress)) * (max_learning_rate - min_learning_rate)
    return min_learning_rate

def gradient_clipping(parameters, max_norm: float):
    acc = 0
    for p in parameters:
        if p.grad is not None:
            acc += p.grad.data.square().sum()
    total_norm = acc.sqrt()
    if total_norm > max_norm:  # TODO make this unconditional
        total_norm += 1e-6
        for p in parameters:
            if p.grad is not None:
                p.grad.data *= max_norm / total_norm


if __name__ == '__main__':
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=1e3)
    for t in range(10):
        opt.zero_grad() # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean() # Compute a scalar loss value.
        print(loss.cpu().item())
        loss.backward() # Run backward pass, which computes gradients.
        opt.step() # Run optimizer step