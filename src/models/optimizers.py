import inspect
import math
import torch
import torch.nn as nn
from typing import Tuple
from configs.base import Config


def _get_params(model: nn.Module):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    return decay_params, nodecay_params


def adamw(
    model: nn.Module,
    cfg: Config,
):
    decay_params, nodecay_params = _get_params(model)
    # Create AdamW optimizer and use the fused version if it is available
    optim_groups = [
        {"params": decay_params, "weight_decay": cfg.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and cfg.device != "cpu"
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(
        optim_groups, lr=cfg.learning_rate, betas=cfg.betas, **extra_args
    )

    return optimizer


class PlaceHolderScheduledOptim:
    """A simple wrapper class for learning rate scheduling"""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        cfg: Config,
    ):
        self._optimizer = optimizer
        self.learning_rate = optimizer.param_groups[0]["lr"]
        self.warmup_iters = cfg.warmup_iters
        self.lr_decay_iters = cfg.lr_decay_iters
        self.min_lr = cfg.min_lr
        self.n_steps = 0

    def step(self):
        pass


class ScheduledOptim:
    """A simple wrapper class for learning rate scheduling"""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        cfg: Config,
    ):
        self._optimizer = optimizer
        self.learning_rate = optimizer.param_groups[0]["lr"]
        self.warmup_iters = cfg.warmup_iters
        self.lr_decay_iters = cfg.lr_decay_iters
        self.min_lr = cfg.min_lr
        self.n_steps = 0

    def step(self):
        "Step with the inner optimizer"
        self.n_steps += 1
        lr = self.learning_rate
        # 1) linear warmup for warmup_iters steps
        if self.n_steps < self.warmup_iters:
            lr = self.learning_rate * self.n_steps / self.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        elif self.n_steps > self.lr_decay_iters:
            lr = self.min_lr
        # 3) in between, use cosine decay down to min learning rate
        else:
            decay_ratio = (self.n_steps - self.warmup_iters) / (
                self.lr_decay_iters - self.warmup_iters
            )
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
            lr = self.min_lr + coeff * (self.learning_rate - self.min_lr)
        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr
