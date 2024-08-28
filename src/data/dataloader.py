import os
import torch
import numpy as np
from typing import Tuple


class Dataloader:
    def __init__(
        self,
        bin_path: str,
        batch_size: int,
        block_size: int,
        pin_memory: bool = False,
        device: torch.device = torch.device("cpu"),
    ):
        super(Dataloader, self).__init__()
        self.bin_path = os.path.abspath(bin_path)
        self.block_size = block_size
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.device = device

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Recreate np.memmap every batch to avoid a memory leak
        data = np.memmap(self.bin_path, dtype=np.uint16, mode="r")
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack(
            [
                torch.from_numpy((data[i : i + self.block_size]).astype(np.int64))
                for i in ix
            ]
        )
        y = torch.stack(
            [
                torch.from_numpy(
                    (data[i + 1 : i + 1 + self.block_size]).astype(np.int64)
                )
                for i in ix
            ]
        )

        if self.pin_memory and self.device != torch.device("cpu"):
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y

    def __len__(self) -> int:
        return 0
