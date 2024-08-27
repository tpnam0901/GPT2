import logging
import os
from abc import ABC, abstractmethod
from typing import Tuple
import importlib
import sys


class Base(ABC):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def show(self):
        pass

    @abstractmethod
    def save(self, save_folder: str):
        pass

    @abstractmethod
    def load(self, cfg_path: str):
        pass


class BaseConfig(Base):
    def __init__(self, **kwargs):
        super(BaseConfig, self).__init__(**kwargs)

    def show(self):
        for key, value in self.__dict__.items():
            logging.info(f"{key}: {value}")

    def save(self, save_folder: str):
        message = "\n"
        for k, v in sorted(vars(self).items()):
            message += f"{str(k):>30}: {str(v):<40}\n"

        os.makedirs(os.path.join(save_folder), exist_ok=True)
        out_cfg = os.path.join(save_folder, "cfg.log")
        with open(out_cfg, "w") as cfg_file:
            cfg_file.write(message)
            cfg_file.write("\n")

        logging.info(message)

    def load(self, cfg_path: str):
        def decode_value(value: str):
            value = value.strip()
            value_converted = None
            if "." in value and value.replace(".", "").isdigit():
                value_converted = float(value)
            elif value.isdigit():
                value_converted = int(value)
            elif value == "True":
                value_converted = True
            elif value_converted == "False":
                value_converted = False
            elif (
                value.startswith("'")
                and value.endswith("'")
                or value.startswith('"')
                and value.endswith('"')
            ):
                value_converted = value[1:-1]
            elif value.startswith("(") or value.startswith("["):
                value_converted = []
                for temp in value.strip("(").strip(")").split(","):
                    value_converted.append(decode_value(temp))
                if value.startswith("("):
                    value_converted = tuple(value_converted)
            else:
                value_converted = value
            return value_converted

        with open(cfg_path, "r") as f:
            data = f.read().split("\n")
            # remove all empty strings
            data = list(filter(None, data))
            # convert to dict
            data_dict = {}
            for i in range(len(data)):
                key, value = (
                    data[i].split(":")[0].strip(),
                    data[i].split(":")[1].strip(),
                )
                if value.startswith("[") and value.endswith("]"):
                    value = value[1:-1].split(",")
                    value = [decode_value(x) for x in value]
                else:
                    value = decode_value(value)

                data_dict[key] = value
        for key, value in data_dict.items():
            setattr(self, key, value)


class Config(BaseConfig):
    # Base
    def __init__(self, **kwargs):
        super(Config, self).__init__(**kwargs)
        self.name = "default"
        self.set_args()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def set_args(self, **kwargs):
        # ---------------------------------- Training settings ----------------------------------#
        self.num_iters: int = 10000
        self.num_val_iters: int = 5000
        self.batch_size: int = 32
        self.checkpoint_dir: "str" = "working/checkpoints"
        self.ckpt_save_fred: int = 4000
        self.device: str = "cpu"

        # ---------------------------------- Optim settings ----------------------------------#
        self.optimizer: str = "adamw"
        self.learning_rate: float = 6e-4
        self.weight_decay: float = 1e-1
        self.betas: Tuple[float, float] = (0.9, 0.95)

        # PlaceHolderScheduledOptim, ScheduledOptim
        self.lr_scheduler: str = "ScheduledOptim"
        self.warmup_iters: int = 2000
        self.lr_decay_iters: int = 600_000
        self.min_lr: float = 6e-5

        # ---------------------------------- Model settings ----------------------------------#
        self.pretrained: str = ""
        self.block_size: int = 1024
        self.crop_block_size: int = 1024
        # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
        self.vocab_size: int = 50257
        self.n_layer: int = 12
        self.n_head: int = 12
        self.n_embd: int = 768
        self.dropout: float = 0.0
        # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
        self.bias: bool = True

        # ---------------------------------- Dataset ----------------------------------#
        self.train_bin: str = "working/dataset/openwebtext/train.bin"
        self.val_bin: str = "working/dataset/openwebtext/val.bin"
        self.pin_memory: bool = True

        for key, value in kwargs.items():
            setattr(self, key, value)


def import_config(
    path: str,
):
    """Get arguments for training and evaluate
    Returns:
        cfg: ArgumentParser
    """
    # Import config from path
    spec = importlib.util.spec_from_file_location("config", path)
    config = importlib.util.module_from_spec(spec)
    sys.modules["config"] = config
    spec.loader.exec_module(config)
    cfg = config.Config()
    return cfg
