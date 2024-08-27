from configs.gpt2 import Config as BaseConfig


class Config(BaseConfig):
    # Base
    def __init__(self, **kwargs):
        super(Config, self).__init__(**kwargs)
        super().add_args()
        self.add_args()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def add_args(self, **kwargs):
        self.name = "GPT2-openwebtext_gpt2"
        # ---------------------------------- Model settings ----------------------------------#
        self.crop_block_size = 512

        # ---------------------------------- Dataset ----------------------------------#
        self.train_bin: str = "working/dataset/openwebtext/train.bin"
        self.val_bin: str = "working/dataset/openwebtext/val.bin"
        self.pin_memory: bool = True

        for key, value in kwargs.items():
            setattr(self, key, value)
