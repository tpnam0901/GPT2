from configs.base import Config as BaseConfig


class Config(BaseConfig):
    # Base
    def __init__(self, **kwargs):
        super(Config, self).__init__(**kwargs)
        self.add_args()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def add_args(self, **kwargs):
        self.name = "GPT2_Medium"
        # ---------------------------------- Model settings ----------------------------------#
        self.pretrained = "models/pretrained/gpt2-medium.pth"
        self.n_layer = 24
        self.n_head = 16
        self.n_embd = 1024

        for key, value in kwargs.items():
            setattr(self, key, value)
