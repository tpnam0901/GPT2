from configs.base import Config as BaseConfig


class Config(BaseConfig):
    # Base
    def __init__(self, **kwargs):
        super(Config, self).__init__(**kwargs)
        self.add_args()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def add_args(self, **kwargs):
        self.name = "GPT2_Large"
        # ---------------------------------- Model settings ----------------------------------#
        self.pretrained = ""
        self.n_layer = 36
        self.n_head = 20
        self.n_embd = 1280

        for key, value in kwargs.items():
            setattr(self, key, value)
