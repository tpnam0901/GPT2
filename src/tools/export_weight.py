import os
import sys

lib_path = os.path.abspath("").replace("/tools", "")
sys.path.append(lib_path)

import torch
from transformers import GPT2LMHeadModel
from models.gpt import GPT2
from configs.base import import_config

model_type = "gpt2-xl"
config = {
    "gpt2": "../configs/gpt2.py",  # 124M params
    "gpt2-medium": "../configs/gpt2_medium.py",  # 350M params
    "gpt2-large": "../configs/gpt2_large.py",  # 774M params
    "gpt2-xl": "../configs/gpt2_xlarge.py",  # 1558M params
}

cfg = import_config(config[model_type])
model = GPT2(cfg)

model_hf = GPT2LMHeadModel.from_pretrained(model_type)
sd_hf = model_hf.state_dict()
sd = model.state_dict()

sd_keys = sd.keys()
sd_keys = [
    k for k in sd_keys if not k.endswith(".attn.bias")
]  # discard this mask / buffer, not a param

# copy while ensuring all of the parameters are aligned and match in names and shapes
sd_keys_hf = sd_hf.keys()
sd_keys_hf = [
    k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
]  # ignore these, just a buffer
sd_keys_hf = [
    k for k in sd_keys_hf if not k.endswith(".attn.bias")
]  # same, just the mask (buffer)
transposed = [
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
]

assert len(sd_keys_hf) == len(
    sd_keys
), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
for k in sd_keys_hf:
    if any(k.endswith(w) for w in transposed):
        # special treatment for the Conv1D weights we need to transpose
        assert sd_hf[k].shape[::-1] == sd[k].shape
        with torch.no_grad():
            sd[k].copy_(sd_hf[k].t())
    else:
        # vanilla copy over the other parameters
        # print(sd_hf[k].shape, sd[k].shape)
        assert sd_hf[k].shape == sd[k].shape
        with torch.no_grad():
            sd[k].copy_(sd_hf[k])

torch.save(model.state_dict(), f"../models/pretrained/{model_type}.pth")
