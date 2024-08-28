import torch

SEED = 1996
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import os
import argparse
import logging
import tiktoken
from configs.base import Config
from models.gpt import GPT2


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def main(cfg: Config, args: argparse.Namespace):
    device = cfg.device
    logger = logging.getLogger(cfg.name)
    weight_best_path = os.path.join(cfg.checkpoint_dir, "weight_best.pth")
    weight_last_path = os.path.join(cfg.checkpoint_dir, "weight_last.pt")
    cpu_device = torch.device("cpu")
    device = torch.device(device)
    model = GPT2(cfg)
    model.to(cpu_device)
    if args.best_ckpt:
        ckpt = torch.load(weight_best_path, map_location=cpu_device)
    else:
        ckpt = torch.load(weight_last_path, map_location=cpu_device)["state_dict_model"]
    model.load_state_dict(ckpt)
    model.to(device)
    logger.info("Number of parameters: {:.2f}M".format(model.get_num_params() / 1e6))
    if cfg.compile:
        raise NotImplementedError
        # model = torch.compile(model)

    # Build automatic mixed precision
    if cfg.device == "cpu":
        cfg.use_amp = False

    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

    with torch.autocast(
        device_type="cuda" if cfg.device != "cpu" else "cpu",
        dtype=torch.float16,
        enabled=cfg.use_amp,
    ):
        temperature = 0.8
        top_k = 200
        max_new_tokens = 500
        start_ids = encode(args.prompt)
        x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
        y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
        logger.info(decode(y[0].tolist()))


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str, help="User input")
    parser.add_argument(
        "-cfg_path",
        "--config_path",
        type=str,
        default="../src/working/checkpoints/cfg.log",
        help="Path to config.py file",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
    )

    parser.add_argument(
        "--top_k",
        type=float,
        default=200,
    )

    parser.add_argument(
        "--max_new_tokens",
        type=float,
        default=500,
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    cfg = Config()
    cfg.load(args.cfg_path)
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main(cfg, args)
