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

import json
import cherrypy


def arg_parser():
    parser = argparse.ArgumentParser()
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
        type=int,
        default=50,
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
    )
    parser.add_argument("--best_ckpt", action="store_true")
    parser.add_argument(
        "--port",
        type=int,
        default=8123,
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


args = arg_parser()
cfg = Config()
cfg.load(args.config_path)
level = logging.DEBUG if args.debug else logging.INFO
logging.basicConfig(
    level=level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

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


conf = {
    "/": {
        "request.dispatch": cherrypy.dispatch.MethodDispatcher(),
        "tools.sessions.on": True,
        "tools.response_headers.on": True,
        "tools.response_headers.headers": [("Content-Type", "application/json")],
    }
}


@cherrypy.expose
class StringGeneratorWebService(object):

    @cherrypy.tools.accept(media="application/json")
    @cherrypy.tools.json_out()
    def POST(self):
        # Why this need to decode two time?
        content = cherrypy.request.body.read().decode("utf-8")
        content = content.replace('\\"', '"')
        content = content[1:-1]
        logger.info(f"Received data: {content}")
        content = json.loads(content)
        data = {"text": "INVALID TOKEN!"}
        if str(content["TOKEN"]) == "THIS_IS_MY_CUSTOM_TOKEN":
            try:
                with torch.autocast(
                    device_type="cuda" if cfg.device != "cpu" else "cpu",
                    dtype=torch.float16,
                    enabled=cfg.use_amp,
                ):
                    start_ids = []
                    for msg in content["messages"]:
                        start_ids += encode(msg["text"])
                        start_ids.append(enc.eot_token)
                    x = torch.tensor(start_ids, dtype=torch.long, device=device)[
                        None, ...
                    ]
                    y = model.generate(
                        x,
                        args.max_new_tokens,
                        temperature=args.temperature,
                        top_k=args.top_k,
                    )
                    response = decode(y[0].tolist())
                    response = response.split("<|endoftext|>")[1].replace('"', "''")
                    logger.info(response)
                data["text"] = response
            except Exception as e:
                logger.info(e)
                data["text"] = (
                    "My server is having a problem! Please get in touch with the admin to help me!"
                )
        data_json = json.dumps(data)
        return data_json


cherrypy.config.update({"server.socket_port": args.port})
cherrypy.quickstart(StringGeneratorWebService(), "/", conf)
