import random
import numpy as np
import torch
import torch.nn as nn

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

import os
import argparse
import logging
import mlflow
import datetime
from tqdm.auto import tqdm
from configs.base import Config, import_config
from models import optimizers
from models.gpt import GPT2
from models.losses import GPTLoss
from data.dataloader import Dataloader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def main(cfg: Config):
    # ---------------------------------- Logging, Folder Initalize ----------------------------------#

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    cfg.checkpoint_dir = os.path.join(cfg.checkpoint_dir, cfg.name, current_time)

    # Log, weight, mlflow folder
    log_dir = os.path.join(cfg.checkpoint_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Add logger to log folder
    logging.getLogger().setLevel(logging.INFO)
    file_handler = logging.FileHandler(os.path.join(log_dir, "train.log"))
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger = logging.getLogger(cfg.name)
    logger.addHandler(file_handler)
    logger.addHandler(logging.StreamHandler())

    # Add mlflow to log folder
    mlflow.set_tracking_uri(
        uri=f'file://{os.path.abspath(os.path.join(log_dir, "mlruns"))}'
    )

    # Save configs
    logger.info("Saving config to {}".format(cfg.checkpoint_dir))
    cfg.save(cfg.checkpoint_dir)
    cfg.show()

    # ---------------------------------- Model Initalize ----------------------------------#
    logger.info("Building model...")
    cpu_device = torch.device("cpu")
    device = torch.device(cfg.device)
    model = GPT2(cfg)
    model.to(cpu_device)
    if cfg.pretrained:
        model.load_state_dict(torch.load(cfg.pretrained, map_location=cpu_device))
    model.to(device)
    logger.info("Number of parameters: {:.2f}M".format(model.get_num_params() / 1e6))

    # Preparing checkpoint output
    weight_best_path = os.path.join(cfg.checkpoint_dir, "weight_best.pth")
    weight_last_path = os.path.join(cfg.checkpoint_dir, "weight_last.pt")

    # ---------------------------------- Train Initalize ----------------------------------#
    # Build dataset
    logger.info("Building dataset...")
    train_dataloader = Dataloader(
        cfg.train_bin,
        cfg.batch_size,
        cfg.block_size,
        cfg.pin_memory,
        device,
    )

    val_dataloader = Dataloader(
        cfg.val_bin,
        cfg.batch_size,
        cfg.block_size,
        cfg.pin_memory,
        device,
    )

    logger.info("Building optimizer and loss functions")
    # Build optimizer
    optimizer = getattr(optimizers, cfg.optimizer)(model, cfg)
    lr_scheduler = getattr(optimizers, cfg.lr_scheduler)(optimizer, cfg)

    # Build loss functions
    criterion = GPTLoss()

    # Build automatic mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)

    best_loss = float(np.inf)

    global_train_step = 0
    global_val_step = 0

    if False:
        checkpoint = torch.load(weight_last_path)
        global_train_step = checkpoint["global_train_step"]
        global_val_step = checkpoint["global_val_step"]
        best_loss = checkpoint["best_loss"]
        model.load_state_dict(checkpoint["state_dict_model"])
        optimizer.load_state_dict(checkpoint["state_dict_optim_model"])
        lr_scheduler.n_steps = checkpoint["lr_scheduler_step"]
        logger.info(
            "Resume training from Step {}/{}".format(global_train_step, cfg.num_iters)
        )
    else:
        logger.info("Start training...")

    with mlflow.start_run():
        total_loss_train = []
        model.train()
        with tqdm(total=cfg.num_iters, ascii=True) as pbar:
            for step, (inputs, targets) in enumerate(iter(train_dataloader)):
                if step > cfg.num_iters - 1:
                    break
                global_train_step += 1

                with torch.autocast(
                    device_type="cuda" if cfg.device != "cpu" else "cpu",
                    dtype=torch.float16,
                    enabled=cfg.use_amp,
                ):
                    logits = model(inputs)
                    loss = criterion(logits, targets)

                scaler.scale(loss).backward()
                scaler.scale(optimizer).step()
                scaler.update()
                optimizer.zero_grad()
                lr_scheduler.step()

                loss = loss.detach().cpu().numpy()
                total_loss_train.append(loss.item())
                mlflow.log_metric("loss", loss.item(), step=global_train_step)

                postfix = "Loss: {:.4f}".format(loss.item())
                pbar.set_description(postfix)
                pbar.update(1)

                if global_train_step % cfg.ckpt_save_fred == 0:
                    logger.info(
                        "Loss after {} iters: {:.4f}".format(
                            cfg.ckpt_save_fred,
                            np.mean(total_loss_train).item(),
                        )
                    )
                    total_loss_train = []
                    checkpoint = {
                        "global_train_step": global_train_step,
                        "global_val_step": global_val_step,
                        "best_loss": best_loss,
                        "state_dict_model": model.state_dict(),
                        "state_dict_optim_model": optimizer.state_dict(),
                        "state_dict_scaler": scaler.state_dict(),
                        "lr_scheduler_step": lr_scheduler.n_steps,
                    }
                    torch.save(checkpoint, weight_last_path)

                    total_loss_val = []
                    model.eval()
                    global_val_step += 1
                    logger.info("Evaluating model on the validation dataset")
                    for inputs, targets in tqdm(
                        iter(val_dataloader), total=cfg.num_val_iters
                    ):

                        with torch.no_grad():
                            logits = model(inputs)
                            loss = criterion(logits, targets)
                            loss = loss.detach().cpu().numpy()
                        total_loss_val.append(loss.item())

                    val_loss = np.mean(total_loss_val).item()
                    mlflow.log_metric("val_loss", val_loss, step=global_val_step)
                    logger.info(
                        "Iter {}/{} - val_loss: {:.4f} ".format(
                            global_train_step,
                            cfg.num_iters,
                            val_loss,
                        )
                    )

                    if val_loss < best_loss:
                        best_loss = val_loss
                        torch.save(model.state_dict(), weight_best_path)

    end_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logger.info("Training finished at {}".format(end_time))


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cfg",
        "--config",
        type=str,
        default="../src/configs/base.py",
        help="Path to config.py file",
    )
    parser.add_argument(
        "-rcp",
        "--resume_config_path",
        type=str,
        default=None,
        help="Path to resume cfg.log file if want to resume training",
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    cfg: Config = import_config(args.config)
    if False and cfg.resume and cfg.opt_path:
        resume = cfg.resume
        resume_path = cfg.resume_path
        cfg.load(cfg.opt_path)
        cfg.resume = resume
        cfg.resume_path = resume_path

    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main(cfg)
