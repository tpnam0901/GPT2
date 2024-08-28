import random
import numpy as np
import torch

SEED = 1996
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

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

# DistributedDataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


def main(cfg: Config):
    # ---------------------------------- DDP Settings ----------------------------------#
    device = cfg.device
    seed_offset = SEED
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        init_process_group(
            backend=cfg.ddp_backend, world_size=int(os.environ["WORLD_SIZE"])
        )
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        # this process will do logging, checkpointing etc.
        master_process = ddp_rank == 0
        # each process gets a different seed
        seed_offset = ddp_rank + seed_offset
    else:
        master_process = True
        ddp_local_rank = -1

    # ---------------------------------- Logging, Folder Initalize ----------------------------------#
    logger = logging.getLogger(cfg.name)
    if master_process:
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
    device = torch.device(device)
    model = GPT2(cfg)
    model.to(cpu_device)
    if cfg.pretrained:
        model.load_state_dict(torch.load(cfg.pretrained, map_location=cpu_device))
    if cfg.crop_block_size != cfg.block_size:
        model.crop_block_size(cfg.crop_block_size)
        cfg.save(cfg.checkpoint_dir)
    model.to(device)
    logger.info("Number of parameters: {:.2f}M".format(model.get_num_params() / 1e6))
    if cfg.compile:
        raise NotImplementedError
        # model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

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
    if cfg.device == "cpu":
        cfg.use_amp = False
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)

    best_loss = float(np.inf)

    global_train_step = 0
    global_val_step = 0

    if cfg.resume:
        raise NotImplementedError
        # checkpoint = torch.load(weight_last_path)
        # global_train_step = checkpoint["global_train_step"]
        # global_val_step = checkpoint["global_val_step"]
        # best_loss = checkpoint["best_loss"]
        # model.load_state_dict(checkpoint["state_dict_model"])
        # optimizer.load_state_dict(checkpoint["state_dict_optim_model"])
        # lr_scheduler.n_steps = checkpoint["lr_scheduler_step"]
        # logger.info(
        # "Resume training from Step {}/{}".format(global_train_step, cfg.num_iters)
        # )

    logger.info("Start training...")
    log_rank = seed_offset - SEED
    # unwrap DDP container if needed
    raw_model = model.module if ddp else model

    with mlflow.start_run():
        total_loss_train = []
        model.train()
        for inputs, targets in iter(train_dataloader):
            if global_train_step >= cfg.num_iters:
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
            # clip the gradient
            if cfg.grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            lr_scheduler.step()

            loss = loss.detach().cpu().numpy()
            total_loss_train.append(loss.item())
            mlflow.log_metric(f"loss_{log_rank}", loss.item(), step=global_train_step)
            if global_train_step % cfg.log_freq == 0:
                logger.info("Loss: {:.4f}".format(loss.item()))

            if global_train_step % cfg.ckpt_save_freq == 0 and master_process:
                logger.info(
                    "Loss after {} iters: {:.4f}".format(
                        cfg.ckpt_save_freq,
                        np.mean(total_loss_train).item(),
                    )
                )
                total_loss_train = []
                checkpoint = {
                    "global_train_step": global_train_step,
                    "global_val_step": global_val_step,
                    "best_loss": best_loss,
                    "state_dict_model": raw_model.state_dict(),
                    "state_dict_optim_model": optimizer.state_dict(),
                    "state_dict_scaler": scaler.state_dict(),
                    "lr_scheduler_step": lr_scheduler.n_steps,
                }
                torch.save(checkpoint, weight_last_path)

                total_loss_val = []
                model.eval()
                global_val_step += 1
                logger.info("Evaluating model on the validation dataset")
                for step, (inputs, targets) in tqdm(
                    enumerate(iter(val_dataloader)), total=cfg.num_val_iters
                ):
                    if step > cfg.num_val_iters:
                        break

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
                    torch.save(raw_model.state_dict(), weight_best_path)
                model.train()

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
    if cfg.resume:
        cfg.load(cfg.resume)

    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main(cfg)
