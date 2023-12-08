import os
import yaml
import numpy as np
import argparse
import importlib
import functools
import logging
import sys
from pathlib import Path
import transforms as T

import torch
from learning import Learning
from models import MSResUNet
from torch.utils.data import DataLoader
from dataloader import RibDataset
from losses import ComboLoss
from utils import dice_round_fn, search_thresholds

def load_yaml(file_name):
    with open(file_name) as stream:
        config = yaml.load(stream, Loader=yaml.SafeLoader)
    return config

def init_seed(SEED=1):
    os.environ["PYTHONHASHSEED"] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

def init_eval_fns(train_config):
    score_threshold = train_config["evaluation"]["score_threshold"]
    area_threshold = train_config["evaluation"]["area_threshold"]
    thr_search_list = train_config["evaluation"]["threshold_search_list"]
    area_search_list = train_config["evaluation"]["area_search_list"]
    local_metric_fn = functools.partial(
        dice_round_fn, score_threshold=score_threshold, area_threshold=area_threshold
    )

    global_metric_fn = functools.partial(
        search_thresholds, thr_list=thr_search_list, area_list=area_search_list
    )
    return local_metric_fn, global_metric_fn

def init_logger(directory, log_file_name):
    formatter = logging.Formatter(
        "\n%(asctime)s\t%(message)s", datefmt="%m/%d/%Y %H:%M:%S"
    )
    log_path = Path(directory, log_file_name)
    if log_path.exists():
        log_path.unlink()
    handler = logging.FileHandler(filename=log_path)
    handler.setFormatter(formatter)

    logger = logging.getLogger(log_file_name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger

def argparser():
    parser = argparse.ArgumentParser(description="DRR Rib Segmentation pipeline")
    parser.add_argument("train_config", default='', type=str, help="train config path")
    return parser.parse_args()


def train(train_config,
          log_folder,
          log_dir,
          train_dataloader,
          test_dataloader,
          local_metric_fn,
          global_metric_fn):

    fold_logger = init_logger(log_dir, "train.log")
    best_checkpoint_folder = Path(
        log_folder, train_config["checkpoints"]["best_folder"])
    best_checkpoint_folder.mkdir(exist_ok=True, parents=True)

    checkpoints_history_folder = Path(
        fold_logger,
        train_config["checkpoints"]["full_folder"],
    )
    checkpoints_history_folder.mkdir(exist_ok=True, parents=True)
    checkpoints_topk = train_config["checkpoints"]["topk"]

    device = train_config["device"]

    model = MSResUNet(n_classes=train_config['model']['n_classes'],pretrained=train_config['model']['pretrained'])
    if len(train_config["device_list"]) > 1:
        model = torch.nn.DataParallel(model)
    freeze_model = train_config["model"]["freeze"]

    loss_fn = ComboLoss(**train_config["loss"])

    optimizer = torch.optim.Adam(model.parameters(), lr=train_config["lr"], weight_decay=train_config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)

    n_epoches = train_config["epoches"]
    grad_clip = train_config["grad_clip"]
    grad_accum = train_config["grad_accum"]
    early_stopping = train_config["early_stopping"]

    exper_name = train_config["exper_name"]

    Learning(
        optimizer,
        loss_fn,
        device,
        n_epoches,
        scheduler,
        freeze_model,
        grad_clip,
        grad_accum,
        early_stopping,
        exper_name,
        best_checkpoint_folder,
        checkpoints_history_folder,
        checkpoints_topk,
        fold_logger,
    ).run_train(
        model, train_dataloader, test_dataloader, local_metric_fn, global_metric_fn
    )



def main():
    args = argparser()
    train_config_folder = Path(args.train_cfg.strip("/"))
    train_config = load_yaml(train_config_folder)

    log_folder = train_config_folder.parents[0]
    log_dir = Path(log_folder, train_config["logger_dir"])
    log_dir.mkdir(exist_ok=True, parents=True)
    main_logger = init_logger(log_dir, "train_main.log")

    json_path = train_config["json_path"]
    seed = train_config["seed"]
    init_seed(seed)
    main_logger.info(train_config)

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, train_config["device_list"]))

    img_size = train_config["image_size"]
    train_tfm, test_tfm = T.transformA(img_size)

    num_workers = train_config["num_workers"]
    batch_size = train_config["batch_size"]

    local_metric_fn, global_metric_fn = init_eval_fns(train_config)

    train_dataloader = DataLoader(
        dataset=RibDataset(mode="train", transforms=train_tfm, json_path=json_path),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True)
    
    test_dataloader = DataLoader(
        dataset=RibDataset(mode="test", transforms=test_tfm, json_path=json_path),
        batch_size=1,
        num_workers=num_workers,
        shuffle=False)
    
    main_logger.info("Start training ...")
    train(
        train_config,
        log_folder,
        log_dir,
        train_dataloader,
        test_dataloader,
        local_metric_fn,
        global_metric_fn
    )
    



if __name__ == "__main__":
    main()