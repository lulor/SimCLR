import logging
import os
import random
import shutil
import sys
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader


def setup_logging(
    output_folder,
    console="debug",
    info_filename="info.log",
    debug_filename="debug.log",
):
    """Set up logging files and console output.
    Creates one file for INFO logs and one for DEBUG logs.
    Args:
        output_folder (str): creates the folder where to save the files.
        debug (str):
            if == "debug" prints on console debug messages and higher
            if == "info"  prints on console info messages and higher
            if == None does not use console (useful when a logger has already been set)
        info_filename (str): the name of the info file. if None, don't create info file
        debug_filename (str): the name of the debug file. if None, don't create debug file
    """

    if os.path.exists(output_folder):
        raise FileExistsError(f"{output_folder} already exists!")
    os.makedirs(output_folder, exist_ok=True)

    # logging.Logger.manager.loggerDict.keys() to check which loggers are in use
    base_formatter = logging.Formatter("%(asctime)s : %(message)s", "%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger("")
    logger.setLevel(logging.DEBUG)

    if info_filename is not None:
        info_file_handler = logging.FileHandler(
            os.path.join(output_folder, info_filename)
        )
        info_file_handler.setLevel(logging.INFO)
        info_file_handler.setFormatter(base_formatter)
        logger.addHandler(info_file_handler)

    if debug_filename is not None:
        debug_file_handler = logging.FileHandler(
            os.path.join(output_folder, debug_filename)
        )
        debug_file_handler.setLevel(logging.DEBUG)
        debug_file_handler.setFormatter(base_formatter)
        logger.addHandler(debug_file_handler)

    if console is not None:
        console_handler = logging.StreamHandler()
        if console == "debug":
            console_handler.setLevel(logging.DEBUG)
        if console == "info":
            console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(base_formatter)
        logger.addHandler(console_handler)

    def exception_handler(ex_type, ex, tb):
        logger.info("\n%s", os.path.join(traceback.format_exception(ex_type, ex, tb)))

    sys.excepthook = exception_handler


def get_run_out_dir(out_dir):
    run_number = 0
    while True:
        run_out_dir = os.path.join(out_dir, f"run_{run_number}")
        if not os.path.exists(run_out_dir):
            os.makedirs(run_out_dir)
            return run_out_dir
        run_number += 1


def make_deterministic(seed):
    # Setting the seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        device = "cuda"  # cuda:0
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return device


def split_dataset(dataset, split):
    rnd_indexes = np.arange(len(dataset))
    np.random.shuffle(rnd_indexes)

    cut = int(split * len(dataset))
    training_indexes = rnd_indexes[:cut]
    validation_indexes = rnd_indexes[cut:]

    # training set
    train_dataset = torch.utils.data.Subset(dataset, training_indexes)

    # validation set
    val_dataset = torch.utils.data.Subset(dataset, validation_indexes)

    return train_dataset, val_dataset


def build_data_loader(dataset, batch_size, mode):
    if mode == "train":
        shuffle = True
        drop_last = True
    elif mode == "eval":
        shuffle = False
        drop_last = False
    else:
        raise RuntimeError(f"Unknown mode recieved {mode}")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=True,
        num_workers=os.cpu_count(),
    )


def postfix_str(postfix_dict):
    str_ = ""
    for name, value in postfix_dict.items():
        if len(str_) > 0:
            str_ += ", "
        str_ += f"{name}: {value:5.3f}"
    return str_


def build_state(model, optimizer, best_val_top5_acc, epoch):
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_top5": best_val_top5_acc,
        "epoch": epoch,
    }


def save_checkpoint(output_folder, state, is_best, filename):
    file_path = os.path.join(output_folder, filename)
    torch.save(state, file_path)
    if is_best:
        shutil.copyfile(file_path, os.path.join(output_folder, "best_model.pth"))


def load_checkpoint(file_path, device):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Checkpoint '{file_path}' does not exist")
    return torch.load(file_path, map_location=device)


def resume_from_state(state, model, optimizer):
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    epoch = state["epoch"]
    best_val_top5_acc = state["best_val_top5"]
    return epoch, best_val_top5_acc


def reload_simclr_encoder(state, model):
    if model.projection is not None:
        raise RuntimeError("Unexpected projection in model to reload")
    model_state_dict = state["model_state_dict"]
    encoder_state_dict = {
        k: v for k, v in model_state_dict.items() if not k.startswith("projection")
    }
    model.load_state_dict(encoder_state_dict)
