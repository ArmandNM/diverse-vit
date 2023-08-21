import os

import torch
import torch.optim as optim
import torchvision

import numpy as np
import random

from diverse_vit.datasets import MNISTCIFARDataset
from diverse_vit.models import Recorder, DiverseViT
from diverse_vit.logging import CSVLogger


def split_dataset(dataset: torch.utils.data.Dataset, ratio: float):
    """Split a dataset into two datasets with a given ratio."""
    n = len(dataset)
    split = int(n * ratio)
    return torch.utils.data.random_split(dataset, [split, n - split])


def get_dataset(name, config, run=None):
    """Prepare datasets for training ID and testing both ID and OOD"""
    if name == "MNIST-CIFAR":
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
            ]
        )

        datasets = {}
        for split in ["train", "val"]:
            for domain in ["ID", "OOD"]:
                datasets[f"{split}{domain}"] = MNISTCIFARDataset(
                    data_path=os.path.join(config["data_path"], "mnist-cifar"),
                    split=split,
                    domain=domain,
                    causal_noise=config[domain]["causal_noise"],
                    spurious_ratio=config[domain]["spurious_ratio"],
                    transform=transform,
                )

        # Add spurious val dataset
        datasets["val_spurious"] = MNISTCIFARDataset(
            data_path=os.path.join(config["data_path"], "mnist-cifar"),
            split="val",
            domain="",
            causal_noise=0.5,
            spurious_ratio=0.0,
            transform=transform,
        )

        # Split val dataset into val and test
        for domain in ["ID", "OOD"]:
            datasets[f"val{domain}"], datasets[f"test{domain}"] = split_dataset(
                datasets[f"val{domain}"], config["val_ratio"]
            )
        return datasets

    else:
        raise ValueError(f"{name} is not a valid dataset name.")


def get_model(name, config):
    if name == "DiverseViT":
        model = DiverseViT(**config)
        model = Recorder(model)
    else:
        raise ValueError(f"{name} is not a valid model name.")

    return model


def get_optimizer(name, model_params, config):
    if name.lower() == "adam":
        optimizer = optim.Adam(model_params, **config)
        return optimizer
    else:
        raise ValueError(f"{name} is not a valid optimizer name.")


def get_scheduler(name, optimizer, config):
    if name == "MultiStepLR":
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, **config)
        return scheduler
    elif name.lower() == "none":
        return None
    else:
        raise ValueError(f"{name} is not a valid scheduler name.")


def get_logger(csv_path, num_heads):
    columns = list()

    columns.append("epoch")
    columns.append("Accuracy/trainID")
    columns.append("Loss/trainID")
    columns.append("Loss_diversity/trainID")

    for split in ["val", "test"]:
        for domain in ["ID", "OOD"]:
            split_domain = f"{split}{domain}"
            columns.append(f"Accuracy (All Heads)/{split_domain}")
            columns.append(f"Accuracy Head Best/{split_domain}")
            columns.append(f"Loss (All Heads)/{split_domain}")
            columns.append(f"Loss Head Best/{split_domain}")
            columns.append(f"Loss_diversity (All Heads)/{split_domain}")

            for head_idx in range(num_heads):
                columns.append(f"Accuracy Head No{head_idx}/{split_domain}")
                columns.append(f"Loss Head No{head_idx}/{split_domain}")

    logger = CSVLogger(csv_path=csv_path, columns=columns)
    return logger


def set_random_seed(random_seed):
    # From STAI-tuned
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # to suppress warning
    # torch.use_deterministic_algorithms(True, warn_only=True)
    torch.use_deterministic_algorithms(True)
