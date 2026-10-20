"""
Train and validate the models, saving the best weights and the training curves.

Stage 3: baseline classifier only. The Siamese model is added later via --model siamese.

Usage example:
    python train.py --model baseline --subset 1000 --epochs 1

The test split is never touched here; it is only used for the final evaluation.
"""

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")  # save plots to file without needing a display (e.g. on Rangpur)
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Subset

from dataset import LesionDataset, build_transforms, get_splits
from modules import BaselineClassifier, count_parameters

OUT_DIR = "outputs"
DATA_ROOT = "data"

def parse_args():
    parser = argparse.ArgumentParser(description="Train the melanoma classifiers.")
    parser.add_argument("--model", choices=["baseline"], default="baseline")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def pick_device():
    """Best available device: CUDA (Rangpur), MPS (Mac GPU), else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def random_subset(dataset, size, seed):
    """A reproducible random subset, for quick smoke tests."""
    if size is None or size >= len(dataset):
        return dataset
    g = torch.Generator().manual_seed(seed)
    
    indices = torch.randperm(len(dataset), generator=g)[:size].tolist()
    return Subset(dataset, indices)


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    """One pass over the training data. Returns the mean training loss."""
    model.train()
    total_loss, n = 0.0, 0
    for images, targets, _ in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(images), targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(targets)
        n += len(targets)
    return total_loss / n


def main():
    args = parse_args()
    # set the random seed for reproducibility
    torch.manual_seed(args.seed)
    device = pick_device()
    os.makedirs(OUT_DIR, exist_ok=True)

    # Data: train with augmentation, validate without. The test split is not loaded.
    df = get_splits(DATA_ROOT)
    train_set = random_subset(LesionDataset(df, "train", build_transforms(train=True)), args.subset, args.seed)
    val_set = random_subset(LesionDataset(df, "val"), args.subset and args.subset // 2, args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = BaselineClassifier(pretrained=True).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    history, best_auroc, epochs_since_best = [], -1.0, 0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        # train an epoch and evaluate on the validation set  

if __name__ == "__main__":
    main()
