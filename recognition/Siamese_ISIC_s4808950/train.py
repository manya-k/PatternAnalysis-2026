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
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--subset", type=int, default=None)
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


def subset_targets(dataset):
    """Labels of a LesionDataset, or of a Subset of one."""
    if isinstance(dataset, Subset):
        return dataset.dataset.targets[dataset.indices]
    return dataset.targets


def class_weights(targets):
    """
    Inverse-frequency weights so each class contributes equally to the loss.

    weight_c = N / (2 * N_c). With ~21% melanoma this gives about [0.64, 2.35], so a
    missed melanoma costs ~3.7x more than a misclassified benign lesion.
    """
    counts = torch.bincount(targets, minlength=2).float()
    return len(targets) / (2 * counts)


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


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    """
    Mean loss, AUROC and accuracy on a validation loader.

    AUROC uses the softmax probability of melanoma, so it does not depend on any threshold.
    """
    model.eval()
    total_loss, probs, labels = 0.0, [], []
    for images, targets, _ in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        total_loss += loss_fn(logits, targets).item() * len(targets)
        probs.append(logits.softmax(dim=1)[:, 1].cpu())
        labels.append(targets.cpu())
    probs, labels = torch.cat(probs), torch.cat(labels)
    return {
        "loss": total_loss / len(labels),
        "auroc": roc_auc_score(labels, probs),
        "accuracy": ((probs >= 0.5).long() == labels).float().mean().item(),
    }



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
    weights = class_weights(subset_targets(train_set))
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"device {device} | train {len(train_set)} | val {len(val_set)} | "
          f"class weights {[round(w, 2) for w in weights.tolist()]} | parameters {count_parameters(model):,}")

    history, best_auroc, epochs_since_best = [], -1.0, 0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val = evaluate(model, val_loader, loss_fn, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val["loss"],
                        "val_auroc": val["auroc"], "val_accuracy": val["accuracy"],
                        "seconds": time.time() - epoch_start})
        print(f"epoch {epoch:>3} | train loss {train_loss:.4f} | val loss {val['loss']:.4f} | "
              f"val AUROC {val['auroc']:.4f} | val acc {val['accuracy']:.4f} | {history[-1]['seconds']:.0f}s")

        # Keep the weights with the best validation AUROC; stop early if it stops improving.
        if val["auroc"] > best_auroc:
            best_auroc, epochs_since_best = val["auroc"], 0

        else:
            epochs_since_best += 1

    summary = {
        "model": args.model, "device": str(device), "args": vars(args),
        "best_val_auroc": best_auroc, "parameters": count_parameters(model),
        "total_minutes": (time.time() - start) / 60,
        "history": history,
    }
    print(summary)

if __name__ == "__main__":
    main()
