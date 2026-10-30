"""
train.py

Train, validate, tune, test and save the models of the ISIC melanoma Siamese classifier.

  --model baseline: ResNet-18 classifier trained with class-weighted cross-entropy.
  --model siamese:  Siamese network trained on image pairs with contrastive loss,
                    validated by k-nearest-neighbour classification against the
                    embedded training images.

After training, the best checkpoint is tuned on validation only (temperature for the
baseline; k and a plain or class-balanced kNN vote for the Siamese model; the reject
threshold for both) and then tested once on the test split. Every output (weights,
curves, tuned settings, test results, per-image predictions and plots) goes to outputs/.

Usage example:
    python train.py --model baseline
    python train.py --model siamese
    python train.py --model siamese --skip_training   # re-tune and re-test the saved checkpoint
    python train.py --model siamese --test_only       # re-test with the saved tuned settings
"""

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")  # save plots to file without needing a display (e.g. on Rangpur)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from dataset import ID_SOURCES, LesionDataset, PairDataset, build_transforms, get_splits
from modules import BaselineClassifier, ContrastiveLoss, SiameseNet, count_parameters
from metrics import (calibration, confidence_and_correct, discrimination_metrics, plot_k_tuning,
                     plot_reliability, plot_risk_coverage, plot_roc_pr, risk_coverage,
                     single_image_latency_ms)

OUT_DIR = "outputs"
DATA_ROOT = "data"

MODEL_CHOICE = "baseline"  # "baseline" or "siamese"
EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-4
PATIENCE = 5  # early stopping: stop if val AUROC does not improve
NUM_WORKERS = 4  # for DataLoader
SEED = 0  # for reproducibility
MARGIN = 1.0  # for contrastive loss
K = 5  # for kNN classification of validation images
TEST = False

RECALL_TARGET = 0.95
K_CANDIDATES = (3, 5, 10, 15, 20, 25, 30, 50, 75, 100, 150, 200) 
def parse_args():
    """
    Read the command-line options and copy them into the module-level settings.

    Returns:
        argparse.Namespace: The parsed options, including the --test_only and
            --skip_training flags that main() checks.
    """
    global MODEL_CHOICE, EPOCHS, BATCH_SIZE, PATIENCE, NUM_WORKERS, SEED, MARGIN

    parser = argparse.ArgumentParser(description="Train the melanoma classifiers.")
    parser.add_argument("--model", choices=["baseline", "siamese"], default=MODEL_CHOICE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--margin", type=float, default=MARGIN, help="contrastive loss margin")
    parser.add_argument("--test_only", action="store_true",
                        help="skip training: re-test the saved checkpoint with its saved validation settings")
    parser.add_argument("--skip_training", action="store_true",
                        help="skip training: re-tune the saved checkpoint on validation, then test it")

    args = parser.parse_args()
    MODEL_CHOICE, EPOCHS, BATCH_SIZE, PATIENCE, NUM_WORKERS, SEED, MARGIN = (
        args.model, args.epochs, args.batch_size,
        args.patience, args.num_workers, args.seed, args.margin
    )

    return args

def pick_device():
    """
    Best available device: CUDA (Rangpur), MPS (Mac GPU), else CPU.

    Returns:
        torch.device: The device to train and evaluate on.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class PeakMemory:
    """
    Tracks peak GPU memory during training, for the README's resource profiling table.

    CUDA: PyTorch records the peak itself (torch.cuda.max_memory_allocated).
    MPS (Apple GPU): there is no built-in peak, so update() is called after every
    training batch and keeps the largest value of torch.mps.driver_allocated_memory().
    """

    def __init__(self, device):
        self.device = device
        self.peak_bytes = 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

    def update(self):
        """Record the current MPS memory if it is a new peak (CUDA tracks its own peak)."""
        if self.device.type == "mps":
            self.peak_bytes = max(self.peak_bytes, torch.mps.driver_allocated_memory())

    def peak_gib(self):
        """
        Peak memory in GiB, or None on CPU.

        Returns:
            float or None: Peak memory in GiB, or None on CPU.
        """
        if self.device.type == "cuda":
            return torch.cuda.max_memory_allocated() / 2**30
        if self.device.type == "mps":
            return self.peak_bytes / 2**30
        return None


def sample_split(df, split, size, seed):
    """
    The rows of one split, or a reproducible random sample of `size` of them.

    Sampling the table (rather than a Dataset) lets every dataset built from it, single
    images, pairs and the kNN reference bank, share the same smaller set for smoke tests.

    Args:
        df (pd.DataFrame): All rows, with a "split" column.
        split (str): "train", "val" or "test".
        size (int or None): Number of rows to sample; None keeps the whole split.
        seed (int): Random seed for the sample.

    Returns:
        pd.DataFrame: The rows of the split, or a sample of them.
    """
    rows = df[df["split"] == split]
    if size is None or size >= len(rows):
        return rows
    # added extra for smoke test
    return rows.sample(n=size, random_state=seed)


def class_weights(targets):
    """
    Inverse-frequency weights so each class contributes equally to the loss.

    weight_c = N / (2 * N_c). With ~21% melanoma this gives about [0.64, 2.35], so a
    missed melanoma costs ~3.7x more than a misclassified benign lesion.

    Args:
        targets (torch.Tensor): Training labels [N], 0 = benign, 1 = melanoma.

    Returns:
        torch.Tensor: One weight per class [2].
    """
    counts = torch.bincount(targets, minlength=2).float()
    return len(targets) / (2 * counts)


def train_one_epoch(model, loader, loss_fn, optimizer, device, memory):
    """
    One pass over the training data. Returns the mean training loss.

    Args:
        model (BaselineClassifier): The model being trained.
        loader (DataLoader): Training images.
        loss_fn (nn.Module): Class-weighted cross-entropy.
        optimizer (torch.optim.Optimizer): The optimiser.
        device (torch.device): Device to train on.
        memory (PeakMemory): Peak GPU memory tracker, updated after every batch.

    Returns:
        float: Mean training loss over the epoch.
    """
    model.train()
    total_loss, n = 0.0, 0
    for images, targets, _ in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(images), targets)
        loss.backward()
        optimizer.step()
        memory.update()
        total_loss += loss.item() * len(targets)
        n += len(targets)
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    """
    Mean loss, AUROC and accuracy on a validation loader.

    AUROC uses the softmax probability of melanoma, so it does not depend on any threshold.

    Args:
        model (BaselineClassifier): The model to evaluate.
        loader (DataLoader): Validation images.
        loss_fn (nn.Module): The training loss.
        device (torch.device): Device to run on.

    Returns:
        dict: {"loss", "auroc", "accuracy"}.
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


def train_one_epoch_siamese(model, loader, loss_fn, optimizer, device, memory):
    """
    One pass over the training pairs. Returns the mean contrastive loss.

    Args:
        model (SiameseNet): The model being trained.
        loader (DataLoader): Training pairs (image a, image b, same).
        loss_fn (ContrastiveLoss): The contrastive loss.
        optimizer (torch.optim.Optimizer): The optimiser.
        device (torch.device): Device to train on.
        memory (PeakMemory): Peak GPU memory tracker, updated after every batch.

    Returns:
        float: Mean contrastive loss over the epoch.
    """
    model.train()
    total_loss, n = 0.0, 0
    for img_a, img_b, same in loader:
        img_a, img_b, same = img_a.to(device), img_b.to(device), same.to(device)
        optimizer.zero_grad()
        emb_a, emb_b = model(img_a, img_b)
        loss = loss_fn(emb_a, emb_b, same)
        loss.backward()
        optimizer.step()
        memory.update()
        total_loss += loss.item() * len(same)
        n += len(same)
    return total_loss / n

@torch.no_grad()
def embed_all(model, loader, device):
    """
    Embed every image in a LesionDataset loader. Returns (embeddings [N, 128], labels [N]).

    Args:
        model (SiameseNet): The Siamese network.
        loader (DataLoader): Single images (a LesionDataset loader).
        device (torch.device): Device to embed on.

    Returns:
        tuple: (embeddings [N, 128], labels [N]), both on the CPU.
    """
    model.eval()
    embeddings, labels = [], []
    for images, targets, _ in loader:
        embeddings.append(model.embed(images.to(device)).cpu())
        labels.append(targets)
    return torch.cat(embeddings), torch.cat(labels)


def knn_melanoma_probability(query, bank, bank_labels, k, melanoma_weight=1.0):
    """
    Distance-weighted kNN: for each query embedding, the weighted share of its k nearest
    reference embeddings that are melanoma. Closer neighbours count more (weight 1/d).

    melanoma_weight multiplies the vote of every melanoma neighbour. With
    balanced_melanoma_weight() both classes count equally, like the baseline's
    class-weighted loss; with 1.0 the ~79% benign bank pulls ambiguous cases towards benign.

    Args:
        query (torch.Tensor): Embeddings to classify [Q, D].
        bank (torch.Tensor): Reference embeddings [N, D].
        bank_labels (torch.Tensor): Labels of the reference embeddings [N].
        k (int): Number of neighbours.
        melanoma_weight (float): Vote multiplier for melanoma neighbours.

    Returns:
        torch.Tensor: Melanoma score in [0, 1] per query [Q], used like the baseline's
            softmax probability.
    """
    distances = torch.cdist(query, bank)                         # [Q, N]
    nearest_d, nearest_i = distances.topk(k, dim=1, largest=False)
    is_melanoma = bank_labels[nearest_i].float()                 # [Q, k]
    weights = 1.0 / (nearest_d + 1e-6) * (1 + (melanoma_weight - 1) * is_melanoma)
    return (weights * is_melanoma).sum(dim=1) / weights.sum(dim=1)


def balanced_melanoma_weight(bank_labels):
    """
    Melanoma vote weight that makes both classes count equally: n_benign / n_melanoma.

    Args:
        bank_labels (torch.Tensor): Labels of the reference bank [N].

    Returns:
        float: n_benign / n_melanoma.
    """
    n_melanoma = int(bank_labels.sum())
    return (len(bank_labels) - n_melanoma) / n_melanoma


def training_bank(model, train_df, device):
    """
    Embeddings of every training image: the reference bank the kNN vote searches.

    The bank is cached in outputs/siamese_bank.pt and reused while siamese_best.pt is
    unchanged (same modification time) and the training split has the same size, so
    tuning, testing and predict.py do not re-embed the training images each time.

    Args:
        model (SiameseNet): The trained Siamese network.
        train_df (pd.DataFrame): The training rows.
        device (torch.device): Device to embed on.

    Returns:
        tuple: (embeddings [N, 128], labels [N]), both on the CPU.

    Raises:
        FileNotFoundError: If outputs/siamese_best.pt does not exist.
    """
    weights_time = os.path.getmtime(os.path.join(OUT_DIR, "siamese_best.pt"))
    cache_path = os.path.join(OUT_DIR, "siamese_bank.pt")
    if os.path.exists(cache_path):
        try:
            cache = torch.load(cache_path)
        except Exception:  # e.g. an old cache in a different format: rebuild it
            cache = {}
        if cache.get("weights_time") == weights_time and len(cache["labels"]) == len(train_df):
            return cache["embeddings"], cache["labels"]
    print("embedding the training images (reference bank) ...")
    embeddings, labels = embed_all(model, loader(LesionDataset(train_df, "train")), device)
    torch.save({"weights_time": weights_time, "embeddings": embeddings, "labels": labels}, cache_path)
    return embeddings, labels


@torch.no_grad()
def evaluate_siamese(model, bank_loader, val_loader, val_pair_loader, loss_fn, device, k):
    """
    Validate the Siamese model the way it will be used: embed the training images as a
    reference bank, classify validation images by kNN, and compute AUROC/accuracy.

    The loss is the contrastive loss on a fixed set of validation pairs, for the curves.

    Args:
        model (SiameseNet): The model to evaluate.
        bank_loader (DataLoader): Training images (the reference bank).
        val_loader (DataLoader): Validation images.
        val_pair_loader (DataLoader): Fixed validation pairs, for the loss.
        loss_fn (ContrastiveLoss): The contrastive loss.
        device (torch.device): Device to run on.
        k (int): Number of neighbours in the vote.

    Returns:
        dict: {"loss", "auroc", "accuracy"}.
    """
    model.eval()
    total_loss, n = 0.0, 0
    for img_a, img_b, same in val_pair_loader:
        img_a, img_b, same = img_a.to(device), img_b.to(device), same.to(device)
        total_loss += loss_fn(*model(img_a, img_b), same).item() * len(same)
        n += len(same)

    bank, bank_labels = embed_all(model, bank_loader, device)
    query, labels = embed_all(model, val_loader, device)
    probs = knn_melanoma_probability(query, bank, bank_labels, k)
    return {
        "loss": total_loss / n,
        "auroc": roc_auc_score(labels, probs),
        "accuracy": ((probs >= 0.5).long() == labels).float().mean().item(),
    }


def plot_history(history, path):
    """
    Training/validation loss and validation AUROC per epoch.

    Args:
        history (list): One dict per epoch with train_loss, val_loss and val_auroc.
        path (str): Where to save the PNG.
    """
    epochs = [h["epoch"] for h in history]
    fig, (ax_loss, ax_auc) = plt.subplots(1, 2, figsize=(10, 4))
    ax_loss.plot(epochs, [h["train_loss"] for h in history], marker="o", label="train")
    ax_loss.plot(epochs, [h["val_loss"] for h in history], marker="o", label="validation")

    loss_label = "weighted cross-entropy" if MODEL_CHOICE == "baseline" else "Contrastive Loss"
    ax_loss.set(xlabel="epoch", ylabel=loss_label, title="Loss")
    ax_loss.legend()
    ax_auc.plot(epochs, [h["val_auroc"] for h in history], marker="o", color="tab:green")
    ax_auc.set(xlabel="epoch", ylabel="AUROC", title="Validation AUROC", ylim=(0.5, 1.0))  # 0.5 = chance
    for ax in (ax_loss, ax_auc):
        ax.xaxis.get_major_locator().set_params(integer=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def loader(dataset, shuffle=False):
    """
    DataLoader with the batch size and worker count used everywhere in the project.

    Args:
        dataset (torch.utils.data.Dataset): A LesionDataset or PairDataset.
        shuffle (bool): Shuffle each epoch (True only for training).

    Returns:
        DataLoader: The loader.
    """
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=NUM_WORKERS)

def setup_baseline(train_set, device):
    """
    Model, loss and training loader for the baseline.

    Args:
        train_set (pd.DataFrame): Training rows.
        device (torch.device): Device to put the model on.

    Returns:
        dict: The model, loss_fn, train_loader and a details string for the log.
    """
    weights = class_weights(torch.tensor(train_set["target"].values))
    return {
        "model": BaselineClassifier(pretrained=True).to(device),
        "loss_fn": nn.CrossEntropyLoss(weight=weights.to(device)),
        "train_loader": loader(LesionDataset(train_set, "train", build_transforms(train=True)), True),
        "details": f"class weights {[round(w, 2) for w in weights.tolist()]}",
    }


def setup_siamese(train_set, val_set, device):
    """
    Model, loss and loaders for the Siamese network.

    Args:
        train_set (pd.DataFrame): Training rows.
        val_set (pd.DataFrame): Validation rows.
        device (torch.device): Device to put the model on.

    Returns:
        dict: The model, loss_fn, training pairs and loaders, validation pair loader,
            reference bank loader and a details string for the log.
    """
    # Pair sampling balances the classes, so no class weights are needed.
    train_pairs = PairDataset(train_set, "train", seed=SEED)
    return {
        "model": SiameseNet(pretrained=True).to(device),
        "loss_fn": ContrastiveLoss(margin=MARGIN),
        "train_pairs": train_pairs,
        "train_loader": loader(train_pairs, shuffle=True),
        "val_pair_loader": loader(PairDataset(val_set, "val", transform=build_transforms(train=False), seed=SEED)),
        "bank_loader": loader(LesionDataset(train_set, "train")),
        "details": f"pairs/epoch {len(train_pairs)} | margin {MARGIN} | k {K}",
    }


def save_training_summary(summary, weights_path):
    """
    Save the history (for logs and the README), plot the curves and print a summary line.

    Args:
        summary (dict): Training settings, timings, peak memory and the per-epoch history.
        weights_path (str): Path of the best checkpoint.
    """
    with open(os.path.join(OUT_DIR, f"{MODEL_CHOICE}_history.json"), "w") as f:
        json.dump(summary, f, indent=2)
    plot_history(summary["history"], os.path.join(OUT_DIR, f"{MODEL_CHOICE}_curves.png"))
    peak, device = summary["peak_gpu_memory_gib"], summary["device"]
    print(f"\nbest val AUROC {summary['best_val_auroc']:.3f} | "
          f"total {summary['total_minutes']:.1f} min | peak GPU memory "
          f"{'n/a (CPU)' if peak is None else f'{peak:.2f} GiB ({device})'}")

def train_epochs(setup, optimizer, val_loader, device):
    """
    Train with early stopping on val AUROC. Returns the path of the best checkpoint.

    Args:
        setup (dict): Output of setup_baseline or setup_siamese.
        optimizer (torch.optim.Optimizer): The optimiser.
        val_loader (DataLoader): Validation images.
        device (torch.device): Device to train on.

    Returns:
        str: Path of the best checkpoint.
    """
    history = []
    weights_path = os.path.join(OUT_DIR, f"{MODEL_CHOICE}_best.pt")
    start = time.time()
    memory = PeakMemory(device)
    best_auroc, epochs_since_best = -1.0, 0
    model = setup["model"]
    loss_fn = setup["loss_fn"]

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        if MODEL_CHOICE == "baseline":
            train_loss = train_one_epoch(model, setup["train_loader"], loss_fn, optimizer, device, memory)
            val = evaluate(model, val_loader, loss_fn, device)
        else:
            setup["train_pairs"].set_epoch(epoch)  # fresh pairs every epoch
            train_loss = train_one_epoch_siamese(model, setup["train_loader"], loss_fn, optimizer, device, memory)
            val = evaluate_siamese(model, setup["bank_loader"], val_loader, setup["val_pair_loader"], loss_fn, device, K)
        
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val["loss"],
                    "val_auroc": val["auroc"], "val_accuracy": val["accuracy"],
                    "seconds": time.time() - epoch_start})
        
        print(f"epoch {epoch} | train loss {train_loss:.3f} | val loss {val['loss']:.3f} | "
            f"val AUROC {val['auroc']:.3f} | val acc {val['accuracy']:.3f} | {history[-1]['seconds']:.0f}s")
            
        if val["auroc"] > best_auroc:
            best_auroc, epochs_since_best = val["auroc"], 0
            torch.save(model.state_dict(), weights_path)
        else:
            epochs_since_best += 1
            if epochs_since_best >= PATIENCE:
                print(f"early stopping: no improvement for {PATIENCE} epochs")
                break
    
    # training_k is the k used to choose the best epoch; the tuned k is saved by tune_model.
    save_training_summary({
        "model": MODEL_CHOICE, "device": str(device), "training_k": K, "lr": LR,
        "best_val_auroc": best_auroc, "parameters": count_parameters(model),
        "total_minutes": (time.time() - start) / 60,
        "peak_gpu_memory_gib": memory.peak_gib(), "history": history,
    }, weights_path)
    return weights_path


def print_heading(title):
    """
    A blank line and a heading, so the training / tuning / testing output is easy to scan.

    Args:
        title (str): The heading text.
    """
    print(f"\n{'=' * 20} {title} {'=' * 20}\n")


def main():
    """
    Run the full pipeline for one model: train (unless skipped), tune on validation,
    test once, and draw the baseline vs Siamese test plots when both have been tested.

    Raises:
        FileNotFoundError: With --test_only or --skip_training, if the saved checkpoint or
            tuned settings are missing.
    """
    args = parse_args()
    torch.manual_seed(SEED)
    device = pick_device()
    os.makedirs(OUT_DIR, exist_ok=True)

    df = get_splits(DATA_ROOT)
    train_set = sample_split(df, "train", None, SEED)
    val_set = sample_split(df, "val", None, SEED)
    test_set = sample_split(df, "test", None, SEED)

    if args.test_only:
        model = load_model(MODEL_CHOICE, device)
        with open(os.path.join(OUT_DIR, f"{MODEL_CHOICE}_tuned.json")) as f:
            tuned = json.load(f)
    elif args.skip_training:
        model = load_model(MODEL_CHOICE, device)
        tuned = tune_model(model, device, train_set, loader(LesionDataset(val_set, "val")))
    else:
        val_loader = loader(LesionDataset(val_set, "val"))
        if MODEL_CHOICE == "baseline":
            setup = setup_baseline(train_set, device)
        else:
            setup = setup_siamese(train_set, val_set, device)

        model = setup["model"]
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        print_heading(f"TRAINING {MODEL_CHOICE}")
        print(f"{MODEL_CHOICE} | device {device} | train {len(train_set)} | val {len(val_set)} | "
              f"{setup['details']} | parameters {count_parameters(model):,}")
        weights_path = train_epochs(setup, optimizer, val_loader, device)

        # Tune and test the best checkpoint, not the weights from the last epoch.
        model.load_state_dict(torch.load(weights_path, map_location=device))
        tuned = tune_model(model, device, train_set, val_loader)

    test_model(model, device, train_set, test_set, tuned)
    # Baseline vs Siamese plots on the test split, once both models have been tested.
    plot_test_comparison()


def load_model(name, device):
    """
    Rebuild the architecture and load the best weights saved by train_epochs.

    Args:
        name (str): "baseline" or "siamese".
        device (torch.device): Device to load the model onto.

    Returns:
        nn.Module: The model in eval mode.

    Raises:
        FileNotFoundError: If outputs/<name>_best.pt does not exist.
    """
    model = BaselineClassifier(pretrained=False) if name == "baseline" else SiameseNet(pretrained=False)
    model.load_state_dict(torch.load(os.path.join(OUT_DIR, f"{name}_best.pt"), map_location=device))
    return model.to(device).eval()


def closure(optimizer, logits, log_t, labels):
    """
    One LBFGS evaluation: the cross-entropy of the logits divided by T = exp(log_t).

    Args:
        optimizer (torch.optim.LBFGS): The optimiser fitting log_t.
        logits (torch.Tensor): Validation logits [N, 2].
        log_t (torch.Tensor): Log of the temperature, the only parameter being fitted.
        labels (torch.Tensor): Validation labels [N].

    Returns:
        torch.Tensor: The loss, detached, as LBFGS expects.
    """
    optimizer.zero_grad()
    loss = torch.nn.functional.cross_entropy(logits / log_t.exp(), labels)
    loss.backward()
    return loss.detach()
    
def fit_temperature(logits, labels):
    """
    Temperature scaling (Guo et al., 2017): find the single T that makes the softmax of
    logits / T best match the validation labels (lowest cross-entropy). T rescales
    confidence only, so predictions and AUROC do not change.

    Args:
        logits (torch.Tensor): Validation logits [N, 2].
        labels (torch.Tensor): Validation labels [N].

    Returns:
        float: The fitted temperature T (> 0).
    """
    log_t = torch.zeros(1, requires_grad=True)  # optimise log T so that T stays positive
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)
    # LBFGS calls the closure itself (several times per step), so it must take no arguments.
    optimizer.step(lambda: closure(optimizer, logits, log_t, labels))
    return log_t.exp().item()


@torch.no_grad()
def baseline_logits(model, data_loader, device):
    """
    Raw 2-class logits and labels for every image in a loader.

    Args:
        model (BaselineClassifier): The trained baseline.
        data_loader (DataLoader): Images to score.
        device (torch.device): Device to run on.

    Returns:
        tuple: (logits [N, 2], labels [N]), both on the CPU.
    """
    logits, labels = [], []
    for images, targets, _ in data_loader:
        logits.append(model(images.to(device)).cpu())
        labels.append(targets)
    return torch.cat(logits), torch.cat(labels)

def melanoma_probability_from_logits(logits, temperature=1.0):
    """
    Softmax probability of melanoma from the baseline's logits.

    Args:
        logits (torch.Tensor): Logits [N, 2].
        temperature (float): Temperature T; 1.0 gives the uncalibrated softmax.

    Returns:
        torch.Tensor: Melanoma probability per image [N].
    """
    return (logits / temperature).softmax(dim=1)[:, 1]


def handle_baseline_eval(model, val_loader, result, device):
    """
    Tune the baseline on validation: fit the temperature and report ECE before and after.

    Args:
        model (BaselineClassifier): The best baseline checkpoint.
        val_loader (DataLoader): Validation images.
        result (dict): Tuned settings so far; temperature and ECE values are added.
        device (torch.device): Device to run on.

    Returns:
        tuple: (calibrated validation melanoma probabilities, validation labels, result).
    """
    val_logits, val_labels = baseline_logits(model, val_loader, device)
    
    # bc baseline produces logits, the temperature needs to be adjusted
    temperature = fit_temperature(val_logits, val_labels)
    val_probs = melanoma_probability_from_logits(val_logits, temperature)

    # Before/after: T only changes how confident the probabilities are, so ECE should drop.
    labels = val_labels.numpy()
    ece_before, _ = calibration(*confidence_and_correct(labels, melanoma_probability_from_logits(val_logits).numpy()))
    ece_after, _ = calibration(*confidence_and_correct(labels, val_probs.numpy()))
    print(f"temperature T = {temperature:.3f} | val ECE {ece_before:.3f} (T = 1) -> {ece_after:.3f} (tuned)")

    result["temperature"] = temperature
    result["val_ece_before_temperature"], result["val_ece_after_temperature"] = ece_before, ece_after
    return val_probs, val_labels, result


def handle_siamese_eval(model, val_loader, result, device, train_df):
    """
    Tune the kNN on validation: k, and whether the vote is plain or class-balanced.

    The chosen setting refers the fewest val cases while still meeting the reject rule's
    melanoma-recall target (README criterion 4); ties are broken by AURC. AURC alone would
    always prefer the plain vote, which never meets the recall target (every case referred).

    Args:
        model (SiameseNet): The best Siamese checkpoint.
        val_loader (DataLoader): Validation images.
        result (dict): Tuned settings so far; k, vote, melanoma weight and k_scores are added.
        device (torch.device): Device to run on.
        train_df (pd.DataFrame): Training rows (the reference bank).

    Returns:
        tuple: (validation melanoma scores with the chosen setting, validation labels, result).
    """
    bank, bank_labels = training_bank(model, train_df, device)
    val_embeddings, val_labels = embed_all(model, val_loader, device)
    labels = val_labels.numpy()
    votes = {"plain": 1.0, "balanced": balanced_melanoma_weight(bank_labels)}

    k_scores = {vote: {} for vote in votes}
    for vote, melanoma_weight in votes.items():
        for k in K_CANDIDATES:
            probs = knn_melanoma_probability(val_embeddings, bank, bank_labels, k, melanoma_weight).numpy()
            confidence, correct = confidence_and_correct(labels, probs)
            _, _, aurc = risk_coverage(confidence, correct)
            reject = apply_reject_rule(labels, probs, choose_threshold(labels, probs))
            k_scores[vote][k] = {
                "val_aurc": aurc,
                "val_auroc": float(roc_auc_score(labels, probs)),
                "val_melanoma_recall": float((probs[labels == 1] >= 0.5).mean()),
                "val_referred": reject["referred"],
            }
            print(f"{vote:>8} vote | k {k:>3} | val AURC {aurc:.3f} | val AUROC {k_scores[vote][k]['val_auroc']:.3f} | "
                  f"melanoma recall {k_scores[vote][k]['val_melanoma_recall']:.3f} | referred {reject['referred']:.1%}")

    def selection_key(vote_and_k):
        """Sort key for a (vote, k) setting: fewest referred first, then lowest AURC."""
        scores = k_scores[vote_and_k[0]][vote_and_k[1]]
        return scores["val_referred"], scores["val_aurc"]

    best_vote, best_k = min(((vote, k) for vote in votes for k in K_CANDIDATES), key=selection_key)
    # Before/after: K with a plain vote is what was used during training to pick the best epoch.
    if K in k_scores["plain"]:
        before, after = k_scores["plain"][K], k_scores[best_vote][best_k]
        print(f"\nplain vote, k {K} (training) -> {best_vote} vote, k {best_k} (tuned) | "
              f"melanoma recall {before['val_melanoma_recall']:.3f} -> {after['val_melanoma_recall']:.3f} | "
              f"referred {before['val_referred']:.1%} -> {after['val_referred']:.1%}")

    plot_k_tuning(k_scores, best_k, best_vote, os.path.join(OUT_DIR, "siamese_k_tuning.png"))
    result["k"], result["vote"], result["melanoma_weight"] = best_k, best_vote, votes[best_vote]
    result["k_scores"] = k_scores
    val_probs = knn_melanoma_probability(val_embeddings, bank, bank_labels, best_k, votes[best_vote])
    return val_probs, val_labels, result


def tune_model(model, device, train_df, val_loader):
    """
    Fix every setting on validation data, then choose the reject threshold and save it all
    to outputs/<model>_tuned.json for the test step and predict.py.

    Args:
        model (nn.Module): The best checkpoint of the current model.
        device (torch.device): Device to run on.
        train_df (pd.DataFrame): Training rows (the Siamese reference bank).
        val_loader (DataLoader): Validation images.

    Returns:
        dict: The tuned settings (temperature, or k, vote and melanoma weight) and the
            reject rule's validation outcome, including its threshold.
    """
    print_heading(f"VALIDATION TUNING ({MODEL_CHOICE})")
    result = {}
    start = time.time()
    if MODEL_CHOICE == "baseline":
        val_probs, val_labels, result = handle_baseline_eval(
            model, val_loader, result, device)
    else:
        val_probs, val_labels, result = handle_siamese_eval(
                    model, val_loader, result, device, train_df
                )
    result["scoring_seconds"] = time.time() - start
    labels, probs = val_labels.numpy(), val_probs.numpy()

    #check for threshbhold, k, r
    threshold = choose_threshold(labels, probs)
    result["reject_rule_validation"] = apply_reject_rule(labels, probs, threshold)
    reject = result["reject_rule_validation"]
    threshold_text = "none (refer every case)" if reject["threshold"] is None else f"{reject['threshold']:.3f}"
    print(f"\nreject threshold {threshold_text} | val referred {reject['referred']:.1%}")

    # Save the validation-tuned settings so the test step and predict.py use the tuned k,
    # temperature and threshold rather than the defaults the model was trained with.
    with open(os.path.join(OUT_DIR, f"{MODEL_CHOICE}_tuned.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result

def apply_reject_rule(labels, probs, threshold):
    """
    Outcome of the reject rule on one split.

    Args:
        labels (np.ndarray): True labels [N].
        probs (np.ndarray): Melanoma probabilities [N].
        threshold (float): Confidence threshold; inf refers every case.

    Returns:
        dict: Threshold, share auto-decided and referred, accuracy and melanoma recall
            among auto-decided cases, melanomas missed and system sensitivity.
    """
    confidence, correct = confidence_and_correct(labels, probs)
    preds = (probs >= 0.5).astype(int)
    auto = confidence >= threshold
    auto_melanoma = auto & (labels == 1)
    missed = int((auto_melanoma & (preds == 0)).sum())
    return {
        "threshold": threshold if np.isfinite(threshold) else None,  # None = refer every case
        "auto_decided": float(auto.mean()),
        "referred": float(1 - auto.mean()),
        "accuracy_auto_decided": float(correct[auto].mean()) if auto.any() else None,
        "melanoma_recall_auto_decided": float((preds[auto_melanoma] == 1).mean()) if auto_melanoma.any() else None,
        "melanomas_missed_by_automation": missed,
        # If every referred case is correctly diagnosed by the specialist:
        "system_melanoma_sensitivity": float(1 - missed / max(int(labels.sum()), 1)),
    }
def choose_threshold(labels, probs, target_recall=RECALL_TARGET):
    """
    Reject rule, fitted on VALIDATION: auto-decide a case only if its confidence >= t,
    otherwise refer it. Returns the smallest t (i.e. the most cases automated) for which
    melanoma recall among the auto-decided cases is >= target_recall.

    Args:
        labels (np.ndarray): Validation labels [N].
        probs (np.ndarray): Validation melanoma probabilities [N].
        target_recall (float): Required melanoma recall among auto-decided cases.

    Returns:
        float: The threshold, or inf if no threshold reaches the target.
    """
    confidence, _ = confidence_and_correct(labels, probs)
    preds = (probs >= 0.5).astype(int)
    for t in np.unique(confidence):  # ascending: lower t automates more cases
        auto_melanoma = (confidence >= t) & (labels == 1)
        if auto_melanoma.any() and (preds[auto_melanoma] == 1).mean() >= target_recall:
            return float(t)
    return float("inf")  # no threshold reaches the target: refer every case


@torch.no_grad()
def melanoma_scores(name, model, data_loader, device, tuned, train_df):
    """
    Melanoma probability for every image in a loader, using the tuned settings: the
    temperature-scaled softmax for the baseline, the tuned kNN vote for the Siamese model.

    Args:
        name (str): "baseline" or "siamese".
        model (nn.Module): The trained model.
        data_loader (DataLoader): Images to score.
        device (torch.device): Device to run on.
        tuned (dict): Settings from tune_model.
        train_df (pd.DataFrame): Training rows (the Siamese reference bank).

    Returns:
        tuple: (labels, melanoma probabilities), both NumPy arrays [N].
    """
    model.eval()
    if name == "baseline":
        logits, labels = baseline_logits(model, data_loader, device)
        probs = melanoma_probability_from_logits(logits, tuned["temperature"])
    else:
        bank, bank_labels = training_bank(model, train_df, device)
        query, labels = embed_all(model, data_loader, device)
        probs = knn_melanoma_probability(query, bank, bank_labels, tuned["k"], tuned.get("melanoma_weight", 1.0))
    return labels.numpy(), probs.numpy()


def one_image_predictor(name, model, device, tuned, train_df):
    """
    A function that scores one image the way the deployed tool would, for the latency test.

    The Siamese bank is embedded once here, so the timed call includes only the image's
    own embedding and the kNN search.

    Args:
        name (str): "baseline" or "siamese".
        model (nn.Module): The trained model.
        device (torch.device): Device to run on.
        tuned (dict): Settings from tune_model.
        train_df (pd.DataFrame): Training rows (the Siamese reference bank).

    Returns:
        callable: Maps an image batch [1, 3, 224, 224] to its melanoma probability.
    """
    if name == "baseline":
        return lambda image: melanoma_probability_from_logits(model(image), tuned["temperature"])
    bank, bank_labels = training_bank(model, train_df, device)
    k, melanoma_weight = tuned["k"], tuned.get("melanoma_weight", 1.0)
    return lambda image: knn_melanoma_probability(model.embed(image).cpu(), bank, bank_labels, k, melanoma_weight)


def results_by_source(sources, labels, probs, threshold):
    """
    Discrimination and reject-rule results for each ID source (IP / HAM / dummy) separately,
    to check whether performance depends on the data source (README risk 2).

    Args:
        sources (np.ndarray): ID source of each image ("IP", "HAM" or "dummy").
        labels (np.ndarray): True labels [N].
        probs (np.ndarray): Melanoma probabilities [N].
        threshold (float): The reject threshold chosen on validation.

    Returns:
        dict: Per source, {"discrimination": ..., "reject_rule": ...}.
    """
    results = {}
    for source in ID_SOURCES:
        mask = sources == source  # True for the test images from this source
        if mask.any():
            results[source] = {
                "discrimination": discrimination_metrics(labels[mask], probs[mask]),
                "reject_rule": apply_reject_rule(labels[mask], probs[mask], threshold),
            }
    return results


def test_model(model, device, train_df, test_df, tuned):
    """
    Evaluate the model once on the test split with the settings fixed on validation.

    Saves outputs/<model>_test_results.json (all metrics, overall and per ID source) and
    outputs/<model>_test_predictions.csv (one row per test image), and prints a summary.

    Args:
        model (nn.Module): The trained model.
        device (torch.device): Device to run on.
        train_df (pd.DataFrame): Training rows (the Siamese reference bank).
        test_df (pd.DataFrame): Test rows.
        tuned (dict): Settings from tune_model.

    Returns:
        dict: The test results that were saved.
    """
    print_heading(f"TESTING {MODEL_CHOICE}")
    test_dataset = LesionDataset(test_df, "test")
    labels, probs = melanoma_scores(MODEL_CHOICE, model, loader(test_dataset), device, tuned, train_df)
    predict_one = one_image_predictor(MODEL_CHOICE, model, device, tuned, train_df)
    confidence, correct = confidence_and_correct(labels, probs)
    threshold = tuned["reject_rule_validation"]["threshold"]
    threshold = float("inf") if threshold is None else threshold 
    ece, calibration_table = calibration(confidence, correct)
    _, _, test_aurc = risk_coverage(confidence, correct)

    sources = test_df["id_source"].values
    result = {
        "model": MODEL_CHOICE, "k": tuned.get("k"), "temperature": tuned.get("temperature"),
        "discrimination": discrimination_metrics(labels, probs),
        "confidence_auroc": float(roc_auc_score(correct, confidence)) if 0 < correct.mean() < 1 else None,
        "ece": ece, 
        "calibration_table": calibration_table, 
        "aurc": test_aurc,
        "reject_rule": apply_reject_rule(labels, probs, threshold),
        "latency_ms_per_image": single_image_latency_ms(predict_one, test_dataset, device),
        "by_source": results_by_source(sources, labels, probs, threshold),
    }
    with open(os.path.join(OUT_DIR, f"{MODEL_CHOICE}_test_results.json"), "w") as f:
        json.dump(result, f, indent=2)

    auto = confidence >= threshold
    test_df.assign(prob_melanoma=probs, predicted=(probs >= 0.5).astype(int), confidence=confidence,
                   correct=correct, referred=~auto)[
        ["isic_id", "patient_id", "id_source", "target", "prob_melanoma", "predicted", "confidence",
         "correct", "referred"]
    ].to_csv(os.path.join(OUT_DIR, f"{MODEL_CHOICE}_test_predictions.csv"), index=False)

    d, reject = result["discrimination"], result["reject_rule"]
    recall_auto = reject["melanoma_recall_auto_decided"]
    print(f"discrimination | acc {d['accuracy']:.3f} | AUROC {d['auroc']:.3f} | PR-AUC {d['pr_auc']:.3f} | "
          f"macro-F1 {d['macro_f1']:.3f} | melanoma recall {d['recall_melanoma']:.3f}")
    print(f"confidence     | ECE {ece:.3f} | AURC {test_aurc:.3f}")
    print(f"reject rule    | referred {reject['referred']:.1%} | melanoma recall among auto-decided "
          f"{'n/a (nothing auto-decided)' if recall_auto is None else f'{recall_auto:.3f}'}")
    print(f"resources      | latency {result['latency_ms_per_image']:.1f} ms/image")
    return result


def plot_test_comparison():
    """
    Draw the baseline vs Siamese test plots (ROC and PR curves, reliability diagram,
    risk-coverage) from both models' saved test predictions. Skipped with a message if
    either model has not been tested yet.
    """
    paths = {}
    for name in ("baseline", "siamese"):
        paths[name] = os.path.join(OUT_DIR, f"{name}_test_predictions.csv")
    
    if not all(os.path.exists(path) for path in paths.values()):
        print("joint test plots skipped: test the other model too")
        return
    
    scores, calibrations, curves = {}, {}, {}
    for name, path in paths.items():
        predictions = pd.read_csv(path)
        confidence, correct = predictions["confidence"].values, predictions["correct"].values.astype(bool)
        scores[name] = (predictions["target"].values, predictions["prob_melanoma"].values)
        ece, table = calibration(confidence, correct)
        calibrations[name] = {"test_ece": ece, "test_calibration_table": table}
        curves[name] = risk_coverage(confidence, correct)
    # all the plots
    plot_roc_pr(scores, os.path.join(OUT_DIR, "roc_pr_curves.png"))
    plot_reliability(calibrations, os.path.join(OUT_DIR, "reliability_diagram.png"))
    plot_risk_coverage(curves, os.path.join(OUT_DIR, "risk_coverage.png"))
    print(f"joint test plots in: {OUT_DIR}/roc_pr_curves.png, reliability_diagram.png, risk_coverage.png")


if __name__ == "__main__":
    main()
