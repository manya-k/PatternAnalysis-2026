"""
Train and validate the models, saving the best weights and the training curves.

  --model baseline: ResNet-18 classifier trained with class-weighted cross-entropy.
  --model siamese:  Siamese network trained on image pairs with contrastive loss,
                    validated by k-nearest-neighbour classification against the
                    embedded training images.

Usage example:
    python train.py --model baseline --subset 1000 --epochs 1
    python train.py --model siamese  --subset 1000 --epochs 1

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
from torch.utils.data import DataLoader

from dataset import LesionDataset, PairDataset, build_transforms, get_splits
from modules import BaselineClassifier, ContrastiveLoss, SiameseNet, count_parameters

OUT_DIR = "outputs"
DATA_ROOT = "data"

MODEL_CHOICE = "baseline"  # "baseline" or "siamese"
EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-4
PATIENCE = 5  # early stopping: stop if val AUROC does not improve
SUBSET = None  # for quick smoke tests, use a random subset of the training data
NUM_WORKERS = 4  # for DataLoader
SEED = 0  # for reproducibility
MARGIN = 1.0  # for contrastive loss
K = 5  # for kNN classification of validation images

def parse_args():
    global MODEL_CHOICE, EPOCHS, BATCH_SIZE, PATIENCE, SUBSET, NUM_WORKERS, SEED, MARGIN, K

    parser = argparse.ArgumentParser(description="Train the melanoma classifiers.")
    parser.add_argument("--model", choices=["baseline", "siamese"], default=MODEL_CHOICE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--subset", type=int, default=SUBSET)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--margin", type=float, default=MARGIN, help="contrastive loss margin")
    parser.add_argument("--k", type=int, default=K, help="k for kNN classification of validation images")

    args = parser.parse_args()
    MODEL_CHOICE, EPOCHS, BATCH_SIZE, PATIENCE, SUBSET, NUM_WORKERS, SEED, MARGIN, K = (
        args.model, args.epochs, args.batch_size,
        args.patience, args.subset, args.num_workers, args.seed, args.margin, args.k
    )

    return args

def pick_device():
    """Best available device: CUDA (Rangpur), MPS (Mac GPU), else CPU."""
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
        if self.device.type == "mps":
            self.peak_bytes = max(self.peak_bytes, torch.mps.driver_allocated_memory())

    def peak_gib(self):
        """Peak memory in GiB, or None on CPU."""
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
    """
    rows = df[df["split"] == split]
    if size is None or size >= len(rows):
        return rows
    return rows.sample(n=size, random_state=seed)


def class_weights(targets):
    """
    Inverse-frequency weights so each class contributes equally to the loss.

    weight_c = N / (2 * N_c). With ~21% melanoma this gives about [0.64, 2.35], so a
    missed melanoma costs ~3.7x more than a misclassified benign lesion.
    """
    counts = torch.bincount(targets, minlength=2).float()
    return len(targets) / (2 * counts)


def train_one_epoch(model, loader, loss_fn, optimizer, device, memory):
    """One pass over the training data. Returns the mean training loss."""
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
    """One pass over the training pairs. Returns the mean contrastive loss."""
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
    """Embed every image in a LesionDataset loader. Returns (embeddings [N, 128], labels [N])."""
    model.eval()
    embeddings, labels = [], []
    for images, targets, _ in loader:
        embeddings.append(model.embed(images.to(device)).cpu())
        labels.append(targets)
    return torch.cat(embeddings), torch.cat(labels)


def knn_melanoma_probability(query, bank, bank_labels, k):
    """
    Distance-weighted kNN: for each query embedding, the weighted share of its k nearest
    reference embeddings that are melanoma. Closer neighbours count more (weight 1/d).

    Returns a score in [0, 1] per query, used like the baseline's softmax probability.
    """
    distances = torch.cdist(query, bank)                         # [Q, N]
    nearest_d, nearest_i = distances.topk(k, dim=1, largest=False)
    weights = 1.0 / (nearest_d + 1e-6)
    is_melanoma = bank_labels[nearest_i].float()                 # [Q, k]
    return (weights * is_melanoma).sum(dim=1) / weights.sum(dim=1)


@torch.no_grad()
def evaluate_siamese(model, bank_loader, val_loader, val_pair_loader, loss_fn, device, k):
    """
    Validate the Siamese model the way it will be used: embed the training images as a
    reference bank, classify validation images by kNN, and compute AUROC/accuracy.

    The loss is the contrastive loss on a fixed set of validation pairs, for the curves.
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


def plot_history(history, path, loss_label):
    """Training/validation loss and validation AUROC per epoch."""
    epochs = [h["epoch"] for h in history]
    fig, (ax_loss, ax_auc) = plt.subplots(1, 2, figsize=(10, 4))
    ax_loss.plot(epochs, [h["train_loss"] for h in history], marker="o", label="train")
    ax_loss.plot(epochs, [h["val_loss"] for h in history], marker="o", label="validation")
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
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=NUM_WORKERS)

def main():
    args = parse_args()
    torch.manual_seed(SEED)
    device = pick_device()
    os.makedirs(OUT_DIR, exist_ok=True)

    # splitting the data to train with augmentation, validate without.
    df = get_splits(DATA_ROOT)
    train_set = sample_split(df, "train", SUBSET, SEED)
    val_set = sample_split(df, "val", SUBSET and SUBSET // 2, SEED)
    val_loader = loader(LesionDataset(val_set, "val"))

    if MODEL_CHOICE == "baseline":
        train_loader = loader(LesionDataset(train_set, "train", build_transforms(train=True)), shuffle=True)
        model = BaselineClassifier(pretrained=True).to(device)
        weights = class_weights(torch.tensor(train_set["target"].values))
        loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))
        loss_label = "weighted cross-entropy"
        details = f"class weights {[round(w, 2) for w in weights.tolist()]}"
    else:
        # Pair sampling balances the classes, so no class weights are needed.
        train_pairs = PairDataset(train_set, "train", seed=SEED)
        train_loader = loader(train_pairs, shuffle=True)
        # need another loader for the validation pairs, so the contrastive loss can be computed for the curves.
        val_pair_loader = loader(PairDataset(val_set, "val", transform=build_transforms(train=False), seed=SEED))
        
        # Reference bank: training images without augmentation, embedded each epoch for kNN.
        bank_loader = loader(LesionDataset(train_set, "train"))
        model = SiameseNet(pretrained=True).to(device)
        loss_fn = ContrastiveLoss(margin=MARGIN)
        loss_label = "contrastive loss"
        details = f"pairs/epoch {len(train_pairs)} | margin {MARGIN} | k {K}"

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    print(f"{MODEL_CHOICE} | device {device} | train {len(train_set)} | val {len(val_set)} | "
          f"{details} | parameters {count_parameters(model):,}")

    weights_path = os.path.join(OUT_DIR, f"{MODEL_CHOICE}_best.pt")
    history, best_auroc, epochs_since_best = [], -1.0, 0
    memory = PeakMemory(device)
    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        if MODEL_CHOICE == "baseline":
            train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device, memory)
            val = evaluate(model, val_loader, loss_fn, device)
        else:
            train_pairs.set_epoch(epoch)  # fresh pairs every epoch
            train_loss = train_one_epoch_siamese(model, train_loader, loss_fn, optimizer, device, memory)
            val = evaluate_siamese(model, bank_loader, val_loader, val_pair_loader, loss_fn, device, K)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val["loss"],
                        "val_auroc": val["auroc"], "val_accuracy": val["accuracy"],
                        "seconds": time.time() - epoch_start})
        print(f"epoch {epoch:>3} | train loss {train_loss:.4f} | val loss {val['loss']:.4f} | "
              f"val AUROC {val['auroc']:.4f} | val acc {val['accuracy']:.4f} | {history[-1]['seconds']:.0f}s")

        # Keep the weights with the best validation AUROC; stop early if it stops improving.
        if val["auroc"] > best_auroc:
            best_auroc, epochs_since_best = val["auroc"], 0
            # save the weights of the best model so far, for later evaluation on the test split
            torch.save(model.state_dict(), weights_path)
        else:
            epochs_since_best += 1
            if epochs_since_best >= PATIENCE:
                print(f"early stopping: no improvement for {PATIENCE} epochs")
                break

    summary = {
        "model": MODEL_CHOICE, "device": str(device), "args": vars(args), "lr": LR,
        "best_val_auroc": best_auroc, "parameters": count_parameters(model),
        "total_minutes": (time.time() - start) / 60,
        "peak_gpu_memory_gib": memory.peak_gib(),
        "history": history,
    }
    # saving for logs and later analysis
    with open(os.path.join(OUT_DIR, f"{MODEL_CHOICE}_history.json"), "w") as f:
        json.dump(summary, f, indent=2)
    plot_history(history, os.path.join(OUT_DIR, f"{MODEL_CHOICE}_curves.png"), loss_label)

    peak = memory.peak_gib()
    print(f"best val AUROC {best_auroc:.4f} | weights -> {weights_path} | "
          f"total {summary['total_minutes']:.1f} min | peak GPU memory "
          f"{'n/a (CPU)' if peak is None else f'{peak:.2f} GiB ({device.type})'}")


if __name__ == "__main__":
    main()
