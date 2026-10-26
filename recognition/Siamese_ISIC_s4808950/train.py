"""
Train and validate the models, saving the best weights and the training curves.

  --model baseline: ResNet-18 classifier trained with class-weighted cross-entropy.
  --model siamese:  Siamese network trained on image pairs with contrastive loss,
                    validated by k-nearest-neighbour classification against the
                    embedded training images.

Usage example:
    python train.py --model baseline --subset 1000 --epochs 1
    python train.py --model siamese  --subset 1000 --epochs 1
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
import numpy as np

from dataset import LesionDataset, PairDataset, build_transforms, get_splits
from modules import BaselineClassifier, ContrastiveLoss, SiameseNet, count_parameters

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
LOSS_LABEL = "weighted cross-entropy" if MODEL_CHOICE == "baseline" else "contrastive loss"



def parse_args():
    global MODEL_CHOICE, EPOCHS, BATCH_SIZE, PATIENCE, NUM_WORKERS, SEED, MARGIN, K

    parser = argparse.ArgumentParser(description="Train the melanoma classifiers.")
    parser.add_argument("--model", choices=["baseline", "siamese"], default=MODEL_CHOICE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--margin", type=float, default=MARGIN, help="contrastive loss margin")
    parser.add_argument("--k", type=int, default=K, help="k for kNN classification of validation images")

    args = parser.parse_args()
    MODEL_CHOICE, EPOCHS, BATCH_SIZE, PATIENCE, NUM_WORKERS, SEED, MARGIN, K = (
        args.model, args.epochs, args.batch_size,
        args.patience, args.num_workers, args.seed, args.margin, args.k
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
    # added extra for smoke test
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


def plot_history(history, path):
    """Training/validation loss and validation AUROC per epoch."""
    epochs = [h["epoch"] for h in history]
    fig, (ax_loss, ax_auc) = plt.subplots(1, 2, figsize=(10, 4))
    ax_loss.plot(epochs, [h["train_loss"] for h in history], marker="o", label="train")
    ax_loss.plot(epochs, [h["val_loss"] for h in history], marker="o", label="validation")


    ax_loss.set(xlabel="epoch", ylabel=LOSS_LABEL, title="Loss")
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

def setup_baseline(train_set, device):
    """Model, loss and training loader for the baseline."""
    weights = class_weights(torch.tensor(train_set["target"].values))
    return {
        "model": BaselineClassifier(pretrained=True).to(device),
        "loss_fn": nn.CrossEntropyLoss(weight=weights.to(device)),
        "train_loader": loader(LesionDataset(train_set, "train", build_transforms(train=True)), True),
        "details": f"class weights {[round(w, 2) for w in weights.tolist()]}",
    }


def setup_siamese(train_set, val_set, device):
    """Model, loss and loaders for the Siamese network."""
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


def run_epoch(setup, epoch, optimizer, val_loader, device, memory):
    """Train for one epoch, then validate. Returns (train loss, validation results)."""
    model = setup["model"]
    loss_fn = setup["loss_fn"]

    if MODEL_CHOICE == "baseline":
        train_loss = train_one_epoch(model, setup["train_loader"], loss_fn, optimizer, device, memory)
        return train_loss, evaluate(model, val_loader, loss_fn, device)
    
    setup["train_pairs"].set_epoch(epoch)  # fresh pairs every epoch
    train_loss = train_one_epoch_siamese(model, setup["train_loader"], loss_fn, optimizer, device, memory)
    val = evaluate_siamese(model, setup["bank_loader"], val_loader, setup["val_pair_loader"], loss_fn, device, K)
    return train_loss, val


def save_training_summary(summary, weights_path):
    """Save the history (for logs and the README), plot the curves and print a summary line."""
    with open(os.path.join(OUT_DIR, f"{MODEL_CHOICE}_history.json"), "w") as f:
        json.dump(summary, f, indent=2)
    plot_history(summary["history"], os.path.join(OUT_DIR, f"{MODEL_CHOICE}_curves.png"), LOSS_LABEL)
    peak, device = summary["peak_gpu_memory_gib"], summary["device"]
    print(f"best val AUROC {summary['best_val_auroc']:.4f} | "
          f"total {summary['total_minutes']:.1f} min | peak GPU memory "
          f"{'n/a (CPU)' if peak is None else f'{peak:.2f} GiB ({device})'}")

def train_epochs(setup, optimizer, val_loader, device):
    history = []
    weights_path = os.path.join(OUT_DIR, f"{MODEL_CHOICE}_best.pt")
    start = time.time()
    memory = PeakMemory(device)
    best_auroc, epochs_since_best = -1.0, 0
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        model = setup["model"]
        loss_fn = setup["loss_fn"]

        if MODEL_CHOICE == "baseline":
            train_loss = train_one_epoch(model, setup["train_loader"], loss_fn, optimizer, device, memory)
            val = evaluate(model, val_loader, loss_fn, device)
        else:
            setup["train_pairs"].set_epoch(epoch)  # fresh pairs every epoch
            train_loss = train_one_epoch_siamese(model, setup["train_loader"], loss_fn, optimizer, device, memory)
            val = evaluate_siamese(model, setup["bank_loader"], val_loader, setup["val_pair_loader"], loss_fn, device, K)
        
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val["loss"],
                    "val_auroc": val["auroc"], "val_accuracy": val["accuracy"], "k": K,
                    "seconds": time.time() - epoch_start})
        
        print(f"epoch {epoch} | train loss {train_loss:.4f} | val loss {val['loss']:.4f} | "
            f"val AUROC {val['auroc']:.4f} | val acc {val['accuracy']:.4f} | {history[-1]['seconds']:.0f}s")
            
        if val["auroc"] > best_auroc:
            best_auroc, epochs_since_best = val["auroc"], 0
            torch.save(model.state_dict(), weights_path)
        else:
            epochs_since_best += 1
            if epochs_since_best >= PATIENCE:
                print(f"early stopping: no improvement for {PATIENCE} epochs")
    
    save_training_summary({
        "model": MODEL_CHOICE, "device": str(device), "lr": LR, "best_val_auroc": best_auroc,
        "parameters": count_parameters(model), "total_minutes": (time.time() - start) / 60, 
        "peak_gpu_memory_gib": memory.peak_gib(), "history": history,
    }, weights_path)
    

def main():
    args = parse_args()
    torch.manual_seed(SEED)
    device = pick_device()
    os.makedirs(OUT_DIR, exist_ok=True)

    df = get_splits(DATA_ROOT)
    train_set = sample_split(df, "train", None, SEED)
    val_set = sample_split(df, "val", None, SEED)
    test_set = sample_split(df, "test", None, SEED)
    # each model calls its own train loader as appropriate
    test_loader = loader(LesionDataset(test_set, "test"))
    val_loader = loader(LesionDataset(val_set, "val"))

    setup = {}
    if MODEL_CHOICE == "baseline":
        setup = setup_baseline(train_set, device)
    else:
        setup = setup_siamese(train_set, val_set, device)

    model = setup["model"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    print(f"{MODEL_CHOICE} | device {device} | train {len(train_set)} | val {len(val_set)} | "
          f"{setup['details']} | parameters {count_parameters(model):,}")
    
    train_epochs(setup, optimizer, val_loader, device)
    tune_model(setup["model"], device, train_set, val_loader)


def closure(optimizer, logits, log_t, labels):
    optimizer.zero_grad()
    loss = torch.nn.functional.cross_entropy(logits / log_t.exp(), labels)
    loss.backward()
    return loss.detach()
    
def fit_temperature(logits, labels):
    """
    Temperature scaling
    """
    log_t = torch.zeros(1, requires_grad=True)  # optimise log T so that T stays positive
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)
    closure(optimizer, logits, log_t, labels)


@torch.no_grad()
def baseline_logits(model, data_loader, device):
    """Raw 2-class logits and labels for every image in a loader."""
    logits, labels = [], []
    for images, targets, _ in data_loader:
        logits.append(model(images.to(device))).cpu()
        labels.append(targets)
    return torch.cat(logits), torch.cat(labels)

def melanoma_probability_from_logits(logits, temperature=1.0):
    return (logits / temperature).softmax(dim=1)[:, 1]


def handle_baseline_eval(model, val_loader, result, device):
    val_logits, val_labels = baseline_logits(model, val_loader, device)
    
    # bc baseline produces logits, the temperature needs to be adjusted
    temperature = fit_temperature(val_logits, val_labels)
    val_probs = (val_logits / temperature).softmax(dim=1)[:, 1]

    result["temperature"] = temperature
    return val_logits, val_labels, result

def handle_siamese_eval(model, val_loader, result, device, dfT):      
    bank, bank_labels = embed_all(model, loader(LesionDataset(dfT, "train")), device)
    val_embeddings, val_labels = embed_all(model, val_loader, device)
    
    test_probs = knn_melanoma_probability(val_embeddings, bank, bank_labels, K)
    return test_probs, val_labels, result


def tune_model(model, device, train_df, val_loader):
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
    labels, probs = val_labels.numpy(), val_labels.numpy()

    # checking if confidence track correctness
    confidence, correct = confidence_and_correct(labels, probs)
    print(confidence, correct)
    print(result)
    return result

def confidence_and_correct(labels, probs):
    """Confidence = probability of the predicted class, max(p, 1 - p), in [0.5, 1]."""
    preds = (probs >= 0.5).astype(int)
    return np.maximum(probs, 1 - probs), preds == labels


if __name__ == "__main__":
    main()
