"""
Train and validate the models, saving the best weights and the training curves.

  --model baseline: ResNet-18 classifier trained with class-weighted cross-entropy.
  --model siamese:  Siamese network trained on image pairs with contrastive loss,
                    validated by k-nearest-neighbour classification against the
                    embedded training images.

After training, the best checkpoint is tuned on validation (k for the Siamese model,
temperature for the baseline, reject threshold for both) and then tested once on the test split.

Usage example:
    python train.py --model baseline
    python train.py --model siamese
    python train.py --model siamese --test_only 
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
    global MODEL_CHOICE, EPOCHS, BATCH_SIZE, PATIENCE, NUM_WORKERS, SEED, MARGIN, TESR

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

    args = parser.parse_args()
    MODEL_CHOICE, EPOCHS, BATCH_SIZE, PATIENCE, NUM_WORKERS, SEED, MARGIN, TEST = (
        args.model, args.epochs, args.batch_size,
        args.patience, args.num_workers, args.seed, args.margin, args.test_only
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


def save_training_summary(summary, weights_path):
    """Save the history (for logs and the README), plot the curves and print a summary line."""
    with open(os.path.join(OUT_DIR, f"{MODEL_CHOICE}_history.json"), "w") as f:
        json.dump(summary, f, indent=2)
    plot_history(summary["history"], os.path.join(OUT_DIR, f"{MODEL_CHOICE}_curves.png"))
    peak, device = summary["peak_gpu_memory_gib"], summary["device"]
    print(f"best val AUROC {summary['best_val_auroc']:.4f} | "
          f"total {summary['total_minutes']:.1f} min | peak GPU memory "
          f"{'n/a (CPU)' if peak is None else f'{peak:.2f} GiB ({device})'}")

def train_epochs(setup, optimizer, val_loader, device):
    """Train with early stopping on val AUROC. Returns the path of the best checkpoint."""
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
        
        print(f"epoch {epoch} | train loss {train_loss:.4f} | val loss {val['loss']:.4f} | "
            f"val AUROC {val['auroc']:.4f} | val acc {val['accuracy']:.4f} | {history[-1]['seconds']:.0f}s")
            
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


def main():
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
    else:
        val_loader = loader(LesionDataset(val_set, "val"))
        if MODEL_CHOICE == "baseline":
            setup = setup_baseline(train_set, device)
        else:
            setup = setup_siamese(train_set, val_set, device)

        model = setup["model"]
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        print(f"{MODEL_CHOICE} | device {device} | train {len(train_set)} | val {len(val_set)} | "
              f"{setup['details']} | parameters {count_parameters(model):,}")
        weights_path = train_epochs(setup, optimizer, val_loader, device, args)

        # Tune and test the best checkpoint, not the weights from the last epoch.
        model.load_state_dict(torch.load(weights_path, map_location=device))
        tuned = tune_model(model, device, train_set, val_loader)

    test_model(model, device, train_set, test_set, tuned)



def load_model(name, device):
    """Rebuild the architecture and load the best weights saved by train_epochs."""
    model = BaselineClassifier(pretrained=False) if name == "baseline" else SiameseNet(pretrained=False)
    model.load_state_dict(torch.load(os.path.join(OUT_DIR, f"{name}_best.pt"), map_location=device))
    return model.to(device).eval()


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
    # LBFGS calls the closure itself (several times per step), so it must take no arguments.
    optimizer.step(lambda: closure(optimizer, logits, log_t, labels))
    return log_t.exp().item()


@torch.no_grad()
def baseline_logits(model, data_loader, device):
    """Raw 2-class logits and labels for every image in a loader."""
    logits, labels = [], []
    for images, targets, _ in data_loader:
        logits.append(model(images.to(device)).cpu())
        labels.append(targets)
    return torch.cat(logits), torch.cat(labels)

def melanoma_probability_from_logits(logits, temperature=1.0):
    return (logits / temperature).softmax(dim=1)[:, 1]


def handle_baseline_eval(model, val_loader, result, device):
    val_logits, val_labels = baseline_logits(model, val_loader, device)
    
    # bc baseline produces logits, the temperature needs to be adjusted
    temperature = fit_temperature(val_logits, val_labels)
    val_probs = melanoma_probability_from_logits(val_logits, temperature)

    # Before/after: T only changes how confident the probabilities are, so ECE should drop.
    labels = val_labels.numpy()
    ece_before, _ = calibration(*confidence_and_correct(labels, melanoma_probability_from_logits(val_logits).numpy()))
    ece_after, _ = calibration(*confidence_and_correct(labels, val_probs.numpy()))
    print(f"temperature T = {temperature:} | val ECE {ece_before:} (T = 1) -> {ece_after:} (tuned)")

    result["temperature"] = temperature
    result["val_ece_before_temperature"], result["val_ece_after_temperature"] = ece_before, ece_after
    return val_probs, val_labels, result


def handle_siamese_eval(model, val_loader, result, device, train_df):

    bank, bank_labels = embed_all(model, loader(LesionDataset(train_df, "train")), device)
    val_embeddings, val_labels = embed_all(model, val_loader, device)
    # K in KNN Tunning
    k_scores = {}
    for k in K_CANDIDATES:
        probs = knn_melanoma_probability(val_embeddings, bank, bank_labels, k)
        confidence, correct = confidence_and_correct(val_labels.numpy(), probs.numpy())
        _, _, aurc = risk_coverage(confidence, correct)
        k_scores[k] = {
            "val_aurc": aurc,
            "val_auroc": float(roc_auc_score(val_labels, probs)),
        }
        print(f"k {k:} | val AURC {k_scores[k]['val_aurc']:.4f} | val AUROC {k_scores[k]['val_auroc']:.4f} ")

    best_k = min(k_scores, key=lambda k: k_scores[k]["val_aurc"])
    # Before/after: K is the k used during training to pick the best epoch.
    if K in k_scores:
        print(f"k {K} (training) -> {best_k} (tuned) | "
              f"val AURC {k_scores[K]['val_aurc']:.4f} -> {k_scores[best_k]['val_aurc']:.4f} | "
              f"val AUROC {k_scores[K]['val_auroc']:.4f} -> {k_scores[best_k]['val_auroc']:.4f}")
    
    plot_k_tuning(k_scores, best_k, os.path.join(OUT_DIR, "siamese_k_tuning.png"))
    result["k"], result["k_scores"] = best_k, k_scores
    val_probs = knn_melanoma_probability(val_embeddings, bank, bank_labels, best_k)
    return val_probs, val_labels, result


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
    labels, probs = val_labels.numpy(), val_probs.numpy()

    #check for threshbhold, k, r
    threshold = choose_threshold(labels, probs)
    result["reject_rule_validation"] = apply_reject_rule(labels, probs, threshold)
    print(result)

    # Save the validation-tuned settings so the test step and predict.py use the tuned k,
    # temperature and threshold rather than the defaults the model was trained with.
    with open(os.path.join(OUT_DIR, f"{MODEL_CHOICE}_tuned.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result

def apply_reject_rule(labels, probs, threshold):
    """Outcome of the reject rule on one split."""
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
    """
    confidence, _ = confidence_and_correct(labels, probs)
    preds = (probs >= 0.5).astype(int)
    for t in np.unique(confidence):  # ascending: lower t automates more cases
        auto_melanoma = (confidence >= t) & (labels == 1)
        if auto_melanoma.any() and (preds[auto_melanoma] == 1).mean() >= target_recall:
            return float(t)
    return float("inf")  # no threshold reaches the target: refer every case


@torch.no_grad()
def test_scores(model, device, train_df, test_loader, tuned):
    model.eval()
    if MODEL_CHOICE == "baseline":
        temperature = tuned["temperature"]
        logits, labels = baseline_logits(model, test_loader, device)
        probs = melanoma_probability_from_logits(logits, temperature)
        predict_one = lambda image: melanoma_probability_from_logits(model(image), temperature)
    else:
        k = tuned["k"]
        bank, bank_labels = embed_all(model, loader(LesionDataset(train_df, "train")), device)
        query, labels = embed_all(model, test_loader, device)
        probs = knn_melanoma_probability(query, bank, bank_labels, k)
        predict_one = lambda image: knn_melanoma_probability(model.embed(image).cpu(), bank, bank_labels, k)
    return labels.numpy(), probs.numpy(), predict_one


def results_by_source(sources, labels, probs, threshold):
    """
    Discrimination and reject-rule results for each ID source (IP / HAM / dummy) separately,
    to check whether performance depends on the data source (README risk 2).
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
    test_dataset = LesionDataset(test_df, "test")
    labels, probs, predict_one = test_scores(model, device, train_df, loader(test_dataset), tuned)
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
    print(f"TEST {MODEL_CHOICE} | acc {d['accuracy']: } | AUROC {d['auroc']:} | PR-AUC {d['pr_auc']:} | "
          f"macro-F1 {d['macro_f1']: } | melanoma recall {d['recall_melanoma']:}")
    print(f"TEST {MODEL_CHOICE} | ECE {ece:} | AURC {test_aurc:} | referred {reject['referred']:.1%} | "
          f"melanoma recall among auto-decided {reject['melanoma_recall_auto_decided']} | "
          f"latency {result['latency_ms_per_image']:.} ms/image")
    return result

if __name__ == "__main__":
    main()

"""
things remaining in this file:
3. and all the plots 
need to organise the code file ong
"""