"""
Test-set evaluation of the baseline and Siamese models,

Usage:
    python evaluate.py               # both models, full splits
    python evaluate.py --subset 300  # quick smoke test on a sample of each split
"""

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import torch
import numpy as np
from dataset import LesionDataset, get_splits
from modules import BaselineClassifier, SiameseNet
from train import DATA_ROOT, OUT_DIR, embed_all, knn_melanoma_probability, loader, pick_device, sample_split

MODELS = ("baseline", "siamese")

def load_model(name, device):
    """Rebuild the architecture and load the best weights saved by train.py."""
    model = BaselineClassifier(pretrained=False) if name == "baseline" else SiameseNet(pretrained=False)
    path = os.path.join(OUT_DIR, f"{name}_best.pt")
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()


@torch.no_grad()
def baseline_logits(model, data_loader, device):
    """Raw 2-class logits and labels for every image in a loader."""
    logits, labels = [], []
    for images, targets, _ in data_loader:
        logits.append(model(images.to(device)))
        labels.append(targets)
    return torch.cat(logits), torch.cat(labels)

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


def melanoma_probability_from_logits(logits, temperature=1.0):
    return (logits / temperature).softmax(dim=1)[:, 1]


def confidence_and_correct(labels, probs):
    """Confidence = probability of the predicted class, max(p, 1 - p), in [0.5, 1]."""
    preds = (probs >= 0.5).astype(int)
    return np.maximum(probs, 1 - probs), preds == labels


def handle_baseline_eval(model, val_loader, test_loader, result, device):
    val_logits, val_labels = baseline_logits(model, val_loader, device)
    test_logits, test_labels = baseline_logits(model, test_loader, device)
    
    # bc baseline produces logits, the temperature needs to be adjusted
    temperature = fit_temperature(val_logits, val_labels)
    test_probs = melanoma_probability_from_logits(test_logits, temperature)

    result["temperature"] = temperature
    return test_probs, test_labels, result

def handle_siamese_eval(model, test_loader, result, device, dfT):
    with open(os.path.join(OUT_DIR, "siamese_history.json")) as f:
        k = json.load(f)["args"]["k"]  # use the same k as during training
    
    bank, bank_labels = embed_all(model, loader(LesionDataset(dfT, "train")), device)
    test_embeddings, test_labels = embed_all(model, test_loader, device)
       
    test_probs = knn_melanoma_probability(test_embeddings, bank, bank_labels, k)
    
    result["k"] = k
    return test_probs, test_labels, result


def evaluate_model(name, device, train_df, val_df, test_df):
    """All test results for one model. Also writes <name>_test_predictions.csv."""
    model = load_model(name, device)
    val_set = LesionDataset(val_df, "val")
    test_set = LesionDataset(test_df, "test")
    val_loader = loader(val_set)
    test_loader = loader(test_set)
    
    result = {}
    start = time.time()
    if name == "baseline":
        test_probs, test_labels, result = handle_baseline_eval(
            model, val_loader, test_loader, result, device
        )
    else:
        test_probs, test_labels, result = handle_baseline_eval(
                    model, val_loader, test_loader, result, device, train_df
                )
    result["scoring_seconds"] = time.time() - start
    labels, probs = test_labels.numpy(), test_probs.numpy()

    # checking if confidence track correctness
    confidence, correct = confidence_and_correct(labels, probs)
    print(confidence, correct)
    print(result)
    return result

def main():
    parser = argparse.ArgumentParser(description="Evaluate the trained models on the test split.")
    parser.add_argument("--subset", type=int, default=None, help="sample this many images per split (smoke test)")
    args = parser.parse_args()

    device = pick_device()
    df = get_splits(DATA_ROOT)
    train_df = sample_split(df, "train", args.subset, 0)
    val_df = sample_split(df, "val", args.subset, 0)
    test_df = sample_split(df, "test", args.subset, 0)

    for name in MODELS:
        if not os.path.exists(os.path.join(OUT_DIR, f"{name}_best.pt")):
            print(f"no {name}_best.pt found")
            continue
        print(f"evaluating {name}")
        evaluate_model(name, device, train_df, val_df, test_df)


if __name__ == "__main__":
    main()
