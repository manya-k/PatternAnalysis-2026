"""
Test-set evaluation of the baseline and Siamese models,

Usage:
    python evaluate.py               # both models, full splits
    python evaluate.py --subset 300  # quick smoke test on a sample of each split
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import torch
from dataset import LesionDataset, get_splits
from modules import BaselineClassifier, SiameseNet
from train import DATA_ROOT, OUT_DIR, loader, pick_device, sample_split

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



def evaluate_model(name, device, val_df):
    """All test results for one model. Also writes <name>_test_predictions.csv."""
    model = load_model(name, device)
    val_set = LesionDataset(val_df, "val")
    val_loader = loader(val_set)

    if name == "baseline":
        val_logits, val_labels = baseline_logits(model, val_loader, device)      
        temp = fit_temperature(val_logits, val_labels)
        prob = (val_logits / temp).softmax(dim=1)[:, 1]
        ## TODO
        
def main():
    parser = argparse.ArgumentParser(description="Evaluate the trained models on the test split.")
    parser.add_argument("--subset", type=int, default=None, help="sample this many images per split (smoke test)")
    args = parser.parse_args()

    device = pick_device()
    df = get_splits(DATA_ROOT)
    val_df = sample_split(df, "val", args.subset, 0)

    for name in MODELS:
        if not os.path.exists(os.path.join(OUT_DIR, f"{name}_best.pt")):
            print(f"no {name}_best.pt found")
            continue
        print(f"evaluating {name}")
        evaluate_model(name, device, val_df)


if __name__ == "__main__":
    main()
