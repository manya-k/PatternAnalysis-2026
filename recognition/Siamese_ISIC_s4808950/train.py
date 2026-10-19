"""
Train and validate the models, saving the best weights and the training curves.

Usage example:
    python train.py --model baseline --subset 1000 --epochs 1

The test split is never touched here; it is only used for the final evaluation.
"""

import argparse
import os
import torch

from modules import BaselineClassifier

OUT_DIR = "outputs"
DATA_ROOT = "data"

def parse_args():
    parser = argparse.ArgumentParser(description="Train the melanoma classifiers.")
    parser.add_argument("--model", choices=["baseline"], default="baseline")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def pick_device():
    """Best available device: CUDA (Rangpur), MPS (Mac GPU), else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def main():
    args = parse_args()
    # set the random seed for reproducibility
    torch.manual_seed(args.seed)
    device = pick_device()
    os.makedirs(OUT_DIR, exist_ok=True)

    model = BaselineClassifier(pretrained=True).to(device)


if __name__ == "__main__":
    main()
