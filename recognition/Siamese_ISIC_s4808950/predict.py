import argparse
import os
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from dataset import LesionDataset, get_splits
from train import (DATA_ROOT, OUT_DIR, SEED, embed_all, load_model, loader, melanoma_scores,
                   pick_device, sample_split, training_bank)

def load_settings(name):
    path = os.path.join(OUT_DIR, f"{name}_tuned.json")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found. Requires training")
    with open(path) as f:
        return json.load(f)


def print_test_summary(name):
    path = os.path.join(OUT_DIR, f"{name}_test_results.json")
    if not os.path.exists(path):
        print(f"no test results yet. A test run is required for this flag")
        return
    
    with open(path) as f:
        result = json.load(f)
    d, reject = result["discrimination"], result["reject_rule"]
    print(f"{name} on the full test split ({d['n']} images):")
    print(f"accuracy {d['accuracy']:.3f} | AUROC {d['auroc']:.3f} | PR-AUC {d['pr_auc']:.3f} | "
          f"macro-F1 {d['macro_f1']:.3f} | melanoma recall {d['recall_melanoma']:.3f}")
    
    recall_auto = reject["melanoma_recall_auto_decided"]
    print(f"ECE {result['ece']:.3f} | AURC {result['aurc']:.4f} | referred {reject['referred']:.1%} | "
          f"melanoma recall among auto-decided {'n/a' if recall_auto is None else f'{recall_auto:.3f}'} | "
          f"latency {result['latency_ms_per_image']:.1f} ms/image\n")

def choose_rows(name, test_df, args):
    """The test rows to show: given IDs, failure cases (--autopsy) or a random sample."""
    if args.images:
        rows = test_df[test_df["isic_id"].isin(args.images)]
        missing = set(args.images) - set(rows["isic_id"])
        if missing:
            raise SystemExit(f"not in the test split: {sorted(missing)}")
        return rows

    if args.autopsy:
        print("hello")
    return sample_split(test_df, "test", N_RANDOM)



def main():
    parser = argparse.ArgumentParser(description="Classify test images with a trained model.")
    parser.add_argument("--model", choices=["baseline", "siamese"], default="siamese")
    parser.add_argument("--show_full_test", action="store_true",
                        help="only print the saved results on the full test split")

    parser.add_argument("--images", nargs="+", help="isic_ids from the test split")
    parser.add_argument("--autopsy", action="store_true", help="show the most confident test errors")
    args = parser.parse_args()
    if args.show_full_test:
        print_test_summary(model)
        return
    device = pick_device()
    model = load_model(model, device) 
    df = get_splits(DATA_ROOT)
    test_df = df[df["split"] == "test"]
    train_df = df[df["split"] == "train"]
    tuned = load_settings(model)
    print(f"Running trained {model} model at threshold {tuned["threshold"]},"
          f"temp {tuned["temperature"]} and k {tuned["k"]}") 


    rows = choose_rows(model, test_df, args).reset_index(drop=True)
    labels, probs = melanoma_scores(args.model, model, loader(LesionDataset(rows, "test")), device, tuned, train_df)


if __name__ == "__main__":
    main()
