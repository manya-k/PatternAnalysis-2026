import argparse
import os
import json
import torch
from train import (load_model, OUT_DIR, SEED, DATA_ROOT, 
                   pick_device, sample_split, test_model)
from dataset import get_splits

def load_settings(name):
    path = os.path.join(OUT_DIR, f"{name}_tuned.json")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found. Requires training")
    with open(path) as f:
        tuned = json.load(f)
    return {
        "threshold": tuned["reject_rule_validation"]["threshold"], 
        "temperature": tuned.get("temperature", 1.0),
        "k": tuned.get("k"),
        "melanoma_weight": tuned.get("melanoma_weight", 1.0),  # Siamese only: 1.0 = plain vote                           
    }

def print_test_summary(name):
    path = os.path.join(OUT_DIR, f"{name}_test_results.json")
    if not os.path.exists(path):
        print(f"no test results yet. A test run is required for this flag")
        return
    
    with open(path) as f:
        result = json.load(f)
    d, reject = result["discrimination"], result["reject_rule"]
    print(f"{name} on the full test split:")
    print(f"accuracy {d['accuracy']:.3f} | AUROC {d['auroc']:.3f} | PR-AUC {d['pr_auc']:.3f} | "
          f"macro-F1 {d['macro_f1']:.3f} | melanoma recall {d['recall_melanoma']:.3f}")
    
    recall_auto = reject["melanoma_recall_auto_decided"]
    print(f"ECE {result['ece']:.3f} | AURC {result['aurc']:.4f} | referred {reject['referred']:.1%} | "
          f"melanoma recall among auto-decided {'n/a' if recall_auto is None else f'{recall_auto:.3f}'} | "
          f"latency {result['latency_ms_per_image']:.1f} ms/image\n")


def main():
    parser = argparse.ArgumentParser(description="Classify test images with a trained model.")
    parser.add_argument("--model", choices=["baseline", "siamese"], default="siamese")
    parser.add_argument("--show_full_test", default=False)
    parser.add_argument("--images", nargs="+", help="isic_ids from the test split")
    args = parser.parse_args()
    model = args.model
    device = pick_device()
    model = load_model(model, device)

    if args.show_full_test:
        print_test_summary(model)
        return

    tuned = load_settings(model)
    df = get_splits(DATA_ROOT, SEED)
    test_df = df[df["split"] == "test"]
    train_df = df[df["split"] == "train"]
    test_samples = sample_split(df, "test", 6, SEED)
    print(f"Running trained {model} model at threshold {tuned["threshold"]},"
          f"temp {tuned["temperature"]} and k {tuned["k"]}") 
    results = test_model(model, device, train_df, test_samples, tuned)

if __name__ == "__main__":
    main()