import argparse
import os
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image

from dataset import LesionDataset, get_splits
from train import (DATA_ROOT, OUT_DIR, SEED, embed_all, load_model, loader, melanoma_scores,
                   pick_device, plot_test_comparison, sample_split, training_bank)


N_RANDOM = 6           
N_NEIGHBOURS = 5      
N_FALSE_NEGATIVES = 3
N_FALSE_POSITIVES = 2


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


def decision(confidence, tuned):
    """'auto' or 'REFER' with the reject threshold chosen on validation."""
    threshold = tuned["reject_rule_validation"]["threshold"]
    return "REFER" if threshold is None or confidence < threshold else "auto"


def choose_rows(name, test_df, args):
    """The test rows to show: given IDs, failure cases (--autopsy) or a random sample."""
    if args.images:
        rows = test_df[test_df["isic_id"].isin(args.images)]
        missing = set(args.images) - set(rows["isic_id"])
        if missing:
            raise SystemExit(f"not in the test split: {sorted(missing)}")
        return rows

    if args.autopsy:
        path = os.path.join(OUT_DIR, f"{name}_test_predictions.csv")
        if not os.path.exists(path):
            raise SystemExit(f"{path} not found, running train.py or predict.py without autospy required first")
        
        predictions = pd.read_csv(path)
        print_errors_by_source(predictions)
        # The most confident errors are the dangerous ones: the model was sure and wrong.
        wrong = predictions[~predictions["correct"]].sort_values("confidence", ascending=False)
        false_negatives = wrong[wrong["target"] == 1].head(N_FALSE_NEGATIVES)
        false_positives = wrong[wrong["target"] == 0].head(N_FALSE_POSITIVES)
        chosen = pd.concat([false_negatives, false_positives])["isic_id"]
        return test_df.set_index("isic_id").loc[chosen].reset_index()

    return sample_split(test_df, "test", N_RANDOM)


def print_errors_by_source(predictions):
    """Test errors per ID source (README risk 2): do mistakes cluster in one data source?"""
    wrong = ~predictions["correct"]
    preds = predictions.assign(wrong=wrong,
        missed_melanoma=wrong & (predictions["target"] == 1),
        false_alarm=wrong & (predictions["target"] == 0),
        wrong_but_auto=wrong & ~predictions["referred"],
    )
    table = preds.groupby("id_source").agg(
        images=("isic_id", "size"),
        error_rate=("wrong", "mean"),
        missed_melanomas=("missed_melanoma", "sum"),
        false_alarms=("false_alarm", "sum"),
        referred=("referred", "mean"),
        wrong_but_auto_decided=("wrong_but_auto", "sum"),
    )
    print("Errors by ID source (full test split):")
    print(table.round(3).to_string(), "\n")



def main():
    parser = argparse.ArgumentParser(description="Classify test images with a trained model.")
    parser.add_argument("--model", choices=["baseline", "siamese"], default="siamese")
    parser.add_argument("--show_full_test", action="store_true",
                        help="only print the saved results on the full test split")

    parser.add_argument("--images", nargs="+", help="isic_ids from the test split")
    parser.add_argument("--autopsy", action="store_true", help="show the most confident test errors")
    args = parser.parse_args()
    if args.show_full_test:
        print_test_summary(args.model)
        return
    device = pick_device()
    model = load_model(args.model, device) 
    df = get_splits(DATA_ROOT)
    test_df = df[df["split"] == "test"]
    train_df = df[df["split"] == "train"]
    tuned = load_settings(model)   
    print(f"Running trained {model} model at threshold {tuned["threshold"]},"
          f"temp {tuned["temperature"]} and k {tuned["k"]}") 
    rows = choose_rows(args.model, test_df, args).reset_index(drop=True)
    labels, probs = melanoma_scores(args.model, model, loader(LesionDataset(rows, "test")), device, tuned, train_df)


if __name__ == "__main__":
    main()
