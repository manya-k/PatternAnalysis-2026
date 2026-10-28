"""
predict.py

Example usage of the trained models: load the saved weights and the settings tuned on
validation, classify a few test images, print the results and save a figure.

    python predict.py --model siamese                     # 6 random test images
    python predict.py --model baseline --images ISIC_0033837 ISIC_0066921
    python predict.py --model siamese --autopsy           # most confident test errors (3 FN + 2 FP)
    python predict.py --model siamese --show_full_test    # saved results on the full test split

Each image gets what the triage tool would return: a benign/melanoma decision, a
confidence score, and "auto" or "REFER" (to a specialist) from the reject threshold chosen
on validation. For the Siamese model, each image is also shown next to its nearest
training images, the evidence behind its prediction. Figures are saved to outputs/.

Needs the files written by train.py: <model>_best.pt and <model>_tuned.json, plus
<model>_test_results.json and <model>_test_predictions.csv for --show_full_test and
--autopsy. Nothing here changes those files: the full test is train.py's job.
"""

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
                   pick_device, sample_split, training_bank)

CLASS_NAMES = {0: "benign", 1: "melanoma"}
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
            raise SystemExit(f"{path} not found: run train.py --model {name} first")
        
        predictions = pd.read_csv(path)
        print_errors_by_source(predictions)
        # The most confident errors are the dangerous ones: the model was sure and wrong.
        wrong = predictions[~predictions["correct"]].sort_values("confidence", ascending=False)
        false_negatives = wrong[wrong["target"] == 1].head(N_FALSE_NEGATIVES)
        false_positives = wrong[wrong["target"] == 0].head(N_FALSE_POSITIVES)
        chosen = pd.concat([false_negatives, false_positives])["isic_id"]
        return test_df.set_index("isic_id").loc[chosen].reset_index()

    return sample_split(test_df, "test", N_RANDOM, SEED)


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


@torch.no_grad()
def nearest_training_images(model, rows, train_df, device):
    """For each Siamese query, its N_NEIGHBOURS closest training images and their distances."""
    bank, _ = training_bank(model, train_df, device)
    bank_rows = train_df.reset_index(drop=True) 
    query, _ = embed_all(model, loader(LesionDataset(rows, "test")), device)
    
    distances, indices = torch.cdist(query, bank).topk(N_NEIGHBOURS, dim=1, largest=False)
    return [[(bank_rows.loc[int(j)], float(d)) for d, j in zip(dist_row, idx_row)]
            for dist_row, idx_row in zip(distances, indices)]


def print_results(rows, probs, neighbours, tuned):
    print(f"{'isic_id':>26} {'source':>6} {'true':>9} {'pred':>9} {'p(mel)':>7} {'conf':>5} {'decision':>8}")
    for r, (row, prob) in enumerate(zip(rows.itertuples(), probs)):
        predicted = int(prob >= 0.5)
        confidence = max(prob, 1 - prob)
        print(f"{row.isic_id:>26} {row.id_source:>6} {CLASS_NAMES[row.target]:>9} {CLASS_NAMES[predicted]:>9} "
              f"{prob:>7.3f} {confidence:>5.3f} {decision(confidence, tuned):>8}")
        if neighbours:
            summary = ", ".join(f"{CLASS_NAMES[int(n['target'])][:3]} {d:.2f}" for n, d in neighbours[r])
            print(f"{'':>26} nearest training images: {summary}")


def show(name, rows, probs, neighbours, tuned, path):
    n_cols = 1 + (N_NEIGHBOURS if neighbours else 0)
    # At least 6 inches wide, so the title fits when there is only one column (baseline).
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(max(2.4 * n_cols, 6), 2.8 * len(rows)), squeeze=False)
    for r, (row, prob) in enumerate(zip(rows.itertuples(), probs)):
        predicted = int(prob >= 0.5)
        confidence = max(prob, 1 - prob)
        axes[r][0].imshow(Image.open(row.path).convert("RGB"))
        axes[r][0].set_title(f"{row.isic_id} ({row.id_source})\n"
                             f"true {CLASS_NAMES[row.target]} | pred {CLASS_NAMES[predicted]}\n"
                             f"p(mel) {prob:.2f} | conf {confidence:.2f} | {decision(confidence, tuned)}",
                             fontsize=7, color="green" if predicted == row.target else "red")
        if neighbours:
            for c, (neighbour, distance) in enumerate(neighbours[r], start=1):
                axes[r][c].imshow(Image.open(neighbour["path"]).convert("RGB"))
                axes[r][c].set_title(f"neighbour {c}: {CLASS_NAMES[int(neighbour['target'])]}\n"
                                     f"distance {distance:.2f}", fontsize=7)
    for ax in axes.flat:
        ax.axis("off")

    if name == "siamese":
        setting = f"k = {tuned['k']}, {tuned.get('vote', 'plain')} vote"
    else:
        setting = f"T = {tuned['temperature']:.2f}"
    threshold = tuned["reject_rule_validation"]["threshold"]
    rule = "refer every case" if threshold is None else f"refer if conf < {threshold:.3f}"
    title = "test image (left) and its nearest training images" if neighbours else "test images"
    fig.suptitle(f"{name}: {title}\n({setting}, {rule})", fontsize=10)
    # Keep the top 0.8 inch free for the two-line title.
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.8 / fig.get_figheight()))
    fig.savefig(path, dpi=150)
    plt.close(fig)


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
    tuned = load_settings(args.model)
    threshold = tuned["reject_rule_validation"]["threshold"]
    print(f"Running the trained {args.model} model: k {tuned.get('k')}, vote {tuned.get('vote')}, "
          f"temperature {tuned.get('temperature')}, reject threshold "
          f"{'none (refer every case)' if threshold is None else f'{threshold:.3f}'}\n")
    
    rows = choose_rows(args.model, test_df, args).reset_index(drop=True)
    labels, probs = melanoma_scores(args.model, model, loader(LesionDataset(rows, "test")), device, tuned, train_df)
    neighbours = nearest_training_images(model, rows, train_df, device) if args.model == "siamese" else None
    
    print_results(rows, probs.tolist(), neighbours, tuned)
    figure = os.path.join(OUT_DIR, f"{'autopsy' if args.autopsy else 'predict'}_{args.model}.png")
    show(args.model, rows, probs.tolist(), neighbours, tuned, figure)
    print(f"\nfigure -> {figure}")


if __name__ == "__main__":
    main()
