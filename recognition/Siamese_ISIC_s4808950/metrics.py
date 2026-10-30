"""
metrics.py

Measurement and plotting helpers shared by train.py: discrimination metrics, calibration
(ECE), risk-coverage (AURC), single-image latency and the evaluation plots.

Everything here only measures or draws; nothing is fitted. Choosing k, the kNN vote, the
temperature and the reject threshold happens in train.py, on validation data only.
"""

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score,
                            precision_recall_curve, roc_curve)

N_BINS = 15            # bins for the expected calibration error
LATENCY_IMAGES = 50    # images timed one at a time for the latency measurement



def discrimination_metrics(labels, probs):
    """
    Standard metrics at the 0.5 decision threshold, plus threshold-free AUROC/PR-AUC.

    Args:
        labels (np.ndarray): True labels [N].
        probs (np.ndarray): Melanoma probabilities [N].

    Returns:
        dict: Counts, accuracy, per-class precision and recall, macro-F1, AUROC, PR-AUC
            and the confusion matrix. AUROC and PR-AUC are None if only one class is present.
    """
    preds = (probs >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, labels=[0, 1], zero_division=0)
    both_classes = len(np.unique(labels)) == 2
    return {
        "n": int(len(labels)),
        "melanoma": int(labels.sum()),
        "accuracy": float((preds == labels).mean()),
        "precision_benign": float(precision[0]), "recall_benign": float(recall[0]),
        "precision_melanoma": float(precision[1]), "recall_melanoma": float(recall[1]),
        "macro_f1": float(f1.mean()),
        "auroc": float(roc_auc_score(labels, probs)) if both_classes else None,
        "pr_auc": float(average_precision_score(labels, probs)) if both_classes else None,
        "confusion_matrix [[TN, FP], [FN, TP]]": confusion_matrix(labels, preds, labels=[0, 1]).tolist(),
    }


def confidence_and_correct(labels, probs):
    """
    Confidence = probability of the predicted class, max(p, 1 - p), in [0.5, 1].

    Args:
        labels (np.ndarray): True labels [N].
        probs (np.ndarray): Melanoma probabilities [N].

    Returns:
        tuple: (confidence [N], correct [N] as booleans).
    """
    preds = (probs >= 0.5).astype(int)
    return np.maximum(probs, 1 - probs), preds == labels


def calibration(confidence, correct, n_bins=N_BINS):
    """
    Expected calibration error: split predictions into confidence bins and average the gap
    between mean confidence and actual accuracy in each bin, weighted by bin size. A model
    that says "90% sure" should be right 90% of the time; ECE measures how far off it is.

    This block of code is inspired from https://github.com/feder-cr/jev/wiki/expected-calibration-error-explained

    Args:
        confidence (np.ndarray): Confidence of each prediction, in [0.5, 1].
        correct (np.ndarray): True where the prediction was correct.
        n_bins (int): Number of equal-width confidence bins.

    Returns:
        tuple: (ECE as a float, list of {"confidence", "accuracy", "count"} per non-empty bin).
    """
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(confidence, edges[1:-1]), 0, n_bins - 1)
    ece, table = 0.0, []
    for b in range(n_bins):
        in_bin = bins == b
        if in_bin.any():
            gap = abs(confidence[in_bin].mean() - correct[in_bin].mean())
            ece += in_bin.mean() * gap
            table.append({"confidence": float(confidence[in_bin].mean()),
                          "accuracy": float(correct[in_bin].mean()), "count": int(in_bin.sum())})
    return float(ece), table


def risk_coverage(confidence, correct):
    """
    Risk-coverage curve and its area, AURC (Geifman and El-Yaniv, 2017).

    Predictions are sorted from most to least confident. Auto-deciding only the top k
    gives coverage k / N and risk = error rate among those k. A useful confidence score
    makes risk fall steeply as coverage shrinks; lower AURC is better. Tied confidences
    cannot be separated by any threshold, so each tied case gets the risk of accepting
    its whole group, which makes the result independent of the order of ties.

    Args:
        confidence (np.ndarray): Confidence of each prediction.
        correct (np.ndarray): True where the prediction was correct.

    Returns:
        tuple: (coverage array, risk array, AURC as a float).
    """

    order = np.argsort(-confidence, kind="stable")
    conf, errors = confidence[order], np.cumsum(~correct[order])
    n = len(order)
    k = np.arange(1, n + 1)
    coverage, risk = k / n, errors / k
    last_of_group = np.r_[conf[1:] != conf[:-1], True]
    group_end = np.minimum.accumulate(np.where(last_of_group, np.arange(n), n)[::-1])[::-1]
    risk = risk[group_end]
    return coverage, risk, float(risk.mean())


def synchronize(device):
    """
    Wait for queued GPU work to finish so timings are real.

    Args:
        device (torch.device): The device whose queued work to wait for.
    """
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def single_image_latency_ms(predict_one, dataset, device):
    """
    Median time to classify one image (batch size 1). predict_one(image) runs the whole
    prediction, so for the Siamese model it should include the kNN search.

    Args:
        predict_one (callable): Maps one image [1, 3, 224, 224] to a prediction.
        dataset (LesionDataset): Images to time; the first 5 are warm-up.
        device (torch.device): Device to run on.

    Returns:
        float: Median latency in milliseconds.
    """
    times = []
    for i in range(min(LATENCY_IMAGES + 5, len(dataset))):
        image = dataset[i][0].unsqueeze(0).to(device)
        synchronize(device)
        start = time.perf_counter()
        predict_one(image)
        synchronize(device)
        if i >= 5:  # skip warm-up iterations
            times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times))


def plot_reliability(results, path):
    """
    Reliability diagram: mean confidence against accuracy per confidence bin, per model.

    Args:
        results (dict): Model name to {"test_ece": float, "test_calibration_table": list}.
        path (str): Where to save the PNG.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0.5, 1], [0.5, 1], "--", color="grey", label="perfect calibration")
    for name, r in results.items():
        table = r["test_calibration_table"]
        ax.plot([b["confidence"] for b in table], [b["accuracy"] for b in table], marker="o",
                label=f"{name} (ECE {r['test_ece']:.3f})")
    ax.set(xlabel="confidence", ylabel="accuracy", title="Reliability diagram (test)", xlim=(0.5, 1), ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_risk_coverage(curves, path):
    """
    Risk-coverage curves of several models on one figure.

    Args:
        curves (dict): Model name to the (coverage, risk, aurc) tuple from risk_coverage.
        path (str): Where to save the PNG.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, (coverage, risk, aurc) in curves.items():
        ax.plot(coverage, risk, label=f"{name} (AURC {aurc:.3f})")
    ax.set(xlabel="coverage (fraction auto-decided)", ylabel="error rate among auto-decided",
           title="Risk-coverage (test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc_pr(scores, path):
    """
    ROC and precision-recall curves of several models, side by side.

    Args:
        scores (dict): Model name to (labels, melanoma probabilities).
        path (str): Where to save the PNG.
    """
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(10, 4))
    for name, (labels, probs) in scores.items():
        fpr, tpr, _ = roc_curve(labels, probs)
        precision, recall, _ = precision_recall_curve(labels, probs)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUROC {roc_auc_score(labels, probs):.3f})")
        ax_pr.plot(recall, precision, label=f"{name} (PR-AUC {average_precision_score(labels, probs):.3f})")
    
    ax_roc.plot([0, 1], [0, 1], "--", color="grey")
    ax_roc.set(xlabel="false positive rate", ylabel="true positive rate (recall)", title="ROC (test)")
    ax_pr.set(xlabel="recall", ylabel="precision", title="Precision-recall (test)")
    for ax in (ax_roc, ax_pr):
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)



def plot_k_tuning(k_scores, chosen_k, chosen_vote, path):
    """
    Validation AURC and referral rate for each candidate k, one line per kNN vote
    (plain / class-balanced), with the chosen setting marked.

    Args:
        k_scores (dict): Vote name to {k: {"val_aurc", "val_referred", ...}}.
        chosen_k (int): The selected k.
        chosen_vote (str): The selected vote ("plain" or "balanced").
        path (str): Where to save the PNG.
    """
    fig, (ax_aurc, ax_referred) = plt.subplots(1, 2, figsize=(10, 4))
    for vote, scores in k_scores.items():
        ks = sorted(scores)
        ax_aurc.plot(ks, [scores[k]["val_aurc"] for k in ks], marker="o", label=f"{vote} vote")
        ax_referred.plot(ks, [scores[k]["val_referred"] for k in ks], marker="o", label=f"{vote} vote")
    ax_aurc.set(ylabel="val AURC (lower is better)", title="Risk-coverage area vs k")
    ax_referred.set(ylabel="val cases referred", title="Referral at the recall target vs k", ylim=(0, 1.05))
    for ax in (ax_aurc, ax_referred):
        ax.axvline(chosen_k, linestyle="--", color="grey", label=f"chosen: {chosen_vote}, k = {chosen_k}")
        ax.set(xscale="log", xlabel="k (neighbours)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
