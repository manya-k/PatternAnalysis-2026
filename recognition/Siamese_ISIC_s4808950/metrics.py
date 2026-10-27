import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def confidence_and_correct(labels, probs):
    """Confidence = probability of the predicted class, max(p, 1 - p), in [0.5, 1]."""
    preds = (probs >= 0.5).astype(int)
    return np.maximum(probs, 1 - probs), preds == labels


def risk_coverage(confidence, correct):
    order = np.argsort(-confidence, kind="stable")
    conf, errors = confidence[order], np.cumsum(~correct[order])
    n = len(order)
    k = np.arange(1, n + 1)
    coverage, risk = k / n, errors / k
    last_of_group = np.r_[conf[1:] != conf[:-1], True]
    group_end = np.minimum.accumulate(np.where(last_of_group, np.arange(n), n)[::-1])[::-1]
    risk = risk[group_end]
    return coverage, risk, float(risk.mean())


def plot_k_tuning(k_scores, chosen_k, path):
    """Validation AURC and AUROC for each candidate k, with the chosen k marked."""
    ks = sorted(k_scores)
    fig, (ax_aurc, ax_auroc) = plt.subplots(1, 2, figsize=(10, 4))
    ax_aurc.plot(ks, [k_scores[k]["val_aurc"] for k in ks], marker="o")
    ax_aurc.set(ylabel="val AURC (lower is better)", title="Risk-coverage area vs k")
    ax_auroc.plot(ks, [k_scores[k]["val_auroc"] for k in ks], marker="o")
    ax_auroc.set(ylabel="val AUROC (higher is better)", title="AUROC vs k")
    for ax in (ax_aurc, ax_auroc):
        ax.axvline(chosen_k, linestyle="--", color="grey", label=f"chosen k = {chosen_k}")
        ax.set(xscale="log", xlabel="k (neighbours)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
