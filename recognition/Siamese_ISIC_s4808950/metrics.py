import matplotlib
matplotlib.use("Agg")
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




