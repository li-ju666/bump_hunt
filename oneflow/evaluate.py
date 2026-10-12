import numpy as np


def auroc(pos, neg):
    """P(score of a positive > score of a negative)."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())
