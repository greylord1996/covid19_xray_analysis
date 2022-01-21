import numpy as np
from sklearn.metrics import average_precision_score


def sklearn_mean_ap(preds, targs):
    """
    Difference from COCO is precision is not interpolated
    """
    return np.mean([average_precision_score(targs == i, preds[:, i]) for i in range(4)])*2/3