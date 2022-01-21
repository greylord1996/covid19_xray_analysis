import numpy as np
from sklearn.metrics import roc_auc_score


def macro_multilabel_auc(label, pred, target_cols):
    aucs = []
    for i in range(len(target_cols)):
        aucs.append(roc_auc_score(label[:, i], pred[:, i]))
    # print(np.round(aucs, 4))
    return np.mean(aucs)
