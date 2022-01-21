import torch
import numpy as np
from src.classification.macro_multilabel_auc import macro_multilabel_auc
from src.classification.ap_metric import sklearn_mean_ap


def compute_all_metrics(predictions_list, targets_list, losses, target_cols):
    predictions = torch.cat(predictions_list).detach().cpu().numpy()
    targets = torch.cat(targets_list).cpu().numpy()
    roc_auc = macro_multilabel_auc(targets, predictions, target_cols)
    average_precision = sklearn_mean_ap(predictions, np.argmax(targets, axis=1))
    loss = np.mean(losses)
    return loss, roc_auc, average_precision