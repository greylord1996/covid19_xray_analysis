from src.classification.compute_all_metrics import compute_all_metrics
import torch
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F


def valid_func(model, device, criterion, valid_loader, target_cols):
    model.eval()
    bar = tqdm(valid_loader)

    targets_list = []
    losses = []
    predictions_list = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(bar):
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            predictions_list += [F.sigmoid(logits)]
            targets_list += [targets.detach().cpu()]
            loss = criterion(logits, targets)
            losses.append(loss.item())
            smooth_loss = np.mean(losses[-30:])
            bar.set_description(f'loss: {loss.item():.5f}, smth: {smooth_loss:.5f}')

    loss_valid, roc_auc_valid, average_precision_valid = compute_all_metrics(predictions_list, targets_list, losses,
                                                                             target_cols)
    return loss_valid, roc_auc_valid, average_precision_valid