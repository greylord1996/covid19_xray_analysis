import numpy as np
import torch
from tqdm import tqdm
from src.classification.compute_all_metrics import compute_all_metrics
from torch.nn import functional as F


def train_one_epoch(model, device, criterion, optimizer, train_loader, use_amp, target_cols):
    predictions_list = []
    targets_list = []

    model.train()
    bar = tqdm(train_loader)
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
    losses = []
    for batch_idx, (images, targets) in enumerate(bar):

        images, targets = images.to(device), targets.to(device)

        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(images)

                predictions_list += [F.sigmoid(logits)]
                targets_list += [targets.detach().cpu()]

                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            logits = model(images)

            predictions_list += [F.sigmoid(logits)]
            targets_list += [targets.detach().cpu()]

            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        losses.append(loss.item())
        smooth_loss = np.mean(losses[-30:])

        bar.set_description(f'loss: {loss.item():.5f}, smth: {smooth_loss:.5f}')

    loss_train, roc_auc_train, average_precision_train = compute_all_metrics(predictions_list, targets_list,
                                                                             losses, target_cols)
    return loss_train, roc_auc_train, average_precision_train
