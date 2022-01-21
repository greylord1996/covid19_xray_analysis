import torch
import torch.nn as nn


# class FocalLoss(nn.modules.loss._WeightedLoss):
#     def __init__(self, weight=None, gamma=2, reduction='mean'):
#         super(FocalLoss, self).__init__(weight, reduction=reduction)
#         self.gamma = gamma
#         self.weight = weight #weight parameter will act as the alpha parameter to balance class weights
#
#     def forward(self, logits, targets, gamma=2):
#         l = logits.reshape(-1)
#         t = targets.reshape(-1)
#         p = torch.sigmoid(l)
#         p = torch.where(t >= 0.5, p, 1-p)
#         logp = - torch.log(torch.clamp(p, 1e-4, 1-1e-4))
#         loss = logp*((1-p)**gamma)
#         loss = 4*loss.mean()
#         return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits

    def forward(self, inputs, targets):
        bce_criterion = nn.BCEWithLogitsLoss()
        BCE_loss = bce_criterion(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        return F_loss