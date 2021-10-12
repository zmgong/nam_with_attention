from sklearn.metrics import roc_auc_score
import torch


def mae(logits, targets):
    return ((logits.view(-1) - targets.view(-1)).abs().sum() / logits.numel()).item()


def accuracy(logits, targets):
    return (((targets.view(-1) > 0) == (logits.view(-1) > 0.5)).sum() / targets.numel()).item()


def roc_auc(logits, targets):
    preds = torch.sigmoid(logits)
    y_pred = preds.view(-1).cpu().detach().numpy()
    y_true = targets.view(-1).cpu().detach().numpy()
    return roc_auc_score(y_true, y_pred)