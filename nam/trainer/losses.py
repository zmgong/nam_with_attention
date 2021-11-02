import torch
import torch.nn as nn
import torch.nn.functional as F


def bce_loss(logits: torch.Tensor, targets: torch.Tensor, weights: torch.tensor) -> torch.Tensor:
    """Cross entropy loss for binary classification.

    Args:
      logits: NAM model outputs
      targets: Binary class labels.

    Returns:
      Binary Cross-entropy loss between model predictions and the targets.
    """
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    loss *= weights
    loss = torch.sum(loss, dim=0)
    loss = loss / torch.sum(weights, dim=0)
    return torch.mean(loss)


def mse_loss(logits: torch.Tensor, targets: torch.Tensor, weights: torch.tensor) -> torch.Tensor:
    """Mean squared error loss for regression."""
    loss = F.mse_loss(logits, targets, reduction='none')
    loss *= weights
    loss = torch.sum(loss, dim=0)
    loss = loss / torch.sum(weights, dim=0)
    return loss


def reg_penalty(fnn_out: torch.Tensor, model: nn.Module,
    output_regularization: float, l2_regularization: float
) -> torch.Tensor:
    """Computes penalized loss with L2 regularization and output penalty.

    Args:
      config: Global config.
      model: Neural network model.
      inputs: Input values to be fed into the model for computing predictions.
      targets: Target values containing either real values or binary labels.

    Returns:
      The penalized loss.
    """

    def features_loss(per_feature_outputs):
        b, f = per_feature_outputs.shape[0], per_feature_outputs.shape[-1]
        out = torch.sum(per_feature_outputs ** 2) / (b * f)

        return output_regularization * out

    def weight_decay(model: nn.Module) -> torch.Tensor:
        """Penalizes the L2 norm of weights in each feature net."""
        num_networks = len(model.feature_nns)
        l2_losses = [(x**2).sum() for x in model.parameters()]
        return sum(l2_losses) / num_networks

    reg_loss = 0.0
    if output_regularization > 0:
        reg_loss += features_loss(fnn_out)

    if l2_regularization > 0:
        reg_loss += l2_regularization * weight_decay(model)

    return reg_loss


def make_penalized_loss_func(model, regression, output_regularization, l2_regularization):
    loss_func = mse_loss if regression else bce_loss
    def penalized_loss_func(logits, targets, weights, fnn_out):
        loss = loss_func(logits, targets, weights)
        loss += reg_penalty(fnn_out, model, output_regularization, l2_regularization)
        return loss
    return penalized_loss_func


