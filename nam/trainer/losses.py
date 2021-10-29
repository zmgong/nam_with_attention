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
    loss = loss / torch.count_nonzero(weights, dim=0)
    return torch.mean(loss)


def mse_loss(logits: torch.Tensor, targets: torch.Tensor, weights: torch.tensor) -> torch.Tensor:
    """Mean squared error loss for regression."""
    loss = F.mse_loss(logits, targets, reduction='none')
    loss *= weights
    loss = torch.sum(loss, dim=0)
    loss = loss / torch.sum(weights, dim=0)
    return loss


def penalized_loss(loss: torch.Tensor, logits: torch.Tensor, targets: torch.Tensor, weights: torch.tensor,
                   fnn_out: torch.Tensor, model: nn.Module) -> torch.Tensor:
    """Computes penalized loss with L2 regularization and output penalty.

    Args:
      config: Global config.
      model: Neural network model.
      inputs: Input values to be fed into the model for computing predictions.
      targets: Target values containing either real values or binary labels.

    Returns:
      The penalized loss.
    """

    def features_loss(per_feature_outputs: torch.Tensor) -> torch.Tensor:
        """Penalizes the L2 norm of the prediction of each feature net."""
        per_feature_norm = [  # L2 Regularization
            torch.mean(torch.square(outputs)) for outputs in per_feature_outputs
        ]
        return sum(per_feature_norm) / len(per_feature_norm)

    def weight_decay(model: nn.Module) -> torch.Tensor:
        """Penalizes the L2 norm of weights in each feature net."""
        num_networks = len(model.feature_nns)
        l2_losses = [(x**2).sum() for x in model.parameters()]
        return sum(l2_losses) / num_networks

    # loss_func = mse_loss if model.regression else bce_loss
    # loss = loss_func(logits, targets, weights)

    reg_loss = 0.0
    if model.output_regularization > 0:
        reg_loss += model.output_regularization * features_loss(fnn_out)

    if model.l2_regularization > 0:
        reg_loss += model.l2_regularization * weight_decay(model)

    return loss + reg_loss


def penalized_mse_loss(logits: torch.Tensor, targets: torch.Tensor, weights: torch.tensor,
                   fnn_out: torch.Tensor, model: nn.Module) -> torch.Tensor:
    loss = mse_loss(logits, targets, weights)
    return penalized_loss(loss, logits, targets, weights, fnn_out, model)


def penalized_bce_loss(logits: torch.Tensor, targets: torch.Tensor, weights: torch.tensor,
                   fnn_out: torch.Tensor, model: nn.Module) -> torch.Tensor:
    loss = bce_loss(logits, targets, weights)
    return penalized_loss(loss, logits, targets, weights, fnn_out, model)
