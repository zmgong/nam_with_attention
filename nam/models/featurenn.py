import torch
import torch.nn as nn
import torch.nn.functional as F

from nam.models.base import Model

from .activation import ExU
from .activation import LinReLU


class FeatureNN(torch.nn.Module):
    """Neural Network model for each individual feature."""

    def __init__(
        self,
        input_shape: int,
        num_units: int,
        dropout: float,
        feature_num: int,
        hidden_sizes: list = [64, 32],
        activation: str = 'relu'
    ) -> None:
        """Initializes FeatureNN hyperparameters.

        Args:
          num_units: Number of hidden units in first hidden layer.
          dropout: Coefficient for dropout regularization.
          feature_num: Feature Index used for naming the hidden layers.
        """
        super(FeatureNN, self).__init__()
        self._input_shape = input_shape
        self._num_units = num_units
        self._feature_num = feature_num
        self.dropout = nn.Dropout(p=dropout)
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        
        all_hidden_sizes = [self._num_units] + self.hidden_sizes

        layers = []

        ## First layer is ExU
        if self.activation == "exu":
            layers.append(ExU(in_features=input_shape, out_features=num_units))
        else:
            layers.append(LinReLU(in_features=input_shape, out_features=num_units))

        ## Hidden Layers
        for in_features, out_features in zip(all_hidden_sizes, all_hidden_sizes[1:]):
            layers.append(LinReLU(in_features, out_features))

        ## Last Linear Layer
        layers.append(nn.Linear(in_features=all_hidden_sizes[-1], out_features=1, bias=False))

        self.model = nn.ModuleList(layers)

    def forward(self, inputs) -> torch.Tensor:
        """Computes FeatureNN output with either evaluation or training
        mode."""
        outputs = inputs.unsqueeze(1)
        for layer in self.model:
            outputs = self.dropout(layer(outputs))
        return outputs


class MultiFeatureNN(Model):
    def __init__(
        self,
        config,
        name,
        *,
        input_shape: int,
        num_units: int,
        feature_num: int,
        num_subnets: int,
        num_tasks: int
    ) -> None:
        """Initializes FeatureNN hyperparameters.
        Args:
            num_units: Number of hidden units in first hidden layer.
            dropout: Coefficient for dropout regularization.
            feature_num: Feature Index used for naming the hidden layers.
        """
        super(MultiFeatureNN, self).__init__(config, name)
        subnets = [
            FeatureNN(
            config,
            name,
            input_shape=input_shape,
            num_units=num_units,
            feature_num=feature_num,
            )
            for i in range(num_subnets)
        ]
        self.feature_nns = nn.ModuleList(subnets)
        self.linear = torch.nn.Linear(num_subnets, num_tasks)

    def forward(self, inputs) -> torch.Tensor:
        """Computes FeatureNN output with either evaluation or training mode."""
        individual_outputs = []
        for fnn in self.feature_nns:
            individual_outputs.append(fnn(inputs)) 

        # (batch_size, num_subnets)
        stacked = torch.stack(individual_outputs, dim=-1)
        # (batch_size, num_tasks)
        weighted = self.linear(stacked)
        return weighted