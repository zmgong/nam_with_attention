from typing import Sequence
from typing import Tuple

import torch
import torch.nn as nn

from nam.models.featurenn import FeatureNN, MultiFeatureNN
from torch.nn import MultiheadAttention


class NAM(torch.nn.Module):

    def __init__(
            self,
            num_inputs: int,
            num_units: list,
            hidden_sizes: list,
            dropout: float,
            feature_dropout: float
    ) -> None:
        super(NAM, self).__init__()
        assert len(num_units) == num_inputs
        self.num_inputs = num_inputs
        self.num_units = num_units
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.feature_dropout = feature_dropout

        self.dropout_layer = nn.Dropout(p=self.feature_dropout)
        ## Builds the FeatureNNs on the first call.
        self.feature_nns = nn.ModuleList([
            FeatureNN(
                input_shape=1,
                num_units=self.num_units[i],
                dropout=self.dropout, feature_num=i,
                hidden_sizes=self.hidden_sizes
            )
            for i in range(num_inputs)
        ])

        self.multihead_attn_before = nn.MultiheadAttention(num_inputs + 1, 1, batch_first=True)
        self.multihead_attn_after = nn.MultiheadAttention(num_inputs + 1, 1, batch_first=True)

        self._bias = torch.nn.Parameter(data=torch.zeros(1))

    def calc_outputs(self, inputs: torch.Tensor) -> Sequence[torch.Tensor]:
        """Returns the output computed by each feature net."""
        return [self.feature_nns[i](inputs[:, i]) for i in range(self.num_inputs)]

    def attention_forward(self, inputs, att_module):
        inputs = torch.unsqueeze(inputs, -1)
        index_emb = torch.zeros((inputs.shape[1], inputs.shape[1]))
        for index, row in enumerate(index_emb):
            row[index] = 1
        index_emb = torch.reshape(index_emb.repeat(inputs.shape[0], 1),
                                  (inputs.shape[0], inputs.shape[1], inputs.shape[1]))
        inputs = torch.cat((index_emb, inputs), dim=-1)

        attn_output, attn_output_weights = att_module(inputs, inputs, inputs)
        inputs = attn_output[:, :, -1]
        return inputs

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # query_before = self.query_proj_before(inputs)
        att_before_flag = True
        if att_before_flag:
            inputs = self.attention_forward(inputs, self.multihead_attn_before)

        individual_outputs = self.calc_outputs(inputs)
        conc_out = torch.cat(individual_outputs, dim=-1)
        dropout_out = self.dropout_layer(conc_out)

        att_after_flag = False
        if att_after_flag:
            dropout_out = self.attention_forward(dropout_out, self.multihead_attn_after)
        out = torch.sum(dropout_out, dim=-1)
        return out + self._bias, dropout_out


class MultiTaskNAM(torch.nn.Module):

    def __init__(
            self,
            num_inputs: list,
            num_units: int,
            num_subnets: int,
            num_tasks: int,
            hidden_sizes: list,
            dropout: float,
            feature_dropout: float
    ) -> None:
        super(MultiTaskNAM, self).__init__()

        assert len(num_units) == num_inputs
        self.num_inputs = num_inputs
        self.num_units = num_units
        self.num_subnets = num_subnets
        self.num_tasks = num_tasks
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.feature_dropout = feature_dropout
        self.dropout_layer = nn.Dropout(p=self.feature_dropout)
        ## Builds the FeatureNNs on the first call.
        self.feature_nns = nn.ModuleList([
            MultiFeatureNN(
                input_shape=1,
                feature_num=i,
                num_units=self.num_units[i],
                num_subnets=self.num_subnets,
                num_tasks=self.num_tasks,
                dropout=self.dropout,
                hidden_sizes=self.hidden_sizes
            )
            for i in range(self.num_inputs)
        ])

        self.multihead_attn_before = nn.MultiheadAttention(num_inputs + 1, 1)
        self.multihead_attn_after = nn.MultiheadAttention(num_inputs + 1, 1)

        self._bias = torch.nn.Parameter(data=torch.zeros(1, self.num_tasks))

    def calc_outputs(self, inputs: torch.Tensor) -> Sequence[torch.Tensor]:
        """Returns the output computed by each feature net."""
        return [self.feature_nns[i](inputs[:, i]) for i in range(self.num_inputs)]

    def attention_forward(self, inputs, att_module):
        inputs = torch.unsqueeze(inputs, -1)
        index_emb = torch.zeros((inputs.shape[1], inputs.shape[1]))
        for index, row in enumerate(index_emb):
            row[index] = 1
        index_emb = torch.reshape(index_emb.repeat(inputs.shape[0], 1),
                                  (inputs.shape[0], inputs.shape[1], inputs.shape[1]))
        inputs = torch.cat((index_emb, inputs), dim=-1)

        attn_output, attn_output_weights = att_module(inputs, inputs, inputs)
        inputs = attn_output[:, :, -1]
        return inputs

    def forward(
            self,
            inputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        att_before_flag = True
        if att_before_flag:
            inputs = self.attention_forward(inputs, self.multihead_attn_before)

        # tuple: (batch, num_tasks) x num_inputs
        individual_outputs = self.calc_outputs(inputs)
        # (batch, num_tasks, num_inputs)
        stacked_out = torch.stack(individual_outputs, dim=-1).squeeze(dim=1)
        dropout_out = self.dropout_layer(stacked_out)
        # (batch, num_tasks)
        summed_out = torch.sum(dropout_out, dim=2) + self._bias
        return summed_out, dropout_out

    def feature_output(self, feature_index, inputs):
        return self.feature_nns[feature_index](inputs)
