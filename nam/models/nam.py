import math
from typing import Sequence
from typing import Tuple
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from nam.models.featurenn import FeatureNN, MultiFeatureNN
from torch.nn import MultiheadAttention


# Referenced from https://github.com/sooftware/attentions/blob/master/attentions.py
class RelativeMultiHeadAttention(nn.Module):
    """
    Multi-head attention with relative positional encoding.
    This concept was proposed in the "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"
    Args:
        d_model (int): The dimension of model
        num_heads (int): The number of attention heads.
        dropout_p (float): probability of dropout
    Inputs: query, key, value, pos_embedding, mask
        - **query** (batch, time, dim): Tensor containing query vector
        - **key** (batch, time, dim): Tensor containing key vector
        - **value** (batch, time, dim): Tensor containing value vector
        - **pos_embedding** (batch, time, dim): Positional embedding tensor
        - **mask** (batch, 1, time2) or (batch, time1, time2): Tensor containing indices to be masked
    Returns:
        - **outputs**: Tensor produces by relative multi head attention module.
    """

    def __init__(
            self,
            d_model: int = 512,
            num_heads: int = 16,
            dropout_p: float = 0.1,
            pos_dim: int = 512,
    ):

        super(RelativeMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model % num_heads should be zero."

        self.d_model = d_model
        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(d_model)

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.pos_proj = nn.Linear(pos_dim, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout_p)
        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            pos_embedding: Tensor,
            mask: Optional[Tensor] = None,
    ) -> (Tensor, Tensor):

        batch_size = value.size(0)
        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        pos_embedding = self.pos_proj(pos_embedding)
        pos_embedding = pos_embedding.view(batch_size, -1, self.num_heads, self.d_head)



        content_score = torch.matmul((query + self.u_bias).transpose(1, 2), key.transpose(2, 3))
        pos_score = torch.matmul((query + self.v_bias).transpose(1, 2), pos_embedding.permute(0, 2, 3, 1))
        pos_score = self._compute_relative_positional_encoding(pos_score)

        score = (content_score + pos_score) / self.sqrt_dim

        if mask is not None:
            mask = mask.unsqueeze(1)
            score.masked_fill_(mask, -1e9)

        attn = F.softmax(score, -1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, value).transpose(1, 2)
        context = context.contiguous().view(batch_size, -1, self.d_model)

        return self.out_proj(context), attn

    def _compute_relative_positional_encoding(self, pos_score: Tensor) -> Tensor:
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)

        padded_pos_score = padded_pos_score.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)

        return pos_score


class NAM(torch.nn.Module):

    def __init__(
            self,
            enable_att: bool,
            num_inputs: int,
            num_units: list,
            hidden_sizes: list,
            dropout: float,
            feature_dropout: float,
            embed_dim: int,
            pos_embed: int,
            head_attention: int,
    ) -> None:
        super(NAM, self).__init__()
        assert len(num_units) == num_inputs
        self.enable_att = enable_att
        self.num_inputs = num_inputs
        self.num_units = num_units
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.feature_dropout = feature_dropout
        self.embed_dim = embed_dim
        self.pos_embed = pos_embed
        self.head_attention = head_attention

        self.dropout_layer = nn.Dropout(p=self.feature_dropout)
        ## Builds the FeatureNNs on the first call.
        self.feature_nns = nn.ModuleList([
            FeatureNN(
                input_shape=embed_dim-pos_embed,
                num_units=self.num_units[i],
                dropout=self.dropout, feature_num=i,
                hidden_sizes=self.hidden_sizes
            )
            for i in range(num_inputs)
        ])

        if self.enable_att:

            # self.multihead_attn_after = nn.MultiheadAttention(embed_dim, 1, batch_first=True)

            if self.pos_embed is not None:
                self.multihead_relative_attn = RelativeMultiHeadAttention(embed_dim-self.pos_embed, self.head_attention, pos_dim=self.pos_embed)
            else:
                self.multihead_attn_before = nn.MultiheadAttention(embed_dim, self.head_attention, batch_first=True)

        self._bias = torch.nn.Parameter(data=torch.zeros(1))

    def calc_outputs(self, inputs: torch.Tensor) -> Sequence[torch.Tensor]:
        """Returns the output computed by each feature net."""
        return [self.feature_nns[i](inputs[:, i]) for i in range(self.num_inputs)]

    def attention_forward(self, inputs, att_module):
        attn_output, attn_output_weights = att_module(inputs, inputs, inputs)
        return attn_output, attn_output_weights

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # query_before = self.query_proj_before(inputs)
        att_before_flag = self.enable_att
        attn_output_weights = None
        if att_before_flag:
            if self.pos_embed is not None:
                x_dim = self.embed_dim - self.pos_embed
                x = inputs[:, :, 0:x_dim]
                pos = inputs[:, :, x_dim:]
                inputs, attn_output_weights = self.multihead_relative_attn(x, x, x, pos)
            else:
                inputs, attn_output_weights = self.attention_forward(inputs, self.multihead_attn_before)
        feature_after_att = inputs
        individual_outputs = self.calc_outputs(inputs)
        conc_out = torch.cat(individual_outputs, dim=-1)
        dropout_out = self.dropout_layer(conc_out)

        out = torch.sum(dropout_out, dim=-1)

        return out + self._bias, dropout_out, feature_after_att, attn_output_weights


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
