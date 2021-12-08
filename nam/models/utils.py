from typing import List, Union

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import torch
import torch.nn as nn


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.01)

def get_num_units(
    units_multiplier: int,
    num_basis_functions: int,
    X: Union[ArrayLike, pd.DataFrame]
) -> List:
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    num_unique_vals = [len(np.unique(X[:, i])) for i in range(X.shape[1])]
    num_units = [min(num_basis_functions, i * units_multiplier) for i in num_unique_vals]

    return num_units
