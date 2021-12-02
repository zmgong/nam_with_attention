from typing import Tuple, Union

from numpy.typing import ArrayLike
import pandas as pd
import torch


class NAMDataset(torch.utils.data.Dataset):

    def __init__(self,
                 X: Union[ArrayLike, pd.DataFrame],
                 y: Union[ArrayLike, pd.DataFrame],
                 w: Union[ArrayLike, pd.DataFrame] = None):
        """Dataset for NAMs that handles default weights.

        Args:
            X (Union[ArrayLike, pd.DataFrame]): Feature array.
            y (Union[ArrayLike, pd.DataFrame]): Target array.
            w (Union[ArrayLike, pd.DataFrame]): Weight array.
        """
        self.X = torch.tensor(X, requires_grad=False, dtype=torch.float)
        self.y = torch.tensor(y, requires_grad=False, dtype=torch.float)
        if not w:
            self.w = torch.clone(self.y)
            self.w[~torch.isnan(self.w)] = 1.0
            self.w[torch.isnan(self.w)] = 0.0
        else:
            self.w = torch.tensor(w, requires_grad=False, dtype=torch.float)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[ArrayLike, ...]:
        return self.X[idx], self.y[idx], self.w[idx]