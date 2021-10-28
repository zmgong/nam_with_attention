import torch

from data.data import NAMDataset
from models.nam import NAM, MultiTaskNAM
from nam.models import get_num_units
from trainer import Trainer


class NAMBase:
    def __init__(
        self,
        units_multiplier: int = 2,
        num_basis_functions: int = 1000,
        dropout: float = 0.1,
        val_split: float = 0.1, 
        test_split: float = 0.2,
        batch_size: int = 1024,
        device: str = 'cpu'
    ) -> None:
        self.units_multiplier = units_multiplier
        self.num_basis_functions = num_basis_functions
        self.dropout = dropout
        self.val_split = val_split
        self.test_split = test_split
        self.batch_size = batch_size
        self.device = device

    def fit(self, X, y, w=None) -> None:
        # (1) Create dataset
        self.dataset = NAMDataset(X, y, w)
        
        # (2) Initialize model
        self.model = NAM(
            num_inputs=X.shape[1],
            num_units=get_num_units(self.units_multiplier, self.num_basis_functions, X.shape[1]),
            dropout=self.dropout
        )

        # (3) Train model
        self.trainer = Trainer(
            model=self.model,
            device=self.device
        )
        
        self.trainer.train()

    def predict_proba(self, X) -> None:
        return self.model.forward(X) 

    def predict(self, X) -> None:
        raise NotImplementedError

    def plot(self, feature_index) -> None:
        pass


class NAMClassifier(NAMBase):
    def __init__(self) -> None:
        super().__init__()

    def predict_proba(self, X) -> None:
        return torch.sigmoid(super().predict_proba(X))

    def predict(self, X) -> None:
        return self.predict_proba(X) > 0.5

    
class NAMClassifier(NAMBase):
    def __init__(self) -> None:
        super().__init__()

    def predict_proba(self, X) -> None:
        return super().predict_proba(X)