from typing import Callable
import torch

from nam.data import NAMDataset
from nam.models import NAM, MultiTaskNAM
from nam.models import get_num_units
from nam.trainer import Trainer
from nam.trainer.losses import make_penalized_loss_func

class NAMBase:
    def __init__(
        self,
        units_multiplier: int = 2,
        num_basis_functions: int = 1000,
        dropout: float = 0.1, 
        batch_size: int = 1024,
        num_workers: int = 0,
        num_epochs: int = 1000,
        log_dir: str = None,
        val_split: float = 0.15,
        device: str = 'cpu',
        lr: float = 0.02082,
        decay_rate: float = 0.0,
        output_reg: float = 0.0,
        l2_reg: float = 0.0,
        save_model_frequency: int = 10,
        patience: int = 40
    ) -> None:
        self.units_multiplier = units_multiplier
        self.num_basis_functions = num_basis_functions
        self.dropout = dropout
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_epochs = num_epochs
        self.log_dir = log_dir
        self.val_split = val_split
        self.device = device
        self.lr = lr
        self.decay_rate = decay_rate
        self.output_reg = output_reg
        self.l2_reg = l2_reg
        self.save_model_frequency = save_model_frequency
        self.patience = patience
        self.criterion = None

    def fit(self, X, y, w=None) -> None:
        self.dataset = NAMDataset(X, y, w)
        
        self.model = NAM(
            num_inputs=X.shape[1],
            num_units=get_num_units(self.units_multiplier, self.num_basis_functions, self.dataset.X),
            dropout=self.dropout
        )

        self.criterion = make_penalized_loss_func(self.model, self.regression, 
            self.output_reg, self.l2_reg)

        self.trainer = Trainer(
            model=self.model,
            dataset=self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            num_epochs=self.num_epochs,
            log_dir=self.log_dir,
            val_split=self.val_split,
            train_split=1-self.val_split,
            test_split=None,
            device=self.device,
            lr=self.lr,
            decay_rate=self.decay_rate,
            save_model_frequency=self.save_model_frequency,
            patience=self.patience,
            criterion=self.criterion
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
        super(NAMClassifier, self).__init__()
        self.regression = False

    def predict_proba(self, X) -> None:
        return torch.sigmoid(super().predict_proba(X))

    def predict(self, X) -> None:
        return self.predict_proba(X) > 0.5

    
class NAMRegressor(NAMBase):
    def __init__(self) -> None:
        super(NAMRegressor).__init__()
        self.regression = True

    def predict_proba(self, X) -> None:
        return super().predict_proba(X)