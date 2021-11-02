from sklearn.exceptions import NotFittedError
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
        num_basis_functions: int = 64,
        dropout: float = 0.1,
        feature_dropout: float = 0.05, 
        batch_size: int = 1024,
        num_workers: int = 0,
        num_epochs: int = 1000,
        log_dir: str = None,
        val_split: float = 0.15,
        device: str = 'cpu',
        lr: float = 0.02082,
        decay_rate: float = 0.0,
        output_reg: float = 0.2078,
        l2_reg: float = 0.0,
        save_model_frequency: int = 10,
        patience: int = 60
    ) -> None:
        self.units_multiplier = units_multiplier
        self.num_basis_functions = num_basis_functions
        self.dropout = dropout
        self.feature_dropout = feature_dropout
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
        self._fitted = False

    def fit(self, X, y, w=None) -> None:
        self.dataset = NAMDataset(X, y, w)
        
        self.model = NAM(
            num_inputs=X.shape[1],
            num_units=get_num_units(self.units_multiplier, self.num_basis_functions, self.dataset.X),
            dropout=self.dropout,
            feature_dropout=self.feature_dropout
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
            test_split=None,
            device=self.device,
            lr=self.lr,
            decay_rate=self.decay_rate,
            save_model_frequency=self.save_model_frequency,
            patience=self.patience,
            criterion=self.criterion
        )
        
        self.trainer.train()
        self._fitted = True

    def predict_proba(self, X) -> None:
        if not self._fitted:
            raise NotFittedError('''This NAM instance is not fitted yet. Call \'fit\' 
                with appropriate arguments before using this method.''')
        return self.model.forward(X) 

    def predict(self, X) -> None:
        raise NotImplementedError

    def plot(self, feature_index) -> None:
        pass


class NAMClassifier(NAMBase):
    def __init__(
        self,
        units_multiplier: int = 2,
        num_basis_functions: int = 64,
        dropout: float = 0.1,
        feature_dropout: float = 0.05, 
        batch_size: int = 1024,
        num_workers: int = 0,
        num_epochs: int = 1000,
        log_dir: str = None,
        val_split: float = 0.15,
        device: str = 'cpu',
        lr: float = 0.02082,
        decay_rate: float = 0.0,
        output_reg: float = 0.2078,
        l2_reg: float = 0.0,
        save_model_frequency: int = 10,
        patience: int = 60
    ) -> None:
        super(NAMClassifier, self).__init__(
            units_multiplier=units_multiplier,
            num_basis_functions=num_basis_functions,
            dropout=dropout,
            feature_dropout=feature_dropout,
            batch_size=batch_size,
            num_workers=num_workers,
            num_epochs=num_epochs,
            log_dir=log_dir,
            val_split=val_split,
            device=device,
            lr=lr,
            decay_rate=decay_rate,
            output_reg=output_reg,
            l2_reg=l2_reg,
            save_model_frequency=save_model_frequency,
            patience=patience
        )
        self.regression = False

    def predict_proba(self, X) -> None:
        return torch.sigmoid(super().predict_proba(X))

    def predict(self, X) -> None:
        return self.predict_proba(X) > 0.5

    
class NAMRegressor(NAMBase):
    def __init__(
        self,
        units_multiplier: int = 2,
        num_basis_functions: int = 64,
        dropout: float = 0.1,
        feature_dropout: float = 0.05, 
        batch_size: int = 1024,
        num_workers: int = 0,
        num_epochs: int = 1000,
        log_dir: str = None,
        val_split: float = 0.15,
        device: str = 'cpu',
        lr: float = 0.02082,
        decay_rate: float = 0.0,
        output_reg: float = 0.2078,
        l2_reg: float = 0.0,
        save_model_frequency: int = 10,
        patience: int = 60
    ) -> None:
        super(NAMRegressor, self).__init__(
            units_multiplier=units_multiplier,
            num_basis_functions=num_basis_functions,
            dropout=dropout,
            feature_dropout=feature_dropout,
            batch_size=batch_size,
            num_workers=num_workers,
            num_epochs=num_epochs,
            log_dir=log_dir,
            val_split=val_split,
            device=device,
            lr=lr,
            decay_rate=decay_rate,
            output_reg=output_reg,
            l2_reg=l2_reg,
            save_model_frequency=save_model_frequency,
            patience=patience
        )
        self.regression = True

    def predict_proba(self, X) -> None:
        return super().predict_proba(X)