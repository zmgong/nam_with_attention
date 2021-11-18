import random
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike
import scipy
from sklearn.exceptions import NotFittedError
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
        hidden_sizes: list = [64, 32],
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
        patience: int = 60,
        monitor_loss: bool = True,
        early_stop_mode: str = 'min',
        loss_func: Callable = None,
        metric: str = None,
        num_learners: int = 1,
        random_state: int = 42
    ) -> None:
        self.units_multiplier = units_multiplier
        self.num_basis_functions = num_basis_functions
        self.hidden_sizes = hidden_sizes
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
        self.monitor_loss = monitor_loss
        self.early_stop_mode = early_stop_mode
        self.loss_func = loss_func
        self.metric = metric
        self.num_learners = num_learners
        self.random_state = random_state

        self._fitted = False

    def _set_random_state(self):
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        return
    
    def _initialize_models(self, X, y):
        self.num_tasks = y.shape[1] if len(y.shape) > 1 else 1
        self.num_inputs = X.shape[1]
        self.models = [
            NAM(num_inputs=self.num_inputs,
                num_units=get_num_units(self.units_multiplier, self.num_basis_functions, X),
                dropout=self.dropout,
                feature_dropout=self.feature_dropout,
                hidden_sizes=self.hidden_sizes)
            for _ in range(self.num_learners)
        ]
        return

    def partial_fit(self):
        # TODO: Implement for warm start. Ask Rich about warm start + ensembling.
        pass

    def fit(self, X, y, w=None) -> None:
        self._set_random_state()
        self._initialize_models(X, y)
        # self.preprocessor = MinMaxScaler(feature_range = (-1, 1) )

        dataset = NAMDataset(X, y, w)

        self.criterion = make_penalized_loss_func(self.loss_func, 
            self.regression, self.output_reg, self.l2_reg)

        self.trainer = Trainer(
            models=self.models,
            dataset=dataset,
            metric=self.metric,
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
            monitor_loss=self.monitor_loss,
            early_stop_mode=self.early_stop_mode,
            criterion=self.criterion,
            regression=self.regression,
            num_learners=self.num_learners,
            random_state=self.random_state
        )
        
        self.trainer.train_ensemble()
        self.trainer.close()
        self._fitted = True

    def predict(self, X) -> ArrayLike:
        if not self._fitted:
            raise NotFittedError('''This NAM instance is not fitted yet. Call \'fit\' 
                with appropriate arguments before using this method.''')
        
        prediction = np.zeros((X.shape[0],))
        if self.num_tasks > 1:
            prediction = np.zeros((X.shape[0], self.num_tasks))

        for model in self.models:
            preds, _ = model.forward(X)
            prediction += preds.detach().cpu().numpy()
        return prediction / self.num_learners

    def plot(self, feature_index) -> None:
        num_samples = 1000
        X = np.zeros((num_samples, self.num_inputs))
        X[:, feature_index] = np.linspace(-1.0, 1.0, num_samples)
        
        feature_outputs = []
        for model in self.models:
            # (examples, tasks, features)
            _, fnns_out = model.forward(torch.tensor(X, dtype=torch.float32))
            if self.num_tasks == 1:
                fnns_out = fnns_out.unsqueeze(dim=1)
            # (examples, tasks)
            feature_outputs.append(fnns_out[:, :, feature_index].detach().cpu().numpy())

        # (learners, examples, tasks)
        feature_outputs = np.stack(feature_outputs, axis=0)
        # (examples, tasks)
        y = np.mean(feature_outputs, axis=0)
        conf_int = self._get_confidence_interval(feature_outputs)

        if self.num_tasks == 1:
            y, conf_int = y.squeeze(1), conf_int.squeeze(1)
        
        return {'x': X[:, feature_index], 'y': y, 'conf_int': conf_int}

    @staticmethod
    def _get_confidence_interval(data, axis=0, confidence=0.95):
        n = len(data)
        se = scipy.stats.sem(data, axis=axis)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
        return h


class NAMClassifier(NAMBase):
    def __init__(
        self,
        units_multiplier: int = 2,
        num_basis_functions: int = 64,
        hidden_sizes: list = [64, 32],
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
        patience: int = 60,
        monitor_loss: bool = True,
        early_stop_mode: str = 'min',
        loss_func: Callable = None,
        metric: str = None,
        num_learners: int = 1
    ) -> None:
        super(NAMClassifier, self).__init__(
            units_multiplier=units_multiplier,
            num_basis_functions=num_basis_functions,
            hidden_sizes=hidden_sizes,
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
            patience=patience,
            monitor_loss=monitor_loss,
            early_stop_mode=early_stop_mode,
            loss_func=loss_func,
            metric=metric,
            num_learners=num_learners
        )
        self.regression = False

    def predict_proba(self, X) -> ArrayLike:
        out = scipy.special.expit(super().predict(X))
        return out

    def predict(self, X) -> ArrayLike:
        return self.predict_proba(X).round()

    
class NAMRegressor(NAMBase):
    def __init__(
        self,
        units_multiplier: int = 2,
        num_basis_functions: int = 64,
        hidden_sizes: list = [64, 32],
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
        patience: int = 60,
        monitor_loss: bool = True,
        early_stop_mode: str = 'min',
        loss_func: Callable = None,
        metric: str = None,
        num_learners: int = 1
    ) -> None:
        super(NAMRegressor, self).__init__(
            units_multiplier=units_multiplier,
            num_basis_functions=num_basis_functions,
            hidden_sizes=hidden_sizes,
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
            patience=patience,
            monitor_loss=monitor_loss,
            early_stop_mode=early_stop_mode,
            loss_func=loss_func,
            metric=metric,
            num_learners=num_learners
        )
        self.regression = True


class MultiTaskNAMClassifier(NAMClassifier):
    def __init__(
        self,
        units_multiplier: int = 2,
        num_basis_functions: int = 64,
        hidden_sizes: list = [64, 32],
        num_subnets: int = 2,
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
        patience: int = 60,
        monitor_loss: bool = True,
        early_stop_mode: str = 'min',
        loss_func: Callable = None,
        metric: str = None,
        num_learners: int = 1
    ) -> None:
        super(MultiTaskNAMClassifier, self).__init__(
            units_multiplier=units_multiplier,
            num_basis_functions=num_basis_functions,
            hidden_sizes=hidden_sizes,
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
            patience=patience,
            monitor_loss=monitor_loss,
            early_stop_mode=early_stop_mode,
            loss_func=loss_func,
            metric=metric,
            num_learners=num_learners
        )
        self.num_subnets = num_subnets

    def _initialize_models(self, X, y):
        self.num_inputs = X.shape[1]
        self.num_tasks = y.shape[1] if len(y.shape) > 1 else 1
        self.models = [
            MultiTaskNAM(num_inputs=X.shape[1],
                num_units=get_num_units(self.units_multiplier, self.num_basis_functions, X),
                num_subnets=self.num_subnets,
                num_tasks=y.shape[1],
                dropout=self.dropout,
                feature_dropout=self.feature_dropout,
                hidden_sizes=self.hidden_sizes)
            for _ in range(self.num_learners)
        ]


class MultiTaskNAMRegressor(NAMRegressor):
    def __init__(
        self,
        units_multiplier: int = 2,
        num_basis_functions: int = 64,
        hidden_sizes: list = [64, 32],
        num_subnets: int = 2,
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
        patience: int = 60,
        monitor_loss: bool = True,
        early_stop_mode: str = 'min',
        loss_func: Callable = None,
        metric: str = None,
        num_learners: int = 1
    ) -> None:
        super(MultiTaskNAMRegressor, self).__init__(
            units_multiplier=units_multiplier,
            num_basis_functions=num_basis_functions,
            hidden_sizes=hidden_sizes,
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
            patience=patience,
            monitor_loss=monitor_loss,
            early_stop_mode=early_stop_mode,
            loss_func=loss_func,
            metric=metric,
            num_learners=num_learners
        )
        self.num_subnets = num_subnets

    def _initialize_models(self, X, y):
        self.num_inputs = X.shape[1]
        self.num_tasks = y.shape[1] if len(y.shape) > 1 else 1
        self.models = [
            MultiTaskNAM(num_inputs=X.shape[1],
                num_units=get_num_units(self.units_multiplier, self.num_basis_functions, X),
                num_subnets=self.num_subnets,
                num_tasks=y.shape[1],
                dropout=self.dropout,
                feature_dropout=self.feature_dropout,
                hidden_sizes=self.hidden_sizes)
            for _ in range(self.num_learners)
        ]