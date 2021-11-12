import gc
import os
from re import T
from types import SimpleNamespace
from typing import Callable, Mapping
from typing import Sequence

from ignite.contrib.metrics import ROC_AUC
from ignite.metrics import Accuracy
from ignite.metrics.epoch_metric import EpochMetric
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.autonotebook import tqdm

from nam.models.saver import Checkpointer
from nam.utils.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Subset
from torch.utils.data import random_split


class Trainer:

    def __init__(self, 
        models: Sequence[nn.Module], 
        dataset: torch.utils.data.Dataset,
        criterion: Callable,
        metric: str,
        batch_size: int = 1024,
        num_workers: int = 0,
        num_epochs: int = 1000,
        log_dir: str = None,
        val_split: float = 0.15,
        test_split: float = None,
        device: str = 'cpu',
        lr: float = 0.02082,
        decay_rate: float = 0.0,
        save_model_frequency: int = 0,
        patience: int = 40,
        monitor_loss: bool = True,
        early_stop_mode: str = 'min',
        regression: bool = True,
        num_learners: int = 1,
        random_state: int = 0
    ) -> None:
        self.models = [model.to(device) for model in models]
        self.dataset = dataset
        self.criterion = criterion
        self.metric_name = metric.upper()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_epochs = num_epochs
        self.device = device
        self.lr = lr
        self.decay_rate = decay_rate
        self.save_model_frequency = save_model_frequency
        self.patience = patience
        self.monitor_loss = monitor_loss
        self.early_stop_mode = early_stop_mode
        self.regression = regression
        self.num_learners = num_learners
        self.random_state = random_state

        self.log_dir = log_dir
        if not self.log_dir:
            self.log_dir = 'output'

        self.val_split = val_split
        self.test_split = test_split

        self.setup_dataloaders()
        
    def setup_dataloaders(self):
        test_size = int(self.test_split * len(self.dataset)) if self.test_split else 0
        val_size = int(self.val_split * (len(self.dataset) - test_size))
        train_size = len(self.dataset) - val_size - test_size

        train_subset, val_subset, test_subset = random_split(self.dataset, [train_size, val_size, test_size])

        # TODO: Possibly find way not to store data longterm -- maybe use close function
        self.train_dl = DataLoader(train_subset, batch_size=self.batch_size, 
            shuffle=True, num_workers=self.num_workers)

        self.val_dl = DataLoader(val_subset, batch_size=self.batch_size, 
            shuffle=False, num_workers=self.num_workers)

        self.test_dl = None
        if test_size > 0:
            self.test_dl = DataLoader(test_subset, batch_size=self.batch_size, 
                shuffle=False, num_workers=self.num_workers)

    def train_step(self, batch: torch.Tensor, model: nn.Module, 
                   optimizer: optim.Optimizer, metric: EpochMetric) -> torch.Tensor:
        """Performs a single gradient-descent optimization step."""
        features, targets, weights = [t.to(self.device) for t in batch]

        # Resets optimizer's gradients.
        optimizer.zero_grad()

        # Forward pass from the model.
        predictions, fnn_out = model(features)

        loss = self.criterion(predictions, targets, weights, fnn_out, model)
        self.update_metric(metric, predictions, targets, weights)

        # Backward pass.
        loss.backward()

        # Performs a gradient descent step.
        optimizer.step()

        return loss

    def train_epoch(self, model: nn.Module, optimizer: optim.Optimizer,
                    dataloader: torch.utils.data.DataLoader, metric: EpochMetric) -> torch.Tensor:
        """Performs an epoch of gradient descent optimization on
        `dataloader`."""
        model.train()
        loss = 0.0
        with tqdm(dataloader, leave=False) as pbar:
            for batch in pbar:
                # Performs a gradient-descent step.
                step_loss = self.train_step(batch, model, optimizer, metric)
                loss += step_loss

        metric_train = None
        if metric:
            metric_train = metric.compute()
            metric.reset()

        return loss / len(dataloader), metric_train

    def evaluate_step(self, model: nn.Module, batch: Mapping[str, torch.Tensor],
                      metric: EpochMetric) -> torch.Tensor:
        """Evaluates `model` on a `batch`."""
        features, targets, weights = [t.to(self.device) for t in batch]

        # Forward pass from the model.
        predictions, fnn_out = model(features)

        # Calculates loss on mini-batch.
        loss = self.criterion(predictions, targets, weights, fnn_out, model)
        self.update_metric(metric, predictions, targets, weights)

        return loss

    def evaluate_epoch(self, model: nn.Module, dataloader: torch.utils.data.DataLoader,
                       metric: EpochMetric) -> torch.Tensor:
        """Performs an evaluation of the `model` on the `dataloader`."""
        model.eval()
        loss = 0.0
        with tqdm(dataloader, leave=False) as pbar:
            for batch in pbar:
                # Accumulates loss in dataset.
                with torch.no_grad():
                    step_loss = self.evaluate_step(model, batch, metric)
                    loss += step_loss

        metric_val = None
        if metric:
            metric_val = metric.compute()
            metric.reset()

        return loss / len(dataloader), metric_val

    def train_ensemble(self):
        if self.regression:
            ss = ShuffleSplit(n_splits=self.num_learners, 
                test_size=self.val_split, random_state=self.random_state)
        else:
            ss = StratifiedShuffleSplit(n_splits=self.num_learners, 
                test_size=self.val_split, random_state=self.random_state)

        # TODO: Add parallelism for cpu
        for i, (train_ind, val_ind) in enumerate(ss.split(self.dataset.X, self.dataset.y)):
            train_subset = Subset(self.dataset, train_ind)
            val_subset = Subset(self.dataset, val_ind)
            
            train_dl = DataLoader(train_subset, batch_size=self.batch_size, 
                shuffle=True, num_workers=self.num_workers)

            val_dl = DataLoader(val_subset, batch_size=self.batch_size, 
                shuffle=False, num_workers=self.num_workers)

            log_subdir = os.path.join(self.log_dir, str(i))
            writer = TensorBoardLogger(log_dir=log_subdir)
            checkpointer = Checkpointer(model=self.models[i], log_dir=log_subdir)

            optimizer = torch.optim.Adam(self.models[i].parameters(),
                                         lr=self.lr,
                                         weight_decay=self.decay_rate)
            
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                gamma=0.995,
                                                step_size=1)

            metric = self.create_metric()

            self.train(i, train_dl, val_dl, optimizer, scheduler, writer, checkpointer, metric)

    def train(self, model_index, train_dl, val_dl, optimizer, scheduler, writer, checkpointer, metric):
        """Train the model for a specified number of epochs."""
        num_epochs = self.num_epochs
        best_loss_or_metric = float('inf')
        best_checkpoint = -1
        epochs_since_best = 0
        model = self.models[model_index]

        with tqdm(range(num_epochs)) as pbar_epoch:
            for epoch in pbar_epoch:
                # Trains model on whole training dataset, and writes on `TensorBoard`.
                loss_train, metric_train = self.train_epoch(model, optimizer, train_dl, metric)
                writer.write({"loss_train_epoch": loss_train.detach().cpu().numpy().item()})
                if metric:
                    writer.write({f"{self.metric_name}_train_epoch": metric_train})

                # Evaluates model on whole validation dataset, and writes on `TensorBoard`.
                loss_val, metric_val = self.evaluate_epoch(model, val_dl, metric)
                writer.write({"loss_val_epoch": loss_val.detach().cpu().numpy().item()})
                if metric:
                    writer.write({f"{self.metric_name}_val_epoch": metric_val})

                scheduler.step()

                # Updates progress bar description.
                desc = f"""Epoch({epoch}):
                    Training Loss: {loss_train.detach().cpu().numpy().item():.3f} |
                    Validation Loss: {loss_val.detach().cpu().numpy().item():.3f}"""
                if metric:    
                    desc += f' | {self.metric_name}: {metric_train:.3f}'
                pbar_epoch.set_description(desc)

                # Checkpoints model weights.
                if self.save_model_frequency > 0 and epoch % self.save_model_frequency == 0:
                    checkpointer.save(epoch)

                # Save best checkpoint for early stopping
                loss_or_metric = loss_val if self.monitor_loss else metric_val
                if self.early_stop_mode == 'max':
                    loss_or_metric = -1 * loss_or_metric

                if self.patience > 0 and loss_or_metric < best_loss_or_metric:
                    best_loss_or_metric = loss_or_metric
                    epochs_since_best = 0
                    checkpointer.save(epoch)
                    best_checkpoint = epoch

                # Stop training if early stopping patience exceeded
                epochs_since_best += 1
                if self.patience > 0 and epochs_since_best > self.patience:
                    self.models[model_index] = checkpointer.load(best_checkpoint)
                    break

    def close(self):
        del self.dataset
        gc.collect()
        return
    
    def create_metric(self):
        if self.metric_name.lower() == 'auroc':
            return ROC_AUC(lambda p: (torch.sigmoid(p[0]), p[1]))
        # TODO: Come up with a wrapper scheme to handle necessary data
        # transformations for different metrics, e.g. conver predictions
        # to 1's and 0's for accuracy.
        if self.metric_name.lower() == 'accuracy':
            return Accuracy(lambda p: ((p[0] > 0).type(torch.int32), p[1]))
        
        return None

    def update_metric(self, metric, predictions, targets, weights):
        if metric:
            predictions, targets = predictions.view(-1), targets.view(-1)
            indices = weights.view(-1) > 0
            predictions, targets = predictions[indices], targets[indices]
            metric.update((predictions, targets))
