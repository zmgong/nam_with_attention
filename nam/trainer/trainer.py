from types import SimpleNamespace
from typing import Callable, Mapping
from typing import Sequence

from ignite.contrib.metrics import ROC_AUC
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm.autonotebook import tqdm

from nam.models.saver import Checkpointer
# from nam.trainer.losses import make_penalized_loss_func
from nam.utils.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torch.utils.data import random_split


class Trainer:

    def __init__(self, 
        model: Sequence[nn.Module], 
        dataset: torch.utils.data.Dataset,
        criterion: Callable,
        batch_size: int = 1024,
        num_workers: int = 0,
        num_epochs: int = 1000,
        log_dir: str = None,
        val_split: float = 0.15,
        train_split: float = 0.85,
        test_split: float = None,
        device: str = 'cpu',
        lr: float = 0.02082,
        decay_rate: float = 0.0,
        save_model_frequency: int = 0,
        patience: int = 40
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_epochs = num_epochs
        self.device = device
        self.lr = lr
        self.decay_rate = decay_rate
        self.save_model_frequency = save_model_frequency
        self.patience = patience


        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.lr,
                                          weight_decay=self.decay_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                gamma=0.995,
                                                step_size=1)

        self.log_dir = log_dir
        if not self.log_dir:
            self.log_dir = 'output'
        self.writer = TensorBoardLogger(log_dir=self.log_dir)
        self.checkpointer = Checkpointer(model=model, log_dir=self.log_dir)
        self.best_checkpoint = None

        self.val_split = val_split
        self.train_split = train_split
        self.test_split = test_split

        # self.criterion = lambda inputs, targets, weights, fnns_out, model: penalized_loss(
        #     inputs, targets, weights, fnns_out, model)
        self.criterion = criterion#penalized_loss

        output_transform_fn = lambda output: (torch.sigmoid(output[0]), output[1])
        self.metric_train = ROC_AUC(output_transform_fn)
        self.metric_val = ROC_AUC(output_transform_fn)
        self.metric_name = 'AUROC'
        
        # TODO: Enable test split for users who want to use the Trainer outside of
        # NAMClassifier or NAMRegressor
        val_size = int(self.val_split * len(self.dataset))
        train_size = len(self.dataset) - val_size
        train_subset, val_subset = random_split(self.dataset, [train_size, val_size])
        self.train_dl = DataLoader(train_subset, batch_size=self.batch_size, 
            shuffle=True, num_workers=self.num_workers)
        self.val_dl = DataLoader(val_subset, batch_size=self.batch_size, 
            shuffle=False, num_workers=self.num_workers)

    def train_step(self, batch: torch.Tensor) -> torch.Tensor:
        """Performs a single gradient-descent optimization step."""
        features, targets, weights = batch
        features = features.to(self.device)
        targets = targets.to(self.device)
        weights = weights.to(self.device)

        # Resets optimizer's gradients.
        self.optimizer.zero_grad()

        # Forward pass from the model.
        predictions, fnn_out = self.model(features)

        loss = self.criterion(predictions, targets, weights, fnn_out)
        self.metric_train.update((predictions, targets))

        # Backward pass.
        loss.backward()

        # Performs a gradient descent step.
        self.optimizer.step()

        return loss

    def train_epoch(self, model: nn.Module, optimizer: optim.Optimizer,
                    dataloader: torch.utils.data.DataLoader) -> torch.Tensor:
        """Performs an epoch of gradient descent optimization on
        `dataloader`."""
        model.train()
        loss = 0.0
        with tqdm(dataloader, leave=False) as pbar:
            for batch in pbar:
                # Performs a gradient-descent step.
                step_loss = self.train_step(batch)
                loss += step_loss

        metric = self.metric_train.compute()
        self.metric_train.reset()

        return loss / len(dataloader), metric

    def evaluate_step(self, model: nn.Module, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Evaluates `model` on a `batch`."""
        features, targets, weights = batch
        features = features.to(self.device)
        targets = targets.to(self.device)
        weights = weights.to(self.device)

        # Forward pass from the model.
        predictions, fnn_out = self.model(features)

        # Calculates loss on mini-batch.
        loss = self.criterion(predictions, targets, weights, fnn_out)
        self.metric_val.update((predictions, targets))

        return loss

    def evaluate_epoch(self, model: nn.Module, dataloader: torch.utils.data.DataLoader) -> torch.Tensor:
        """Performs an evaluation of the `model` on the `dataloader`."""
        model.eval()
        loss = 0.0
        with tqdm(dataloader, leave=False) as pbar:
            for batch in pbar:
                # Accumulates loss in dataset.
                with torch.no_grad():
                    step_loss = self.evaluate_step(model, batch)
                    loss += step_loss

        metric = self.metric_val.compute()
        self.metric_val.reset()

        return loss / len(dataloader), metric

    def train(self):
        """Train the model for a specified number of epochs."""
        num_epochs = self.num_epochs
        best_loss = -float('inf')
        epochs_since_best = 0

        with tqdm(range(num_epochs)) as pbar_epoch:
            for epoch in pbar_epoch:
                # Trains model on whole training dataset, and writes on `TensorBoard`.
                loss_train, metric_train = self.train_epoch(self.model, self.optimizer, self.train_dl)
                self.writer.write({
                    "loss_train_epoch": loss_train.detach().cpu().numpy().item(),
                    f"{self.metric_name}_train_epoch": metric_train,
                })

                # Evaluates model on whole validation dataset, and writes on `TensorBoard`.
                loss_val, metric_val = self.evaluate_epoch(self.model, self.val_dl)
                self.writer.write({
                    "loss_val_epoch": loss_val.detach().cpu().numpy().item(),
                    f"{self.metric_name}_val_epoch": metric_val,
                })

                self.scheduler.step()

                # Updates progress bar description.
                pbar_epoch.set_description(f"""Epoch({epoch}):
                    Training Loss: {loss_train.detach().cpu().numpy().item():.3f} |
                    Validation Loss: {loss_val.detach().cpu().numpy().item():.3f} |
                    {self.metric_name}: {metric_train:.3f}""")

                # Checkpoints model weights.
                if self.save_model_frequency > 0 and epoch % self.save_model_frequency == 0:
                    self.checkpointer.save(epoch)

                # Save best checkpoint for early stopping
                # if self.config.patience > 0 and metric_val > best_loss:
                if self.patience > 0 and metric_val > best_loss:#loss_val < best_loss:
                    # TODO: support early stopping on both loss and metric
                    best_loss = metric_val
                    # best_loss = loss_val
                    epochs_since_best = 0
                    self.checkpointer.save(epoch)
                    self.best_checkpoint = epoch

                # Stop training if early stopping patience exceeded
                epochs_since_best += 1
                if self.patience > 0 and epochs_since_best > self.patience:
                    break

    def test(self):
        """Evaluate the model on the test set."""
        if not self.test_split:
            # TODO: Find correct exception to throw here.
            raise Exception() 

        if self.config.patience > 0:
            self.model = self.checkpointer.load(self.best_checkpoint)
        
        num_epochs = 1
        with tqdm(range(num_epochs)) as pbar_epoch:
            for epoch in pbar_epoch:
                # Evaluates model on whole test set, and writes on `TensorBoard`.
                loss_test, metrics_test = self.evaluate_epoch(self.model, self.test_dl)
                self.writer.write({
                    "loss_test_epoch": loss_test.detach().cpu().numpy().item(),
                    f"{self.metric_name}_test_epoch": metrics_test,
                })

                # Updates progress bar description.
                pbar_epoch.set_description(f"""Epoch({epoch}):
                    Test Loss: {loss_test.detach().cpu().numpy().item():.3f} |
                    Test {self.metric_name}: {metrics_test:.3f}""")
