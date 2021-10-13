from types import SimpleNamespace
from typing import Mapping
from typing import Sequence

from ignite.contrib.metrics import ROC_AUC
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm.autonotebook import tqdm

from nam.config import Config
from nam.models.saver import Checkpointer
from nam.trainer.losses import penalized_loss, mse_loss
from nam.utils.loggers import TensorBoardLogger


class Trainer:

    def __init__(self, config: SimpleNamespace, model: Sequence[nn.Module], dataset: torch.utils.data.Dataset) -> None:
        self.config = Config(**vars(config))  #config
        self.model = model
        self.dataset = dataset
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.config.lr,
                                          weight_decay=self.config.decay_rate)

        self.writer = TensorBoardLogger(config)
        self.checkpointer = Checkpointer(model=model, config=config)

        self.criterion = lambda inputs, targets, weights, fnns_out, model: penalized_loss(
            self.config, inputs, targets, weights, fnns_out, model)

        output_transform_fn = lambda output: (torch.sigmoid(output[0]), output[1])
        self.metric_train = ROC_AUC(output_transform_fn)
        self.metric_val = ROC_AUC(output_transform_fn)
        self.metric_name = 'AUROC'

        if config.wandb:
            wandb.watch(models=self.model, log='all', log_freq=10)

        self.dataloader_train, self.dataloader_val = self.dataset.train_dataloaders()
        self.dataloader_test = self.dataset.test_dataloaders()

    def train_step(self, model: nn.Module, optimizer: optim.Optimizer, batch: torch.Tensor) -> torch.Tensor:
        """Performs a single gradient-descent optimization step."""

        features, targets = batch

        # Resets optimizer's gradients.
        self.optimizer.zero_grad()

        # Forward pass from the model.
        predictions, fnn_out = self.model(features)

        loss = self.criterion(predictions, targets, None, fnn_out, self.model)
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
                step_loss = self.train_step(model, optimizer, batch)
                loss += step_loss

        metric = self.metric_train.compute()
        self.metric_train.reset()

        return loss / len(dataloader), metric

    def evaluate_step(self, model: nn.Module, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Evaluates `model` on a `batch`."""
        features, targets = batch

        # Forward pass from the model.
        predictions, fnn_out = self.model(features)

        # Calculates loss on mini-batch.
        loss = self.criterion(predictions, targets, None, fnn_out, self.model)
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
        num_epochs = self.config.num_epochs

        with tqdm(range(num_epochs)) as pbar_epoch:
            for epoch in pbar_epoch:
                # Trains model on whole training dataset, and writes on `TensorBoard`.
                loss_train, metric_train = self.train_epoch(self.model, self.optimizer, self.dataloader_train)
                self.writer.write({
                    "loss_train_epoch": loss_train.detach().cpu().numpy().item(),
                    f"{self.metric_name}_train_epoch": metric_train,
                })

                # Evaluates model on whole validation dataset, and writes on `TensorBoard`.
                loss_val, metrics_val = self.evaluate_epoch(self.model, self.dataloader_val)
                self.writer.write({
                    "loss_val_epoch": loss_val.detach().cpu().numpy().item(),
                    f"{self.metric_name}_val_epoch": metrics_val,
                })

                # Checkpoints model weights.
                if epoch % self.config.save_model_frequency == 0:
                    self.checkpointer.save(epoch)

                # Updates progress bar description.
                pbar_epoch.set_description(f"""Epoch({epoch}):
                    Training Loss: {loss_train.detach().cpu().numpy().item():.3f} |
                    Validation Loss: {loss_val.detach().cpu().numpy().item():.3f} |
                    {self.metric_name}: {metric_train:.3f}""")

    def test(self):
        """Evaluate the model on the test set."""
        num_epochs = 1

        with tqdm(range(num_epochs)) as pbar_epoch:
            for epoch in pbar_epoch:
                # Evaluates model on whole test set, and writes on `TensorBoard`.
                loss_test, metrics_test = self.evaluate_epoch(self.model, self.dataloader_test)
                self.writer.write({
                    "loss_test_epoch": loss_test.detach().cpu().numpy().item(),
                    f"{self.metric_name}_test_epoch": metrics_test,
                })

                # Updates progress bar description.
                pbar_epoch.set_description(f"""Epoch({epoch}):
                    Test Loss: {loss_test.detach().cpu().numpy().item():.3f} |
                    Test {self.metric_name}: {metrics_test:.3f}""")
