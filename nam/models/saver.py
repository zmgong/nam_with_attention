"""Utility classes for saving model checkpoints."""

import os

import torch
import torch.nn as nn


class Checkpointer:
    """A simple `PyTorch` model load/save wrapper."""

    def __init__(
        self,
        model: nn.Module,
        log_dir: str = 'output',
        device: str = 'cpu'
    ) -> None:
        """Constructs a simple load/save checkpointer."""
        self._model = model
        self._ckpt_dir = os.path.join(log_dir, "ckpts")
        self._device = device
        os.makedirs(self._ckpt_dir, exist_ok=True)

    def save(
        self,
        epoch: int,
    ) -> str:
        """Saves the model to the `ckpt_dir/epoch/model.pt` file."""
        ckpt_path = os.path.join(self._ckpt_dir, "model-{}.pt".format(epoch))
        torch.save(self._model.state_dict(), ckpt_path)
        return ckpt_path

    def load(
        self,
        epoch: int,
    ) -> nn.Module:
        """Loads the model from the `ckpt_dir/epoch/model.pt` file."""
        ckpt_path = os.path.join(self._ckpt_dir, "model-{}.pt".format(epoch))
        self._model.load_state_dict(torch.load(ckpt_path, map_location=self._device))
        return self._model
