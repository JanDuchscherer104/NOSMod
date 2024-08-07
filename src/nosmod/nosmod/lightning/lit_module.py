from typing import Any, Dict, Optional, Type, Union

import pytorch_lightning as pl
import torch
from pydantic import Field
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

from ..network_components.mod_trans_demod_system import NosmodSystemParams
from ..network_components.shared_params import TemporalParams
from ..utils import CONSOLE, BaseConfig


class HParams(BaseConfig["LitNOSModModule"]):
    target: Type["LitNOSModModule"] = Field(default_factory=lambda: LitNOSModModule)

    lr: float = 1e-3
    weight_decay: float = 1e-4

    nosmod_system: NosmodSystemParams = Field(default_factory=NosmodSystemParams)


class LitNOSModModule(pl.LightningModule):
    config: HParams

    def __init__(self, params: HParams):
        """
        Initialize the LitNOSModModule with configuration and hyperparameters.

        Args:
            config (Config): Configuration object containing various settings.
            hparams (HParams): Hyperparameters for the model and training.
        """
        super().__init__()

        self.params = params
        self.save_hyperparameters(params.model_dump())

        self.model = self.params.nosmod_system.setup_target().to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        print(f"Moved {self.model.__class__.__name__} to {self.device}.")

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Model output tensor.
        """
        return self.model(x)

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        """
        Perform a single training step.

        Args:
            batch (Tensor): Batch of data containing input tensor.
            batch_idx (int): Index of the batch.

        Returns:
            Tensor: Loss tensor.
        """
        y_pred = self(batch)
        # Assuming the target is the same as the input for now
        loss = self.criterion(y_pred, batch)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Tensor, batch_idx: int):
        """
        Perform a single validation step.

        Args:
            batch (Tensor): Batch of data containing input tensor.
            batch_idx (int): Index of the batch.

        Returns:
            Tensor: Loss tensor.
        """
        y_pred = self(batch)
        # Assuming the target is the same as the input for now
        loss = self.criterion(y_pred, batch)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configure the optimizers and learning rate schedulers.

        Returns:
            Dict[str, Any]: Dictionary containing optimizer and scheduler.
        """
        optimizer = AdamW(
            params=self.model.parameters(),
            weight_decay=self.params.weight_decay,
            lr=self.params.lr,
        )

        scheduler = {
            "scheduler": StepLR(optimizer, step_size=10, gamma=0.1),
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
