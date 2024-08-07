from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Type
from warnings import warn

import torch
from pydantic import Field
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.types import STEP_OUTPUT

from ..config import _ExperimentConfig
from ..utils import CONSOLE
from .lit_datamodule import LitNosmodDatamodule
from .lit_module import LitNOSModModule


class ExperimentConfig(_ExperimentConfig):
    target: Type["TrainerFactory"] = Field(default_factory=lambda: TrainerFactory)


class TrainerFactory:
    def __init__(self, config: _ExperimentConfig):
        """Private constructor to set config and hyper_params."""
        self.config = config

    @classmethod
    def setup_target(
        cls,
        config: _ExperimentConfig,
    ) -> Trainer:
        """Create and initialize Callback instances."""

        torch.set_float32_matmul_precision(config.matmul_precision)

        factory_instance = TrainerFactory(config)
        factory_instance.config = config

        # if config.is_optuna:
        #     config.active_callbacks["OptunaPruning"] = True

        if config.is_debug:
            config.is_gpu = False
            config.is_fast_dev_run = True
            config.num_workers = 0
            config.is_multiproc = False
            config.verbose = True
            # config.is_mlflow = False
            torch.autograd.set_detect_anomaly(True)

            config.active_callbacks["ModelCheckpoint"] = False

        # Create Trainer
        return factory_instance

    def create_all(
        self,
        **trainer_kwargs,
    ) -> Tuple[Trainer, LitNOSModModule, LitNosmodDatamodule]:
        """Create and initialize Callback instances."""
        callbacks = self._assemble_callbacks()
        tb_logger = self._assemble_loggers()
        config = self.config

        trainer = Trainer(
            accelerator="auto" if config.is_gpu else "cpu",
            logger=tb_logger,
            callbacks=callbacks,
            max_epochs=config.max_epochs,
            default_root_dir=config.paths.root,
            fast_dev_run=config.is_fast_dev_run,
            log_every_n_steps=config.log_every_n_steps,
            enable_model_summary=not config.active_callbacks["ModelSummary"],
            **trainer_kwargs,
        )
        if isinstance(config.from_ckpt, Path):
            print(f"Loading model from checkpoint: {config.from_ckpt}")
            lit_module = LitNOSModModule.load_from_checkpoint(
                config.from_ckpt, hparams=config.module
            )
        else:
            lit_module = config.module.setup_target()
        lit_datamodule = config.datamodule.setup_target()

        CONSOLE.log(config)
        return (
            trainer,
            lit_module,
            lit_datamodule,
        )

    def _get_callback_map(self) -> Dict[str, Callable]:
        return {
            "ModelCheckpoint": self._create_model_checkpoint,
            "TQDMProgressBar": self._create_tqdm_progress_bar,
            "EarlyStopping": self._create_early_stopping,
            # "BatchSizeFinder": self._create_batch_size_finder,
            "LearningRateMonitor": self._create_lr_monitor,
            "ModelSummary": self._create_model_summary,
            # "OptunaPruning": self._create_optuna_pruning,
        }

    def _create_model_checkpoint(self):
        return ModelCheckpoint(
            dirpath=self.config.paths.checkpoints,
            filename=f"{self.config.mlflow_config.run_name}-{{epoch:02d}}-{{val_loss:.2f}}",
            monitor="val_loss",
            verbose=self.config.verbose,
        )

    def _create_tqdm_progress_bar(self):
        return CustomTQDMProgressBar()

    def _create_lr_monitor(self):
        return LearningRateMonitor(logging_interval="step", log_momentum=True)

    def _create_early_stopping(self):
        return EarlyStopping(
            monitor="val_loss",
            patience=self.config.early_stopping_patience,
            verbose=self.config.verbose,
            mode="min",
        )

    # def _create_optuna_pruning(self):
    #     if self.trial is not None:
    #         return PyTorchLightningPruningCallback(
    #             trial=self.trial,
    #             monitor=self.config.optuna_config.monitor,
    #         )

    def _create_model_summary(self):
        return ModelSummary(max_depth=4)

    # def _create_batch_size_finder(self):
    #     return BatchSizeFinder(
    #         mode="binsearch",
    #         steps_per_trial=3,
    #         init_val=self.hparams.optimizer.batch_size,
    #         max_trials=25,
    #         batch_arg_name="batch_size",
    #     )

    def _assemble_loggers(self):
        return [
            TensorBoardLogger(
                save_dir=self.config.paths.tb_logs,
                name=self.config.mlflow_config.run_name,
            ),
        ]

    def _assemble_callbacks(self) -> List[Callback]:
        callbacks = []
        callback_map = self._get_callback_map()
        for key, is_active in self.config.active_callbacks.items():
            if is_active:
                create_callback = callback_map.get(key)
                if create_callback:
                    callback = create_callback()
                    if callback:
                        callbacks.append(callback)
                    else:
                        CONSOLE.warn(f"Callback {key} could not be created.")
                else:
                    warn(f"No method found in TrainerFactory for key {key}.")

        return callbacks


class CustomTQDMProgressBar(TQDMProgressBar):
    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        description = f"train_loss: {trainer.callback_metrics.get('train_loss', 0):.2f}"
        self.train_progress_bar.set_postfix_str(description, refresh=True)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        super().on_validation_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )
        description = f"val_loss: {trainer.callback_metrics.get('val_loss', 0):.2f}"
        self.val_progress_bar.set_postfix_str(description, refresh=True)
