from typing import Any, Callable, Dict, List
from warnings import warn

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import (
    BatchSizeFinder,
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.types import STEP_OUTPUT

from .config import Config, HyperParameters


class LitTrainerFactory:
    @classmethod
    def create_trainer(
        cls,
        config: Config,
        hparams: HyperParameters,
        **trainer_kwargs,
    ) -> Trainer:
        """Create and initialize Callback instances."""
        factory_instance = cls(config, hparams)

        callbacks = factory_instance._assemble_callbacks()
        tb_logger = factory_instance._assemble_loggers()

        if config.is_debug:
            config.is_gpu = False
            config.is_fast_dev_run = True
            config.num_workers = 0
            config.is_multiproc = False
            config.verbose = True

            config.active_callbacks["ModelCheckpoint"] = False

        # Create Trainer
        return Trainer(
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

    def _get_callback_map(self) -> Dict[str, Callable]:
        return {
            "ModelCheckpoint": self._create_model_checkpoint,
            "TQDMProgressBar": self._create_tqdm_progress_bar,
            "EarlyStopping": self._create_early_stopping,
            "BatchSizeFinder": self._create_batch_size_finder,
            "LearningRateMonitor": self._create_lr_monitor,
            "ModelSummary": self._create_model_summary,
        }

    def __init__(
        self,
        config: Config,
        hparams: HyperParameters,
    ):
        """Private constructor to set config and hyper_params."""
        self.config = config
        self.hparams = hparams

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

    def _create_model_summary(self):
        return ModelSummary(max_depth=4)

    def _create_batch_size_finder(self):
        return BatchSizeFinder(
            mode="power",
            steps_per_trial=3,
            init_val=self.hparams.batch_size,
            max_trials=25,
            batch_arg_name="batch_size",
        )

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
                        warn(f"Callback {key} could not be created.")
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
        description = f"train_loss: {trainer.callback_metrics.get('train_loss', float('inf')):.2f}"
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
        description = (
            f"val_loss: {trainer.callback_metrics.get('val_loss', float('inf')):.2f}"
        )
        self.val_progress_bar.set_postfix_str(description, refresh=True)
