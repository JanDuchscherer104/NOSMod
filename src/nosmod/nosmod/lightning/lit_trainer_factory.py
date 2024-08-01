import re
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Callable, Dict, List, Literal, Optional, Tuple, Type
from warnings import warn

import mlflow
import psutil
import torch
from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator
from pytorch_lightning import Callback, LightningModule, Trainer
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
from typing_extensions import Self

from ..network_components.data_generator import AlphabetType, NosmodDataGeneratorParams
from ..network_components.mod_trans_demod_system import NosmodSystemParams
from ..network_components.raised_cos_filter import RaisedCosParams
from ..utils import CONSOLE, BaseConfig, Stage
from .lit_datamodule import DatamoduleParams, LitNosmodDatamodule
from .lit_module import HParams, LitNOSModModule

ROOT = Path(__file__).parents[3].resolve()


class Paths(BaseConfig):
    target: Type["Paths"] = Field(default_factory=lambda: Paths)
    root: Path = Field(default=ROOT)
    data: Annotated[Path, Field(default=".data", validate_default=True)]
    checkpoints: Annotated[
        Path, Field(default=".logs/checkpoints", validate_default=True)
    ]
    tb_logs: Annotated[Path, Field(default=".logs/tb_logs", validate_default=True)]
    configs: Annotated[Path, Field(default=".configs", validate_default=True)]
    mlflow_uri: Annotated[
        str, Field(default=".logs/mlflow_logs/mlflow", validate_default=True)
    ]
    sqrt_filter_file_name: str = Field(default="sqrt_filter", validate_default=True)

    @field_validator("data", "checkpoints", "tb_logs", "configs")
    @classmethod
    def __convert_to_path(cls, v: str, info: ValidationInfo) -> Path:
        root: Path = info.data.get("root")
        path = (root / v).resolve() if not Path(v).is_absolute() else Path(v)
        path.mkdir(parents=True, exist_ok=True)
        if info.field_name == "data":
            (path / "sqrt_filter").mkdir(parents=True, exist_ok=True)
        return path

    @field_validator("mlflow_uri")
    @classmethod
    def __convert_to_uri(cls, v: str, info: ValidationInfo) -> str:
        if v.startswith("file://"):
            return v
        root: Path = info.data.get("root")
        uri_path = root / v if not Path(v).is_absolute() else Path(v)
        uri_path.parent.mkdir(parents=True, exist_ok=True)
        if not uri_path.exists():
            uri_path.mkdir(parents=True, exist_ok=True)
        return uri_path.resolve().as_uri()  # type: ignore

    @field_validator("sqrt_filter_file_name", mode="after")
    @classmethod
    def __update_label(cls, v: str, info: ValidationInfo) -> str:
        if v.endswith(".pickle"):
            return v
        data_dir: Path = info.data.get("data") / "sqrt_filter"

        # Extract the run numbers from the file names
        run_numbers = [
            int(re.search(rf"{v}_(\d+).pickle", str(file)).group(1))  # type: ignore
            for file in data_dir.glob(f"{v}_*.pickle")
            if re.search(rf"{v}_(\d+).pickle", str(file)) is not None
        ]

        next_run_number = max(run_numbers, default=0) + 1

        return f"{v}_{next_run_number}.pickle"

    @property
    def sqrt_filter_file(self) -> Path:
        return self.data / "sqrt_filter" / self.sqrt_filter_file_name


class MLflowConfig(BaseConfig):
    target: Type["MLflowConfig"] = Field(default_factory=lambda: MLflowConfig)
    experiment_name: str = "nosmod"
    run_name: Annotated[str, Field(default="None")]
    experiment_id: Annotated[str, Field(default="None")]


class ExperimentConfig(BaseConfig["TrainerFactory"]):
    """
    TODO: add MachineConfig
    TODO: add TrainerConfig
    """

    target: Type["TrainerFactory"] = Field(default_factory=lambda: TrainerFactory)
    is_debug: bool = False
    verbose: bool = True
    from_ckpt: Optional[str] = None
    is_multiproc: bool = True
    num_workers: Optional[int] = None
    is_optuna: bool = False
    max_epochs: int = 50
    early_stopping_patience: int = 2
    log_every_n_steps: int = 8
    is_gpu: bool = True
    matmul_precision: Literal["medium", "high"] = "medium"
    is_fast_dev_run: bool = False
    active_callbacks: Dict[
        Literal[
            "ModelCheckpoint",
            "TQDMProgressBar",
            "EarlyStopping",
            "BatchSizeFinder",
            "LearningRateMonitor",
            "ModelSummary",
        ],
        bool,
    ] = {
        "ModelCheckpoint": True,
        "TQDMProgressBar": True,
        "EarlyStopping": True,
        "BatchSizeFinder": False,
        "LearningRateMonitor": False,
        "ModelSummary": True,
    }
    paths: Paths = Field(default_factory=Paths)
    mlflow_config: MLflowConfig = Field(default_factory=MLflowConfig)
    datamodule: DatamoduleParams = Field(default_factory=DatamoduleParams)
    module: HParams = Field(
        default_factory=lambda: HParams(
            nosmod_system=NosmodSystemParams(
                raised_cos_params=RaisedCosParams(
                    rolloff_fact=0.5,
                    sampling_freq=1e4,
                    center_freq=1e3,
                ),
            ),
        )
    )
    module_type: Type["LitNOSModModule"] = Field(default=LitNOSModModule)
    datamodule_type: Type["LitNosmodDatamodule"] = Field(default=LitNosmodDatamodule)

    def dump_yaml(self) -> None:
        self.to_yaml(self.paths.configs)

    @model_validator(mode="after")
    def __setup_mlflow(self) -> "ExperimentConfig":
        mlflow.set_tracking_uri(self.paths.mlflow_uri)
        experiment = mlflow.get_experiment_by_name(self.mlflow_config.experiment_name)
        experiment_id = (
            experiment.experiment_id
            if experiment is not None
            else mlflow.create_experiment(self.mlflow_config.experiment_name)
        )
        self.mlflow_config.experiment_id = experiment_id
        last_run = mlflow.search_runs(
            order_by=["start_time DESC"], max_results=1, experiment_ids=[experiment_id]
        )

        if last_run.empty:
            next_run_num = 1
        else:
            last_run_label = last_run.iloc[0]["tags.mlflow.runName"]
            last_run_num = int(last_run_label.split("-")[0][1:])
            next_run_num = last_run_num + 1

        self.mlflow_config.run_name = (
            f"R{next_run_num:03d}-{datetime.now().strftime('%b%d-%H:%M')}"
        )

        if self.num_workers is None and self.is_multiproc:
            self.num_workers = psutil.cpu_count(logical=True)

        return self

    # def dump(self):
    #     i = 1
    #     config_file = self.paths.configs / f"{self.mlflow_config.run_name}.yaml"
    #     while config_file.exists():
    #         config_file = self.paths.configs / f"{self.mlflow_config.run_name}-{i}.yaml"
    #         i += 1
    #     self.to_yaml(config_file)

    # @classmethod
    # def read(cls, file_name: str) -> Self:
    #     config_file = (ROOT / ".configs" / file_name).with_suffix(".yaml")
    #     assert config_file.exists(), f"{config_file} does not exist"
    #     return cls.from_yaml(config_file)  # type: ignore


class TrainerFactory:
    def __init__(self, config: ExperimentConfig):
        """Private constructor to set config and hyper_params."""
        self.config = config

    @classmethod
    def setup_target(
        cls,
        config: ExperimentConfig,
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
