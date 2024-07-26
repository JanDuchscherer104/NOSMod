import re
from datetime import datetime
from pathlib import Path
from typing import Annotated, Dict, Literal, Optional

import mlflow
import psutil
from pydantic import Field, ValidationInfo, field_validator, model_validator

from .utils import YamlBaseModel

ROOT = Path(__file__).parents[3].resolve()


class Paths(YamlBaseModel):
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


class SqrtFilterParams(YamlBaseModel):
    nos: int = 128 * 1
    sps: int = 32
    symbolrate: float = 40e9
    num_samples: int = 30000
    n_features: int = 4
    samplen: bool = False
    batch_size: int = 128
    num_epochs: int = 100


class MLflowConfig(YamlBaseModel):
    experiment_name: str = "DL-EXP"
    run_name: Annotated[str, Field(default=None)]
    experiment_id: Annotated[str, Field(default=None)]


class Config(YamlBaseModel):
    is_debug: bool = False
    verbose: bool = True
    from_ckpt: Optional[str] = None
    is_multiproc: bool = True
    num_workers: Optional[int] = None
    is_optuna: bool = False
    is_lr_scheduler: bool = False
    pin_memory: bool = True
    max_epochs: int = 50
    early_stopping_patience: int = 2
    log_every_n_steps: int = 8
    is_gpu: bool = True
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

    def dump_yaml(self) -> None:
        self.to_yaml(self.paths.configs)

    @model_validator(mode="after")
    def __setup_mlflow(self) -> "Config":
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

    def dump(self):
        i = 1
        config_file = self.paths.configs / f"{self.mlflow_config.run_name}.yaml"
        while config_file.exists():
            config_file = self.paths.configs / f"{self.mlflow_config.run_name}-{i}.yaml"
            i += 1
        self.to_yaml(config_file)

    @classmethod
    def read(cls, file_name: str) -> "Config":
        config_file = (ROOT / ".configs" / file_name).with_suffix(".yaml")
        assert config_file.exists(), f"{config_file} does not exist"
        return cls.from_yaml(config_file)  # type: ignore


class HyperParameters(YamlBaseModel):
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_epochs: int = 100


if __name__ == "__main__":
    config = Config.read("R001-Apr22")
    # config.dump()
    pass