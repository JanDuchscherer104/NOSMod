from typing import Callable, Optional, Type

import pytorch_lightning as pl
from pydantic import Field
from torch.utils.data import DataLoader

from ..network_components.data_generator import (
    NosmodDataGenerator,
    NosmodDataGeneratorParams,
)
from ..utils import BaseConfig, Stage


class DatamoduleParams(BaseConfig["LitNosmodDatamodule"]):
    target: Type["LitNosmodDatamodule"] = Field(
        default_factory=lambda: LitNosmodDatamodule
    )

    batch_size: int = 512
    num_workers: int = -1
    pin_memory: bool = True

    dataset: dict[Stage, NosmodDataGeneratorParams] = {
        split: NosmodDataGeneratorParams(
            split=split,
        )
        for split in Stage
    }
    transforms: Callable = lambda x: x


class LitNosmodDatamodule(pl.LightningDataModule):
    params: DatamoduleParams

    nosmod_train: NosmodDataGenerator
    nosmod_val: NosmodDataGenerator
    nosmod_test: NosmodDataGenerator

    def __init__(self, params: DatamoduleParams):
        super().__init__()
        self.params = params

    def setup(self, stage: Optional[str] = None):
        split = Stage.from_str(stage)
        match split:
            case Stage.TRAIN | Stage.VAL | Stage.TEST:
                self.nosmod_train = self.params.dataset[split].setup_target()
            case _:
                raise ValueError(f"Unknown stage: {stage}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.nosmod_train,
            batch_size=self.params.batch_size,
            pin_memory=self.params.pin_memory,
            num_workers=self.params.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.nosmod_val,
            batch_size=self.params.batch_size,
            pin_memory=self.params.pin_memory,
            num_workers=self.params.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.nosmod_test,
            batch_size=self.params.batch_size,
            pin_memory=self.params.pin_memory,
            num_workers=self.params.num_workers,
        )
