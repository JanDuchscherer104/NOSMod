from typing import Annotated, Literal, Tuple

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from ..config import Config, SqrtFilterParams
from .sqrt_filter import generate_sqrt_filter_train_data


class SqrtFilterDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (2, 2, #symbols)
        """
        idx = np.array(idx)
        return torch.from_numpy(self.X[idx]).to(torch.float32), torch.from_numpy(
            self.Y[idx]
        ).to(torch.float32)


class SqrtFilterDataModule(LightningDataModule):
    config: Config
    hparams: SqrtFilterParams

    df: Annotated[pd.DataFrame, None]

    test_data: Annotated[SqrtFilterDataset, None]
    train_data: Annotated[SqrtFilterDataset, None]
    val_data: Annotated[SqrtFilterDataset, None]

    def __init__(self, config: Config, hparams: SqrtFilterParams):
        super().__init__()
        self.config = config
        self.save_hyperparameters(hparams.model_dump())

        self.df = None

        self.train_data = None
        self.test_data = None
        self.val_data = None

    def prepare_data(self):
        if not (filter_file := self.config.paths.sqrt_filter_file).exists():
            generate_sqrt_filter_train_data(self.config, self.hparams)
        df: pd.DataFrame = pd.read_pickle(filter_file)

        if "split" not in df.columns:
            df = df.assign(rand=np.random.rand(len(df))).assign(
                split=lambda x: pd.cut(
                    x["rand"], bins=[0, 0.6, 0.8, 1], labels=["train", "val", "test"]
                ).drop(columns=["rand"])
            )
            df.to_pickle(filter_file)
        self.df = df

    def setup(self, stage: Literal["fit", "validate", "test", "predict"]) -> None:
        match stage:
            case "fit":
                df = self.df.query("split == 'train'")
                self.train_data = SqrtFilterDataset(
                    df["E_in"].values, df["E_out"].values
                )
            case "validate":
                df = self.df.query("split == 'val'")
                self.val_data = SqrtFilterDataset(df["E_in"].values, df["E_out"].values)
            case "test" | "predict":
                df = self.df.query("split == 'test'")
                self.test_data = SqrtFilterDataset(
                    df["E_in"].values, df["E_out"].values
                )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data,
            batch_size=self.hparams.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_data,
            batch_size=self.hparams.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_data,
            batch_size=self.hparams.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_data,
            batch_size=self.hparams.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )
