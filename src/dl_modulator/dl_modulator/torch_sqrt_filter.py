import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

from config import Config, SqrtFilterParams
from sqrt_RNN_filter import generate_sqrt_filter_train_data


class SqrtFilterDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]).to(torch.float32), torch.from_numpy(
            self.Y[idx]
        ).to(torch.float32)


class SqrtFilterDataModule(LightningDataModule):
    test_data: SqrtFilterDataset
    train_data: SqrtFilterDataset
    val_data: SqrtFilterDataset

    def __init__(self, config: Config, batch_size: int):
        super().__init__()
        self.config = config
        self.batch_size = batch_size
        self.train_data = None  # type: ignore
        self.test_data = None  # type: ignore
        self.val_data = None  # type: ignore

    def prepare_data(self):
        df = pd.read_pickle(
            self.config.paths.data / "sqrt_filter" / "sqrt_filter_12.pickle"
        )
        X = np.array(df["E_in"].tolist())
        Y = np.array(df["E_out"].tolist())

        # Split the data into training and temporary sets (60:40)
        X_train, X_temp, Y_train, Y_temp = train_test_split(
            X, Y, test_size=0.4, random_state=42
        )

        # Split the temporary set into validation and test sets (50:50)
        X_val, X_test, Y_val, Y_test = train_test_split(
            X_temp, Y_temp, test_size=0.5, random_state=42
        )

        self.train_data = SqrtFilterDataset(X_train, Y_train)
        self.val_data = SqrtFilterDataset(X_val, Y_val)
        self.test_data = SqrtFilterDataset(X_test, Y_test)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.config.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.config.num_workers,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.config.num_workers,
        )


class LinearModel(LightningModule):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        with torch.no_grad():
            y_hat = self(x)
        return {"y": y.detach().cpu().numpy(), "y_hat": y_hat.detach().cpu().numpy()}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters())


def main():
    config = Config(sqrt_filter_params=SqrtFilterParams(num_trials=30000))
    generate_sqrt_filter_train_data(config)
    print("Data generated")
    data_module = SqrtFilterDataModule(
        config, batch_size=config.sqrt_filter_params.batch_size
    )
    data_module.prepare_data()

    model = LinearModel(
        config.sqrt_filter_params.nos * config.sqrt_filter_params.sps,
        config.sqrt_filter_params.nos * config.sqrt_filter_params.sps,
    )

    trainer = Trainer(max_epochs=config.sqrt_filter_params.num_epochs)
    trainer.fit(model, datamodule=data_module)

    outputs = trainer.predict(model, datamodule=data_module)

    # Extract the true and predicted y values
    Y = np.concatenate([x["y"] for x in outputs])
    Y2 = np.concatenate([x["y_hat"] for x in outputs])

    # Plot the true vs predicted y values
    plt.figure(1)
    plt.plot(Y2, Y, "g.")
    plt.title("Validation Set")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")

    n = 0
    plt.figure(2)
    plt.plot(Y[n], "k")
    plt.plot(Y2[n], "r:")
    plt.show()

    torch.save(model.state_dict(), "model_B20_sqrt_singlefeature.pt")


def test():

    config = Config(sqrt_filter_params=SqrtFilterParams(num_trials=1000))
    trainer = Trainer(max_epochs=config.sqrt_filter_params.num_epochs)
    # Initialize the model
    model = LinearModel(
        config.sqrt_filter_params.nos * config.sqrt_filter_params.sps,
        config.sqrt_filter_params.nos * config.sqrt_filter_params.sps,
    )
    data_module = SqrtFilterDataModule(
        config, batch_size=config.sqrt_filter_params.batch_size
    )

    trainer.fit(model, datamodule=data_module)
    torch.save(model.state_dict(), "model_B20_sqrt_singlefeature.pt")
    outputs = trainer.predict(model, datamodule=data_module)

    # Extract the true and predicted y values
    Y = torch.cat([x["y"] for x in outputs]).numpy()
    Y2 = torch.cat([x["y_hat"] for x in outputs]).numpy()

    # Plot the true vs predicted y values
    plt.figure(1)
    plt.plot(Y2, Y, "g.")
    plt.title("Validation Set")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")

    n = 0
    plt.figure(2)
    plt.plot(Y[n], "k")
    plt.plot(Y2[n], "r:")
    plt.show()


if __name__ == "__main__":
    main()
    # test()
