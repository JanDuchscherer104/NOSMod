import numpy as np
import torch
from matplotlib import pyplot as plt
from pytorch_lightning import LightningModule, Trainer
from torch import nn

from ..config import ExperimentConfig, SqrtFilterParams
from .filter_dataset import SqrtFilterDataModule
from .sqrt_filter import generate_sqrt_filter_train_data


class LitSqrtFilterModule(LightningModule):
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
    config = ExperimentConfig()
    generate_sqrt_filter_train_data(config)
    print("Data generated")
    data_module = SqrtFilterDataModule(
        config, batch_size=config.sqrt_filter_params.batch_size
    )
    data_module.prepare_data()

    model = LitSqrtFilterModule(
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

    config = ExperimentConfig(sqrt_filter_params=SqrtFilterParams(num_samples=1000))
    trainer = Trainer(max_epochs=config.sqrt_filter_params.num_epochs)
    # Initialize the model
    model = LitSqrtFilterModule(
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
