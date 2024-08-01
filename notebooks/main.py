from torchsummary import summary

from nosmod.lightning.lit_trainer_factory import ExperimentConfig
from nosmod.utils import CONSOLE

trainer, module, datamodule = ExperimentConfig(is_gpu=False).setup_target().create_all()
CONSOLE.log(module)
summary(module, (4096,), device="cpu")
