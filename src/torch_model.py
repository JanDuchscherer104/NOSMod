import torch
from torch import nn
from torch.nn import functional as F


class RingPaddingLayer(nn.Module):
    def __init__(self, wow):
        super(RingPaddingLayer, self).__init__()
        self.wow = wow

    def forward(self, inputs):
        padding = inputs[:, -self.wow + 1 :, :]
        inputs_with_ring_padding = torch.cat([padding, inputs], dim=1)
        output_list = []
        for i in range(inputs.shape[1]):
            window = inputs_with_ring_padding[:, i : i + self.wow, :]
            output_list.append(window)
        output = torch.stack(output_list, dim=1)
        flattened_tensor = output.view(-1, inputs.shape[1], self.wow * inputs.shape[-1])
        return flattened_tensor


class CustomReshapeLayer(nn.Module):
    def __init__(self):
        super(CustomReshapeLayer, self).__init__()

    def forward(self, inputs):
        skipped = inputs[:, 64::128, :]
        shape = skipped.shape
        new_shape = (shape[0], shape[1] // 2, shape[2] * 2)
        reshaped = skipped.view(new_shape)
        return reshaped


class Model(nn.Module):
    def __init__(self, nos: int, sps: int):
        super(Model, self).__init__()
        self.ring_padding_layer = RingPaddingLayer(121)
        self.dense_layer_0 = nn.Linear(121 * 4, sps // 4)
        self.dense_layer_1 = nn.Linear(sps // 4, 4 * sps)
        self.reshape_layer = nn.Linear(nos * sps, 4)
        self.shared_dense_0 = nn.Linear(4 * sps, nos * sps)
        self.ring_padding_layer_2 = RingPaddingLayer(5)
        self.dense_layer_2 = nn.Linear(5 * 4, 8 * 4)
        self.dense_layer_3 = nn.Linear(8 * 4, 4)

        self.activ_1 = nn.Tanh()  # Why Tanh?
        self.activ_2 = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ring_padding_layer(x)
        x = self.dense_layer_0(x)
        x = self.activ_1(x)
        x = self.dense_layer_1(x)
        x = self.reshape_layer(x)
        x = self.shared_dense_0(x)
        x = self.ring_padding_layer_2(x)
        x = self.dense_layer_2(x)
        x = self.activ_2(x)
        x = self.dense_layer_3(x)
        return x


model = Model(nos=128, sps=32)
