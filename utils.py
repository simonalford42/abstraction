import torch
import torch.nn as nn
from typing import List, Any


def assertEqual(a, b):
    assert a == b, f'a != b: a:{a}, b:{b}'


def num_params(model):
    return sum([torch.prod(torch.tensor(p.shape))
                for p in list(model.parameters())])


class FC(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 num_hidden=1,
                 hidden_dim=512,
                 batch_norm=False):
        super().__init__()
        layers: List[Any]
        if num_hidden == 0:
            layers = [nn.Linear(input_dim, output_dim)]
        else:
            layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            for i in range(num_hidden - 1):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
                if batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
