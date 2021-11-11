import torch
import torch.nn as nn
from typing import List, Any
import os


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
            # print(f"layers: {layers}")

            # layers = [Print(l) for l in layers]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Print(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        print(x)
        for p in self.parameters():
            print(p)
        return self.layer(x)


def save_model(model, path):
    path2 = next_unused_path(path)
    torch.save(model.state_dict(), path2)
    print('Saved model at ' + path2)


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    print('Loaded model from ' + path)


def next_unused_path(path):
    last_dot = path.rindex('.')
    extension = path[last_dot:]
    file_name = path[:last_dot]

    i = 0
    while os.path.isfile(path):
        path = file_name + f"__{i}" + extension
        i += 1

    return path
