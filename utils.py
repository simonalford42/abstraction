import torch
import torch.nn as nn
import os
import time
import mlflow


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def print_torch_device():
    if torch.cuda.is_available():
        print('Training on ' + torch.cuda.get_device_name(DEVICE))
    else:
        print('Training on CPU')


def save_mlflow_model(net: nn.Module, model_name='model'):
    mlflow.pytorch.log_model(net, model_name)
    print(f"Saved model for run\n{mlflow.active_run().info.run_id}",
          f"with name {model_name}")


def load_mlflow_model(run_id: str, model_name: str = 'model') -> nn.Module:
    model_uri = f"runs:/{run_id}/{model_name}"
    model = mlflow.pytorch.load_model(model_uri)
    print(f"Loaded model from run {run_id} with name {model_name}")
    return model


class Timing(object):
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, type, value, traceback):
        dt = time.time() - self.start
        if isinstance(self.message, str):
            message = self.message
        elif callable(self.message):
            message = self.message(dt)
        else:
            raise ValueError("Timing message should be string function")
        print(f"{message} in {dt:.1f} seconds")


def assertEqual(a, b):
    assert a == b, f'a != b: a:{a}, b:{b}'


def num_params(model):
    return sum([torch.prod(torch.tensor(p.shape))
                for p in list(model.parameters())])


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
