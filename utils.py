import torch
import torch.nn as nn
import os
import time
import mlflow
from typing import Tuple
import numpy as np
import uuid
import psutil


POS = Tuple[int, int]

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print('using cpu!!')
# DEVICE = torch.device("cpu")


def log(s: str):
    with open('log.txt', 'r+') as f:
        f.write(s)


def get_memory_usage(prnt=False):
    # https://stackoverflow.com/a/21632554/4383594
    mem = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    if prnt:
        print(f'Using {mem} MB')
    return mem


def print_torch_device():
    if torch.cuda.is_available():
        print('Using torch device ' + torch.cuda.get_device_name(DEVICE))
    else:
        print('Using torch device CPU')


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


def assert_equal(a, b):
    if np.ndarray in [type(a), type(b)]:
        assert np.array_equal(a, b), f'a != b: a:{a}, b:{b}'
    else:
        assert a == b, f'a != b: a:{a}, b:{b}'


def assert_shape(a: torch.Tensor, shape: tuple):
    assert_equal(a.shape, shape)


def num_params(model):
    return sum([torch.prod(torch.tensor(p.shape))
                for p in list(model.parameters())])


def save_model(model, path, assert_clear=False):
    path2 = next_unused_path(path)
    torch.save(model, path2)
    print('Saved model at ' + path2)
    return path2


def load_model(path):
    model = torch.load(path)
    print('Loaded model from ' + path)
    return model


def generate_uuid():
    return uuid.uuid4().hex


def next_unused_path(path, extend_fn=None):
    if not extend_fn:
        extend_fn = lambda i: f'__({i})'

    last_dot = path.rindex('.')
    extension = path[last_dot:]
    file_name = path[:last_dot]

    i = 0
    while os.path.isfile(path):
        path = file_name + extend_fn(i) + extension
        i += 1

    return path


def logaddexp(tensor, other, mask=None):
    if mask is None:
        mask = torch.tensor([1, 1])
    else:
        assert mask.shape == (2, ), 'Invalid mask shape'

    a = torch.max(tensor, other)
    return a + ((tensor - a).exp()*mask[0] + (other - a).exp()*mask[1]).log()


class NoLogRun():
    def __setitem__(self, key, item):
        pass

    def __getitem__(self, key):
        class NoLogRunInner():
            def log(self, *args, **kwargs):
                pass
            def upload(self, *args, **kwards):
                pass

        return NoLogRunInner()

    def stop(self):
        pass
