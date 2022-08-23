import torch
import os
import time
import numpy as np
import uuid
import psutil
from contextlib import nullcontext
from collections import namedtuple


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

WARNINGS = set()


def gpu_check():
    print_torch_device()
    print(torch.arange(3).to(DEVICE))
    print('passed gpu check')


def warn(s):
    if s not in WARNINGS:
        print('WARNING:', s)
    WARNINGS.add(s)


def hash_tensor(t):
    return (t * torch.arange(torch.numel(t)).reshape(t.shape)**2).sum() % 1000


class CustomDictOne(dict):
    def __init__(self,*arg,**kw):
        super(CustomDictOne, self).__init__(*arg, **kw)


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


def save_model(model, path, overwrite=False):
    if not overwrite:
        path = next_unused_path(path)
    torch.save(model, path)
    print('Saved model at ' + path)
    return path


def load_model(path):
    model = torch.load(path, map_location=DEVICE)
    print('Loaded model from ' + path)
    return model


def generate_uuid():
    return uuid.uuid4().hex


def next_unused_path(path, extend_fn=lambda i: f'__({i})'):
    last_dot = path.rindex('.')
    extension = path[last_dot:]
    file_name = path[:last_dot]

    i = 0
    while os.path.isfile(path):
        path = file_name + extend_fn(i) + extension
        i += 1

    return path


def logaddexp(tensor, other, mask=[1, 1]):
    if type(mask) in [list, tuple]:
        mask = torch.tensor(mask)

    assert mask.shape == (2, ), 'Invalid mask shape'

    a = torch.max(tensor, other)
    # if max is -inf, set a to zero, to avoid making nan's
    a = torch.where(a == float('-inf'), torch.zeros(a.shape), a)

    return a + ((tensor - a).exp()*mask[0] + (other - a).exp()*mask[1]).log()


def log1minus(x):
    """
    Returns log(1 - x.exp())
    This is the logged version of finding 1 - p
    """
    return torch.log1p(-x.exp())


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


class NoMlflowRun():
    def start_run(self):
        return nullcontext()

    def log_params(self, *args, **kwargs):
        pass

    def log_metrics(self, *args, **kwargs):
        pass

    def active_run(self):
        Out = namedtuple('Out', 'info')
        Out2 = namedtuple('Out2', 'run_id')
        out = Out(info=Out2(run_id=generate_uuid()))
        return out

    def set_experiment(self, *args, **kwargs):
        pass


if __name__ == '__main__':
    c = torch.tensor(float('-inf'))
    print(logaddexp(c, c))
