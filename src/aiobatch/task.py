import asyncio
from concurrent.futures import Future
from typing import Union

import numpy as np
import torch


class AsyncTask(asyncio.Future):
    def __init__(self, args: list, kwargs: dict, is_tensor=False) -> None:
        super().__init__()
        self.args = args
        self.kwargs = kwargs
        self._result = None

        if is_tensor:
            preprocess_tensor_args(self)
        else:
            self.tensor_args_idx = []
            self.tensor_kwargs_idx = []
            self.sample_size = 1


class Task(Future):
    def __init__(self, args: list, kwargs: dict, is_tensor=False) -> None:
        super().__init__()
        self.args = args
        self.kwargs = kwargs
        self._result = None

        if is_tensor:
            preprocess_tensor_args(self)
        else:
            self.tensor_args_idx = []
            self.tensor_kwargs_idx = []
            self.sample_size = 1


def preprocess_tensor_args(task: Union[AsyncTask, Task]):
    args_idx = []
    kwargs_idx = []
    sample_size = None
    for i, arg in enumerate(task.args):
        if isinstance(arg, torch.Tensor) or isinstance(arg, np.ndarray):
            args_idx.append(i)
            if sample_size is None:
                sample_size = arg.shape[0]
            elif sample_size != arg.shape[0]:
                raise ValueError("All tensor arguments should have the same batch size")

    for key, value in task.kwargs.items():
        if isinstance(value, torch.Tensor) or isinstance(value, np.ndarray):
            kwargs_idx.append(key)
            if sample_size is None:
                sample_size = value.shape[0]
            elif sample_size != value.shape[0]:
                raise ValueError("All tensor arguments should have the same batch size")

    task.tensor_args_idx = args_idx
    task.tensor_kwargs_idx = kwargs_idx
    task.sample_size = sample_size
    if sample_size is None:
        raise ValueError("No tensor arguments found")
