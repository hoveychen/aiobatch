from typing import List, Tuple

import numpy as np
import torch

from .task import AsyncTask, Task


class TensorBatcher(object):
    def __init__(self, fixed_batch_size=0) -> None:
        self._fixed_batch_size = fixed_batch_size

    def batch(self, tasks: List[Task | AsyncTask]) -> Tuple[list, dict]:
        # We assume that only the tensor arguments are batched, and the rest are the same
        args = tasks[0].args.copy()
        kwargs = tasks[0].kwargs.copy()

        tensor_args_idx = tasks[0].tensor_args_idx
        tensor_kwargs_idx = tasks[0].tensor_kwargs_idx

        for i in tensor_args_idx:
            if isinstance(tasks[0].args[i], torch.Tensor):
                args[i] = torch.cat([task.args[i] for task in tasks], dim=0)
                if len(args[i]) < self._fixed_batch_size:
                    padding = torch.zeros(
                        (self._fixed_batch_size - len(args[i]), *args[i].shape[1:]),
                        dtype=args[i].dtype,
                        device=args[i].device,
                    )
                    args[i] = torch.cat([args[i], padding], dim=0)
            elif isinstance(tasks[0].args[i], np.ndarray):
                args[i] = np.concatenate([task.args[i] for task in tasks], axis=0)
                if len(args[i]) < self._fixed_batch_size:
                    padding = np.zeros(
                        (self._fixed_batch_size - len(args[i]), *args[i].shape[1:]),
                        dtype=args[i].dtype,
                    )
                    args[i] = np.concatenate([args[i], padding], axis=0)

        for key in tensor_kwargs_idx:
            if isinstance(tasks[0].kwargs[key], torch.Tensor):
                kwargs[key] = torch.cat([task.kwargs[key] for task in tasks], dim=0)
                if len(kwargs[key]) < self._fixed_batch_size:
                    padding = torch.zeros(
                        (
                            self._fixed_batch_size - len(kwargs[key]),
                            *kwargs[key].shape[1:],
                        ),
                        dtype=kwargs[key].dtype,
                        device=kwargs[key].device,
                    )
                    kwargs[key] = torch.cat([kwargs[key], padding], dim=0)
            elif isinstance(tasks[0].kwargs[key], np.ndarray):
                kwargs[key] = np.concatenate(
                    [task.kwargs[key] for task in tasks], axis=0
                )
                if len(kwargs[key]) < self._fixed_batch_size:
                    padding = np.zeros(
                        (
                            self._fixed_batch_size - len(kwargs[key]),
                            *kwargs[key].shape[1:],
                        ),
                        dtype=kwargs[key].dtype,
                    )
                    kwargs[key] = np.concatenate([kwargs[key], padding], axis=0)

        return (args, kwargs)

    def unbatch(self, result, tasks: List[Task | AsyncTask]):
        expected_size = (
            sum(task.sample_size for task in tasks)
            if self._fixed_batch_size == 0
            else self._fixed_batch_size
        )

        if (
            isinstance(result, (torch.Tensor, np.ndarray, list))
            and len(result) == expected_size
        ):
            last_idx = 0
            for i, task in enumerate(tasks):
                new_result = result[last_idx : last_idx + task.sample_size]
                last_idx += task.sample_size
                task.set_result(new_result)
        elif isinstance(result, tuple):
            last_idx = 0
            for i, task in enumerate(tasks):
                new_result = list(result)
                for key, value in enumerate(result):
                    if (
                        isinstance(value, (torch.Tensor, np.ndarray, list))
                        and len(value) == expected_size
                    ):
                        new_result[key] = value[last_idx : last_idx + task.sample_size]

                last_idx += task.sample_size
                task.set_result(tuple(new_result))

        elif isinstance(result, dict):
            last_idx = 0
            for i, task in enumerate(tasks):
                new_result = result.copy()
                for key, value in result.items():
                    if (
                        isinstance(value, (torch.Tensor, np.ndarray))
                        and len(value) == expected_size
                    ):
                        new_result[key] = value[last_idx : last_idx + task.sample_size]

                last_idx += task.sample_size
                task.set_result(new_result)

        else:
            # This is a scalar result
            for task in tasks:
                task.set_result(result)


class ListBatcher(object):
    def batch(self, tasks: List[Task | AsyncTask]) -> Tuple[List[list], List[dict]]:
        args = [[]] * len(tasks[0].args)
        kwargs = {key: [] for key in tasks[0].kwargs.keys()}

        for task in tasks:
            for i, arg in enumerate(task.args):
                args[i].append(arg)
            for key, value in task.kwargs.items():
                kwargs[key].append(value)

        return (args, kwargs)

    def unbatch(self, result, tasks: List[Task | AsyncTask]):
        num = len(tasks)

        if isinstance(result, (torch.Tensor, np.ndarray, list)) and len(result) == num:
            for i, task in enumerate(tasks):
                task.set_result(result[i])
        elif isinstance(result, tuple):
            for i, task in enumerate(tasks):
                new_result = list(result)
                for key, value in enumerate(result):
                    if (
                        isinstance(value, (torch.Tensor, np.ndarray, list))
                        and len(value) == num
                    ):
                        new_result[key] = value[i]

                task.set_result(tuple(new_result))

        elif isinstance(result, dict):
            for i, task in enumerate(tasks):
                new_result = result.copy()
                for key, value in result.items():
                    if (
                        isinstance(value, (torch.Tensor, np.ndarray, list))
                        and len(value) == num
                    ):
                        new_result[key] = value[i]

                task.set_result(new_result)
        else:
            # This is a scalar result
            for task in tasks:
                task.set_result(result)
