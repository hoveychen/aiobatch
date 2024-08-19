import numpy as np
import torch


class TensorBatcher(object):
    def __init__(self, fixed_batch_size=0) -> None:
        self._fixed_batch_size = fixed_batch_size

    def batch(self, tasks: list):
        args = tasks[0].args.copy()
        kwargs = tasks[0].kwargs.copy()

        args_idx = tasks[0].tensor_args_idx
        kwargs_idx = tasks[0].tensor_kwargs_idx

        for i in args_idx:
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

        for key in kwargs_idx:
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

    def unbatch(self, tasks: list, result):
        if isinstance(result, (tuple, list)):
            last_idx = 0
            for i, task in enumerate(tasks):
                new_result = result.copy()
                for key, value in enumerate(result):
                    if isinstance(value, (torch.Tensor, np.ndarray)):
                        new_result[key] = value[
                            last_idx : last_idx + task.tensor_sample_num
                        ]

                last_idx += task.tensor_sample_num
                task.set_result(new_result)

        elif isinstance(result, dict):
            last_idx = 0
            for i, task in enumerate(tasks):
                new_result = result.copy()
                for key, value in result.items():
                    if isinstance(value, (torch.Tensor, np.ndarray)):
                        new_result[key] = value[
                            last_idx : last_idx + task.tensor_sample_num
                        ]

                last_idx += task.tensor_sample_num
                task.set_result(new_result)

        elif isinstance(result, (torch.Tensor, np.ndarray)):
            last_idx = 0
            for i, task in enumerate(tasks):
                new_result = result[last_idx : last_idx + task.tensor_sample_num]
                last_idx += task.tensor_sample_num
                task.set_result(new_result)
        else:
            raise ValueError("Unsupported return type")
