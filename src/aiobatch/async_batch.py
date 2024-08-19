import asyncio
import functools
import time

import numpy as np
import torch

from .batcher import TensorBatcher


class AsyncTask(object):
    def __init__(self, args: list, kwargs: dict) -> None:
        self.args = args
        self.kwargs = kwargs
        self._result = None
        self._done = asyncio.Event()

        self._prepare()

    def _prepare(self):
        args_idx = []
        kwargs_idx = []
        tensor_sample_num = None
        for i, arg in enumerate(self.args):
            if isinstance(arg, torch.Tensor) or isinstance(arg, np.ndarray):
                args_idx.append(i)
                if tensor_sample_num is None:
                    tensor_sample_num = arg.shape[0]
                elif tensor_sample_num != arg.shape[0]:
                    raise ValueError(
                        "All tensor arguments should have the same batch size"
                    )

        for key, value in self.kwargs.items():
            if isinstance(value, torch.Tensor) or isinstance(value, np.ndarray):
                kwargs_idx.append(key)
                if tensor_sample_num is None:
                    tensor_sample_num = value.shape[0]
                elif tensor_sample_num != value.shape[0]:
                    raise ValueError(
                        "All tensor arguments should have the same batch size"
                    )

        self.tensor_args_idx = args_idx
        self.tensor_kwargs_idx = kwargs_idx
        self.tensor_sample_num = tensor_sample_num
        if tensor_sample_num is None:
            raise ValueError("No tensor arguments found")

    def set_result(self, result):
        self._result = result
        self._done.set()

    async def get_result(self):
        await self._done.wait()
        return self._result


class AsyncScheduler(object):
    def __init__(self, batch_size, timeout) -> None:
        self._queue = asyncio.Queue()
        self._worker = None
        self._stopped = False
        self._timeout = timeout
        self._batch_size = batch_size
        self._batcher = TensorBatcher(fixed_batch_size=batch_size)

    async def start(self, executor):
        if self._worker is not None:
            return
        self._executor = executor
        self._worker = asyncio.create_task(self._scheduler())
        self._stopped = False

    async def submit(self, task):
        await self._queue.put(task)

    async def _scheduler(self):
        last_task = None
        while not self._stopped:
            # Start a new batch
            if last_task is not None:
                task = last_task
                last_task = None
            else:
                task = await self._queue.get()
            tasks = [task]
            total_size = task.tensor_sample_num

            since = time.time()
            # Add more tasks to the batch, until the batch is full or timeout
            while total_size < self._batch_size and not self._stop_event.is_set():
                try:
                    elapsed = time.time() - since
                    wait = self._timeout - elapsed
                    if wait < 0:
                        break
                    task = await asyncio.wait_for(self._queue.get(), timeout=wait)

                    if total_size + task.tensor_sample_num <= self._batch_size:
                        tasks.append(task)
                        total_size += task.tensor_sample_num
                    else:
                        last_task = task
                        break

                except asyncio.TimeoutError:
                    break

            if self._stopped:
                break

            args, kwargs = self._batcher.batch(tasks)
            # Process the batch
            # TODO(yuheng): pass stop event to executor
            result = await asyncio.to_thread(self._executor, *args, **kwargs)
            self._batcher.unbatch(tasks, result)

            tasks.clear()

    def stop(self):
        if self._worker is None:
            return
        self._stop_event.set()
        self._worker.join()
        self._worker = None
        self._executor = None


def async_batching(func, batch_size, timeout):
    scheduler = AsyncScheduler(batch_size, timeout)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        async def async_wrapper(*args, **kwargs):
            await scheduler.start(func)

            task = AsyncTask(args, kwargs)
            await scheduler.submit(task)

            return await task.get_result()

        return async_wrapper(*args, **kwargs)

    return wrapper
