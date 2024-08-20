import asyncio
import functools
import queue
import threading
import time
from concurrent.futures import Future
from typing import List, Tuple, Union

from .batcher import ListBatcher, TensorBatcher
from .task import AsyncTask, Task


class AsyncBatchScheduler(object):
    def __init__(self, batch_size: int, timeout: float, is_tensor=False) -> None:
        self._queue = asyncio.Queue()
        self._worker = None
        self._stopped = False
        self._timeout = timeout
        self._batch_size = batch_size
        if is_tensor:
            self._batcher = TensorBatcher(fixed_batch_size=batch_size)
        else:
            self._batcher = ListBatcher()
        self._is_tensor = is_tensor

    async def start(self, executor):
        if self._worker is not None:
            return
        self._executor = executor
        self._worker = asyncio.create_task(self._scheduler())
        self._stopped = False

    async def submit(self, task: AsyncTask):
        await self._queue.put(task)

    async def _scheduler(self):
        last_task: AsyncTask = None
        while not self._stopped:
            # Start a new batch
            if last_task is not None:
                task = last_task
                last_task = None
            else:
                task: AsyncTask = await self._queue.get()
            tasks = [task]
            total_size = task.sample_size

            since = time.time()
            # Add more tasks to the batch, until the batch is full or timeout
            while total_size < self._batch_size and not self._stopped:
                try:
                    if self._timeout == 0:
                        task = await self._queue.get_nowait()
                    else:
                        elapsed = time.time() - since
                        wait = self._timeout - elapsed
                        if wait < 0:
                            break
                        task = await asyncio.wait_for(self._queue.get(), timeout=wait)

                    if total_size + task.sample_size <= self._batch_size:
                        tasks.append(task)
                        total_size += task.sample_size
                    else:
                        last_task = task
                        break

                except asyncio.TimeoutError:
                    break
                except asyncio.QueueEmpty:
                    break

            if self._stopped:
                break

            args, kwargs = self._batcher.batch(tasks)
            # Process the batch
            # TODO(yuheng): pass stop event to executor
            result = await asyncio.to_thread(self._executor, *args, **kwargs)
            self._batcher.unbatch(result, tasks)

            tasks.clear()

    def stop(self):
        if self._worker is None:
            return
        self._stopped = True
        self._worker.cancel()
        self._worker = None
        self._executor = None

    def __call__(self, func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            await self.start(func)

            task = AsyncTask(args, kwargs, is_tensor=self._is_tensor)
            await self.submit(task)

            return await task

        return wrapper


class AsyncSingleScheduler(object):
    def __init__(self) -> None:
        self._queue = asyncio.Queue()
        self._worker = None
        self._stopped = False

    async def start(self, executor):
        if self._worker is not None:
            return
        self._executor = executor
        self._worker = asyncio.create_task(self._scheduler())
        self._stopped = False

    async def submit(self, task: AsyncTask):
        await self._queue.put(task)

    async def _scheduler(self):
        while not self._stopped:
            task: AsyncTask = await self._queue.get()

            if self._stopped:
                break

            # TODO(yuheng): pass stop event to executor
            result = await asyncio.to_thread(self._executor, *task.args, **task.kwargs)
            task.set_result(result)

    def stop(self):
        if self._worker is None:
            return
        self._stopped = True
        self._worker.cancel()
        self._worker = None
        self._executor = None

    def __call__(self, func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            await self.start(func)

            task = AsyncTask(args, kwargs)
            await self.submit(task)

            return await task

        return wrapper


class BatchScheduler(object):
    def __init__(self, batch_size: int, timeout: float, is_tensor=False) -> None:
        self._queue = queue.Queue()
        self._worker_thread = None
        self._stop_event = threading.Event()
        self._timeout = timeout
        self._batch_size = batch_size
        if is_tensor:
            self._batcher = TensorBatcher(fixed_batch_size=batch_size)
        else:
            self._batcher = ListBatcher()
        self._is_tensor = is_tensor

    def start(self, executor):
        if self._worker_thread is not None:
            return
        self._executor = executor
        self._worker_thread = threading.Thread(target=self._scheduler, daemon=True)
        self._worker_thread.start()

    def submit(self, task: Task):
        self._queue.put(task)

    def _scheduler(self):
        last_task: Task = None
        while not self._stop_event.is_set():
            # Start a new batch
            if last_task is not None:
                task = last_task
                last_task = None
            else:
                task: Task = self._queue.get()
            tasks = [task]
            total_size = task.sample_size

            since = time.time()
            # Add more tasks to the batch, until the batch is full or timeout
            while total_size < self._batch_size and not self._stop_event.is_set():
                try:
                    if self._timeout == 0:
                        task = self._queue.get(block=False)
                    else:
                        elapsed = time.time() - since
                        wait = self._timeout - elapsed
                        if wait < 0:
                            break
                        task = self._queue.get(timeout=wait)

                    if total_size + task.sample_size <= self._batch_size:
                        tasks.append(task)
                        total_size += task.sample_size
                    else:
                        last_task = task
                        break

                except queue.Empty:
                    break

            if self._stop_event.is_set():
                break

            args, kwargs = self._batcher.batch(tasks)
            # Process the batch
            # TODO(yuheng): pass stop event to executor
            result = self._executor(*args, **kwargs)
            self._batcher.unbatch(result, tasks)

            tasks.clear()

    def stop(self):
        if self._worker_thread is None:
            return
        self._stop_event.set()
        self._worker_thread.join()
        self._worker_thread = None
        self._executor = None

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self.start(func)

            task = Task(args, kwargs)
            self.submit(task)

            return task.result()

        return wrapper


class SingleScheduler(object):
    def __init__(self) -> None:
        self._queue = queue.Queue()
        self._worker_thread = None
        self._stop_event = threading.Event()

    def start(self, executor):
        if self._worker_thread is not None:
            return
        self._executor = executor
        self._worker_thread = threading.Thread(target=self._scheduler, daemon=True)
        self._worker_thread.start()

    def submit(self, task: Task):
        self._queue.put(task)

    def _scheduler(self):
        while not self._stop_event.is_set():
            task: Task = self._queue.get()
            if self._stop_event.is_set():
                break

            # TODO(yuheng): pass stop event to executor
            result = self._executor(*task.args, **task.kwargs)
            task.set_result(result)

    def stop(self):
        if self._worker_thread is None:
            return
        self._stop_event.set()
        self._worker_thread.join()
        self._worker_thread = None
        self._executor = None

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self.start(func)

            task = Task(args, kwargs)
            self.submit(task)
            return task.result()

        return wrapper
