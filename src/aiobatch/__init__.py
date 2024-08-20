from .scheduler import (
    AsyncBatchScheduler,
    AsyncSingleScheduler,
    BatchScheduler,
    SingleScheduler,
)


def async_queuing(batch_size=1, timeout=0, is_tensor=False):
    if batch_size == 1:
        return AsyncSingleScheduler()
    else:
        return AsyncBatchScheduler(batch_size, timeout, is_tensor)


def queuing(batch_size=1, timeout=0, is_tensor=False):
    if batch_size == 1:
        return SingleScheduler()
    else:
        return BatchScheduler(batch_size, timeout, is_tensor)
