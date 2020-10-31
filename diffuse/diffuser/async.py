import asyncio
import logging
import math

from diffuse import worker
from diffuse.diffuser import base


class _Task:
    """Task that runs Asynchronously."""

    def __init__(self, target, *args, **kwargs):
        self.target = target
        self.args = args
        self.kwargs = kwargs
        self.future = asyncio.Future()

    async def run(self):
        try:
            result = await self.target(*self.args, **self.kwargs)
        except BaseException as exc:
            # Catch any exception raised by target callable and set on
            # corresponding future.
            self.future.set_exception(exc)
        else:
            self.future.set_result(result)
            return result


class AsyncDiffuser(base._ASyncDiffuser):
    _QUEUE_EMPTY_EXCEPTION = asyncio.QueueEmpty
    _TASK_CLASS = _Task
    _WORKER_CLASS = worker.AsyncWorker

    def _init_task_queue(self):
        return asyncio.Queue()

    def _get_max_workers(self):
        # Allow infinite workers.
        # TODO(sandeep): May be revisit and see if this practical.
        return math.inf
