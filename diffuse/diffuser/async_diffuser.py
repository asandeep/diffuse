import asyncio
import logging
import math
import threading

from diffuse.diffuser import base
from diffuse.worker import async_worker

LOGGER = logging.getLogger(__name__)


class _AsyncTask(object):
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
            # A catch-all block to prevent thread from getting killed abruptly
            # in event of an exception.
            LOGGER.error("Error while running target: %s", str(exc))
            self.future.set_exception(exc)
        else:
            self.future.set_result(result)
            return result


class AsyncDiffuser(base._ASyncDiffuser):
    _WORKER_CLASS = async_worker.AsyncWorker
    _TASK_CLASS = _AsyncTask
    _QUEUE_EMPTY_EXCEPTION = asyncio.QueueEmpty

    def __init__(self, target, ephemeral=False, max_workers=None):

        super().__init__(target, ephemeral)

        # Allow infinite workers.
        # TODO(sandeep): May be revisit and see if this practical.
        self._max_workers = math.inf
        LOGGER.debug("Max workers: %s", self._max_workers)

        self._task_queue = asyncio.Queue()
        self._close_lock = asyncio.Lock()

    @property
    def task_queue(self):
        return self._task_queue

    @property
    def max_workers(self):
        return self._max_workers

    @property
    def close_lock(self):
        return self._close_lock
