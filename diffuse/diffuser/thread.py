import logging
import os
import queue as thread_safe_queue
import threading
from concurrent import futures

from diffuse.diffuser import base
from diffuse.worker import thread as thread_worker

LOGGER = logging.getLogger(__name__)


class _Task(object):
    def __init__(self, target, *args, **kwargs):
        self.target = target
        self.args = args
        self.kwargs = kwargs
        self.future = futures.Future()

    def run(self):
        if not self.future.set_running_or_notify_cancel():
            return

        try:
            result = self.target(*self.args, **self.kwargs)
        except BaseException as exc:
            # A catch-all block to prevent thread from getting killed abruptly
            # in event of an exception.
            LOGGER.error("Error while running target: %s", str(exc))
            self.future.set_exception(exc)
            # Break a reference cycle with the exception 'exc'
            self = None
        else:
            self.future.set_result(result)


class ThreadDiffuser(base._SyncDiffuser):
    _WORKER_CLASS = thread_worker.ThreadWorker
    _TASK_CLASS = _Task
    _QUEUE_EMPTY_EXCEPTION = thread_safe_queue.Empty

    def __init__(self, target, ephemeral=False, max_workers=None):
        if max_workers is None:
            max_workers = min(32, (os.cpu_count() or 1) + 4)

        if max_workers <= 0:
            raise ValueError("max_workers must be greater than 0")

        LOGGER.debug("Max workers: %s", max_workers)
        self._max_workers = max_workers

        self._task_queue = thread_safe_queue.Queue()
        self._close_lock = threading.Lock()

        super().__init__(target, ephemeral)

    @property
    def task_queue(self):
        return self._task_queue

    @property
    def max_workers(self):
        return self._max_workers

    @property
    def close_lock(self):
        return self._close_lock
