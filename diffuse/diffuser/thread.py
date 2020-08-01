import logging
import os
import queue as thread_safe_queue
import threading
from concurrent import futures

from diffuse.diffuser import base
from diffuse.worker import thread as thread_worker

LOGGER = logging.getLogger(__name__)


class ThreadDiffuser(base._BaseDiffuser):
    _WORKER_CLASS = thread_worker.ThreadWorker

    def __init__(self, target, ephemeral=False, max_workers=None):
        if max_workers is None:
            max_workers = min(32, (os.cpu_count() or 1) + 4)

        if max_workers <= 0:
            raise ValueError("max_workers must be greater than 0")

        LOGGER.debug("Max workers: %s", max_workers)
        self._max_workers = max_workers

        self._task_queue = thread_safe_queue.Queue()

        super().__init__(target, ephemeral)

    @property
    def task_queue(self):
        return self._task_queue

    @property
    def max_workers(self):
        return self._max_workers

    def _drain_tasks(self):
        while True:
            try:
                task = self.task_queue.get_nowait()
                if task is not None:
                    task.future.cancel()
            except thread_safe_queue.Empty:
                break
