import logging
import os
import queue
from concurrent import futures

from diffuse import worker
from diffuse.diffuser import base


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
            self.future.set_exception(exc)
        else:
            self.future.set_result(result)
            return result


class ThreadDiffuser(base._SyncDiffuser):
    _QUEUE_EMPTY_EXCEPTION = queue.Empty
    _TASK_CLASS = _Task
    _WORKER_CLASS = worker.ThreadWorker

    def _init_task_queue(self):
        return queue.Queue()

    def _get_max_workers(self):
        return min(32, (os.cpu_count() or 1) + 4)
