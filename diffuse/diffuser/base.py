import logging
import threading
from concurrent import futures

from diffuse import pool

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


class _BaseDiffuser:
    # Worker implementation compatible with this diffuser.
    _WORKER_CLASS = None

    def __init__(self, target, ephemeral=False):
        if not callable(target):
            raise TypeError("target must be a callable.")

        self._target = target
        self._ephemeral = ephemeral

        self._closed = False
        self._close_lock = threading.Lock()

        self._worker_pool = None

    @property
    def task_queue(self):
        raise NotImplementedError("Must be implemented by child class.")

    @property
    def max_workers(self):
        raise NotImplementedError("Must be implemented by child class.")

    @property
    def worker_pool(self):
        if not self._worker_pool:
            self._worker_pool = pool.WorkerPool(
                self.max_workers,
                self.task_queue,
                self._WORKER_CLASS,
                self.__class__.__name__,
                self._ephemeral,
            )

        return self._worker_pool

    def diffuse(self, *args, **kwargs):
        with self._close_lock:
            if self._closed:
                raise RuntimeError("Cannot diffuse on closed Diffuser.")

            task = _Task(self._target, *args, **kwargs)
            self.task_queue.put(task)
            self.worker_pool.notify()
            return task.future

    def close(self, wait=True, cancel_pending=False):
        """
        Stops worker processes and blocks till all the tasks have been
        processed.
        """
        LOGGER.debug("Close request received.")
        with self._close_lock:
            self._closed = True

            if cancel_pending:
                LOGGER.debug("Cancelling pending tasks.")
                self._drain_tasks()

            LOGGER.debug("Shutting down worker pool.")
            self._worker_pool.shutdown(wait=wait)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close(wait=True)
        return False
