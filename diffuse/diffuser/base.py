import asyncio
import logging
from concurrent import futures

from diffuse import pool

LOGGER = logging.getLogger(__name__)


class _BaseDiffuser:
    """
    The Base diffuser implementation. This class should only be inherited by
    base implementations for Sync and ASync diffusers.

    The class provides implementation agnostic boilerplate code, which can be
    called by specific implementations.
    """

    # Worker implementation compatible with this diffuser.
    _WORKER_CLASS = None
    # The task implementation compatible with this diffuser.
    _TASK_CLASS = None
    # Exception raised by the task queue when it is empty. Since every diffuser
    # utilize a different type of queue, adding the class here allows us to
    # refactor some of the implementation agnostic logic.
    _QUEUE_EMPTY_EXCEPTION = None

    def __init__(self, target, ephemeral=False):
        if not callable(target):
            raise TypeError("target must be a callable.")

        self._target = target
        self._ephemeral = ephemeral

        self._closed = False

        self._worker_pool = pool.WorkerPool()

    @property
    def task_queue(self):
        """Implementation specific task queue."""
        raise NotImplementedError("Must be implemented by child class.")

    @property
    def max_workers(self):
        raise NotImplementedError("Must be implemented by child class.")

    @property
    def close_lock(self):
        """Returns Diffuser specific lock implementation."""
        raise NotImplementedError("Must be implemented by child class.")

    @property
    def closed(self):
        """Whether the diffuser is closed."""
        return self._closed

    def diffuse(self, *args, **kwargs):
        """Diffuses given arguments and keyword arguments to target callable."""
        raise NotImplementedError()

    def close(self, wait=True, cancel_pending=False):
        """
        Closes the diffuser and stops all worker processes. Optionally blocks
        till all the tasks have been processed. Note that no futher tasks can be
        processed once Diffuser is closed and one must initialize a new diffuser
        if required.

        Args:
            wait: Whether to wait for currently processing tasks to finish.
            cancel_pending: Whether to cancel pending tasks that haven't yet
                been picked for processing. The tasks will be cancelled and
                removed from queue.
        """
        raise NotImplementedError()

    def _init_worker(self, **kwargs):
        """Initializes and returns a new worker if required.

        Below conditions should meet for worker to get initialized:
          1. Pool size should be less than allowed max worker limit.
          2. No. of tasks in queue should be more than current running worker
             count.

        This method should be called when diffusing a new task. If all the above
        pre-requisites are met, then a new worker will be initialized and
        returned. However, starting the worker is implementation specific and
        left with diffuser to do.
        """

        # No need to initialize new worker if there aren't any pending tasks in
        # queue.
        if not self.task_queue.qsize():
            return

        if self._worker_pool.size >= self.max_workers:
            return

        worker = self._WORKER_CLASS(self.task_queue, self._ephemeral, **kwargs)
        self._worker_pool.add(worker)
        LOGGER.debug("Created new worker: %s", worker.id)
        return worker

    def _drain_tasks(self):
        """Removes pending tasks from queue.

        The method should be called from `close` method by Diffuser
        implementation, when close signal is received.
        """
        LOGGER.debug("Cancelling pending tasks.")
        while True:
            try:
                task = self.task_queue.get_nowait()
                if task is not None:
                    task.future.cancel()
            except self._QUEUE_EMPTY_EXCEPTION:
                break


class _SyncDiffuser(_BaseDiffuser):
    """Base implementation for Synchronous diffusers i.e. Threads/Processes."""

    def diffuse(self, *args, **kwargs):
        with self.close_lock:
            if self._closed:
                raise RuntimeError("Cannot diffuse on closed Diffuser.")

            task = self._TASK_CLASS(self._target, *args, **kwargs)
            self._diffuse(task)

            worker = self._init_worker(**self._worker_init_kwargs())
            if worker:
                worker.start()

            return task.future

    def _diffuse(self, task):
        """Adds task to queue."""
        self.task_queue.put(task)

    def _worker_init_kwargs(self):
        """
        Returns implementation specific additional arguments that are passed
        while initializing a new worker instance.
        """
        return {}

    def close(self, wait=True, cancel_pending=False):
        LOGGER.debug("Close request received.")
        with self.close_lock:
            self._closed = True

            if cancel_pending:
                self._drain_tasks()

            self._worker_pool.shutdown(wait=wait)
            self._cleanup(wait)

    def _cleanup(self, wait: bool):
        """
        Any implementation specific cleanup actions to be performed when
        diffuser is closed.

        This method is called after clearing pending tasks from queue and
        stopping all running workers.

        Args:
            wait: Whether to wait for exiting tasks to complete processing.
        """
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class _ASyncDiffuser(_BaseDiffuser):
    """Base implementation for ASynchronous diffuser."""

    async def diffuse(self, *args, **kwargs):
        async with self.close_lock:
            if self._closed:
                raise RuntimeError("Cannot diffuse on closed Diffuser.")

            task = self._TASK_CLASS(self._target, *args, **kwargs)
            await self.task_queue.put(task)
            worker = self._init_worker()
            if worker:
                asyncio.ensure_future(worker.start())

            # Allow worker to pick up and start processing task from queue.
            await asyncio.sleep(0)
            return task.future

    async def close(self, wait=True, cancel_pending=False):
        LOGGER.debug("Close request received.")
        async with self.close_lock:
            self._closed = True

            if cancel_pending:
                self._drain_tasks()

            await self._worker_pool.shutdown_async(wait=wait)

            await self._cleanup(wait)

    async def _cleanup(self, wait):
        """
        Any implementation specific cleanup actions to be performed when
        diffuser is closed.
        """
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
