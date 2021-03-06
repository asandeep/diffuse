import asyncio
import logging
import queue
import os


# Duration in Seconds that an ephemeral worker will wait before stopping, if
# there are no tasks in queue for processing. This can be used to prevent
# aggressive killing of ephemeral workers when the tasks being processed are
# short-lived. The default duration is 1 second.
EPHEMERAL_WORKER_IDLE_TIMEOUT = int(
    os.environ.get("EPHEMERAL_WORKER_IDLE_TIMEOUT", "1")
)


class _BaseWorker:
    """
    Base worker implementation.

    Not meant to be directly inherited by concrete worker implementations.

    Attributes:
        task_queue: A queue to be monitored for tasks to arrive.
        ephemeral: Whether to continue running or die if there are no more
            tasks in queue to be processed.
    """

    def __init__(self, task_queue, ephemeral):
        self.log = logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )

        self._task_queue = task_queue
        self._ephemeral = ephemeral

        # When true, indicates that the worker has detected stop signal, is in
        # process of shutting down and won't be picking up new tasks. This can
        # happen in either of below two cases:
        # 1. When `stop` method is called on worker instance by Diffuser.
        # 2. For **ephemeral** workers, when there are no more tasks in queue to
        #    be processed.
        self._stop_signal = False

    @property
    def id(self):
        """
        Returns a unique ID for this worker instance.

        Depending on implementation, the ID might not be available till the
        worker is started.
        """
        raise NotImplementedError

    def is_running(self):
        """Returns whether worker is currently alive and running."""
        raise NotImplementedError

    def start(self):
        """
        Starts this worker instance.

        The worker will start pulling and processing tasks from queue as soon as
        this method is called, and it will continue to running until a stop is
        received or there are no more tasks in queue (for ephemeral workers).
        """
        raise NotImplementedError

    def _get_task(self):
        """
        Convenience method to retrieve task from queue.

        Blocks if there isn't any task in queue and this is not an ephemeral
        worker.
        """
        raise NotImplementedError

    def _process_result(self, result):
        """Implementation specific processing of task result."""

        pass

    def stop(self):
        """
        Notifies worker instance to stop running.

        The worker will shut itself down as soon as it receives the stop signal.
        However it should be noted that calling this method won't immediately
        terminate the worker instance. The worker will continue processing the
        task that it picked before it received the signal.
        """
        # Ephemeral tasks automatically dies as soon as there are no tasks left
        # in queue to process.
        if self._ephemeral:
            return

        # Put a None task in queue. Worker identifies this as a shutdown signal.
        self._task_queue.put_nowait(None)

    def wait(self):
        """
        Waits for worker to complete processing current task.

        Must be called after *stop* call or else the worker might block forever.
        """
        raise NotImplementedError


class _SyncWorker(_BaseWorker):
    """
    Base class for worker implementations that uses synchronous processing
    methods i.e. Thread/Subprocess.
    """

    def is_running(self):
        return self.is_alive()

    def wait(self):
        return self.join()

    def _run(self):
        """
        Defines the task retrieval and processing logic for Thread/process
        workers. Should be called from **run** method of respective
        implementations.
        """
        while not self._stop_signal:
            self.log.debug("%s reading message from queue.", self.id)

            task = self._get_task()
            if task is None:
                self._stop_signal = True
                continue

            result = task.run()
            self._process_result(result)

        self.log.info(
            "%s - stopped. Pending tasks: %s", self.id, self._task_queue.qsize()
        )

    def _get_task(self):
        timeout = EPHEMERAL_WORKER_IDLE_TIMEOUT if self._ephemeral else None
        try:
            return self._task_queue.get(timeout=timeout)
        except queue.Empty:
            pass

        return None


class _AsyncWorker(_BaseWorker):
    """
    Base class for worker implementation that uses asyncio for processing tasks.
    """

    def is_running(self):
        return not self._stop_signal

    async def wait(self):
        while self.is_running():
            # Yield control to running tasks.
            await asyncio.sleep(0)

    async def start(self):
        while not self._stop_signal:
            self.log.debug("%s reading message from queue.", self.id)

            task = await self._get_task()
            if task is None:
                self._stop_signal = True
                continue

            result = await task.run()
            self._process_result(result)

        self.log.debug(
            "%s - stopped. Pending tasks: %s", self.id, self._task_queue.qsize()
        )

    async def _get_task(self):
        if self._ephemeral:
            try:
                return self._task_queue.get_nowait()
            except asyncio.QueueEmpty:
                return None

        return await self._task_queue.get()
