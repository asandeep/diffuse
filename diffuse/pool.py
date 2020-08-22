import logging

LOGGER = logging.getLogger(__name__)


class WorkerPool:
    """Worker pool that keeps track of currently running worker processes."""

    def __init__(self):
        self._workers = []

    @property
    def size(self):
        """The current pool size.

        Returns the count of workers that are currently alive.
        """
        self._workers = [
            worker for worker in self._workers if worker.is_alive()
        ]
        return len(self._workers)

    def add(self, worker):
        """Adds a new worker to pool.

        Args:
            worker: The implementation specific worker instance.
        """
        self._workers.append(worker)

    def shutdown(self, wait):
        """Shuts down pool and stops all the worker processes.

        This is a synchronous implementation and works with Sync workers.

        Args:
            wait: Whether to wait for workers to complete the currently assigned
                task.
        """
        self._stop_workers()

        if not wait:
            return

        # Wait for worker processes to complete. Due to the way worker processes
        # decides that it is time to shutdown, joining of worker can't be done
        # just after workers are asked to stop above.
        for task_worker in list(self._workers):
            task_worker.join()
            LOGGER.debug("Stopped worker: %s", task_worker.id)

    async def shutdown_async(self, wait):
        """
        Asynchronous implementation for shutdown that works with ASync workers.
        """
        self._stop_workers()

        if not wait:
            return

        for task_worker in list(self._workers):
            await task_worker.join()
            LOGGER.debug("Stopped worker: %s", task_worker.id)

    def _stop_workers(self):
        """Stops running worker processes."""
        LOGGER.debug("Shutting down worker pool.")

        # When using ephemeral workers, the workers might have already stopped
        # when this method is called. If that is the case, return.
        if not self.size:
            return

        for task_worker in list(self._workers):
            LOGGER.debug("Stopping worker: %s", task_worker.id)
            task_worker.stop()
