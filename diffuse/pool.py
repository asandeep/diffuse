import logging
import weakref

LOGGER = logging.getLogger(__name__)


class WorkerPool:
    """Worker pool that keeps track of currently running worker processes."""

    def __init__(self):
        # Keep week ref for workers added to the pool. This simplifies pool
        # logic as it is no longer required to keep track of dead workers.
        self._workers = weakref.WeakSet()

    @property
    def size(self):
        """
        The current pool size.

        Returns the count of workers that are currently alive.
        """
        # Weekrefs are maintained till they are garbage collected. Simply
        # returning the length of pool will give incorrect result in case worker
        # is dead but not yet garbage collected.
        return len([worker.is_alive() for worker in self._workers])

    def add(self, worker):
        """Adds a new worker to pool.

        Args:
            worker: The implementation specific worker instance.
        """
        self._workers.add(worker)

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
        for task_worker in self._workers:
            task_worker.join()
            LOGGER.debug("Stopped worker: %s", task_worker.id)

    async def shutdown_async(self, wait):
        """
        Asynchronous implementation for shutdown that works with ASync workers.
        """
        self._stop_workers()

        if not wait:
            return

        for task_worker in self._workers:
            await task_worker.join()
            LOGGER.debug("Stopped worker: %s", task_worker.id)

    def _stop_workers(self):
        """Stops running worker processes."""
        LOGGER.debug("Shutting down worker pool.")

        # Since task queue is shared among workers, stop signal sent to a worker
        # might actually be read by completely different worker. This will
        # result in the receiving worker initiating its termination process and
        # getting auto evicted from pool due to weak reference.
        # To make sure every worker receives stop signal at least once, we need
        # to send as many signals as there are workers in pool.
        workers_copy = [worker for worker in self._workers]
        for task_worker in workers_copy:
            LOGGER.debug("Stopping worker: %s", task_worker.id)
            task_worker.stop()
