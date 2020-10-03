import logging
import weakref

LOGGER = logging.getLogger(__name__)


class WorkerPool:
    """
    Worker pool that keeps track of currently running worker processes.
    """

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
        return len([worker for worker in self._workers if worker.is_running()])

    def add(self, worker):
        """
        Adds a new worker to pool.

        Args:
            worker: The implementation specific worker instance.
        """
        self._workers.add(worker)

    def shutdown(self, wait):
        """
        Shuts down pool and stops all the worker processes.

        This is a synchronous implementation and works with Sync workers.

        Args:
            wait: Whether to wait for workers to complete the currently assigned
                task.
        """
        LOGGER.debug("Shutting down worker pool.")
        self._stop_workers()

        if not wait:
            return

        # Wait for worker processes to complete.
        for task_worker in self._workers:
            task_worker.wait()
            LOGGER.debug("Stopped worker: %s", task_worker.id)

    async def shutdown_async(self, wait):
        """
        Asynchronous implementation for shutdown that works with ASync workers.
        """
        LOGGER.debug("Shutting down worker pool.")
        self._stop_workers()

        if not wait:
            return

        for task_worker in self._workers:
            await task_worker.wait()
            LOGGER.debug("Stopped worker: %s", task_worker.id)

    def _stop_workers(self):
        """
        Stops running worker processes.
        """
        # Since task queue is shared among workers, stop signal sent to a worker
        # might actually be read by completely different worker. This will
        # result in the receiving worker initiating its termination process and
        # getting auto evicted from pool due to weak reference and changing the
        # pool size during iteration.
        # This can result in less stop signals being sent than actual count of
        # workers. To prevent this from happening, we create a strong reference
        # to available workers in pool before sending stop signal to make sure
        # worker won't be auto-removed from pool even when it received the stop
        # signal and decided to shut itself down.
        workers_copy = [worker for worker in self._workers]
        for task_worker in workers_copy:
            LOGGER.debug("Stopping worker: %s", task_worker.id)
            task_worker.stop()
