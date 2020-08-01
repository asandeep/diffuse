import logging

LOGGER = logging.getLogger(__name__)


class WorkerPool:
    def __init__(
        self,
        pool_size,
        task_queue,
        worker_impl_class,
        name_prefix=None,
        ephemeral=False,
    ):
        self._worker_impl_class = worker_impl_class
        self._ephemeral = ephemeral
        self._pool_size = pool_size
        self._task_queue = task_queue
        self._worker_name_prefix = name_prefix
        self._workers = []

    def notify(self):
        LOGGER.debug("New task added to queue.")
        self._workers = [
            worker for worker in self._workers if worker.is_alive()
        ]

        if len(self._workers) > self._task_queue.qsize():
            return

        if len(self._workers) > self._pool_size:
            return

        worker = self._worker_impl_class(
            self._task_queue, self._ephemeral, self._worker_name_prefix
        )
        worker.start()
        LOGGER.debug("Started new worker process: %s", worker.id)
        self._workers.append(worker)

    def shutdown(self, wait):
        for task_worker in list(self._workers):
            LOGGER.debug("Stopping worker: %s", task_worker.id)
            task_worker.stop()

        if wait:
            # Wait for worker processes to complete. Due to the way worker
            # processes decides that it is time to shutdown, joining of
            # worker can't be done just after workers are asked to stop
            # above.
            for task_worker in list(self._workers):
                task_worker.join()
                LOGGER.debug("Stopped worker: %s", task_worker.id)
