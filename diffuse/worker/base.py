import logging

LOGGER = logging.getLogger(__name__)


class _BaseWorker(object):
    def __init__(self, queue, ephemeral):
        # Queue to be monitored for tasks.
        self._queue = queue

        self._ephemeral = ephemeral

        # Indicates that there are no more tasks to be processed and that the
        # worker is now going to shut down.
        self._is_done = False

    def start(self):
        """
        Provides a way to call run method in a separate thread/process.
        """
        raise NotImplementedError

    def stop(self):
        """
        Sets conditions that will cause worker process to stop running.

        It should be noted that calling this method won't terminate the worker
        immediately. The worker will keep running until there are no more real
        pending tasks left in queue.
        """
        # Even when execution flag is set to false, the thread might be already
        # done with processing all the pending tasks and is currently blocked on
        # queue waiting for new task. At this point we insert a dummy task to
        # unblock the thread.
        self._queue.put(None)

    def _run(self):
        """
        Fetches task from queue and starts processing.

        This method should be called from Thread/Process run methods.
        """
        # Keep running until parent process calls stop to indicate that there
        # are no more tasks or we have encountered a dummy task. while
        # self.execution_flag or not self._is_done:
        # TODO(sandeep): Checking execution flag shouldn't really be required
        # as, as soon as worker finds a dummy task, it knows that it should
        # shutdown.
        while not self._is_done:
            LOGGER.debug("Worker:%s reading message from queue.", self.id)

            task = self._get_task()
            # Since None task is only inserted at the end of queue when caller
            # has asked thread to stop, we can be sure that there are no more
            # pending tasks.
            # This assertion doesn't however hold when caller thread itself gets
            # killed and calls child's thread stop method before dying.
            if task is None:
                self._is_done = True
            else:
                task.run()
            # self._queue.task_done()

        LOGGER.debug(
            "Worker: %s - stopped. Pending task count: %s",
            self.id,
            self._queue.qsize(),
        )
