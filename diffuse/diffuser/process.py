import logging
import multiprocessing
import queue
import threading
import time
from concurrent import futures

from diffuse.diffuser import base
from diffuse.worker import process as process_worker

LOGGER = logging.getLogger(__name__)


class _TaskInstance(object):
    """
    Reflects the object that is added to task queue.

    Since tasks are processed in separate child process, sending the actual task
    and setting future result there won't cascade the result back to client. So,
    instead we send this instance in queue which holds the reference to task
    result/exception depending on how task was executed. The child process then
    adds the same task instance to result queue which is read by separate result
    processing thread.
    """

    def __init__(self, target, *args, **kwargs):
        self.id = id(self)

        self.target = target
        self.args = args
        self.kwargs = kwargs
        self.result = None
        self.exception = None

    def run(self):
        """
        Calls the task target with supplied args and kwargs parameters, and
        keeps reference to task output/exception.
        """
        try:
            self.result = self.target(*self.args, **self.kwargs)
        except BaseException as exc:
            self.exception = exc

        return self


class _Task(object):
    def __init__(self, target, *args, **kwargs):
        self.future = futures.Future()
        self.instance = _TaskInstance(target, *args, **kwargs)


class ProcessDiffuser(base._SyncDiffuser):
    """
    A diffuser implementation that executes tasks in separate child process.
    """

    _WORKER_CLASS = process_worker.ProcessWorker
    _TASK_CLASS = _Task
    _QUEUE_EMPTY_EXCEPTION = queue.Empty

    def __init__(self, target, ephemeral=False, max_workers=None):

        if max_workers is None:
            max_workers = multiprocessing.cpu_count()

        if max_workers <= 0:
            raise ValueError("max_workers must be greater than 0")

        if max_workers > multiprocessing.cpu_count():
            raise ValueError(
                "Worker count %s is more than supported by system (%s)",
                max_workers,
                multiprocessing.cpu_count(),
            )

        LOGGER.debug("Max workers: %s", max_workers)
        self._max_workers = max_workers

        self._task_queue = multiprocessing.Queue()
        self._close_lock = threading.Lock()

        super().__init__(target, ephemeral)

        # Apart from task queue, we also need a result queue to receive results
        # from worker processes.
        self._result_queue = multiprocessing.Queue()

        # Maintains the list of tasks that have been sent to workers for
        # processing for which results haven't been received yet.
        self._pending_tasks = {}

        # Whether the stop result processor thread.
        self._stop_result_processor = False

        # A separate result processing thread that receives results pushed by
        # child processes in result queue, and updates the associated task's
        # future instance based on result.
        self._result_processor = threading.Thread(
            target=self._process_task_results
        )
        self._result_processor.start()

    @property
    def task_queue(self):
        return self._task_queue

    @property
    def max_workers(self):
        return self._max_workers

    @property
    def close_lock(self):
        return self._close_lock

    def _diffuse(self, task):
        # Since task is processed in completely separate process, it would be
        # hard to know when task actually started running. So, we set it to
        # running as soon as task is submitted for processing.
        # Also, the future state should be set at the earliest, as there are
        # chances that task gets picked up by a worker as soon as it is added to
        # queue, and for small tasks the processing might complete even before
        # future state is set.
        task.future.set_running_or_notify_cancel()

        self._pending_tasks[task.instance.id] = task

        super(ProcessDiffuser, self)._diffuse(task.instance)

    def _worker_init_kwargs(self):
        """
        Sends result_queue to worker.
        """
        return {"result_queue": self._result_queue}

    def _cleanup(self, wait):
        """
        Notifies result processor to shutdown.

        Args:
            wait: Whether to wait for all results to be processed from queue.
        """
        self._result_queue.put_nowait(None)

        if wait:
            self._result_processor.join()

    def _process_task_results(self):
        """
        Retrieves task results from result queue and updates corresponding
        future objects.
        """
        while not self._stop_result_processor:
            task_instance = self._result_queue.get()
            if task_instance is None:
                self._stop_result_processor = True
                continue

            task = self._pending_tasks[task_instance.id]
            # If task was already cancelled by user, skip further processing.
            if task.future.cancelled():
                return

            if task_instance.result:
                task.future.set_result(task_instance.result)
            else:
                task.future.set_exception(task_instance.exception)

            del self._pending_tasks[task_instance.id]
