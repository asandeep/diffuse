import logging
import multiprocessing
import queue
import threading
from concurrent import futures

from diffuse import worker
from diffuse.diffuser import base


class _TaskInstance:
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


class _Task:
    def __init__(self, target, *args, **kwargs):
        self.future = futures.Future()
        self.instance = _TaskInstance(target, *args, **kwargs)


class ProcessDiffuser(base._SyncDiffuser):
    """
    A diffuser implementation that executes tasks in separate child processes.
    """

    _QUEUE_EMPTY_EXCEPTION = queue.Empty
    _TASK_CLASS = _Task
    _WORKER_CLASS = worker.ProcessWorker

    def __init__(self, target, ephemeral=False, max_workers=None):

        if max_workers and max_workers > multiprocessing.cpu_count():
            raise ValueError(
                f"Worker count {max_workers} is more than supported "
                f"by system ({multiprocessing.cpu_count()})."
            )

        super().__init__(target, ephemeral, max_workers)

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

    def _init_task_queue(self):
        return multiprocessing.Queue()

    def _get_max_workers(self):
        return multiprocessing.cpu_count()

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

        super()._diffuse(task.instance)

    def _worker_init_kwargs(self):
        """
        Sends result_queue to worker.
        """
        return {"result_queue": self._result_queue}

    def _cleanup(self, wait):
        """
        Notifies result processor to shutdown.

        Args:
            wait: Whether to wait for cleanup actions to be completed.
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

            if task_instance.result:
                task.future.set_result(task_instance.result)
            else:
                task.future.set_exception(task_instance.exception)

            del self._pending_tasks[task_instance.id]

    def _drain_tasks(self):
        """
        Overrides base implementation to skip cancelling the task future.

        Since future is set to running as soon as it is added to queue, same
        cannot be cancelled during drain operation.
        """
        while True:

            try:
                # During unit tests, `get_nowait` isn't somehow removing task
                # from queue. Using `get` with timeout works as expected.
                self._task_queue.get(timeout=1 / 1000)
            except self._QUEUE_EMPTY_EXCEPTION:
                break
