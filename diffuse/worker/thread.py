import threading

from diffuse.worker import base


class ThreadWorker(threading.Thread, base._SyncWorker):
    """
    A worker implementation that runs in a separate thread.

    The order in which parent classes are inherited is important due to the way
    python MRO works.
    """

    def __init__(self, task_queue, ephemeral):
        # Since Thread is the first class mentioned in inheritance list, this
        # call will go to Thread class and make sure that worker thread is
        # correctly initialized.
        super().__init__()

        # This call is required to initialize the standalone worker
        # functionality.
        base._BaseWorker.__init__(self, task_queue, ephemeral)

    @property
    def id(self):
        return "__".join((self.__class__.__name__, str(self.ident)))

    def run(self):
        """Overrides base implementation and calls worker's _run method."""
        self.log.info(
            "%s - started. Task count: %s", self.id, self._task_queue.qsize()
        )

        self._run()
