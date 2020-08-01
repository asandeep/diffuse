import threading
from diffuse.worker import base
import queue as thread_safe_queue


class ThreadWorker(threading.Thread, base._BaseWorker):
    """
    A worker implementation that runs in a separate thread.

    The order in which parent classes are inherited is important due to the way
    python MRO works.
    """

    def __init__(self, queue, ephemeral, name_prefix):
        # Since Thread is the first class mentioned in inheritance list, this
        # call will go to Thread class and make sure that worker thread is
        # correctly initialized.
        super().__init__()

        # This call is required to initialize the standalone worker
        # functionality.
        base._BaseWorker.__init__(self, queue, ephemeral)

        self._name_prefix = name_prefix or self.__class__.__name__

    @property
    def id(self):
        return "__".join((self._name_prefix, str(self.ident)))

    def run(self):
        """Overrides base implementation and calls worker's _run method."""
        self._run()

    def _get_task(self):
        try:
            return self._queue.get(block=not self._ephemeral)
        except thread_safe_queue.Empty:
            pass

        return None
