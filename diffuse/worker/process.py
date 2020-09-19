import logging
import multiprocessing
import queue

from diffuse.worker import base

LOGGER = logging.getLogger(__name__)


class ProcessWorker(multiprocessing.Process, base._BaseWorker):
    """
    A worker implementation that runs in a separate child process.

    The order in which parent classes are inherited is important due to the way
    python MRO works.
    """

    def __init__(self, queue, ephemeral, result_queue):
        # Since Process is the first class mentioned in inheritance list, this
        # call will go to Process class and make sure that worker process is
        # correctly initialized.
        super().__init__()

        # This call is required to initialize the standalone worker
        # functionality.
        base._BaseWorker.__init__(self, queue, ephemeral)

        self._result_queue = result_queue

    @property
    def id(self):
        return "__".join((self.__class__.__name__, str(self.pid)))

    def run(self):
        """Overrides base implementation and calls worker's _run method."""
        self._run()

    def _process_result(self, result):
        self._result_queue.put(result)