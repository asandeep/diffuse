import asyncio
import logging

from diffuse.worker import base

LOGGER = logging.getLogger(__name__)


class AsyncWorker(base._BaseWorker):
    """A worker implementation that runs asynchronously."""

    def __init__(self, queue, ephemeral):
        super(AsyncWorker, self).__init__(queue, ephemeral)

        self._finished = asyncio.Event()

    @property
    def id(self):
        return "__".join((self.__class__.__name__, str(id(self))))

    def is_alive(self):
        """Returns whether the worker is currently alive."""
        return not self._finished.is_set()

    async def join(self):
        """Blocks till worker is finished processing tasks."""
        if self.is_alive():
            await self._finished.wait()

    async def start(self):
        await self._run()

    async def _run(self):
        while not self._finished.is_set():
            LOGGER.debug("%s reading message from queue.", self.id)
            task = await self._get_task()

            if task is None:
                self._finished.set()
            else:
                result = await task.run()
                await self._process_result(result)

        LOGGER.debug(
            "%s - finished. Pending task count: %s",
            self.id,
            self._queue.qsize(),
        )

    async def _get_task(self):
        if self._ephemeral:
            try:
                return self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return None

        return await self._queue.get()

    async def _process_result(self, result):
        pass
