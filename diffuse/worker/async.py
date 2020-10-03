from diffuse.worker import base


class AsyncWorker(base._AsyncWorker):
    """A worker implementation that runs asynchronously."""

    @property
    def id(self):
        return "__".join((self.__class__.__name__, str(id(self))))
