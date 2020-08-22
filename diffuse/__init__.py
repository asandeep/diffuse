__version__ = "0.1.0"

import enum
from diffuse.diffuser import thread, async_diffuser


class Diffuser(enum.Enum):
    THREAD = thread.ThreadDiffuser
    ASYNC = async_diffuser.AsyncDiffuser

    @classmethod
    def create(cls, target, *, diffuser_type=None, ephemeral=False, **kwargs):
        if not diffuser_type:
            diffuser_type = cls.THREAD

        if not isinstance(diffuser_type, cls):
            raise TypeError("Unknown diffuser: %s" % diffuser_type)

        diffuser_class = cls(diffuser_type).value
        return diffuser_class(target, ephemeral, **kwargs)
