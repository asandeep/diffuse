import enum

from diffuse import diffuser


class Diffuser(enum.Enum):
    THREAD = diffuser.ThreadDiffuser
    PROCESS = diffuser.ProcessDiffuser
    ASYNC = diffuser.AsyncDiffuser

    @classmethod
    def create(cls, target, *, diffuser_type=None, ephemeral=False, **kwargs):
        if not diffuser_type:
            diffuser_type = cls.THREAD

        if not isinstance(diffuser_type, cls):
            raise TypeError("Unknown diffuser: %s" % diffuser_type)

        diffuser_class = cls(diffuser_type).value
        return diffuser_class(target, ephemeral, **kwargs)
