import pytest

import diffuse


def test__unknown_diffuser():
    with pytest.raises(TypeError, match="Unknown diffuser: Unknown"):
        diffuse.Diffuser.create(target=lambda: "hello", diffuser_type="Unknown")
