import os
from concurrent import futures
from contextlib import contextmanager
from unittest import mock

import pytest

import diffuse
from diffuse.diffuser.base import pool
from diffuse.diffuser.tests import base


@contextmanager
def noop():
    yield


@pytest.mark.parametrize("diffuser_type", [diffuse.Diffuser.THREAD])
class TestProcessDiffuser(base.BaseDiffuserTest):
    @pytest.fixture
    def future_running(self):
        return False

    @pytest.fixture
    def future_cancelled(self):
        return True

    def test__default_diffuser(self, diffuser_type):
        del diffuser_type

        default_diffuser = diffuse.Diffuser.create(target=lambda: "test")
        assert isinstance(default_diffuser, diffuse.Diffuser.THREAD.value)

    @pytest.fixture
    def expected(self, request):
        if request.param is None:
            return min(32, (os.cpu_count() or 1) + 4)

        return request.param

    @pytest.mark.parametrize(
        "input,expected,might_raise",
        [
            (
                -1,
                None,
                pytest.raises(
                    ValueError, match="max_workers must be greater than 0."
                ),
            ),
            (None, None, noop()),
            (10, 10, noop()),
        ],
        indirect=["expected"],
    )
    def test__max_workers(self, diffuser_type, input, expected, might_raise):
        with might_raise:
            diffuser = diffuse.Diffuser.create(
                target=lambda: "test",
                diffuser_type=diffuser_type,
                max_workers=input,
            )
            assert diffuser._max_workers == expected
