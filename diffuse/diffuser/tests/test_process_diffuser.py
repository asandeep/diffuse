import enum
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


class MaxWorkerInput(enum.Enum):
    NEGATIVE = enum.auto()
    NONE = enum.auto()
    ABOVE_OS_CPU_COUNT = enum.auto()
    BELOW_OS_CPU_COUNT = enum.auto()


class MaxWorkerExpected(enum.Enum):
    ERROR = enum.auto()
    OS_CPU_COUNT = enum.auto()
    BELOW_OS_CPU_COUNT = enum.auto()


@pytest.mark.parametrize("diffuser_type", [diffuse.Diffuser.PROCESS])
class TestProcessDiffuser(base.BaseDiffuserTest):
    @pytest.fixture
    def future_running(self):
        return True

    @pytest.fixture
    def future_cancelled(self):
        return False

    @pytest.fixture
    def input(self, request):
        if request.param is MaxWorkerInput.NEGATIVE:
            return -1

        if request.param is MaxWorkerInput.NONE:
            return None

        if request.param is MaxWorkerInput.ABOVE_OS_CPU_COUNT:
            return os.cpu_count() + 1

        if request.param is MaxWorkerInput.BELOW_OS_CPU_COUNT:
            return os.cpu_count() - 1

    @pytest.fixture
    def expected(self, request):
        if request.param is MaxWorkerExpected.OS_CPU_COUNT:
            return os.cpu_count()

        if request.param is MaxWorkerExpected.BELOW_OS_CPU_COUNT:
            return os.cpu_count() - 1

        return None

    @pytest.mark.parametrize(
        "input,expected,might_raise",
        [
            (
                MaxWorkerInput.NEGATIVE,
                MaxWorkerExpected.ERROR,
                pytest.raises(
                    ValueError, match="max_workers must be greater than 0."
                ),
            ),
            (
                MaxWorkerInput.ABOVE_OS_CPU_COUNT,
                MaxWorkerExpected.ERROR,
                pytest.raises(
                    ValueError,
                    match=r"Worker count \d+ is more than supported by system \(\d+\).",
                ),
            ),
            (MaxWorkerInput.NONE, MaxWorkerExpected.OS_CPU_COUNT, noop()),
            (
                MaxWorkerInput.BELOW_OS_CPU_COUNT,
                MaxWorkerExpected.BELOW_OS_CPU_COUNT,
                noop(),
            ),
        ],
        indirect=["input", "expected"],
    )
    def test__max_workers(self, diffuser_type, input, expected, might_raise):
        with might_raise:
            with diffuse.Diffuser.create(
                target=lambda: "test",
                diffuser_type=diffuser_type,
                max_workers=input,
            ) as diffuser:
                assert diffuser._max_workers == expected

    def test__diffuse(self, mocker, diffuser_type, future_running):
        diffuser = super().test__diffuse(mocker, diffuser_type, future_running)

        assert not diffuser._pending_tasks
        assert diffuser._stop_result_processor is True
        assert not diffuser._result_processor.is_alive()

    def test__diffuse__task_exception(
        self, mocker, diffuser_type, future_running
    ):
        diffuser = super().test__diffuse__task_exception(
            mocker, diffuser_type, future_running
        )

        assert not diffuser._pending_tasks
        assert diffuser._stop_result_processor is True
        assert not diffuser._result_processor.is_alive()

    def test__diffuse__close(self, mocker, diffuser_type):
        diffuser = super().test__diffuse__close(mocker, diffuser_type)

        assert not diffuser._pending_tasks
        assert diffuser._stop_result_processor is True
        assert not diffuser._result_processor.is_alive()
