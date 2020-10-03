import os
import time
from concurrent import futures
from contextlib import contextmanager
from unittest import mock

import pytest

import diffuse
from diffuse.diffuser.base import pool


def target(msg):
    return f"hello {msg}"


def target_exception(msg):
    raise ValueError("Test")


def target_long_running(msg):
    time.sleep(1 / 10)
    return f"hello {msg}"


class BaseDiffuserTest:
    def test__target_not_callable(self, diffuser_type):
        with pytest.raises(TypeError, match="target must be a callable."):
            diffuse.Diffuser.create(target=None, diffuser_type=diffuser_type)

    def test__diffuse(self, mocker, diffuser_type, future_running):
        with diffuse.Diffuser.create(
            target=target, diffuser_type=diffuser_type
        ) as diffuser:
            spy_async_worker = mocker.spy(diffuser, "_WORKER_CLASS")

            future = diffuser.diffuse("world")
            futures.wait([future])

            assert isinstance(future, futures.Future)
            assert not future.cancelled()
            assert future.done()
            assert future.result() == "hello world"

            assert diffuser._task_queue.qsize() == 0

            spy_async_worker.assert_called_once_with(
                diffuser._task_queue, False, **diffuser._worker_init_kwargs()
            )
            assert diffuser._worker_pool.size == 1

        return diffuser

    def test__diffuse__task_exception(
        self, mocker, diffuser_type, future_running
    ):
        with diffuse.Diffuser.create(
            target=target_exception, diffuser_type=diffuser_type
        ) as diffuser:
            spy_async_worker = mocker.spy(diffuser, "_WORKER_CLASS")

            future = diffuser.diffuse("world")
            futures.wait([future])

            assert isinstance(future, futures.Future)
            assert not future.cancelled()
            assert future.done()

            with pytest.raises(ValueError, match="Test"):
                assert future.result()

            assert diffuser._task_queue.qsize() == 0

            spy_async_worker.assert_called_once_with(
                diffuser._task_queue, False, **diffuser._worker_init_kwargs()
            )
            assert diffuser._worker_pool.size == 1

        return diffuser

    def test__diffuse__task_consumed_by_worker(
        self, mocker, diffuser_type, future_running
    ):
        """
        Verifies that no new worker is initialized when there are no task in
        queue i.e. task was consumed by another worker as soon as it was added
        to queue.
        """
        with diffuse.Diffuser.create(
            target=target, diffuser_type=diffuser_type
        ) as diffuser:
            mocker.patch.object(diffuser._task_queue, "qsize", return_value=0)

            future = diffuser.diffuse("world")
            assert isinstance(future, futures.Future)
            assert future.running() == future_running
            assert not future.done()

            assert diffuser._worker_pool.size == 0

        return diffuser

    def test__diffuse__max_pool_size(
        self, mocker, diffuser_type, future_running
    ):
        with diffuse.Diffuser.create(
            target=target, diffuser_type=diffuser_type, max_workers=1
        ) as diffuser:
            mock_worker = mocker.MagicMock()
            diffuser._worker_pool.add(mock_worker)

            future = diffuser.diffuse("world")
            assert isinstance(future, futures.Future)
            assert future.running() == future_running
            assert not future.done()

            assert diffuser._worker_pool.size == 1

        return diffuser

    def test__diffuse__close(self, mocker, diffuser_type):
        diffuser = diffuse.Diffuser.create(
            target=target, diffuser_type=diffuser_type
        )
        spy_pool_shutdown = mocker.spy(diffuser._worker_pool, "shutdown")

        future = diffuser.diffuse("world")
        diffuser.close()

        assert diffuser.closed
        assert not future.cancelled()
        assert future.done()
        spy_pool_shutdown.assert_called_once_with(wait=True)

        return diffuser

    def test__diffuse__close__no_wait(self, mocker, diffuser_type):
        diffuser = diffuse.Diffuser.create(
            target=target_long_running, diffuser_type=diffuser_type
        )
        spy_pool_shutdown = mocker.spy(diffuser._worker_pool, "shutdown")

        future = diffuser.diffuse("world")
        diffuser.close(wait=False)

        assert diffuser.closed
        assert not future.cancelled()
        assert not future.done()
        spy_pool_shutdown.assert_called_once_with(wait=False)

        return diffuser

    def test__diffuse__close__cancel_pending(
        self, mocker, diffuser_type, future_cancelled
    ):
        diffuser = diffuse.Diffuser.create(
            target=target, diffuser_type=diffuser_type
        )
        spy_pool_shutdown = mocker.spy(diffuser._worker_pool, "shutdown")
        mock_worker = mocker.patch.object(diffuser, "_WORKER_CLASS")

        future = diffuser.diffuse("world")
        diffuser.close(cancel_pending=True)

        assert diffuser.closed
        assert future.cancelled() == future_cancelled
        assert diffuser._task_queue.qsize() == 0
        spy_pool_shutdown.assert_called_once_with(wait=True)

        return diffuser

    def test__diffuse__closed_diffuser(self, mocker, diffuser_type):
        diffuser = diffuse.Diffuser.create(
            target=target, diffuser_type=diffuser_type
        )

        diffuser.close()

        with pytest.raises(
            RuntimeError, match="Cannot diffuse on closed Diffuser."
        ):
            diffuser.diffuse("world")

        assert diffuser._task_queue.qsize() == 0
        assert diffuser._worker_pool.size == 0

        return diffuser
