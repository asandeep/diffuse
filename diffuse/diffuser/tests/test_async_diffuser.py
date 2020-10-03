import asyncio
import math
import os
from contextlib import contextmanager
from unittest import mock

import asynctest
import pytest

import diffuse
from diffuse import worker
from diffuse.diffuser.base import pool


@contextmanager
def noop():
    yield


async def target(msg):
    return f"hello {msg}"


async def target_exception(msg):
    raise ValueError("Test")


async def target_long_running(msg):
    await asyncio.sleep(1 / 10)
    return f"hello {msg}"


class TestAsyncDiffuser:
    def test__target_not_callable(self):
        with pytest.raises(TypeError, match="target must be a callable."):
            diffuse.Diffuser.create(
                target=None, diffuser_type=diffuse.Diffuser.ASYNC
            )

    @pytest.fixture
    def expected(self, request):
        if request.param is None:
            return math.inf

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
    def test__max_workers(self, input, expected, might_raise):
        with might_raise:
            diffuser = diffuse.Diffuser.create(
                target=target,
                diffuser_type=diffuse.Diffuser.ASYNC,
                max_workers=input,
            )
            assert diffuser._max_workers == expected

    @pytest.mark.asyncio
    async def test__diffuse(self, mocker):
        async with diffuse.Diffuser.create(
            target=target, diffuser_type=diffuse.Diffuser.ASYNC
        ) as diffuser:
            spy_async_worker = mocker.spy(diffuser, "_WORKER_CLASS")

            future = await diffuser.diffuse("world")
            await future

            assert isinstance(future, asyncio.Future)
            assert not future.cancelled()
            assert future.done()
            assert future.result() == "hello world"

            assert diffuser._task_queue.qsize() == 0

            spy_async_worker.assert_called_once_with(
                diffuser._task_queue, False
            )
            assert diffuser._worker_pool.size == 1

    @pytest.mark.asyncio
    async def test__diffuse__task_exception(self, mocker):
        async with diffuse.Diffuser.create(
            target=target_exception, diffuser_type=diffuse.Diffuser.ASYNC
        ) as diffuser:
            spy_async_worker = mocker.spy(diffuser, "_WORKER_CLASS")

            future = await diffuser.diffuse("world")

            with pytest.raises(ValueError, match="Test"):
                await future

            assert isinstance(future, asyncio.Future)
            assert not future.cancelled()
            assert future.done()

            assert diffuser._task_queue.qsize() == 0

            spy_async_worker.assert_called_once_with(
                diffuser._task_queue, False
            )
            assert diffuser._worker_pool.size == 1

    @pytest.mark.asyncio
    async def test__diffuse__task_consumed_by_worker(self, mocker):
        async with diffuse.Diffuser.create(
            target=target, diffuser_type=diffuse.Diffuser.ASYNC
        ) as diffuser:
            mocker.patch.object(diffuser._task_queue, "qsize", return_value=0)

            future = await diffuser.diffuse("world")

            assert isinstance(future, asyncio.Future)
            assert not future.cancelled()
            assert not future.done()

            assert diffuser._worker_pool.size == 0

    @pytest.mark.asyncio
    async def test__diffuse__max_pool_size(self, mocker):
        async with diffuse.Diffuser.create(
            target=target, diffuser_type=diffuse.Diffuser.ASYNC, max_workers=1
        ) as diffuser:
            mock_worker = asynctest.Mock(
                diffuse.worker.AsyncWorker(diffuser._task_queue, False)
            )
            diffuser._worker_pool.add(mock_worker)

            future = await diffuser.diffuse("world")
            assert isinstance(future, asyncio.Future)
            assert not future.cancelled()
            assert not future.done()

            assert diffuser._worker_pool.size == 1

    @pytest.mark.asyncio
    async def test__diffuse__close(self, mocker):
        diffuser = diffuse.Diffuser.create(
            target=target, diffuser_type=diffuse.Diffuser.ASYNC
        )
        spy_pool_shutdown = mocker.spy(diffuser._worker_pool, "shutdown_async")

        future = await diffuser.diffuse("world")
        await diffuser.close()

        assert diffuser.closed
        assert not future.cancelled()
        assert future.done()
        spy_pool_shutdown.assert_called_once_with(wait=True)

    @pytest.mark.asyncio
    async def test__diffuse__close__no_wait(self, mocker):
        diffuser = diffuse.Diffuser.create(
            target=target_long_running, diffuser_type=diffuse.Diffuser.ASYNC
        )
        spy_pool_shutdown = mocker.spy(diffuser._worker_pool, "shutdown_async")

        future = await diffuser.diffuse("world")
        await diffuser.close(wait=False)

        assert diffuser.closed
        assert not future.cancelled()
        assert not future.done()
        spy_pool_shutdown.assert_called_once_with(wait=False)

    @pytest.mark.asyncio
    async def test__diffuse__close__cancel_pending(self, mocker):
        diffuser = diffuse.Diffuser.create(
            target=target, diffuser_type=diffuse.Diffuser.ASYNC, max_workers=1
        )
        spy_pool_shutdown = mocker.spy(diffuser._worker_pool, "shutdown_async")

        mock_worker = asynctest.Mock(
            diffuse.worker.AsyncWorker(diffuser._task_queue, False)
        )
        diffuser._worker_pool.add(mock_worker)

        future = await diffuser.diffuse("world")
        await diffuser.close(cancel_pending=True)

        assert diffuser.closed
        assert future.cancelled()
        assert future.done()
        assert diffuser._task_queue.qsize() == 0
        spy_pool_shutdown.assert_called_once_with(wait=True)

    @pytest.mark.asyncio
    async def test__diffuse__closed_diffuser(self, mocker):
        diffuser = diffuse.Diffuser.create(
            target=target, diffuser_type=diffuse.Diffuser.ASYNC
        )

        await diffuser.close()

        with pytest.raises(
            RuntimeError, match="Cannot diffuse on closed Diffuser."
        ):
            await diffuser.diffuse("world")

        assert diffuser._task_queue.qsize() == 0
        assert diffuser._worker_pool.size == 0
