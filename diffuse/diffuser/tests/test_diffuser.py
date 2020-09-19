import diffuse
import pytest
import os
from contextlib import contextmanager
from diffuse.diffuser.base import pool
from unittest import mock
from concurrent import futures


@contextmanager
def noop():
    yield


class TestThreadDiffuser:
    def test__default_thread_diffuser(self):
        default_diffuser = diffuse.Diffuser.create(target=lambda: "test")
        assert isinstance(default_diffuser, diffuse.Diffuser.THREAD.value)

    def test__target_not_callable(self):
        with pytest.raises(TypeError, match="target must be a callable."):
            diffuse.Diffuser.create(target=None)

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
    def test__max_workers(self, input, expected, might_raise):
        with might_raise:
            diffuser = diffuse.Diffuser.create(
                target=lambda: "test", max_workers=input
            )
            assert diffuser.max_workers == expected

    def test__diffuse(self, mocker):
        diffuser = diffuse.Diffuser.create(target=lambda: "hello")
        spy_pool_add = mocker.spy(diffuser._worker_pool, "add")
        mock_thread_worker = mocker.patch.object(diffuser, "_WORKER_CLASS")

        future = diffuser.diffuse("world")
        assert isinstance(future, futures.Future)
        assert not future.running()
        assert not future.done()

        assert diffuser.task_queue.qsize() == 1

        mock_thread_worker.assert_called_once_with(diffuser.task_queue, False)

        mock_worker_instance = mock_thread_worker.return_value
        spy_pool_add.assert_called_once_with(mock_worker_instance)
        mock_worker_instance.start.assert_called_once_with()

    def test__diffuse__task_consumed_by_worker(self, mocker):
        diffuser = diffuse.Diffuser.create(target=lambda: "hello")
        mocker.patch.object(diffuser.task_queue, "qsize", return_value=0)

        spy_pool_add = mocker.spy(diffuser._worker_pool, "add")
        mock_thread_worker = mocker.patch.object(diffuser, "_WORKER_CLASS")

        future = diffuser.diffuse("world")
        assert isinstance(future, futures.Future)
        assert not future.running()
        assert not future.done()

        mock_thread_worker.assert_not_called()
        spy_pool_add.assert_not_called()

    def test__diffuse__max_pool_size(self, mocker):
        diffuser = diffuse.Diffuser.create(
            target=lambda: "hello", max_workers=1
        )
        mock_worker = mocker.MagicMock()
        diffuser._worker_pool.add(mock_worker)

        spy_pool_add = mocker.spy(diffuser._worker_pool, "add")
        mock_thread_worker = mocker.patch.object(diffuser, "_WORKER_CLASS")

        future = diffuser.diffuse("world")
        assert isinstance(future, futures.Future)
        assert not future.running()
        assert not future.done()

        mock_thread_worker.assert_not_called()
        spy_pool_add.assert_not_called()

    def test__diffuse__close(self, mocker):
        diffuser = diffuse.Diffuser.create(target=lambda: "hello")
        spy_pool_shutdown = mocker.spy(diffuser._worker_pool, "shutdown")
        mock_thread_worker = mocker.patch.object(diffuser, "_WORKER_CLASS")

        future = diffuser.diffuse("world")
        diffuser.close()

        assert diffuser.closed
        assert not future.cancelled()
        spy_pool_shutdown.assert_called_once_with(wait=True)

    def test__diffuse__close__no_wait(self, mocker):
        diffuser = diffuse.Diffuser.create(target=lambda: "hello")
        spy_pool_shutdown = mocker.spy(diffuser._worker_pool, "shutdown")
        mock_thread_worker = mocker.patch.object(diffuser, "_WORKER_CLASS")

        future = diffuser.diffuse("world")
        diffuser.close(wait=False)

        assert diffuser.closed
        assert not future.cancelled()
        spy_pool_shutdown.assert_called_once_with(wait=False)

    def test__diffuse__close__cancel_pending(self, mocker):
        diffuser = diffuse.Diffuser.create(target=lambda: "hello")
        spy_pool_shutdown = mocker.spy(diffuser._worker_pool, "shutdown")
        mock_thread_worker = mocker.patch.object(diffuser, "_WORKER_CLASS")

        future = diffuser.diffuse("world")
        diffuser.close(cancel_pending=True)

        assert diffuser.closed
        assert future.cancelled()
        spy_pool_shutdown.assert_called_once_with(wait=True)

    def test__diffuse__closed_diffuser(self, mocker):
        diffuser = diffuse.Diffuser.create(target=lambda: "hello")
        spy_pool_add = mocker.spy(diffuser._worker_pool, "add")
        mock_thread_worker = mocker.patch.object(diffuser, "_WORKER_CLASS")

        diffuser.close()

        with pytest.raises(
            RuntimeError, match="Cannot diffuse on closed Diffuser."
        ):
            diffuser.diffuse("world")

        assert diffuser.task_queue.qsize() == 0
        mock_thread_worker.assert_not_called()
        spy_pool_add.assert_not_called()
