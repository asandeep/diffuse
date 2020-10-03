import functools
import gc
from unittest import mock

import asynctest
import pytest

from diffuse import pool


class TestWorkerPoolSize:
    def test__size__no_workers(self):
        worker_pool = pool.WorkerPool()

        assert worker_pool.size == 0

    def test__size__worker__state_change(self):
        mock_worker = mock.MagicMock()
        mock_worker.is_running.side_effect = [True, False]

        worker_pool = pool.WorkerPool()
        worker_pool.add(mock_worker)

        assert worker_pool.size == 1
        assert worker_pool.size == 0

        assert mock_worker.is_running.call_count == 2

    def test__size__worker__no_strong_reference(self):
        mock_worker = mock.MagicMock()
        mock_worker.is_running.return_value = True

        worker_pool = pool.WorkerPool()
        worker_pool.add(mock_worker)

        assert worker_pool.size == 1

        del mock_worker
        gc.collect()
        assert worker_pool.size == 0


class TestWorkerPoolShutdown:
    @pytest.fixture
    def make_worker(self):
        def _mock_worker_stop(mock_worker):
            mock_worker.is_running.return_value = False

        def _make_worker(coro=False):
            mock_worker = mock.MagicMock()
            mock_worker.stop.side_effect = functools.partial(
                _mock_worker_stop, mock_worker
            )

            if coro:
                mock_worker.wait = asynctest.CoroutineMock()

            return mock_worker

        return _make_worker

    def test__shutdown(self, make_worker):
        mock_worker_1 = make_worker()
        mock_worker_2 = make_worker()

        worker_pool = pool.WorkerPool()
        worker_pool.add(mock_worker_1)
        worker_pool.add(mock_worker_2)

        worker_pool.shutdown(wait=True)

        assert worker_pool.size == 0
        mock_worker_1.stop.assert_called_once()
        mock_worker_1.wait.assert_called_once()
        mock_worker_2.stop.assert_called_once()
        mock_worker_2.wait.assert_called_once()

    def test__shutdown__no_wait(self, make_worker):
        mock_worker_1 = make_worker()
        mock_worker_2 = make_worker()

        worker_pool = pool.WorkerPool()
        worker_pool.add(mock_worker_1)
        worker_pool.add(mock_worker_2)

        worker_pool.shutdown(wait=False)

        assert worker_pool.size == 0
        mock_worker_1.stop.assert_called_once()
        mock_worker_1.wait.assert_not_called()
        mock_worker_2.stop.assert_called_once()
        mock_worker_2.wait.assert_not_called()

    @pytest.mark.asyncio
    async def test__shutdown_async(self, make_worker):
        mock_worker_1 = make_worker(coro=True)
        mock_worker_2 = make_worker(coro=True)

        worker_pool = pool.WorkerPool()
        worker_pool.add(mock_worker_1)
        worker_pool.add(mock_worker_2)

        await worker_pool.shutdown_async(wait=True)

        assert worker_pool.size == 0
        mock_worker_1.stop.assert_called_once()
        mock_worker_1.wait.assert_called_once()
        mock_worker_2.stop.assert_called_once()
        mock_worker_2.wait.assert_called_once()

    @pytest.mark.asyncio
    async def test__shutdown_async__no_wait(self, make_worker):
        mock_worker_1 = make_worker(coro=True)
        mock_worker_2 = make_worker(coro=True)

        worker_pool = pool.WorkerPool()
        worker_pool.add(mock_worker_1)
        worker_pool.add(mock_worker_2)

        await worker_pool.shutdown_async(wait=False)

        assert worker_pool.size == 0
        mock_worker_1.stop.assert_called_once()
        mock_worker_1.wait.assert_not_called()
        mock_worker_2.stop.assert_called_once()
        mock_worker_2.wait.assert_not_called()
