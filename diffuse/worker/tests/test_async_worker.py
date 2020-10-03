import asyncio

import asynctest
import pytest

from diffuse import worker


class TestAsyncWorker:
    @pytest.mark.asyncio
    async def test__worker__process_task(self, mocker):
        task_queue = asyncio.Queue()
        mock_task = mocker.MagicMock()
        mock_task.run = asynctest.CoroutineMock(return_value="testing")

        async_worker = worker.AsyncWorker(
            task_queue=task_queue, ephemeral=False
        )
        spy_worker_process_result = mocker.spy(async_worker, "_process_result")

        asyncio.ensure_future(async_worker.start())
        assert async_worker.is_running()

        await task_queue.put(mock_task)

        async_worker.stop()
        await async_worker.wait()

        assert not async_worker.is_running()
        mock_task.run.assert_called_once()
        spy_worker_process_result.assert_called_once_with("testing")

    @pytest.mark.asyncio
    async def test__worker__stop(self, mocker):
        task_queue = asyncio.Queue()

        async_worker = worker.AsyncWorker(
            task_queue=task_queue, ephemeral=False
        )
        spy_worker_process_result = mocker.spy(async_worker, "_process_result")

        asyncio.ensure_future(async_worker.start())
        assert async_worker.is_running()

        async_worker.stop()
        await async_worker.wait()

        assert not async_worker.is_running()
        spy_worker_process_result.assert_not_called()

    @pytest.mark.asyncio
    async def test__ephemeral_worker__process_task(self, mocker):
        task_queue = asyncio.Queue()
        mock_task = mocker.MagicMock()
        mock_task.run = asynctest.CoroutineMock(return_value="testing")

        await task_queue.put(mock_task)

        async_worker = worker.AsyncWorker(task_queue=task_queue, ephemeral=True)
        spy_worker_process_result = mocker.spy(async_worker, "_process_result")

        asyncio.ensure_future(async_worker.start())
        await async_worker.wait()

        assert not async_worker.is_running()
        mock_task.run.assert_called_once()
        spy_worker_process_result.assert_called_once_with("testing")
