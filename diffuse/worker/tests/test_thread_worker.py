import queue

import pytest

from diffuse import worker


class TestThreadWorker:
    def test__worker__process_task(self, mocker):
        task_queue = queue.Queue()
        mock_task = mocker.MagicMock()
        mock_task.run.return_value = "testing"

        thread_worker = worker.ThreadWorker(
            task_queue=task_queue, ephemeral=False
        )
        spy_worker_process_result = mocker.spy(thread_worker, "_process_result")

        thread_worker.start()
        assert thread_worker.is_running()

        task_queue.put_nowait(mock_task)
        thread_worker.stop()
        thread_worker.wait()

        assert not thread_worker.is_running()
        mock_task.run.assert_called_once()
        spy_worker_process_result.assert_called_once_with("testing")

    def test__worker__stop(self, mocker):
        task_queue = queue.Queue()

        thread_worker = worker.ThreadWorker(
            task_queue=task_queue, ephemeral=False
        )
        spy_worker_process_result = mocker.spy(thread_worker, "_process_result")

        thread_worker.start()
        assert thread_worker.is_running()

        thread_worker.stop()
        thread_worker.wait()

        assert not thread_worker.is_running()
        spy_worker_process_result.assert_not_called()

    def test__ephemeral_worker__process_task(self, mocker):
        task_queue = queue.Queue()
        mock_task = mocker.MagicMock()
        mock_task.run.return_value = "testing"
        task_queue.put_nowait(mock_task)

        thread_worker = worker.ThreadWorker(
            task_queue=task_queue, ephemeral=True
        )
        spy_worker_process_result = mocker.spy(thread_worker, "_process_result")

        thread_worker.start()
        thread_worker.wait()

        assert not thread_worker.is_running()
        mock_task.run.assert_called_once()
        spy_worker_process_result.assert_called_once_with("testing")
