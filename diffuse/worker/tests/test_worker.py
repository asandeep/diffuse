import queue
from diffuse.worker import thread
import pytest


class TestThreadWorker:
    def test__worker__process_task(self, mocker):
        task_queue = queue.Queue()
        mock_task = mocker.MagicMock()
        mock_task.run.return_value = "testing"

        worker = thread.ThreadWorker(queue=task_queue, ephemeral=False)
        spy_worker_process_result = mocker.spy(worker, "_process_result")

        worker.start()
        assert worker.is_alive()

        task_queue.put_nowait(mock_task)
        worker.stop()
        worker.join()

        assert not worker.is_alive()
        mock_task.run.assert_called_once()
        spy_worker_process_result.assert_called_once_with("testing")

    def test__worker__stop(self, mocker):
        task_queue = queue.Queue()

        worker = thread.ThreadWorker(queue=task_queue, ephemeral=False)
        spy_worker_process_result = mocker.spy(worker, "_process_result")

        worker.start()
        assert worker.is_alive()

        worker.stop()
        worker.join()

        assert not worker.is_alive()
        spy_worker_process_result.assert_not_called()

    def test__ephemeral_worker__process_task(self, mocker):
        task_queue = queue.Queue()
        mock_task = mocker.MagicMock()
        mock_task.run.return_value = "testing"
        task_queue.put_nowait(mock_task)

        worker = thread.ThreadWorker(queue=task_queue, ephemeral=True)
        spy_worker_process_result = mocker.spy(worker, "_process_result")

        worker.start()

        assert not worker.is_alive()
        mock_task.run.assert_called_once()
        spy_worker_process_result.assert_called_once_with("testing")
