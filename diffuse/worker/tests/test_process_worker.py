import multiprocessing

import pytest

from diffuse import worker


class TaskMock:
    def run(self):
        return "testing"


class TestProcessWorker:
    def test__worker__process_task(self, mocker):
        task_queue = multiprocessing.Queue()
        result_queue = multiprocessing.Queue()

        mock_task = TaskMock()

        process_worker = worker.ProcessWorker(
            task_queue=task_queue, ephemeral=False, result_queue=result_queue
        )

        process_worker.start()
        assert process_worker.is_running()

        task_queue.put_nowait(mock_task)
        process_worker.stop()
        process_worker.wait()

        assert not process_worker.is_running()

        assert result_queue.qsize() == 1
        result = result_queue.get()
        assert result == "testing"

    def test__worker__stop(self, mocker):
        task_queue = multiprocessing.Queue()
        result_queue = multiprocessing.Queue()

        process_worker = worker.ProcessWorker(
            task_queue=task_queue, ephemeral=False, result_queue=result_queue
        )
        spy_worker_process_result = mocker.spy(
            process_worker, "_process_result"
        )

        process_worker.start()
        assert process_worker.is_running()

        process_worker.stop()
        process_worker.wait()

        assert not process_worker.is_running()
        assert result_queue.qsize() == 0
        spy_worker_process_result.assert_not_called()

    def test__ephemeral_worker__process_task(self, mocker):
        task_queue = multiprocessing.Queue()
        result_queue = multiprocessing.Queue()

        mock_task = TaskMock()
        task_queue.put_nowait(mock_task)

        process_worker = worker.ProcessWorker(
            task_queue=task_queue, ephemeral=True, result_queue=result_queue
        )

        process_worker.start()
        process_worker.wait()

        assert not process_worker.is_running()

        assert result_queue.qsize() == 1
        result = result_queue.get()
        assert result == "testing"
