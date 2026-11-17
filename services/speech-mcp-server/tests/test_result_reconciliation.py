"""Completed runtime tasks remain recoverable while new GPU admission is busy."""
import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from speech_mcp.domain import Failure, new_id
from speech_mcp.worker import Worker
from test_speech import MemoryStore, fixture


class ResultReconciliationTest(unittest.IsolatedAsyncioTestCase):
    def make_worker(self, status="succeeded"):
        _, _, row = fixture()
        row.update(phase="inference", status="running")
        tid = new_id()
        lease = {"lease_id": "original-lease", "fencing_token": 77,
                 "state": "running", "version": 3}
        row["state"].update(runtime_task_id=tid, gpu_scheduler_lease=lease.copy())
        response = {"task_id": tid, "model_manifest_sha256": "a" * 64, "status": status,
                    "outputs": [{"name": name, "size": 1, "sha256": "b" * 64,
                                 "mime": "application/json" if name.endswith("json") else "audio/wav"}
                                for name in ("original.wav", "editing.wav", "report.json")]}
        if status == "failed":
            response["error"] = {"code": "stale_fencing_token"}
        runtime = SimpleNamespace(json=AsyncMock(return_value=response))
        peers = SimpleNamespace(open=AsyncMock(), runtime=runtime)
        scheduler = SimpleNamespace(enabled=True, heartbeat=Mock(return_value=lease.copy()),
                                    acquire=Mock(side_effect=RuntimeError("capacity unavailable")),
                                    release=Mock())
        store = MemoryStore(row)
        store.renew = Mock()
        with patch("speech_mcp.worker.threading.Thread.start"):
            worker = Worker(SimpleNamespace(store=store, peers=peers,
                                            gpu=SimpleNamespace(enabled=False), scheduler=scheduler))
        worker._ensure_gpu = AsyncMock(side_effect=Failure("waiting_gpu_lease", 503, True))
        return worker, row, response

    async def test_completed_result_archives_despite_busy_admission(self):
        worker, row, _ = self.make_worker()
        await worker.process(row)
        self.assertEqual(row["phase"], "archive")
        self.assertTrue(row["state"]["gpu_execution_finished"])
        self.assertEqual(len(row["state"]["archive_entries"]), 4)
        worker._ensure_gpu.assert_not_awaited()
        worker.scheduler.acquire.assert_not_called()
        worker.scheduler.heartbeat.assert_not_called()
        self.assertEqual(worker.peers.runtime.json.await_args.args[1:3],
                         ("GET", "/v1/tasks/" + row["state"]["runtime_task_id"]))

    async def test_running_task_renews_original_lease_without_resubmission(self):
        worker, row, _ = self.make_worker("running")
        await worker.process(row)
        self.assertEqual(row["phase"], "inference")
        worker.scheduler.heartbeat.assert_called_once_with("original-lease", 77)
        worker.scheduler.acquire.assert_not_called()
        worker._ensure_gpu.assert_not_awaited()
        self.assertEqual(worker.peers.runtime.json.await_count, 1)
        self.assertNotIn("gpu_attempt_seq", row["state"])

    async def test_failed_receiver_is_not_hidden_by_admission_wait(self):
        worker, row, _ = self.make_worker("failed")
        await worker.process(row)
        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["last_error"], "stale_fencing_token")
        worker.scheduler.acquire.assert_not_called()

    async def test_result_identity_mismatch_is_rejected_before_reclaim(self):
        for field, value in (("task_id", new_id()), ("model_manifest_sha256", "c" * 64)):
            with self.subTest(field=field):
                worker, row, response = self.make_worker()
                response[field] = value
                await worker.process(row)
                self.assertEqual(row["status"], "failed")
                self.assertEqual(row["last_error"], "runtime_identity_mismatch")
                worker.scheduler.release.assert_not_called()
                self.assertNotIn("gpu_execution_finished", row["state"])

    async def test_unavailable_result_retries_original_task_without_dispatch(self):
        worker, row, _ = self.make_worker()
        tid = row["state"]["runtime_task_id"]
        worker.peers.runtime.json.side_effect = Failure("runtime_unavailable", 503, True)
        await worker.process(row)
        self.assertEqual((row["status"], row["phase"]), ("running", "inference"))
        self.assertEqual(row["state"]["runtime_task_id"], tid)
        self.assertNotIn("gpu_execution_finished", row["state"])
        worker.scheduler.release.assert_not_called()
        worker.scheduler.acquire.assert_not_called()

    async def test_result_read_heartbeat_keeps_db_claim_without_gpu_gate(self):
        for phase in ("inference", "archive"):
            with self.subTest(phase=phase):
                worker, row, _ = self.make_worker()
                row["phase"] = phase
                worker.scheduler.heartbeat.side_effect = RuntimeError("lease unavailable")
                with patch("speech_mcp.worker.asyncio.sleep", new=AsyncMock(
                        side_effect=[None, asyncio.CancelledError()])):
                    with self.assertRaises(asyncio.CancelledError):
                        await worker.renew(row)
                worker.store.renew.assert_called_once_with(row)
                worker.scheduler.heartbeat.assert_not_called()

    async def test_new_submission_still_requires_admission(self):
        worker, row, _ = self.make_worker()
        row["phase"] = "submit"
        await worker.process(row)
        self.assertEqual(row["phase"], "submit")
        self.assertEqual(row["last_error"], "waiting_gpu_lease")
        worker._ensure_gpu.assert_awaited_once()
        worker.peers.runtime.json.assert_not_awaited()
