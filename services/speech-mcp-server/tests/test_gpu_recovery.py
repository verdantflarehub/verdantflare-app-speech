import copy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx

from speech_mcp.domain import Failure
from speech_mcp.gpu_lease import GpuCoordinator, K8sApi, MemoryApi
from speech_mcp.worker import Worker
from test_speech import MemoryStore, fixture


class ClaimStore(MemoryStore):
    def __init__(self, row):
        super().__init__(copy.deepcopy(row))
        self.claim = row["lease_token"]
        self.valid = True

    def renew(self, row):
        if not self.valid or row["lease_token"] != self.claim:
            raise Failure("lease_lost", 409, True)

    def checkpoint(self, row, phase=None):
        self.renew(row)
        super().checkpoint(row, phase)

    def save(self, row, **kwargs):
        self.renew(row)
        super().save(row, **kwargs)
        self.valid = False


class RecoveryTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        _, _, self.row = fixture()
        self.store = ClaimStore(self.row)
        self.api = MemoryApi()
        self.t = 0.0
        self.gpu = GpuCoordinator(self.api, "indextts2", "speech/test", deployment="speech",
            enabled=True, queue_seconds=1, ready_seconds=0.1, idle_seconds=600,
            now=lambda: datetime(2026, 11, 3, tzinfo=timezone.utc) + timedelta(seconds=self.t),
            sleep=lambda seconds: setattr(self, "t", self.t + seconds), monotonic=lambda: self.t)
        self.peers = SimpleNamespace(runtime=SimpleNamespace(json=AsyncMock(return_value={"ready": True})),
                                     open=AsyncMock())
        with patch("speech_mcp.worker.threading.Thread.start"):
            self.worker = Worker(SimpleNamespace(store=self.store, peers=self.peers, gpu=self.gpu))

    def holder(self):
        return ((self.api.get_lease(self.gpu.lease_name) or {}).get("spec") or {}).get("holderIdentity", "")

    def persist_held(self):
        self.row["state"]["gpu_held"] = True
        self.store.checkpoint(self.row)

    async def test_loading_checkpoint_retains_claim_and_input_identity(self):
        identity = (self.row["task_id"], self.row["request"]["idempotency_key"])
        await self.worker._ensure_gpu(self.row)
        self.store.renew(self.row)
        self.assertEqual(self.store.row["phase"], "loading_model")
        self.assertTrue(self.store.gpu_active())
        self.assertEqual(identity, (self.row["task_id"], self.row["request"]["idempotency_key"]))

    async def test_restart_held_flag_cannot_bypass_music_admission(self):
        self.persist_held()
        music = GpuCoordinator(self.api, "music3", "music/test", enabled=True,
                               now=self.gpu._now)
        self.assertTrue(music.try_acquire())
        with self.assertRaisesRegex(Failure, "waiting_gpu_lease"):
            await self.worker._ensure_gpu(self.row)
        self.assertEqual(self.api.get_scale("speech"), 0)
        self.peers.runtime.json.assert_not_awaited()
        self.assertEqual(self.holder(), "")

    async def test_restart_reacquires_and_verifies_backend_without_inference(self):
        self.persist_held()
        await self.worker._ensure_gpu(self.row)
        self.assertEqual(self.holder(), "speech/test")
        self.assertEqual(self.api.get_scale("speech"), 1)
        self.assertEqual([call.args[1:3] for call in self.peers.runtime.json.await_args_list], [("GET", "/health")])

    async def test_repeated_stages_do_not_accumulate_memory_counts(self):
        await self.worker._ensure_gpu(self.row)
        self.row["phase"] = "submit"
        await self.worker._ensure_gpu(self.row)
        self.assertEqual(self.gpu.in_flight, 0)
        self.assertEqual(self.holder(), "speech/test")

    async def test_retry_after_load_timeout_does_not_skip_readiness(self):
        self.gpu.ready_seconds = 0
        for _ in range(2):
            with self.assertRaisesRegex(Failure, "model_load_timeout"):
                await self.worker._ensure_gpu(self.row)

    async def test_lost_claim_blocks_before_scaling_or_runtime_calls(self):
        self.persist_held()
        self.store.valid = False
        with self.assertRaisesRegex(Failure, "lease_lost"):
            await self.worker._ensure_gpu(self.row)
        self.assertEqual(self.api.get_scale("speech"), 0)
        self.peers.runtime.json.assert_not_awaited()

    async def test_unclaimed_persisted_task_prevents_idle_scale_down(self):
        self.persist_held()
        self.api.patch_scale("speech", 1)
        self.gpu.idle_at = 0
        self.t = 1000
        self.gpu.reconcile()
        self.assertEqual(self.api.get_scale("speech"), 1)
        self.assertEqual(self.holder(), "speech/test")

    async def test_store_outage_does_not_imply_idle(self):
        await self.worker._ensure_gpu(self.row)
        self.gpu.activity = Mock(side_effect=OSError("test database unavailable"))
        with self.assertRaises(OSError):
            self.gpu.reconcile()
        self.assertEqual(self.api.get_scale("speech"), 1)
        self.assertEqual(self.holder(), "speech/test")

    async def test_pre_submission_failure_persists_release_before_handoff(self):
        self.persist_held()
        self.gpu.try_acquire()
        self.worker.step = AsyncMock(side_effect=Failure("model_version_changed", 409))
        await self.worker.process(self.row)
        self.assertFalse(self.store.row["state"].get("gpu_held"))
        self.assertFalse(self.store.gpu_active())
        self.assertEqual(self.holder(), "")

    async def test_unconfirmed_inference_failure_keeps_occupancy(self):
        self.row["phase"] = "inference"
        self.persist_held()
        self.gpu.try_acquire()
        self.worker.step = AsyncMock(side_effect=Failure("runtime_identity_mismatch", 502))
        await self.worker.process(self.row)
        self.assertTrue(self.store.gpu_active())
        self.assertEqual(self.holder(), "speech/test")

    async def test_verified_terminal_failure_releases_occupancy(self):
        self.row["phase"] = "inference"
        self.row["state"]["gpu_execution_finished"] = True
        self.persist_held()
        self.gpu.try_acquire()
        self.worker.step = AsyncMock(side_effect=Failure("inference_failed", 409))
        await self.worker.process(self.row)
        self.assertFalse(self.store.gpu_active())
        self.assertEqual(self.holder(), "")

    async def test_archive_completion_persists_before_release_without_wake(self):
        self.row["phase"] = "archive"
        self.row["state"]["archive_entries"] = []
        self.persist_held()
        self.gpu.try_acquire()
        await self.worker.step(self.row)
        self.assertEqual(self.store.row["status"], "succeeded")
        self.assertFalse(self.store.gpu_active())
        self.assertEqual(self.holder(), "")
        self.peers.runtime.json.assert_not_awaited()

    async def test_idle_shutdown_waits_for_durable_completion_and_interval(self):
        await self.worker._ensure_gpu(self.row)
        self.row["state"].pop("gpu_held")
        self.store.save(self.row, status="succeeded", phase="complete")
        self.gpu.reconcile()
        self.assertEqual(self.api.get_scale("speech"), 1)
        self.t = 601
        self.gpu.reconcile()
        self.assertEqual(self.api.get_scale("speech"), 0)

    async def test_idle_process_cannot_stop_a_new_holder(self):
        self.api.patch_scale("speech", 1)
        self.gpu.idle_at = 0
        self.t = 601
        peer = GpuCoordinator(self.api, "indextts2", "speech/other", enabled=True, now=self.gpu._now)
        self.assertTrue(peer.try_acquire())
        self.gpu.reconcile()
        self.assertEqual(self.api.get_scale("speech"), 1)
        self.assertEqual(self.holder(), "speech/other")

    async def test_lease_creation_conflict_does_not_overwrite_winner(self):
        api, methods = K8sApi("test"), []
        def request(method, path, **kwargs):
            methods.append(method)
            return httpx.Response(409 if method == "POST" else 200,
                json={"metadata": {"resourceVersion": "1"}, "spec": {"holderIdentity": "winner"}},
                request=httpx.Request(method, "http://test" + path))
        api._request = request
        self.assertEqual(api.replace_lease("gpu-task-indextts2", {"holderIdentity": "loser"}, None), 409)
        self.assertEqual(methods, ["POST"])


if __name__ == "__main__":
    unittest.main()
