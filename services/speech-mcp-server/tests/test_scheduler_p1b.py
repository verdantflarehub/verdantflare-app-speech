"""P1b tests for speech-mcp-server :5056 SchedulerLeaseClient, bind-target, lock order, and receiver-guard warm release."""
from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from speech_mcp.domain import Failure
from speech_mcp.gpu_lease import GpuCoordinator, MemoryApi, SchedulerError, SchedulerLeaseClient
from speech_mcp.worker import Worker
from test_speech import MemoryStore, fixture


class SpeechSchedulerP1bTest(unittest.IsolatedAsyncioTestCase):
    async def test_ensure_gpu_acquires_slot_binds_target_and_releases_with_receiver_proof(self) -> None:
        _, _, row = fixture()
        row = copy.deepcopy(row)
        store = MemoryStore(row)
        api = MemoryApi()
        gpu = GpuCoordinator(
            api,
            "indextts2",
            "speech/test",
            deployment="speech-indextts2",
            enabled=True,
            queue_seconds=5,
            ready_seconds=2,
            idle_seconds=600,
            now=lambda: datetime(2026, 11, 15, tzinfo=timezone.utc) + timedelta(seconds=1),
        )

        sched_calls: list[tuple[str, str, dict]] = []

        def sched_handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content.decode("utf-8")) if request.content else {}
            sched_calls.append((request.method, request.url.path, body))
            if request.url.path.endswith("/acquire"):
                # Verify deployment has NOT been scaled to 1 before :5056 acquire AND K8s lease NOT yet acquired!
                self.assertEqual(api.get_scale("speech-indextts2"), 0)
                self.assertIsNone(api.get_lease("gpu-task-indextts2"))
                return httpx.Response(
                    200,
                    json={
                        "lease_id": "lease-speech-99",
                        "fencing_token": 77,
                        "version": 1,
                        "state": "leased",
                        "attempt_id": body["attempt_id"],
                        "stage_id": body["stage_id"],
                        "gpu_uuids": ["GPU-11111111-2222-3333-4444-555555555555"],
                        "expires_at": "2099-01-01T00:00:00Z",
                    },
                )
            if request.url.path.endswith("/bind-target"):
                return httpx.Response(
                    200,
                    json={
                        "lease_id": "lease-speech-99",
                        "fencing_token": 77,
                        "version": 2,
                        "state": "running",
                        "attempt_id": "att-1",
                        "stage_id": "indextts2_inference",
                        "pod_uid": body["pod_uid"],
                        "container_id": body["container_id"],
                        "boot_id": body["boot_id"],
                        "gpu_uuids": body.get("actual_gpu_uuids") or ["GPU-11111111-2222-3333-4444-555555555555"],
                        "expires_at": "2099-01-01T00:00:00Z",
                    },
                )
            if request.url.path.endswith("/release"):
                return httpx.Response(200, json={"lease_id": "lease-speech-99", "state": "released"})
            return httpx.Response(200, json={"lease_id": "lease-speech-99", "fencing_token": 77, "state": "running"})

        sched_http = httpx.Client(transport=httpx.MockTransport(sched_handler))
        self.addCleanup(sched_http.close)
        scheduler = SchedulerLeaseClient(
            base_url="http://station-runtime:5056",
            token="sched-token",
            enabled=True,
            client=sched_http,
        )

        runtime_calls: list[tuple[str, str, dict]] = []

        async def runtime_json(p, method, path, payload=None, **kwargs):
            runtime_calls.append((method, path, kwargs))
            if path == "/health":
                return {
                    "ready": True,
                    "runtime_identity": {
                        "pod_uid": "pod-indextts2-1",
                        "container_id": "cid-indextts2-1",
                        "boot_id": "boot-node-1",
                        "supervisor_pid": 6100,
                        "supervisor_starttime_ticks": 71000,
                        "visible_gpu_uuids": ["GPU-11111111-2222-3333-4444-555555555555"],
                    },
                }
            if path == "/v1/capabilities":
                return {"model_manifest_sha256": row["state"]["profile"]["model_manifest_sha256"], "ready": True}
            if path == "/v1/tasks":
                return {"task_id": row["task_id"], "status": "pending"}
            if path.startswith("/v1/tasks/"):
                return {
                    "task_id": row["task_id"],
                    "model_manifest_sha256": row["state"]["profile"]["model_manifest_sha256"],
                    "status": "failed",
                    "error": {"code": "stale_fencing_token"},
                    "reclaim_proof": {
                        "reclaimed": True,
                        "active_tasks": 0,
                        "proof_source": "receiver_guard",
                        "token_sealed": True,
                        "supervisor_pid": 6100,
                        "supervisor_starttime_ticks": 71000,
                        "sealed_pod_uid": "pod-indextts2-1",
                        "sealed_container_id": "cid-indextts2-1",
                        "sealed_boot_id": "boot-node-1",
                        "revoked_lease_id": "lease-speech-99",
                        "consumed_fencing_token": 77,
                    },
                }
            raise AssertionError(f"unexpected path {path}")

        peers = SimpleNamespace(runtime=SimpleNamespace(json=AsyncMock(side_effect=runtime_json)), open=AsyncMock())
        with patch("speech_mcp.worker.threading.Thread.start"):
            worker = Worker(SimpleNamespace(store=store, peers=peers, gpu=gpu, scheduler=scheduler))

        row["phase"] = "submit"
        row["state"]["runtime_inputs"] = {
            "idempotency_key": "speech-1",
            "text": "你好",
            "language": "zh-CN",
            "reference_sha256": "a" * 64,
        }
        await worker.step(row)

        acq = [c for c in sched_calls if c[1] == "/internal/v1/gpu-scheduler/leases/acquire"]
        binds = [c for c in sched_calls if c[1] == "/internal/v1/gpu-scheduler/leases/lease-speech-99/bind-target"]
        self.assertEqual(len(acq), 1)
        self.assertEqual(acq[0][2]["k8s_slots_required"], 1)
        self.assertEqual(acq[0][2]["peak_budget_mib"], 6144)
        self.assertEqual(len(binds), 1)
        self.assertEqual(binds[0][2]["pod_uid"], "pod-indextts2-1")
        self.assertEqual(api.get_scale("speech-indextts2"), 1)

        submit_call = next(c for c in runtime_calls if c[0] == "POST" and c[1] == "/v1/tasks")
        hdrs = submit_call[2]["extra_headers"]
        self.assertEqual(hdrs["X-VF-GPU-Lease-ID"], "lease-speech-99")
        self.assertEqual(hdrs["X-VF-GPU-Fencing-Token"], "77")
        self.assertEqual(hdrs["X-VF-GPU-Pod-UID"], "pod-indextts2-1")
        self.assertEqual(hdrs["X-VF-GPU-Boot-ID"], "boot-node-1")

        # Inference failure with receiver_guard reclaim_proof releases :5056 lease as warm_idle_settled
        row["phase"] = "inference"
        with self.assertRaises(Exception):
            await worker.step(row)
        rel = [c for c in sched_calls if c[1] == "/internal/v1/gpu-scheduler/leases/lease-speech-99/release"]
        self.assertEqual(len(rel), 1)
        self.assertTrue(rel[0][2]["worker_settled"])
        self.assertEqual(rel[0][2]["reclaim_mode"], "warm_idle_settled")
        self.assertEqual(rel[0][2]["pod_uid"], "pod-indextts2-1")

    async def test_k8s_lease_contention_rolls_back_scheduler_with_abort_unstarted_and_missing_proof_never_fakes_settled(self) -> None:
        _, _, row = fixture()
        row = copy.deepcopy(row)
        store = MemoryStore(row)
        api = MemoryApi()
        now_dt = datetime(2026, 11, 15, tzinfo=timezone.utc)
        # Pre-hold conflicting music3 K8s Lease
        api.replace_lease(
            "gpu-task-music3",
            {
                "holderIdentity": "music3/peer-pod",
                "leaseDurationSeconds": 90,
                "acquireTime": "2026-11-15T00:00:00.000000Z",
                "renewTime": "2026-11-15T00:00:00.000000Z",
            },
            None,
        )
        gpu = GpuCoordinator(
            api,
            "indextts2",
            "speech/test",
            deployment="speech-indextts2",
            enabled=True,
            now=lambda: now_dt + timedelta(seconds=5),
        )

        sched_calls: list[tuple[str, str, dict]] = []

        def sched_handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content.decode("utf-8")) if request.content else {}
            sched_calls.append((request.method, request.url.path, body))
            if request.url.path.endswith("/acquire"):
                return httpx.Response(
                    200,
                    json={
                        "lease_id": "lease-speech-contended",
                        "fencing_token": 88,
                        "version": 1,
                        "state": "leased",
                        "attempt_id": body["attempt_id"],
                        "stage_id": body["stage_id"],
                        "gpu_uuids": ["GPU-11111111-2222-3333-4444-555555555555"],
                        "expires_at": "2099-01-01T00:00:00Z",
                    },
                )
            if request.url.path.endswith("/release"):
                lid = request.url.path.split("/")[-2]
                return httpx.Response(200, json={"lease_id": lid, "state": "released"})
            return httpx.Response(404, json={"error_code": "LEASE_NOT_FOUND", "message": "not found"})

        sched_http = httpx.Client(transport=httpx.MockTransport(sched_handler))
        self.addCleanup(sched_http.close)
        scheduler = SchedulerLeaseClient(
            base_url="http://station-runtime:5056",
            token="sched-token",
            enabled=True,
            client=sched_http,
        )
        peers = SimpleNamespace(
            runtime=SimpleNamespace(json=AsyncMock(side_effect=RuntimeError("unreachable"))),
            open=AsyncMock(),
        )
        with patch("speech_mcp.worker.threading.Thread.start"):
            worker = Worker(SimpleNamespace(store=store, peers=peers, gpu=gpu, scheduler=scheduler))

        with self.assertRaises(Failure) as ctx:
            await worker._ensure_gpu(row)
        self.assertEqual(ctx.exception.code, "waiting_gpu_lease")
        rel = [c for c in sched_calls if c[1].endswith("/release")]
        self.assertEqual(len(rel), 1)
        self.assertEqual(rel[0][2]["reclaim_mode"], "abort_unstarted")
        self.assertFalse(rel[0][2]["worker_settled"])

        # Now test bound lease with unreachable receiver reclaim endpoint -> MUST send worker_settled=False, reclaim_mode=""
        sched_calls.clear()
        row["state"]["gpu_scheduler_lease"] = {
            "lease_id": "lease-speech-bound",
            "fencing_token": 89,
            "version": 2,
            "state": "running",
            "pod_uid": "pod-indextts2-1",
        }
        row["state"]["gpu_target_identity"] = {"pod_uid": "pod-indextts2-1"}
        await worker._release_scheduler_lease(row, reason="timeout_no_proof")
        rel2 = [c for c in sched_calls if c[1].endswith("/release")]
        self.assertEqual(len(rel2), 1)
        self.assertFalse(rel2[0][2]["worker_settled"])
        self.assertNotIn("reclaim_mode", rel2[0][2])
        self.assertNotIn("reclaim_proof", rel2[0][2])

    async def test_r2_m1_durable_pending_release_journal_restart_and_unconfirmed_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            journal_path = Path(tmpdir) / "gpu_pending_releases.json"
            mode = {"phase": "transport_fail"}
            flushed_calls: list[tuple[str, dict]] = []

            def handler(request: httpx.Request) -> httpx.Response:
                body = json.loads(request.content.decode("utf-8")) if request.content else {}
                if request.url.path.endswith("/release"):
                    lid = request.url.path.split("/")[-2]
                    flushed_calls.append((lid, body))
                    p = mode["phase"]
                    if p == "transport_fail":
                        raise httpx.ConnectError("network down", request=request)
                    if p == "not_found":
                        return httpx.Response(404, json={"error_code": "LEASE_NOT_FOUND", "message": "unknown"})
                    if p == "wrong_lease":
                        return httpx.Response(200, json={"lease_id": "other-lease", "state": "released"})
                    if p == "non_terminal":
                        return httpx.Response(200, json={"lease_id": lid, "state": "reclaiming"})
                    return httpx.Response(
                        200,
                        json={"lease_id": lid, "state": "expired" if lid.endswith("-b") else "released"},
                    )
                if request.url.path.endswith("/acquire"):
                    return httpx.Response(
                        200,
                        json={
                            "lease_id": "lease-new-1",
                            "fencing_token": 101,
                            "version": 1,
                            "state": "leased",
                            "attempt_id": body["attempt_id"],
                            "stage_id": body["stage_id"],
                        },
                    )
                return httpx.Response(200, json={})

            http_client = httpx.Client(transport=httpx.MockTransport(handler))
            self.addCleanup(http_client.close)

            c1 = SchedulerLeaseClient(
                base_url="http://station-runtime:5056",
                token="sched-token",
                enabled=True,
                client=http_client,
                pending_releases_path=journal_path,
            )

            # 1) Transport error persists retry_body (without expected_version) BEFORE HTTP call
            with self.assertRaises(SchedulerError) as e1:
                c1.release("lease-a", 41, expected_version=7, worker_settled=False, reclaim_mode="abort_unstarted")
            self.assertEqual(e1.exception.code, "RELEASE_TRANSPORT_FAILED")
            saved = json.loads(journal_path.read_text(encoding="utf-8"))
            self.assertIn("lease-a", saved)
            self.assertNotIn("expected_version", saved["lease-a"])

            # 2) 404, wrong lease_id, and state="reclaiming" MUST retain pending journal entry
            for bad_phase in ("not_found", "wrong_lease", "non_terminal"):
                mode["phase"] = bad_phase
                with self.assertRaises(SchedulerError):
                    c1.release("lease-b", 42, expected_version=3, worker_settled=False, reclaim_mode="abort_unstarted")
                saved_bad = json.loads(journal_path.read_text(encoding="utf-8"))
                self.assertIn("lease-a", saved_bad)
                self.assertIn("lease-b", saved_bad)

            # 3) Simulate process restart -> new client reloads journal and flushes BOTH entries on acquire()
            mode["phase"] = "ok"
            flushed_calls.clear()
            c2 = SchedulerLeaseClient(
                base_url="http://station-runtime:5056",
                token="sched-token",
                enabled=True,
                client=http_client,
                pending_releases_path=journal_path,
            )
            acq_res = c2.acquire(parent_task_id="task-speech-1", attempt_id="att-1")
            self.assertEqual(acq_res["lease_id"], "lease-new-1")
            self.assertEqual({lid for lid, _ in flushed_calls}, {"lease-a", "lease-b"})
            for _, rbody in flushed_calls:
                self.assertNotIn("expected_version", rbody)
            self.assertEqual(json.loads(journal_path.read_text(encoding="utf-8")), {})

            # 4) Corrupted journal file fails closed on startup
            journal_path.write_text("{broken-json", encoding="utf-8")
            with self.assertRaises(SchedulerError) as e_corrupt:
                SchedulerLeaseClient(
                    base_url="http://station-runtime:5056",
                    token="sched-token",
                    enabled=True,
                    client=http_client,
                    pending_releases_path=journal_path,
                )
            self.assertEqual(e_corrupt.exception.code, "JOURNAL_CORRUPTED")

    async def test_r3_m1_m2_quarantine_400_and_postgres_missing_store_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            journal_path = Path(tmpdir) / "gpu_pending_releases_400.json"
            mode = {"status": 400}
            calls: list[tuple[str, dict]] = []

            def handler(request: httpx.Request) -> httpx.Response:
                body = json.loads(request.content.decode("utf-8")) if request.content else {}
                lid = request.url.path.split("/")[-2]
                calls.append((lid, body))
                if mode["status"] == 400:
                    return httpx.Response(
                        400,
                        json={"error_code": "INVALID_RECLAIM_PROOF", "message": "proof missing token_sealed"},
                    )
                return httpx.Response(200, json={"lease_id": lid, "state": "released"})

            http_client = httpx.Client(transport=httpx.MockTransport(handler))
            self.addCleanup(http_client.close)

            client = SchedulerLeaseClient(
                base_url="http://station-runtime:5056",
                token="sched-token",
                enabled=True,
                client=http_client,
                pending_releases_path=journal_path,
            )

            # 400 MUST quarantine entry instead of deleting it
            with self.assertRaises(SchedulerError) as e_400:
                client.release("lease-q-400", 51, worker_settled=False, reclaim_mode="abort_unstarted")
            self.assertEqual(e_400.exception.code, "INVALID_RECLAIM_PROOF")
            q_entry = client.get_pending_release("lease-q-400")
            self.assertIsNotNone(q_entry)
            assert q_entry is not None
            self.assertTrue(q_entry.get("_quarantined_400"))
            self.assertIn("INVALID_RECLAIM_PROOF", str(q_entry.get("_last_error") or ""))

            # Default flush_pending_releases skips quarantined entry (no hot-loop 400 storm)
            calls.clear()
            self.assertEqual(client.flush_pending_releases(), 0)
            self.assertEqual(len(calls), 0)
            self.assertIsNotNone(client.get_pending_release("lease-q-400"))

            # Explicit include_quarantined=True recovers and clears once server accepts
            mode["status"] = 200
            self.assertEqual(client.flush_pending_releases(include_quarantined=True), 1)
            self.assertEqual(len(calls), 1)
            self.assertNotIn("_quarantined_400", calls[0][1])
            self.assertIsNone(client.get_pending_release("lease-q-400"))

            # Postgres backend without Store MUST fail closed with JOURNAL_STORE_REQUIRED
            pg_no_store = SchedulerLeaseClient(
                base_url="http://station-runtime:5056",
                token="sched-token",
                enabled=True,
                client=http_client,
                pending_releases_backend="postgres",
                store=None,
            )
            with self.assertRaises(SchedulerError) as e_no_store:
                pg_no_store.release("lease-pg-nostore", 1, worker_settled=False, reclaim_mode="abort_unstarted")
            self.assertEqual(e_no_store.exception.code, "JOURNAL_STORE_REQUIRED")

    async def test_warm_replica_probes_runtime_identity_before_scheduler_acquire(self) -> None:
        _, _, row = fixture()
        row = copy.deepcopy(row)
        store = MemoryStore(row)
        api = MemoryApi()
        api.patch_scale("speech-indextts2", 1)
        gpu = GpuCoordinator(
            api,
            "indextts2",
            "speech/test",
            deployment="speech-indextts2",
            enabled=True,
            queue_seconds=5,
            ready_seconds=2,
            idle_seconds=600,
            now=lambda: datetime(2026, 11, 15, tzinfo=timezone.utc) + timedelta(seconds=1),
        )

        sched_calls: list[tuple[str, str, dict]] = []

        def sched_handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content.decode("utf-8")) if request.content else {}
            sched_calls.append((request.method, request.url.path, body))
            if request.url.path.endswith("/acquire"):
                return httpx.Response(
                    200,
                    json={
                        "lease_id": "lease-warm-gpu1",
                        "fencing_token": 91,
                        "version": 1,
                        "state": "leased",
                        "attempt_id": body["attempt_id"],
                        "stage_id": body["stage_id"],
                        "pod_uid": body.get("pod_uid", ""),
                        "container_id": body.get("container_id", ""),
                        "boot_id": body.get("boot_id", ""),
                        "gpu_uuids": body.get("candidate_gpu_uuids") or [],
                        "expires_at": "2099-01-01T00:00:00Z",
                    },
                )
            return httpx.Response(404)

        sched_http = httpx.Client(transport=httpx.MockTransport(sched_handler))
        self.addCleanup(sched_http.close)
        scheduler = SchedulerLeaseClient(
            base_url="http://station-runtime:5056",
            token="sched-token",
            enabled=True,
            client=sched_http,
        )

        async def runtime_json(p, method, path, payload=None, **kwargs):
            if path == "/health":
                return {
                    "ready": True,
                    "runtime_identity": {
                        "pod_uid": "pod-indextts2-gpu1",
                        "container_id": "pod-pod-indextts2-gpu1-speech",
                        "boot_id": "boot-node-5090",
                        "supervisor_pid": 6200,
                        "supervisor_starttime_ticks": 72000,
                        "visible_gpu_uuids": ["GPU-798fb508-830f-b538-6733-72ed6d210d40"],
                    },
                }
            raise AssertionError(f"unexpected path {path}")

        peers = SimpleNamespace(runtime=SimpleNamespace(json=AsyncMock(side_effect=runtime_json)), open=AsyncMock())
        with patch("speech_mcp.worker.threading.Thread.start"):
            worker = Worker(SimpleNamespace(store=store, peers=peers, gpu=gpu, scheduler=scheduler))

        await worker._ensure_gpu(row)
        acq = [c for c in sched_calls if c[1].endswith("/acquire")]
        binds = [c for c in sched_calls if c[1].endswith("/bind-target")]
        self.assertEqual(len(acq), 1)
        self.assertNotIn("k8s_slots_required", acq[0][2])
        self.assertEqual(
            acq[0][2]["candidate_gpu_uuids"],
            ["GPU-798fb508-830f-b538-6733-72ed6d210d40"],
        )
        self.assertEqual(acq[0][2]["pod_uid"], "pod-indextts2-gpu1")
        self.assertEqual(len(binds), 0)
