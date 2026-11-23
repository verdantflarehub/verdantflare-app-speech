"""T2.2 tests: retired legacy K8s Lease under :5056, startup orphan reconciliation,
4-condition strict scale-to-0 + Pod-exit settled release, and report-first execution timings.
"""
from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from speech_mcp import __version__
from speech_mcp.clients import Client
from speech_mcp.domain import Failure, canonical, digest, new_id, strict_json, validate
from speech_mcp.gpu_lease import GpuCoordinator, MemoryApi, SchedulerLeaseClient
from speech_mcp.service import Service
from speech_mcp.worker import Worker
from test_speech import MemoryStore, fixture


class SpeechT22LeaseAndTimingsTest(unittest.IsolatedAsyncioTestCase):
    async def test_scheduler_enabled_disables_legacy_k8s_lease_and_warns_once(self) -> None:
        self.assertEqual(__version__, "0.2.15")
        env = {
            "STATION_GPU_SCHEDULER_URL": "http://station-runtime:5056",
            "STATION_GPU_SCHEDULER_TOKEN": "s" * 36,
            "GPU_TASK_ROLE": "indextts2",
            "GPU_TASK_DEPLOYMENT": "speech-indextts2",
            "GPU_IDLE_SECONDS": "600",
        }
        with patch.dict(os.environ, env, clear=False):
            with self.assertLogs("speech_gpu_lease", level="WARNING") as cm:
                gpu = GpuCoordinator.from_environment()
            self.assertEqual(len(cm.output), 1)
            self.assertIn("speech_legacy_gpu_task_role_ignored", cm.output[0])
            self.assertFalse(gpu.enabled)
            self.assertTrue(gpu.scale_enabled)

            api = MemoryApi()
            gpu.api = api
            p, req, row = fixture()
            store = MemoryStore(row)
            peers = AsyncMock()
            peers.open.return_value = {"head_revision_id": req["revision_id"], "can_write": True}
            peers.profile.return_value = (row["state"]["profile"], "b" * 64)
            peers.content.return_value = (b"audio", {"mime": "audio/wav"})
            peers.runtime.json.side_effect = Failure("dependency_unavailable", 503, True)

            sched_http = httpx.Client(
                transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"leases": []}))
            )
            self.addCleanup(sched_http.close)
            scheduler = SchedulerLeaseClient(
                base_url="http://station-runtime:5056",
                token="s" * 36,
                enabled=True,
                client=sched_http,
            )
            svc = Service(store, peers, "a" * 64, gpu=gpu, scheduler=scheduler)
            with patch("speech_mcp.worker.threading.Thread.start"):
                _ = Worker(svc)

        # Cold-start speech.create succeeds when scaled to 0 without touching runtime or K8s Leases
        created = await svc.call("speech.create", req, p)
        self.assertEqual(created["status"], "pending")
        self.assertEqual(created["service_version"], "0.2.15")
        rt_info = await svc.engine_runtime_info(p)
        self.assertEqual(rt_info["reason"], "scaled_to_zero")
        self.assertTrue(rt_info["layers"]["orchestration_alive"])
        self.assertEqual(api.lease_calls, [])

    async def test_early_503_health_binds_target_immediately_and_pre_dispatch_failure_releases_lease(self) -> None:
        p, _, row = fixture()
        row = copy.deepcopy(row)
        store = MemoryStore(row)
        api = MemoryApi()
        gpu = GpuCoordinator(
            api,
            "",
            "",
            deployment="speech-indextts2",
            enabled=False,
            ready_seconds=10,
            sleep=lambda _: None,
        )

        sched_events: list[tuple[str, dict]] = []

        def sched_handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content.decode("utf-8")) if request.content else {}
            sched_events.append((request.url.path, body))
            if request.url.path.endswith("/acquire"):
                return httpx.Response(
                    200,
                    json={
                        "lease_id": "lease-early-bind",
                        "fencing_token": 55,
                        "version": 1,
                        "state": "leased",
                        "attempt_id": body["attempt_id"],
                        "stage_id": body["stage_id"],
                        "gpu_uuids": ["GPU-11111111-2222-3333-4444-555555555555"],
                    },
                )
            if request.url.path.endswith("/bind-target"):
                return httpx.Response(
                    200,
                    json={
                        "lease_id": "lease-early-bind",
                        "fencing_token": 55,
                        "version": 2,
                        "state": "running",
                        "pod_uid": body["pod_uid"],
                        "container_id": body["container_id"],
                        "boot_id": body["boot_id"],
                        "gpu_uuids": body.get("actual_gpu_uuids") or ["GPU-11111111-2222-3333-4444-555555555555"],
                    },
                )
            if request.url.path.endswith("/leases"):
                return httpx.Response(200, json={"leases": []})
            if request.url.path.endswith("/release"):
                return httpx.Response(200, json={"lease_id": "lease-early-bind", "state": "released"})
            return httpx.Response(200, json={})

        sched_http = httpx.Client(transport=httpx.MockTransport(sched_handler))
        self.addCleanup(sched_http.close)
        scheduler = SchedulerLeaseClient(
            base_url="http://station-runtime:5056",
            token="s" * 36,
            enabled=True,
            client=sched_http,
        )

        health_polls = [0]
        bind_count_on_503 = [0]

        def runtime_handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/health":
                health_polls[0] += 1
                is_ready = health_polls[0] >= 2
                if is_ready:
                    bind_count_on_503[0] = sum(1 for path, _ in sched_events if path.endswith("/bind-target"))
                return httpx.Response(
                    200 if is_ready else 503,
                    json={
                        "ready": is_ready,
                        "active_tasks": 0,
                        "runtime_identity": {
                            "pod_uid": "pod-loading-1",
                            "container_id": "cid-loading-1",
                            "boot_id": "boot-5090-1",
                            "supervisor_pid": 4100,
                            "supervisor_starttime_ticks": 91000,
                            "visible_gpu_uuids": ["GPU-11111111-2222-3333-4444-555555555555"],
                        },
                    },
                )
            if request.url.path == "/v1/capabilities":
                return httpx.Response(200, json={"ready": True, "model_manifest_sha256": "c" * 64})
            if request.url.path == "/internal/v1/gpu-guard/reclaim":
                return httpx.Response(
                    200,
                    json={
                        "ok": True,
                        "active_tasks": 0,
                        "settled": True,
                        "reclaim_proof": {
                            "reclaimed": True,
                            "active_tasks": 0,
                            "proof_source": "receiver_guard",
                            "token_sealed": True,
                            "revoked_lease_id": "lease-early-bind",
                            "consumed_fencing_token": 55,
                        },
                    },
                )
            return httpx.Response(404)

        rt_client = Client(
            "http://speech-indextts2:8080",
            "r" * 36,
            transport=httpx.MockTransport(runtime_handler),
            allow_http=True,
        )
        self.addCleanup(rt_client.close)
        peers = SimpleNamespace(open=AsyncMock(), runtime=rt_client)

        with patch("speech_mcp.worker.threading.Thread.start"):
            worker = Worker(SimpleNamespace(store=store, peers=peers, gpu=gpu, scheduler=scheduler))
        worker._ready_poll_seconds = 0.01
        await worker.process(row)

        # Bound immediately during HTTP 503 model-loading response!
        self.assertEqual(bind_count_on_503[0], 1)
        # Pre-dispatch model_version_changed failure scaled back to 0 and released :5056 with worker_settled=True
        self.assertEqual(store.row["status"], "failed")
        self.assertEqual(store.row["last_error"], "model_version_changed")
        self.assertEqual(api.get_scale("speech-indextts2"), 0)
        releases = [b for path, b in sched_events if path.endswith("/release")]
        self.assertEqual(len(releases), 1)
        self.assertTrue(releases[0]["worker_settled"])
        self.assertNotIn("reclaim_mode", releases[0])

    async def test_startup_reconciliation_adopts_inflight_and_settles_orphan_leases(self) -> None:
        _, _, row_active = fixture()
        row_active = copy.deepcopy(row_active)
        row_active["phase"] = "inference"
        row_active["status"] = "running"
        row_active["state"]["runtime_task_id"] = new_id()
        row_active["state"]["gpu_scheduler_lease"] = {
            "lease_id": "lease-adopt-1",
            "fencing_token": 101,
            "version": 2,
            "state": "running",
            "pod_uid": "pod-1",
        }

        _, _, row_done = fixture()
        row_done = copy.deepcopy(row_done)
        row_done["phase"] = "complete"
        row_done["status"] = "succeeded"
        row_done["state"]["gpu_scheduler_lease"] = {
            "lease_id": "lease-residual-done",
            "fencing_token": 99,
            "version": 3,
            "state": "running",
            "pod_uid": "pod-1",
            "container_id": "cid-1",
            "boot_id": "boot-1",
        }
        row_done["state"]["gpu_target_identity"] = {
            "pod_uid": "pod-1",
            "container_id": "cid-1",
            "boot_id": "boot-1",
        }

        store = MemoryStore(row_active)
        store.rows[row_done["task_id"]] = row_done
        api = MemoryApi()
        api.patch_scale("speech-indextts2", 1)
        gpu = GpuCoordinator(api, "", "", deployment="speech-indextts2", enabled=False, sleep=lambda _: None)

        sched_State = {
            "lease-adopt-1": {
                "lease_id": "lease-adopt-1",
                "holder_app_id": "speech-mcp",
                "fencing_token": 101,
                "version": 2,
                "state": "running",
                "pod_uid": "pod-1",
            },
            "lease-orphan-unbound": {
                "lease_id": "lease-orphan-unbound",
                "holder_app_id": "speech-mcp",
                "fencing_token": 100,
                "version": 1,
                "state": "leased",
                "pod_uid": "",
            },
            "lease-residual-done": {
                "lease_id": "lease-residual-done",
                "holder_app_id": "speech-mcp",
                "fencing_token": 99,
                "version": 3,
                "state": "running",
                "pod_uid": "pod-1",
                "container_id": "cid-1",
                "boot_id": "boot-1",
            },
        }
        released_calls: dict[str, dict] = {}
        heartbeats: list[str] = []

        def sched_handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content.decode("utf-8")) if request.content else {}
            path = request.url.path
            if path == "/internal/v1/gpu-scheduler/leases":
                return httpx.Response(
                    200,
                    json={
                        "leases": [
                            v for v in sched_State.values() if v["state"] not in {"released", "expired"}
                        ]
                    },
                )
            if path.endswith("/heartbeat"):
                lid = path.split("/")[-2]
                heartbeats.append(lid)
                return httpx.Response(200, json=sched_State[lid])
            if path.endswith("/release"):
                lid = path.split("/")[-2]
                released_calls[lid] = body
                sched_State[lid]["state"] = "released"
                return httpx.Response(200, json={"lease_id": lid, "state": "released"})
            lid = path.rsplit("/", 1)[-1]
            if lid in sched_State:
                return httpx.Response(200, json=sched_State[lid])
            return httpx.Response(404)

        sched_http = httpx.Client(transport=httpx.MockTransport(sched_handler))
        self.addCleanup(sched_http.close)
        scheduler = SchedulerLeaseClient(
            base_url="http://station-runtime:5056",
            token="s" * 36,
            enabled=True,
            client=sched_http,
        )

        def runtime_handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/health":
                return httpx.Response(
                    200,
                    json={
                        "ready": True,
                        "active_tasks": 0,
                        "runtime_identity": {
                            "pod_uid": "pod-1",
                            "container_id": "cid-1",
                            "boot_id": "boot-1",
                        },
                    },
                )
            if request.url.path == "/internal/v1/gpu-guard/reclaim":
                body = strict_json(request.content)
                return httpx.Response(
                    200,
                    json={
                        "ok": True,
                        "active_tasks": 0,
                        "settled": True,
                        "reclaim_proof": {
                            "reclaimed": True,
                            "active_tasks": 0,
                            "proof_source": "receiver_guard",
                            "token_sealed": True,
                            "revoked_lease_id": body["lease_id"],
                            "consumed_fencing_token": body["fencing_token"],
                        },
                    },
                )
            return httpx.Response(404)

        rt_client = Client(
            "http://speech-indextts2:8080",
            "r" * 36,
            transport=httpx.MockTransport(runtime_handler),
            allow_http=True,
        )
        self.addCleanup(rt_client.close)
        peers = SimpleNamespace(open=AsyncMock(), runtime=rt_client)

        with patch("speech_mcp.worker.threading.Thread.start"):
            worker = Worker(SimpleNamespace(store=store, peers=peers, gpu=gpu, scheduler=scheduler))

        summary = await worker.reconcile_startup_leases()
        # 1) Active in-flight task lease was adopted & heartbeated; Deployment stayed at 1 while active task exists!
        self.assertEqual(summary["adopted"], ["lease-adopt-1"])
        self.assertIn("lease-adopt-1", heartbeats)
        self.assertNotIn("lease-adopt-1", released_calls)
        self.assertEqual(api.get_scale("speech-indextts2"), 1)
        # 2) Unbound orphan was aborted immediately, and bound residual lease warm-released without killing active Pod!
        self.assertEqual(set(summary["released"]), {"lease-orphan-unbound", "lease-residual-done"})
        self.assertEqual(released_calls["lease-orphan-unbound"]["reclaim_mode"], "abort_unstarted")
        self.assertEqual(released_calls["lease-residual-done"]["reclaim_mode"], "warm_idle_settled")
        self.assertNotIn("gpu_scheduler_lease", row_done["state"])

        # 3) Now simulate restart when ONLY an orphan bound lease exists -> scales to 0, waits for Pod exit, releases!
        store.rows.pop(row_active["task_id"])
        sched_State.clear()
        sched_State["lease-sole-orphan"] = {
            "lease_id": "lease-sole-orphan",
            "holder_app_id": "speech-mcp",
            "fencing_token": 120,
            "version": 1,
            "state": "running",
            "pod_uid": "pod-1",
            "container_id": "cid-1",
            "boot_id": "boot-1",
        }
        released_calls.clear()
        summary2 = await worker.reconcile_startup_leases()
        self.assertEqual(summary2["released"], ["lease-sole-orphan"])
        self.assertEqual(api.get_scale("speech-indextts2"), 0)
        self.assertTrue(released_calls["lease-sole-orphan"]["worker_settled"])
        self.assertNotIn("reclaim_mode", released_calls["lease-sole-orphan"])

    async def test_four_strict_scale_down_preconditions_and_409_reclaim_and_version_conflict(self) -> None:
        _, _, row1 = fixture()
        row1 = copy.deepcopy(row1)
        _, _, row2 = fixture()
        row2 = copy.deepcopy(row2)
        store = MemoryStore(row1)
        store.rows[row2["task_id"]] = row2

        api = MemoryApi()
        api.patch_scale("speech-indextts2", 1)
        gpu = GpuCoordinator(
            api,
            "",
            "",
            deployment="speech-indextts2",
            enabled=False,
            ready_seconds=5,
            sleep=lambda _: None,
        )

        server_version = [2]
        conflict_triggered = [False]
        release_bodies: list[dict] = []

        def sched_handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content.decode("utf-8")) if request.content else {}
            path = request.url.path
            if path == "/internal/v1/gpu-scheduler/leases":
                return httpx.Response(200, json={"leases": []})
            if path.endswith("/release"):
                lid = path.split("/")[-2]
                if lid == "lease-final" and not conflict_triggered[0]:
                    conflict_triggered[0] = True
                    server_version[0] = 4
                    return httpx.Response(
                        409,
                        json={"error_code": "LEASE_VERSION_CONFLICT", "message": "version mismatch"},
                    )
                release_bodies.append({"lease_id": lid, **body})
                return httpx.Response(200, json={"lease_id": lid, "state": "released"})
            if path.endswith("/lease-final"):
                return httpx.Response(
                    200,
                    json={
                        "lease_id": "lease-final",
                        "fencing_token": 202,
                        "version": server_version[0],
                        "state": "running",
                        "pod_uid": "pod-2",
                        "container_id": "cid-2",
                        "boot_id": "boot-2",
                    },
                )
            return httpx.Response(200, json={})

        sched_http = httpx.Client(transport=httpx.MockTransport(sched_handler))
        self.addCleanup(sched_http.close)
        scheduler = SchedulerLeaseClient(
            base_url="http://station-runtime:5056",
            token="s" * 36,
            enabled=True,
            client=sched_http,
        )

        rt_state = {
            "active_tasks": 0,
            "reclaim_status": 200,
            "reclaimed": True,
            "token_sealed": True,
            "resident_count": 4,
        }

        def runtime_handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/health":
                return httpx.Response(
                    200,
                    json={
                        "ready": True,
                        "active_tasks": rt_state["active_tasks"],
                        "runtime_identity": {
                            "pod_uid": "pod-2",
                            "container_id": "cid-2",
                            "boot_id": "boot-2",
                        },
                    },
                )
            if request.url.path == "/internal/v1/gpu-guard/reclaim":
                body = strict_json(request.content)
                return httpx.Response(
                    rt_state["reclaim_status"],
                    json={
                        "ok": rt_state["reclaimed"],
                        "active_tasks": rt_state["active_tasks"],
                        "settled": rt_state["reclaimed"],
                        "reclaim_proof": {
                            "reclaimed": rt_state["reclaimed"],
                            "active_tasks": rt_state["active_tasks"],
                            "proof_source": "receiver_guard",
                            "token_sealed": rt_state["token_sealed"],
                            "revoked_lease_id": body["lease_id"],
                            "consumed_fencing_token": body["fencing_token"],
                            "resident_procs": [
                                {"pid": 5000 + i, "starttime_ticks": 80000 + i}
                                for i in range(rt_state["resident_count"])
                            ],
                        },
                    },
                )
            return httpx.Response(404)

        rt_client = Client(
            "http://speech-indextts2:8080",
            "r" * 36,
            transport=httpx.MockTransport(runtime_handler),
            allow_http=True,
        )
        self.addCleanup(rt_client.close)
        peers = SimpleNamespace(open=AsyncMock(), runtime=rt_client)

        with patch("speech_mcp.worker.threading.Thread.start"):
            worker = Worker(SimpleNamespace(store=store, peers=peers, gpu=gpu, scheduler=scheduler))

        # (a) Blocked from scaling to 0 because row2 is pending in 'reference'; warm-releases lease-1 instead
        row1["state"]["gpu_scheduler_lease"] = {
            "lease_id": "lease-1",
            "fencing_token": 201,
            "version": 1,
            "state": "running",
            "pod_uid": "pod-2",
        }
        row1["state"]["gpu_target_identity"] = {"pod_uid": "pod-2", "container_id": "cid-2", "boot_id": "boot-2"}
        await worker._release_scheduler_lease(row1, reason="task_succeeded")
        self.assertEqual(api.get_scale("speech-indextts2"), 1)
        self.assertEqual(release_bodies[-1]["reclaim_mode"], "warm_idle_settled")
        release_bodies.clear()

        # Remove row1; now row2 is the ONLY task, test blocking on (b) active_tasks > 0
        store.rows.pop(row1["task_id"])
        row2["status"] = "succeeded"
        row2["phase"] = "complete"
        lease_final = {
            "lease_id": "lease-final",
            "fencing_token": 202,
            "version": 2,
            "state": "running",
            "pod_uid": "pod-2",
        }
        ident_2 = {"pod_uid": "pod-2", "container_id": "cid-2", "boot_id": "boot-2"}

        rt_state["active_tasks"] = 1
        ok_b = await worker._settle_and_release_lease(lease_final, target_identity=ident_2, exclude_task_id=row2["task_id"])
        self.assertFalse(ok_b)
        self.assertEqual(api.get_scale("speech-indextts2"), 1)
        self.assertEqual(release_bodies, [])

        # (c) Blocked when token_sealed is False or resident_procs > 128
        rt_state["active_tasks"] = 0
        rt_state["token_sealed"] = False
        self.assertFalse(
            await worker._settle_and_release_lease(lease_final, target_identity=ident_2, exclude_task_id=row2["task_id"])
        )
        rt_state["token_sealed"] = True
        rt_state["resident_count"] = 129
        self.assertFalse(
            await worker._settle_and_release_lease(lease_final, target_identity=ident_2, exclude_task_id=row2["task_id"])
        )
        self.assertEqual(api.get_scale("speech-indextts2"), 1)
        self.assertEqual(release_bodies, [])

        # (c)+(d) HTTP 409 reclaim_proof with 128 resident_procs + token_sealed=True + Pod terminating wait + 409 retry!
        rt_state["reclaim_status"] = 409
        rt_state["reclaimed"] = False
        rt_state["resident_count"] = 128
        api.pods["speech-indextts2"] = [{"metadata": {"name": "speech-indextts2-terminating"}}]
        pod_polls = [0]
        orig_deployment_pods = api.deployment_pods

        def staged_pods(name: str) -> list[dict]:
            pod_polls[0] += 1
            if api.get_scale(name) == 0 and pod_polls[0] >= 3:
                api.pods[name] = []
            return orig_deployment_pods(name)

        api.deployment_pods = staged_pods  # type: ignore[method-assign]
        ok_final = await worker._settle_and_release_lease(
            lease_final,
            target_identity=ident_2,
            exclude_task_id=row2["task_id"],
        )
        self.assertTrue(ok_final)
        self.assertEqual(api.get_scale("speech-indextts2"), 0)
        self.assertGreaterEqual(pod_polls[0], 3)
        self.assertTrue(conflict_triggered[0])
        self.assertEqual(len(release_bodies), 1)
        self.assertEqual(release_bodies[0]["expected_version"], 4)
        self.assertTrue(release_bodies[0]["worker_settled"])
        self.assertNotIn("reclaim_mode", release_bodies[0])

    async def test_delayed_result_retrieval_and_three_report_timing_structures(self) -> None:
        # 1) Delayed retrieval (~10,136s turnaround) with top-level elapsed_seconds = 2.1168 (report.top)
        _, req, row = fixture()
        row = copy.deepcopy(row)
        now_dt = datetime.now(timezone.utc)
        row["created_at"] = now_dt - timedelta(seconds=10136)
        row["phase"] = "inference"
        row["status"] = "running"
        rt_tid = new_id()
        row["state"]["runtime_task_id"] = rt_tid
        row["state"]["submit_started_at"] = (now_dt - timedelta(seconds=10136)).timestamp()

        report_top = canonical(
            {
                "task_id": rt_tid,
                "elapsed_seconds": 2.1168,
                "resolved_emotion_vector": [0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.7],
            }
        ).encode("utf-8")
        payloads = {
            "original.wav": b"orig-wav",
            "editing.wav": b"edit-wav",
            "report.json": report_top,
        }
        entries = [
            {
                "name": name,
                "sha256": digest(blob),
                "size": len(blob),
                "mime": "application/json" if name.endswith(".json") else "audio/wav",
            }
            for name, blob in payloads.items()
        ]
        store = MemoryStore(row)
        runtime = SimpleNamespace(
            json=AsyncMock(
                return_value={
                    "task_id": rt_tid,
                    "model_manifest_sha256": "a" * 64,
                    "status": "succeeded",
                    "outputs": copy.deepcopy(entries),
                }
            ),
            request=AsyncMock(side_effect=lambda p, method, path, **kw: payloads[path.rsplit("/", 1)[-1]]),
        )
        archived: dict[str, bytes] = {}

        async def fake_archive(p, r, entry, data):
            self.assertEqual(digest(data), entry["sha256"])
            archived[entry["name"]] = data
            return {
                "name": entry["name"],
                "content_ref": req["voice_profile"],
                "sha256": entry["sha256"],
                "size": entry["size"],
                "mime": entry["mime"],
            }

        peers = SimpleNamespace(open=AsyncMock(), runtime=runtime, archive=AsyncMock(side_effect=fake_archive))
        with patch("speech_mcp.worker.threading.Thread.start"):
            worker = Worker(
                SimpleNamespace(
                    store=store,
                    peers=peers,
                    gpu=SimpleNamespace(enabled=False),
                    scheduler=SimpleNamespace(enabled=False),
                )
            )

        await worker.step(row)
        self.assertTrue(row["state"]["generation_finalized"])
        gen_entry_sha_before_archive = next(
            e["sha256"] for e in row["state"]["archive_entries"] if e["name"] == "generation.json"
        )
        timings = row["state"]["timings"]
        self.assertEqual(timings["execution_seconds"], 2.1168)
        self.assertEqual(timings["execution_source"], "report.top")
        self.assertEqual(timings["inference_seconds"], 2.1168)
        self.assertGreaterEqual(timings["turnaround_seconds"], 10135.0)
        self.assertIsNone(timings["queue_wait_seconds"])

        for _ in range(5):
            await worker.step(row)
        self.assertEqual(row["status"], "succeeded")
        gen_doc = strict_json(archived["generation.json"])
        validate("generation_record", gen_doc)
        self.assertEqual(digest(archived["generation.json"]), gen_entry_sha_before_archive)
        self.assertEqual(gen_doc["timings"]["execution_seconds"], 2.1168)
        self.assertEqual(gen_doc["timings"]["inference_seconds"], 2.1168)

        # 2) report.metrics fallback + explicit queue_wait_seconds
        _, _, row_m = fixture()
        row_m["created_at"] = now_dt - timedelta(seconds=20)
        t_m = worker._compute_final_timings(
            row_m,
            {"created_at": "2026-11-18T00:00:00Z", "started_at": "2026-11-18T00:00:03.500000Z"},
            {"metrics": {"elapsed_seconds": 1.4321, "load_seconds": 0.75}},
            now_epoch=now_dt.timestamp(),
        )
        self.assertEqual(t_m["execution_seconds"], 1.4321)
        self.assertEqual(t_m["execution_source"], "report.metrics")
        self.assertEqual(t_m["inference_seconds"], 1.4321)
        self.assertEqual(t_m["queue_wait_seconds"], 3.5)
        self.assertEqual(t_m["load_seconds"], 0.75)

        # 3) missing fallback -> None (never wall-clock turnaround!)
        _, _, row_none = fixture()
        row_none["created_at"] = now_dt - timedelta(seconds=50)
        t_none = worker._compute_final_timings(row_none, {}, {"metrics": {"cer": 0.0}}, now_epoch=now_dt.timestamp())
        self.assertIsNone(t_none["execution_seconds"])
        self.assertIsNone(t_none["inference_seconds"])
        self.assertEqual(t_none["execution_source"], "missing")
        self.assertGreaterEqual(t_none["turnaround_seconds"], 49.9)
        self.assertIsNone(t_none["queue_wait_seconds"])

    async def test_invalid_execution_seconds_anomaly_guard_and_tampered_report_rejected(self) -> None:
        _, _, row = fixture()
        now_dt = datetime.now(timezone.utc)
        row["created_at"] = now_dt - timedelta(seconds=5)
        with patch("speech_mcp.worker.threading.Thread.start"):
            worker = Worker(
                SimpleNamespace(
                    store=MemoryStore(row),
                    peers=SimpleNamespace(open=AsyncMock(), runtime=SimpleNamespace()),
                    gpu=SimpleNamespace(enabled=False),
                    scheduler=SimpleNamespace(enabled=False),
                )
            )

        for bad_elapsed in (-0.25, 999.0):
            with self.assertLogs("speech_worker", level="WARNING") as cm:
                t_bad = worker._compute_final_timings(
                    copy.deepcopy(row),
                    {},
                    {"elapsed_seconds": bad_elapsed},
                    now_epoch=now_dt.timestamp(),
                )
            self.assertTrue(any("speech_invalid_execution_seconds" in line for line in cm.output))
            self.assertIsNone(t_bad["execution_seconds"])
            self.assertIsNone(t_bad["inference_seconds"])
            self.assertEqual(t_bad["execution_source"], "report.top")

        # Tampered report.json SHA-256 fails in inference BEFORE state["generation"] is constructed
        _, _, row_tamper = fixture()
        row_tamper["phase"] = "inference"
        row_tamper["status"] = "running"
        rt_tid = new_id()
        row_tamper["state"]["runtime_task_id"] = rt_tid
        good_report = b'{"elapsed_seconds":1.2}'
        entries = [
            {"name": "original.wav", "sha256": digest(b"a"), "size": 1, "mime": "audio/wav"},
            {"name": "editing.wav", "sha256": digest(b"b"), "size": 1, "mime": "audio/wav"},
            {"name": "report.json", "sha256": digest(good_report), "size": len(good_report), "mime": "application/json"},
        ]
        runtime = SimpleNamespace(
            json=AsyncMock(
                return_value={
                    "task_id": rt_tid,
                    "model_manifest_sha256": "a" * 64,
                    "status": "succeeded",
                    "outputs": entries,
                }
            ),
            request=AsyncMock(return_value=b'{"elapsed_seconds":9.9}'),
        )
        worker.peers = SimpleNamespace(open=AsyncMock(), runtime=runtime)
        with self.assertRaisesRegex(Failure, "output_integrity_failed"):
            await worker.step(row_tamper)
        self.assertNotIn("generation", row_tamper["state"])

    async def test_reclaiming_startup_lease_settled_and_queued_yield_reuses_same_lease(self) -> None:
        _, _, row_reclaim = fixture()
        row_reclaim["status"] = "running"
        row_reclaim["phase"] = "submit"
        row_reclaim["lease_token"] = None
        row_reclaim["state"]["gpu_scheduler_lease"] = {
            "lease_id": "lease-reclaim-1",
            "state": "leased",
            "fencing_token": 77,
            "version": 3,
            "pod_uid": "pod-u1",
            "container_id": "cid-1",
            "boot_id": "boot-1",
        }
        _, _, row_other = fixture()
        row_other["task_id"] = new_id()
        row_other["status"] = "running"
        row_other["phase"] = "reference"

        store = MemoryStore(row_reclaim)
        store.rows[row_other["task_id"]] = row_other

        released_modes: dict[str, str] = {}

        def fake_release(lid, tok, **kwargs):
            released_modes[lid] = kwargs.get("reclaim_mode") or ""
            return {"lease_id": lid, "state": "released"}

        acquire_calls = []
        q_get_calls = []

        def fake_acquire(**kwargs):
            acquire_calls.append(kwargs)
            return {"lease_id": "lease-q-reuse", "state": "queued", "fencing_token": 88, "queue_position": 1}

        def fake_get_lease(lid):
            if lid == "lease-reclaim-1":
                return {
                    "lease_id": "lease-reclaim-1",
                    "state": "reclaiming",
                    "fencing_token": 77,
                    "version": 4,
                    "pod_uid": "pod-u1",
                    "container_id": "cid-1",
                    "boot_id": "boot-1",
                }
            q_get_calls.append(lid)
            if len(q_get_calls) < 1:
                return {"lease_id": "lease-q-reuse", "state": "queued", "fencing_token": 88, "queue_position": 1}
            return {
                "lease_id": "lease-q-reuse",
                "state": "leased",
                "fencing_token": 88,
                "version": 2,
                "gpu_uuids": ["GPU-0000"],
                "pod_uid": "pod-u1",
                "container_id": "cid-1",
                "boot_id": "boot-1",
            }

        sched = SimpleNamespace(
            enabled=True,
            flush_pending_releases=lambda **_: 0,
            list_own_leases=lambda: [],
            get_lease=fake_get_lease,
            heartbeat=lambda lid, tok: {"lease_id": lid, "state": "running", "fencing_token": tok},
            acquire=fake_acquire,
            release=fake_release,
        )
        runtime = SimpleNamespace(
            request_status_json=AsyncMock(
                side_effect=lambda method, path, *args, **kwargs: (
                    (
                        200,
                        {
                            "active_tasks": 0,
                            "runtime_identity": {"pod_uid": "pod-u1", "container_id": "cid-1", "boot_id": "boot-1"},
                        },
                    )
                    if path == "/health"
                    else (
                        200,
                        {
                            "reclaim_proof": {
                                "proof_source": "receiver_guard",
                                "token_sealed": True,
                                "reclaimed": True,
                                "active_tasks": 0,
                                "revoked_lease_id": "lease-reclaim-1",
                                "resident_procs": [{"pid": 100, "starttime_ticks": 200}],
                            }
                        },
                    )
                )
            )
        )
        gpu = SimpleNamespace(
            enabled=False,
            scale_enabled=True,
            queue_seconds=60,
            replicas_for=lambda: 1,
            deployment_pods=lambda: ["speech-indextts2-abc"],
        )
        with patch("threading.Thread"):
            w = Worker(
                SimpleNamespace(
                    store=store,
                    peers=SimpleNamespace(open=AsyncMock(), runtime=runtime),
                    gpu=gpu,
                    scheduler=sched,
                )
            )
        w._queue_step_yield_seconds = 0.0
        summary = await w.reconcile_startup_leases()
        self.assertEqual(summary["adopted"], [])
        self.assertEqual(summary["released"], ["lease-reclaim-1"])
        self.assertEqual(released_modes["lease-reclaim-1"], "warm_idle_settled")
        self.assertNotIn("gpu_scheduler_lease", row_reclaim["state"])

        # Step yield preserves queued lease_id and resumes polling without minting a second acquire attempt!
        row_other["lease_token"] = "tok-1"
        with self.assertRaisesRegex(Failure, "waiting_gpu_lease"):
            await w._acquire_scheduler_grant(row_other)
        self.assertEqual(len(acquire_calls), 1)
        self.assertEqual(row_other["state"]["gpu_scheduler_lease"]["lease_id"], "lease-q-reuse")

        w._queue_step_yield_seconds = 5.0
        await w._acquire_scheduler_grant(row_other)
        self.assertEqual(len(acquire_calls), 1)
        self.assertEqual(row_other["state"]["gpu_scheduler_lease"]["state"], "leased")

    async def test_preinference_stale_fencing_token_auto_retries_and_requeues_on_startup(self) -> None:
        p, req, row = fixture()
        row = copy.deepcopy(row)
        row["status"] = "failed"
        row["phase"] = "inference"
        row["last_error"] = "stale_fencing_token"
        row["state"]["outputs"] = []
        row["state"]["submit_dispatched"] = True
        row["state"]["runtime_task_id"] = new_id()
        row["state"]["runtime_inputs"] = {
            "idempotency_key": "speech-" + row["task_id"],
            "text": "你好",
            "language": "zh-CN",
            "reference_sha256": "a" * 64,
        }
        store = MemoryStore(row)
        sched = SimpleNamespace(
            enabled=True,
            flush_pending_releases=lambda **_: 0,
            list_own_leases=lambda: [],
            acquire=lambda **_: {"lease_id": "lease-fresh-1", "fencing_token": 999, "state": "leased", "version": 1},
            heartbeat=lambda lid, tok: {"lease_id": lid, "fencing_token": tok, "state": "running", "version": 2},
            release=lambda *args, **kwargs: {"state": "released"},
            build_grant_headers=lambda lease, **_: {"X-VF-GPU-Fencing-Token": "999"},
        )
        runtime_posts: list[dict] = []

        async def runtime_json(principal, method, path, payload=None, **kwargs):
            if path == "/v1/capabilities":
                return {"model_manifest_sha256": row["state"]["profile"]["model_manifest_sha256"], "ready": True}
            if method == "PUT" and path.startswith("/v1/references/"):
                return {"ok": True}
            if method == "POST" and path == "/v1/tasks":
                runtime_posts.append(dict(payload))
                if len(runtime_posts) == 1:
                    raise Failure("stale_fencing_token", 409)
                return {"task_id": new_id(), "status": "pending"}
            raise AssertionError(f"unexpected {method} {path}")

        ref_bytes = b"ref-bytes"
        row["state"]["profile"]["reference"]["sha256"] = digest(ref_bytes)
        peers = SimpleNamespace(
            open=AsyncMock(),
            content=AsyncMock(return_value=(ref_bytes, {"mime": "audio/wav"})),
            runtime=SimpleNamespace(json=AsyncMock(side_effect=runtime_json)),
        )
        gpu = SimpleNamespace(enabled=False, scale_enabled=False, queue_seconds=60)
        with patch("threading.Thread"):
            w = Worker(SimpleNamespace(store=store, peers=peers, gpu=gpu, scheduler=sched))
        w._ensure_gpu = AsyncMock()

        await w.reconcile_startup_leases()
        self.assertEqual(row["status"], "running")
        self.assertEqual(row["phase"], "reference")
        self.assertIsNone(row["last_error"])
        self.assertEqual(row["state"]["fence_retry_count"], 1)
        self.assertNotIn("runtime_task_id", row["state"])
        self.assertNotIn("submit_dispatched", row["state"])

        # Step reference -> submit with -r1 idempotency suffix
        await w.process(row)
        self.assertEqual(row["phase"], "submit")
        self.assertEqual(row["state"]["runtime_inputs"]["idempotency_key"], f"speech-{row['task_id']}-r1")

        # First POST /v1/tasks hits 409 stale_fencing_token -> retryable reset to reference with -r2
        await w.process(row)
        self.assertEqual(row["status"], "running")
        self.assertEqual(row["phase"], "reference")
        self.assertEqual(row["state"]["fence_retry_count"], 2)

        # Next reference -> submit succeeds with -r2
        await w.process(row)
        await w.process(row)
        self.assertEqual(row["phase"], "inference")
        self.assertEqual(runtime_posts[-1]["idempotency_key"], f"speech-{row['task_id']}-r2")


if __name__ == "__main__":
    unittest.main()
