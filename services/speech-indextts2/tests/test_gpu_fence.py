"""Unit tests for speech-indextts2 6-tuple LeaseFenceGuard, durable token sealing, reclaim proof, and S2 evict-vram."""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
from pathlib import Path
import struct
import tempfile
import threading
import unittest
import wave

import httpx

from speech_runtime.domain import Failure, new_id
from speech_runtime.gpu_guard import FencePersistenceError, LeaseFenceGuard
from speech_runtime.server import create_app


class DummyASR:
    ready = True

    def load(self) -> None:
        self.ready = True

    def check_audio(self, *_args, **_kwargs):
        return {"transcript_check": {"state": "passed"}, "asr_metrics": None}


class DummyEngine:
    def __init__(self) -> None:
        self.asr = DummyASR()
        self.load_seconds = 0.05
        self.unloaded = False

    def load(self) -> None:
        self.unloaded = False

    def unload(self) -> None:
        self.unloaded = True


class SpeechIndexTTS2FenceTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self._old_policy = os.environ.get("SPEECH_GPU_GRANT_POLICY")
        self._old_pod = os.environ.get("POD_UID")
        self._old_cid = os.environ.get("CONTAINER_ID")
        self._old_boot = os.environ.get("NODE_BOOT_ID")
        self._old_gpu = os.environ.get("VF_VISIBLE_GPU_UUIDS")
        os.environ["SPEECH_GPU_GRANT_POLICY"] = "require"
        os.environ["POD_UID"] = "pod-speech-1"
        os.environ["CONTAINER_ID"] = "containerd://cid-speech-1"
        os.environ["NODE_BOOT_ID"] = "boot-speech-1"
        os.environ["VF_VISIBLE_GPU_UUIDS"] = "GPU-aaaaaaaa-1111-2222-3333-444444444444"

        def _restore() -> None:
            for k, v in (
                ("SPEECH_GPU_GRANT_POLICY", self._old_policy),
                ("POD_UID", self._old_pod),
                ("CONTAINER_ID", self._old_cid),
                ("NODE_BOOT_ID", self._old_boot),
                ("VF_VISIBLE_GPU_UUIDS", self._old_gpu),
            ):
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

        self.addCleanup(_restore)
        self.token = "s" * 40
        self.engine = DummyEngine()
        self.rt_root = str(Path(self.tmp.name) / "rt")
        self.model_root = str(Path(self.tmp.name) / "models")
        self.app = create_app(
            root=self.rt_root,
            token=self.token,
            model_root=self.model_root,
            engine=self.engine,
            model_hash="a" * 64,
            start_worker=False,
        )
        self.app.state.worker.ready = True
        self.addCleanup(lambda: None if self.app.state.store.lockfile.closed else self.app.state.store.close())

    def _scope_headers(self, body_len: int, extra: dict[str, str] | None = None) -> dict[str, str]:
        hdrs = {
            "Authorization": f"Bearer {self.token}",
            "X-Organization-Id": new_id(),
            "X-User-Id": new_id(),
            "X-Project-Id": new_id(),
            "X-Request-Id": new_id(),
            "Content-Type": "application/json",
            "Content-Length": str(body_len),
        }
        if extra:
            hdrs.update(extra)
        return hdrs

    async def test_health_residency_revoke_and_evict_vram(self) -> None:
        transport = httpx.ASGITransport(app=self.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://indextts2") as client:
            h = await client.get("/health")
            self.assertEqual(h.status_code, 200)
            h_json = h.json()
            self.assertEqual(h_json["residency_state"], "S2_warm_idle")
            self.assertEqual(h_json["runtime_identity"]["pod_uid"], "pod-speech-1")

            # Missing 6-tuple rejected in require mode
            body = json.dumps(
                {
                    "idempotency_key": "speech-idem-1",
                    "text": "你好世界",
                    "language": "zh-CN",
                    "reference_sha256": hashlib.sha256(b"wav").hexdigest(),
                }
            ).encode()
            resp_no_grant = await client.post("/v1/tasks", content=body, headers=self._scope_headers(len(body)))
            self.assertEqual(resp_no_grant.status_code, 403)
            self.assertEqual(resp_no_grant.json()["error"]["code"], "missing_or_invalid_gpu_grant")

            # Internal revoke-lease works without user scope headers and stamps receiver_guard proof
            rev_body = json.dumps({"lease_id": "lease-speech-1", "fencing_token": 15}).encode()
            rev_resp = await client.post(
                "/internal/v1/gpu-residency/revoke-lease",
                content=rev_body,
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Content-Type": "application/json",
                    "Content-Length": str(len(rev_body)),
                },
            )
            self.assertEqual(rev_resp.status_code, 200)
            self.assertTrue(rev_resp.json()["settled"])
            proof = rev_resp.json()["reclaim_proof"]
            self.assertTrue(proof["reclaimed"])
            self.assertEqual(proof["proof_source"], "receiver_guard")
            self.assertTrue(proof["token_sealed"])

            # Internal warm S2 evict-vram unloads idle model and transitions ready=False
            evict_body = json.dumps({"lease_id": "lease-evict-1", "fencing_token": 16}).encode()
            evict_resp = await client.post(
                "/internal/v1/gpu-residency/evict-vram",
                content=evict_body,
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Content-Type": "application/json",
                    "Content-Length": str(len(evict_body)),
                },
            )
            self.assertEqual(evict_resp.status_code, 200)
            self.assertTrue(evict_resp.json()["evicted"])
            self.assertTrue(self.engine.unloaded)
            self.assertFalse(self.app.state.worker.ready)

    async def test_unknown_identity_fail_closed_and_durable_token_sealing_across_restart(self) -> None:
        transport = httpx.ASGITransport(app=self.app)
        wav_buf = io.BytesIO()
        rate = 22050
        frames = bytearray()
        for idx in range(rate * 2):
            sample = int(4800 * math.sin(2.0 * math.pi * 220.0 * idx / rate))
            frames.extend(struct.pack("<h", sample))
        with wave.open(wav_buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(rate)
            wf.writeframes(bytes(frames))
        ref_bytes = wav_buf.getvalue()
        ref_sha = hashlib.sha256(ref_bytes).hexdigest()

        async with httpx.AsyncClient(transport=transport, base_url="http://indextts2") as client:
            scope_org, scope_user, scope_proj = new_id(), new_id(), new_id()
            put_hdrs = {
                "Authorization": f"Bearer {self.token}",
                "X-Organization-Id": scope_org,
                "X-User-Id": scope_user,
                "X-Project-Id": scope_proj,
                "X-Request-Id": new_id(),
                "Content-Type": "audio/wav",
                "Content-Length": str(len(ref_bytes)),
            }
            r_put = await client.put(f"/v1/references/{ref_sha}", content=ref_bytes, headers=put_hdrs)
            self.assertEqual(r_put.status_code, 200)

            # Unknown boot_id fails closed under enforce
            os.environ["NODE_BOOT_ID"] = "unknown-boot-id"
            body1 = json.dumps(
                {
                    "idempotency_key": "speech-idem-unk",
                    "text": "测试未知身份拒绝",
                    "language": "zh-CN",
                    "reference_sha256": ref_sha,
                }
            ).encode()
            grant_hdrs = {
                "Authorization": f"Bearer {self.token}",
                "X-Organization-Id": scope_org,
                "X-User-Id": scope_user,
                "X-Project-Id": scope_proj,
                "X-Request-Id": new_id(),
                "Content-Type": "application/json",
                "Content-Length": str(len(body1)),
                "X-VF-GPU-Lease-ID": "lease-speech-21",
                "X-VF-GPU-Fencing-Token": "21",
                "X-VF-GPU-Attempt-ID": "att-1",
                "X-VF-GPU-Stage-ID": "indextts2_inference",
                "X-VF-GPU-Parent-Task-ID": "task-parent-21",
                "X-VF-GPU-Pod-UID": "pod-speech-1",
                "X-VF-GPU-Container-ID": "cid-speech-1",
                "X-VF-GPU-Boot-ID": "boot-speech-1",
                "X-VF-GPU-UUID": "GPU-aaaaaaaa-1111-2222-3333-444444444444",
                "X-VF-GPU-Expires-At": "2099-01-01T00:00:00Z",
            }
            r_unk = await client.post("/v1/tasks", content=body1, headers=grant_hdrs)
            self.assertEqual(r_unk.status_code, 403)

            # Restore boot_id -> normalized container_id ("cid-speech-1" == "containerd://cid-speech-1") succeeds
            os.environ["NODE_BOOT_ID"] = "boot-speech-1"
            r_ok = await client.post("/v1/tasks", content=body1, headers=grant_hdrs)
            self.assertEqual(r_ok.status_code, 202)
            created_task_id = r_ok.json()["task_id"]

            # Reclaim endpoint seals token 21 once active_tasks == 0 (mark task failed in DB first so active_count==0)
            with self.app.state.store.mutex:
                self.app.state.store.db.execute("UPDATE tasks SET status='failed' WHERE id=?", (created_task_id,))
                self.app.state.store.db.commit()

            rec_body = json.dumps(
                {"lease_id": "lease-speech-21", "fencing_token": 21, "task_id": created_task_id}
            ).encode()
            r_rec = await client.post(
                "/internal/v1/gpu-guard/reclaim",
                content=rec_body,
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Content-Type": "application/json",
                    "Content-Length": str(len(rec_body)),
                },
            )
            self.assertEqual(r_rec.status_code, 200)
            rec_json = r_rec.json()
            self.assertTrue(rec_json["settled"])
            self.assertTrue(rec_json["reclaim_proof"]["reclaimed"])
            self.assertEqual(rec_json["reclaim_proof"]["proof_source"], "receiver_guard")
            self.assertTrue(rec_json["reclaim_proof"]["token_sealed"])
            self.assertEqual(rec_json["reclaim_proof"]["consumed_fencing_token"], 21)

            # Replay with new attempt_id and same token 21 MUST fail with 409 stale_fencing_token
            body2 = json.dumps(
                {
                    "idempotency_key": "speech-idem-replay",
                    "text": "重放旧令牌必须拒绝",
                    "language": "zh-CN",
                    "reference_sha256": ref_sha,
                }
            ).encode()
            replay_hdrs = dict(grant_hdrs)
            replay_hdrs["X-Request-Id"] = new_id()
            replay_hdrs["Content-Length"] = str(len(body2))
            replay_hdrs["X-VF-GPU-Attempt-ID"] = "att-2"
            r_replay = await client.post("/v1/tasks", content=body2, headers=replay_hdrs)
            self.assertEqual(r_replay.status_code, 409)
            self.assertEqual(r_replay.json()["error"]["code"], "stale_fencing_token")

        # Restart app on same rt_root -> close first store lock, then verify persisted gpu_guard_state.json blocks token <= 21!
        self.app.state.store.close()
        app2 = create_app(
            root=self.rt_root,
            token=self.token,
            model_root=self.model_root,
            engine=DummyEngine(),
            model_hash="a" * 64,
            start_worker=False,
        )
        app2.state.worker.ready = True
        self.addCleanup(app2.state.store.close)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app2), base_url="http://indextts2") as client2:
            replay_hdrs["X-Request-Id"] = new_id()
            replay_hdrs["X-VF-GPU-Attempt-ID"] = "att-3"
            r_after_restart = await client2.post("/v1/tasks", content=body2, headers=replay_hdrs)
            self.assertEqual(r_after_restart.status_code, 409)
            self.assertEqual(r_after_restart.json()["error"]["code"], "stale_fencing_token")

    async def test_r2_h1_concurrent_worker_inference_blocks_reclaim_with_409_until_settled(self) -> None:
        entered = threading.Barrier(2)
        release_worker = threading.Event()

        class BlockingEngine(DummyEngine):
            def generate(self, reference, inputs, output, emo_reference=None):
                entered.wait(timeout=5.0)
                if not release_worker.wait(timeout=5.0):
                    raise TimeoutError("release_worker timed out")
                raise Failure("synthetic_done", 500)

        rt_conc = str(Path(self.tmp.name) / "rt_conc")
        app_conc = create_app(
            root=rt_conc,
            token=self.token,
            model_root=self.model_root,
            engine=BlockingEngine(),
            model_hash="a" * 64,
            start_worker=False,
        )
        app_conc.state.worker.ready = True
        self.addCleanup(lambda: None if app_conc.state.store.lockfile.closed else app_conc.state.store.close())

        wav_buf = io.BytesIO()
        rate = 22050
        frames = bytearray()
        for idx in range(rate * 2):
            sample = int(4800 * math.sin(2.0 * math.pi * 220.0 * idx / rate))
            frames.extend(struct.pack("<h", sample))
        with wave.open(wav_buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(rate)
            wf.writeframes(bytes(frames))
        ref_bytes = wav_buf.getvalue()
        ref_sha = hashlib.sha256(ref_bytes).hexdigest()

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app_conc), base_url="http://indextts2") as client:
            scope_org, scope_user, scope_proj = new_id(), new_id(), new_id()
            r_put = await client.put(
                f"/v1/references/{ref_sha}",
                content=ref_bytes,
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "X-Organization-Id": scope_org,
                    "X-User-Id": scope_user,
                    "X-Project-Id": scope_proj,
                    "X-Request-Id": new_id(),
                    "Content-Type": "audio/wav",
                    "Content-Length": str(len(ref_bytes)),
                },
            )
            self.assertEqual(r_put.status_code, 200)

            body = json.dumps(
                {
                    "idempotency_key": "speech-conc-1",
                    "text": "并发测试",
                    "language": "zh-CN",
                    "reference_sha256": ref_sha,
                }
            ).encode()
            grant_hdrs = {
                "Authorization": f"Bearer {self.token}",
                "X-Organization-Id": scope_org,
                "X-User-Id": scope_user,
                "X-Project-Id": scope_proj,
                "X-Request-Id": new_id(),
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
                "X-VF-GPU-Lease-ID": "lease-speech-55",
                "X-VF-GPU-Fencing-Token": "55",
                "X-VF-GPU-Attempt-ID": "att-1",
                "X-VF-GPU-Stage-ID": "indextts2_inference",
                "X-VF-GPU-Parent-Task-ID": "task-parent-55",
                "X-VF-GPU-Pod-UID": "pod-speech-1",
                "X-VF-GPU-Container-ID": "cid-speech-1",
                "X-VF-GPU-Boot-ID": "boot-speech-1",
                "X-VF-GPU-UUID": "GPU-aaaaaaaa-1111-2222-3333-444444444444",
                "X-VF-GPU-Expires-At": "2099-01-01T00:00:00Z",
            }
            r_create = await client.post("/v1/tasks", content=body, headers=grant_hdrs)
            self.assertEqual(r_create.status_code, 202)
            created_task_id = r_create.json()["task_id"]

            claimed_row = app_conc.state.store.claim()
            self.assertIsNotNone(claimed_row)
            worker_thread = threading.Thread(target=app_conc.state.worker.execute, args=(claimed_row,))
            worker_thread.start()
            entered.wait(timeout=5.0)

            rec_body = json.dumps(
                {"lease_id": "lease-speech-55", "fencing_token": 55, "task_id": created_task_id}
            ).encode()
            r_rec_busy = await client.post(
                "/internal/v1/gpu-guard/reclaim",
                content=rec_body,
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Content-Type": "application/json",
                    "Content-Length": str(len(rec_body)),
                },
            )
            self.assertEqual(r_rec_busy.status_code, 409)
            busy_json = r_rec_busy.json()
            self.assertFalse(busy_json["settled"])
            self.assertGreaterEqual(busy_json["active_tasks"], 1)
            self.assertFalse(busy_json["reclaim_proof"]["reclaimed"])
            self.assertTrue(busy_json["reclaim_proof"]["token_sealed"])

            ev_body = json.dumps({"lease_id": "lease-speech-55", "fencing_token": 55}).encode()
            r_ev_busy = await client.post(
                "/internal/v1/gpu-residency/evict-vram",
                content=ev_body,
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Content-Type": "application/json",
                    "Content-Length": str(len(ev_body)),
                },
            )
            self.assertEqual(r_ev_busy.status_code, 409)

            release_worker.set()
            worker_thread.join(timeout=5.0)
            self.assertFalse(worker_thread.is_alive())

            r_rec_idle = await client.post(
                "/internal/v1/gpu-guard/reclaim",
                content=rec_body,
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Content-Type": "application/json",
                    "Content-Length": str(len(rec_body)),
                },
            )
            self.assertEqual(r_rec_idle.status_code, 200)
            idle_json = r_rec_idle.json()
            self.assertTrue(idle_json["settled"])
            self.assertEqual(idle_json["active_tasks"], 0)
            self.assertTrue(idle_json["reclaim_proof"]["reclaimed"])
            self.assertEqual(idle_json["reclaim_proof"]["consumed_fencing_token"], 55)

    async def test_r2_h3_short_writes_persist_failure_and_corrupted_state_fail_closed_503(self) -> None:
        state_file = Path(self.tmp.name) / "guard_short.json"
        short_calls = 0

        def one_byte_write(fd: int, buf) -> int:
            nonlocal short_calls
            short_calls += 1
            return os.write(fd, bytes(buf[:1]))

        g1 = LeaseFenceGuard(state_path=state_file, write_fn=one_byte_write)
        ident = {
            "pod_uid": "pod-speech-1",
            "container_id": "cid-speech-1",
            "boot_id": "boot-speech-1",
            "supervisor_pid": os.getpid(),
            "supervisor_starttime_ticks": 1,
            "visible_gpu_uuids": ["GPU-aaaaaaaa-1111-2222-3333-444444444444"],
        }
        hdrs = {
            "X-VF-GPU-Lease-ID": "lease-s1",
            "X-VF-GPU-Fencing-Token": "31",
            "X-VF-GPU-Attempt-ID": "att-1",
            "X-VF-GPU-Stage-ID": "indextts2_inference",
            "X-VF-GPU-Parent-Task-ID": "p-1",
            "X-VF-GPU-Pod-UID": "pod-speech-1",
            "X-VF-GPU-Container-ID": "cid-speech-1",
            "X-VF-GPU-Boot-ID": "boot-speech-1",
            "X-VF-GPU-UUID": "GPU-aaaaaaaa-1111-2222-3333-444444444444",
            "X-VF-GPU-Expires-At": "2099-01-01T00:00:00Z",
        }
        grant, dup = g1.extract_and_validate(hdrs, key_id="k1", request_digest="dig1", identity=ident)
        self.assertIsNotNone(grant)
        self.assertIsNone(dup)
        self.assertGreater(short_calls, 10)

        # Simulate write failure -> raises FencePersistenceError and poisons guard fail-closed without advancing _min_fencing_token
        fail_file = Path(self.tmp.name) / "guard_fail.json"
        g_fail = LeaseFenceGuard(
            state_path=fail_file,
            write_fn=lambda _fd, _buf: (_ for _ in ()).throw(OSError("enospc")),
        )
        with self.assertRaises(FencePersistenceError):
            g_fail.extract_and_validate(hdrs, key_id="k2", request_digest="dig2", identity=ident)
        self.assertEqual(g_fail._min_fencing_token, 0)
        with self.assertRaises(FencePersistenceError):
            g_fail.reclaim_and_seal(lease_id="lease-s1", fencing_token=31, identity=ident)

        # Corrupted state file on startup -> HTTP 503 fence_state_corrupted
        rt_bad = Path(self.tmp.name) / "rt_bad"
        rt_bad.mkdir(parents=True, exist_ok=True)
        (rt_bad / "gpu_guard_state.json").write_text("{corrupt-json", encoding="utf-8")
        app_bad = create_app(
            root=str(rt_bad),
            token=self.token,
            model_root=self.model_root,
            engine=DummyEngine(),
            model_hash="a" * 64,
            start_worker=False,
        )
        app_bad.state.worker.ready = True
        self.addCleanup(lambda: None if app_bad.state.store.lockfile.closed else app_bad.state.store.close())
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app_bad), base_url="http://indextts2") as bad_client:
            rec_body = json.dumps({"lease_id": "lease-x", "fencing_token": 99}).encode()
            r_bad = await bad_client.post(
                "/internal/v1/gpu-guard/reclaim",
                content=rec_body,
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Content-Type": "application/json",
                    "Content-Length": str(len(rec_body)),
                },
            )
            self.assertEqual(r_bad.status_code, 503)
            self.assertEqual(r_bad.json()["error"]["code"], "fence_state_corrupted")
