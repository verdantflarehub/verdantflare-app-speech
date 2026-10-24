import copy
from datetime import datetime, timezone
import unittest
from unittest.mock import AsyncMock
from types import SimpleNamespace

import httpx

from speech_mcp.clients import Client, Peers
from speech_mcp.domain import (
    Failure,
    IndexTTS2Adapter,
    TaskNotFound,
    apply_lexicon,
    build_text_record,
    canonical,
    check_take_staleness,
    compute_input_digest,
    digest,
    new_id,
    normalize_text,
    normalize_world_voice,
    strict_json,
    validate,
)
from speech_mcp.server import applications
from speech_mcp.service import Service
from speech_mcp.worker import Worker


def fixture():
    p = {k: new_id() for k in ("organization_id", "subject_id", "request_id")}
    ref = {k: new_id() for k in ("store_id", "artifact_id", "version_id")}
    req = dict(project_id=new_id(), revision_id=new_id(), voice_profile=ref,
               idempotency_key="test-take-1", utterance_id="N01", text="风停了。", language="zh-CN")
    profile = dict(schema_version=1, voice_id="mengsk", label="Mengsk", engine="indextts2",
                   model_manifest_sha256="a" * 64, reference={"content_ref": ref, "sha256": digest(b"audio")},
                   candidate_state="candidate", listening_review="pending")
    voice_source = {"kind": "project_candidate", "voice_profile": ref, "profile_sha256": "b" * 64}
    text_record = build_text_record(req["text"])
    input_digest = compute_input_digest(text_record["sent"], voice_source, "a" * 64, req)
    row = dict(task_id=new_id(), project_id=req["project_id"], request=req, principal=p,
               state={
                   "profile": profile,
                   "profile_sha256": "b" * 64,
                   "voice_source": voice_source,
                   "text": text_record,
                   "input_digest": input_digest,
                   "timings": {"queue_seconds": 0.0, "load_seconds": 0.0, "inference_seconds": 0.0},
                   "capabilities": {"ready": True},
                   "outputs": [],
               },
               phase="reference", status="pending", created_at=datetime.now(timezone.utc),
               updated_at=datetime.now(timezone.utc), lease_token=new_id())
    return p, req, row


class MemoryStore:
    def __init__(self, row):
        self.row = row
        self.rows = {row["task_id"]: row}
        self.prior = None
        self.saves = 0

    def healthy(self): pass
    def existing(self, p, req):
        if self.prior and req != self.prior["request"]:
            raise Failure("idempotency_conflict", 409)
        return self.prior
    def create(self, p, req, state):
        self.row["state"] = copy.deepcopy(state)
        self.row["request"] = copy.deepcopy(req)
        self.prior = self.row
        self.rows[self.row["task_id"]] = self.row
        return self.row
    def get(self, p, pid, tid):
        r = self.rows.get(tid)
        if not r or p["organization_id"] != r["principal"]["organization_id"] or p["subject_id"] != r["principal"]["subject_id"] or pid != r["project_id"]:
            raise Failure("task_not_found", 404)
        return r
    def get_by_id(self, p, tid):
        r = self.rows.get(tid)
        if not r or p["organization_id"] != r["principal"]["organization_id"] or p["subject_id"] != r["principal"]["subject_id"]:
            raise Failure("task_not_found", 404)
        return r
    def list_for_principal(self, p, project_id=None):
        out = [
            copy.deepcopy(r)
            for r in self.rows.values()
            if r["principal"]["organization_id"] == p["organization_id"]
            and r["principal"]["subject_id"] == p["subject_id"]
            and (project_id is None or r["project_id"] == project_id)
        ]
        out.sort(key=lambda r: r["created_at"], reverse=True)
        return out
    def utterance_takes(self, p, pid, utterance_id):
        return [
            copy.deepcopy(r)
            for r in self.rows.values()
            if r["principal"]["organization_id"] == p["organization_id"]
            and r["principal"]["subject_id"] == p["subject_id"]
            and r["project_id"] == pid
            and r["request"].get("utterance_id") == utterance_id
        ]
    def update_review(self, p, project_id, task_id, decision, note=None, reviewed_revision_id=None):
        r = self.get(p, project_id, task_id)
        if r["status"] != "succeeded":
            raise Failure("result_not_ready", 409)
        uid = r["request"].get("utterance_id")
        if decision == "selected" and uid:
            for other in self.rows.values():
                if (
                    other["task_id"] != task_id
                    and other["project_id"] == project_id
                    and other["request"].get("utterance_id") == uid
                ):
                    other_st = other["state"]
                    if other_st.get("candidate_state") == "selected" or other_st.get("review_state") == "selected":
                        other_st["candidate_state"] = "candidate"
                        other_st["review_state"] = "pending"
        state = dict(r["state"])
        if decision == "selected":
            state["review_state"] = "selected"
            state["candidate_state"] = "selected"
            state["listening_review"] = "approved"
            state["review"] = "approved"
        else:
            state["review_state"] = "needs_revision"
            state["candidate_state"] = "candidate"
            state["listening_review"] = "rejected"
            state["review"] = "rejected"
        state["review_note"] = note or ""
        if reviewed_revision_id:
            state["reviewed_revision_id"] = reviewed_revision_id
        r["state"] = state
        r["updated_at"] = datetime.now(timezone.utc)
        return copy.deepcopy(r)
    def save(self, row, phase=None, status="running", error=None, delay=0):
        self.saves += 1
        row.update(phase=phase or row["phase"], status=status, last_error=error, updated_at=datetime.now(timezone.utc))
        self.row = copy.deepcopy(row)
        self.rows[self.row["task_id"]] = self.row
    def renew(self, row): pass
    def rename_task(self, task_id: str, title: str, p=None):
        if not isinstance(title, str):
            raise Failure("invalid_arguments", 400)
        clean_title = title.strip()
        if not clean_title or len(clean_title) > 120:
            raise Failure("invalid_arguments", 400)
        r = self.rows.get(task_id)
        if not r or (
            p
            and (
                p.get("organization_id") != r["principal"]["organization_id"]
                or p.get("subject_id") != r["principal"]["subject_id"]
            )
        ):
            raise TaskNotFound("task_not_found", 404)
        r["title"] = clean_title
        r["state"]["title"] = clean_title
        r["updated_at"] = datetime.now(timezone.utc)
        return copy.deepcopy(r)
    def delete_task(self, task_id: str, p=None):
        r = self.rows.get(task_id)
        if not r or (
            p
            and (
                p.get("organization_id") != r["principal"]["organization_id"]
                or p.get("subject_id") != r["principal"]["subject_id"]
            )
        ):
            raise TaskNotFound("task_not_found", 404)
        self.rows.pop(task_id, None)
        if self.prior and self.prior.get("task_id") == task_id:
            self.prior = None
        return {"status": "deleted", "task_id": task_id}


class ContractTests(unittest.TestCase):
    def test_unknown_capabilities_and_vectors_rejected(self):
        _, req, _ = fixture()
        validate("create", req)
        for extra in ({"speed": 1.2}, {"target_duration": 8}, {"emotion_vector": [0.4] * 8}, {"emotion_vector": [True] * 8}, {"language": "en-US"}):
            with self.assertRaises(Failure): validate("create", req | extra)

    def test_duplicate_and_nonfinite_json_rejected(self):
        for raw in (b'{"text":1,"text":2}', b'{"x":NaN}', b'[]x'):
            with self.assertRaises(Failure): strict_json(raw)

    def test_fixed_reference_must_belong_to_revision(self):
        _, req, _ = fixture()
        with self.assertRaises(Failure): Peers.contains({"manifest": {"files": []}}, req["voice_profile"])

    def test_emotion_modes_mutually_exclusive(self):
        _, req, _ = fixture()
        emo_ref = {"content_ref": {k: new_id() for k in ("store_id", "artifact_id", "version_id")}, "alpha": 0.7}
        validate("create", req | {"emotion_vector": [0.1] * 8})
        validate("create", req | {"emotion_text": "低沉克制、带一点颤抖"})
        validate("create", req | {"emotion_reference": emo_ref})
        with self.assertRaises(Failure):
            validate("create", req | {"emotion_vector": [0.1] * 8, "emotion_text": "低沉"})
        with self.assertRaises(Failure):
            validate("create", req | {"emotion_text": "低沉", "emotion_reference": emo_ref})
        with self.assertRaises(Failure):
            validate("create", req | {"emotion_vector": [0.1] * 8, "emotion_reference": emo_ref})

    def test_world_voice_and_voice_profile_mutually_exclusive(self):
        _, req, _ = fixture()
        wv = {"asset_id": new_id(), "version_id": new_id()}
        req_wv = {k: v for k, v in req.items() if k != "voice_profile"} | {"world_voice": wv}
        validate("create", req_wv)
        with self.assertRaises(Failure):
            validate("create", req | {"world_voice": wv})
        with self.assertRaises(Failure):
            validate("create", {k: v for k, v in req.items() if k != "voice_profile"})

    def test_conservative_text_normalization_and_lexicon(self):
        raw = "  谢小娥  在东市  寻人\r\n第二行 "
        self.assertEqual(normalize_text(raw), "谢小娥 在东市 寻人\n第二行")
        entries = [
            {"term": "解", "replacement": "懈"},
            {"term": "解县", "replacement": "懈县"},
            {"term": "单于", "replacement": "缠于", "note": "古音"},
        ]
        rec = build_text_record("谢小娥去解县见单于。", entries)
        self.assertEqual(rec["source"], "谢小娥去解县见单于。")
        self.assertEqual(rec["sent"], "谢小娥去懈县见缠于。")
        self.assertEqual([h["term"] for h in rec["lexicon_hits"]], ["解县", "单于"])

    def test_input_digest_and_take_staleness_detection(self):
        aid, v1, v2 = new_id(), new_id(), new_id()
        vs_old = {"kind": "world", "world_voice": {"asset_id": aid, "version_id": v1}, "reference_sha256": "1" * 64}
        vs_new = {"kind": "world", "world_voice": {"asset_id": aid, "version_id": v2}, "reference_sha256": "2" * 64}
        gen = {
            "input_digest": compute_input_digest("风停了。", vs_old, "a" * 64, {}),
            "voice_source": vs_old,
            "model_manifest_sha256": "a" * 64,
            "text": {"source": "风停了。", "sent": "风停了。", "lexicon_hits": []},
        }
        ok = check_take_staleness(gen, vs_old, current_sent_text="风停了。", current_model_manifest_sha256="a" * 64)
        self.assertFalse(ok["stale"])
        stale_voice = check_take_staleness(gen, vs_new, current_sent_text="风停了。")
        self.assertTrue(stale_voice["stale"])
        self.assertIn("world_voice_version_changed", stale_voice["reasons"])


class AsyncTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.p, self.req, self.row = fixture()
        self.store = MemoryStore(self.row)
        self.peers = AsyncMock()
        self.peers.open.return_value = {"head_revision_id": self.req["revision_id"], "can_write": True}
        self.peers.profile.return_value = (self.row["state"]["profile"], "b" * 64)
        self.peers.content.return_value = (b"audio", {"mime": "audio/wav"})
        self.peers.runtime.json.return_value = {"ready": True, "model_manifest_sha256": "a" * 64}
        self.svc = Service(self.store, self.peers, "a" * 64)

    async def test_capabilities_returns_engines_declaration(self):
        res = await self.svc.call("speech.capabilities", {}, self.p)
        validate("capabilities_result", res)
        self.assertEqual(len(res["engines"]), 1)
        eng = res["engines"][0]
        self.assertEqual(eng["engine"], "indextts2")
        self.assertTrue(eng["ready"])
        self.assertEqual(eng["emotion_inputs"], ["emotion_vector", "emotion_text", "emotion_reference"])

    async def test_create_idempotency_and_model_mismatch(self):
        first = await self.svc.call("speech.create", self.req, self.p)
        self.peers.profile.reset_mock()
        second = await self.svc.call("speech.create", self.req, self.p)
        self.assertEqual(first["task_id"], second["task_id"])
        self.peers.profile.assert_not_awaited()
        with self.assertRaisesRegex(Failure, "idempotency_conflict"):
            await self.svc.call("speech.create", self.req | {"text": "别出声。"}, self.p)
        self.store.prior = None
        self.peers.runtime.json.return_value = {"ready": True, "model_manifest_sha256": "c" * 64}
        with self.assertRaisesRegex(Failure, "model_version_changed"):
            await self.svc.call("speech.create", self.req, self.p)

    async def test_world_voice_create_and_emotion_text(self):
        aid, vid = new_id(), new_id()
        ref = {k: new_id() for k in ("store_id", "artifact_id", "version_id")}
        profile_synth = {
            "schema_version": 1,
            "voice_id": "xie_xiaoe",
            "label": "谢小娥",
            "engine": "indextts2",
            "model_manifest_sha256": "a" * 64,
            "reference": {
                "content_ref": ref,
                "sha256": digest(b"audio"),
                "asset_id": aid,
                "asset_version_id": vid,
            },
            "candidate_state": "selected",
            "listening_review": "approved",
        }
        voice_source = {
            "kind": "world",
            "world_voice": {"asset_id": aid, "version_id": vid},
            "reference_sha256": digest(b"audio"),
        }
        self.peers.world_voice.return_value = (profile_synth, voice_source, {})
        self.peers.asset_content.return_value = (b"audio", {"mime": "audio/wav"})
        req_wv = {k: v for k, v in self.req.items() if k != "voice_profile"} | {
            "world_voice": {"asset_id": aid, "version_id": vid},
            "emotion_text": "克制冷静",
        }
        created = await self.svc.call("speech.create", req_wv, self.p)
        self.assertEqual(created["status"], "pending")
        self.assertEqual(self.store.row["state"]["voice_source"]["kind"], "world")
        self.peers.asset_content.assert_awaited_once()

    async def test_read_checks_live_permission_and_owner(self):
        args = {"project_id": self.req["project_id"], "task_id": self.row["task_id"]}
        with self.assertRaisesRegex(Failure, "task_not_found"):
            await self.svc.call("speech.status", args, self.p | {"subject_id": new_id()})
        self.peers.open.side_effect = Failure("permission_denied", 403)
        with self.assertRaises(Failure): await self.svc.call("speech.status", args, self.p)

    async def test_managed_cold_start_accepts_once_without_runtime_access(self):
        self.svc.gpu = SimpleNamespace(enabled=True)
        self.peers.runtime.json.side_effect = Failure("dependency_unavailable", 503, True)
        first = await self.svc.call("speech.create", self.req, self.p)
        second = await self.svc.call("speech.create", self.req, self.p)
        self.assertEqual(first["task_id"], second["task_id"])
        self.peers.runtime.json.assert_not_awaited()
        self.peers.content.assert_awaited_once()
        self.assertIsNone(self.store.row["state"]["capabilities"])

    async def test_managed_cold_start_still_rejects_bad_reference(self):
        self.svc.gpu = SimpleNamespace(enabled=True)
        self.peers.content.return_value = (b"wrong audio", {"mime": "audio/wav"})
        with self.assertRaisesRegex(Failure, "reference_integrity_failed"):
            await self.svc.call("speech.create", self.req, self.p)
        self.assertIsNone(self.store.prior)
        self.peers.runtime.json.assert_not_awaited()

    async def test_reference_wakes_before_upload_and_records_real_capabilities(self):
        worker = Worker(self.svc)
        events = []
        async def wake(row): events.append("lease-and-ready")
        worker._ensure_gpu = wake
        async def runtime(p, method, path, **kw):
            events.append(path)
            if path == "/v1/capabilities":
                return {"ready": True, "model_manifest_sha256": "a" * 64, "observed": True}
            return {}
        self.peers.runtime.json.side_effect = runtime
        self.row["state"]["capabilities"] = None
        await worker.step(self.row)
        self.assertEqual(events, ["lease-and-ready", "/v1/capabilities", "/v1/references/" + digest(b"audio")])
        self.assertEqual(self.store.row["phase"], "submit")
        self.assertTrue(self.store.row["state"]["capabilities"]["observed"])

    async def test_loading_model_phase_and_timeout(self):
        self.svc.gpu = SimpleNamespace(
            enabled=True,
            ready_seconds=0.01,
            acquire_for_queue=lambda: None,
            scale_to=lambda n: None,
            renew=lambda: None,
            release=lambda: None,
        )
        worker = Worker(self.svc)
        self.peers.runtime.json.side_effect = Failure("dependency_unavailable", 503, True)
        with self.assertRaisesRegex(Failure, "model_load_timeout"):
            await worker.step(self.row)
        self.assertEqual(self.store.row["phase"], "loading_model")
        self.assertGreaterEqual(self.row["state"]["timings"]["load_seconds"], 0.0)

    async def test_reference_blocks_upload_until_ready_and_matching_model(self):
        worker = Worker(self.svc)
        worker._ensure_gpu = AsyncMock()
        for caps, error in [({"ready": False, "model_manifest_sha256": "a" * 64}, "model_not_ready"),
                            ({"ready": True, "model_manifest_sha256": "c" * 64}, "model_version_changed")]:
            self.peers.runtime.json.reset_mock()
            self.peers.runtime.json.return_value = caps
            with self.assertRaisesRegex(Failure, error): await worker.step(self.row)
            self.peers.runtime.json.assert_awaited_once()
        self.assertEqual(self.store.saves, 0)

    async def test_submission_response_loss_reuses_original_key(self):
        worker = Worker(self.svc)
        self.row.update(phase="submit")
        self.row["state"]["runtime_inputs"] = {"idempotency_key": "speech-" + self.row["task_id"]}
        calls = []
        async def runtime(p, method, path, payload=None, **kw):
            if path == "/v1/capabilities": return {"model_manifest_sha256": "a" * 64}
            calls.append(copy.deepcopy(payload))
            if len(calls) == 1: raise Failure("dependency_unavailable", 503, True)
            return {"task_id": new_id()}
        self.peers.runtime.json.side_effect = runtime
        await worker.process(self.row)
        self.assertEqual(self.store.row["phase"], "submit")
        await worker.process(copy.deepcopy(self.store.row))
        self.assertEqual(calls[0], calls[1])
        self.assertEqual(self.store.row["phase"], "inference")

    async def test_archive_failure_recovers_without_inference_or_new_write_id(self):
        worker = Worker(self.svc)
        self.row.update(phase="inference", status="running")
        tid = new_id()
        self.row["state"]["runtime_task_id"] = tid
        report_bytes = canonical({
            "schema_version": 2,
            "resolved_emotion_vector": [0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.5],
            "metrics": {"duration_seconds": 1.6, "cer": 0.0},
            "verdict": {"auto_qc": "pass", "failure_reasons": []},
        }).encode()
        payloads = {
            "original.wav": b"original.wav",
            "editing.wav": b"editing.wav",
            "report.json": report_bytes,
        }
        entries = [{"name": n, "sha256": digest(b), "size": len(b), "mime": "application/json" if n.endswith("json") else "audio/wav"}
                   for n, b in payloads.items()]
        self.peers.runtime.json.return_value = {"task_id": tid, "model_manifest_sha256": "a" * 64,
                                               "status": "succeeded", "outputs": entries}
        await worker.step(self.row)
        self.assertEqual(self.store.row["phase"], "archive")
        self.peers.runtime.json.reset_mock()
        declarations = []
        async def archive(p, row, entry, data):
            self.assertEqual(digest(data), entry["sha256"])
            if entry["name"] == "generation.json":
                gen_doc = strict_json(data)
                validate("generation_record", gen_doc)
                self.assertEqual(gen_doc["resolved_emotion_vector"], [0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.5])
            declarations.append(copy.deepcopy(entry))
            if len(declarations) == 1: raise Failure("dependency_unavailable", 503, True)
            return {"name": entry["name"], "content_ref": self.req["voice_profile"], "sha256": entry["sha256"], "size": entry["size"], "mime": entry["mime"]}
        self.peers.archive.side_effect = archive
        self.peers.runtime.request.side_effect = lambda p, method, path, **kw: payloads[path.rsplit("/", 1)[-1]]
        for _ in range(6): await worker.process(copy.deepcopy(self.store.row))
        self.assertEqual(declarations[0], declarations[1])
        self.assertEqual(self.store.row["status"], "succeeded")
        self.assertEqual(len(self.store.row["state"]["outputs"]), 4)
        self.peers.runtime.json.assert_not_awaited()

    async def test_runtime_uncertainty_is_terminal(self):
        self.row.update(phase="inference", status="running")
        tid = new_id()
        self.row["state"]["runtime_task_id"] = tid
        self.peers.runtime.json.return_value = {"task_id": tid, "model_manifest_sha256": "a" * 64,
                                               "status": "failed", "error": {"code": "recovery_required"}}
        await Worker(self.svc).process(self.row)
        self.assertEqual(self.store.row["status"], "failed")
        self.assertEqual(self.store.row["last_error"], "recovery_required")

    async def test_mcp_boundary_and_dashboard_routes(self):
        self.row["status"] = "succeeded"
        self.row["phase"] = "complete"
        mcp = applications(self.svc, "m" * 32, start_worker=False)
        headers = {"Authorization": "Bearer " + "m" * 32,
                   "X-Organization-Id": self.p["organization_id"], "X-User-Id": self.p["subject_id"],
                   "X-Request-Id": self.p["request_id"]}
        body = {"jsonrpc": "2.0", "id": self.p["request_id"], "method": "tools/call",
                "params": {"name": "speech.capabilities", "arguments": {}}}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=mcp), base_url="https://test") as c:
            self.assertEqual((await c.get("/health")).status_code, 200)
            self.assertEqual((await c.post("/mcp", json=body)).status_code, 401)
            self.assertEqual((await c.post("/mcp", json=body, headers=headers)).status_code, 200)
            self.assertEqual((await c.post("/mcp", json=body, headers=headers | {"X-Project-Id": new_id()})).status_code, 400)
            self.assertEqual((await c.post("/internal/artifacts/authorize", json={}, headers=headers)).status_code, 404)

            # Dashboard bundle routes
            decl = await c.get("/internal/dashboard-declaration", headers=headers)
            self.assertEqual(decl.status_code, 200)
            self.assertEqual(decl.json()["appId"], "speech-mcp")
            idx = await c.get("/internal/dashboard-index", headers=headers)
            self.assertEqual(idx.status_code, 200)
            self.assertIn("index.html", idx.json()["files"])
            html_res = await c.get("/internal/dashboard/assets/index.html", headers=headers)
            self.assertEqual(html_res.status_code, 200)
            self.assertIn(b"dashboard.mjs", html_res.content)

            # Dashboard API routes
            ov = await c.get("/api/overview", headers=headers)
            self.assertEqual(ov.status_code, 200)
            self.assertEqual(ov.json()["kpis"]["total_tasks"], 1)

            tasks = await c.get("/api/tasks", headers=headers)
            self.assertEqual(tasks.status_code, 200)
            self.assertEqual(len(tasks.json()["items"]), 1)

            rev = await c.post(
                f"/api/tasks/{self.row['task_id']}/review",
                headers=headers,
                json={
                    "project_id": self.req["project_id"],
                    "task_id": self.row["task_id"],
                    "expected_revision_id": self.req["revision_id"],
                    "decision": "selected",
                    "note": "语气自然，节奏准确",
                },
            )
            self.assertEqual(rev.status_code, 200)
            self.assertEqual(rev.json()["summary"]["review"], "selected")
            self.assertEqual(rev.json()["review_note"], "语气自然，节奏准确")

            # Stale expected_revision_id returns 409
            conflict = await c.post(
                f"/api/tasks/{self.row['task_id']}/review",
                headers=headers,
                json={
                    "project_id": self.req["project_id"],
                    "task_id": self.row["task_id"],
                    "expected_revision_id": new_id(),
                    "decision": "needs_revision",
                    "note": "重录",
                },
            )
            self.assertEqual(conflict.status_code, 409)

            # Cross-user access denied
            other_headers = headers | {"X-User-Id": new_id()}
            denied = await c.get(f"/api/tasks/{self.row['task_id']}", headers=other_headers)
            self.assertEqual(denied.status_code, 404)

    async def test_internal_client_rejects_redirects_and_checks_digest(self):
        seen = []
        def handler(request):
            seen.append(str(request.url))
            return httpx.Response(302, headers={"location": "https://other.test"}, json={})
        c = Client("https://internal.test", "secret", transport=httpx.MockTransport(handler))
        with self.assertRaises(Failure): await c.json(self.p, "GET", "/health")
        self.assertEqual(seen, ["https://internal.test/health"])
        await c.close()

    async def test_project_read_omits_empty_optional_write_flag(self):
        def handler(request):
            body = strict_json(request.content)
            if body.get("require_write") is False:
                return httpx.Response(400, json={"code": "INVALID_ARGUMENT"})
            return httpx.Response(200, json={"manifest": {"files": []}})
        c = Client("https://internal.test", "secret", transport=httpx.MockTransport(handler))
        peers = Peers(c, None, None)
        for write in (False, True):
            result = await peers.open(self.p, self.req["project_id"], write=write)
            self.assertEqual(result, {"manifest": {"files": []}})
        await c.close()

    async def test_peers_world_voice_validation_and_consent_privacy(self):
        aid, vid = new_id(), new_id()
        rv_ref = {k: new_id() for k in ("store_id", "artifact_id", "version_id")}
        audio_ref = {k: new_id() for k in ("store_id", "artifact_id", "version_id")}
        audition_ref = {k: new_id() for k in ("store_id", "artifact_id", "version_id")}
        consent_ref = {k: new_id() for k in ("store_id", "artifact_id", "version_id")}
        opened = {"manifest": {"asset_refs": [{"asset_id": aid, "asset_version_id": vid}]}}
        world_client = AsyncMock()
        world_client.json.return_value = {
            "manifest": {
                "files": [
                    {"path": "reference-voice.json", "content_ref": rv_ref},
                    {"path": "reference.wav", "content_ref": audio_ref},
                    {"path": "audition.wav", "content_ref": audition_ref},
                    {"path": "consent/authorization.pdf", "content_ref": consent_ref},
                ]
            }
        }
        peers = Peers(AsyncMock(), AsyncMock(), AsyncMock(), world=world_client)
        base_doc = {
            "schema_version": 1,
            "kind": "reference-voice",
            "voice_id": "xie_xiaoe",
            "label": "谢小娥",
            "speaker": "谢小娥",
            "language": "zh-CN",
            "consent": {
                "source_kind": "human_recording",
                "basis": "actor_signed_release",
                "scope": "project_and_world",
                "evidence_path": "consent/authorization.pdf",
                "recorded_at": "2026-10-21T00:00:00Z",
            },
            "reference": {
                "path": "reference.wav",
                "sha256": digest(b"audio"),
                "duration_seconds": 8.2,
                "sample_rate_hz": 24000,
            },
            "audition": {
                "path": "audition.wav",
                "text": "风停了，东市的人声却还没散。",
                "technical_status": "passed",
                "listening_review": "approved",
            },
            "engines": [
                {
                    "service_id": "speech",
                    "engine": "indextts2",
                    "model_manifest_sha256": "a" * 64,
                    "verified_at": "2026-10-21T00:00:00Z",
                    "result": "passed",
                }
            ],
        }
        fetched_refs = []
        async def fake_asset_content(p, ref, a_id, v_id, limit):
            fetched_refs.append(ref)
            return canonical(base_doc).encode(), {"mime": "application/json"}
        peers.asset_content = fake_asset_content

        # 1. Valid world_voice resolves without reading consent.evidence_path
        profile, vs, _ = await peers.world_voice(
            self.p, opened, {"asset_id": aid, "version_id": vid}, engine="indextts2", model_manifest_sha256="a" * 64
        )
        self.assertEqual(profile["voice_id"], "xie_xiaoe")
        self.assertEqual(vs["kind"], "world")
        self.assertEqual(fetched_refs, [rv_ref])

        # 2. Asset outside revision is rejected
        with self.assertRaisesRegex(Failure, "reference_outside_revision"):
            await peers.world_voice(self.p, {"manifest": {"asset_refs": []}}, {"asset_id": aid, "version_id": vid})

        # 3. Unverified engine is rejected
        bad_doc = copy.deepcopy(base_doc)
        bad_doc["engines"] = [{"service_id": "speech", "engine": "other_engine", "model_manifest_sha256": "a" * 64, "verified_at": "2026-10-21T00:00:00Z", "result": "passed"}]
        async def fake_bad_content(p, ref, a_id, v_id, limit):
            return canonical(bad_doc).encode(), {"mime": "application/json"}
        peers.asset_content = fake_bad_content
        with self.assertRaisesRegex(Failure, "unverified_voice_engine"):
            await peers.world_voice(self.p, opened, {"asset_id": aid, "version_id": vid}, engine="indextts2")


if __name__ == "__main__": unittest.main()
