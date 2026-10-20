import copy
from datetime import datetime, timezone
import unittest
from unittest.mock import AsyncMock

import httpx

from speech_mcp.clients import Client, Peers
from speech_mcp.domain import Failure, canonical, digest, new_id, strict_json, validate
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
    row = dict(task_id=new_id(), project_id=req["project_id"], request=req, principal=p,
               state={"profile": profile, "profile_sha256": "b" * 64, "capabilities": {"ready": True}, "outputs": []},
               phase="reference", status="pending", created_at=datetime.now(timezone.utc),
               updated_at=datetime.now(timezone.utc), lease_token=new_id())
    return p, req, row


class MemoryStore:
    def __init__(self, row):
        self.row = row
        self.prior = None
        self.saves = 0

    def healthy(self): pass
    def existing(self, p, req):
        if self.prior and req != self.prior["request"]:
            raise Failure("idempotency_conflict", 409)
        return self.prior
    def create(self, p, req, state):
        self.row["state"] = copy.deepcopy(state)
        self.prior = self.row
        return self.row
    def get(self, p, pid, tid):
        if p["organization_id"] != self.row["principal"]["organization_id"] or p["subject_id"] != self.row["principal"]["subject_id"] or pid != self.row["project_id"] or tid != self.row["task_id"]:
            raise Failure("task_not_found", 404)
        return self.row
    def save(self, row, phase=None, status="running", error=None, delay=0):
        self.saves += 1
        row.update(phase=phase or row["phase"], status=status, last_error=error)
        self.row = copy.deepcopy(row)
    def renew(self, row): pass


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


class AsyncTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.p, self.req, self.row = fixture()
        self.store = MemoryStore(self.row)
        self.peers = AsyncMock()
        self.peers.profile.return_value = (self.row["state"]["profile"], "b" * 64)
        self.peers.content.return_value = (b"audio", {"mime": "audio/wav"})
        self.peers.runtime.json.return_value = {"ready": True, "model_manifest_sha256": "a" * 64}
        self.svc = Service(self.store, self.peers, "a" * 64)

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

    async def test_read_checks_live_permission_and_owner(self):
        args = {"project_id": self.req["project_id"], "task_id": self.row["task_id"]}
        with self.assertRaisesRegex(Failure, "task_not_found"):
            await self.svc.call("speech.status", args, self.p | {"subject_id": new_id()})
        self.peers.open.side_effect = Failure("permission_denied", 403)
        with self.assertRaises(Failure): await self.svc.call("speech.status", args, self.p)

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
        entries = [{"name": n, "sha256": digest(n.encode()), "size": len(n), "mime": "application/json" if n.endswith("json") else "audio/wav"}
                   for n in ("original.wav", "editing.wav", "report.json")]
        self.peers.runtime.json.return_value = {"task_id": tid, "model_manifest_sha256": "a" * 64,
                                               "status": "succeeded", "outputs": entries}
        await worker.step(self.row)
        self.assertEqual(self.store.row["phase"], "archive")
        self.peers.runtime.json.reset_mock()
        declarations = []
        async def archive(p, row, entry, data):
            self.assertEqual(digest(data), entry["sha256"])
            declarations.append(copy.deepcopy(entry))
            if len(declarations) == 1: raise Failure("dependency_unavailable", 503, True)
            return {"name": entry["name"]}
        self.peers.archive.side_effect = archive
        self.peers.runtime.request.side_effect = lambda p, method, path, **kw: path.rsplit("/", 1)[-1].encode()
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

    async def test_mcp_boundary_and_retired_authority_route(self):
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


if __name__ == "__main__": unittest.main()
