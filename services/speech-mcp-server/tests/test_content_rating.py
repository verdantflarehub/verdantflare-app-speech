from __future__ import annotations

import copy
from datetime import datetime, timezone
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock

import httpx

_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from speech_mcp.content_rating import ContentRatingError, ContentRatingStore
from speech_mcp.domain import (
    SPEECH_TOOLS_SCHEMA,
    TaskNotFound,
    build_text_record,
    compute_input_digest,
    digest,
    new_id,
)
from speech_mcp.server import applications
from speech_mcp.service import Service

from tests.test_speech import MemoryStore


def _make_speech_task(p: dict[str, str], project_id: str, revision_id: str, utterance_id: str, text: str):
    voice_ref = {k: new_id() for k in ("store_id", "artifact_id", "version_id")}
    orig_ref = {k: new_id() for k in ("store_id", "artifact_id", "version_id")}
    edit_ref = {k: new_id() for k in ("store_id", "artifact_id", "version_id")}
    req = {
        "project_id": project_id,
        "revision_id": revision_id,
        "voice_profile": voice_ref,
        "idempotency_key": f"idem-{utterance_id}",
        "utterance_id": utterance_id,
        "text": text,
        "language": "zh-CN",
    }
    profile = {
        "schema_version": 1,
        "voice_id": "xie_xiaoe",
        "label": "谢小娥",
        "engine": "indextts2",
        "model_manifest_sha256": "a" * 64,
        "reference": {"content_ref": voice_ref, "sha256": digest(b"audio")},
        "candidate_state": "candidate",
        "listening_review": "pending",
    }
    voice_source = {"kind": "project_candidate", "voice_profile": voice_ref, "profile_sha256": "b" * 64}
    text_record = build_text_record(text)
    input_digest = compute_input_digest(text_record["sent"], voice_source, "a" * 64, req)
    outputs = [
        {
            "name": "original.wav",
            "content_ref": orig_ref,
            "sha256": digest(b"orig"),
            "size": 4,
            "mime": "audio/wav",
        },
        {
            "name": "editing.wav",
            "content_ref": edit_ref,
            "sha256": digest(b"edit"),
            "size": 4,
            "mime": "audio/wav",
        },
    ]
    now = datetime.now(timezone.utc)
    row = {
        "task_id": new_id(),
        "project_id": project_id,
        "request": req,
        "principal": copy.deepcopy(p),
        "state": {
            "profile": profile,
            "profile_sha256": "b" * 64,
            "voice_source": voice_source,
            "text": text_record,
            "input_digest": input_digest,
            "timings": {"queue_seconds": 0.1, "load_seconds": 0.2, "inference_seconds": 1.2},
            "capabilities": {"ready": True},
            "outputs": outputs,
        },
        "phase": "complete",
        "status": "succeeded",
        "created_at": now,
        "updated_at": now,
        "lease_token": None,
    }
    return row, edit_ref


class TestContentRatingStore(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "speech_ratings.db"
        self.store = ContentRatingStore(self.db_path, domain="speech")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_add_only_and_human_override_lifecycle(self) -> None:
        self.assertEqual(self.store.domain, "speech")

        # 1. Rate general
        r1 = self.store.rate_content(
            version_id="ver-1",
            project_id="proj-1",
            rating="general",
            reason="clean narration line",
            rubric_version="rubric v1.0",
        )
        self.assertEqual(r1["rating"], "general")
        self.assertFalse(r1["ignored"])

        # 2. Upgrade general -> nsfw
        r2 = self.store.rate_content(
            version_id="ver-1",
            project_id="proj-1",
            rating="nsfw",
            reason="explicit_sexual_audio",
            rubric_version="rubric v1.0",
        )
        self.assertEqual(r2["rating"], "nsfw")
        self.assertFalse(r2["ignored"])
        self.assertTrue(self.store.is_nsfw("ver-1"))

        # 3. Attempt to downgrade nsfw -> general via MCP (must be ignored)
        r3 = self.store.rate_content(
            version_id="ver-1",
            project_id="proj-1",
            rating="general",
            reason="attempt to downgrade via mcp",
            rubric_version="rubric v1.0",
        )
        self.assertEqual(r3["rating"], "nsfw")
        self.assertTrue(r3["ignored"])
        self.assertTrue(self.store.is_nsfw("ver-1"))

        # 4. Clear by human
        r4 = self.store.clear_rating(
            version_id="ver-1",
            project_id="proj-1",
            actor_user_id="user-admin",
            note="false positive dramatic sigh",
        )
        self.assertEqual(r4["rating"], "general")
        self.assertEqual(r4["source"], "human")
        self.assertTrue(r4["human_cleared"])
        self.assertFalse(self.store.is_nsfw("ver-1"))

        # 4b. AI attempts to re-rate nsfw after human clear -> ignored!
        r4b = self.store.rate_content(
            version_id="ver-1",
            project_id="proj-1",
            rating="nsfw",
            reason="ai re-rate attempt",
            rubric_version="rubric v1.0",
        )
        self.assertEqual(r4b["rating"], "general")
        self.assertTrue(r4b["ignored"])
        self.assertFalse(self.store.is_nsfw("ver-1"))

        # 5. Mark by human
        r5 = self.store.mark_rating(
            version_id="ver-1",
            project_id="proj-1",
            actor_user_id="user-admin",
            reason="manual 18+ mark",
        )
        self.assertEqual(r5["rating"], "nsfw")
        self.assertEqual(r5["source"], "human")
        self.assertTrue(self.store.is_nsfw("ver-1"))

        events = self.store.list_events("ver-1")
        self.assertEqual(
            [e["event_type"] for e in events],
            ["rate", "rate", "rate", "clear", "ignored_after_clear", "rate"],
        )

    def test_validation_rejects_invalid_rating_or_reason(self) -> None:
        with self.assertRaises(ContentRatingError) as ctx_clear:
            self.store.rate_content(
                version_id="ver-1",
                project_id="proj-1",
                rating="unrated",
                reason="attempt clear",
            )
        self.assertEqual(ctx_clear.exception.error_code, "RATING_CLEAR_FORBIDDEN")

        with self.assertRaises(ContentRatingError):
            self.store.rate_content(
                version_id="ver-1",
                project_id="proj-1",
                rating="r18",
                reason="bad enum",
            )
        with self.assertRaises(ContentRatingError):
            self.store.rate_content(
                version_id="ver-1",
                project_id="proj-1",
                rating="nsfw",
                reason="   ",
            )


class TestSpeechMcpAndHttpContentRating(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "speech_ratings.db"
        self.ratings = ContentRatingStore(self.db_path, domain="speech")
        self.p = {k: new_id() for k in ("organization_id", "subject_id", "request_id")}
        self.project_id = new_id()
        self.revision_id = new_id()

        self.row_gen, self.ref_gen = _make_speech_task(
            self.p, self.project_id, self.revision_id, "U01", "风停了，东市的人声却还没散。"
        )
        self.row_nsfw, self.ref_nsfw = _make_speech_task(
            self.p, self.project_id, self.revision_id, "U02", "成人向台词测试样本。"
        )

        self.store = MemoryStore(self.row_gen)
        self.store.rows[self.row_nsfw["task_id"]] = self.row_nsfw
        self.peers = AsyncMock()
        self.peers.open.return_value = {"head_revision_id": self.revision_id, "can_write": True}
        self.peers.runtime.json.return_value = {"ready": True, "model_manifest_sha256": "a" * 64}
        self.peers.task_output_content.return_value = (b"RIFF\x24\x00\x00\x00WAVEfmt ", {"mime": "audio/wav"})
        self.svc = Service(self.store, self.peers, "a" * 64, ratings=self.ratings)
        self.token = "s" * 32
        self.app = applications(self.svc, self.token, start_worker=False)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    async def test_capabilities_and_tools_include_rate_content(self) -> None:
        tool_names = {t["name"] for t in SPEECH_TOOLS_SCHEMA}
        self.assertIn("speech.rate_content", tool_names)
        self.assertIn("speech.list", tool_names)

        caps = await self.svc.call("speech.capabilities", {}, self.p)
        self.assertIn("content_rating", caps)
        self.assertTrue(caps["content_rating"]["supported"])
        self.assertEqual(caps["content_rating"]["tool"], "speech.rate_content")
        self.assertFalse(caps["content_rating"]["mcp_clear_supported"])

    async def test_mcp_rate_content_hides_from_mcp_and_http_supports_local_switch_and_clear(self) -> None:
        # 1. Initially both tasks are visible in MCP speech.list
        initial_list = await self.svc.call("speech.list", {"project_id": self.project_id}, self.p)
        self.assertEqual(initial_list["total"], 2)

        # 2. Rate second speech take as nsfw via MCP speech.rate_content
        rate_res = await self.svc.call(
            "speech.rate_content",
            {
                "version_id": self.ref_nsfw["version_id"],
                "rating": "nsfw",
                "reason": "explicit_adult_dialogue",
                "rubric_version": "rubric v1.0",
            },
            self.p,
        )
        self.assertEqual(rate_res["rating"], "nsfw")
        self.assertFalse(rate_res["ignored"])

        # 3. Attempting to downgrade via MCP general must not clear nsfw
        downgrade_res = await self.svc.call(
            "speech.rate_content",
            {
                "version_id": self.ref_nsfw["version_id"],
                "rating": "general",
                "reason": "try downgrade",
                "rubric_version": "rubric v1.0",
            },
            self.p,
        )
        self.assertEqual(downgrade_res["rating"], "nsfw")
        self.assertTrue(downgrade_res["ignored"])

        # 4. MCP list now hides the nsfw task and decrements total (no placeholder)
        after_list = await self.svc.call("speech.list", {"project_id": self.project_id}, self.p)
        self.assertEqual(after_list["total"], 1)
        self.assertEqual([t["task_id"] for t in after_list["tasks"]], [self.row_gen["task_id"]])

        # 5. MCP status and result raise TaskNotFound for the nsfw task
        with self.assertRaises(TaskNotFound):
            await self.svc.call(
                "speech.status",
                {"project_id": self.project_id, "task_id": self.row_nsfw["task_id"]},
                self.p,
            )
        with self.assertRaises(TaskNotFound):
            await self.svc.call(
                "speech.result",
                {"project_id": self.project_id, "task_id": self.row_nsfw["task_id"]},
                self.p,
            )

        # 6. Unknown or cross-project version_id raises TaskNotFound
        with self.assertRaises(TaskNotFound):
            await self.svc.call(
                "speech.rate_content",
                {"version_id": new_id(), "rating": "general", "reason": "missing"},
                self.p,
            )
        with self.assertRaises(TaskNotFound):
            await self.svc.call(
                "speech.rate_content",
                {
                    "version_id": self.ref_nsfw["version_id"],
                    "rating": "general",
                    "reason": "wrong project",
                    "project_id": new_id(),
                },
                self.p,
            )

        headers = {
            "Authorization": f"Bearer {self.token}",
            "X-Organization-Id": self.p["organization_id"],
            "X-User-Id": self.p["subject_id"],
            "X-Request-Id": self.p["request_id"],
        }

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app),
            base_url="https://test",
        ) as client:
            # HTTP /api/tasks and /speech/api/tasks without X-VF-Include-NSFW hide the nsfw task
            resp_default = await client.get(f"/api/tasks?project_id={self.project_id}", headers=headers)
            self.assertEqual(resp_default.status_code, 200)
            self.assertEqual(resp_default.json()["total"], 1)

            resp_prefixed = await client.get(f"/speech/api/tasks?project_id={self.project_id}", headers=headers)
            self.assertEqual(resp_prefixed.status_code, 200)
            self.assertEqual(resp_prefixed.json()["total"], 1)

            # HTTP /api/tasks/stats and /speech/api/tasks/stats without X-VF-Include-NSFW exclude nsfw
            resp_stats = await client.get("/api/tasks/stats", headers=headers)
            self.assertEqual(resp_stats.status_code, 200)
            self.assertEqual(resp_stats.json()["total"], 1)

            resp_stats_pref = await client.get("/speech/api/tasks/stats", headers=headers)
            self.assertEqual(resp_stats_pref.status_code, 200)
            self.assertEqual(resp_stats_pref.json()["total"], 1)

            # HTTP /artifacts/{id}/content and /speech/artifacts/{id}/content without X-VF-Include-NSFW return 404
            resp_art_hidden = await client.get(f"/artifacts/{self.ref_nsfw['version_id']}/content")
            self.assertEqual(resp_art_hidden.status_code, 404)
            resp_art_hidden_pref = await client.get(f"/speech/artifacts/{self.ref_nsfw['version_id']}/content")
            self.assertEqual(resp_art_hidden_pref.status_code, 404)

            # HTTP /api/tasks with X-VF-Include-NSFW: 1 includes the nsfw task with UI fields
            resp_inc = await client.get(
                f"/api/tasks?project_id={self.project_id}",
                headers={**headers, "X-VF-Include-NSFW": "1"},
            )
            self.assertEqual(resp_inc.status_code, 200)
            inc_data = resp_inc.json()
            self.assertEqual(inc_data["total"], 2)
            nsfw_item = next(t for t in inc_data["tasks"] if t["task_id"] == self.row_nsfw["task_id"])
            self.assertEqual(nsfw_item["content_rating"]["rating"], "nsfw")
            self.assertTrue(nsfw_item["can_clear_rating"])
            self.assertFalse(nsfw_item["can_mark_rating"])

            # HTTP /api/tasks/stats with X-VF-Include-NSFW: 1 includes the nsfw task
            resp_stats_inc = await client.get(
                "/api/tasks/stats",
                headers={**headers, "X-VF-Include-NSFW": "1"},
            )
            self.assertEqual(resp_stats_inc.status_code, 200)
            self.assertEqual(resp_stats_inc.json()["total"], 2)

            # HTTP /artifacts/{id}/content with X-VF-Include-NSFW: 1 returns 200
            resp_art_shown = await client.get(
                f"/artifacts/{self.ref_nsfw['version_id']}/content",
                headers={"X-VF-Include-NSFW": "1"},
            )
            self.assertEqual(resp_art_shown.status_code, 200)

            # MCP JSON-RPC endpoint ignores X-VF-Include-NSFW: 1 and still hides nsfw task
            mcp_rpc = await client.post(
                "/mcp",
                headers={
                    **headers,
                    "X-Project-Id": self.project_id,
                    "X-VF-Include-NSFW": "1",
                },
                json={
                    "jsonrpc": "2.0",
                    "id": self.p["request_id"],
                    "method": "tools/call",
                    "params": {
                        "name": "speech.status",
                        "arguments": {
                            "project_id": self.project_id,
                            "task_id": self.row_nsfw["task_id"],
                        },
                    },
                },
            )
            self.assertEqual(mcp_rpc.status_code, 200)
            self.assertTrue(mcp_rpc.json()["result"]["isError"])
            self.assertEqual(mcp_rpc.json()["result"]["structuredContent"]["error"]["code"], "task_not_found")

            # Unauthenticated call to internal clear endpoint returns 401
            resp_unauth = await client.post(
                f"/internal/content-ratings/{self.ref_nsfw['version_id']}/clear",
                json={"note": "unauthorized attempt"},
            )
            self.assertEqual(resp_unauth.status_code, 401)

            # Internal human clear endpoint restores general visibility
            resp_clear = await client.post(
                f"/internal/content-ratings/{self.ref_nsfw['version_id']}/clear",
                json={"note": "误标撤销"},
                headers=headers,
            )
            self.assertEqual(resp_clear.status_code, 200)
            self.assertEqual(resp_clear.json()["rating"], "general")
            self.assertEqual(resp_clear.json()["source"], "human")
            self.assertTrue(resp_clear.json()["human_cleared"])

            # Now MCP list and status see both tasks again
            restored_list = await self.svc.call("speech.list", {"project_id": self.project_id}, self.p)
            self.assertEqual(restored_list["total"], 2)
            status_res = await self.svc.call(
                "speech.status",
                {"project_id": self.project_id, "task_id": self.row_nsfw["task_id"]},
                self.p,
            )
            self.assertEqual(status_res["status"], "succeeded")

            # Internal human mark endpoint (/speech/internal/...) marks nsfw again
            resp_mark = await client.post(
                f"/speech/internal/content-ratings/{self.ref_nsfw['version_id']}/mark",
                json={"reason": "人工复核确认 18+"},
                headers=headers,
            )
            self.assertEqual(resp_mark.status_code, 200)
            self.assertEqual(resp_mark.json()["rating"], "nsfw")
            self.assertEqual(resp_mark.json()["source"], "human")

            marked_list = await self.svc.call("speech.list", {"project_id": self.project_id}, self.p)
            self.assertEqual(marked_list["total"], 1)


if __name__ == "__main__":
    unittest.main()
