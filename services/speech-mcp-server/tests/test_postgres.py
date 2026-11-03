"""Run against the 5090 development database, never start a local service."""
from concurrent.futures import ThreadPoolExecutor
import os
import unittest

import psycopg
from psycopg.conninfo import make_conninfo

from speech_mcp.domain import Failure, new_id
from speech_mcp.store import Store


@unittest.skipUnless(os.environ.get("SPEECH_TEST_DATABASE_URL"), "5090 database required")
class PostgresTests(unittest.TestCase):
    def test_concurrent_idempotency_leases_restart_and_fencing(self):
        dsn = os.environ["SPEECH_TEST_DATABASE_URL"]
        schema = "speech_test_" + new_id().replace("-", "")
        with psycopg.connect(dsn) as c:
            c.execute(psycopg.sql.SQL("CREATE SCHEMA {}").format(psycopg.sql.Identifier(schema)))
        class IsolatedStore(Store):
            def connect(self):
                c = super().connect()
                c.execute(psycopg.sql.SQL("SET search_path TO {}").format(psycopg.sql.Identifier(schema)))
                return c
        try:
            store = IsolatedStore(dsn)
            store.migrate()
            p = {k: new_id() for k in ("organization_id", "subject_id", "request_id")}
            req = {"project_id": new_id(), "idempotency_key": "same", "text": "one"}
            with ThreadPoolExecutor(max_workers=6) as pool:
                rows = list(pool.map(lambda _: store.create(p, req, {"outputs": []}), range(6)))
            self.assertEqual(len({r["task_id"] for r in rows}), 1)
            with self.assertRaises(Failure): store.create(p, req | {"text": "two"}, {})
            with self.assertRaises(Failure): store.get(p | {"subject_id": new_id()}, req["project_id"], rows[0]["task_id"])
            with ThreadPoolExecutor(max_workers=6) as pool:
                claimed = [r for r in pool.map(lambda _: store.claim(), range(6)) if r]
            self.assertEqual(len(claimed), 1)
            old = claimed[0]
            old["state"]["gpu_held"] = True
            store.checkpoint(old, phase="loading_model")
            store.renew(old)
            self.assertIsNone(store.claim())
            self.assertTrue(store.gpu_active())
            self.assertEqual(store.get(p, req["project_id"], old["task_id"])["phase"], "loading_model")
            with store.connect() as c:
                c.execute("UPDATE speech_tasks SET lease_until=now()-interval '1 second'")
            restarted = IsolatedStore(dsn)
            new = restarted.claim()
            self.assertEqual(old["task_id"], new["task_id"])
            self.assertNotEqual(old["lease_token"], new["lease_token"])
            with self.assertRaisesRegex(Failure, "lease_lost"): store.save(old, status="failed")
            with self.assertRaisesRegex(Failure, "lease_lost"): store.checkpoint(old, phase="submit")
            self.assertTrue(restarted.gpu_active())
            restarted.save(new, status="succeeded", phase="complete")
            self.assertFalse(restarted.gpu_active())
            self.assertIsNone(store.claim())
            self.assertEqual(store.existing(p, req)["status"], "succeeded")
        finally:
            with psycopg.connect(dsn) as c:
                c.execute(psycopg.sql.SQL("DROP SCHEMA {} CASCADE").format(psycopg.sql.Identifier(schema)))

    def test_cold_start_and_restart_continue_one_runtime_task(self):
        import asyncio
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, patch
        from speech_mcp.gpu_lease import GpuCoordinator, MemoryApi
        from speech_mcp.worker import Worker
        from test_speech import fixture

        dsn = os.environ["SPEECH_TEST_DATABASE_URL"]
        schema = "speech_test_" + new_id().replace("-", "")
        with psycopg.connect(dsn) as c:
            c.execute(psycopg.sql.SQL("CREATE SCHEMA {}").format(psycopg.sql.Identifier(schema)))
        class IsolatedStore(Store):
            def connect(self):
                c = super().connect()
                c.execute(psycopg.sql.SQL("SET search_path TO {}").format(psycopg.sql.Identifier(schema)))
                return c
        try:
            store = IsolatedStore(dsn)
            store.migrate()
            principal, request, original = fixture()
            row = store.create(principal, request, original["state"])
            api, submitted, runtime_id = MemoryApi(), [], new_id()
            async def runtime(p, method, path, payload=None, **kwargs):
                if path == "/health": return {"ready": True}
                if path == "/v1/capabilities": return {"ready": True, "model_manifest_sha256": "a" * 64}
                if method == "PUT": return {}
                if path == "/v1/tasks":
                    submitted.append(payload)
                    return {"task_id": runtime_id}
                self.assertEqual((method, path), ("GET", "/v1/tasks/" + runtime_id))
                return {"task_id": runtime_id, "model_manifest_sha256": "a" * 64, "status": "running"}
            peers = SimpleNamespace(open=AsyncMock(), content=AsyncMock(return_value=(b"audio", {})),
                                    runtime=SimpleNamespace(json=runtime))
            def worker():
                gpu = GpuCoordinator(api, "indextts2", "speech/test", deployment="speech", enabled=True)
                with patch("speech_mcp.worker.threading.Thread.start"):
                    return Worker(SimpleNamespace(store=store, peers=peers, gpu=gpu))
            first = worker()
            asyncio.run(first.process(store.claim()))
            self.assertEqual(store.get(principal, request["project_id"], row["task_id"])["phase"], "submit")
            asyncio.run(first.process(store.claim()))
            with store.connect() as c:
                c.execute("UPDATE speech_tasks SET next_attempt=now() WHERE task_id=%s", (row["task_id"],))
            recovered = worker()
            asyncio.run(recovered.process(store.claim()))
            current = store.get(principal, request["project_id"], row["task_id"])
            self.assertEqual(current["state"]["runtime_task_id"], runtime_id)
            self.assertEqual([p["idempotency_key"] for p in submitted], ["speech-" + row["task_id"]])
            self.assertEqual(recovered.gpu.in_flight, 0)
            self.assertTrue(store.gpu_active())
        finally:
            with psycopg.connect(dsn) as c:
                c.execute(psycopg.sql.SQL("DROP SCHEMA {} CASCADE").format(psycopg.sql.Identifier(schema)))


if __name__ == "__main__": unittest.main()
