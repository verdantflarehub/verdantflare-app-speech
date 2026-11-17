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
    def test_result_recovery_precedes_new_gpu_admission(self):
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
            principal = {k: new_id() for k in ("organization_id", "subject_id", "request_id")}
            request = {"project_id": new_id(), "text": "test"}
            queued = store.create(principal, request | {"idempotency_key": "queued"}, {})
            result = store.create(principal, request | {"idempotency_key": "result"}, {})
            with store.connect() as c:
                c.execute("UPDATE speech_tasks SET phase='inference' WHERE task_id=%s", (result["task_id"],))
                c.execute("UPDATE speech_tasks SET phase='reference',next_attempt=now()-interval '1 hour' WHERE task_id=%s", (queued["task_id"],))
            first = store.claim()
            self.assertEqual(first["task_id"], result["task_id"])
            # Priority cannot steal an active lease; the other worker gets queued work.
            second = store.claim()
            self.assertEqual(second["task_id"], queued["task_id"])
            store.save(first, status="running", phase="inference", delay=60)
            self.assertIsNone(store.claim())  # Future retry is still respected.
        finally:
            with psycopg.connect(dsn) as c:
                c.execute(psycopg.sql.SQL("DROP SCHEMA {} CASCADE").format(psycopg.sql.Identifier(schema)))

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

    def test_postgres_pending_gpu_release_pod_replacement_and_400_quarantine(self):
        import asyncio
        import json
        import shutil
        import tempfile
        import time
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, patch

        import httpx
        from speech_mcp.gpu_lease import GpuCoordinator, MemoryApi, SchedulerError, SchedulerLeaseClient
        from speech_mcp.worker import Worker

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
            mode = {"phase": "crash_mid_flight"}
            release_calls: list[tuple[str, dict]] = []

            def sched_handler(request: httpx.Request) -> httpx.Response:
                body = json.loads(request.content.decode("utf-8")) if request.content else {}
                if request.url.path.endswith("/release"):
                    lid = request.url.path.split("/")[-2]
                    release_calls.append((lid, body))
                    if mode["phase"] == "bad_400":
                        return httpx.Response(
                            400,
                            json={"error_code": "INVALID_RECLAIM_PROOF", "message": "proof malformed"},
                        )
                    return httpx.Response(200, json={"lease_id": lid, "state": "released"})
                return httpx.Response(200, json={})

            http_client = httpx.Client(transport=httpx.MockTransport(sched_handler))
            self.addCleanup(http_client.close)

            # 1) Simulate Pod A ("pod-uid-old") writing durable PG pending release right before crashing mid-HTTP
            pod_a_tmp = tempfile.mkdtemp(prefix="pod-a-")
            store.upsert_pending_gpu_release(
                "lease-orphan-pod-a",
                701,
                {
                    "idempotency_key": "rel:lease-orphan-pod-a:701:abort_unstarted",
                    "fencing_token": 701,
                    "worker_settled": False,
                    "reclaim_mode": "abort_unstarted",
                },
                owner_pod_uid="pod-uid-old",
                claim_seconds=1,
            )
            # Destroy Pod A's ephemeral /tmp completely (Recreate replacement)
            shutil.rmtree(pod_a_tmp, ignore_errors=True)

            # While Pod A's 1s claim_until is still active, Pod B does not steal an in-flight claim
            sched_pod_b = SchedulerLeaseClient(
                base_url="http://station-runtime:5056",
                token="sched-token",
                enabled=True,
                client=http_client,
                store=store,
                pending_releases_backend="postgres",
                owner_pod_uid="pod-uid-new",
                claim_seconds=5,
            )
            self.assertEqual(sched_pod_b.flush_pending_releases(), 0)

            # Wait for Pod A's claim_until (1s) to expire; Pod B starts Worker.run() with ZERO speech tasks
            time.sleep(1.2)
            gpu = GpuCoordinator(MemoryApi(), "indextts2", "speech/pod-b", deployment="speech", enabled=False)
            peers = SimpleNamespace(open=AsyncMock(), runtime=SimpleNamespace(json=AsyncMock()))
            with patch("speech_mcp.worker.threading.Thread.start"):
                worker_b = Worker(SimpleNamespace(store=store, peers=peers, gpu=gpu, scheduler=sched_pod_b))

            async def run_briefly():
                t = asyncio.create_task(worker_b.run())
                await asyncio.sleep(0.15)
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass

            mode["phase"] = "ok"
            asyncio.run(run_briefly())
            self.assertEqual(len(release_calls), 1)
            self.assertEqual(release_calls[0][0], "lease-orphan-pod-a")
            self.assertEqual(release_calls[0][1]["fencing_token"], 701)
            self.assertIsNone(store.get_pending_gpu_release("lease-orphan-pod-a"))

            # 2) Verify HTTP 400 in Postgres mode quarantines row (never deletes) and recovers on corrected release()
            release_calls.clear()
            mode["phase"] = "bad_400"
            with self.assertRaises(SchedulerError) as e_400:
                sched_pod_b.release("lease-pg-400", 702, expected_version=4, worker_settled=False, reclaim_mode="abort_unstarted")
            self.assertEqual(e_400.exception.code, "INVALID_RECLAIM_PROOF")
            row_q = store.get_pending_gpu_release("lease-pg-400")
            self.assertIsNotNone(row_q)
            self.assertEqual(row_q["status"], "quarantined_400")
            self.assertIn("INVALID_RECLAIM_PROOF", row_q["last_error"])

            # Normal flush skips quarantined_400
            release_calls.clear()
            self.assertEqual(sched_pod_b.flush_pending_releases(), 0)
            self.assertEqual(len(release_calls), 0)

            # Corrected release() resets row and clears it once 200 confirmed
            mode["phase"] = "ok"
            res_ok = sched_pod_b.release("lease-pg-400", 702, worker_settled=False, reclaim_mode="abort_unstarted")
            self.assertEqual(res_ok["state"], "released")
            self.assertIsNone(store.get_pending_gpu_release("lease-pg-400"))

            # 3) R4-M1: Late 400/delete from stale (tok_a, ver_a) CANNOT quarantine or delete newer (tok_b, ver_b)
            #    even after Pod B released its claim back to NULL on a transient 503.
            tok_a, ver_a = store.upsert_pending_gpu_release(
                "lease-r4-fence",
                710,
                {"idempotency_key": "rel:lease-r4-fence:710:bad", "fencing_token": 710, "worker_settled": False},
                owner_pod_uid="pod-uid-old",
                claim_seconds=1,
            )
            self.assertEqual(ver_a, 1)
            # While Pod A's claim is still active, a foreign Pod B cannot overwrite it prematurely
            with self.assertRaisesRegex(Failure, "pending_gpu_release_claimed_or_stale"):
                store.upsert_pending_gpu_release(
                    "lease-r4-fence",
                    710,
                    {"idempotency_key": "rel:lease-r4-fence:710:good", "fencing_token": 710, "worker_settled": True},
                    owner_pod_uid="pod-uid-new",
                    claim_seconds=5,
                )
            time.sleep(1.15)
            # After Pod A's claim expired, Pod B updates the row with corrected payload (payload_version -> 2)
            tok_b, ver_b = store.upsert_pending_gpu_release(
                "lease-r4-fence",
                710,
                {"idempotency_key": "rel:lease-r4-fence:710:good", "fencing_token": 710, "worker_settled": True},
                owner_pod_uid="pod-uid-new",
                claim_seconds=5,
            )
            self.assertEqual(ver_b, 2)
            # Pod B hits transient network failure and releases claim -> claim_token IS NULL in DB
            self.assertTrue(store.release_pending_gpu_claim("lease-r4-fence", claim_token=tok_b, payload_version=ver_b))
            row_unclaimed = store.get_pending_gpu_release("lease-r4-fence")
            self.assertIsNotNone(row_unclaimed)
            self.assertIsNone(row_unclaimed["claim_token"])
            self.assertEqual(row_unclaimed["status"], "pending")
            # Now Pod A's late HTTP 400 or late HTTP 200 arrives with stale (tok_a, ver_a=1): MUST be fenced out!
            self.assertFalse(
                store.quarantine_pending_gpu_release(
                    "lease-r4-fence",
                    "INVALID_RECLAIM_PROOF:late_pod_a",
                    claim_token=tok_a,
                    payload_version=ver_a,
                )
            )
            self.assertFalse(
                store.delete_pending_gpu_release(
                    "lease-r4-fence",
                    claim_token=tok_a,
                    payload_version=ver_a,
                )
            )
            row_still_good = store.get_pending_gpu_release("lease-r4-fence")
            self.assertEqual(row_still_good["status"], "pending")
            self.assertEqual(row_still_good["payload_version"], 2)
            self.assertTrue(row_still_good["payload"]["worker_settled"])
            # Subsequent flush sends Pod B's corrected payload and deletes the row cleanly
            release_calls.clear()
            self.assertEqual(sched_pod_b.flush_pending_releases(), 1)
            self.assertIsNone(store.get_pending_gpu_release("lease-r4-fence"))
            self.assertEqual(release_calls[0][1]["worker_settled"], True)

            # 4) R4-M1: 1-by-1 claiming right before each HTTP call prevents local batch queue claim expiration
            for idx in range(3):
                lid = f"lease-r4-batch-{idx}"
                t_c, v_c = store.upsert_pending_gpu_release(
                    lid,
                    720 + idx,
                    {
                        "idempotency_key": f"rel:{lid}:{720 + idx}:abort_unstarted",
                        "fencing_token": 720 + idx,
                        "worker_settled": False,
                        "reclaim_mode": "abort_unstarted",
                    },
                    owner_pod_uid="pod-uid-batch",
                    claim_seconds=5,
                )
                store.release_pending_gpu_claim(lid, claim_token=t_c, payload_version=v_c)

            def slow_sched_handler(request: httpx.Request) -> httpx.Response:
                if request.url.path.endswith("/release"):
                    time.sleep(0.45)
                    lid = request.url.path.split("/")[-2]
                    return httpx.Response(200, json={"lease_id": lid, "state": "released"})
                return httpx.Response(200, json={})

            slow_client = httpx.Client(transport=httpx.MockTransport(slow_sched_handler))
            self.addCleanup(slow_client.close)
            sched_short_claim = SchedulerLeaseClient(
                base_url="http://station-runtime:5056",
                token="sched-token",
                enabled=True,
                client=slow_client,
                store=store,
                pending_releases_backend="postgres",
                owner_pod_uid="pod-uid-short",
                claim_seconds=1,
            )
            # Total wall time ~1.35s (> claim_seconds=1s), yet each row is claimed fresh immediately before its 0.45s HTTP call
            self.assertEqual(sched_short_claim.flush_pending_releases(), 3)
            for idx in range(3):
                self.assertIsNone(store.get_pending_gpu_release(f"lease-r4-batch-{idx}"))
        finally:
            with psycopg.connect(dsn) as c:
                c.execute(psycopg.sql.SQL("DROP SCHEMA {} CASCADE").format(psycopg.sql.Identifier(schema)))


if __name__ == "__main__": unittest.main()
