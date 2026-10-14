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
            with store.connect() as c:
                c.execute("UPDATE speech_tasks SET lease_until=now()-interval '1 second'")
            restarted = IsolatedStore(dsn)
            new = restarted.claim()
            self.assertEqual(old["task_id"], new["task_id"])
            self.assertNotEqual(old["lease_token"], new["lease_token"])
            with self.assertRaisesRegex(Failure, "lease_lost"): store.save(old, status="failed")
            restarted.save(new, status="succeeded", phase="complete")
            self.assertIsNone(store.claim())
            self.assertEqual(store.existing(p, req)["status"], "succeeded")
        finally:
            with psycopg.connect(dsn) as c:
                c.execute(psycopg.sql.SQL("DROP SCHEMA {} CASCADE").format(psycopg.sql.Identifier(schema)))


if __name__ == "__main__": unittest.main()
