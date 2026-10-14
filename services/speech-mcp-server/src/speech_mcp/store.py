"""PostgreSQL durable intent and fenced leases. External I/O never holds locks."""
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .domain import Failure, canonical, digest, new_id, scope


class Store:
    def __init__(self, dsn):
        self.dsn = dsn

    def connect(self):
        return psycopg.connect(self.dsn, row_factory=dict_row, connect_timeout=5,
                               options="-c statement_timeout=5000 -c lock_timeout=4000")

    def migrate(self):
        with self.connect() as c:
            c.execute("SELECT pg_advisory_xact_lock(73140914)")
            c.execute(Path(__file__).with_name("schema.sql").read_text())

    def healthy(self):
        with self.connect() as c:
            c.execute("SELECT task_id FROM speech_tasks LIMIT 0")

    def existing(self, p, req):
        with self.connect() as c:
            row = c.execute("SELECT * FROM speech_tasks WHERE scope=%s AND idem=%s",
                            (scope(p, req["project_id"]), req["idempotency_key"])).fetchone()
        if row and row["input_hash"] != digest(canonical(req).encode()):
            raise Failure("idempotency_conflict", 409)
        return row

    def create(self, p, req, state):
        with self.connect() as c:
            c.execute("""INSERT INTO speech_tasks(task_id,organization_id,subject_id,project_id,scope,idem,input_hash,request,principal,state)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT(scope,idem) DO NOTHING""",
                      (new_id(), p["organization_id"], p["subject_id"], req["project_id"], scope(p, req["project_id"]),
                       req["idempotency_key"], digest(canonical(req).encode()), Jsonb(req), Jsonb(p), Jsonb(state)))
        return self.existing(p, req)

    def get(self, p, project_id, task_id):
        with self.connect() as c:
            row = c.execute("SELECT * FROM speech_tasks WHERE task_id=%s AND scope=%s",
                            (task_id, scope(p, project_id))).fetchone()
        if not row:
            raise Failure("task_not_found", 404)
        return row

    def claim(self):
        with self.connect() as c:
            return c.execute("""WITH candidate AS (
                SELECT task_id FROM speech_tasks WHERE status IN ('pending','running')
                AND next_attempt<=now() AND (lease_until IS NULL OR lease_until<now())
                ORDER BY next_attempt,created_at FOR UPDATE SKIP LOCKED LIMIT 1)
                UPDATE speech_tasks t SET status='running',lease_token=%s,
                lease_until=now()+interval '120 seconds',updated_at=now()
                FROM candidate WHERE t.task_id=candidate.task_id RETURNING t.*""", (new_id(),)).fetchone()

    def renew(self, row):
        with self.connect() as c:
            n = c.execute("""UPDATE speech_tasks SET lease_until=now()+interval '120 seconds'
                WHERE task_id=%s AND lease_token=%s AND lease_until>now()""",
                          (row["task_id"], row["lease_token"])).rowcount
        if n != 1:
            raise Failure("lease_lost", 409, True)

    def save(self, row, *, phase=None, status="running", error=None, delay=0):
        with self.connect() as c:
            n = c.execute("""UPDATE speech_tasks SET state=%s,phase=%s,status=%s,last_error=%s,
                next_attempt=now()+(%s * interval '1 second'),updated_at=now(),lease_token=NULL,lease_until=NULL
                WHERE task_id=%s AND lease_token=%s AND lease_until>now()""",
                          (Jsonb(row["state"]), phase or row["phase"], status, error, delay,
                           row["task_id"], row["lease_token"])).rowcount
        if n != 1:
            raise Failure("lease_lost", 409, True)
