"""PostgreSQL durable intent and fenced leases. External I/O never holds locks."""
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .domain import Failure, TaskNotFound, canonical, digest, new_id, scope


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
            raise TaskNotFound("task_not_found", 404)
        return row

    def get_by_id(self, p, task_id):
        with self.connect() as c:
            row = c.execute(
                "SELECT * FROM speech_tasks WHERE task_id=%s AND organization_id=%s AND subject_id=%s",
                (task_id, p["organization_id"], p["subject_id"]),
            ).fetchone()
        if not row:
            raise TaskNotFound("task_not_found", 404)
        return row

    def find_by_version(self, version_id, project_id=None):
        with self.connect() as c:
            if project_id:
                rows = c.execute(
                    "SELECT * FROM speech_tasks WHERE project_id=%s ORDER BY created_at DESC LIMIT 500",
                    (project_id,),
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT * FROM speech_tasks ORDER BY created_at DESC LIMIT 500"
                ).fetchall()
        for r in rows:
            if r["task_id"] == version_id or r.get("artifact_id") == version_id:
                return r
            for out in (r.get("state") or {}).get("outputs") or []:
                cref = out.get("content_ref") if isinstance(out, dict) else None
                if isinstance(cref, dict) and version_id in (cref.get("version_id"), cref.get("artifact_id")):
                    return r
        return None

    def list_for_principal(self, p, project_id=None):
        with self.connect() as c:
            if project_id:
                return c.execute(
                    "SELECT * FROM speech_tasks WHERE scope=%s ORDER BY created_at DESC LIMIT 500",
                    (scope(p, project_id),),
                ).fetchall()
            return c.execute(
                "SELECT * FROM speech_tasks WHERE organization_id=%s AND subject_id=%s ORDER BY created_at DESC LIMIT 500",
                (p["organization_id"], p["subject_id"]),
            ).fetchall()

    def utterance_takes(self, p, project_id, utterance_id):
        with self.connect() as c:
            return c.execute(
                "SELECT * FROM speech_tasks WHERE scope=%s AND request->>'utterance_id'=%s ORDER BY created_at ASC",
                (scope(p, project_id), utterance_id),
            ).fetchall()

    def update_review(self, p, project_id, task_id, decision, note=None, reviewed_revision_id=None):
        sc = scope(p, project_id)
        with self.connect() as c:
            row = c.execute(
                "SELECT * FROM speech_tasks WHERE task_id=%s AND scope=%s FOR UPDATE",
                (task_id, sc),
            ).fetchone()
            if not row:
                raise Failure("task_not_found", 404)
            if row["status"] != "succeeded":
                raise Failure("result_not_ready", 409)
            utterance_id = row["request"].get("utterance_id")
            if decision == "selected" and utterance_id:
                siblings = c.execute(
                    "SELECT * FROM speech_tasks WHERE scope=%s AND request->>'utterance_id'=%s AND task_id<>%s FOR UPDATE",
                    (sc, utterance_id, task_id),
                ).fetchall()
                for sib in siblings:
                    sib_state = dict(sib["state"])
                    if sib_state.get("candidate_state") == "selected" or sib_state.get("review_state") == "selected":
                        sib_state["candidate_state"] = "candidate"
                        if sib_state.get("review_state") == "selected":
                            sib_state["review_state"] = "pending"
                        if isinstance(sib_state.get("generation"), dict):
                            sib_state["generation"]["candidate_state"] = "candidate"
                        c.execute(
                            "UPDATE speech_tasks SET state=%s, updated_at=now() WHERE task_id=%s AND scope=%s",
                            (Jsonb(sib_state), sib["task_id"], sc),
                        )
            state = dict(row["state"])
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
            if isinstance(state.get("generation"), dict):
                state["generation"]["candidate_state"] = state["candidate_state"]
                state["generation"]["listening_review"] = state["listening_review"]
            updated = c.execute(
                "UPDATE speech_tasks SET state=%s, updated_at=now() WHERE task_id=%s AND scope=%s RETURNING *",
                (Jsonb(state), task_id, sc),
            ).fetchone()
        return updated

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
