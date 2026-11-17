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
                ORDER BY CASE
                             WHEN phase IN ('inference','archive','complete') THEN 0
                             WHEN phase = 'submit' OR (state->'gpu_scheduler_lease'->>'state') IN ('leased','running','granted') THEN 1
                             ELSE 2
                         END,
                         next_attempt,created_at FOR UPDATE SKIP LOCKED LIMIT 1)
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

    def checkpoint(self, row, *, phase=None):
        """Persist preparation without handing the running step to a peer."""
        target_phase = phase or row["phase"]
        with self.connect() as c:
            n = c.execute("""UPDATE speech_tasks SET state=%s,phase=%s,updated_at=now()
                WHERE task_id=%s AND lease_token=%s AND lease_until>now()""",
                          (Jsonb(row["state"]), target_phase, row["task_id"], row["lease_token"])).rowcount
        if n != 1:
            raise Failure("lease_lost", 409, True)
        row["phase"] = target_phase

    def gpu_active(self):
        """Worker-only aggregate; never expose another principal's task data."""
        with self.connect() as c:
            return c.execute("""SELECT EXISTS (
                SELECT 1 FROM speech_tasks WHERE state->>'gpu_held'='true'
                AND status!='succeeded' AND phase!='complete'
                AND NOT (status='failed' AND phase IN ('reference','loading_model'))
                ) AS active""").fetchone()["active"]

    def has_active_gpu_tasks(
        self,
        *,
        exclude_task_id: str | None = None,
        include_unarchived_runtime_outputs: bool = True,
    ) -> bool:
        excl = str(exclude_task_id or "").strip()
        phases = (
            ("reference", "loading_model", "submit", "inference", "archive")
            if include_unarchived_runtime_outputs
            else ("reference", "loading_model", "submit", "inference")
        )
        with self.connect() as c:
            if include_unarchived_runtime_outputs:
                row = c.execute(
                    """SELECT EXISTS (
                        SELECT 1 FROM speech_tasks
                        WHERE status IN ('pending','running')
                          AND phase = ANY(%s)
                          AND (%s = '' OR task_id <> %s)
                    ) AS active""",
                    (list(phases), excl, excl),
                ).fetchone()
            else:
                row = c.execute(
                    """SELECT EXISTS (
                        SELECT 1 FROM speech_tasks
                        WHERE status IN ('pending','running')
                          AND (
                              phase IN ('reference','loading_model','submit')
                              OR (phase = 'inference' AND COALESCE(state->>'gpu_execution_finished', 'false') <> 'true')
                          )
                          AND (%s = '' OR task_id <> %s)
                    ) AS active""",
                    (excl, excl),
                ).fetchone()
        return bool(row and row["active"])

    def list_tasks_with_scheduler_leases(self) -> list[dict]:
        with self.connect() as c:
            return c.execute(
                """SELECT task_id, project_id, status, phase, state, principal
                   FROM speech_tasks
                   WHERE jsonb_typeof(state->'gpu_scheduler_lease') = 'object'
                   ORDER BY updated_at ASC"""
            ).fetchall()

    def clear_task_scheduler_lease(self, task_id: str, lease_id: str) -> bool:
        with self.connect() as c:
            n = c.execute(
                """UPDATE speech_tasks
                   SET state = (state - 'gpu_scheduler_lease') - 'gpu_queue',
                       updated_at = now()
                   WHERE task_id = %s
                     AND state->'gpu_scheduler_lease'->>'lease_id' = %s
                     AND (lease_until IS NULL OR lease_until <= now() OR status IN ('succeeded','failed'))""",
                (str(task_id), str(lease_id)),
            ).rowcount
        return n == 1

    def requeue_preinference_stale_fence_tasks(self, *, max_retries: int = 5) -> list[str]:
        requeued: list[str] = []
        with self.connect() as c:
            rows = c.execute(
                """SELECT task_id, state
                   FROM speech_tasks
                   WHERE status = 'failed'
                     AND last_error = 'stale_fencing_token'
                     AND phase IN ('reference', 'loading_model', 'submit', 'inference')
                     AND jsonb_array_length(
                         CASE WHEN jsonb_typeof(state->'outputs') = 'array'
                              THEN state->'outputs'
                              ELSE '[]'::jsonb
                         END
                     ) = 0
                     AND COALESCE((state->>'fence_retry_count')::int, 0) < %s
                     AND (lease_until IS NULL OR lease_until <= now())
                   ORDER BY created_at ASC
                   FOR UPDATE SKIP LOCKED""",
                (int(max_retries),),
            ).fetchall()
            for r in rows:
                tid = str(r["task_id"])
                st = dict(r.get("state") or {})
                retry_cnt = int(st.get("fence_retry_count") or 0) + 1
                st["fence_retry_count"] = retry_cnt
                st.pop("submit_dispatched", None)
                st.pop("runtime_task_id", None)
                st.pop("gpu_execution_finished", None)
                st.pop("gpu_reclaim_proof", None)
                st.pop("gpu_scheduler_lease", None)
                st.pop("gpu_queue", None)
                if isinstance(st.get("runtime_inputs"), dict):
                    rt_in = dict(st["runtime_inputs"])
                    rt_in["idempotency_key"] = f"speech-{tid}-r{retry_cnt}"
                    st["runtime_inputs"] = rt_in
                target_phase = (
                    "reference"
                    if isinstance(st.get("profile"), dict) and isinstance(st["profile"].get("reference"), dict)
                    else "submit"
                )
                c.execute(
                    """UPDATE speech_tasks
                       SET status = 'running',
                           phase = %s,
                           last_error = NULL,
                           state = %s,
                           next_attempt = now(),
                           updated_at = now()
                       WHERE task_id = %s""",
                    (target_phase, Jsonb(st), tid),
                )
                requeued.append(tid)
        return requeued

    def rename_task(self, task_id: str, title: str, p=None):
        if not isinstance(title, str):
            raise Failure("invalid_arguments", 400)
        clean_title = title.strip()
        if not clean_title or len(clean_title) > 120:
            raise Failure("invalid_arguments", 400)
        with self.connect() as c:
            if p and p.get("organization_id") and p.get("subject_id"):
                row = c.execute(
                    "SELECT * FROM speech_tasks WHERE task_id=%s AND organization_id=%s AND subject_id=%s FOR UPDATE",
                    (task_id, p["organization_id"], p["subject_id"]),
                ).fetchone()
            else:
                row = c.execute(
                    "SELECT * FROM speech_tasks WHERE task_id=%s FOR UPDATE",
                    (task_id,),
                ).fetchone()
            if not row:
                raise TaskNotFound("task_not_found", 404)
            state = dict(row.get("state") or {})
            state["title"] = clean_title
            updated = c.execute(
                "UPDATE speech_tasks SET state=%s, updated_at=now() WHERE task_id=%s RETURNING *",
                (Jsonb(state), task_id),
            ).fetchone()
        if updated is not None:
            updated["title"] = clean_title
        return updated

    def delete_task(self, task_id: str, p=None):
        with self.connect() as c:
            if p and p.get("organization_id") and p.get("subject_id"):
                deleted = c.execute(
                    "DELETE FROM speech_tasks WHERE task_id=%s AND organization_id=%s AND subject_id=%s RETURNING task_id",
                    (task_id, p["organization_id"], p["subject_id"]),
                ).fetchone()
            else:
                deleted = c.execute(
                    "DELETE FROM speech_tasks WHERE task_id=%s RETURNING task_id",
                    (task_id,),
                ).fetchone()
        if not deleted:
            raise TaskNotFound("task_not_found", 404)
        return {"status": "deleted", "task_id": task_id}

    def upsert_pending_gpu_release(
        self,
        lease_id: str,
        fencing_token: int,
        payload: dict,
        *,
        owner_pod_uid: str = "",
        claim_seconds: int = 30,
    ) -> tuple[str, int]:
        clean_id = str(lease_id or "").strip()
        if not clean_id or not isinstance(payload, dict):
            raise Failure("invalid_pending_gpu_release", 400)
        token = new_id()
        owner = str(owner_pod_uid or "")
        ttl = max(1, int(claim_seconds))
        with self.connect() as c:
            row = c.execute(
                """INSERT INTO speech_gpu_pending_releases(
                    lease_id, fencing_token, payload, payload_version, status, owner_pod_uid,
                    claim_token, claim_until, last_error, created_at, updated_at
                ) VALUES (
                    %s, %s, %s, 1, 'pending', %s,
                    %s, now() + (%s * interval '1 second'), NULL, now(), now()
                )
                ON CONFLICT (lease_id) DO UPDATE SET
                    fencing_token = EXCLUDED.fencing_token,
                    payload = EXCLUDED.payload,
                    payload_version = speech_gpu_pending_releases.payload_version + 1,
                    status = 'pending',
                    owner_pod_uid = EXCLUDED.owner_pod_uid,
                    claim_token = EXCLUDED.claim_token,
                    claim_until = EXCLUDED.claim_until,
                    last_error = NULL,
                    updated_at = now()
                WHERE speech_gpu_pending_releases.fencing_token <= EXCLUDED.fencing_token
                  AND (
                      speech_gpu_pending_releases.claim_until IS NULL
                      OR speech_gpu_pending_releases.claim_until <= now()
                      OR (%s <> '' AND speech_gpu_pending_releases.owner_pod_uid = %s)
                  )
                RETURNING claim_token, payload_version""",
                (clean_id, int(fencing_token), Jsonb(payload), owner, token, ttl, owner, owner),
            ).fetchone()
        if not row:
            raise Failure("pending_gpu_release_claimed_or_stale", 409, True)
        return str(row["claim_token"]), int(row["payload_version"])

    def claim_next_pending_gpu_release(
        self,
        *,
        owner_pod_uid: str = "",
        include_quarantined: bool = False,
        claim_seconds: int = 30,
        exclude_lease_ids: tuple[str, ...] | list[str] = (),
    ) -> dict | None:
        statuses = ("pending", "quarantined_400") if include_quarantined else ("pending",)
        excluded = [str(x) for x in exclude_lease_ids if str(x)]
        token = new_id()
        with self.connect() as c:
            return c.execute(
                """WITH candidate AS (
                    SELECT lease_id FROM speech_gpu_pending_releases
                    WHERE status = ANY(%s)
                      AND (claim_until IS NULL OR claim_until <= now())
                      AND NOT (lease_id = ANY(%s))
                    ORDER BY updated_at ASC
                    FOR UPDATE SKIP LOCKED
                    LIMIT 1
                )
                UPDATE speech_gpu_pending_releases r
                SET owner_pod_uid = %s,
                    claim_token = %s,
                    claim_until = now() + (%s * interval '1 second'),
                    updated_at = now()
                FROM candidate
                WHERE r.lease_id = candidate.lease_id
                RETURNING r.*""",
                (list(statuses), excluded, str(owner_pod_uid or ""), token, max(1, int(claim_seconds))),
            ).fetchone()

    def quarantine_pending_gpu_release(
        self,
        lease_id: str,
        last_error: str,
        *,
        claim_token: str,
        payload_version: int,
    ) -> bool:
        if not claim_token or int(payload_version) <= 0:
            return False
        with self.connect() as c:
            n = c.execute(
                """UPDATE speech_gpu_pending_releases
                   SET status = 'quarantined_400',
                       last_error = %s,
                       claim_token = NULL,
                       claim_until = NULL,
                       updated_at = now()
                   WHERE lease_id = %s
                     AND claim_token = %s
                     AND payload_version = %s""",
                (str(last_error or "")[:2000], str(lease_id), str(claim_token), int(payload_version)),
            ).rowcount
        return n == 1

    def release_pending_gpu_claim(
        self,
        lease_id: str,
        *,
        claim_token: str,
        payload_version: int,
    ) -> bool:
        if not claim_token or int(payload_version) <= 0:
            return False
        with self.connect() as c:
            n = c.execute(
                """UPDATE speech_gpu_pending_releases
                   SET claim_token = NULL,
                       claim_until = NULL,
                       updated_at = now()
                   WHERE lease_id = %s
                     AND claim_token = %s
                     AND payload_version = %s""",
                (str(lease_id), str(claim_token), int(payload_version)),
            ).rowcount
        return n == 1

    def delete_pending_gpu_release(
        self,
        lease_id: str,
        *,
        claim_token: str,
        payload_version: int,
    ) -> bool:
        if not claim_token or int(payload_version) <= 0:
            return False
        with self.connect() as c:
            n = c.execute(
                """DELETE FROM speech_gpu_pending_releases
                   WHERE lease_id = %s
                     AND claim_token = %s
                     AND payload_version = %s""",
                (str(lease_id), str(claim_token), int(payload_version)),
            ).rowcount
        return n == 1

    def get_pending_gpu_release(self, lease_id: str) -> dict | None:
        with self.connect() as c:
            return c.execute(
                "SELECT * FROM speech_gpu_pending_releases WHERE lease_id = %s",
                (str(lease_id),),
            ).fetchone()

    def list_pending_gpu_releases(self) -> list[dict]:
        with self.connect() as c:
            return c.execute(
                "SELECT lease_id, fencing_token, status, payload FROM speech_gpu_pending_releases ORDER BY updated_at ASC"
            ).fetchall()


TaskStore = Store
