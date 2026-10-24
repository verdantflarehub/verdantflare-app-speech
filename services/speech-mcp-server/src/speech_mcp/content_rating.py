"""Domain-local ContentRating storage, state machine, and audit log (content-rating-v1)."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Mapping
import uuid


class ContentRatingError(Exception):
    def __init__(self, status_code: int, error_code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.message = message


CONTENT_RATING_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS content_ratings (
    version_id TEXT PRIMARY KEY,
    task_id TEXT,
    project_id TEXT NOT NULL,
    content_ref_json TEXT NOT NULL,
    rating TEXT NOT NULL,
    source TEXT NOT NULL,
    rubric_version TEXT,
    evaluator_model TEXT,
    reasons_json TEXT NOT NULL,
    rated_by TEXT NOT NULL,
    rated_at TEXT NOT NULL,
    human_cleared INTEGER NOT NULL DEFAULT 0,
    cleared_by TEXT,
    cleared_at TEXT,
    clear_note TEXT
);
CREATE INDEX IF NOT EXISTS idx_content_ratings_task ON content_ratings(task_id);
CREATE INDEX IF NOT EXISTS idx_content_ratings_project ON content_ratings(project_id);
CREATE INDEX IF NOT EXISTS idx_content_ratings_rating ON content_ratings(rating);

CREATE TABLE IF NOT EXISTS content_rating_events (
    event_id TEXT PRIMARY KEY,
    version_id TEXT NOT NULL,
    task_id TEXT,
    project_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    rating TEXT NOT NULL,
    source TEXT NOT NULL,
    rubric_version TEXT,
    evaluator_model TEXT,
    reasons_json TEXT NOT NULL,
    actor_id TEXT NOT NULL,
    note TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_content_rating_events_version ON content_rating_events(version_id);
"""


def default_unrated_dict() -> dict[str, Any]:
    return {
        "rating": "unrated",
        "source": None,
        "rubric_version": None,
        "evaluator_model": None,
        "reasons": [],
        "reason": None,
        "rated_by": None,
        "rated_at": None,
        "human_cleared": False,
        "cleared_by": None,
        "cleared_at": None,
        "clear_note": None,
    }


def include_nsfw_from_headers(headers: Mapping[str, str] | None) -> bool:
    if not headers:
        return False
    for k, v in headers.items():
        if k.lower() == "x-vf-include-nsfw":
            return str(v).strip() == "1"
    return False


def normalize_reasons(reasons: list[str] | str | None, reason: str | None = None) -> list[str]:
    out: list[str] = []
    if isinstance(reasons, str) and reasons.strip():
        out.append(reasons.strip())
    elif isinstance(reasons, list):
        for item in reasons:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
    if isinstance(reason, str) and reason.strip() and reason.strip() not in out:
        out.append(reason.strip())
    if not out:
        raise ContentRatingError(400, "INVALID_ARGUMENT", "reason or reasons must be a non-empty string/list")
    for item in out:
        if len(item) > 500:
            raise ContentRatingError(400, "INVALID_ARGUMENT", "Each rating reason must not exceed 500 characters")
    return out


class _ClosingConnection(sqlite3.Connection):
    def __exit__(self, *args: Any) -> Any:
        try:
            return super().__exit__(*args)
        finally:
            self.close()


class ContentRatingStore:
    def __init__(self, db_path: Path, domain: str = "speech") -> None:
        self.db_path = db_path
        self.domain = domain

    def _get_conn(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            check_same_thread=False,
            factory=_ClosingConnection,
        )
        conn.row_factory = sqlite3.Row
        return conn

    def ensure_ready(self) -> None:
        with self._get_conn() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.executescript(CONTENT_RATING_SCHEMA_SQL)

    def _row_to_dict(self, row: sqlite3.Row | None) -> dict[str, Any]:
        if row is None:
            return default_unrated_dict()
        try:
            reasons = json.loads(row["reasons_json"] or "[]")
        except Exception:
            reasons = []
        try:
            content_ref = json.loads(row["content_ref_json"] or "{}")
        except Exception:
            content_ref = {}
        return {
            "version_id": row["version_id"],
            "task_id": row["task_id"],
            "project_id": row["project_id"],
            "content_ref": content_ref,
            "rating": row["rating"],
            "source": row["source"],
            "rubric_version": row["rubric_version"],
            "evaluator_model": row["evaluator_model"],
            "reasons": reasons,
            "reason": reasons[0] if reasons else None,
            "rated_by": row["rated_by"],
            "rated_at": row["rated_at"],
            "human_cleared": bool(row["human_cleared"]),
            "cleared_by": row["cleared_by"],
            "cleared_at": row["cleared_at"],
            "clear_note": row["clear_note"],
        }

    def get(self, version_id: str | None = None, *, task_id: str | None = None) -> dict[str, Any]:
        self.ensure_ready()
        with self._get_conn() as conn:
            row = None
            if version_id:
                row = conn.execute(
                    "SELECT * FROM content_ratings WHERE version_id = ?", (version_id,)
                ).fetchone()
            if row is None and task_id:
                row = conn.execute(
                    "SELECT * FROM content_ratings WHERE task_id = ? ORDER BY rated_at DESC LIMIT 1",
                    (task_id,),
                ).fetchone()
            return self._row_to_dict(row)

    def is_nsfw(self, version_id: str | None = None, *, task_id: str | None = None) -> bool:
        rec = self.get(version_id, task_id=task_id)
        return rec.get("rating") == "nsfw"

    def ui_fields(
        self,
        version_id: str | None = None,
        *,
        task_id: str | None = None,
        can_write: bool = True,
    ) -> dict[str, Any]:
        rec = self.get(version_id, task_id=task_id)
        rating = rec.get("rating", "unrated")
        return {
            "content_rating": rating,
            "content_rating_detail": {
                "rating": rating,
                "source": rec.get("source"),
                "rubric_version": rec.get("rubric_version"),
                "evaluator_model": rec.get("evaluator_model"),
                "reasons": rec.get("reasons", []),
                "reason": rec.get("reason"),
                "rated_by": rec.get("rated_by"),
                "rated_at": rec.get("rated_at"),
                "human_cleared": bool(rec.get("human_cleared", False)),
                "cleared_by": rec.get("cleared_by"),
                "cleared_at": rec.get("cleared_at"),
                "clear_note": rec.get("clear_note"),
            },
            "can_clear_rating": bool(can_write and rating == "nsfw"),
            "can_mark_rating": bool(can_write and rating != "nsfw"),
        }

    def rate_content(
        self,
        *,
        version_id: str,
        project_id: str,
        rating: str,
        reasons: list[str] | str | None = None,
        reason: str | None = None,
        rubric_version: str | None = "rubric v1.0",
        evaluator_model: str | None = None,
        rated_by: str = "skill:k09",
        task_id: str | None = None,
        content_ref: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.ensure_ready()
        clean_rating = str(rating or "").strip().lower()
        if clean_rating in ("unrated", "clear", "none"):
            raise ContentRatingError(
                403,
                "RATING_CLEAR_FORBIDDEN",
                "MCP rate_content is add-only and cannot clear or downgrade ratings to unrated",
            )
        if clean_rating not in ("general", "nsfw"):
            raise ContentRatingError(
                400,
                "INVALID_ARGUMENT",
                f"rating must be 'general' or 'nsfw', got '{rating}'",
            )
        if not version_id or not str(version_id).strip():
            raise ContentRatingError(400, "INVALID_ARGUMENT", "version_id is required")
        version_id = str(version_id).strip()

        clean_rubric = str(rubric_version if rubric_version is not None else "rubric v1.0").strip()
        if not clean_rubric or len(clean_rubric) > 32:
            raise ContentRatingError(
                400,
                "INVALID_ARGUMENT",
                "rubric_version must be a non-empty string up to 32 characters",
            )
        clean_reasons = normalize_reasons(reasons, reason)
        now = datetime.now(timezone.utc).isoformat()
        resolved_ref = content_ref or {
            "domain": self.domain,
            "project_id": project_id,
            "version_id": version_id,
            "task_id": task_id,
        }

        with self._get_conn() as conn:
            existing_row = conn.execute(
                "SELECT * FROM content_ratings WHERE version_id = ?", (version_id,)
            ).fetchone()
            existing = self._row_to_dict(existing_row)
            previous_rating = existing.get("rating", "unrated")

            # Rule 1: If human_cleared is True, AI submissions of nsfw are ignored!
            if existing_row is not None and bool(existing_row["human_cleared"]):
                event_type = "ignored_after_clear" if clean_rating == "nsfw" else "rate"
                conn.execute(
                    """
                    INSERT INTO content_rating_events (
                        event_id, version_id, task_id, project_id, event_type,
                        rating, source, rubric_version, evaluator_model,
                        reasons_json, actor_id, note, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 'ai', ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"crev-{uuid.uuid4().hex[:16]}",
                        version_id,
                        task_id or existing_row["task_id"],
                        project_id,
                        event_type,
                        clean_rating,
                        clean_rubric,
                        evaluator_model,
                        json.dumps(clean_reasons, ensure_ascii=False),
                        rated_by,
                        "Ignored AI rating after human clear",
                        now,
                    ),
                )
                conn.commit()
                return {
                    "status": "ok",
                    "version_id": version_id,
                    "task_id": task_id or existing_row["task_id"],
                    "project_id": project_id,
                    "rating": existing_row["rating"],
                    "previous_rating": previous_rating,
                    "source": existing_row["source"],
                    "rubric_version": existing_row["rubric_version"],
                    "evaluator_model": existing_row["evaluator_model"],
                    "reasons": existing["reasons"],
                    "reason": existing["reason"],
                    "rated_at": existing_row["rated_at"],
                    "rated_by": existing_row["rated_by"],
                    "human_cleared": True,
                    "ignored": True,
                }

            # Rule 2: nsfw priority — if already nsfw, AI submitting general does not downgrade.
            if existing_row is not None and existing_row["rating"] == "nsfw" and clean_rating == "general":
                conn.execute(
                    """
                    INSERT INTO content_rating_events (
                        event_id, version_id, task_id, project_id, event_type,
                        rating, source, rubric_version, evaluator_model,
                        reasons_json, actor_id, note, created_at
                    ) VALUES (?, ?, ?, ?, 'rate', 'general', 'ai', ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"crev-{uuid.uuid4().hex[:16]}",
                        version_id,
                        task_id or existing_row["task_id"],
                        project_id,
                        clean_rubric,
                        evaluator_model,
                        json.dumps(clean_reasons, ensure_ascii=False),
                        rated_by,
                        "Preserved existing nsfw rating over general submission",
                        now,
                    ),
                )
                conn.commit()
                return {
                    "status": "ok",
                    "version_id": version_id,
                    "task_id": task_id or existing_row["task_id"],
                    "project_id": project_id,
                    "rating": "nsfw",
                    "previous_rating": "nsfw",
                    "source": existing_row["source"],
                    "rubric_version": existing_row["rubric_version"],
                    "evaluator_model": existing_row["evaluator_model"],
                    "reasons": existing["reasons"],
                    "reason": existing["reason"],
                    "rated_at": existing_row["rated_at"],
                    "rated_by": existing_row["rated_by"],
                    "human_cleared": False,
                    "ignored": True,
                }

            conn.execute(
                """
                INSERT INTO content_ratings (
                    version_id, task_id, project_id, content_ref_json,
                    rating, source, rubric_version, evaluator_model,
                    reasons_json, rated_by, rated_at, human_cleared,
                    cleared_by, cleared_at, clear_note
                ) VALUES (?, ?, ?, ?, ?, 'ai', ?, ?, ?, ?, ?, 0, NULL, NULL, NULL)
                ON CONFLICT(version_id) DO UPDATE SET
                    task_id = COALESCE(excluded.task_id, content_ratings.task_id),
                    rating = excluded.rating,
                    source = 'ai',
                    rubric_version = excluded.rubric_version,
                    evaluator_model = excluded.evaluator_model,
                    reasons_json = excluded.reasons_json,
                    rated_by = excluded.rated_by,
                    rated_at = excluded.rated_at
                """,
                (
                    version_id,
                    task_id,
                    project_id,
                    json.dumps(resolved_ref, ensure_ascii=False),
                    clean_rating,
                    clean_rubric,
                    evaluator_model,
                    json.dumps(clean_reasons, ensure_ascii=False),
                    rated_by,
                    now,
                ),
            )
            conn.execute(
                """
                INSERT INTO content_rating_events (
                    event_id, version_id, task_id, project_id, event_type,
                    rating, source, rubric_version, evaluator_model,
                    reasons_json, actor_id, note, created_at
                ) VALUES (?, ?, ?, ?, 'rate', ?, 'ai', ?, ?, ?, ?, NULL, ?)
                """,
                (
                    f"crev-{uuid.uuid4().hex[:16]}",
                    version_id,
                    task_id,
                    project_id,
                    clean_rating,
                    clean_rubric,
                    evaluator_model,
                    json.dumps(clean_reasons, ensure_ascii=False),
                    rated_by,
                    now,
                ),
            )
            conn.commit()

        return {
            "status": "ok",
            "version_id": version_id,
            "task_id": task_id,
            "project_id": project_id,
            "rating": clean_rating,
            "previous_rating": previous_rating,
            "source": "ai",
            "rubric_version": clean_rubric,
            "evaluator_model": evaluator_model,
            "reasons": clean_reasons,
            "reason": clean_reasons[0],
            "rated_at": now,
            "rated_by": rated_by,
            "human_cleared": False,
            "ignored": False,
        }

    def clear_rating(
        self,
        *,
        version_id: str,
        project_id: str,
        actor_user_id: str,
        note: str | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        self.ensure_ready()
        if not actor_user_id or not str(actor_user_id).strip():
            raise ContentRatingError(401, "UNAUTHORIZED", "Authenticated human user is required to clear NSFW rating")
        actor = str(actor_user_id).strip()
        clean_note = str(note).strip() if note else "Cleared by human in Studio"
        if len(clean_note) > 500:
            raise ContentRatingError(400, "INVALID_ARGUMENT", "note must not exceed 500 characters")
        now = datetime.now(timezone.utc).isoformat()
        resolved_ref = {
            "domain": self.domain,
            "project_id": project_id,
            "version_id": version_id,
            "task_id": task_id,
        }

        with self._get_conn() as conn:
            existing_row = conn.execute(
                "SELECT * FROM content_ratings WHERE version_id = ?", (version_id,)
            ).fetchone()
            previous_rating = existing_row["rating"] if existing_row else "unrated"
            resolved_task = task_id or (existing_row["task_id"] if existing_row else None)

            conn.execute(
                """
                INSERT INTO content_ratings (
                    version_id, task_id, project_id, content_ref_json,
                    rating, source, rubric_version, evaluator_model,
                    reasons_json, rated_by, rated_at, human_cleared,
                    cleared_by, cleared_at, clear_note
                ) VALUES (?, ?, ?, ?, 'general', 'human', NULL, NULL, ?, ?, ?, 1, ?, ?, ?)
                ON CONFLICT(version_id) DO UPDATE SET
                    rating = 'general',
                    source = 'human',
                    human_cleared = 1,
                    cleared_by = excluded.cleared_by,
                    cleared_at = excluded.cleared_at,
                    clear_note = excluded.clear_note,
                    rated_by = excluded.rated_by,
                    rated_at = excluded.rated_at
                """,
                (
                    version_id,
                    resolved_task,
                    project_id,
                    json.dumps(resolved_ref, ensure_ascii=False),
                    json.dumps([clean_note], ensure_ascii=False),
                    actor,
                    now,
                    actor,
                    now,
                    clean_note,
                ),
            )
            conn.execute(
                """
                INSERT INTO content_rating_events (
                    event_id, version_id, task_id, project_id, event_type,
                    rating, source, rubric_version, evaluator_model,
                    reasons_json, actor_id, note, created_at
                ) VALUES (?, ?, ?, ?, 'clear', 'general', 'human', NULL, NULL, ?, ?, ?, ?)
                """,
                (
                    f"crev-{uuid.uuid4().hex[:16]}",
                    version_id,
                    resolved_task,
                    project_id,
                    json.dumps([clean_note], ensure_ascii=False),
                    actor,
                    clean_note,
                    now,
                ),
            )
            conn.commit()

        return {
            "status": "ok",
            "version_id": version_id,
            "task_id": resolved_task,
            "project_id": project_id,
            "rating": "general",
            "previous_rating": previous_rating,
            "source": "human",
            "human_cleared": True,
            "cleared_at": now,
            "cleared_by": actor,
            "clear_note": clean_note,
        }

    def mark_rating(
        self,
        *,
        version_id: str,
        project_id: str,
        actor_user_id: str,
        reason: str | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        self.ensure_ready()
        if not actor_user_id or not str(actor_user_id).strip():
            raise ContentRatingError(401, "UNAUTHORIZED", "Authenticated human user is required to mark NSFW rating")
        actor = str(actor_user_id).strip()
        clean_reason = str(reason).strip() if reason and str(reason).strip() else "Manually marked NSFW in Studio"
        if len(clean_reason) > 500:
            raise ContentRatingError(400, "INVALID_ARGUMENT", "reason must not exceed 500 characters")
        now = datetime.now(timezone.utc).isoformat()
        resolved_ref = {
            "domain": self.domain,
            "project_id": project_id,
            "version_id": version_id,
            "task_id": task_id,
        }

        with self._get_conn() as conn:
            existing_row = conn.execute(
                "SELECT * FROM content_ratings WHERE version_id = ?", (version_id,)
            ).fetchone()
            previous_rating = existing_row["rating"] if existing_row else "unrated"
            resolved_task = task_id or (existing_row["task_id"] if existing_row else None)

            conn.execute(
                """
                INSERT INTO content_ratings (
                    version_id, task_id, project_id, content_ref_json,
                    rating, source, rubric_version, evaluator_model,
                    reasons_json, rated_by, rated_at, human_cleared,
                    cleared_by, cleared_at, clear_note
                ) VALUES (?, ?, ?, ?, 'nsfw', 'human', 'manual', NULL, ?, ?, ?, 0, NULL, NULL, NULL)
                ON CONFLICT(version_id) DO UPDATE SET
                    rating = 'nsfw',
                    source = 'human',
                    rubric_version = 'manual',
                    reasons_json = excluded.reasons_json,
                    rated_by = excluded.rated_by,
                    rated_at = excluded.rated_at,
                    human_cleared = 0
                """,
                (
                    version_id,
                    resolved_task,
                    project_id,
                    json.dumps(resolved_ref, ensure_ascii=False),
                    json.dumps([clean_reason], ensure_ascii=False),
                    actor,
                    now,
                ),
            )
            conn.execute(
                """
                INSERT INTO content_rating_events (
                    event_id, version_id, task_id, project_id, event_type,
                    rating, source, rubric_version, evaluator_model,
                    reasons_json, actor_id, note, created_at
                ) VALUES (?, ?, ?, ?, 'rate', 'nsfw', 'human', 'manual', NULL, ?, ?, ?, ?)
                """,
                (
                    f"crev-{uuid.uuid4().hex[:16]}",
                    version_id,
                    resolved_task,
                    project_id,
                    json.dumps([clean_reason], ensure_ascii=False),
                    actor,
                    clean_reason,
                    now,
                ),
            )
            conn.commit()

        return {
            "status": "ok",
            "version_id": version_id,
            "task_id": resolved_task,
            "project_id": project_id,
            "rating": "nsfw",
            "previous_rating": previous_rating,
            "source": "human",
            "rubric_version": "manual",
            "reasons": [clean_reason],
            "reason": clean_reason,
            "rated_at": now,
            "rated_by": actor,
            "human_cleared": False,
        }

    def list_events(self, version_id: str) -> list[dict[str, Any]]:
        self.ensure_ready()
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM content_rating_events WHERE version_id = ? ORDER BY created_at ASC",
                (version_id,),
            ).fetchall()
            out = []
            for r in rows:
                out.append(
                    {
                        "event_id": r["event_id"],
                        "version_id": r["version_id"],
                        "task_id": r["task_id"],
                        "project_id": r["project_id"],
                        "event_type": r["event_type"],
                        "rating": r["rating"],
                        "source": r["source"],
                        "rubric_version": r["rubric_version"],
                        "evaluator_model": r["evaluator_model"],
                        "reasons": json.loads(r["reasons_json"] or "[]"),
                        "actor_id": r["actor_id"],
                        "note": r["note"],
                        "created_at": r["created_at"],
                    }
                )
            return out
