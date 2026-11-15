"""Single-process durable GPU queue; scoped immutable inputs and atomic outputs."""
try:
    import fcntl
except ImportError:
    fcntl = None  # type: ignore[assignment]
    import msvcrt
import hashlib
import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .audio import inspect_wav, digest_file
from .domain import Failure, SHA256, UUID7, canonical, new_id, validate_create


def _lock_file_nonblocking(handle) -> None:
    if fcntl is not None:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return
    handle.seek(0)
    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)


def _unlock_file(handle) -> None:
    if fcntl is not None:
        fcntl.flock(handle, fcntl.LOCK_UN)
        return
    handle.seek(0)
    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    if os.name != "nt":
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


class Store:
    def __init__(self, root, model_hash):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        if not SHA256.fullmatch(model_hash):
            raise ValueError("invalid_model_manifest_hash")
        self.model_hash = model_hash
        self.lockfile = (self.root / "runtime.lock").open("r+" if (self.root / "runtime.lock").exists() else "w+")
        try:
            _lock_file_nonblocking(self.lockfile)
        except OSError:
            self.lockfile.close()
            raise RuntimeError("runtime_already_active") from None
        self.mutex = threading.RLock()
        self.db = sqlite3.connect(self.root / "tasks.sqlite3", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("""CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY, scope TEXT NOT NULL, idem TEXT NOT NULL, input_hash TEXT NOT NULL,
            input_json TEXT NOT NULL, model_hash TEXT NOT NULL, status TEXT NOT NULL, error TEXT,
            created TEXT NOT NULL, updated TEXT NOT NULL, UNIQUE(scope, idem))""")
        self.db.commit()
        self.recover()

    def close(self):
        with self.mutex:
            self.db.close()
            _unlock_file(self.lockfile)
            self.lockfile.close()

    @contextmanager
    def transaction(self):
        with self.mutex:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                yield
                self.db.commit()
            except BaseException:
                self.db.rollback()
                raise

    def reference_path(self, scope, checksum):
        if not SHA256.fullmatch(checksum):
            raise Failure("invalid_reference_hash")
        return self.root / "references" / scope.key / (checksum + ".wav")

    def put_reference(self, scope, checksum, data):
        if hashlib.sha256(data).hexdigest() != checksum:
            raise Failure("reference_hash_mismatch")
        report = inspect_wav(data, reference=True)
        path = self.reference_path(scope, checksum)
        with self.mutex:
            if path.exists():
                if digest_file(path) != checksum:
                    raise Failure("stored_reference_corrupt", 409)
            else:
                atomic_write(path, data)
        return report

    def create(self, scope, inputs):
        validate_create(inputs)
        payload = canonical(inputs)
        checksum = hashlib.sha256(payload.encode()).hexdigest()
        with self.transaction():
            prior = self.db.execute("SELECT * FROM tasks WHERE scope=? AND idem=?", (scope.key, inputs["idempotency_key"])).fetchone()
            if prior:
                if prior["input_hash"] != checksum:
                    raise Failure("idempotency_conflict", 409)
                return self.public(prior)
            reference = self.reference_path(scope, inputs["reference_sha256"])
            if not reference.is_file():
                raise Failure("reference_not_found", 404)
            if digest_file(reference) != inputs["reference_sha256"]:
                raise Failure("stored_reference_corrupt", 409)
            if "emotion_reference" in inputs:
                emo_hash = inputs["emotion_reference"]["reference_sha256"]
                emo_ref = self.reference_path(scope, emo_hash)
                if not emo_ref.is_file():
                    raise Failure("reference_not_found", 404)
                if digest_file(emo_ref) != emo_hash:
                    raise Failure("stored_reference_corrupt", 409)
            count = self.db.execute("SELECT count(*) FROM tasks WHERE status IN ('pending','running')").fetchone()[0]
            if count >= 16:
                raise Failure("queue_full", 429)
            task_id, timestamp = new_id(), now()
            self.db.execute("INSERT INTO tasks VALUES (?,?,?,?,?,?,?,NULL,?,?)", (task_id, scope.key, inputs["idempotency_key"], checksum, payload, self.model_hash, "pending", timestamp, timestamp))
            return self.public(self.db.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone())

    def get(self, scope, task_id):
        if not UUID7.fullmatch(task_id):
            raise Failure("task_not_found", 404)
        with self.mutex:
            row = self.db.execute("SELECT * FROM tasks WHERE id=? AND scope=?", (task_id, scope.key)).fetchone()
        if not row:
            raise Failure("task_not_found", 404)
        return self.public(row)

    def public(self, row):
        out = {"task_id": row["id"], "status": row["status"], "created_at": row["created"], "updated_at": row["updated"], "model_manifest_sha256": row["model_hash"]}
        if row["error"]:
            out["error"] = {"code": row["error"], "recovery_required": row["error"] in ("recovery_required", "model_version_changed")}
        if row["status"] == "succeeded":
            out["outputs"] = self.validate_completion(row["id"], row["model_hash"])
        return out

    def claim(self):
        with self.transaction():
            row = self.db.execute("SELECT * FROM tasks WHERE status='pending' ORDER BY created,id LIMIT 1").fetchone()
            if not row:
                return None
            if row["model_hash"] != self.model_hash:
                self.db.execute("UPDATE tasks SET status='failed',error='model_version_changed',updated=? WHERE id=?", (now(), row["id"]))
                return None
            self.db.execute("UPDATE tasks SET status='running',updated=? WHERE id=?", (now(), row["id"]))
            return dict(row)

    def output_dir(self, task_id):
        if not UUID7.fullmatch(task_id):
            raise Failure("invalid_task_id")
        return self.root / "outputs" / task_id

    def validate_completion(self, task_id, model_hash):
        root = self.output_dir(task_id)
        try:
            manifest = json.loads((root / "complete.json").read_text())
            if manifest["model_manifest_sha256"] != model_hash or manifest["task_id"] != task_id:
                raise ValueError()
            entries = manifest["outputs"]
            if {x["name"] for x in entries} != {"original.wav", "editing.wav", "report.json"} or len(entries) != 3:
                raise ValueError()
            for entry in entries:
                file = root / entry["name"]
                if file.is_symlink() or file.stat().st_size != entry["size"] or digest_file(file) != entry["sha256"]:
                    raise ValueError()
            return entries
        except (OSError, KeyError, ValueError, TypeError):
            raise Failure("output_integrity_failed", 500) from None

    def finish(self, task_id):
        self.validate_completion(task_id, self.model_hash)
        with self.transaction():
            self.db.execute("UPDATE tasks SET status='succeeded',error=NULL,updated=? WHERE id=? AND status='running'", (now(), task_id))

    def fail(self, task_id, code):
        with self.transaction():
            self.db.execute("UPDATE tasks SET status='failed',error=?,updated=? WHERE id=? AND status='running'", (code, now(), task_id))

    def active_count(self) -> int:
        with self.mutex:
            row = self.db.execute("SELECT count(*) FROM tasks WHERE status IN ('pending','running')").fetchone()
            return int(row[0]) if row else 0

    def abort_pending_fence(self, task_id: str) -> bool:
        with self.transaction():
            cur = self.db.execute(
                "UPDATE tasks SET status='failed',error='stale_fencing_token',updated=? WHERE id=? AND status='pending'",
                (now(), task_id),
            )
            return cur.rowcount > 0

    def recover(self):
        with self.transaction():
            for row in self.db.execute("SELECT * FROM tasks WHERE status='running'").fetchall():
                try:
                    self.validate_completion(row["id"], row["model_hash"])
                    status, error = "succeeded", None
                except Failure:
                    status, error = "failed", "recovery_required"
                self.db.execute("UPDATE tasks SET status=?,error=?,updated=? WHERE id=?", (status, error, now(), row["id"]))
