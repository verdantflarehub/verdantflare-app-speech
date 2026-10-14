"""Pure contract validation; no inference or external I/O."""
import hashlib
import json
import re
import secrets
import time
import uuid
from pathlib import Path

from jsonschema import Draft202012Validator

from . import __version__

UUID7 = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
HASH = re.compile(r"^[0-9a-f]{64}$")
CONTRACT = json.loads(Path(__file__).with_name("contract.json").read_text())


class Failure(Exception):
    def __init__(self, code, status=400, retry=False):
        super().__init__(code)
        self.code, self.status, self.retry = code, status, retry


def new_id():
    return str(uuid.UUID(int=(int(time.time() * 1000) << 80) | (7 << 76) |
                         (secrets.randbits(12) << 64) | (2 << 62) | secrets.randbits(62)))


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def strict_json(data):
    def pairs(items):
        out = {}
        for k, v in items:
            if k in out:
                raise Failure("duplicate_json_key")
            out[k] = v
        return out
    try:
        return json.loads(data, object_pairs_hook=pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(Failure("invalid_number")))
    except (ValueError, UnicodeError, RecursionError):
        raise Failure("invalid_json") from None


def validate(kind, value):
    schema = {"$ref": "#/$defs/" + kind, "$defs": CONTRACT["$defs"]}
    if not Draft202012Validator(schema).is_valid(value):
        raise Failure("invalid_arguments")
    if kind == "create":
        if not value["text"].strip() or any(ord(c) < 32 and c not in "\n\t" for c in value["text"]):
            raise Failure("invalid_text")
        if sum(value.get("emotion_vector", [])) > 0.800000001:
            raise Failure("invalid_emotion_vector")
    return value


def principal(value):
    keys = {"organization_id", "subject_id", "request_id"}
    if not isinstance(value, dict) or set(value) != keys or not all(isinstance(v, str) and UUID7.fullmatch(v) for v in value.values()):
        raise Failure("invalid_identity", 403)
    return value


def scope(p, project_id):
    return digest(canonical([p["organization_id"], p["subject_id"], project_id]).encode())


DESCRIPTIONS = {
    "capabilities": "Discover the ready self-hosted speech engine and its actual limits.",
    "voices": "List fixed voice profiles in a readable project revision. References are conditional inputs, not trained weights.",
    "create": "Generate one Chinese dialogue or narration take using a fixed project voice profile. Persisted idempotency key required; save candidates through Project after completion.",
    "status": "Recover the original speech task and phase. Do not resubmit inference after an archive failure.",
    "result": "Return central Artifact ContentRefs for a completed speech take. Download through artifact.read and explicitly archive in the same Project.",
}
TOOLS = [{"name": "speech." + k, "description": v,
          "inputSchema": {**CONTRACT["$defs"][k], "$defs": CONTRACT["$defs"]}}
         for k, v in DESCRIPTIONS.items()]


def public(row):
    state = row["state"]
    out = {k: row[k] for k in ("task_id", "project_id", "status", "phase")}
    out.update(created_at=row["created_at"].isoformat(), updated_at=row["updated_at"].isoformat(),
               service_version=__version__, runtime_task_id=state.get("runtime_task_id"),
               project_archive="pending", candidate_state="candidate", listening_review="pending")
    if row.get("last_error"):
        out["error"] = {"code": row["last_error"], "retrying": row["status"] in ("pending", "running"),
                        "recovery_required": row["last_error"] in ("recovery_required", "model_version_changed")}
    if row["status"] == "succeeded":
        out["outputs"] = state["outputs"]
        out["model_manifest_sha256"] = state["profile"]["model_manifest_sha256"]
    return out
