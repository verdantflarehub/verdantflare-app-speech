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


class TaskNotFound(Failure):
    def __init__(self, code="task_not_found", status=404, retry=False):
        actual_code = "task_not_found" if (str(code).startswith("NOT_FOUND") or ":" in str(code) or " " in str(code)) else str(code)
        super().__init__(actual_code, status=status, retry=retry)
        self.detail = str(code)


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
    if kind == "rate_content":
        if not isinstance(value, dict):
            raise Failure("invalid_arguments")
        return value
    schema = {"$ref": "#/$defs/" + kind, "$defs": CONTRACT["$defs"]}
    if not Draft202012Validator(schema).is_valid(value):
        raise Failure("invalid_arguments")
    if kind == "create":
        if not value["text"].strip() or any(ord(c) < 32 and c not in "\n\t\r" for c in value["text"]):
            raise Failure("invalid_text")
        emotion_keys = [k for k in ("emotion_vector", "emotion_text", "emotion_reference") if k in value]
        if len(emotion_keys) > 1:
            raise Failure("invalid_arguments")
        if sum(value.get("emotion_vector", [])) > 0.800000001:
            raise Failure("invalid_emotion_vector")
        if "emotion_text" in value and not value["emotion_text"].strip():
            raise Failure("invalid_arguments")
    return value


def normalize_text(text):
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u3000", " ").replace("\t", " ")
    lines = [" ".join(line.split()) for line in cleaned.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def apply_lexicon(text, entries=None):
    normalized = normalize_text(text)
    sent = normalized
    hits = []
    if entries:
        ordered = sorted(entries, key=lambda item: len(item.get("term", "")), reverse=True)
        for item in ordered:
            term = item.get("term", "")
            replacement = item.get("replacement", "")
            if not term:
                continue
            count = sent.count(term)
            if count > 0:
                sent = sent.replace(term, replacement)
                hits.append({"term": term, "replacement": replacement})
    return normalized, sent, hits


def build_text_record(source, entries=None):
    _, sent, hits = apply_lexicon(source, entries)
    return {"source": source, "sent": sent, "lexicon_hits": hits}


def normalize_world_voice(wv):
    version_id = wv.get("version_id") or wv.get("asset_version_id")
    return {"asset_id": wv["asset_id"], "version_id": version_id}


def compute_input_digest(sent_text, voice_source, model_manifest_sha256, req):
    emotion = None
    if "emotion_vector" in req:
        emotion = {"mode": "emotion_vector", "value": req["emotion_vector"]}
    elif "emotion_text" in req:
        emotion = {"mode": "emotion_text", "value": req["emotion_text"]}
    elif "emotion_reference" in req:
        emotion = {
            "mode": "emotion_reference",
            "content_ref": req["emotion_reference"]["content_ref"],
            "alpha": req["emotion_reference"]["alpha"],
        }
    payload = {
        "sent_text": sent_text,
        "voice_source": voice_source,
        "model_manifest_sha256": model_manifest_sha256,
        "emotion": emotion,
    }
    return digest(canonical(payload).encode())


def check_take_staleness(generation, current_voice_source, current_sent_text=None, current_model_manifest_sha256=None):
    reasons = []
    recorded_voice = generation.get("voice_source") or {}
    if recorded_voice.get("kind") != current_voice_source.get("kind"):
        reasons.append("voice_source_kind_changed")
    elif current_voice_source.get("kind") == "world":
        rec_wv = normalize_world_voice(recorded_voice.get("world_voice") or {})
        cur_wv = normalize_world_voice(current_voice_source.get("world_voice") or {})
        if (rec_wv != cur_wv or
                recorded_voice.get("reference_sha256") != current_voice_source.get("reference_sha256")):
            reasons.append("world_voice_version_changed")
    else:
        if (recorded_voice.get("voice_profile") != current_voice_source.get("voice_profile") or
                recorded_voice.get("profile_sha256") != current_voice_source.get("profile_sha256")):
            reasons.append("voice_profile_changed")
    if current_sent_text is not None and (generation.get("text") or {}).get("sent") != current_sent_text:
        reasons.append("text_changed")
    if current_model_manifest_sha256 is not None and generation.get("model_manifest_sha256") != current_model_manifest_sha256:
        reasons.append("model_version_changed")
    return {
        "stale": bool(reasons),
        "reasons": reasons,
        "input_digest": generation.get("input_digest"),
    }


class EngineAdapter:
    engine_id = ""

    def capability(self, ready_runtime=None, model_manifest_sha256=None):
        raise NotImplementedError


class IndexTTS2Adapter(EngineAdapter):
    engine_id = "indextts2"
    emotion_inputs = ["emotion_vector", "emotion_text", "emotion_reference"]
    unsupported = ["exact_duration", "speed", "cancel", "non_verbal", "streaming"]

    def capability(self, ready_runtime=None, model_manifest_sha256=None):
        is_ready = bool(ready_runtime and ready_runtime.get("ready"))
        manifest_hash = (ready_runtime or {}).get("model_manifest_sha256") or model_manifest_sha256 or ("0" * 64)
        max_chars = int((ready_runtime or {}).get("max_text_chars", 300))
        ref_audio = (ready_runtime or {}).get("reference_audio") or {
            "min_seconds": 3,
            "max_seconds": 15,
            "max_bytes": 10485760,
            "sample_rates_hz": [16000, 22050, 24000, 44100, 48000],
            "channels": 1,
        }
        return {
            "engine": self.engine_id,
            "ready": is_ready,
            "model_manifest_sha256": manifest_hash,
            "languages": ["zh-CN"],
            "max_text_chars": max_chars,
            "reference_audio": ref_audio,
            "emotion_inputs": list(self.emotion_inputs),
            "unsupported": list(self.unsupported),
        }


DEFAULT_ADAPTER = IndexTTS2Adapter()



def principal(value):
    keys = {"organization_id", "subject_id", "request_id"}
    if not isinstance(value, dict) or set(value) != keys or not all(isinstance(v, str) and UUID7.fullmatch(v) for v in value.values()):
        raise Failure("invalid_identity", 403)
    return value


def scope(p, project_id):
    return digest(canonical([p["organization_id"], p["subject_id"], project_id]).encode())


DESCRIPTIONS = {
    "capabilities": "Discover registered self-hosted speech engines and their per-engine capabilities.",
    "voices": "List World reference-voice assets and project candidate voice profiles in a readable project revision.",
    "create": "Generate one Chinese dialogue or narration take using a pinned World reference voice or project voice profile.",
    "status": "Recover the original speech task, phase, timings, and review state. Do not resubmit inference after an archive failure.",
    "result": "Return central Artifact ContentRefs for a completed speech take. Download through artifact.read and explicitly archive in the same Project.",
    "list": "List speech tasks for the caller (NSFW items are unconditionally excluded from MCP responses).",
    "rate_content": "Record content rating (general or nsfw) for a speech task or output artifact version. Add-only; cannot clear an existing nsfw rating.",
}
TOOLS = [{"name": "speech." + k, "description": v,
          "inputSchema": {**CONTRACT["$defs"][k], "$defs": CONTRACT["$defs"]}}
         for k, v in DESCRIPTIONS.items()]
SPEECH_TOOLS_SCHEMA = TOOLS


def public(row, *, filter_nsfw_outputs=None):
    state = row["state"]
    out = {k: row[k] for k in ("task_id", "project_id", "status", "phase")}
    out.update(
        created_at=row["created_at"].isoformat(),
        updated_at=row["updated_at"].isoformat(),
        service_version=__version__,
        runtime_task_id=state.get("runtime_task_id"),
        project_archive=state.get("project_archive", "pending"),
        candidate_state=state.get("candidate_state", "candidate"),
        listening_review=state.get("listening_review", "pending"),
    )
    if state.get("review") in ("pending", "approved", "rejected"):
        out["review"] = state["review"]
    if state.get("review_note"):
        out["review_note"] = state["review_note"]
    if state.get("timings"):
        out["timings"] = state["timings"]
    if state.get("voice_source"):
        out["voice_source"] = state["voice_source"]
    if row.get("last_error"):
        out["error"] = {
            "code": row["last_error"],
            "retrying": row["status"] in ("pending", "running"),
            "recovery_required": row["last_error"] in ("recovery_required", "model_version_changed", "unverified_voice_engine"),
        }
    if row["status"] == "succeeded":
        outputs = state["outputs"]
        if filter_nsfw_outputs is not None:
            filtered_outputs = []
            for entry in outputs:
                cref = entry.get("content_ref") if isinstance(entry, dict) else None
                vid = (cref or {}).get("version_id") or (cref or {}).get("artifact_id") or (entry.get("version_id") if isinstance(entry, dict) else None)
                if vid and filter_nsfw_outputs.is_nsfw(vid):
                    continue
                filtered_outputs.append(entry)
            outputs = filtered_outputs
        out["outputs"] = outputs
        out["model_manifest_sha256"] = state["profile"]["model_manifest_sha256"]
    return out

