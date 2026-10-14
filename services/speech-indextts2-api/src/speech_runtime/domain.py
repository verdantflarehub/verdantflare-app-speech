"""Strict internal inputs; no model imports or network access."""
import hashlib
import json
import math
import re
import secrets
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from jsonschema import Draft202012Validator

UUID7 = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
CONTRACT = json.loads(Path(__file__).with_name("contract.json").read_text())
CREATE_VALIDATOR = Draft202012Validator(CONTRACT["$defs"]["create"])


class Failure(Exception):
    def __init__(self, code, status=400):
        super().__init__(code)
        self.code, self.status = code, status


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise Failure("duplicate_json_key")
            result[key] = value
        return result
    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda _: (_ for _ in ()).throw(Failure("nonfinite_number")))
    except (ValueError, UnicodeError):
        raise Failure("invalid_json") from None


def validate_create(value):
    if not CREATE_VALIDATOR.is_valid(value):
        raise Failure("invalid_input")
    if not value["text"].strip() or any(ord(c) < 32 and c not in "\n\t" for c in value["text"]):
        raise Failure("invalid_text")
    vector = value.get("emotion_vector")
    if vector is not None and (any(not math.isfinite(x) for x in vector) or sum(vector) > 0.800000001):
        raise Failure("invalid_emotion_vector")
    if "target_duration_seconds" in value and not math.isfinite(value["target_duration_seconds"]):
        raise Failure("invalid_duration")
    return value


def new_id():
    millis = int(time.time() * 1000)
    return str(uuid.UUID(int=(millis << 80) | (7 << 76) | (secrets.randbits(12) << 64) | (2 << 62) | secrets.randbits(62)))


@dataclass(frozen=True)
class Scope:
    organization_id: str
    subject_id: str
    project_id: str

    def __post_init__(self):
        if not all(UUID7.fullmatch(v) for v in (self.organization_id, self.subject_id, self.project_id)):
            raise Failure("invalid_scope", 403)

    @property
    def key(self):
        return hashlib.sha256(canonical(self.__dict__).encode()).hexdigest()
