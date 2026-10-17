"""Validate requests without importing a speech model."""

import json
import re
import unicodedata
from importlib.resources import files

from jsonschema import Draft202012Validator


class CandidateError(Exception):
    """A stable error code safe to display without input values or local paths."""


def parse_request(data):
    try:
        schema = json.loads(files("voice_design").joinpath("request.schema.json").read_text())
        Draft202012Validator(schema).validate(data)
    except Exception as exc:
        raise CandidateError("invalid_request") from exc
    for key in ("text", "reference_traits", "change_request", "target_description", "style_instruction"):
        if key not in data:
            continue
        value = data[key]
        if not value.strip() or any(unicodedata.category(c).startswith("C") and c not in "\n\t\r" for c in value):
            raise CandidateError("invalid_text_characters")
        if any(c in value for c in "()（）<>"):
            raise CandidateError("embedded_control_instruction")
    description = data.get("target_description") if data["mode"] == "design" else data.get("style_instruction")
    model_text = f"({description}){data['text']}" if description else data["text"]
    # Match the pinned upstream's unconditional whitespace handling, recording both forms.
    model_text = re.sub(r"\s+", " ", model_text.replace("\n", " "))
    return model_text


def generation_kwargs(request, model_text, reference_path):
    result = {
        "text": model_text, "seed": request["seed"], "cfg_value": 2.0,
        "inference_timesteps": 10, "min_len": 2, "max_len": 64,
        "normalize": False, "denoise": False,
        "retry_badcase": False, "retry_badcase_max_times": 1,
    }
    if request["mode"] == "clone":
        result["reference_wav_path"] = str(reference_path)
    return result
