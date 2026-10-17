"""Run in a controlled dev Job. No network service or automatic task submission."""

import argparse
from importlib.resources import files
import json
from pathlib import Path

from .assets import prepare_model, verify_model
from .engine import gpu_probe
from .request import CandidateError, parse_request
from .runner import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare-model", "model-probe"):
        command = sub.add_parser(name)
        command.add_argument("--model-root", type=Path, required=True)
    sub.add_parser("gpu-probe")
    check = sub.add_parser("check-request")
    check.add_argument("--request", type=Path, required=True)
    execute = sub.add_parser("generate")
    for name in ("request", "reference", "model-root", "out"):
        execute.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    lock = files("voice_design").joinpath("model.lock.json")
    try:
        if args.command == "prepare-model":
            result = prepare_model(args.model_root, lock)
        elif args.command == "model-probe":
            result = verify_model(args.model_root, lock)
        elif args.command == "gpu-probe":
            result = gpu_probe()
        else:
            request = json.loads(args.request.read_text(encoding="utf-8"))
            if args.command == "check-request":
                parse_request(request)
                result = {"status": "valid_request", "mode": request["mode"], "inference_run": False}
            else:
                report = run(request, args.reference, args.model_root, lock, args.out)
                result = {"status": "candidate_created", "candidate_id": request["candidate_id"], "quality": report["quality"]}
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0
    except Exception as exc:
        code = str(exc) if isinstance(exc, CandidateError) else "operation_failed"
        print(json.dumps({"status": "failed", "error_code": code}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
