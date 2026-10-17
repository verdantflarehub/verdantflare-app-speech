"""Hash-locked, offline assets; preparation is an explicit separate operation."""

import fcntl
import hashlib
import json
import os
from pathlib import Path
import urllib.request

from .request import CandidateError


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def load_lock(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def asset_path(root, name):
    # Locks have flat names. Disallow symlinks even if they point inside the root.
    if not name or name in (".", "..") or "/" in name or "\\" in name:
        raise CandidateError("invalid_asset_name")
    path = Path(root) / name
    if path.is_symlink():
        raise CandidateError("asset_symlink")
    return path


def matches(path, entry):
    return path.is_file() and path.stat().st_size == entry["size"] and digest(path) == entry["sha256"]


def verify_model(root, lock_path):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise CandidateError("invalid_model_root")
    lock = load_lock(lock_path)
    expected = {entry["path"] for entry in lock["files"]}
    # Prevent upstream loading an unverified alternative VAE or weights file.
    if any(p.name not in expected | {".prepare.lock"} for p in root.iterdir()):
        raise CandidateError("unexpected_model_asset")
    for entry in lock["files"]:
        if not matches(asset_path(root, entry["path"]), entry):
            raise CandidateError("model_integrity_failed")
    if load_lock(root / "config.json").get("architecture") != "voxcpm2":
        raise CandidateError("model_architecture_mismatch")
    return {"repository": lock["repository"], "revision": lock["revision"], "manifest_sha256": digest(lock_path)}


def prepare_model(root, lock_path):
    """Download once into a dedicated model directory. Never overwrite existing assets."""
    root = Path(root)
    if root.is_symlink():
        raise CandidateError("invalid_model_root")
    root.mkdir(parents=True, exist_ok=True)
    lock_file = asset_path(root, ".prepare.lock")
    with lock_file.open("a") as guard:
        try:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CandidateError("model_prepare_in_progress") from exc
        lock = load_lock(lock_path)
        expected = {e["path"] for e in lock["files"]} | {e["path"] + ".part" for e in lock["files"]} | {".prepare.lock"}
        if any(p.name not in expected for p in root.iterdir()):
            raise CandidateError("unexpected_model_asset")
        for entry in lock["files"]:
            path = asset_path(root, entry["path"])
            if path.exists():
                if not matches(path, entry):
                    raise CandidateError("existing_model_asset_mismatch")
                continue
            part = asset_path(root, entry["path"] + ".part")
            # Preserve partial downloads on failure; a repeat resumes that same file.
            start = part.stat().st_size if part.exists() else 0
            if start > entry["size"]:
                raise CandidateError("partial_model_asset_oversized")
            if start < entry["size"]:
                headers = {"Range": f"bytes={start}-"} if start else {}
                with urllib.request.urlopen(urllib.request.Request(entry["url"], headers=headers), timeout=120) as source:
                    if start and (source.status != 206 or not source.headers.get("Content-Range", "").startswith(f"bytes {start}-")):
                        raise CandidateError("model_resume_not_supported")
                    with part.open("ab" if part.exists() else "xb") as target:
                        count = start
                        while block := source.read(1024 * 1024):
                            count += len(block)
                            if count > entry["size"]:
                                raise CandidateError("model_download_oversized")
                            target.write(block)
                        target.flush()
                        os.fsync(target.fileno())
            if not matches(part, entry):
                raise CandidateError("download_integrity_failed")
            # Hard link is atomic and refuses to replace a concurrently created final file.
            os.link(part, path)
            part.unlink()
    return verify_model(root, lock_path)
