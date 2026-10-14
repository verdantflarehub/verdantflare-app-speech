"""Explicit model preparation and offline verification; never used to fetch at inference."""
import concurrent.futures
import hashlib
import json
import os
import shutil
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from .audio import digest_file
from .store import atomic_write

BUNDLED_MANIFEST = Path(os.environ.get("SPEECH_MODEL_MANIFEST", "/opt/service/models/manifest.lock.json"))


def safe_target(root, relative):
    path = PurePosixPath(relative)
    if path.is_absolute() or any(part in (".", "..") for part in path.parts) or "\\" in relative or not relative:
        raise ValueError("invalid_model_path")
    result = Path(root).joinpath(*path.parts)
    if any(p.is_symlink() for p in [result, *result.parents]):
        raise ValueError("model_symlink_rejected")
    return result


def read_manifest(path):
    raw = Path(path).read_bytes()
    manifest = json.loads(raw)
    seen = set()
    for item in manifest["files"]:
        if item["path"] in seen:
            raise ValueError("duplicate_model_path")
        seen.add(item["path"])
        if len(item["sha256"]) != 64 or item["size"] < 0:
            raise ValueError("invalid_model_digest")
        url = urllib.parse.urlparse(item["url"])
        if url.scheme != "https" or url.hostname not in ("huggingface.co", "raw.githubusercontent.com") or url.username or url.query:
            raise ValueError("untrusted_model_source")
    return raw, manifest


def check_file(path, item):
    if not path.is_file() or path.stat().st_size != item["size"] or digest_file(path) != item["sha256"]:
        raise ValueError("model_integrity_failed: " + item["path"])


def extract_resource(root, item, verify_only=False):
    if item.get("archive") != "zip":
        return
    destination = safe_target(root, item["extract_to"])
    with zipfile.ZipFile(safe_target(root, item["path"])) as archive:
        for entry in archive.infolist():
            target = safe_target(destination, entry.filename)
            if entry.is_dir():
                if not verify_only:
                    target.mkdir(parents=True, exist_ok=True)
                continue
            if entry.file_size > 64 * 1024 * 1024:
                raise ValueError("resource_too_large")
            data = archive.read(entry)
            if target.exists():
                if digest_file(target) != hashlib.sha256(data).hexdigest():
                    raise ValueError("resource_integrity_failed")
            elif verify_only:
                raise ValueError("resource_missing")
            else:
                atomic_write(target, data)


def prepare_models(root, manifest_path=BUNDLED_MANIFEST):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    raw, manifest = read_manifest(manifest_path)
    installed = root / "manifest.lock.json"
    if installed.exists() and installed.read_bytes() != raw:
        raise ValueError("different_model_manifest_exists")
    endpoint = os.environ.get("SPEECH_MODEL_ENDPOINT", "https://huggingface.co")
    if endpoint not in ("https://huggingface.co", "https://hf-mirror.com"):
        raise ValueError("untrusted_model_endpoint")

    def download(item):
        path = safe_target(root, item["path"])
        if path.exists():
            check_file(path, item)
            extract_resource(root, item)
            print(json.dumps({"file": item["path"], "state": "reused"}), flush=True)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".download")
        url = item["url"].replace("https://huggingface.co/", endpoint + "/", 1)
        request = urllib.request.Request(url, headers={"User-Agent": "VerdantFlare-Speech/0.1"})
        # Public fixed objects only; no credentials are attached to redirected requests.
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            count = 0
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                count += len(block)
                if count > item["size"]:
                    raise ValueError("model_download_oversize")
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
        check_file(temporary, item)
        os.replace(temporary, path)
        extract_resource(root, item)
        print(json.dumps({"file": item["path"], "state": "verified"}), flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(download, manifest["files"]))
    atomic_write(installed, raw)
    verify_models(root, manifest_path)


def verify_models(root, manifest_path=BUNDLED_MANIFEST):
    root = Path(root)
    raw, manifest = read_manifest(manifest_path)
    if (root / "manifest.lock.json").read_bytes() != raw:
        raise ValueError("model_manifest_mismatch")
    for item in manifest["files"]:
        check_file(safe_target(root, item["path"]), item)
        extract_resource(root, item, verify_only=True)
    return {"model_manifest_sha256": hashlib.sha256(raw).hexdigest(), "files": len(manifest["files"])}


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) == 2 else ""
    root = Path(os.environ.get("SPEECH_MODEL_ROOT", "/models"))
    if mode == "prepare":
        prepare_models(root)
    elif mode == "verify":
        print(json.dumps(verify_models(root)))
    else:
        raise SystemExit("usage: python -m speech_runtime.models prepare|verify")
