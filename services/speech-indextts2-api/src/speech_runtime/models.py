"""Explicit model preparation and offline verification; never used to fetch at inference."""
import concurrent.futures
import fcntl
import hashlib
import http.client
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from .audio import digest_file
from .store import atomic_write

BUNDLED_MANIFEST = Path(os.environ.get("SPEECH_MODEL_MANIFEST", "/opt/service/models/manifest.lock.json"))
BUNDLED_SOURCES = Path("/opt/service/models/download-sources.lock.json")


def read_sources(manifest, path=BUNDLED_SOURCES):
    sources = json.loads(Path(path).read_bytes())
    if sources.get("schema_version") != 1:
        raise ValueError("invalid_download_sources")
    expected = {item["path"]: item for item in manifest["files"]}
    result = {}
    for item in sources["files"]:
        original = expected.get(item["path"])
        if original is None or item["path"] in result or not item["path"].startswith("indextts2/"):
            raise ValueError("unexpected_download_source")
        if item["sha256"] != original["sha256"] or item["size"] != original["size"]:
            raise ValueError("download_source_identity_mismatch")
        if not re.fullmatch(r"[0-9a-f]{40}", item["revision"]):
            raise ValueError("download_source_revision_required")
        url = "https://www.modelscope.cn/models/IndexTeam/IndexTTS-2/resolve/" + item["revision"] + "/" + urllib.parse.quote(item["path"][10:], safe="/")
        if item["url"] != url:
            raise ValueError("untrusted_download_source")
        result[item["path"]] = item["url"]
    return result


def download_file(path, item, endpoint, attempts=4, source_url=None):
    """Resume only a fixed object, validating both transport range and final digest."""
    temporary = path.with_name(path.name + ".download")
    url = source_url or item["url"].replace("https://huggingface.co/", endpoint + "/", 1)
    if source_url is None and urllib.parse.urlparse(item["url"]).hostname == "huggingface.co":
        url += "?download=true"
    for attempt in range(attempts):
        offset = temporary.stat().st_size if temporary.exists() else 0
        if offset >= item["size"] and temporary.exists():
            check_file(temporary, item)
            os.replace(temporary, path)
            return
        headers = {"User-Agent": "huggingface_hub/0.34.4; python/3.11; VerdantFlare-Speech", "Accept-Encoding": "identity"}
        if source_url:
            headers["User-Agent"] = "modelscope/1.18.0; python/3.11; VerdantFlare-Speech"
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as response:
                if response.status == 206:
                    expected = f"bytes {offset}-{item['size'] - 1}/{item['size']}"
                    if response.headers.get("Content-Range") != expected:
                        raise ValueError("invalid_model_content_range")
                    mode = "ab" if offset else "wb"
                elif response.status == 200:
                    offset, mode = 0, "wb"
                else:
                    raise ValueError("invalid_model_http_status")
                with temporary.open(mode) as output:
                    count, reported = offset, offset
                    while block := response.read(1024 * 1024):
                        count += len(block)
                        if count > item["size"]:
                            raise ValueError("model_download_oversize")
                        output.write(block)
                        if count - reported >= 64 * 1024 * 1024:
                            print(json.dumps({"file": item["path"], "state": "downloading", "bytes": count, "total": item["size"]}), flush=True)
                            reported = count
                    output.flush()
                    os.fsync(output.fileno())
                if count != item["size"]:
                    raise OSError("incomplete_model_download")
            check_file(temporary, item)
            os.replace(temporary, path)
            return
        except (OSError, urllib.error.URLError, http.client.IncompleteRead) as error:
            print(json.dumps({"file": item["path"], "state": "download_retry" if attempt + 1 < attempts else "download_failed",
                              "attempt": attempt + 1, "error_class": type(error).__name__, "http_status": getattr(error, "code", None)}), flush=True)
            if attempt + 1 == attempts:
                raise RuntimeError("model_download_failed: " + item["path"]) from None
            time.sleep(min(2 ** attempt, 8))


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
    with (root / ".prepare.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("model_preparation_already_active") from None
        _prepare_models(root, manifest_path)


def _prepare_models(root, manifest_path):
    raw, manifest = read_manifest(manifest_path)
    installed = root / "manifest.lock.json"
    if installed.exists() and installed.read_bytes() != raw:
        raise ValueError("different_model_manifest_exists")
    endpoint = os.environ.get("SPEECH_MODEL_ENDPOINT", "https://huggingface.co")
    if endpoint not in ("https://huggingface.co", "https://hf-mirror.com"):
        raise ValueError("untrusted_model_endpoint")
    enabled = os.environ.get("SPEECH_MODEL_USE_MIRRORS", "1")
    if enabled not in ("0", "1"):
        raise ValueError("invalid_model_mirror_setting")
    sources = read_sources(manifest) if enabled == "1" else {}

    def download(item):
        path = safe_target(root, item["path"])
        if path.exists():
            check_file(path, item)
            extract_resource(root, item)
            print(json.dumps({"file": item["path"], "state": "reused"}), flush=True)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        # Public fixed objects only; no credentials are attached to redirected requests.
        download_file(path, item, endpoint, source_url=sources.get(item["path"]))
        extract_resource(root, item)
        print(json.dumps({"file": item["path"], "state": "verified"}), flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(download, item) for item in manifest["files"]]
        try:
            for future in concurrent.futures.as_completed(futures):
                future.result()
        except Exception:
            for future in futures:
                future.cancel()
            raise
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
