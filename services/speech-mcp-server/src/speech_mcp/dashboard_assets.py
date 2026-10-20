"""Deterministic package builder for Speech MCP Dashboard assets."""
from pathlib import Path

from . import __version__
from .domain import Failure, canonical, digest

ASSET_DIR = Path(__file__).with_name("dashboard_src")
MIME_BY_SUFFIX = {
    ".html": "text/html",
    ".css": "text/css",
    ".mjs": "text/javascript",
    ".js": "text/javascript",
}


def _load_files():
    files = {}
    for name in ("index.html", "dashboard.css", "dashboard.mjs"):
        path = ASSET_DIR / name
        text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        files[name] = text.encode("utf-8")
    return files


def build_bundle(version=None):
    app_version = version or __version__
    raw_files = _load_files()
    index_payload = {
        "schemaVersion": 1,
        "files": {name: digest(data) for name, data in sorted(raw_files.items())},
    }
    index_bytes = canonical(index_payload).encode("utf-8")
    artifact_digest = "sha256:" + digest(index_bytes)
    declaration_payload = {
        "schemaVersion": 1,
        "appId": "speech-mcp",
        "appVersion": app_version,
        "dashboard": {
            "entryRef": "dashboard/index.html",
            "artifactDigest": artifact_digest,
            "adapter": "wujie",
            "sdkMajor": 1,
            "minimumSdkVersion": "1.2.0",
            "requiredCapabilities": ["app.request", "context.read"],
            "optionalCapabilities": ["navigation.open"],
            "appApiPolicyRef": "speech-mcp-dashboard-v1",
            "keepAlive": False,
        },
    }
    declaration_bytes = canonical(declaration_payload).encode("utf-8")
    declaration_digest = "sha256:" + digest(declaration_bytes)
    return {
        "files": raw_files,
        "index": index_payload,
        "index_bytes": index_bytes,
        "artifact_digest": artifact_digest,
        "declaration": declaration_payload,
        "declaration_bytes": declaration_bytes,
        "declaration_digest": declaration_digest,
    }


def get_asset(name):
    bundle = build_bundle()
    if name not in bundle["files"]:
        raise Failure("asset_not_found", 404)
    suffix = Path(name).suffix
    mime = MIME_BY_SUFFIX.get(suffix, "application/octet-stream")
    return bundle["files"][name], mime
