"""Controlled internal clients. No caller URLs, redirects or user bearer forwarding."""
import os
import ssl
from urllib.parse import urlencode, urlsplit

import httpx

from .domain import Failure, canonical, digest, strict_json, validate

MAX_OUTPUT = 64 << 20


def token(key):
    value = os.environ.get(key, "")
    if len(value) < 32 or any(c.isspace() for c in value):
        raise RuntimeError("invalid_service_credential:" + key)
    return value


class Client:
    def __init__(self, base, credential, ca="", transport=None, allow_http=False):
        u = urlsplit(base)
        if (not u.hostname or u.username or u.password or u.query or u.fragment or
                u.path not in ("", "/") or u.scheme not in (("http", "https") if allow_http else ("https",))):
            raise RuntimeError("invalid_internal_origin")
        self.base, self.token = base.rstrip("/"), credential
        context = ssl.create_default_context()
        if ca:
            context.load_verify_locations(cafile=ca)
        self.http = httpx.AsyncClient(verify=context, trust_env=False, follow_redirects=False,
                                     timeout=30, transport=transport)

    async def close(self):
        await self.http.aclose()

    async def request(self, p, method, path, payload=None, data=None, mime="application/json", project_id=None, limit=4 << 20):
        headers = {"Authorization": "Bearer " + self.token, "X-Organization-Id": p["organization_id"],
                   "X-User-Id": p["subject_id"], "X-Request-Id": p["request_id"], "Content-Type": mime}
        if project_id:
            headers["X-Project-Id"] = project_id
        if payload is not None:
            data = canonical(payload).encode()
        try:
            async with self.http.stream(method, self.base + path, headers=headers, content=data) as r:
                if r.status_code >= 300:
                    if r.status_code in (401, 403):
                        raise Failure("permission_denied", 403, True)
                    # Server details can contain internal URLs/credentials; never relay them.
                    raw = bytearray()
                    async for chunk in r.aiter_bytes():
                        raw.extend(chunk)
                        if len(raw) > 4096:
                            break
                    if len(raw) <= 4096:
                        try:
                            err = strict_json(raw).get("error", {})
                            if isinstance(err, dict) and err.get("code") in ("model_version_changed", "recovery_required", "idempotency_conflict"):
                                raise Failure(err["code"], 409)
                        except (AttributeError, ValueError):
                            pass
                    if r.status_code in (400, 413, 415, 422):
                        raise Failure("dependency_rejected_input", 400)
                    if r.status_code == 404:
                        raise Failure("dependency_not_found", 404)
                    raise Failure("dependency_unavailable", 503, True)
                result = bytearray()
                async for chunk in r.aiter_bytes():
                    result.extend(chunk)
                    if len(result) > limit:
                        raise Failure("dependency_output_too_large", 502)
                return bytes(result)
        except httpx.HTTPError:
            raise Failure("dependency_unavailable", 503, True) from None

    async def json(self, p, method, path, payload=None, **kwargs):
        return strict_json(await self.request(p, method, path, payload, **kwargs))


class Peers:
    def __init__(self, project, artifact, runtime):
        self.project, self.artifact, self.runtime = project, artifact, runtime

    @classmethod
    def environment(cls):
        return cls(Client(os.environ["SPEECH_PROJECT_URL"], token("SPEECH_PROJECT_TOKEN"), os.environ["SPEECH_PLATFORM_CA_FILE"]),
                   Client(os.environ["SPEECH_ARTIFACT_URL"], token("SPEECH_ARTIFACT_TOKEN"), os.environ["SPEECH_PLATFORM_CA_FILE"]),
                   Client(os.environ["SPEECH_RUNTIME_URL"], token("SPEECH_RUNTIME_TOKEN"), allow_http=True))

    async def close(self):
        for c in (self.project, self.artifact, self.runtime):
            await c.close()

    async def open(self, p, project_id, revision_id=None, write=False):
        args = {"project_id": project_id}
        # Project strict JSON rejects explicit zero values for omitempty fields.
        if write:
            args["require_write"] = True
        if revision_id:
            args["revision_id"] = revision_id
        result = await self.project.json(p, "POST", "/internal/v1/project/open", args)
        return result

    @staticmethod
    def files(opened):
        return opened["manifest"]["files"]

    @classmethod
    def contains(cls, opened, ref):
        if not any(f["content_ref"] == ref for f in cls.files(opened)):
            raise Failure("reference_outside_revision", 403)

    async def content(self, p, ref, project_id, revision_id, limit):
        validate("content_ref", ref)
        query = urlencode({"store_id": ref["store_id"], "artifact_id": ref["artifact_id"],
                           "project_id": project_id, "project_revision_id": revision_id})
        path = "/v2/artifacts/" + ref["version_id"]
        meta = await self.artifact.json(p, "GET", path + "?" + query)
        if meta.get("size", limit + 1) > limit or any(meta.get(k) != v for k, v in ref.items()):
            raise Failure("invalid_reference_metadata")
        data = await self.artifact.request(p, "GET", path + "/content?" + query, limit=limit)
        if digest(data) != meta.get("sha256") or len(data) != meta["size"]:
            raise Failure("reference_integrity_failed", 502)
        return data, meta

    async def profile(self, p, opened, project_id, revision_id, ref):
        self.contains(opened, ref)
        raw, _ = await self.content(p, ref, project_id, revision_id, 16384)
        profile = validate("profile", strict_json(raw))
        self.contains(opened, profile["reference"]["content_ref"])
        return profile, digest(raw)

    async def archive(self, p, row, entry, data):
        source = {"kind": "task_output", "project_id": row["project_id"], "service_id": "speech",
                  "task_id": row["task_id"], "run_id": row["state"]["runtime_task_id"]}
        declaration = {"write_id": entry["write_id"], "source": source, "sha256": entry["sha256"],
                       "size": entry["size"], "mime": entry["mime"]}
        upload = await self.artifact.json(p, "POST", "/v2/artifacts/uploads", declaration)
        validate("id", upload.get("upload_id"))
        path = "/v2/artifacts/uploads/" + upload["upload_id"]
        if upload.get("content_path") != path + "/content" or upload.get("state") not in ("prepared", "committed"):
            raise Failure("invalid_upload_response", 502)
        if upload["state"] != "committed":
            await self.artifact.request(p, "PUT", path + "/content", data=data, mime=entry["mime"])
        version = await self.artifact.json(p, "POST", path + "/commit")
        ref = {k: version.get(k) for k in ("store_id", "artifact_id", "version_id")}
        validate("content_ref", ref)
        if (version.get("sha256") != entry["sha256"] or version.get("size") != entry["size"] or
                version.get("mime") != entry["mime"] or version.get("source") != source or
                version["artifact_id"] != upload["artifact_id"] or version["version_id"] != upload["version_id"]):
            raise Failure("archive_integrity_failed", 502)
        return {"name": entry["name"], "content_ref": ref, "sha256": entry["sha256"],
                "size": entry["size"], "mime": entry["mime"], "source": source}
