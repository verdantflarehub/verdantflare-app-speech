import asyncio
import contextlib
import inspect
import logging
import secrets
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
import starlette.routing as _starlette_routing
from starlette.middleware.base import BaseHTTPMiddleware

if "on_startup" not in inspect.signature(_starlette_routing.Router.__init__).parameters:
    _orig_router_init = _starlette_routing.Router.__init__

    def _compat_router_init(self, *args, on_startup=None, on_shutdown=None, **kwargs):
        return _orig_router_init(self, *args, **kwargs)

    _starlette_routing.Router.__init__ = _compat_router_init

if not hasattr(FastAPI, "max_body_size"):
    FastAPI.max_body_size = None

from . import __version__
from .content_rating import ContentRatingError, include_nsfw_from_headers
from .dashboard import DashboardAPI
from .dashboard_assets import build_bundle, get_asset
from .domain import (
    SPEECH_TOOLS_SCHEMA,
    TOOLS,
     UUID7,
    Failure,
    TaskNotFound,
    canonical,
    new_id,
    principal,
    strict_json,
)
from .worker import Worker


class BearerAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, credential: str = ""):
        super().__init__(app)
        self.credential = credential


def applications(service, mcp_token, registry=None, start_worker=True):
    if len(mcp_token) < 32 or any(c.isspace() for c in mcp_token):
        raise RuntimeError("invalid_service_credential")
    dash = DashboardAPI(service)

    @asynccontextmanager
    async def lifespan(app):
        tasks = []
        if start_worker:
            tasks.append(asyncio.create_task(Worker(service).run()))
        if registry:
            tasks.append(asyncio.create_task(registry.run()))
        yield
        for task in tasks:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        if registry:
            await registry.close()
        await service.peers.close()

    mcp = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    mcp.state.service = service
    mcp.state.ratings = service.active_ratings()

    def install(app, credential):
        @app.middleware("http")
        async def boundary(request, call_next):
            path = request.url.path
            is_health = path in ("/health", "/speech/health")
            is_artifact = path.startswith("/artifacts/") or path.startswith("/speech/artifacts/")
            if not is_health:
                values = request.headers.getlist("authorization")
                if is_artifact and not values:
                    pass
                elif len(values) != 1 or not secrets.compare_digest(values[0].encode(), ("Bearer " + credential).encode()):
                    return JSONResponse({"code": "UNAUTHORIZED", "error": "unauthorized"}, status_code=401)
            raw_path = request.scope.get("raw_path", b"").split(b"?", 1)[0].decode()
            disallow_query = path in ("/mcp", "/speech/mcp") and bool(request.url.query)
            if disallow_query or request.headers.get("content-encoding") or (raw_path and raw_path != path):
                return JSONResponse({"code": "INVALID_TRANSPORT", "error": "invalid_transport"}, status_code=400)
            result = await call_next(request)
            result.headers["Cache-Control"] = "no-store"
            result.headers["X-Content-Type-Options"] = "nosniff"
            return result

        @app.exception_handler(Failure)
        async def failure(request, error):
            return JSONResponse(
                {"code": error.code.upper(), "error": {"code": error.code}},
                status_code=error.status,
            )

    install(mcp, mcp_token)

    def request_principal(request: Request, *, allow_fallback: bool = False):
        names = {"organization_id": "x-organization-id", "subject_id": "x-user-id", "request_id": "x-request-id"}
        present = {k: request.headers.getlist(v) for k, v in names.items()}
        if all(len(vals) == 1 for vals in present.values()):
            return principal({k: vals[0] for k, vals in present.items()})
        if allow_fallback:
            default_p = service.default_principal()
            if default_p and all(len(vals) == 0 for vals in present.values()):
                return default_p
            if default_p and len(present["subject_id"]) == 1 and UUID7.fullmatch(present["subject_id"][0]):
                return {
                    "organization_id": present["organization_id"][0] if len(present["organization_id"]) == 1 else default_p["organization_id"],
                    "subject_id": present["subject_id"][0],
                    "request_id": present["request_id"][0] if len(present["request_id"]) == 1 else default_p.get("request_id", new_id()),
                }
        raise Failure("invalid_identity", 403)

    def try_principal(request: Request):
        try:
            return request_principal(request, allow_fallback=True)
        except Failure:
            return None

    async def read(request, *, allow_empty: bool = False):
        raw = bytearray()
        async for chunk in request.stream():
            raw.extend(chunk)
            if len(raw) > 65536:
                raise Failure("body_too_large", 413)
        if not raw and allow_empty:
            return {}
        if request.headers.get("content-type", "").split(";", 1)[0] != "application/json":
            raise Failure("invalid_transport")
        return strict_json(raw)

    @mcp.get("/health")
    @mcp.get("/speech/health")
    async def health():
        try:
            await asyncio.to_thread(service.store.healthy)
        except Exception:
            return JSONResponse({"ready": False}, status_code=503)
        return {"ready": True, "version": __version__}

    @mcp.get("/internal/dashboard-declaration")
    @mcp.get("/speech/internal/dashboard-declaration")
    async def dashboard_declaration(request: Request):
        request_principal(request)
        bundle = build_bundle()
        return Response(content=bundle["declaration_bytes"], media_type="application/json")

    @mcp.get("/internal/dashboard-index")
    @mcp.get("/speech/internal/dashboard-index")
    async def dashboard_index(request: Request):
        request_principal(request)
        bundle = build_bundle()
        return Response(content=bundle["index_bytes"], media_type="application/json")

    @mcp.get("/internal/dashboard/assets/{name:path}")
    @mcp.get("/speech/internal/dashboard/assets/{name:path}")
    async def dashboard_asset(name: str, request: Request):
        request_principal(request)
        data, mime = get_asset(name)
        return Response(content=data, media_type=mime)

    @mcp.get("/api/overview")
    @mcp.get("/speech/api/overview")
    @mcp.get("/api/dashboard")
    @mcp.get("/speech/api/dashboard")
    async def api_overview(request: Request):
        p = request_principal(request, allow_fallback=True)
        include_nsfw = include_nsfw_from_headers(request.headers)
        return await dash.overview(p, include_nsfw=include_nsfw)

    @mcp.get("/api/tasks")
    @mcp.get("/speech/api/tasks")
    async def api_tasks(request: Request):
        p = request_principal(request, allow_fallback=True)
        include_nsfw = include_nsfw_from_headers(request.headers)
        query = dict(request.query_params)
        return await dash.query_tasks(p, query, include_nsfw=include_nsfw)

    @mcp.post("/api/tasks/query")
    @mcp.post("/speech/api/tasks/query")
    async def api_tasks_query(request: Request):
        p = request_principal(request, allow_fallback=True)
        body = await read(request)
        include_nsfw = include_nsfw_from_headers(request.headers)
        return await dash.query_tasks(p, body, include_nsfw=include_nsfw)

    @mcp.get("/api/tasks/stats")
    @mcp.get("/speech/api/tasks/stats")
    async def api_tasks_stats(request: Request):
        p = request_principal(request, allow_fallback=True)
        include_nsfw = include_nsfw_from_headers(request.headers)
        project_id = request.query_params.get("project_id")
        return await dash.task_stats(p, project_id=project_id, include_nsfw=include_nsfw)

    @mcp.get("/api/tasks/{task_id}")
    @mcp.get("/speech/api/tasks/{task_id}")
    async def api_task_detail(task_id: str, request: Request):
        p = request_principal(request, allow_fallback=True)
        include_nsfw = include_nsfw_from_headers(request.headers)
        return await dash.task_detail(p, task_id, include_nsfw=include_nsfw)

    @mcp.post("/api/tasks/{task_id}/title")
    @mcp.post("/speech/api/tasks/{task_id}/title")
    async def api_task_title(task_id: str, request: Request):
        p = request_principal(request, allow_fallback=True)
        body = await read(request)
        if not isinstance(body, dict):
            raise Failure("invalid_arguments", 400)
        return await dash.rename_task(p, task_id, body.get("title"))

    @mcp.post("/api/tasks/{task_id}/delete")
    @mcp.post("/speech/api/tasks/{task_id}/delete")
    @mcp.delete("/api/tasks/{task_id}")
    @mcp.delete("/speech/api/tasks/{task_id}")
    async def api_task_delete(task_id: str, request: Request):
        p = request_principal(request, allow_fallback=True)
        return await dash.delete_task(p, task_id)

    @mcp.post("/api/tasks/{task_id}/content-rating")
    @mcp.post("/speech/api/tasks/{task_id}/content-rating")
    async def api_task_content_rating(task_id: str, request: Request):
        p = request_principal(request, allow_fallback=True)
        body = await read(request, allow_empty=True)
        if not isinstance(body, dict):
            raise Failure("invalid_arguments", 400)
        action = body.get("action")
        reason = body.get("reason") or body.get("note")
        actor_id = (
            request.headers.get("x-user-id", "").strip()
            or str(body.get("actor_user_id") or body.get("user_id") or body.get("actor_id") or "").strip()
            or ((p or {}).get("subject_id") or "").strip()
        )
        try:
            return await dash.update_content_rating(
                p,
                task_id,
                action,
                reason=reason,
                actor_user_id=actor_id,
            )
        except ContentRatingError as exc:
            return JSONResponse(
                {"code": exc.error_code, "error": exc.error_code, "error_code": exc.error_code, "message": exc.message},
                status_code=exc.status_code,
            )

    @mcp.get("/api/tasks/{task_id}/audio/{kind}")
    @mcp.get("/speech/api/tasks/{task_id}/audio/{kind}")
    async def api_task_audio(task_id: str, kind: str, request: Request):
        p = request_principal(request, allow_fallback=True)
        include_nsfw = include_nsfw_from_headers(request.headers)
        return await dash.task_audio(p, task_id, kind, include_nsfw=include_nsfw)

    @mcp.post("/api/tasks/{task_id}/review")
    @mcp.post("/speech/api/tasks/{task_id}/review")
    async def api_task_review(task_id: str, request: Request):
        p = request_principal(request, allow_fallback=True)
        body = await read(request)
        return await dash.write_review(p, task_id, body)

    @mcp.get("/artifacts/{artifact_id}/content")
    @mcp.get("/speech/artifacts/{artifact_id}/content")
    async def artifact_content(artifact_id: str, request: Request):
        include_nsfw = include_nsfw_from_headers(request.headers)
        p = try_principal(request)
        row = await asyncio.to_thread(service.find_row_by_version, artifact_id, p=p)
        if row is None and p is not None:
            row = await asyncio.to_thread(service.find_row_by_version, artifact_id, p=None)
        store = service.active_ratings()
        if not include_nsfw:
            if store.is_nsfw(artifact_id, task_id=row["task_id"] if row else None) or (row and service.is_row_nsfw(row)):
                return JSONResponse({"code": "NOT_FOUND", "error": "artifact_not_found"}, status_code=404)
        if row is None:
            return JSONResponse({"code": "NOT_FOUND", "error": "artifact_not_found"}, status_code=404)
        outputs = (row.get("state") or {}).get("outputs") or []
        matched_entry = None
        for entry in outputs:
            if not isinstance(entry, dict):
                continue
            cref = entry.get("content_ref")
            if isinstance(cref, dict) and artifact_id in (cref.get("version_id"), cref.get("artifact_id")):
                matched_entry = entry
                break
            if artifact_id in (entry.get("version_id"), entry.get("artifact_id")):
                matched_entry = entry
                break
        if not matched_entry or not matched_entry.get("content_ref"):
            return JSONResponse({"code": "NOT_FOUND", "error": "artifact_not_found"}, status_code=404)
        reader = getattr(service.peers, "task_output_content", None)
        if not reader:
            logging.getLogger(__name__).error("task_output_content not configured on service.peers for artifact %s", artifact_id)
            return JSONResponse({"code": "UPSTREAM_ERROR", "error": "artifact_read_failed"}, status_code=502)
        try:
            eff_p = p or row.get("principal")
            res = await reader(eff_p, matched_entry["content_ref"], row["project_id"], 16 << 20)
            if isinstance(res, tuple) and len(res) == 2 and isinstance(res[0], (bytes, bytearray)):
                return Response(content=bytes(res[0]), media_type=matched_entry.get("mime", "audio/wav"))
            logging.getLogger(__name__).warning("invalid task_output_content payload for artifact %s", artifact_id)
            return JSONResponse({"code": "UPSTREAM_ERROR", "error": "artifact_read_failed"}, status_code=502)
        except Failure as exc:
            logging.getLogger(__name__).warning("failed reading artifact %s: %s (status=%s)", artifact_id, exc.code, exc.status)
            if exc.status in (403, 404):
                return JSONResponse({"code": exc.code.upper(), "error": exc.code}, status_code=exc.status)
            return JSONResponse({"code": "UPSTREAM_ERROR", "error": "artifact_read_failed"}, status_code=502)
        except Exception:
            logging.getLogger(__name__).exception("unexpected error reading artifact %s", artifact_id)
            return JSONResponse({"code": "UPSTREAM_ERROR", "error": "artifact_read_failed"}, status_code=502)

    @mcp.post("/internal/content-ratings/{version_id}/clear")
    @mcp.post("/speech/internal/content-ratings/{version_id}/clear")
    async def internal_clear_rating(version_id: str, request: Request):
        body = await read(request, allow_empty=True)
        if not isinstance(body, dict):
            raise Failure("invalid_arguments", 400)
        project_id = body.get("project_id") or request.headers.get("x-project-id") or None
        note = body.get("note") or body.get("reason")
        actor_id = (
            request.headers.get("x-user-id", "").strip()
            or str(body.get("actor_user_id") or body.get("user_id") or body.get("actor_id") or "").strip()
        )
        p = try_principal(request)
        try:
            rec = await asyncio.to_thread(
                service.clear_content_rating,
                version_id,
                actor_user_id=actor_id,
                note=note,
                project_id=project_id,
                p=p,
            )
            return JSONResponse(rec, status_code=200)
        except (TaskNotFound, Failure) as exc:
            if getattr(exc, "status", 404) == 404:
                return JSONResponse({"code": "NOT_FOUND", "error": "not_found", "error_code": "NOT_FOUND"}, status_code=404)
            return JSONResponse({"code": exc.code.upper(), "error": exc.code}, status_code=exc.status)
        except ContentRatingError as exc:
            return JSONResponse(
                {"code": exc.error_code, "error": exc.error_code, "error_code": exc.error_code, "message": exc.message},
                status_code=exc.status_code,
            )

    @mcp.post("/internal/content-ratings/{version_id}/mark")
    @mcp.post("/speech/internal/content-ratings/{version_id}/mark")
    async def internal_mark_rating(version_id: str, request: Request):
        body = await read(request, allow_empty=True)
        if not isinstance(body, dict):
            raise Failure("invalid_arguments", 400)
        project_id = body.get("project_id") or request.headers.get("x-project-id") or None
        reason = body.get("reason") or body.get("note")
        actor_id = (
            request.headers.get("x-user-id", "").strip()
            or str(body.get("actor_user_id") or body.get("user_id") or body.get("actor_id") or "").strip()
        )
        p = try_principal(request)
        try:
            rec = await asyncio.to_thread(
                service.mark_content_rating,
                version_id,
                actor_user_id=actor_id,
                reason=reason,
                project_id=project_id,
                p=p,
            )
            return JSONResponse(rec, status_code=200)
        except (TaskNotFound, Failure) as exc:
            if getattr(exc, "status", 404) == 404:
                return JSONResponse({"code": "NOT_FOUND", "error": "not_found", "error_code": "NOT_FOUND"}, status_code=404)
            return JSONResponse({"code": exc.code.upper(), "error": exc.code}, status_code=exc.status)
        except ContentRatingError as exc:
            return JSONResponse(
                {"code": exc.error_code, "error": exc.error_code, "error_code": exc.error_code, "message": exc.message},
                status_code=exc.status_code,
            )

    @mcp.get("/api/models")
    @mcp.get("/speech/api/models")
    async def api_models(request: Request):
        p = request_principal(request, allow_fallback=True)
        return await dash.models(p)

    @mcp.get("/api/mcp/status")
    @mcp.get("/speech/api/mcp/status")
    async def api_mcp_status(request: Request):
        p = request_principal(request, allow_fallback=True)
        return await dash.mcp_status(p)

    @mcp.post("/mcp")
    @mcp.post("/speech/mcp")
    async def rpc(request: Request):
        names = {"organization_id": "x-organization-id", "subject_id": "x-user-id", "request_id": "x-request-id"}
        if any(len(request.headers.getlist(v)) != 1 for v in names.values()):
            raise Failure("invalid_identity", 403)
        p = principal({k: request.headers[v] for k, v in names.items()})
        body = await read(request)
        if (not isinstance(body, dict) or set(body) - {"jsonrpc", "id", "method", "params"} or
                body.get("jsonrpc") != "2.0" or body.get("id") != p["request_id"]):
            raise Failure("invalid_rpc")
        method = body.get("method")
        if method == "tools/list":
            result = {"tools": TOOLS}
        elif method == "initialize":
            result = {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}},
                      "serverInfo": {"name": "speech-mcp-server", "version": __version__}}
        elif method == "tools/call":
            params = body.get("params")
            if not isinstance(params, dict) or set(params) != {"name", "arguments"} or not isinstance(params["name"], str) or not isinstance(params["arguments"], dict):
                raise Failure("invalid_rpc")
            args = params["arguments"]
            project_headers = request.headers.getlist("x-project-id")
            if ((args.get("project_id") and project_headers != [args["project_id"]]) or
                    (not args.get("project_id") and project_headers)):
                raise Failure("project_scope_mismatch", 400)
            try:
                data = await service.call(params["name"], args, p)
                result = {"isError": False, "structuredContent": data, "content": [{"type": "text", "text": canonical(data)}]}
            except ContentRatingError as error:
                data = {"error": {"code": error.error_code.lower(), "error_code": error.error_code, "message": error.message, "retryable": False}}
                result = {"isError": True, "structuredContent": data, "content": [{"type": "text", "text": canonical(data)}]}
            except Failure as error:
                data = {"error": {"code": error.code, "retryable": error.retry}}
                result = {"isError": True, "structuredContent": data, "content": [{"type": "text", "text": canonical(data)}]}
            except Exception:
                data = {"error": {"code": "service_unavailable", "retryable": True}}
                result = {"isError": True, "structuredContent": data, "content": [{"type": "text", "text": canonical(data)}]}
        else:
            raise Failure("method_not_found", 404)
        return {"jsonrpc": "2.0", "id": body["id"], "result": result}

    return mcp
