import asyncio
import contextlib
import secrets
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from . import __version__
from .domain import Failure, TOOLS, canonical, principal, strict_json
from .worker import Worker


def applications(service, mcp_token, registry=None, start_worker=True):
    if len(mcp_token) < 32 or any(c.isspace() for c in mcp_token):
        raise RuntimeError("invalid_service_credential")

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

    def install(app, credential):
        @app.middleware("http")
        async def boundary(request, call_next):
            if request.url.path != "/health":
                values = request.headers.getlist("authorization")
                if len(values) != 1 or not secrets.compare_digest(values[0].encode(), ("Bearer " + credential).encode()):
                    return JSONResponse({"error": "unauthorized"}, status_code=401)
            if request.url.query or request.headers.get("content-encoding") or request.scope.get("raw_path", b"").decode() != request.url.path:
                return JSONResponse({"error": "invalid_transport"}, status_code=400)
            result = await call_next(request)
            result.headers["Cache-Control"] = "no-store"
            result.headers["X-Content-Type-Options"] = "nosniff"
            return result

        @app.exception_handler(Failure)
        async def failure(request, error):
            return JSONResponse({"error": {"code": error.code}}, status_code=error.status)

    install(mcp, mcp_token)

    async def read(request):
        if request.headers.get("content-type", "").split(";", 1)[0] != "application/json":
            raise Failure("invalid_transport")
        data = bytearray()
        async for chunk in request.stream():
            data.extend(chunk)
            if len(data) > 65536:
                raise Failure("body_too_large", 413)
        return strict_json(data)

    @mcp.get("/health")
    async def health():
        try:
            await asyncio.to_thread(service.store.healthy)
        except Exception:
            return JSONResponse({"ready": False}, status_code=503)
        return {"ready": True, "version": __version__}

    @mcp.post("/mcp")
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
