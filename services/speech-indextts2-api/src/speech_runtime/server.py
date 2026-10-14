import hashlib
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

from .audio import MAX_REFERENCE_BYTES
from .domain import Failure, Scope, strict_json, validate_create
from .store import Store
from .worker import IndexEngine, Worker


def create_app(root=None, token=None, model_root=None, engine=None, model_hash=None, start_worker=True):
    root = root or os.environ.get("SPEECH_TASK_ROOT", "/data/runtime")
    model_root = Path(model_root or os.environ.get("SPEECH_MODEL_ROOT", "/models"))
    token = token or os.environ.get("SPEECH_RUNTIME_TOKEN", "")
    if len(token) < 32 or any(c.isspace() for c in token):
        raise RuntimeError("runtime_credential_required")
    if model_hash is None:
        model_hash = hashlib.sha256((model_root / "manifest.lock.json").read_bytes()).hexdigest()
    store = Store(root, model_hash)
    worker = Worker(store, engine or IndexEngine(model_root))

    @asynccontextmanager
    async def lifespan(app):
        if start_worker:
            worker.start()
        yield
        if worker.shutdown():
            store.close()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.store, app.state.worker = store, worker

    @app.exception_handler(Failure)
    async def failure(request, error):
        return JSONResponse({"error": {"code": error.code, "request_id": request.headers.get("x-request-id", "")}}, status_code=error.status)

    @app.middleware("http")
    async def boundary(request, call_next):
        if request.url.path not in ("/live", "/health"):
            supplied = request.headers.getlist("authorization")
            if len(supplied) != 1 or not secrets.compare_digest(supplied[0].encode(), ("Bearer " + token).encode()):
                return JSONResponse({"error": {"code": "unauthorized"}}, status_code=401)
            if request.url.query or request.headers.get("content-encoding") or request.headers.get("transfer-encoding"):
                return JSONResponse({"error": {"code": "invalid_transport"}}, status_code=400)
            try:
                names = ("x-organization-id", "x-user-id", "x-project-id", "x-request-id")
                if any(len(request.headers.getlist(name)) != 1 for name in names):
                    raise Failure("invalid_scope", 403)
                if not 1 <= len(request.headers["x-request-id"]) <= 128:
                    raise Failure("invalid_request_id", 400)
                request.state.scope = Scope(*(request.headers[x] for x in names[:3]))
            except Failure as e:
                return JSONResponse({"error": {"code": e.code}}, status_code=e.status)
        result = await call_next(request)
        result.headers["Cache-Control"] = "no-store"
        result.headers["X-Content-Type-Options"] = "nosniff"
        return result

    async def body(request, maximum, content_type):
        if request.headers.get("content-type", "").split(";", 1)[0].lower() != content_type:
            raise Failure("unsupported_media_type", 415)
        lengths = request.headers.getlist("content-length")
        if len(lengths) != 1 or not lengths[0].isdigit() or not 0 < int(lengths[0]) <= maximum:
            raise Failure("body_size_invalid", 413)
        data = bytearray()
        async for chunk in request.stream():
            data.extend(chunk)
            if len(data) > maximum:
                raise Failure("body_too_large", 413)
        if len(data) != int(lengths[0]):
            raise Failure("body_size_mismatch")
        return bytes(data)

    @app.get("/live")
    def live():
        return {"status": "alive"}

    @app.get("/health")
    def health():
        return JSONResponse({"ready": worker.ready, "reason": worker.error or (None if worker.ready else "model_loading")}, status_code=200 if worker.ready else 503)

    @app.get("/v1/capabilities")
    def capabilities():
        return {"engine": "indextts2", "version": "0.1.3", "model_manifest_sha256": model_hash, "ready": worker.ready,
                "languages": ["zh-CN"], "reference_audio_required": True, "reference_max_seconds": 15,
                "emotion_vector_order": ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"],
                "exact_duration": False, "nonverbal_performance": False, "rvc": False, "max_concurrent_inference": 1}

    @app.put("/v1/references/{checksum}")
    async def reference(checksum: str, request: Request):
        data = await body(request, MAX_REFERENCE_BYTES, "audio/wav")
        return store.put_reference(request.state.scope, checksum, data)

    @app.post("/v1/tasks")
    async def create(request: Request):
        inputs = validate_create(strict_json(await body(request, 16384, "application/json")))
        # Existing idempotent results are available while the model reloads.
        with store.mutex:
            prior = store.db.execute("SELECT 1 FROM tasks WHERE scope=? AND idem=?", (request.state.scope.key, inputs["idempotency_key"])).fetchone()
        if not worker.ready and not prior:
            raise Failure("model_not_ready", 503)
        result = store.create(request.state.scope, inputs)
        worker.wake.set()
        return JSONResponse(result, status_code=202)

    @app.get("/v1/tasks/{task_id}")
    def status(task_id: str, request: Request):
        return store.get(request.state.scope, task_id)

    @app.get("/v1/tasks/{task_id}/outputs/{name}")
    def output(task_id: str, name: str, request: Request):
        task = store.get(request.state.scope, task_id)
        if task["status"] != "succeeded":
            raise Failure("result_not_ready", 409)
        meta = next((item for item in task["outputs"] if item["name"] == name), None)
        if meta is None:
            raise Failure("output_not_found", 404)
        return FileResponse(store.output_dir(task_id) / name, media_type=meta["mime"], filename=name,
                            headers={"X-Content-SHA256": meta["sha256"]})

    return app
