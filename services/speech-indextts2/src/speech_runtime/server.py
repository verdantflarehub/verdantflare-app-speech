import hashlib
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

from .asr import ASR_MODEL_ID
from .audio import MAX_REFERENCE_BYTES
from .domain import Failure, Scope, canonical, strict_json, validate_create
from .gpu_guard import FencePersistenceError, LeaseFenceGuard, StaleFenceError, read_runtime_identity
from .store import Store
from .worker import IndexEngine, Worker


def create_app(root=None, token=None, model_root=None, engine=None, model_hash=None, start_worker=True):
    root = root or os.environ.get("SPEECH_TASK_ROOT", "/data/runtime")
    model_root = Path(model_root or os.environ.get("SPEECH_MODEL_ROOT", "/models"))
    token = token or os.environ.get("SPEECH_RUNTIME_TOKEN", "")
    if len(token) < 32 or any(c.isspace() for c in token):
        raise RuntimeError("runtime_credential_required")
    sched_token = os.environ.get("STATION_GPU_SCHEDULER_TOKEN", "").strip()
    if model_hash is None:
        model_hash = hashlib.sha256((model_root / "manifest.lock.json").read_bytes()).hexdigest()
    store = Store(root, model_hash)
    default_state_path = os.environ.get("VF_GPU_GUARD_STATE_PATH", "").strip() or str(Path(root) / "gpu_guard_state.json")
    fence = LeaseFenceGuard(state_path=default_state_path)
    worker = Worker(store, engine or IndexEngine(model_root), fence=fence)

    @asynccontextmanager
    async def lifespan(app):
        if start_worker:
            worker.start()
        yield
        if worker.shutdown():
            store.close()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.store, app.state.worker, app.state.fence = store, worker, fence

    @app.exception_handler(Failure)
    async def failure(request, error):
        return JSONResponse({"error": {"code": error.code, "request_id": request.headers.get("x-request-id", "")}}, status_code=error.status)

    def _valid_bearer(supplied_header: str, *, allow_scheduler: bool = False) -> bool:
        if secrets.compare_digest(supplied_header.encode(), ("Bearer " + token).encode()):
            return True
        if allow_scheduler and sched_token and secrets.compare_digest(supplied_header.encode(), ("Bearer " + sched_token).encode()):
            return True
        return False

    @app.middleware("http")
    async def boundary(request, call_next):
        is_internal_residency = (
            request.url.path.startswith("/internal/v1/gpu-residency/")
            or request.url.path == "/internal/v1/gpu-guard/reclaim"
        )
        if request.url.path not in ("/live", "/health"):
            supplied = request.headers.getlist("authorization")
            if len(supplied) != 1 or not _valid_bearer(supplied[0], allow_scheduler=is_internal_residency):
                return JSONResponse({"error": {"code": "unauthorized"}}, status_code=401)
            if request.url.query or request.headers.get("content-encoding") or request.headers.get("transfer-encoding"):
                return JSONResponse({"error": {"code": "invalid_transport"}}, status_code=400)
            if not is_internal_residency:
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
        active = store.active_count()
        residency = "S1_active" if active > 0 else ("S2_warm_idle" if worker.ready else "S3_cold")
        return JSONResponse(
            {
                "ready": worker.ready,
                "reason": worker.error or (None if worker.ready else "model_loading"),
                "load_seconds": getattr(worker.engine, "load_seconds", 0.0),
                "active_tasks": active,
                "residency_state": residency,
                "runtime_identity": read_runtime_identity(),
            },
            status_code=200 if worker.ready else 503,
        )

    @app.get("/v1/capabilities")
    def capabilities():
        asr = getattr(worker.engine, "asr", None)
        return {"engine": "indextts2", "version": "0.1.5", "model_manifest_sha256": model_hash, "ready": worker.ready,
                "load_seconds": getattr(worker.engine, "load_seconds", 0.0),
                "asr_model": ASR_MODEL_ID, "asr_ready": bool(getattr(asr, "ready", False)),
                "languages": ["zh-CN"], "reference_audio_required": True, "reference_max_seconds": 15,
                "emotion_inputs": ["emotion_vector", "emotion_text", "emotion_reference"],
                "emotion_vector_order": ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"],
                "exact_duration": False, "nonverbal_performance": False, "rvc": False, "max_concurrent_inference": 1}

    @app.put("/v1/references/{checksum}")
    async def reference(checksum: str, request: Request):
        data = await body(request, MAX_REFERENCE_BYTES, "audio/wav")
        return store.put_reference(request.state.scope, checksum, data)

    @app.post("/v1/tasks")
    async def create(request: Request):
        inputs = validate_create(strict_json(await body(request, 16384, "application/json")))
        req_digest = hashlib.sha256(canonical(inputs).encode()).hexdigest()
        key_id = f"{request.state.scope.key}:{inputs['idempotency_key']}"
        with store.mutex:
            try:
                fence.extract_and_validate(request.headers, key_id=key_id, request_digest=req_digest)
            except FencePersistenceError as exc:
                raise Failure("fence_state_corrupted", 503) from exc
            except PermissionError as exc:
                raise Failure("missing_or_invalid_gpu_grant", 403) from exc
            except StaleFenceError as exc:
                raise Failure("stale_fencing_token", 409) from exc
            except ValueError as exc:
                raise Failure("idempotency_conflict", 409) from exc
            try:
                # Existing idempotent results are available while the model reloads.
                prior = store.db.execute("SELECT 1 FROM tasks WHERE scope=? AND idem=?", (request.state.scope.key, inputs["idempotency_key"])).fetchone()
                if not worker.ready and not prior:
                    raise Failure("model_not_ready", 503)
                result = store.create(request.state.scope, inputs)
            except Exception:
                fence.abort_key(key_id)
                raise
            is_active = result.get("status") in ("pending", "running")
            fence.bind_task_id(key_id, result["task_id"], is_active=is_active)
        worker.wake.set()
        return JSONResponse(result, status_code=202)

    @app.get("/v1/tasks/{task_id}")
    def status(task_id: str, request: Request):
        with store.mutex:
            out = store.get(request.state.scope, task_id)
            if out.get("status") in ("succeeded", "failed"):
                active = store.active_count()
                try:
                    proof = fence.reclaim_and_seal(task_id=task_id, active_tasks=active, settled=active == 0)
                except FencePersistenceError as exc:
                    raise Failure("fence_state_corrupted", 503) from exc
                if proof is not None:
                    out = dict(out)
                    out["reclaim_proof"] = proof
        return out

    @app.post("/internal/v1/gpu-guard/reclaim")
    async def reclaim_guard(request: Request):
        payload = strict_json(await body(request, 4096, "application/json"))
        if not isinstance(payload, dict):
            raise Failure("invalid_request", 400)
        lease_id = str(payload.get("lease_id") or "").strip()
        task_id = str(payload.get("task_id") or "").strip() or None
        try:
            fencing_token = int(payload.get("fencing_token") or 0)
        except (TypeError, ValueError) as exc:
            raise Failure("invalid_request", 400) from exc
        if not lease_id and fencing_token <= 0 and not task_id:
            raise Failure("invalid_request", 400)
        with store.mutex:
            try:
                affected = fence.revoke_lease(lease_id, fencing_token)
                aborted = [tid for tid in affected if store.abort_pending_fence(tid)]
                fence.clear_aborted_tasks(aborted)
                active = store.active_count()
                proof = fence.reclaim_and_seal(
                    task_id=task_id,
                    lease_id=lease_id,
                    fencing_token=fencing_token,
                    active_tasks=active,
                    settled=active == 0,
                )
            except FencePersistenceError as exc:
                raise Failure("fence_state_corrupted", 503) from exc
        eff_active = int(proof["active_tasks"]) if isinstance(proof, dict) else active
        settled = bool(isinstance(proof, dict) and proof.get("reclaimed") is True)
        return JSONResponse(
            {
                "ok": settled,
                "active_tasks": eff_active,
                "settled": settled,
                "reclaim_proof": proof,
            },
            status_code=200 if settled else 409,
        )

    @app.post("/internal/v1/gpu-residency/revoke-lease")
    async def revoke_lease(request: Request):
        payload = strict_json(await body(request, 4096, "application/json"))
        if not isinstance(payload, dict):
            raise Failure("invalid_request", 400)
        lease_id = str(payload.get("lease_id") or "").strip()
        try:
            fencing_token = int(payload.get("fencing_token") or 0)
        except (TypeError, ValueError) as exc:
            raise Failure("invalid_request", 400) from exc
        with store.mutex:
            try:
                affected = fence.revoke_lease(lease_id, fencing_token)
                aborted = [tid for tid in affected if store.abort_pending_fence(tid)]
                fence.clear_aborted_tasks(aborted)
                active = store.active_count()
                proof = fence.reclaim_and_seal(
                    lease_id=lease_id,
                    fencing_token=fencing_token,
                    active_tasks=active,
                    settled=active == 0,
                )
            except FencePersistenceError as exc:
                raise Failure("fence_state_corrupted", 503) from exc
        eff_active = int(proof["active_tasks"]) if isinstance(proof, dict) else active
        settled = bool(isinstance(proof, dict) and proof.get("reclaimed") is True)
        return {
            "ok": True,
            "affected_tasks": affected,
            "aborted_pending_tasks": aborted,
            "active_tasks": eff_active,
            "settled": settled,
            "reclaim_proof": proof,
        }

    @app.post("/internal/v1/gpu-residency/evict-vram")
    async def evict_vram(request: Request):
        payload = strict_json(await body(request, 4096, "application/json"))
        if not isinstance(payload, dict):
            raise Failure("invalid_request", 400)
        lease_id = str(payload.get("lease_id") or "").strip()
        try:
            fencing_token = int(payload.get("fencing_token") or 0)
        except (TypeError, ValueError):
            fencing_token = 0
        if not worker.evict_idle_vram():
            active = store.active_count()
            return JSONResponse(
                {"error": {"code": "active_inference_busy"}, "active_tasks": active, "settled": False},
                status_code=409,
            )
        with store.mutex:
            active = store.active_count()
            try:
                proof = fence.reclaim_and_seal(
                    lease_id=lease_id or "warm_evict",
                    fencing_token=max(1, fencing_token),
                    active_tasks=active,
                    settled=active == 0,
                )
            except FencePersistenceError as exc:
                raise Failure("fence_state_corrupted", 503) from exc
        eff_active = int(proof["active_tasks"]) if isinstance(proof, dict) else active
        settled = bool(isinstance(proof, dict) and proof.get("reclaimed") is True)
        return {
            "ok": True,
            "evicted": True,
            "active_tasks": eff_active,
            "settled": settled,
            "reclaim_proof": proof,
        }

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
