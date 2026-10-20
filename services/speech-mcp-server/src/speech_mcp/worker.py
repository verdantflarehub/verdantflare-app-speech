import asyncio
import contextlib
import logging
import threading
import time

from . import __version__
from .clients import MAX_OUTPUT
from .domain import Failure, canonical, digest, new_id, validate
from .gpu_lease import GpuCoordinator, LeaseBusy

log = logging.getLogger("speech_worker")


class Worker:
    def __init__(self, service):
        self.service, self.store, self.peers = service, service.store, service.peers
        self.gpu = getattr(service, "gpu", None) or GpuCoordinator.from_environment()
        if self.gpu.enabled:
            threading.Thread(target=self._idle, name="speech-gpu-idle", daemon=True).start()

    def _idle(self):
        while True:
            time.sleep(30)
            try:
                self.gpu.maybe_scale_down()
            except Exception:
                log.error("speech_gpu_idle_scale_failed")

    async def renew(self, row):
        while True:
            await asyncio.sleep(20)
            await asyncio.to_thread(self.store.renew, row)
            if row["state"].get("gpu_held") and self.gpu.enabled:
                await asyncio.to_thread(self.gpu.renew)

    async def run(self):
        while True:
            try:
                row = await asyncio.to_thread(self.store.claim)
                if row:
                    await self.process(row)
                else:
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                raise
            except Exception:
                # Exception strings can contain database credentials or task text.
                log.error("speech_worker_dependency_failure")
                await asyncio.sleep(5)

    async def process(self, row):
        heartbeat = asyncio.create_task(self.renew(row))
        operation = asyncio.create_task(self.step(row))
        try:
            done, _ = await asyncio.wait([heartbeat, operation], return_when=asyncio.FIRST_COMPLETED)
            if heartbeat in done:
                await heartbeat
            await operation
        except Failure as error:
            if error.code != "lease_lost":
                await asyncio.to_thread(self.store.save, row,
                                        status="running" if error.retry else "failed", error=error.code, delay=15)
            if not error.retry:
                await self._release_gpu(row)
        finally:
            for task in (operation, heartbeat):
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Failure):
                    await task

    async def save(self, row, **kwargs):
        await asyncio.to_thread(self.store.save, row, **kwargs)

    async def step(self, row):
        p, req, state = row["principal"], row["request"], row["state"]
        pid, phase = row["project_id"], row["phase"]
        await self.peers.open(p, pid, write=True)
        if phase in ("reference", "submit", "inference"):
            await self._ensure_gpu(row)
        if phase == "reference":
            profile = state["profile"]
            caps = await self.peers.runtime.json(p, "GET", "/v1/capabilities", project_id=pid)
            if caps.get("model_manifest_sha256") != profile["model_manifest_sha256"]:
                raise Failure("model_version_changed", 409)
            if caps.get("ready") is not True:
                raise Failure("model_not_ready", 503, True)
            state["capabilities"] = caps
            data, _ = await self.peers.content(p, profile["reference"]["content_ref"], pid, req["revision_id"], 2 << 20)
            if digest(data) != profile["reference"]["sha256"]:
                raise Failure("reference_integrity_failed")
            await self.peers.runtime.json(p, "PUT", "/v1/references/" + digest(data), data=data, mime="audio/wav", project_id=pid)
            state["runtime_inputs"] = {"idempotency_key": "speech-" + row["task_id"], "text": req["text"],
                                       "language": req["language"], "reference_sha256": digest(data)}
            if "emotion_vector" in req:
                state["runtime_inputs"]["emotion_vector"] = req["emotion_vector"]
            await self.save(row, phase="submit")
        elif phase == "submit":
            caps = await self.peers.runtime.json(p, "GET", "/v1/capabilities", project_id=pid)
            if caps.get("model_manifest_sha256") != state["profile"]["model_manifest_sha256"]:
                raise Failure("model_version_changed", 409)
            task = await self.peers.runtime.json(p, "POST", "/v1/tasks", state["runtime_inputs"], project_id=pid)
            validate("id", task.get("task_id"))
            state["runtime_task_id"] = task["task_id"]
            await self.save(row, phase="inference", delay=2)
        elif phase == "inference":
            task = await self.peers.runtime.json(p, "GET", "/v1/tasks/" + state["runtime_task_id"], project_id=pid)
            if task.get("task_id") != state["runtime_task_id"] or task.get("model_manifest_sha256") != state["profile"]["model_manifest_sha256"]:
                raise Failure("runtime_identity_mismatch", 502)
            if task["status"] == "failed":
                code = task.get("error", {}).get("code")
                raise Failure(code if code in ("recovery_required", "model_version_changed") else "inference_failed", 409)
            if task["status"] != "succeeded":
                await self.save(row, delay=2)
                return
            entries = task.get("outputs", [])
            if len(entries) != 3 or {x.get("name") for x in entries} != {"original.wav", "editing.wav", "report.json"}:
                raise Failure("invalid_runtime_outputs", 502)
            for e in entries:
                validate("hash", e.get("sha256"))
                if type(e.get("size")) is not int or not 0 < e["size"] <= MAX_OUTPUT or e.get("mime") != ("application/json" if e["name"] == "report.json" else "audio/wav"):
                    raise Failure("invalid_runtime_outputs", 502)
                e["write_id"] = new_id()
            # This immutable journal survives all retries and documents actual inputs.
            state["generation"] = {"schema_version": 1, "service_version": __version__, "task_id": row["task_id"],
                                   "runtime_task_id": state["runtime_task_id"], "request": req,
                                   "voice_profile": state["profile"], "profile_sha256": state["profile_sha256"],
                                   "runtime_capabilities": state["capabilities"], "runtime_outputs": entries,
                                   "candidate_state": "candidate", "listening_review": "pending"}
            raw = canonical(state["generation"]).encode()
            entries.append({"name": "generation.json", "sha256": digest(raw), "size": len(raw),
                            "mime": "application/json", "write_id": new_id()})
            # Copy avoids a recursive reference through generation.runtime_outputs.
            state["generation"]["runtime_outputs"] = [dict(e) for e in entries[:3]]
            state["archive_entries"] = entries
            await self.save(row, phase="archive")
        elif phase == "archive":
            complete = {x["name"] for x in state["outputs"]}
            entry = next((x for x in state["archive_entries"] if x["name"] not in complete), None)
            if entry is None:
                await self._release_gpu(row)
                await self.save(row, phase="complete", status="succeeded")
                return
            if entry["name"] == "generation.json":
                data = canonical(state["generation"]).encode()
            else:
                path = "/v1/tasks/" + state["runtime_task_id"] + "/outputs/" + entry["name"]
                data = await self.peers.runtime.request(p, "GET", path, project_id=pid, limit=MAX_OUTPUT)
            if len(data) != entry["size"] or digest(data) != entry["sha256"]:
                raise Failure("output_integrity_failed", 502)
            result = await self.peers.archive(p, row, entry, data)
            state["outputs"].append(result)
            await self.save(row)
        else:
            raise Failure("invalid_task_phase", 500)

    async def _ensure_gpu(self, row):
        if not self.gpu.enabled:
            return
        if row["state"].get("gpu_held"):
            await asyncio.to_thread(self.gpu.renew)
            return
        try:
            await asyncio.to_thread(self.gpu.acquire_for_queue)
        except LeaseBusy as error:
            raise Failure("waiting_gpu_lease", 503, True) from error
        row["state"]["gpu_held"] = True
        await asyncio.to_thread(self.gpu.scale_to, 1)
        deadline = time.monotonic() + self.gpu.ready_seconds
        while time.monotonic() < deadline:
            try:
                health = await self.peers.runtime.json(row["principal"], "GET", "/health", project_id=row["project_id"])
                if health.get("ready") is True:
                    return
            except Failure:
                pass
            await asyncio.sleep(2)
        raise Failure("model_not_ready", 503, True)

    async def _release_gpu(self, row):
        if not self.gpu.enabled or not row["state"].pop("gpu_held", None):
            return
        await asyncio.to_thread(self.gpu.release)
