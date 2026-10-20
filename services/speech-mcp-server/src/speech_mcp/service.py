import asyncio
import re

from . import __version__
from .domain import Failure, canonical, digest, public, validate
from .gpu_lease import GpuCoordinator

VOICE_PATH = re.compile(r"^speech/voices/[a-z0-9][a-z0-9_-]{0,63}/profile\.json$")


class Service:
    def __init__(self, store, peers, model_hash, gpu=None):
        validate("hash", model_hash)
        self.store, self.peers, self.model_hash = store, peers, model_hash
        self.gpu = gpu if gpu is not None else GpuCoordinator.from_environment()

    async def call(self, name, args, p):
        action = name.removeprefix("speech.")
        if action not in ("capabilities", "voices", "create", "status", "result") or name != "speech." + action:
            raise Failure("tool_not_found", 404)
        validate(action, args)
        if action == "capabilities":
            ready = False
            reason = "connect_failed"
            try:
                health = await self.peers.runtime.json(p, "GET", "/health")
                ready = health.get("ready") is True
                reason = "ok" if ready else "health_not_ready"
            except Failure:
                reason = self._runtime_reason()
            return {"service_version": __version__, "engine": "indextts2", "ready": ready,
                    "reason": reason,
                    "wake_on_task": reason in {"scaled_to_zero", "waiting_gpu_lease"},
                    "layers": {"tool_registered": True, "orchestration_alive": self.gpu.api_ok(),
                               "backend_ready": ready},
                    "configured_model_manifest_sha256": self.model_hash,
                    "languages": ["zh-CN"], "max_text_characters": 500, "reference_audio_required": True,
                    "reference_seconds": {"min": 1, "max": 15}, "reference_max_bytes": 2 << 20,
                    "emotion_vector_order": ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"],
                    "emotion_vector_max_sum": 0.8, "exact_duration": False, "speed_control": False,
                    "nonverbal_performance": False, "rvc": False, "max_concurrent_inference": 1,
                    "outputs": ["original.wav", "editing.wav", "report.json", "generation.json"],
                    "quality_review": "human_listening_required"}
        pid = args["project_id"]
        opened = await self.peers.open(p, pid, args.get("revision_id"), write=action == "create")
        if action == "voices":
            files = [f for f in self.peers.files(opened) if VOICE_PATH.fullmatch(f["path"])]
            if len(files) > 64:
                raise Failure("too_many_voice_profiles")
            voices = []
            for file in files:
                try:
                    profile, sha = await self.peers.profile(p, opened, pid, args["revision_id"], file["content_ref"])
                    voices.append({"path": file["path"], "voice_profile": file["content_ref"], "profile_sha256": sha,
                                   **profile, "compatible": profile["model_manifest_sha256"] == self.model_hash})
                except Failure as error:
                    if error.retry:
                        raise
                    voices.append({"path": file["path"], "error": error.code})
            return {"project_id": pid, "revision_id": args["revision_id"], "voices": voices}
        if action in ("status", "result"):
            row = await asyncio.to_thread(self.store.get, p, pid, args["task_id"])
            if action == "result" and row["status"] != "succeeded":
                raise Failure("result_not_ready", 409)
            return public(row)
        prior = await asyncio.to_thread(self.store.existing, p, args)
        if prior:
            return public(prior)
        if args.get("script_ref"):
            self.peers.contains(opened, args["script_ref"])
            await self.peers.content(p, args["script_ref"], pid, args["revision_id"], 1 << 20)
        profile, sha = await self.peers.profile(p, opened, pid, args["revision_id"], args["voice_profile"])
        if profile["model_manifest_sha256"] != self.model_hash:
            raise Failure("model_version_changed", 409)
        # Validate the actual immutable reference before accepting durable work.
        audio, meta = await self.peers.content(p, profile["reference"]["content_ref"], pid, args["revision_id"], 2 << 20)
        if digest(audio) != profile["reference"]["sha256"] or meta["mime"] not in ("audio/wav", "audio/x-wav"):
            raise Failure("reference_integrity_failed")
        # A managed runtime may be scaled to zero. Persist validated work first;
        # the worker owns its lease, startup and actual model verification.
        # None is intentional: configuration is not an observed capability report.
        caps = None
        if not self.gpu.enabled:
            caps = await self.peers.runtime.json(p, "GET", "/v1/capabilities", project_id=pid)
            if caps.get("model_manifest_sha256") != self.model_hash:
                raise Failure("model_version_changed", 409)
            if not caps.get("ready"):
                raise Failure("model_not_ready", 503, True)
        state = {"profile": profile, "profile_sha256": sha, "capabilities": caps, "outputs": []}
        return public(await asyncio.to_thread(self.store.create, p, args, state))

    def _runtime_reason(self):
        if not self.gpu.enabled:
            return "connect_failed"
        try:
            if self.gpu.peer_holder():
                return "waiting_gpu_lease"
            if self.gpu.replicas() == 0:
                return "scaled_to_zero"
        except Exception:
            return "connect_failed"
        return "connect_failed"
