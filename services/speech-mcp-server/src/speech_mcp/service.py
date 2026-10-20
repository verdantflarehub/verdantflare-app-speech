import asyncio
import re

from . import __version__
from .domain import (
    DEFAULT_ADAPTER,
    Failure,
    build_text_record,
    compute_input_digest,
    digest,
    public,
    validate,
)
from .gpu_lease import GpuCoordinator

VOICE_PATH = re.compile(r"^speech/voices/[a-z0-9][a-z0-9_-]{0,63}/profile\.json$")


class Service:
    def __init__(self, store, peers, model_hash, gpu=None, adapter=None):
        validate("hash", model_hash)
        self.store, self.peers, self.model_hash = store, peers, model_hash
        self.gpu = gpu if gpu is not None else GpuCoordinator.from_environment()
        self.adapter = adapter if adapter is not None else DEFAULT_ADAPTER

    async def engine_runtime_info(self, p):
        ready = False
        reason = "connect_failed"
        info = None
        try:
            health = await self.peers.runtime.json(p, "GET", "/health")
            info = health if isinstance(health, dict) else {}
            ready = info.get("ready") is True
            reason = "ok" if ready else "health_not_ready"
        except Failure:
            reason = self._runtime_reason()
        cap = self.adapter.capability(info, model_manifest_sha256=self.model_hash)
        return {
            "capability": cap,
            "ready": ready,
            "reason": reason,
            "wake_on_task": reason in {"scaled_to_zero", "waiting_gpu_lease"},
            "layers": {
                "tool_registered": True,
                "orchestration_alive": self.gpu.api_ok() if hasattr(self.gpu, "api_ok") else True,
                "backend_ready": ready,
            },
        }

    async def call(self, name, args, p):
        action = name.removeprefix("speech.")
        if action not in ("capabilities", "voices", "create", "status", "result") or name != "speech." + action:
            raise Failure("tool_not_found", 404)
        validate(action, args)
        if action == "capabilities":
            rt = await self.engine_runtime_info(p)
            return validate(
                "capabilities_result",
                {"service_version": __version__, "engines": [rt["capability"]]},
            )
        pid = args["project_id"]
        opened = await self.peers.open(p, pid, args.get("revision_id"), write=action == "create")
        if action == "voices":
            voices = []
            for asset_ref in self.peers.asset_refs(opened):
                if self.peers.world is None:
                    break
                asset_id = asset_ref.get("asset_id")
                ver_id = asset_ref.get("asset_version_id") or asset_ref.get("version_id")
                if not asset_id or not ver_id:
                    continue
                try:
                    profile, voice_source, _ = await self.peers.world_voice(
                        p, opened, {"asset_id": asset_id, "version_id": ver_id},
                        engine=self.adapter.engine_id, model_manifest_sha256=None, check_boundary=False,
                    )
                    voices.append({
                        "voice_source": "world",
                        "world_voice": voice_source["world_voice"],
                        "reference_sha256": voice_source["reference_sha256"],
                        **profile,
                        "compatible": profile["model_manifest_sha256"] == self.model_hash,
                    })
                except Failure as error:
                    if error.retry:
                        raise
                    if error.code != "invalid_reference_voice":
                        voices.append({
                            "voice_source": "world",
                            "world_voice": {"asset_id": asset_id, "version_id": ver_id},
                            "error": error.code,
                        })
            files = [f for f in self.peers.files(opened) if VOICE_PATH.fullmatch(f["path"])]
            if len(files) > 64:
                raise Failure("too_many_voice_profiles")
            for file in files:
                try:
                    profile, sha = await self.peers.profile(p, opened, pid, args["revision_id"], file["content_ref"])
                    voices.append({
                        "voice_source": "project_candidate",
                        "path": file["path"],
                        "voice_profile": file["content_ref"],
                        "profile_sha256": sha,
                        **profile,
                        "compatible": profile["model_manifest_sha256"] == self.model_hash,
                    })
                except Failure as error:
                    if error.retry:
                        raise
                    voices.append({"voice_source": "project_candidate", "path": file["path"], "error": error.code})
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
        if args.get("world_voice"):
            profile, voice_source, _ = await self.peers.world_voice(
                p, opened, args["world_voice"],
                engine=self.adapter.engine_id, model_manifest_sha256=self.model_hash, check_boundary=True,
            )
            audio, meta = await self.peers.asset_content(
                p,
                profile["reference"]["content_ref"],
                profile["reference"]["asset_id"],
                profile["reference"]["asset_version_id"],
                10 << 20,
            )
            if digest(audio) != profile["reference"]["sha256"] or meta["mime"] not in ("audio/wav", "audio/x-wav"):
                raise Failure("reference_integrity_failed")
            profile_sha = voice_source["reference_sha256"]
        else:
            profile, profile_sha = await self.peers.profile(p, opened, pid, args["revision_id"], args["voice_profile"])
            if profile["model_manifest_sha256"] != self.model_hash:
                raise Failure("model_version_changed", 409)
            # Validate the actual immutable reference before accepting durable work.
            audio, meta = await self.peers.content(p, profile["reference"]["content_ref"], pid, args["revision_id"], 2 << 20)
            if digest(audio) != profile["reference"]["sha256"] or meta["mime"] not in ("audio/wav", "audio/x-wav"):
                raise Failure("reference_integrity_failed")
            voice_source = {
                "kind": "project_candidate",
                "voice_profile": args["voice_profile"],
                "profile_sha256": profile_sha,
            }
        emo_ref_sha = None
        if args.get("emotion_reference"):
            emo_ref = args["emotion_reference"]["content_ref"]
            self.peers.contains(opened, emo_ref)
            emo_audio, emo_meta = await self.peers.content(p, emo_ref, pid, args["revision_id"], 2 << 20)
            if emo_meta["mime"] not in ("audio/wav", "audio/x-wav"):
                raise Failure("reference_integrity_failed")
            emo_ref_sha = digest(emo_audio)
        entries = None
        if args.get("lexicon_ref"):
            lex_doc, _ = await self.peers.lexicon(p, opened, pid, args["revision_id"], args["lexicon_ref"])
            entries = lex_doc["entries"]
        text_record = build_text_record(args["text"], entries)
        if len(text_record["sent"]) > 500 or not text_record["sent"]:
            raise Failure("invalid_text")
        input_digest = compute_input_digest(text_record["sent"], voice_source, self.model_hash, args)
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
        state = {
            "profile": profile,
            "profile_sha256": profile_sha,
            "voice_source": voice_source,
            "text": text_record,
            "input_digest": input_digest,
            "capabilities": caps,
            "outputs": [],
            "timings": {"queue_seconds": 0.0, "load_seconds": 0.0, "inference_seconds": 0.0},
        }
        if emo_ref_sha:
            state["emotion_reference_sha256"] = emo_ref_sha
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
