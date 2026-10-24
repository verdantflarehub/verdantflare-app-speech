import asyncio
import os
from pathlib import Path
import re
import tempfile
import uuid

from . import __version__
from .content_rating import ContentRatingStore
from .domain import (
    DEFAULT_ADAPTER,
    Failure,
    TaskNotFound,
    build_text_record,
    compute_input_digest,
    digest,
    public,
    validate,
)
from .gpu_lease import GpuCoordinator

VOICE_PATH = re.compile(r"^speech/voices/[a-z0-9][a-z0-9_-]{0,63}/profile\.json$")


class Service:
    def __init__(self, store, peers, model_hash, gpu=None, adapter=None, ratings=None):
        validate("hash", model_hash)
        self.store, self.peers, self.model_hash = store, peers, model_hash
        self.gpu = gpu if gpu is not None else GpuCoordinator.from_environment()
        self.adapter = adapter if adapter is not None else DEFAULT_ADAPTER
        if ratings is not None:
            self.ratings = ratings
        elif getattr(store, "ratings", None) is not None:
            self.ratings = store.ratings
        else:
            db_path = getattr(store, "db_path", None)
            if not db_path:
                env_db = os.environ.get("SPEECH_CONTENT_RATING_DB", "").strip()
                if env_db:
                    db_path = Path(env_db)
                else:
                    db_path = Path(tempfile.gettempdir()) / f"speech_ratings_{id(store)}_{uuid.uuid4().hex[:8]}.db"
            self.ratings = ContentRatingStore(Path(db_path), domain="speech")
        try:
            self.store.ratings = self.ratings
        except Exception:
            pass

    def active_ratings(self) -> ContentRatingStore:
        store_ratings = getattr(self.store, "ratings", None)
        if isinstance(store_ratings, ContentRatingStore):
            self.ratings = store_ratings
        return self.ratings

    @staticmethod
    def row_version_ids(row) -> list[str]:
        vids: list[str] = []
        if not isinstance(row, dict):
            return vids
        top_art = row.get("artifact_id")
        if top_art and str(top_art).strip() not in vids:
            vids.append(str(top_art).strip())
        state = row.get("state") or {}
        st_art = state.get("artifact_id")
        if st_art and str(st_art).strip() not in vids:
            vids.append(str(st_art).strip())
        for out in state.get("outputs") or []:
            if not isinstance(out, dict):
                continue
            cref = out.get("content_ref")
            if isinstance(cref, dict):
                for k in ("version_id", "artifact_id"):
                    val = cref.get(k)
                    if val and str(val).strip() not in vids:
                        vids.append(str(val).strip())
            for k in ("version_id", "artifact_id"):
                val = out.get(k)
                if val and str(val).strip() not in vids:
                    vids.append(str(val).strip())
        return vids

    @classmethod
    def primary_version_id(cls, row) -> str:
        vids = cls.row_version_ids(row)
        if vids:
            return vids[0]
        return str(row.get("task_id") or "")

    def is_row_nsfw(self, row) -> bool:
        if not isinstance(row, dict):
            return False
        store = self.active_ratings()
        tid = row.get("task_id")
        if tid and store.is_nsfw(str(tid), task_id=str(tid)):
            return True
        for vid in self.row_version_ids(row):
            if store.is_nsfw(vid, task_id=str(tid) if tid else None):
                return True
        return False

    def row_ui_fields(self, row, *, can_write: bool = True) -> dict:
        store = self.active_ratings()
        tid = str(row.get("task_id") or "") or None
        pvid = self.primary_version_id(row) or tid
        for vid in self.row_version_ids(row):
            if store.get(vid, task_id=tid).get("rating") != "unrated":
                pvid = vid
                break
        return store.ui_fields(pvid, task_id=tid, can_write=can_write)

    def find_row_by_version(self, version_id: str, *, p=None, project_id: str | None = None):
        vid = (version_id or "").strip()
        if not vid:
            return None
        candidates = []
        if p is not None and hasattr(self.store, "list_for_principal"):
            candidates = self.store.list_for_principal(p, project_id)
        elif hasattr(self.store, "rows") and isinstance(self.store.rows, dict):
            candidates = list(self.store.rows.values())
        elif hasattr(self.store, "find_by_version"):
            found = self.store.find_by_version(vid, project_id=project_id)
            if found is not None:
                candidates = [found]
        for r in candidates:
            if project_id and r.get("project_id") != project_id:
                continue
            if r.get("task_id") == vid or vid in self.row_version_ids(r):
                return r
        return None

    def default_principal(self):
        if hasattr(self.store, "rows") and isinstance(self.store.rows, dict) and self.store.rows:
            first = next(iter(self.store.rows.values()))
            if isinstance(first, dict) and isinstance(first.get("principal"), dict):
                return dict(first["principal"])
        if hasattr(self.store, "row") and isinstance(self.store.row, dict) and isinstance(self.store.row.get("principal"), dict):
            return dict(self.store.row["principal"])
        return None

    def _sync_rating_across_row(self, row, primary_vid: str, action: str, **kwargs) -> None:
        store = self.active_ratings()
        tid = str(row.get("task_id") or "")
        pid = str(row.get("project_id") or "")
        all_ids = {tid, *self.row_version_ids(row)} - {"", primary_vid}
        for cid in all_ids:
            if action == "rate":
                store.rate_content(
                    version_id=cid,
                    project_id=pid,
                    task_id=tid or None,
                    **kwargs,
                )
            elif action == "clear":
                store.clear_rating(
                    version_id=cid,
                    project_id=pid,
                    task_id=tid or None,
                    **kwargs,
                )
            elif action == "mark":
                store.mark_rating(
                    version_id=cid,
                    project_id=pid,
                    task_id=tid or None,
                    **kwargs,
                )

    def clear_content_rating(
        self,
        version_id: str,
        *,
        actor_user_id: str,
        note: str | None = None,
        project_id: str | None = None,
        p=None,
    ) -> dict:
        vid = (version_id or "").strip()
        row = self.find_row_by_version(vid, p=p, project_id=project_id)
        if row is None and p is not None:
            row = self.find_row_by_version(vid, p=None, project_id=project_id)
        if row is None:
            raise TaskNotFound(f"NOT_FOUND: {vid}")
        pid = row["project_id"]
        tid = row["task_id"]
        store = self.active_ratings()
        res = store.clear_rating(
            version_id=vid,
            project_id=pid,
            actor_user_id=actor_user_id,
            note=note,
            task_id=tid,
        )
        self._sync_rating_across_row(
            row,
            vid,
            "clear",
            actor_user_id=actor_user_id,
            note=note,
        )
        return res

    def mark_content_rating(
        self,
        version_id: str,
        *,
        actor_user_id: str,
        reason: str | None = None,
        project_id: str | None = None,
        p=None,
    ) -> dict:
        vid = (version_id or "").strip()
        row = self.find_row_by_version(vid, p=p, project_id=project_id)
        if row is None and p is not None:
            row = self.find_row_by_version(vid, p=None, project_id=project_id)
        if row is None:
            raise TaskNotFound(f"NOT_FOUND: {vid}")
        pid = row["project_id"]
        tid = row["task_id"]
        store = self.active_ratings()
        res = store.mark_rating(
            version_id=vid,
            project_id=pid,
            actor_user_id=actor_user_id,
            reason=reason,
            task_id=tid,
        )
        self._sync_rating_across_row(
            row,
            vid,
            "mark",
            actor_user_id=actor_user_id,
            reason=reason,
        )
        return res

    def rename_task(self, task_id: str, title: str, *, p=None) -> dict:
        if not isinstance(title, str):
            raise Failure("invalid_arguments", 400)
        clean_title = title.strip()
        if not clean_title or len(clean_title) > 120:
            raise Failure("invalid_arguments", 400)
        if hasattr(self.store, "rename_task"):
            try:
                updated = self.store.rename_task(task_id, clean_title, p=p)
            except TypeError:
                updated = self.store.rename_task(task_id, clean_title)
            return {
                "status": "ok",
                "task_id": task_id,
                "title": clean_title,
                "task": updated if isinstance(updated, dict) else None,
            }
        row = self.find_row_by_version(task_id, p=p)
        if row is None:
            raise TaskNotFound("task_not_found", 404)
        row["title"] = clean_title
        if isinstance(row.get("state"), dict):
            row["state"]["title"] = clean_title
        return {"status": "ok", "task_id": task_id, "title": clean_title}

    def delete_task(self, task_id: str, *, p=None) -> dict:
        if hasattr(self.store, "delete_task"):
            try:
                res = self.store.delete_task(task_id, p=p)
            except TypeError:
                res = self.store.delete_task(task_id)
            if isinstance(res, dict) and res.get("status") == "deleted":
                return res
            return {"status": "deleted", "task_id": task_id}
        if hasattr(self.store, "rows") and isinstance(self.store.rows, dict):
            if task_id not in self.store.rows:
                raise TaskNotFound("task_not_found", 404)
            self.store.rows.pop(task_id, None)
            if getattr(self.store, "prior", None) and self.store.prior.get("task_id") == task_id:
                self.store.prior = None
            return {"status": "deleted", "task_id": task_id}
        raise TaskNotFound("task_not_found", 404)

    def update_task_content_rating(
        self,
        task_id: str,
        action: str,
        *,
        reason: str | None = None,
        actor_user_id: str | None = None,
        p=None,
    ) -> dict:
        clean_action = str(action or "").strip().lower()
        if clean_action not in ("mark", "clear"):
            raise Failure("invalid_arguments", 400)
        row = self.find_row_by_version(task_id, p=p)
        if row is None and p is not None:
            row = self.find_row_by_version(task_id, p=None)
        if row is None:
            raise TaskNotFound("task_not_found", 404)
        actor = (
            (actor_user_id or "").strip()
            or ((p or {}).get("subject_id") or "").strip()
            or "studio-user"
        )
        primary_vid = self.primary_version_id(row) or task_id
        if clean_action == "mark":
            rec = self.mark_content_rating(
                primary_vid,
                actor_user_id=actor,
                reason=reason or "manual mark from card menu",
                project_id=row.get("project_id"),
                p=p,
            )
            return {
                **rec,
                "status": "ok",
                "task_id": row["task_id"],
                "content_rating": "nsfw",
            }
        rec = self.clear_content_rating(
            primary_vid,
            actor_user_id=actor,
            note=reason or "manual clear from card menu",
            project_id=row.get("project_id"),
            p=p,
        )
        return {
            **rec,
            "status": "ok",
            "task_id": row["task_id"],
            "content_rating": "general",
        }

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
        if action not in ("capabilities", "voices", "create", "status", "result", "list", "rate_content") or name != "speech." + action:
            raise Failure("tool_not_found", 404)
        validate(action, args)
        if action == "capabilities":
            rt = await self.engine_runtime_info(p)
            return validate(
                "capabilities_result",
                {
                    "service_version": __version__,
                    "engines": [rt["capability"]],
                    "content_rating": {
                        "supported": True,
                        "tool": "speech.rate_content",
                        "rubric_version": "rubric v1.0",
                        "ratings": ["general", "nsfw"],
                        "mcp_clear_supported": False,
                    },
                },
            )
        if action == "rate_content":
            vid = str(args.get("version_id") or "").strip()
            if not vid and isinstance(args.get("content_ref"), dict):
                vid = str(args["content_ref"].get("version_id") or args["content_ref"].get("artifact_id") or "").strip()
            if not vid and args.get("task_id"):
                vid = str(args["task_id"]).strip()
            if not vid:
                raise TaskNotFound("NOT_FOUND: version_id is required")
            project_hint = args.get("project_id")
            row = await asyncio.to_thread(self.find_row_by_version, vid, p=p, project_id=project_hint)
            if row is None and p is not None:
                row = await asyncio.to_thread(self.find_row_by_version, vid, p=None, project_id=project_hint)
            if row is None:
                raise TaskNotFound(f"NOT_FOUND: {vid}")
            pid = row["project_id"]
            tid = row["task_id"]
            actor_id = (p or {}).get("subject_id") or (p or {}).get("user_id") or "skill:k09"
            store = self.active_ratings()
            res = store.rate_content(
                version_id=vid,
                project_id=pid,
                rating=args.get("rating", ""),
                reasons=args.get("reasons"),
                reason=args.get("reason"),
                rubric_version=args.get("rubric_version", "rubric v1.0"),
                evaluator_model=args.get("evaluator_model"),
                rated_by=actor_id,
                task_id=tid,
                content_ref=args.get("content_ref"),
            )
            self._sync_rating_across_row(
                row,
                vid,
                "rate",
                rating=args.get("rating", ""),
                reasons=args.get("reasons"),
                reason=args.get("reason"),
                rubric_version=args.get("rubric_version", "rubric v1.0"),
                evaluator_model=args.get("evaluator_model"),
                rated_by=actor_id,
                content_ref=args.get("content_ref"),
            )
            return res
        if action == "list":
            pid = args.get("project_id")
            if pid and self.peers is not None and getattr(self.peers, "open", None) is not None:
                await self.peers.open(p, pid, write=False)
            eff_p = p or self.default_principal()
            rows = await asyncio.to_thread(self.store.list_for_principal, eff_p, pid) if eff_p else []
            visible = [r for r in rows if not self.is_row_nsfw(r)]
            if args.get("status"):
                visible = [r for r in visible if r.get("status") == args["status"]]
            limit = int(args.get("limit", 50))
            offset = int(args.get("offset", 0))
            sliced = [public(r, filter_nsfw_outputs=self.active_ratings()) for r in visible[offset : offset + limit]]
            return {
                "tasks": sliced,
                "items": sliced,
                "total": len(visible),
                "limit": limit,
                "offset": offset,
            }
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
            if self.is_row_nsfw(row):
                raise TaskNotFound(f"NOT_FOUND: {args['task_id']}")
            if action == "result" and row["status"] != "succeeded":
                raise Failure("result_not_ready", 409)
            return public(row, filter_nsfw_outputs=self.active_ratings())
        prior = await asyncio.to_thread(self.store.existing, p, args)
        if prior:
            return public(prior, filter_nsfw_outputs=self.active_ratings())
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
