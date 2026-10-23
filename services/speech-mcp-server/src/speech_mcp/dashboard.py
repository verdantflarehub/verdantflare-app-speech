"""Speech Dashboard read/review API handlers scoped to the verified caller."""
import asyncio
import base64
import contextlib

from . import __version__
from .domain import (
    TOOLS,
    Failure,
    TaskNotFound,
    build_text_record,
    check_take_staleness,
    digest,
    public,
    strict_json,
    validate,
)


def _review_state(state):
    rev = state.get("review_state")
    if rev in ("pending", "selected", "needs_revision"):
        return rev
    if state.get("candidate_state") == "selected" or state.get("listening_review") == "approved" or state.get("review") == "approved":
        return "selected"
    if state.get("listening_review") == "rejected" or state.get("review") == "rejected":
        return "needs_revision"
    return "pending"


def _archive_state(row):
    state = row["state"]
    if state.get("project_archive") == "archived":
        return "archived"
    if row["status"] == "succeeded" and len(state.get("outputs") or []) >= 4:
        return "archived_link_pending"
    return "not_archived"


def _elapsed_seconds(row):
    timings = (row["state"] or {}).get("timings") or {}
    total = sum(float(timings.get(k) or 0.0) for k in ("queue_seconds", "load_seconds", "inference_seconds"))
    if total > 0:
        return round(total, 3)
    if hasattr(row.get("created_at"), "timestamp") and hasattr(row.get("updated_at"), "timestamp"):
        return round(max(0.0, row["updated_at"].timestamp() - row["created_at"].timestamp()), 3)
    return 0.0


def _build_take_positions(rows):
    groups = {}
    for r in rows:
        uid = (r.get("request") or {}).get("utterance_id") or r["task_id"]
        key = (r["project_id"], uid)
        groups.setdefault(key, []).append(r)
    positions = {}
    for key, items in groups.items():
        ordered = sorted(items, key=lambda x: (x["created_at"], x["task_id"]))
        count = len(ordered)
        for idx, item in enumerate(ordered, start=1):
            positions[item["task_id"]] = (idx, count)
    return positions


def summarize_row(row, take_index=1, take_count=1, ui_fields=None):
    req = row.get("request") or {}
    state = row.get("state") or {}
    profile = state.get("profile") or {}
    report = state.get("report") or {}
    voice_src_obj = state.get("voice_source") or {}
    voice_kind = voice_src_obj.get("kind") or ("world" if "world_voice" in req else "project_candidate")
    status = row["status"]
    last_error = row.get("last_error")
    recovery_required = bool(
        status == "failed" and last_error in ("recovery_required", "model_version_changed", "unverified_voice_engine")
    )
    summary = {
        "task_id": row["task_id"],
        "project_id": row["project_id"],
        "utterance_id": req.get("utterance_id") or "U01",
        "text": req.get("text") or "",
        "status": status,
        "recovery_required": recovery_required,
        "phase": row["phase"],
        "voice_source": voice_kind,
        "take_index": max(1, int(take_index)),
        "take_count": max(1, int(take_count)),
        "archive": _archive_state(row),
        "review": _review_state(state),
        "created_at": row["created_at"].isoformat(),
        "elapsed_seconds": _elapsed_seconds(row),
    }
    character = req.get("character") or profile.get("speaker") or profile.get("label") or profile.get("voice_id")
    if character:
        summary["character"] = str(character)[:128]
    label = profile.get("label") or profile.get("voice_id")
    if label:
        summary["voice_label"] = str(label)[:128]
    editing_metrics = report.get("editing") if isinstance(report, dict) else None
    if isinstance(editing_metrics, dict) and (editing_metrics.get("duration_seconds") or 0) > 0:
        summary["duration_seconds"] = float(editing_metrics["duration_seconds"])
    shot_win = req.get("shot_window_seconds") or (report.get("shot_window_seconds") if isinstance(report, dict) else None)
    if shot_win and float(shot_win) > 0:
        summary["shot_window_seconds"] = float(shot_win)
    if isinstance(report, dict) and isinstance(report.get("warnings"), list):
        valid_codes = {"clipping", "true_peak_over", "transcript_mismatch", "window_overrun"}
        codes = [
            w.get("code") if isinstance(w, dict) else w
            for w in report["warnings"]
        ]
        filtered = [c for c in codes if c in valid_codes]
        summary["warnings"] = filtered
    else:
        summary["warnings"] = []
    if last_error:
        summary["error_code"] = str(last_error)[:64]
    if isinstance(ui_fields, dict):
        summary.update(ui_fields)
    return validate("task_summary", summary)


class DashboardAPI:
    def __init__(self, service):
        self.service = service
        self.store = service.store
        self.peers = service.peers

    async def overview(self, p, *, include_nsfw=False):
        rows = await asyncio.to_thread(self.store.list_for_principal, p)
        rows = [r for r in rows if include_nsfw or not self.service.is_row_nsfw(r)]
        positions = _build_take_positions(rows)
        summaries = [
            summarize_row(
                r,
                *positions.get(r["task_id"], (1, 1)),
                ui_fields=self.service.row_ui_fields(r, can_write=True),
            )
            for r in rows
        ]
        rt = await self.service.engine_runtime_info(p)
        kpis = {
            "total_tasks": len(summaries),
            "succeeded_tasks": sum(1 for s in summaries if s["status"] == "succeeded"),
            "running_tasks": sum(1 for s in summaries if s["status"] in ("pending", "running")),
            "failed_tasks": sum(1 for s in summaries if s["status"] == "failed"),
            "recovery_required_tasks": sum(1 for s in summaries if s["recovery_required"]),
            "selected_takes": sum(1 for s in summaries if s["review"] == "selected"),
            "needs_revision_takes": sum(1 for s in summaries if s["review"] == "needs_revision"),
            "pending_review_takes": sum(1 for s in summaries if s["status"] == "succeeded" and s["review"] == "pending"),
            "warning_tasks": sum(1 for s in summaries if s.get("warnings")),
        }
        projects = sorted({s["project_id"] for s in summaries})
        characters = sorted({s["character"] for s in summaries if s.get("character")})
        return {
            "service_version": __version__,
            "kpis": kpis,
            "engine": rt,
            "engines": [rt["capability"]],
            "tools": TOOLS,
            "filters": {"projects": projects, "characters": characters},
            "recent_tasks": summaries[:500],
        }

    async def query_tasks(self, p, query=None, *, include_nsfw=False):
        raw_q = dict(query or {})
        limit_override = raw_q.pop("limit", None)
        offset_override = raw_q.pop("offset", None)
        if "page" in raw_q and isinstance(raw_q["page"], str) and raw_q["page"].isdigit():
            raw_q["page"] = int(raw_q["page"])
        if "page_size" in raw_q and isinstance(raw_q["page_size"], str) and raw_q["page_size"].isdigit():
            raw_q["page_size"] = int(raw_q["page_size"])
        q_obj = validate("dashboard_task_query", raw_q)
        rows = await asyncio.to_thread(self.store.list_for_principal, p, q_obj.get("project_id"))
        rows = [r for r in rows if include_nsfw or not self.service.is_row_nsfw(r)]
        positions = _build_take_positions(rows)
        summaries = [
            summarize_row(
                r,
                *positions.get(r["task_id"], (1, 1)),
                ui_fields=self.service.row_ui_fields(r, can_write=True),
            )
            for r in rows
        ]
        filtered = []
        for s in summaries:
            if q_obj.get("project_id") and s["project_id"] != q_obj["project_id"]:
                continue
            if q_obj.get("character") and s.get("character") != q_obj["character"]:
                continue
            if q_obj.get("review") and s["review"] != q_obj["review"]:
                continue
            if q_obj.get("utterance_id") and s["utterance_id"] != q_obj["utterance_id"]:
                continue
            if q_obj.get("status"):
                st = q_obj["status"]
                if st == "recovery_required":
                    if not s["recovery_required"]:
                        continue
                elif s["status"] != st:
                    continue
            if q_obj.get("q"):
                needle = q_obj["q"].strip().lower()
                haystack = " ".join([
                    s["task_id"],
                    s["utterance_id"],
                    s.get("character", ""),
                    s.get("voice_label", ""),
                    s["text"],
                ]).lower()
                if needle and needle not in haystack:
                    continue
            filtered.append(s)
        page = int(q_obj.get("page", 1))
        page_size = int(q_obj.get("page_size", 50))
        if limit_override is not None:
            page_size = max(1, min(100, int(limit_override)))
        start = int(offset_override) if offset_override is not None else (page - 1) * page_size
        end = start + page_size
        page_items = filtered[start:end]
        result = {
            "items": page_items,
            "tasks": page_items,
            "total": len(filtered),
            "page": page,
            "page_size": page_size,
        }
        return validate("task_list_result", result)

    async def task_stats(self, p, *, project_id=None, include_nsfw=False):
        rows = await asyncio.to_thread(self.store.list_for_principal, p, project_id)
        rows = [r for r in rows if include_nsfw or not self.service.is_row_nsfw(r)]
        succeeded = [r for r in rows if r.get("status") == "succeeded"]
        durations = [_elapsed_seconds(r) for r in succeeded]
        avg_dur = round(sum(durations) / len(durations), 2) if durations else 0.0
        total = len(rows)
        pending_cnt = sum(1 for r in rows if r.get("status") == "pending")
        running_cnt = sum(1 for r in rows if r.get("status") == "running")
        failed_cnt = sum(1 for r in rows if r.get("status") == "failed")
        canceled_cnt = sum(1 for r in rows if r.get("status") in ("cancelled", "canceled"))
        return {
            "total": total,
            "total_tasks": total,
            "queued": pending_cnt,
            "pending": pending_cnt,
            "running": running_cnt,
            "running_tasks": pending_cnt + running_cnt,
            "completed": len(succeeded),
            "succeeded": len(succeeded),
            "succeeded_tasks": len(succeeded),
            "failed": failed_cnt,
            "failed_tasks": failed_cnt,
            "canceled": canceled_cnt,
            "avg_duration": avg_dur,
        }

    async def _ensure_report(self, p, row):
        state = row.get("state") or {}
        if isinstance(state.get("report"), dict):
            return state["report"]
        if row["status"] != "succeeded":
            return None
        outputs = state.get("outputs") or []
        report_entry = next((x for x in outputs if x.get("name") == "report.json"), None)
        if not report_entry or not report_entry.get("content_ref"):
            return None
        with contextlib.suppress(Failure):
            raw, _ = await self.peers.task_output_content(p, report_entry["content_ref"], row["project_id"], 1 << 20)
            doc = strict_json(raw)
            if isinstance(doc, dict):
                state["report"] = doc
                return doc
        return None

    async def task_detail(self, p, task_id, *, include_nsfw=False):
        validate("id", task_id)
        row = await asyncio.to_thread(self.store.get_by_id, p, task_id)
        if not include_nsfw and self.service.is_row_nsfw(row):
            raise TaskNotFound("task_not_found", 404)
        opened = await self.peers.open(p, row["project_id"], write=False)
        can_write = bool(opened.get("can_write", True))
        req = row.get("request") or {}
        state = row.get("state") or {}
        utterance_id = req.get("utterance_id") or "U01"
        siblings = await asyncio.to_thread(self.store.utterance_takes, p, row["project_id"], utterance_id)
        siblings = [s for s in siblings if include_nsfw or not self.service.is_row_nsfw(s)]
        if not siblings:
            siblings = [row]
        positions = _build_take_positions(siblings)
        report_doc = await self._ensure_report(p, row)
        takes = []
        for sib in siblings:
            idx, cnt = positions.get(sib["task_id"], (1, len(siblings)))
            takes.append(
                summarize_row(
                    sib,
                    idx,
                    cnt,
                    ui_fields=self.service.row_ui_fields(sib, can_write=can_write),
                )
            )
        cur_idx, cur_cnt = positions.get(row["task_id"], (1, len(siblings)))
        ui_fields = self.service.row_ui_fields(row, can_write=can_write)
        summary = summarize_row(row, cur_idx, cur_cnt, ui_fields=ui_fields)
        voice_source = state.get("voice_source") or (
            {"kind": "world", "world_voice": req["world_voice"], "reference_sha256": state["profile"]["reference"]["sha256"]}
            if "world_voice" in req
            else {"kind": "project_candidate", "voice_profile": req.get("voice_profile"), "profile_sha256": state.get("profile_sha256")}
        )
        generation = state.get("generation")
        staleness = {"stale": False, "reasons": [], "input_digest": state.get("input_digest")}
        if isinstance(generation, dict):
            staleness = check_take_staleness(
                generation,
                voice_source,
                current_sent_text=(state.get("text") or {}).get("sent"),
                current_model_manifest_sha256=self.service.model_hash,
            )
        head_revision_id = (
            opened.get("head_revision_id")
            or opened.get("revision_id")
            or req.get("revision_id")
        )
        text_info = state.get("text") or build_text_record(req.get("text", ""))
        timings = state.get("timings") or {"queue_seconds": 0.0, "load_seconds": 0.0, "inference_seconds": 0.0}
        emotion_mode = "none"
        emotion_input = None
        if "emotion_vector" in req:
            emotion_mode = "emotion_vector"
            emotion_input = req["emotion_vector"]
        elif "emotion_text" in req:
            emotion_mode = "emotion_text"
            emotion_input = req["emotion_text"]
        elif "emotion_reference" in req:
            emotion_mode = "emotion_reference"
            emotion_input = req["emotion_reference"]
        resolved_emotion = (
            (generation or {}).get("resolved_emotion_vector")
            if isinstance(generation, dict) and "resolved_emotion_vector" in generation
            else (report_doc or {}).get("resolved_emotion_vector")
        )
        if resolved_emotion is None and "emotion_vector" in req:
            resolved_emotion = req["emotion_vector"]
        return {
            "summary": summary,
            "task": public(row, filter_nsfw_outputs=None if include_nsfw else self.service.active_ratings()),
            "request": req,
            "head_revision_id": head_revision_id,
            "can_write": can_write,
            **ui_fields,
            "voice_profile": state.get("profile"),
            "voice_source": voice_source,
            "text": text_info,
            "emotion": {
                "mode": emotion_mode,
                "input": emotion_input,
                "resolved_vector": resolved_emotion,
            },
            "timings": timings,
            "report": report_doc,
            "generation": generation,
            "staleness": staleness,
            "takes": takes,
            "review_note": state.get("review_note", ""),
            "reviewed_revision_id": state.get("reviewed_revision_id"),
        }

    async def task_audio(self, p, task_id, kind, *, include_nsfw=False):
        validate("id", task_id)
        if kind not in ("original", "editing"):
            raise Failure("invalid_arguments", 400)
        row = await asyncio.to_thread(self.store.get_by_id, p, task_id)
        if not include_nsfw and self.service.is_row_nsfw(row):
            raise TaskNotFound("task_not_found", 404)
        await self.peers.open(p, row["project_id"], write=False)
        if row["status"] != "succeeded":
            raise Failure("result_not_ready", 409)
        filename = kind + ".wav"
        outputs = (row.get("state") or {}).get("outputs") or []
        entry = next((x for x in outputs if x.get("name") == filename), None)
        if not entry:
            raise Failure("result_not_ready", 409)
        cref = entry.get("content_ref") if isinstance(entry, dict) else None
        vid = (cref or {}).get("version_id") or (cref or {}).get("artifact_id")
        if not include_nsfw and vid and self.service.active_ratings().is_nsfw(vid, task_id=task_id):
            raise TaskNotFound("task_not_found", 404)
        if entry.get("content_ref"):
            data, _ = await self.peers.task_output_content(p, entry["content_ref"], row["project_id"], 16 << 20)
        else:
            rt_id = (row.get("state") or {}).get("runtime_task_id")
            data = await self.peers.runtime.request(
                p, "GET", f"/v1/tasks/{rt_id}/outputs/{filename}", project_id=row["project_id"], limit=16 << 20
            )
        b64 = base64.b64encode(data).decode("ascii")
        return {
            "task_id": task_id,
            "kind": kind,
            "filename": filename,
            "mime": "audio/wav",
            "size": len(data),
            "sha256": digest(data),
            "data_url": "data:audio/wav;base64," + b64,
        }

    async def write_review(self, p, task_id, payload):
        validate("id", task_id)
        body = validate("review_write", payload)
        if body["task_id"] != task_id:
            raise Failure("invalid_arguments", 400)
        row = await asyncio.to_thread(self.store.get, p, body["project_id"], task_id)
        opened = await self.peers.open(p, row["project_id"], write=True)
        if opened.get("can_write") is False:
            raise Failure("permission_denied", 403)
        head_rev = opened.get("head_revision_id") or opened.get("revision_id")
        if head_rev and body["expected_revision_id"] != head_rev:
            raise Failure("revision_conflict", 409)
        await asyncio.to_thread(
            self.store.update_review,
            p,
            body["project_id"],
            task_id,
            body["decision"],
            body.get("note"),
            body["expected_revision_id"],
        )
        return await self.task_detail(p, task_id, include_nsfw=True)

    async def models(self, p):
        rt = await self.service.engine_runtime_info(p)
        return {
            "service_version": __version__,
            "engines": [rt["capability"]],
            "runtime": rt,
        }

    async def mcp_status(self, p):
        rt = await self.service.engine_runtime_info(p)
        return {
            "service_version": __version__,
            "domain": "speech",
            "tools": TOOLS,
            "engine": rt,
        }
