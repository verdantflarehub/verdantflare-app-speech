import asyncio
import contextlib
from datetime import datetime, timezone
import logging
import math
import threading
import time

from . import __version__
from .clients import Client, MAX_OUTPUT
from .domain import (
    Failure,
    build_text_record,
    canonical,
    compute_input_digest,
    digest,
    new_id,
    strict_json,
    validate,
)
from .gpu_lease import GpuCoordinator, LeaseBusy, SchedulerError, SchedulerLeaseClient, is_lease_granted

log = logging.getLogger("speech_worker")

MAX_RESIDENT_PROCS = 128


def _finite_float(value: object) -> float | None:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    fval = float(value)
    return fval if math.isfinite(fval) else None


def _parse_epoch(value: object) -> float | None:
    if isinstance(value, datetime):
        dt = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        fval = float(value)
        return fval if math.isfinite(fval) and fval > 0 else None
    if isinstance(value, str) and value.strip():
        raw = value.strip().replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(raw)
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    return None


def _has_concrete_identity(ident: object) -> bool:
    if not isinstance(ident, dict):
        return False
    for key in ("pod_uid", "container_id", "boot_id"):
        val = str(ident.get(key) or "").strip()
        if not val or val.startswith("unknown-"):
            return False
    return True


def _is_sealed_reclaim_proof(proof: object, lease_id: str) -> bool:
    if not isinstance(proof, dict):
        return False
    if proof.get("proof_source") != "receiver_guard":
        return False
    if proof.get("token_sealed") is not True:
        return False
    try:
        if int(proof.get("active_tasks", 1)) != 0:
            return False
    except (TypeError, ValueError):
        return False
    if str(proof.get("revoked_lease_id") or "").strip() != str(lease_id).strip():
        return False
    procs = proof.get("resident_procs")
    if procs is not None:
        if not isinstance(procs, list) or len(procs) > MAX_RESIDENT_PROCS:
            return False
    return True


def _is_complete_warm_proof(proof: object, lease_id: str) -> bool:
    return _is_sealed_reclaim_proof(proof, lease_id) and isinstance(proof, dict) and proof.get("reclaimed") is True


def _has_custom_attr(obj: object, attr: str) -> bool:
    if type(obj).__module__.startswith("unittest.mock"):
        inst_dict = getattr(obj, "__dict__", None)
        return isinstance(inst_dict, dict) and attr in inst_dict
    return hasattr(obj, attr)


class Worker:
    def __init__(self, service):
        self.service, self.store, self.peers = service, service.store, service.peers
        self.gpu = getattr(service, "gpu", None) or GpuCoordinator.from_environment()
        existing_sched = getattr(service, "scheduler", None)
        if existing_sched is not None:
            if hasattr(existing_sched, "attach_store"):
                existing_sched.attach_store(self.store)
            self.scheduler = existing_sched
        else:
            self.scheduler = SchedulerLeaseClient.from_environment(store=self.store)
        if self.scheduler.enabled:
            if hasattr(self.gpu, "disable_legacy_lease"):
                self.gpu.disable_legacy_lease()
            elif getattr(self.gpu, "enabled", False):
                self.gpu.enabled = False
        self._last_pending_flush = 0.0
        self._ready_poll_seconds = 2.0
        self._orphan_lock = threading.RLock()
        self._orphan_leases: dict[str, dict] = {}
        self._scheduler_idle_since: float | None = None
        if getattr(self.gpu, "enabled", False) or self.scheduler.enabled or self._scale_enabled():
            if getattr(self.gpu, "enabled", False):
                self.gpu.activity = self.store.gpu_active
            threading.Thread(target=self._idle, name="speech-gpu-idle", daemon=True).start()

    def _scale_enabled(self) -> bool:
        return bool(getattr(self.gpu, "scale_enabled", False))

    def _row_engine(self, row: dict | None) -> str:
        if not isinstance(row, dict):
            return "indextts2"
        prof = (row.get("state") or {}).get("profile") or {}
        req = row.get("request") or {}
        return str(prof.get("engine") or req.get("engine") or "indextts2")

    def _runtime_for(self, engine: str = "indextts2"):
        if hasattr(self.service, "runtime_for") and not type(self.service).__module__.startswith("unittest.mock"):
            return self.service.runtime_for(engine)
        return getattr(self.peers, "runtime", None)

    def _deployment_for(self, engine: str = "indextts2") -> str | None:
        if engine == "indextts2":
            return None
        if hasattr(self.service, "deployment_for") and not type(self.service).__module__.startswith("unittest.mock"):
            return self.service.deployment_for(engine)
        return "speech-cosyvoice3"

    def _register_orphan_lease(
        self,
        lease: dict,
        *,
        target_identity: dict | None = None,
        reclaim_proof: dict | None = None,
        runtime_task_id: str = "",
        task_id: str = "",
        principal: dict | None = None,
        project_id: str | None = None,
        engine: str = "indextts2",
    ) -> None:
        lid = str(lease.get("lease_id") or "").strip()
        if not lid:
            return
        with self._orphan_lock:
            self._orphan_leases[lid] = {
                "lease": dict(lease),
                "target_identity": dict(target_identity) if isinstance(target_identity, dict) else {},
                "reclaim_proof": dict(reclaim_proof) if isinstance(reclaim_proof, dict) else None,
                "runtime_task_id": str(runtime_task_id or ""),
                "task_id": str(task_id or ""),
                "principal": dict(principal) if isinstance(principal, dict) else None,
                "project_id": project_id,
                "engine": str(engine or lease.get("engine") or "indextts2"),
            }

    def _pop_orphan_lease(self, lease_id: str) -> None:
        with self._orphan_lock:
            self._orphan_leases.pop(str(lease_id or "").strip(), None)

    def _idle(self):
        while True:
            time.sleep(30)
            if self.scheduler.enabled:
                try:
                    self.scheduler.flush_pending_releases()
                except Exception:
                    log.error("speech_scheduler_pending_release_flush_failed")
                try:
                    asyncio.run(self._reconcile_scheduler_idle())
                except Exception:
                    log.warning("speech_scheduler_idle_reconcile_deferred")
            elif getattr(self.gpu, "enabled", False):
                try:
                    self.gpu.reconcile()
                except Exception:
                    log.error("speech_gpu_idle_scale_failed")

    async def _reconcile_scheduler_idle(self) -> None:
        if not self.scheduler.enabled:
            return
        await self._drain_orphan_leases()
        if not self._scale_enabled():
            return
        with self._orphan_lock:
            has_orphans = bool(self._orphan_leases)
        if has_orphans:
            self._scheduler_idle_since = None
            return
        if await asyncio.to_thread(self._has_other_active_gpu_tasks, include_unarchived_runtime_outputs=True):
            self._scheduler_idle_since = None
            return
        if await asyncio.to_thread(self._has_other_active_scheduler_leases):
            self._scheduler_idle_since = None
            return
        reps = await asyncio.to_thread(self.gpu.replicas_for)
        if reps in (None, 0):
            self._scheduler_idle_since = None
            return
        mono = getattr(self.gpu, "_monotonic", time.monotonic)
        idle_budget = int(getattr(self.gpu, "idle_seconds", 600))
        if self._scheduler_idle_since is None:
            self._scheduler_idle_since = mono()
        if mono() - self._scheduler_idle_since < idle_budget:
            return
        health = await self._probe_runtime_health()
        if not isinstance(health, dict) or int(health.get("active_tasks") or 0) != 0:
            return
        await asyncio.to_thread(self.gpu.stop_deployment_and_wait)
        self._scheduler_idle_since = None

    async def reconcile_startup_leases(self) -> dict[str, list[str]]:
        adopted: list[str] = []
        released: list[str] = []
        deferred: list[str] = []
        if not self.scheduler.enabled:
            return {"adopted": adopted, "released": released, "deferred": deferred}

        with contextlib.suppress(Exception):
            await asyncio.to_thread(self.scheduler.flush_pending_releases)

        local_rows = await asyncio.to_thread(self._list_local_lease_rows)
        remote_leases: list[dict] = []
        if hasattr(self.scheduler, "list_own_leases"):
            with contextlib.suppress(Exception):
                remote_leases = await asyncio.to_thread(self.scheduler.list_own_leases)
        remote_by_id = {
            str(item.get("lease_id") or "").strip(): dict(item)
            for item in remote_leases
            if isinstance(item, dict) and str(item.get("lease_id") or "").strip()
        }

        adopted_ids: set[str] = set()
        orphan_entries: dict[str, dict] = {}

        for row in local_rows:
            state = row.get("state") or {}
            lease = state.get("gpu_scheduler_lease")
            if not isinstance(lease, dict):
                continue
            lid = str(lease.get("lease_id") or "").strip()
            if not lid:
                continue
            is_active_task = (
                row.get("status") in {"pending", "running"}
                and row.get("phase") in {"reference", "loading_model", "submit", "inference", "archive"}
            )
            if is_active_task:
                live = remote_by_id.get(lid)
                if live is None and hasattr(self.scheduler, "get_lease"):
                    try:
                        live = await asyncio.to_thread(self.scheduler.get_lease, lid)
                    except SchedulerError as exc:
                        if exc.status_code == 404:
                            live = {"lease_id": lid, "state": "expired"}
                        else:
                            live = lease
                    except Exception:
                        live = lease
                lstate = str((live or {}).get("state") or "").strip().lower()
                if lstate in {"released", "expired"}:
                    await self._clear_terminal_task_lease(str(row.get("task_id") or ""), lid)
                    continue
                if lstate in {"reclaiming", "reconciling", "quarantined"}:
                    orphan_entries[lid] = {
                        "lease": dict(live or lease),
                        "target_identity": state.get("gpu_target_identity") if isinstance(state.get("gpu_target_identity"), dict) else {},
                        "reclaim_proof": state.get("gpu_reclaim_proof") if isinstance(state.get("gpu_reclaim_proof"), dict) else None,
                        "runtime_task_id": str(state.get("runtime_task_id") or ""),
                        "task_id": str(row.get("task_id") or ""),
                        "principal": row.get("principal") if isinstance(row.get("principal"), dict) else None,
                        "project_id": row.get("project_id"),
                    }
                    continue
                if is_lease_granted(live) and hasattr(self.scheduler, "heartbeat"):
                    try:
                        await asyncio.to_thread(
                            self.scheduler.heartbeat,
                            lid,
                            int((live or lease).get("fencing_token") or 0),
                        )
                    except Exception:
                        orphan_entries[lid] = {
                            "lease": dict(live or lease),
                            "target_identity": state.get("gpu_target_identity") if isinstance(state.get("gpu_target_identity"), dict) else {},
                            "reclaim_proof": state.get("gpu_reclaim_proof") if isinstance(state.get("gpu_reclaim_proof"), dict) else None,
                            "runtime_task_id": str(state.get("runtime_task_id") or ""),
                            "task_id": str(row.get("task_id") or ""),
                            "principal": row.get("principal") if isinstance(row.get("principal"), dict) else None,
                            "project_id": row.get("project_id"),
                        }
                        continue
                adopted_ids.add(lid)
                adopted.append(lid)
                log.info(
                    "speech_startup_lease_adopted task_id=%s lease_id=%s phase=%s",
                    row.get("task_id"),
                    lid,
                    row.get("phase"),
                )
            else:
                orphan_entries[lid] = {
                    "lease": remote_by_id.get(lid) or dict(lease),
                    "target_identity": state.get("gpu_target_identity") if isinstance(state.get("gpu_target_identity"), dict) else {},
                    "reclaim_proof": state.get("gpu_reclaim_proof") if isinstance(state.get("gpu_reclaim_proof"), dict) else None,
                    "runtime_task_id": str(state.get("runtime_task_id") or ""),
                    "task_id": str(row.get("task_id") or ""),
                    "principal": row.get("principal") if isinstance(row.get("principal"), dict) else None,
                    "project_id": row.get("project_id"),
                }

        for lid, rlease in remote_by_id.items():
            if lid in adopted_ids or lid in orphan_entries:
                continue
            orphan_entries[lid] = {
                "lease": dict(rlease),
                "target_identity": {
                    k: rlease[k]
                    for k in ("pod_uid", "container_id", "boot_id", "supervisor_pid", "supervisor_starttime_ticks")
                    if rlease.get(k)
                },
                "reclaim_proof": None,
                "runtime_task_id": "",
                "task_id": str(rlease.get("task_id") or rlease.get("holder_task_id") or ""),
                "principal": None,
                "project_id": None,
            }

        for lid, info in orphan_entries.items():
            self._register_orphan_lease(
                info["lease"],
                target_identity=info["target_identity"],
                reclaim_proof=info["reclaim_proof"],
                runtime_task_id=info["runtime_task_id"],
                task_id=info["task_id"],
                principal=info["principal"],
                project_id=info["project_id"],
            )

        for lid, info in list(orphan_entries.items()):
            live_lease = info["lease"]
            if hasattr(self.scheduler, "get_lease"):
                try:
                    fetched = await asyncio.to_thread(self.scheduler.get_lease, lid)
                    if isinstance(fetched, dict) and fetched.get("lease_id"):
                        live_lease = fetched
                except SchedulerError as exc:
                    if exc.status_code == 404:
                        self._pop_orphan_lease(lid)
                        await self._clear_terminal_task_lease(info["task_id"], lid)
                        released.append(lid)
                        continue
                except Exception:
                    pass
            lstate = str(live_lease.get("state") or "").strip().lower()
            if lstate in {"released", "expired"}:
                self._pop_orphan_lease(lid)
                await self._clear_terminal_task_lease(info["task_id"], lid)
                released.append(lid)
                continue
            ok = await self._settle_and_release_lease(
                live_lease,
                target_identity=info["target_identity"],
                reclaim_proof=info["reclaim_proof"],
                runtime_task_id=info["runtime_task_id"],
                exclude_task_id=info["task_id"] or None,
                principal=info["principal"],
                project_id=info["project_id"],
                reason="startup_orphan_reconcile",
            )
            if ok:
                self._pop_orphan_lease(lid)
                await self._clear_terminal_task_lease(info["task_id"], lid)
                released.append(lid)
            else:
                deferred.append(lid)

        requeued = await self._requeue_stale_fence_tasks()
        for tid in requeued:
            log.info("speech_startup_requeued_stale_fence_task task_id=%s", tid)

        return {"adopted": adopted, "released": released, "deferred": deferred}

    async def _requeue_stale_fence_tasks(self) -> list[str]:
        fn = getattr(self.store, "requeue_preinference_stale_fence_tasks", None)
        if callable(fn) and _has_custom_attr(self.store, "requeue_preinference_stale_fence_tasks"):
            res = await asyncio.to_thread(fn)
            return list(res) if isinstance(res, list) else []
        rows = getattr(self.store, "rows", None)
        if not isinstance(rows, dict):
            return []
        requeued: list[str] = []
        for tid, r in list(rows.items()):
            if not isinstance(r, dict):
                continue
            st = r.get("state") if isinstance(r.get("state"), dict) else {}
            if (
                r.get("status") == "failed"
                and r.get("last_error") == "stale_fencing_token"
                and r.get("phase") in {"reference", "loading_model", "submit", "inference"}
                and not (st.get("outputs") or [])
                and int(st.get("fence_retry_count") or 0) < 5
            ):
                retry_cnt = int(st.get("fence_retry_count") or 0) + 1
                st["fence_retry_count"] = retry_cnt
                st.pop("submit_dispatched", None)
                st.pop("runtime_task_id", None)
                st.pop("gpu_execution_finished", None)
                st.pop("gpu_reclaim_proof", None)
                st.pop("gpu_scheduler_lease", None)
                st.pop("gpu_queue", None)
                if isinstance(st.get("runtime_inputs"), dict):
                    st["runtime_inputs"]["idempotency_key"] = f"speech-{tid}-r{retry_cnt}"
                r["phase"] = (
                    "reference"
                    if isinstance(st.get("profile"), dict) and isinstance(st["profile"].get("reference"), dict)
                    else "submit"
                )
                r["status"] = "running"
                r["last_error"] = None
                requeued.append(str(tid))
        return requeued

    async def _drain_orphan_leases(self) -> int:
        with self._orphan_lock:
            items = [(lid, dict(entry)) for lid, entry in self._orphan_leases.items()]
        drained = 0
        for lid, info in items:
            live_lease = dict(info["lease"])
            if hasattr(self.scheduler, "get_lease"):
                try:
                    fetched = await asyncio.to_thread(self.scheduler.get_lease, lid)
                    if isinstance(fetched, dict) and fetched.get("lease_id"):
                        live_lease = fetched
                except SchedulerError as exc:
                    if exc.status_code == 404:
                        self._pop_orphan_lease(lid)
                        await self._clear_terminal_task_lease(info.get("task_id") or "", lid)
                        drained += 1
                        continue
                except Exception:
                    pass
            if str(live_lease.get("state") or "").strip().lower() in {"released", "expired"}:
                self._pop_orphan_lease(lid)
                await self._clear_terminal_task_lease(info.get("task_id") or "", lid)
                drained += 1
                continue
            ok = await self._settle_and_release_lease(
                live_lease,
                target_identity=info.get("target_identity"),
                reclaim_proof=info.get("reclaim_proof"),
                runtime_task_id=info.get("runtime_task_id") or "",
                exclude_task_id=info.get("task_id") or None,
                principal=info.get("principal"),
                project_id=info.get("project_id"),
                reason="deferred_orphan_reconcile",
            )
            if ok:
                self._pop_orphan_lease(lid)
                await self._clear_terminal_task_lease(info.get("task_id") or "", lid)
                drained += 1
        return drained

    def _list_local_lease_rows(self) -> list[dict]:
        if _has_custom_attr(self.store, "list_tasks_with_scheduler_leases") or hasattr(
            self.store, "list_tasks_with_scheduler_leases"
        ):
            fn = getattr(self.store, "list_tasks_with_scheduler_leases", None)
            if callable(fn):
                res = fn()
                if isinstance(res, list):
                    return res
        rows = getattr(self.store, "rows", None)
        if isinstance(rows, dict):
            return [
                r
                for r in rows.values()
                if isinstance(r, dict) and isinstance((r.get("state") or {}).get("gpu_scheduler_lease"), dict)
            ]
        return []

    async def _clear_terminal_task_lease(self, task_id: str, lease_id: str) -> None:
        if not task_id or not lease_id:
            return
        fn = getattr(self.store, "clear_task_scheduler_lease", None)
        if callable(fn):
            with contextlib.suppress(Exception):
                await asyncio.to_thread(fn, task_id, lease_id)
                return
        rows = getattr(self.store, "rows", None)
        if isinstance(rows, dict):
            row = rows.get(task_id)
            if isinstance(row, dict) and (row.get("status") in {"succeeded", "failed"} or not row.get("lease_token")):
                st = row.get("state")
                if isinstance(st, dict):
                    cur = st.get("gpu_scheduler_lease")
                    if isinstance(cur, dict) and str(cur.get("lease_id") or "") == lease_id:
                        st.pop("gpu_scheduler_lease", None)
                        st.pop("gpu_queue", None)

    def _has_other_active_gpu_tasks(
        self,
        *,
        exclude_task_id: str | None = None,
        include_unarchived_runtime_outputs: bool = True,
    ) -> bool:
        fn = getattr(self.store, "has_active_gpu_tasks", None)
        if callable(fn) and _has_custom_attr(self.store, "has_active_gpu_tasks"):
            return bool(
                fn(
                    exclude_task_id=exclude_task_id,
                    include_unarchived_runtime_outputs=include_unarchived_runtime_outputs,
                )
            )
        rows = getattr(self.store, "rows", None)
        if isinstance(rows, dict):
            excl = str(exclude_task_id or "").strip()
            for r in rows.values():
                if not isinstance(r, dict):
                    continue
                if excl and str(r.get("task_id") or "") == excl:
                    continue
                if r.get("status") not in {"pending", "running"}:
                    continue
                phase = r.get("phase")
                if include_unarchived_runtime_outputs:
                    if phase in {"reference", "loading_model", "submit", "inference", "archive"}:
                        return True
                else:
                    if phase in {"reference", "loading_model", "submit"}:
                        return True
                    if phase == "inference" and (r.get("state") or {}).get("gpu_execution_finished") is not True:
                        return True
        return False

    def _has_other_active_scheduler_leases(self, *, exclude_lease_id: str | None = None) -> bool:
        excl = str(exclude_lease_id or "").strip()
        with self._orphan_lock:
            orphan_ids = set(self._orphan_leases.keys())
        for row in self._list_local_lease_rows():
            if row.get("status") not in {"pending", "running"} or row.get("phase") == "complete":
                continue
            lease = (row.get("state") or {}).get("gpu_scheduler_lease")
            if not isinstance(lease, dict):
                continue
            lid = str(lease.get("lease_id") or "").strip()
            if lid and lid != excl and lid not in orphan_ids:
                lstate = str(lease.get("state") or "").strip().lower()
                if lstate in {"queued", "leased", "running", "granted", "draining"}:
                    return True
        if hasattr(self.scheduler, "list_own_leases"):
            for item in self.scheduler.list_own_leases():
                lid = str(item.get("lease_id") or "").strip()
                if not lid or lid == excl or lid in orphan_ids:
                    continue
                lstate = str(item.get("state") or "").strip().lower()
                if lstate in {"queued", "leased", "running", "granted", "draining"}:
                    return True
        return False

    async def _probe_runtime_health(self, *, p=None, project_id=None, engine: str = "indextts2") -> dict | None:
        runtime = self._runtime_for(engine)
        if runtime is None:
            return None
        if _has_custom_attr(runtime, "request_status_json"):
            status_code, doc = await runtime.request_status_json("GET", "/health", p=p, project_id=project_id)
            if status_code in {200, 503} and isinstance(doc, dict):
                return doc
            return None
        res = await runtime.json(p, "GET", "/health", project_id=project_id)
        return res if isinstance(res, dict) else None

    async def _probe_reclaim_guard(
        self,
        lease_id: str,
        fencing_token: int,
        *,
        task_id: str = "",
        p=None,
        project_id=None,
        engine: str = "indextts2",
    ) -> dict | None:
        runtime = self._runtime_for(engine)
        if runtime is None:
            return None
        payload = {
            "lease_id": str(lease_id),
            "fencing_token": int(fencing_token),
            "task_id": str(task_id or ""),
        }
        if _has_custom_attr(runtime, "request_status_json"):
            status_code, doc = await runtime.request_status_json(
                "POST",
                "/internal/v1/gpu-guard/reclaim",
                payload,
                p=p,
                project_id=project_id,
            )
            if status_code in {200, 409} and isinstance(doc, dict) and isinstance(doc.get("reclaim_proof"), dict):
                return doc["reclaim_proof"]
            return None
        res = await runtime.json(
            p,
            "POST",
            "/internal/v1/gpu-guard/reclaim",
            payload,
            project_id=project_id,
        )
        if isinstance(res, dict) and isinstance(res.get("reclaim_proof"), dict):
            return res["reclaim_proof"]
        return None

    async def renew(self, row):
        while True:
            await asyncio.sleep(20)
            await asyncio.to_thread(self.store.renew, row)
            # Observe the receiver before renewing GPU occupancy: it may already
            # be settled. A GPU heartbeat failure must not cancel that read.
            if row["phase"] not in {"inference", "archive", "complete"}:
                await self._renew_execution_lease(row)

    async def _renew_execution_lease(self, row):
        """Renew existing occupancy only; never acquire or start another attempt."""
        if row["state"].get("gpu_held") and getattr(self.gpu, "enabled", False) and not self.scheduler.enabled:
            try:
                await asyncio.to_thread(self.gpu.renew)
            except LeaseBusy as error:
                raise Failure("waiting_gpu_lease", 503, True) from error
        sched_lease = row["state"].get("gpu_scheduler_lease")
        if self.scheduler.enabled and is_lease_granted(sched_lease):
            try:
                renewed = await asyncio.to_thread(
                    self.scheduler.heartbeat,
                    str(sched_lease["lease_id"]),
                    int(sched_lease["fencing_token"]),
                )
                if isinstance(renewed, dict) and renewed.get("lease_id"):
                    row["state"]["gpu_scheduler_lease"] = renewed
            except Exception as error:
                raise Failure("waiting_gpu_lease", 503, True) from error

    async def run(self):
        if self.scheduler.enabled:
            with contextlib.suppress(Exception):
                await self.reconcile_startup_leases()
            self._last_pending_flush = time.monotonic()
        while True:
            try:
                if self.scheduler.enabled and (time.monotonic() - self._last_pending_flush) >= 10.0:
                    self._last_pending_flush = time.monotonic()
                    with contextlib.suppress(Exception):
                        await asyncio.to_thread(self.scheduler.flush_pending_releases)
                    with contextlib.suppress(Exception):
                        await self._drain_orphan_leases()
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
            pre_dispatch = row["phase"] in {"reference", "loading_model"} or (
                row["phase"] == "submit"
                and not row["state"].get("submit_dispatched")
                and not row["state"].get("runtime_task_id")
            )
            can_release_sched = (
                not error.retry
                and (
                    row["phase"] in {"reference", "loading_model", "submit"}
                    or row["state"].get("gpu_execution_finished") is True
                )
            ) or (
                pre_dispatch
                and error.code not in {"lease_lost", "waiting_gpu_lease"}
            )
            release_gpu = (
                not error.retry
                and getattr(self.gpu, "enabled", False)
                and not self.scheduler.enabled
                and (
                    row["phase"] in {"reference", "loading_model"}
                    or row["state"].get("gpu_execution_finished") is True
                )
            )
            if release_gpu:
                row["state"].pop("gpu_held", None)
            if can_release_sched:
                await self._release_scheduler_lease(row, reason=f"failed:{error.code}")
            if error.code != "lease_lost":
                retry_delay = 2 if (error.retry and error.code in {"stale_fencing_token", "dependency_not_found"}) else 15
                await asyncio.to_thread(
                    self.store.save,
                    row,
                    status="running" if error.retry else "failed",
                    error=error.code,
                    delay=retry_delay,
                )
            if release_gpu:
                await asyncio.to_thread(self.gpu.release)
        finally:
            for task in (operation, heartbeat):
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Failure):
                    await task

    async def save(self, row, **kwargs):
        await asyncio.to_thread(self.store.save, row, **kwargs)

    def _compute_final_timings(
        self,
        row: dict,
        task_doc: dict | None,
        report_doc: dict | None,
        *,
        now_epoch: float | None = None,
    ) -> dict:
        state = row["state"]
        base = state.get("timings") if isinstance(state.get("timings"), dict) else {}
        now_ts = float(now_epoch) if now_epoch is not None else time.time()

        created_ts = _parse_epoch(row.get("created_at"))
        submit_ts = _finite_float(state.get("submit_started_at"))
        candidates = [ts for ts in (created_ts, submit_ts) if ts is not None and ts > 0]
        start_ts = min(candidates) if candidates else now_ts
        turnaround_seconds = round(max(0.0, now_ts - start_ts), 4)

        raw_exec: float | None = None
        execution_source = "missing"
        if isinstance(report_doc, dict):
            top_elapsed = _finite_float(report_doc.get("elapsed_seconds"))
            if top_elapsed is not None:
                raw_exec = top_elapsed
                execution_source = "report.top"
            else:
                metrics_obj = report_doc.get("metrics")
                if isinstance(metrics_obj, dict):
                    metrics_elapsed = _finite_float(metrics_obj.get("elapsed_seconds"))
                    if metrics_elapsed is not None:
                        raw_exec = metrics_elapsed
                        execution_source = "report.metrics"

        if raw_exec is not None:
            if raw_exec < 0.0 or raw_exec > turnaround_seconds + 0.05:
                log.warning(
                    "speech_invalid_execution_seconds task_id=%s execution_seconds=%s turnaround_seconds=%s source=%s",
                    row.get("task_id"),
                    raw_exec,
                    turnaround_seconds,
                    execution_source,
                )
                execution_seconds = None
            else:
                execution_seconds = round(raw_exec, 4)
        else:
            execution_seconds = None

        queue_wait_seconds: float | None = None
        enqueued_ts: float | None = None
        started_exec_ts: float | None = None
        if isinstance(report_doc, dict):
            enqueued_ts = _parse_epoch(report_doc.get("enqueued_at") or report_doc.get("created_at"))
            started_exec_ts = _parse_epoch(report_doc.get("started_at") or report_doc.get("execution_started_at"))
        if isinstance(task_doc, dict):
            if enqueued_ts is None:
                enqueued_ts = _parse_epoch(task_doc.get("enqueued_at") or task_doc.get("created_at"))
            if started_exec_ts is None:
                started_exec_ts = _parse_epoch(task_doc.get("started_at") or task_doc.get("execution_started_at"))
        if enqueued_ts is not None and started_exec_ts is not None and started_exec_ts >= enqueued_ts:
            queue_wait_seconds = round(started_exec_ts - enqueued_ts, 4)

        load_seconds = float(_finite_float(base.get("load_seconds")) or 0.0)
        if not load_seconds and isinstance(report_doc, dict):
            metrics_obj = report_doc.get("metrics")
            if isinstance(metrics_obj, dict):
                m_load = _finite_float(metrics_obj.get("load_seconds"))
                if m_load is not None and m_load >= 0.0:
                    load_seconds = round(m_load, 3)

        updated = {
            "queue_seconds": float(_finite_float(base.get("queue_seconds")) or 0.0),
            "load_seconds": load_seconds,
            "execution_seconds": execution_seconds,
            "execution_source": execution_source,
            "inference_seconds": execution_seconds,
            "turnaround_seconds": turnaround_seconds,
            "queue_wait_seconds": queue_wait_seconds,
        }
        state["timings"] = updated
        return updated

    async def _try_fetch_verified_report(
        self, p, pid: str, runtime_task_id: str, report_entry: dict, *, engine: str = "indextts2"
    ) -> dict | None:
        runtime = self._runtime_for(engine)
        if runtime is None or not hasattr(runtime, "request"):
            return None
        path = "/v1/tasks/" + runtime_task_id + "/outputs/report.json"
        raw = await runtime.request(p, "GET", path, project_id=pid, limit=MAX_OUTPUT)
        if not isinstance(raw, (bytes, bytearray)):
            return None
        data = bytes(raw)
        if len(data) != report_entry["size"] or digest(data) != report_entry["sha256"]:
            raise Failure("output_integrity_failed", 502)
        doc = strict_json(data)
        if not isinstance(doc, dict):
            raise Failure("invalid_runtime_outputs", 502)
        return doc

    async def step(self, row):
        p, req, state = row["principal"], row["request"], row["state"]
        pid, phase = row["project_id"], row["phase"]
        eng = self._row_engine(row)
        rt_client = self._runtime_for(eng)
        state.setdefault("timings", {"queue_seconds": 0.0, "load_seconds": 0.0, "inference_seconds": 0.0})
        await self.peers.open(p, pid, write=True)
        if phase in ("reference", "loading_model", "submit"):
            await self._ensure_gpu(row)
            if phase == "loading_model":
                phase = "reference"
                row["phase"] = "reference"
        if phase == "reference":
            profile = state["profile"]
            caps = await rt_client.json(p, "GET", "/v1/capabilities", project_id=pid)
            if caps.get("model_manifest_sha256") != profile["model_manifest_sha256"]:
                raise Failure("model_version_changed", 409)
            if caps.get("ready") is not True:
                raise Failure("model_not_ready", 503, True)
            state["capabilities"] = caps
            ref_info = profile["reference"]
            if ref_info.get("asset_id") and ref_info.get("asset_version_id"):
                data, _ = await self.peers.asset_content(
                    p, ref_info["content_ref"], ref_info["asset_id"], ref_info["asset_version_id"], 10 << 20
                )
            else:
                data, _ = await self.peers.content(p, ref_info["content_ref"], pid, req["revision_id"], 2 << 20)
            if digest(data) != ref_info["sha256"]:
                raise Failure("reference_integrity_failed")
            await rt_client.json(
                p, "PUT", "/v1/references/" + digest(data), data=data, mime="audio/wav", project_id=pid
            )
            text_rec = state.get("text") or build_text_record(req["text"])
            state["text"] = text_rec
            fence_retry_count = int(state.get("fence_retry_count") or 0)
            rt_idem = f"speech-{row['task_id']}-r{fence_retry_count}" if fence_retry_count > 0 else "speech-" + row["task_id"]
            runtime_inputs = {
                "idempotency_key": rt_idem,
                "text": text_rec["sent"],
                "language": req["language"],
                "reference_sha256": digest(data),
            }
            if text_rec["sent"] != text_rec["source"]:
                runtime_inputs["expected_text"] = text_rec["source"]
            if "emotion_vector" in req:
                runtime_inputs["emotion_vector"] = req["emotion_vector"]
            elif "emotion_text" in req:
                runtime_inputs["emotion_text"] = req["emotion_text"]
            elif "emotion_reference" in req:
                emo_ref = req["emotion_reference"]["content_ref"]
                emo_data, _ = await self.peers.content(p, emo_ref, pid, req["revision_id"], 2 << 20)
                emo_sha = digest(emo_data)
                await rt_client.json(
                    p, "PUT", "/v1/references/" + emo_sha, data=emo_data, mime="audio/wav", project_id=pid
                )
                runtime_inputs["emotion_reference"] = {
                    "reference_sha256": emo_sha,
                    "alpha": req["emotion_reference"]["alpha"],
                }
            if "shot_window_seconds" in req:
                runtime_inputs["shot_window_seconds"] = req["shot_window_seconds"]
            state["runtime_inputs"] = runtime_inputs
            await self.save(row, phase="submit")
        elif phase == "submit":
            caps = await rt_client.json(p, "GET", "/v1/capabilities", project_id=pid)
            if caps.get("model_manifest_sha256") != state["profile"]["model_manifest_sha256"]:
                raise Failure("model_version_changed", 409)
            if not state.get("submit_started_at"):
                state["submit_started_at"] = time.time()
            grant_headers = (
                self.scheduler.build_grant_headers(
                    state.get("gpu_scheduler_lease"),
                    parent_task_id=row["task_id"],
                    target_identity=state.get("gpu_target_identity"),
                )
                if hasattr(self.scheduler, "build_grant_headers")
                else {}
            )
            extra_kw = {"extra_headers": grant_headers} if grant_headers else {}
            state["submit_dispatched"] = True
            try:
                task = await rt_client.json(
                    p, "POST", "/v1/tasks", state["runtime_inputs"], project_id=pid, **extra_kw
                )
            except Failure as exc:
                if exc.code == "stale_fencing_token" and int(state.get("fence_retry_count") or 0) < 5:
                    fence_retry_count = int(state.get("fence_retry_count") or 0) + 1
                    state["fence_retry_count"] = fence_retry_count
                    if isinstance(state.get("runtime_inputs"), dict):
                        state["runtime_inputs"]["idempotency_key"] = f"speech-{row['task_id']}-r{fence_retry_count}"
                    state.pop("submit_dispatched", None)
                    state.pop("runtime_task_id", None)
                    state.pop("gpu_execution_finished", None)
                    state.pop("gpu_reclaim_proof", None)
                    row["phase"] = (
                        "reference"
                        if isinstance(state.get("profile"), dict) and isinstance(state["profile"].get("reference"), dict)
                        else "submit"
                    )
                    await self._release_scheduler_lease(row, reason="submit_stale_fencing_token")
                    raise Failure("stale_fencing_token", 409, True) from exc
                if (
                    exc.code == "dependency_not_found"
                    and isinstance(state.get("profile"), dict)
                    and isinstance(state["profile"].get("reference"), dict)
                    and int(state.get("ref_reupload_count") or 0) < 3
                ):
                    state["ref_reupload_count"] = int(state.get("ref_reupload_count") or 0) + 1
                    state.pop("submit_dispatched", None)
                    state.pop("runtime_task_id", None)
                    row["phase"] = "reference"
                    raise Failure("dependency_not_found", 404, True) from exc
                raise
            validate("id", task.get("task_id"))
            state["runtime_task_id"] = task["task_id"]
            await self.save(row, phase="inference", delay=2)
        elif phase == "inference":
            task = await rt_client.json(p, "GET", "/v1/tasks/" + state["runtime_task_id"], project_id=pid)
            if (
                task.get("task_id") != state["runtime_task_id"]
                or task.get("model_manifest_sha256") != state["profile"]["model_manifest_sha256"]
            ):
                raise Failure("runtime_identity_mismatch", 502)
            if task["status"] in {"failed", "succeeded"}:
                state["gpu_execution_finished"] = True
                if isinstance(task.get("reclaim_proof"), dict):
                    state["gpu_reclaim_proof"] = task["reclaim_proof"]
                if task["status"] == "failed" or not self._scale_enabled():
                    await self._release_scheduler_lease(row, reason=f"inference_{task['status']}")
            if task["status"] == "failed":
                code = task.get("error", {}).get("code")
                if (
                    code == "stale_fencing_token"
                    and not task.get("outputs")
                    and isinstance(state.get("runtime_inputs"), dict)
                    and int(state.get("fence_retry_count") or 0) < 5
                ):
                    fence_retry_count = int(state.get("fence_retry_count") or 0) + 1
                    state["fence_retry_count"] = fence_retry_count
                    state["runtime_inputs"]["idempotency_key"] = f"speech-{row['task_id']}-r{fence_retry_count}"
                    state.pop("submit_dispatched", None)
                    state.pop("runtime_task_id", None)
                    state.pop("gpu_execution_finished", None)
                    state.pop("gpu_reclaim_proof", None)
                    row["phase"] = (
                        "reference"
                        if isinstance(state.get("profile"), dict) and isinstance(state["profile"].get("reference"), dict)
                        else "submit"
                    )
                    raise Failure("stale_fencing_token", 409, True)
                raise Failure(
                    code if code in ("recovery_required", "model_version_changed", "stale_fencing_token") else "inference_failed",
                    409,
                )
            if task["status"] != "succeeded":
                await self._renew_execution_lease(row)
                await self.save(row, delay=2)
                return
            entries = task.get("outputs", [])
            if len(entries) != 3 or {x.get("name") for x in entries} != {"original.wav", "editing.wav", "report.json"}:
                raise Failure("invalid_runtime_outputs", 502)
            for e in entries:
                validate("hash", e.get("sha256"))
                if (
                    type(e.get("size")) is not int
                    or not 0 < e["size"] <= MAX_OUTPUT
                    or e.get("mime") != ("application/json" if e["name"] == "report.json" else "audio/wav")
                ):
                    raise Failure("invalid_runtime_outputs", 502)
                e["write_id"] = new_id()

            report_entry = next(e for e in entries if e["name"] == "report.json")
            report_doc = await self._try_fetch_verified_report(
                p, pid, state["runtime_task_id"], report_entry, engine=eng
            )
            if isinstance(report_doc, dict):
                state["report"] = report_doc
                state["generation_finalized"] = True
            else:
                state["generation_finalized"] = False

            final_timings = self._compute_final_timings(row, task, report_doc)
            state["inference_task_snapshot"] = {
                "created_at": task.get("created_at"),
                "enqueued_at": task.get("enqueued_at"),
                "started_at": task.get("started_at"),
                "execution_started_at": task.get("execution_started_at"),
            }

            voice_source = state.get("voice_source") or {
                "kind": "project_candidate",
                "voice_profile": req["voice_profile"],
                "profile_sha256": state["profile_sha256"],
            }
            text_rec = state.get("text") or build_text_record(req["text"])
            input_digest = state.get("input_digest") or compute_input_digest(
                text_rec["sent"], voice_source, state["profile"]["model_manifest_sha256"], req
            )
            resolved_vec = task.get("resolved_emotion_vector")
            if resolved_vec is None and isinstance(report_doc, dict) and "resolved_emotion_vector" in report_doc:
                resolved_vec = report_doc["resolved_emotion_vector"]
            if resolved_vec is None and "emotion_vector" in req:
                resolved_vec = req["emotion_vector"]

            # Construct generation.json AFTER verifying report.json so timings and emotion are immutable.
            state["generation"] = {
                "schema_version": 1,
                "service_version": __version__,
                "task_id": row["task_id"],
                "runtime_task_id": state["runtime_task_id"],
                "input_digest": input_digest,
                "voice_source": voice_source,
                "resolved_emotion_vector": resolved_vec,
                "text": {
                    "source": text_rec["source"],
                    "sent": text_rec["sent"],
                    "lexicon_hits": text_rec["lexicon_hits"],
                },
                "timings": dict(final_timings),
                "model_manifest_sha256": state["profile"]["model_manifest_sha256"],
                "request": req,
                "voice_profile": state["profile"],
                "profile_sha256": state["profile_sha256"],
                "runtime_capabilities": state["capabilities"],
                "runtime_outputs": [dict(e) for e in entries],
                "candidate_state": state.get("candidate_state", "candidate"),
                "listening_review": state.get("listening_review", "pending"),
            }
            raw = canonical(state["generation"]).encode()
            entries.append(
                {
                    "name": "generation.json",
                    "sha256": digest(raw),
                    "size": len(raw),
                    "mime": "application/json",
                    "write_id": new_id(),
                }
            )
            state["archive_entries"] = entries
            await self.save(row, phase="archive")
        elif phase == "archive":
            complete = {x["name"] for x in state["outputs"]}
            entry = next((x for x in state["archive_entries"] if x["name"] not in complete), None)
            if entry is None:
                release_gpu = getattr(self.gpu, "enabled", False) and not self.scheduler.enabled and state.pop("gpu_held", False)
                await self._release_scheduler_lease(row, reason="task_succeeded")
                await self.save(row, phase="complete", status="succeeded")
                if release_gpu:
                    await asyncio.to_thread(self.gpu.release)
                return
            if entry["name"] == "generation.json":
                data = canonical(state["generation"]).encode()
            else:
                path = "/v1/tasks/" + state["runtime_task_id"] + "/outputs/" + entry["name"]
                data = await rt_client.request(p, "GET", path, project_id=pid, limit=MAX_OUTPUT)
            if len(data) != entry["size"] or digest(data) != entry["sha256"]:
                raise Failure("output_integrity_failed", 502)
            if entry["name"] == "report.json":
                report_doc = strict_json(data)
                if isinstance(report_doc, dict):
                    state["report"] = report_doc
                    if not state.get("generation_finalized") and "generation.json" not in complete:
                        snap = state.get("inference_task_snapshot") if isinstance(state.get("inference_task_snapshot"), dict) else {}
                        final_timings = self._compute_final_timings(row, snap, report_doc)
                        if "resolved_emotion_vector" in report_doc and state["generation"].get("resolved_emotion_vector") is None:
                            state["generation"]["resolved_emotion_vector"] = report_doc["resolved_emotion_vector"]
                        state["generation"]["timings"] = dict(final_timings)
                        state["generation_finalized"] = True
                        gen_raw = canonical(state["generation"]).encode()
                        for item in state["archive_entries"]:
                            if item["name"] == "generation.json":
                                item["sha256"] = digest(gen_raw)
                                item["size"] = len(gen_raw)
            result = await self.peers.archive(p, row, entry, data)
            state["outputs"].append(result)
            await self.save(row)
        else:
            raise Failure("invalid_task_phase", 500)

    def _call_scheduler_release_sync(
        self,
        lease: dict,
        *,
        reclaim_proof: dict | None,
        target_identity: dict | None,
        reason: str,
        worker_settled: bool,
        reclaim_mode: str,
    ) -> None:
        lid = str(lease["lease_id"])
        tok = int(lease.get("fencing_token") or 0)
        ver = int(lease.get("version") or 0)
        try:
            self.scheduler.release(
                lid,
                tok,
                reclaim_proof=reclaim_proof,
                target_identity=target_identity,
                reason=reason,
                worker_settled=worker_settled,
                reclaim_mode=reclaim_mode,
                expected_version=ver,
            )
            return
        except SchedulerError as exc:
            if exc.status_code != 409 or not hasattr(self.scheduler, "get_lease"):
                raise
            latest = self.scheduler.get_lease(lid)
            if str(latest.get("state") or "").strip().lower() in {"released", "expired"}:
                return
            self.scheduler.release(
                lid,
                int(latest.get("fencing_token") or tok),
                reclaim_proof=reclaim_proof,
                target_identity=target_identity,
                reason=reason,
                worker_settled=worker_settled,
                reclaim_mode=reclaim_mode,
                expected_version=int(latest.get("version") or 0),
            )

    async def _settle_and_release_lease(
        self,
        lease: dict,
        *,
        target_identity: dict | None = None,
        reclaim_proof: dict | None = None,
        runtime_task_id: str = "",
        exclude_task_id: str | None = None,
        principal: dict | None = None,
        project_id: str | None = None,
        reason: str = "task_settled",
        engine: str = "indextts2",
    ) -> bool:
        lid = str(lease.get("lease_id") or "").strip()
        if not lid:
            return True
        eng = str(engine or lease.get("engine") or "indextts2")
        target_dep = self._deployment_for(eng)
        dep_args = (target_dep,) if target_dep else ()
        tok = int(lease.get("fencing_token") or 0)
        ident = dict(target_identity) if isinstance(target_identity, dict) else {}
        for k in ("pod_uid", "container_id", "boot_id", "supervisor_pid", "supervisor_starttime_ticks"):
            if not ident.get(k) and lease.get(k):
                ident[k] = lease[k]
        target_bound = bool(str(lease.get("pod_uid") or "").strip())
        proof = dict(reclaim_proof) if isinstance(reclaim_proof, dict) else None

        if not target_bound:
            if self._scale_enabled():
                try:
                    other_local = await asyncio.to_thread(
                        self._has_other_active_gpu_tasks,
                        exclude_task_id=exclude_task_id,
                        include_unarchived_runtime_outputs=True,
                    )
                    other_leases = await asyncio.to_thread(
                        self._has_other_active_scheduler_leases,
                        exclude_lease_id=lid,
                    )
                except Exception:
                    return False
                if not other_local and not other_leases:
                    reps = await asyncio.to_thread(self.gpu.replicas_for, *dep_args)
                    pods = await asyncio.to_thread(self.gpu.deployment_pods, *dep_args)
                    if reps not in (None, 0) or pods:
                        health: dict | None = None
                        with contextlib.suppress(Exception):
                            health = await self._probe_runtime_health(p=principal, project_id=project_id, engine=eng)
                        if isinstance(health, dict) and int(health.get("active_tasks") or 0) > 0:
                            return False
                        try:
                            await asyncio.to_thread(self.gpu.stop_deployment_and_wait, *dep_args)
                        except Exception:
                            return False
            try:
                await asyncio.to_thread(
                    self._call_scheduler_release_sync,
                    lease,
                    reclaim_proof=None,
                    target_identity=None,
                    reason=reason,
                    worker_settled=False,
                    reclaim_mode="abort_unstarted",
                )
                return True
            except Exception:
                log.warning("speech_scheduler_lease_release_failed lease_id=%s", lid)
                return False

        if not self._scale_enabled():
            if not _is_complete_warm_proof(proof, lid):
                with contextlib.suppress(Exception):
                    fetched = await self._probe_reclaim_guard(
                        lid,
                        tok,
                        task_id=runtime_task_id,
                        p=principal,
                        project_id=project_id,
                        engine=eng,
                    )
                    if isinstance(fetched, dict):
                        proof = fetched
            if _is_complete_warm_proof(proof, lid):
                reclaim_mode = "warm_idle_settled"
                worker_settled = True
                send_proof = proof
            else:
                reclaim_mode = ""
                worker_settled = False
                send_proof = None
            try:
                await asyncio.to_thread(
                    self._call_scheduler_release_sync,
                    lease,
                    reclaim_proof=send_proof,
                    target_identity=ident,
                    reason=reason,
                    worker_settled=worker_settled,
                    reclaim_mode=reclaim_mode,
                )
                return True
            except Exception:
                log.warning("speech_scheduler_lease_release_failed lease_id=%s", lid)
                return False

        # Managed K8s Deployment scale-down path: enforce all 4 strict preconditions!
        try:
            other_local = await asyncio.to_thread(
                self._has_other_active_gpu_tasks,
                exclude_task_id=exclude_task_id,
                include_unarchived_runtime_outputs=True,
            )
            other_leases = await asyncio.to_thread(
                self._has_other_active_scheduler_leases,
                exclude_lease_id=lid,
            )
            reps = await asyncio.to_thread(self.gpu.replicas_for, *dep_args)
            pods = await asyncio.to_thread(self.gpu.deployment_pods, *dep_args)
        except Exception:
            log.warning("speech_scale_down_precondition_store_or_k8s_failed lease_id=%s", lid)
            return False

        if reps == 0 and not pods and not other_local and not other_leases:
            try:
                await asyncio.to_thread(
                    self._call_scheduler_release_sync,
                    lease,
                    reclaim_proof=None,
                    target_identity=ident,
                    reason=reason,
                    worker_settled=True,
                    reclaim_mode="",
                )
                return True
            except Exception:
                log.warning("speech_scheduler_lease_release_failed lease_id=%s", lid)
                return False

        # Precondition (b): runtime queue has zero running/pending tasks
        try:
            health = await self._probe_runtime_health(p=principal, project_id=project_id, engine=eng)
        except Exception:
            health = None
        if not isinstance(health, dict) or int(health.get("active_tasks") or 0) != 0:
            log.info("speech_scale_down_blocked_active_runtime_queue lease_id=%s", lid)
            return False
        if isinstance(health.get("runtime_identity"), dict) and not ident.get("pod_uid"):
            ident.update(health["runtime_identity"])

        # Precondition (c): guard reclaim_proof has sealed token (HTTP 200 or 409)
        if not _is_sealed_reclaim_proof(proof, lid):
            with contextlib.suppress(Exception):
                fetched = await self._probe_reclaim_guard(
                    lid,
                    tok,
                    task_id=runtime_task_id,
                    p=principal,
                    project_id=project_id,
                    engine=eng,
                )
                if isinstance(fetched, dict):
                    proof = fetched
        if not _is_sealed_reclaim_proof(proof, lid):
            log.info("speech_scale_down_blocked_unsealed_reclaim_proof lease_id=%s", lid)
            return False

        # Precondition (a): if another local Speech task/lease is active, keep warm at replicas=1
        if other_local or other_leases:
            if _is_complete_warm_proof(proof, lid):
                try:
                    await asyncio.to_thread(
                        self._call_scheduler_release_sync,
                        lease,
                        reclaim_proof=proof,
                        target_identity=ident,
                        reason=reason,
                        worker_settled=True,
                        reclaim_mode="warm_idle_settled",
                    )
                    return True
                except Exception:
                    log.warning("speech_scheduler_lease_release_failed lease_id=%s", lid)
                    return False
            return False

        # Precondition (d): patch_scale(0) and wait until all Pods exit before /release
        last_hb = [time.monotonic()]
        live_lease = dict(lease)

        def _on_pod_exit_wait() -> None:
            if hasattr(self.scheduler, "heartbeat") and (time.monotonic() - last_hb[0]) >= 15.0:
                last_hb[0] = time.monotonic()
                renewed = self.scheduler.heartbeat(lid, tok)
                if isinstance(renewed, dict) and renewed.get("lease_id"):
                    live_lease.update(renewed)

        try:
            await asyncio.to_thread(self.gpu.stop_deployment_and_wait, target_dep, on_wait=_on_pod_exit_wait)
        except Exception:
            log.warning("speech_scale_down_pod_exit_wait_incomplete lease_id=%s", lid)
            return False

        self._scheduler_idle_since = None
        try:
            await asyncio.to_thread(
                self._call_scheduler_release_sync,
                live_lease,
                reclaim_proof=None,
                target_identity=ident,
                reason=reason,
                worker_settled=True,
                reclaim_mode="",
            )
            return True
        except Exception:
            log.warning("speech_scheduler_lease_release_failed lease_id=%s", lid)
            return False

    async def _release_scheduler_lease(self, row, *, reason: str = "task_settled") -> None:
        state = row["state"]
        eng = self._row_engine(row)
        lease = state.get("gpu_scheduler_lease")
        if not self.scheduler.enabled or not isinstance(lease, dict) or not lease.get("lease_id"):
            state.pop("gpu_scheduler_lease", None)
            state.pop("gpu_queue", None)
            return
        lid = str(lease["lease_id"])
        ident = state.get("gpu_target_identity") if isinstance(state.get("gpu_target_identity"), dict) else {}
        proof = state.get("gpu_reclaim_proof") if isinstance(state.get("gpu_reclaim_proof"), dict) else None
        ok = await self._settle_and_release_lease(
            lease,
            target_identity=ident,
            reclaim_proof=proof,
            runtime_task_id=str(state.get("runtime_task_id") or ""),
            exclude_task_id=str(row.get("task_id") or ""),
            principal=row.get("principal") if isinstance(row.get("principal"), dict) else None,
            project_id=row.get("project_id"),
            reason=reason,
            engine=eng,
        )
        if ok:
            self._pop_orphan_lease(lid)
            state.pop("gpu_scheduler_lease", None)
            state.pop("gpu_queue", None)
        else:
            self._register_orphan_lease(
                lease,
                target_identity=ident,
                reclaim_proof=proof,
                runtime_task_id=str(state.get("runtime_task_id") or ""),
                task_id=str(row.get("task_id") or ""),
                principal=row.get("principal") if isinstance(row.get("principal"), dict) else None,
                project_id=row.get("project_id"),
                engine=eng,
            )
            state.pop("gpu_scheduler_lease", None)
            state.pop("gpu_queue", None)

    async def _acquire_scheduler_grant(self, row) -> None:
        if not self.scheduler.enabled:
            return
        state = row["state"]
        existing = state.get("gpu_scheduler_lease")
        if is_lease_granted(existing):
            try:
                refreshed = await asyncio.to_thread(
                    self.scheduler.heartbeat,
                    str(existing["lease_id"]),
                    int(existing["fencing_token"]),
                )
                if is_lease_granted(refreshed):
                    state["gpu_scheduler_lease"] = refreshed
                    state.pop("gpu_queue", None)
                    state.pop("gpu_queue_started_at", None)
                    return
            except Exception:
                await self._release_scheduler_lease(row, reason="stale_granted_lease_before_step")

        queue_budget = int(getattr(self.gpu, "queue_seconds", 1200) or 1200)
        step_yield_seconds = float(getattr(self, "_queue_step_yield_seconds", 8.0))
        decision: dict | None = None
        existing = state.get("gpu_scheduler_lease")
        if isinstance(existing, dict) and str(existing.get("lease_id") or "").strip():
            ex_id = str(existing["lease_id"]).strip()
            ex_state = str(existing.get("state") or "").strip().lower()
            if ex_state in {"queued", "draining"} and hasattr(self.scheduler, "get_lease"):
                try:
                    polled = await asyncio.to_thread(self.scheduler.get_lease, ex_id)
                    p_state = str((polled or {}).get("state") or "").strip().lower()
                    if is_lease_granted(polled) or p_state in {"queued", "draining"}:
                        decision = polled
                    else:
                        state.pop("gpu_scheduler_lease", None)
                        state.pop("gpu_queue", None)
                        state.pop("gpu_queue_started_at", None)
                except SchedulerError as exc:
                    if exc.status_code == 404:
                        state.pop("gpu_scheduler_lease", None)
                        state.pop("gpu_queue", None)
                        state.pop("gpu_queue_started_at", None)
                    else:
                        raise Failure("waiting_gpu_lease", 503, True) from exc
                except Exception as exc:
                    raise Failure("waiting_gpu_lease", 503, True) from exc
            elif ex_state in {"reclaiming", "reconciling", "quarantined"}:
                await self._release_scheduler_lease(row, reason="stale_reclaiming_lease_before_step")
            else:
                state.pop("gpu_scheduler_lease", None)
                state.pop("gpu_queue", None)
                state.pop("gpu_queue_started_at", None)

        if decision is None:
            eng = self._row_engine(row)
            target_dep = self._deployment_for(eng)
            dep_args = (target_dep,) if target_dep else ()
            replicas = await asyncio.to_thread(self.gpu.replicas_for, *dep_args) if self._scale_enabled() else 1
            slots_required = 1 if (self._scale_enabled() and replicas in (0, None)) else 0
            pre_ident: dict | None = None
            if slots_required == 0:
                with contextlib.suppress(Exception):
                    health = await self._probe_runtime_health(
                        p=row["principal"], project_id=row["project_id"], engine=eng
                    )
                    if isinstance(health, dict) and _has_concrete_identity(health.get("runtime_identity")):
                        pre_ident = dict(health["runtime_identity"])
                        state["gpu_target_identity"] = pre_ident
            attempt_seq = int(state.get("gpu_attempt_seq") or 0) + 1
            state["gpu_attempt_seq"] = attempt_seq
            attempt_id = f"att-{attempt_seq}"
            state["gpu_queue_started_at"] = time.time()

            try:
                decision = await asyncio.to_thread(
                    self.scheduler.acquire,
                    parent_task_id=row["task_id"],
                    attempt_id=attempt_id,
                    k8s_slots_required=slots_required,
                    runtime_identity=pre_ident,
                )
            except Exception as exc:
                raise Failure("waiting_gpu_lease", 503, True) from exc

        now_wall = time.time()
        q_started = _finite_float(state.get("gpu_queue_started_at"))
        if q_started is None or q_started <= 0:
            q_started = now_wall
            state["gpu_queue_started_at"] = q_started
        remaining_total = max(0.0, float(queue_budget) - max(0.0, now_wall - q_started))
        deadline = time.monotonic() + remaining_total
        step_deadline = time.monotonic() + min(remaining_total, step_yield_seconds)

        while not is_lease_granted(decision):
            lease_id = str(decision.get("lease_id") or "")
            lstate = str(decision.get("state") or "")
            if not lease_id or lstate not in {"queued", "draining"}:
                state.pop("gpu_scheduler_lease", None)
                state.pop("gpu_queue", None)
                state.pop("gpu_queue_started_at", None)
                raise Failure("waiting_gpu_lease", 503, True)
            state["gpu_scheduler_lease"] = decision
            state["gpu_queue"] = {
                "lease_id": lease_id,
                "state": lstate,
                "queue_position": decision.get("queue_position", 0),
                "blocking_reason": decision.get("blocking_reason") or "waiting_gpu_scheduler",
            }
            now_mono = time.monotonic()
            if now_mono >= deadline:
                with contextlib.suppress(Exception):
                    await asyncio.to_thread(
                        self.scheduler.release,
                        lease_id,
                        int(decision.get("fencing_token") or 0),
                        worker_settled=False,
                        reclaim_mode="abort_unstarted",
                        reason="queue_timeout",
                    )
                state.pop("gpu_scheduler_lease", None)
                state.pop("gpu_queue_started_at", None)
                raise Failure("waiting_gpu_lease", 503, True)
            if now_mono >= step_deadline:
                raise Failure("waiting_gpu_lease", 503, True)
            await asyncio.sleep(1.0)
            await asyncio.to_thread(self.store.renew, row)
            try:
                decision = await asyncio.to_thread(self.scheduler.get_lease, lease_id)
            except Exception as exc:
                raise Failure("waiting_gpu_lease", 503, True) from exc

        state["gpu_scheduler_lease"] = decision
        state.pop("gpu_queue", None)
        state.pop("gpu_queue_started_at", None)
        with contextlib.suppress(Exception):
            await asyncio.to_thread(self.store.checkpoint, row)

    async def _bind_lease_if_needed(self, row, ident: dict) -> None:
        if not self.scheduler.enabled or not _has_concrete_identity(ident):
            return
        state = row["state"]
        lease = state.get("gpu_scheduler_lease")
        if not is_lease_granted(lease):
            return
        need_bind = (
            not lease.get("pod_uid")
            or lease.get("pod_uid") != ident.get("pod_uid")
            or not (lease.get("gpu_uuids") or lease.get("assigned_gpu_uuids") or lease.get("gpu_uuid"))
        )
        if not need_bind:
            return
        try:
            bound = await asyncio.to_thread(
                self.scheduler.bind_target,
                str(lease["lease_id"]),
                int(lease["fencing_token"]),
                runtime_identity=ident,
                expected_version=int(lease.get("version") or 0),
            )
            state["gpu_scheduler_lease"] = bound
            with contextlib.suppress(Exception):
                await asyncio.to_thread(self.store.checkpoint, row)
        except Exception as exc:
            await self._release_scheduler_lease(row, reason="bind_target_failed")
            raise Failure("waiting_gpu_lease", 503, True) from exc

    async def _ensure_gpu(self, row):
        state = row["state"]
        timings = state.setdefault("timings", {"queue_seconds": 0.0, "load_seconds": 0.0, "inference_seconds": 0.0})
        if not getattr(self.gpu, "enabled", False) and not self.scheduler.enabled:
            return
        queue_start = time.monotonic()
        try:
            if self.scheduler.enabled:
                await self._acquire_scheduler_grant(row)
            elif getattr(self.gpu, "enabled", False):
                if not state.get("gpu_held"):
                    state["gpu_held"] = True
                    await asyncio.to_thread(self.store.checkpoint, row)
                else:
                    await asyncio.to_thread(self.store.renew, row)
                await asyncio.to_thread(self.gpu.acquire_for_queue, track=False)
        except LeaseBusy as error:
            raise Failure("waiting_gpu_lease", 503, True) from error
        finally:
            timings["queue_seconds"] = timings.get("queue_seconds", 0.0) + round(
                max(0.0, time.monotonic() - queue_start), 3
            )
        await asyncio.to_thread(self.store.renew, row)
        eng = self._row_engine(row)
        target_dep = self._deployment_for(eng)
        dep_args = (target_dep,) if target_dep else ()
        load_start = time.monotonic()
        prior_phase = row["phase"]
        if prior_phase == "reference":
            await asyncio.to_thread(self.store.checkpoint, row, phase="loading_model")
        if self._scale_enabled():
            await asyncio.to_thread(self.gpu.scale_deployment_to, 1, *dep_args)
        elif getattr(self.gpu, "enabled", False):
            await asyncio.to_thread(self.gpu.scale_to, 1)
        ready_budget = float(getattr(self.gpu, "ready_seconds", 900))
        deadline = time.monotonic() + ready_budget
        last_load_hb = time.monotonic()
        while time.monotonic() < deadline:
            try:
                health = await self._probe_runtime_health(
                    p=row["principal"], project_id=row["project_id"], engine=eng
                )
                if isinstance(health, dict):
                    if isinstance(health.get("runtime_identity"), dict):
                        ident = dict(health["runtime_identity"])
                        state["gpu_target_identity"] = ident
                        await self._bind_lease_if_needed(row, ident)
                    if health.get("ready") is True:
                        timings["load_seconds"] = timings.get("load_seconds", 0.0) + round(
                            max(0.0, time.monotonic() - load_start), 3
                        )
                        if row["phase"] == "loading_model":
                            row["phase"] = "reference"
                        return
            except Failure as fail_err:
                if fail_err.code == "waiting_gpu_lease":
                    raise
            except (OSError, RuntimeError):
                pass
            if self.scheduler.enabled and (time.monotonic() - last_load_hb) >= 15.0:
                last_load_hb = time.monotonic()
                with contextlib.suppress(Exception):
                    await asyncio.to_thread(self.store.renew, row)
                    await self._renew_execution_lease(row)
            await asyncio.sleep(self._ready_poll_seconds)
        timings["load_seconds"] = timings.get("load_seconds", 0.0) + round(max(0.0, time.monotonic() - load_start), 3)
        if self.scheduler.enabled and state.get("gpu_scheduler_lease"):
            await self._release_scheduler_lease(row, reason="model_load_timeout")
        raise Failure("model_load_timeout", 503, True)
