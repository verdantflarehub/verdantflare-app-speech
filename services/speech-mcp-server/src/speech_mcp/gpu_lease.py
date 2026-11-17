"""Shared GPU task lease and on-demand scale.

Approved 2026-10-19 for 5090 Music3, H3, and IndexTTS2. Write this role's
Lease first, then read peers. A live conflicting peer means release and retry.
H3 and IndexTTS2 do not exclude each other. Music3 and IndexTTS2 scale to 1
while held and back to 0 after the idle window. H3 stays resident.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import random
import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode

import httpx

log = logging.getLogger("speech_gpu_lease")

LEASES = {
    "music3": "gpu-task-music3",
    "h3": "gpu-task-h3",
    "indextts2": "gpu-task-indextts2",
}
CONFLICTS = {
    "music3": ("h3", "indextts2"),
    "h3": ("music3",),
    "indextts2": ("music3",),
}


class LeaseBusy(RuntimeError):
    def __init__(self, holder: str | None = None) -> None:
        self.holder = holder or ""
        super().__init__("waiting_gpu_lease")


def format_time(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000000Z")


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def replicas_from_scale(payload: dict) -> int:
    # Kubernetes omits spec.replicas when the desired count is zero.
    spec = payload.get("spec") or {}
    if "replicas" not in spec:
        return 0
    return int(spec["replicas"])


def held_by_other(lease: dict | None, holder: str, now: datetime) -> str | None:
    spec = (lease or {}).get("spec") or {}
    identity = spec.get("holderIdentity") or ""
    if not identity or identity == holder:
        return None
    renew = spec.get("renewTime")
    duration = int(spec.get("leaseDurationSeconds") or 60)
    if not renew:
        return identity
    if parse_time(renew) + timedelta(seconds=duration) > now:
        return identity
    return None


def _scheduler_env_enabled() -> bool:
    url = os.environ.get("STATION_GPU_SCHEDULER_URL", "").strip()
    tok = (
        os.environ.get("STATION_GPU_SCHEDULER_TOKEN", "").strip()
        or os.environ.get("STATION_RUNTIME_GPU_TOKEN", "").strip()
    )
    return bool(url and tok)


class MemoryApi:
    """In-memory Lease and scale API for tests."""

    def __init__(self) -> None:
        self.leases: dict[str, dict] = {}
        self.scales: dict[str, int] = {}
        self.pods: dict[str, list[dict]] = {}
        self.lease_calls: list[tuple[str, str]] = []

    def get_lease(self, name: str) -> dict | None:
        self.lease_calls.append(("get", name))
        item = self.leases.get(name)
        return None if item is None else {
            "metadata": dict(item["metadata"]),
            "spec": dict(item["spec"]),
        }

    def replace_lease(self, name: str, spec: dict, resource_version: str | None) -> int:
        self.lease_calls.append(("replace", name))
        current = self.leases.get(name)
        current_version = None if current is None else current["metadata"].get("resourceVersion")
        if current is not None and current_version != resource_version:
            return 409
        version = str(int(current_version or "0") + 1)
        self.leases[name] = {"metadata": {"name": name, "resourceVersion": version}, "spec": dict(spec)}
        return 200

    def get_scale(self, name: str) -> int:
        return self.scales.get(name, 0)

    def patch_scale(self, name: str, replicas: int) -> None:
        self.scales[name] = replicas

    def deployment_pods(self, name: str) -> list[dict]:
        if name in self.pods:
            return [dict(item) for item in self.pods[name]]
        if self.scales.get(name, 0) > 0:
            return [{"metadata": {"name": f"{name}-pod-0", "uid": f"uid-{name}-0"}, "status": {"phase": "Running"}}]
        return []


class K8sApi:
    def __init__(
        self,
        namespace: str,
        host: str = "https://kubernetes.default.svc",
        token_path: str = "/var/run/secrets/kubernetes.io/serviceaccount/token",
        ca_path: str = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt",
    ) -> None:
        self.namespace = namespace
        self.host = host.rstrip("/")
        self.token_path = token_path
        self.ca_path = ca_path

    def _request(self, method: str, path: str, **kwargs: object) -> httpx.Response:
        token = Path(self.token_path).read_text(encoding="utf-8").strip()
        headers = {"Authorization": f"Bearer {token}"}
        extra = kwargs.pop("headers", None)
        if isinstance(extra, dict):
            headers.update(extra)
        with httpx.Client(verify=self.ca_path, timeout=10, trust_env=False) as client:
            return client.request(method, self.host + path, headers=headers, **kwargs)

    def get_lease(self, name: str) -> dict | None:
        path = f"/apis/coordination.k8s.io/v1/namespaces/{self.namespace}/leases/{name}"
        response = self._request("GET", path)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()

    def replace_lease(self, name: str, spec: dict, resource_version: str | None) -> int:
        path = f"/apis/coordination.k8s.io/v1/namespaces/{self.namespace}/leases/{name}"
        body = {
            "apiVersion": "coordination.k8s.io/v1",
            "kind": "Lease",
            "metadata": {"name": name, "namespace": self.namespace},
            "spec": spec,
        }
        if resource_version:
            body["metadata"]["resourceVersion"] = resource_version
            response = self._request("PUT", path, json=body)
        else:
            response = self._request("POST", f"/apis/coordination.k8s.io/v1/namespaces/{self.namespace}/leases", json=body)
            # A competing creator must be re-observed, never overwritten.
        if response.status_code in {200, 201}:
            return response.status_code
        return response.status_code

    def get_scale(self, name: str) -> int:
        path = f"/apis/apps/v1/namespaces/{self.namespace}/deployments/{name}/scale"
        response = self._request("GET", path)
        response.raise_for_status()
        return replicas_from_scale(response.json())

    def deployment_pods(self, name: str) -> list[dict]:
        response = self._request("GET", f"/apis/apps/v1/namespaces/{self.namespace}/deployments/{name}")
        if response.status_code == 403:
            labels = {"app.kubernetes.io/name": name}
        else:
            response.raise_for_status()
            selector = response.json()["spec"]["selector"]
            labels = selector.get("matchLabels")
            if not labels or selector.get("matchExpressions"):
                raise RuntimeError("unsupported_deployment_selector")
        response = self._request(
            "GET",
            f"/api/v1/namespaces/{self.namespace}/pods",
            params={"labelSelector": ",".join(f"{k}={v}" for k, v in labels.items())},
        )
        response.raise_for_status()
        items = response.json().get("items")
        if not isinstance(items, list):
            raise RuntimeError("invalid_pod_list")
        return items

    def patch_scale(self, name: str, replicas: int) -> None:
        path = f"/apis/apps/v1/namespaces/{self.namespace}/deployments/{name}/scale"
        response = self._request(
            "PATCH",
            path,
            json={"spec": {"replicas": replicas}},
            headers={"Content-Type": "application/merge-patch+json"},
        )
        response.raise_for_status()


class GpuCoordinator:
    def __init__(
        self,
        api: object | None,
        role: str,
        holder: str,
        deployment: str | None = None,
        enabled: bool = False,
        idle_seconds: int = 600,
        queue_seconds: int = 1200,
        ready_seconds: int = 900,
        lease_seconds: int = 90,
        now=None,
        sleep=None,
        monotonic=None,
    ) -> None:
        self.api = api
        self.role = role
        self.holder = holder
        self.deployment = deployment or None
        self.enabled = enabled
        self.idle_seconds = idle_seconds
        self.queue_seconds = queue_seconds
        self.ready_seconds = ready_seconds
        self.lease_seconds = lease_seconds
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._sleep = sleep or time.sleep
        self._monotonic = monotonic or time.monotonic
        self.in_flight = 0
        self.idle_at: float | None = None
        self._lock = threading.RLock()
        self.activity = None
        self._legacy_role_warned = False

    @property
    def lease_name(self) -> str:
        return LEASES[self.role]

    @property
    def scale_enabled(self) -> bool:
        return bool(self.api is not None and self.deployment)

    def disable_legacy_lease(self, *, residual_role: str | None = None) -> None:
        role = (residual_role if residual_role is not None else self.role) or ""
        role = role.strip()
        if role and not self._legacy_role_warned:
            log.warning(
                "speech_legacy_gpu_task_role_ignored role=%s reason=station_gpu_scheduler_enabled",
                role,
            )
            self._legacy_role_warned = True
        self.enabled = False
        self.role = ""
        self.holder = ""

    @classmethod
    def from_environment(cls) -> "GpuCoordinator":
        role = os.environ.get("GPU_TASK_ROLE", "").strip()
        deployment = os.environ.get("GPU_TASK_DEPLOYMENT", "").strip() or None
        namespace = os.environ.get("GPU_TASK_NAMESPACE", "").strip()
        namespace_file = Path("/var/run/secrets/kubernetes.io/serviceaccount/namespace")
        if not namespace and namespace_file.is_file():
            namespace = namespace_file.read_text(encoding="utf-8").strip()
        namespace = namespace or "verdantflare-app"
        if _scheduler_env_enabled():
            coord = cls(
                api=K8sApi(namespace) if deployment else None,
                role="",
                holder="",
                deployment=deployment,
                enabled=False,
                idle_seconds=int(os.environ.get("GPU_IDLE_SECONDS", "600")),
                queue_seconds=int(os.environ.get("GPU_QUEUE_SECONDS", "1200")),
                ready_seconds=int(os.environ.get("GPU_READY_SECONDS", "900")),
                lease_seconds=int(os.environ.get("GPU_LEASE_SECONDS", "90")),
            )
            if role:
                coord.disable_legacy_lease(residual_role=role)
            return coord
        if role not in CONFLICTS:
            return cls(api=None, role="", holder="", enabled=False)
        holder = os.environ.get("GPU_TASK_HOLDER", "").strip() or f"{role}/{os.environ.get('HOSTNAME', 'local')}"
        return cls(
            api=K8sApi(namespace),
            role=role,
            holder=holder,
            deployment=deployment,
            enabled=True,
            idle_seconds=int(os.environ.get("GPU_IDLE_SECONDS", "600")),
            queue_seconds=int(os.environ.get("GPU_QUEUE_SECONDS", "1200")),
            ready_seconds=int(os.environ.get("GPU_READY_SECONDS", "900")),
            lease_seconds=int(os.environ.get("GPU_LEASE_SECONDS", "90")),
        )

    def peer_holder(self) -> str | None:
        if not self.enabled:
            return None
        now = self._now()
        for peer in CONFLICTS[self.role]:
            holder = held_by_other(self.api.get_lease(LEASES[peer]), self.holder, now)
            if holder:
                return holder
        return None

    def api_ok(self) -> bool:
        if self.enabled:
            try:
                self.api.get_lease(self.lease_name)
                return True
            except Exception:
                return False
        if self.scale_enabled:
            try:
                self.api.get_scale(self.deployment)
                return True
            except Exception:
                return False
        return True

    def replicas(self) -> int | None:
        if not self.enabled or not self.deployment:
            return None
        try:
            return int(self.api.get_scale(self.deployment))
        except Exception:
            return None

    def replicas_for(self, deployment: str | None = None) -> int | None:
        target = deployment or self.deployment
        if self.api is None or not target:
            return None
        try:
            return int(self.api.get_scale(target))
        except Exception:
            return None

    def scale_to(self, replicas: int) -> None:
        if not self.enabled or not self.deployment:
            return
        if self.api.get_scale(self.deployment) != replicas:
            self.api.patch_scale(self.deployment, replicas)

    def scale_deployment_to(self, replicas: int, deployment: str | None = None) -> None:
        target = deployment or self.deployment
        if self.api is None or not target:
            return
        if self.api.get_scale(target) != replicas:
            self.api.patch_scale(target, replicas)

    def deployment_pods(self, deployment: str | None = None) -> list[dict]:
        target = deployment or self.deployment
        if self.api is None or not target or not hasattr(self.api, "deployment_pods"):
            return []
        return list(self.api.deployment_pods(target))

    def stop_deployment_and_wait(self, deployment: str | None = None, *, on_wait=None) -> None:
        target = deployment or self.deployment
        if self.api is None or not target:
            raise RuntimeError("gpu_shutdown_unavailable")
        self.scale_deployment_to(0, target)
        deadline = self._monotonic() + self.ready_seconds
        while True:
            # Include terminating/terminal Pods: desired replicas=0 alone is insufficient.
            if not self.deployment_pods(target):
                return
            if self._monotonic() >= deadline:
                raise RuntimeError("gpu_shutdown_timeout")
            if on_wait is not None:
                with contextlib.suppress(Exception):
                    on_wait()
            self._sleep(1)

    def _spec(self, holder: str, acquire_time: str | None = None) -> dict:
        now = format_time(self._now())
        return {
            "holderIdentity": holder,
            "leaseDurationSeconds": self.lease_seconds,
            "acquireTime": acquire_time or now,
            "renewTime": now,
        }

    def _write(self, holder: str) -> bool:
        current = self.api.get_lease(self.lease_name)
        identity = ((current or {}).get("spec") or {}).get("holderIdentity")
        if not holder and identity != self.holder:
            return False
        if holder and held_by_other(current, self.holder, self._now()):
            return False
        version = None if current is None else current.get("metadata", {}).get("resourceVersion")
        acquire_time = None
        if current and (current.get("spec") or {}).get("holderIdentity") == self.holder:
            acquire_time = (current.get("spec") or {}).get("acquireTime")
        return self.api.replace_lease(self.lease_name, self._spec(holder, acquire_time), version) in {200, 201}

    def try_acquire(self) -> bool:
        if not self.enabled:
            return True
        with self._lock:
            now = self._now()
            current = self.api.get_lease(self.lease_name)
            if held_by_other(current, self.holder, now):
                return False
            if held_by_other(current, "", now) == self.holder:
                return self._write(self.holder)
            if not self._write(self.holder):
                return False
            if self.peer_holder():
                self._write("")
                return False
            return True

    def acquire_for_queue(self, *, track: bool = True) -> None:
        if not self.enabled:
            return
        deadline = self._monotonic() + self.queue_seconds
        delay = 0.2
        while True:
            with self._lock:
                if self.try_acquire():
                    if track:
                        self.in_flight += 1
                    self.idle_at = None
                    return
            if self._monotonic() >= deadline:
                raise LeaseBusy(self.peer_holder())
            delay = min(5.0, delay * 2)
            self._sleep(delay * (0.5 + random.random()))

    def renew(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            if self._active() and not self.try_acquire():
                raise LeaseBusy(self.peer_holder())

    def _active(self) -> bool:
        return bool(self.in_flight or (self.activity is not None and self.activity()))

    def reconcile(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            try:
                active = self._active()
            except Exception:
                current = self.api.get_lease(self.lease_name)
                if held_by_other(current, "", self._now()) == self.holder:
                    self._write(self.holder)
                raise
            if active:
                self.idle_at = None
                if not self.try_acquire():
                    raise LeaseBusy(self.peer_holder())
                return
            self._write("")
            self.maybe_scale_down()

    def release(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            if self.in_flight:
                self.in_flight -= 1
            if self._active():
                self.idle_at = None
                return
            self._write("")
            if self.idle_at is None:
                self.idle_at = self._monotonic()

    def maybe_scale_down(self) -> None:
        if not self.enabled or not self.deployment:
            return
        with self._lock:
            if self._active():
                self.idle_at = None
                return
            if self.idle_at is None:
                self.idle_at = self._monotonic()
            if self._monotonic() - self.idle_at < self.idle_seconds:
                return
            if not self._write(self.holder):
                return
            try:
                if not self._active() and self.replicas() not in {None, 0}:
                    self.scale_to(0)
            finally:
                self._write("")

    def hold(self) -> "_Hold":
        return _Hold(self)


class _Hold:
    def __init__(self, coord: GpuCoordinator) -> None:
        self.coord = coord
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "_Hold":
        self.coord.acquire_for_queue()
        if self.coord.enabled:
            self._thread = threading.Thread(target=self._renew, daemon=True)
            self._thread.start()
        return self

    def _renew(self) -> None:
        interval = max(5, self.coord.lease_seconds // 3)
        while not self._stop.wait(interval):
            try:
                self.coord.renew()
            except Exception:
                pass

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._stop.set()
        self.coord.release()
        return False


SCHEDULER_GRANTED_STATES = frozenset({"leased", "running", "granted"})


def is_lease_granted(lease: dict | None) -> bool:
    return isinstance(lease, dict) and str(lease.get("state") or "") in SCHEDULER_GRANTED_STATES


class SchedulerError(RuntimeError):
    def __init__(self, code: str, message: str = "", status_code: int = 500) -> None:
        self.code = code
        self.message = message
        self.status_code = status_code
        super().__init__(f"{code}: {message}" if message else code)


def _fsync_directory(dir_path: Path) -> None:
    if os.name == "nt":
        return
    dfd = os.open(str(dir_path), os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


class SchedulerLeaseClient:
    """HTTP client for station-runtime :5056 unified GPU lease scheduler."""

    def __init__(
        self,
        *,
        base_url: str = "",
        token: str = "",
        holder_app_id: str = "speech-mcp",
        enabled: bool | None = None,
        client: httpx.Client | None = None,
        pending_releases_path: str | Path | None = None,
        store: object | None = None,
        pending_releases_backend: str | None = None,
        owner_pod_uid: str | None = None,
        claim_seconds: int = 30,
    ) -> None:
        self.base_url = base_url.strip().rstrip("/")
        self.token = token.strip()
        self.holder_app_id = holder_app_id.strip() or "speech-mcp"
        self.enabled = bool(self.base_url and self.token) if enabled is None else bool(enabled)
        self._client = client
        self._pending_lock = threading.RLock()
        self._store = store
        self._owner_pod_uid = (
            str(owner_pod_uid).strip()
            if owner_pod_uid is not None
            else (os.environ.get("POD_UID", "").strip() or os.environ.get("HOSTNAME", "").strip())
        )
        self._claim_seconds = max(1, int(claim_seconds))

        explicit_backend = str(pending_releases_backend or "").strip().lower()
        env_backend = os.environ.get("STATION_GPU_PENDING_RELEASES_BACKEND", "").strip().lower()
        env_journal = os.environ.get("STATION_GPU_PENDING_RELEASES_PATH", "").strip()

        if explicit_backend:
            resolved_backend = explicit_backend
        elif pending_releases_path is not None:
            resolved_backend = "file"
        elif env_backend:
            resolved_backend = env_backend
        elif env_journal:
            resolved_backend = "file"
        elif store is not None and hasattr(store, "upsert_pending_gpu_release"):
            resolved_backend = "postgres"
        else:
            resolved_backend = "file"

        if resolved_backend not in {"postgres", "file"}:
            raise SchedulerError(
                "INVALID_JOURNAL_BACKEND",
                f"unsupported pending releases backend: {resolved_backend!r}",
                500,
            )
        self._pending_backend = resolved_backend

        raw_journal = ""
        if self._pending_backend == "file":
            raw_journal = (
                str(pending_releases_path).strip()
                if pending_releases_path is not None
                else env_journal
            )
        self._pending_releases_path: Path | None = Path(raw_journal) if raw_journal else None
        self._pending_releases: dict[str, dict] = {}
        if self._pending_backend == "file":
            self._load_pending_releases_locked()

    @classmethod
    def from_environment(
        cls,
        *,
        store: object | None = None,
        pending_releases_path: str | Path | None = None,
        pending_releases_backend: str | None = None,
        owner_pod_uid: str | None = None,
    ) -> "SchedulerLeaseClient":
        url = os.environ.get("STATION_GPU_SCHEDULER_URL", "").strip()
        tok = (
            os.environ.get("STATION_GPU_SCHEDULER_TOKEN", "").strip()
            or os.environ.get("STATION_RUNTIME_GPU_TOKEN", "").strip()
        )
        holder_app = os.environ.get("STATION_GPU_HOLDER_APP_ID", "speech-mcp").strip() or "speech-mcp"
        explicit_backend = str(pending_releases_backend or "").strip().lower()
        env_backend = os.environ.get("STATION_GPU_PENDING_RELEASES_BACKEND", "").strip().lower()
        env_journal = os.environ.get("STATION_GPU_PENDING_RELEASES_PATH", "").strip()

        if explicit_backend:
            backend = explicit_backend
        elif pending_releases_path is not None:
            backend = "file"
        elif env_backend:
            backend = env_backend
        elif env_journal:
            backend = "file"
        elif store is not None and hasattr(store, "upsert_pending_gpu_release"):
            backend = "postgres"
        else:
            backend = "file"

        resolved_journal: str | Path | None = None
        if backend == "file":
            default_dir = Path("/data/speech-mcp") if Path("/data").is_dir() else Path(tempfile.gettempdir())
            resolved_journal = env_journal or pending_releases_path or (default_dir / "gpu_pending_releases.json")

        return cls(
            base_url=url,
            token=tok,
            holder_app_id=holder_app,
            pending_releases_path=resolved_journal,
            store=store,
            pending_releases_backend=backend,
            owner_pod_uid=owner_pod_uid,
        )

    def attach_store(self, store: object | None) -> None:
        if store is None:
            return
        with self._pending_lock:
            self._store = store
            if (
                self._pending_backend == "file"
                and self._pending_releases_path is None
                and hasattr(store, "upsert_pending_gpu_release")
            ):
                self._pending_backend = "postgres"

    def _require_pg_store(self):
        store = self._store
        if store is None or not hasattr(store, "upsert_pending_gpu_release"):
            raise SchedulerError(
                "JOURNAL_STORE_REQUIRED",
                "postgres pending releases backend requires a PostgreSQL Store",
                503,
            )
        return store

    def _load_pending_releases_locked(self) -> None:
        if self._pending_releases_path is None or not os.path.lexists(self._pending_releases_path):
            return
        if os.path.islink(self._pending_releases_path) or not self._pending_releases_path.is_file():
            raise SchedulerError("JOURNAL_CORRUPTED", "pending releases journal is not a regular file", 503)
        try:
            payload = json.loads(self._pending_releases_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise SchedulerError("JOURNAL_CORRUPTED", f"failed reading pending releases journal: {exc}", 503) from exc
        if not isinstance(payload, dict):
            raise SchedulerError("JOURNAL_CORRUPTED", "pending releases journal root must be an object", 503)
        loaded: dict[str, dict] = {}
        for k, v in payload.items():
            if not isinstance(k, str) or not k.strip() or not isinstance(v, dict):
                raise SchedulerError("JOURNAL_CORRUPTED", "invalid pending release journal entry", 503)
            loaded[k.strip()] = dict(v)
        self._pending_releases = loaded

    def _persist_pending_releases_locked(self) -> None:
        if self._pending_releases_path is None:
            return
        self._pending_releases_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._pending_releases_path.with_name(
            f"{self._pending_releases_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        data = json.dumps(self._pending_releases, separators=(",", ":"), sort_keys=True).encode("utf-8")
        try:
            fd = os.open(str(tmp_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            try:
                view = memoryview(data)
                while view:
                    written = os.write(fd, view)
                    if written <= 0:
                        raise OSError("short_write_zero_bytes")
                    view = view[written:]
                os.fsync(fd)
            finally:
                os.close(fd)
            os.replace(str(tmp_path), str(self._pending_releases_path))
            _fsync_directory(self._pending_releases_path.parent)
        except OSError as exc:
            try:
                if os.path.lexists(tmp_path):
                    os.unlink(tmp_path)
            except OSError:
                pass
            raise SchedulerError("JOURNAL_PERSIST_FAILED", str(exc), 503) from exc

    @staticmethod
    def _is_confirmed_release_receipt(lease_id: str, data: object) -> bool:
        if not isinstance(data, dict):
            return False
        if str(data.get("lease_id") or "").strip() != lease_id:
            return False
        return str(data.get("state") or "").strip().lower() in ("released", "expired")

    def _request_raw(self, method: str, path: str, payload: dict | None = None) -> httpx.Response:
        headers = {"Authorization": f"Bearer {self.token}"}
        if self._client is not None:
            return self._client.request(method, self.base_url + path, json=payload, headers=headers, timeout=15.0)
        with httpx.Client(timeout=15.0, trust_env=False) as http:
            return http.request(method, self.base_url + path, json=payload, headers=headers)

    @staticmethod
    def _parse_error(resp: httpx.Response) -> SchedulerError:
        code = f"HTTP_{resp.status_code}"
        msg = resp.text
        try:
            data = resp.json() if resp.content else {}
        except ValueError:
            data = {}
        if isinstance(data, dict):
            top_code = data.get("error_code")
            top_msg = data.get("message")
            nested = data.get("error")
            if isinstance(top_code, str) and top_code:
                code = top_code
                if isinstance(top_msg, str) and top_msg:
                    msg = top_msg
            elif isinstance(nested, dict):
                if isinstance(nested.get("code"), str) and nested["code"]:
                    code = nested["code"]
                if isinstance(nested.get("message"), str) and nested["message"]:
                    msg = nested["message"]
        return SchedulerError(code, msg, status_code=resp.status_code)

    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        try:
            resp = self._request_raw(method, path, payload)
        except SchedulerError:
            raise
        except Exception as exc:
            raise SchedulerError("TRANSPORT_ERROR", str(exc), status_code=503) from exc
        if resp.status_code >= 400:
            raise self._parse_error(resp)
        return resp.json() if resp.content else {}

    def get_pending_release(self, lease_id: str) -> dict | None:
        if self._pending_backend == "postgres":
            store = self._require_pg_store()
            row = store.get_pending_gpu_release(lease_id)
            if not row:
                return None
            payload = dict(row.get("payload") or {}) if isinstance(row.get("payload"), dict) else {}
            payload["_status"] = str(row.get("status") or "pending")
            payload["_quarantined_400"] = payload["_status"] == "quarantined_400"
            payload["_last_error"] = str(row.get("last_error") or "")
            payload["_owner_pod_uid"] = str(row.get("owner_pod_uid") or "")
            return payload
        with self._pending_lock:
            entry = self._pending_releases.get(lease_id)
            return dict(entry) if isinstance(entry, dict) else None

    def flush_pending_releases(self, *, include_quarantined: bool = False) -> int:
        if not self.enabled:
            return 0
        if self._pending_backend == "postgres":
            store = self._require_pg_store()
            flushed = 0
            visited: list[str] = []
            # Claim strictly 1 row at a time right before sending HTTP so queued items never burn claim_until
            for _ in range(32):
                row = store.claim_next_pending_gpu_release(
                    owner_pod_uid=self._owner_pod_uid,
                    include_quarantined=include_quarantined,
                    claim_seconds=self._claim_seconds,
                    exclude_lease_ids=visited,
                )
                if not row:
                    break
                lease_id = str(row.get("lease_id") or "").strip()
                visited.append(lease_id)
                claim_token = str(row.get("claim_token") or "")
                payload_version = int(row.get("payload_version") or 1)
                raw_body = row.get("payload") if isinstance(row.get("payload"), dict) else {}
                retry_body = {
                    k: v
                    for k, v in raw_body.items()
                    if k != "expected_version" and not str(k).startswith("_")
                }
                try:
                    resp = self._request_raw(
                        "POST",
                        f"/internal/v1/gpu-scheduler/leases/{lease_id}/release",
                        retry_body,
                    )
                    if resp.status_code == 200:
                        data = resp.json() if resp.content else {}
                        if self._is_confirmed_release_receipt(lease_id, data):
                            if store.delete_pending_gpu_release(
                                lease_id,
                                claim_token=claim_token,
                                payload_version=payload_version,
                            ):
                                flushed += 1
                            continue
                    elif resp.status_code == 400:
                        err = self._parse_error(resp)
                        logging.error(
                            "speech_gpu_release_quarantined_400 lease_id=%s code=%s message=%s",
                            lease_id,
                            err.code,
                            err.message,
                        )
                        store.quarantine_pending_gpu_release(
                            lease_id,
                            f"{err.code}:{err.message}",
                            claim_token=claim_token,
                            payload_version=payload_version,
                        )
                        continue
                    store.release_pending_gpu_claim(
                        lease_id,
                        claim_token=claim_token,
                        payload_version=payload_version,
                    )
                except Exception:
                    with contextlib.suppress(Exception):
                        store.release_pending_gpu_claim(
                            lease_id,
                            claim_token=claim_token,
                            payload_version=payload_version,
                        )
                    continue
            return flushed

        with self._pending_lock:
            if not self._pending_releases:
                return 0
            snapshot = [
                (lid, dict(b))
                for lid, b in self._pending_releases.items()
                if include_quarantined or not bool(b.get("_quarantined_400"))
            ]
        if not snapshot:
            return 0
        flushed = 0
        for lease_id, body in snapshot:
            retry_body = {
                k: v
                for k, v in body.items()
                if k != "expected_version" and not str(k).startswith("_")
            }
            try:
                resp = self._request_raw("POST", f"/internal/v1/gpu-scheduler/leases/{lease_id}/release", retry_body)
                if resp.status_code == 200:
                    data = resp.json() if resp.content else {}
                    if self._is_confirmed_release_receipt(lease_id, data):
                        with self._pending_lock:
                            self._pending_releases.pop(lease_id, None)
                            self._persist_pending_releases_locked()
                        flushed += 1
                elif resp.status_code == 400:
                    err = self._parse_error(resp)
                    logging.error(
                        "speech_gpu_release_quarantined_400 lease_id=%s code=%s message=%s",
                        lease_id,
                        err.code,
                        err.message,
                    )
                    with self._pending_lock:
                        cur = self._pending_releases.get(lease_id)
                        if isinstance(cur, dict):
                            upd = dict(cur)
                            upd["_quarantined_400"] = True
                            upd["_last_error"] = f"{err.code}:{err.message}"
                            self._pending_releases[lease_id] = upd
                            self._persist_pending_releases_locked()
            except Exception:
                continue
        return flushed

    def acquire(
        self,
        *,
        parent_task_id: str,
        attempt_id: str,
        stage_id: str = "indextts2_inference",
        vram_budget_mib: int = 6144,
        k8s_slots_required: int = 0,
        lease_ttl_seconds: int = 120,
        runtime_identity: dict | None = None,
        idempotency_key: str | None = None,
    ) -> dict:
        self.flush_pending_releases()
        ident = runtime_identity or {}
        body: dict[str, object] = {
            "idempotency_key": idempotency_key or f"acq:{parent_task_id}:{attempt_id}:{stage_id}",
            "holder_app_id": self.holder_app_id,
            "consumer_kind": "indextts2",
            "task_id": parent_task_id,
            "attempt_id": attempt_id,
            "stage_id": stage_id,
            "card_count": 1,
            "peak_budget_mib": int(vram_budget_mib),
            "priority_class": "shared_slice",
            "ttl_seconds": int(lease_ttl_seconds),
        }
        if k8s_slots_required > 0:
            body["k8s_slots_required"] = int(k8s_slots_required)
        for key in ("pod_uid", "container_id", "boot_id"):
            val = ident.get(key)
            if isinstance(val, str) and val and not val.startswith("unknown-"):
                body[key] = val
        visible = ident.get("visible_gpu_uuids")
        if isinstance(visible, list) and visible:
            body["candidate_gpu_uuids"] = [str(u) for u in visible if str(u)]
        return self._request("POST", "/internal/v1/gpu-scheduler/leases/acquire", body)

    def get_lease(self, lease_id: str) -> dict:
        return self._request("GET", f"/internal/v1/gpu-scheduler/leases/{lease_id}")

    def list_own_leases(self) -> list[dict]:
        if not self.enabled:
            return []
        query = urlencode({"holder_app_id": self.holder_app_id})
        try:
            resp = self._request_raw("GET", f"/internal/v1/gpu-scheduler/leases?{query}")
        except Exception:
            return []
        if resp.status_code in {404, 405}:
            return []
        if resp.status_code >= 400:
            return []
        try:
            data = resp.json() if resp.content else {}
        except ValueError:
            return []
        items = data
        if isinstance(data, dict):
            items = data.get("leases") if isinstance(data.get("leases"), list) else data.get("items")
        if not isinstance(items, list):
            return []
        unsettled = {"queued", "leased", "running", "granted", "draining", "reclaiming"}
        out: list[dict] = []
        for item in items:
            if not isinstance(item, dict) or not str(item.get("lease_id") or "").strip():
                continue
            holder = str(item.get("holder_app_id") or "").strip()
            if holder and holder != self.holder_app_id:
                continue
            state = str(item.get("state") or "").strip().lower()
            if state in unsettled:
                out.append(dict(item))
        return out

    def list_pending_release_ids(self) -> list[str]:
        if self._pending_backend == "postgres":
            store = self._store
            if store is not None and hasattr(store, "list_pending_gpu_releases"):
                return [
                    str(r.get("lease_id") or "").strip()
                    for r in store.list_pending_gpu_releases()
                    if isinstance(r, dict) and str(r.get("lease_id") or "").strip()
                ]
            return []
        with self._pending_lock:
            return [lid for lid in self._pending_releases if lid]

    def bind_target(
        self,
        lease_id: str,
        fencing_token: int,
        *,
        runtime_identity: dict,
        expected_version: int = 0,
        lease_ttl_seconds: int = 120,
        idempotency_key: str | None = None,
    ) -> dict:
        pod_uid = str(runtime_identity.get("pod_uid") or "").strip()
        container_id = str(runtime_identity.get("container_id") or "").strip()
        boot_id = str(runtime_identity.get("boot_id") or "").strip()
        if not pod_uid or not container_id or not boot_id or pod_uid.startswith("unknown-"):
            raise SchedulerError("INVALID_TARGET_IDENTITY", "bind_target requires concrete pod_uid, container_id, boot_id", 400)
        visible = runtime_identity.get("visible_gpu_uuids") or []
        actual_gpus = [str(u) for u in visible if str(u)] if isinstance(visible, (list, tuple)) else []
        body: dict[str, object] = {
            "idempotency_key": idempotency_key or f"bind:{lease_id}:{fencing_token}:{pod_uid}",
            "fencing_token": int(fencing_token),
            "ttl_seconds": int(lease_ttl_seconds),
            "pod_uid": pod_uid,
            "container_id": container_id,
            "boot_id": boot_id,
        }
        if expected_version > 0:
            body["expected_version"] = int(expected_version)
        if actual_gpus:
            body["actual_gpu_uuids"] = actual_gpus
        return self._request("POST", f"/internal/v1/gpu-scheduler/leases/{lease_id}/bind-target", body)

    def heartbeat(
        self,
        lease_id: str,
        fencing_token: int,
        *,
        lease_ttl_seconds: int = 120,
        expected_version: int = 0,
        idempotency_key: str | None = None,
    ) -> dict:
        body: dict[str, object] = {
            "idempotency_key": idempotency_key or f"hb:{lease_id}:{fencing_token}:{int(time.time() // 15)}",
            "fencing_token": int(fencing_token),
            "ttl_seconds": int(lease_ttl_seconds),
        }
        if expected_version > 0:
            body["expected_version"] = int(expected_version)
        return self._request(
            "POST",
            f"/internal/v1/gpu-scheduler/leases/{lease_id}/heartbeat",
            body,
        )

    def release(
        self,
        lease_id: str,
        fencing_token: int,
        *,
        reclaim_proof: dict | None = None,
        target_identity: dict | None = None,
        reason: str = "task_settled",
        worker_settled: bool = True,
        reclaim_mode: str = "warm_idle_settled",
        expected_version: int = 0,
        idempotency_key: str | None = None,
    ) -> dict:
        _ = reason
        ident = target_identity or {}
        body: dict[str, object] = {
            "idempotency_key": idempotency_key or f"rel:{lease_id}:{fencing_token}:{reclaim_mode or 'plain'}",
            "fencing_token": int(fencing_token),
            "worker_settled": bool(worker_settled),
        }
        if expected_version > 0:
            body["expected_version"] = int(expected_version)
        if reclaim_mode:
            body["reclaim_mode"] = reclaim_mode
        if reclaim_proof:
            body["reclaim_proof"] = reclaim_proof
        if isinstance(ident.get("supervisor_pid"), int) and ident["supervisor_pid"] > 0:
            body["supervisor_pid"] = int(ident["supervisor_pid"])
        if isinstance(ident.get("supervisor_starttime_ticks"), int) and ident["supervisor_starttime_ticks"] > 0:
            body["supervisor_starttime_ticks"] = int(ident["supervisor_starttime_ticks"])
        for key in ("pod_uid", "container_id", "boot_id"):
            val = ident.get(key)
            if isinstance(val, str) and val and not val.startswith("unknown-"):
                body[key] = val
        retry_body = {k: v for k, v in body.items() if k != "expected_version"}
        pg_claim_token: str = ""
        pg_payload_version: int = 0
        if self._pending_backend == "postgres":
            store = self._require_pg_store()
            try:
                pg_claim_token, pg_payload_version = store.upsert_pending_gpu_release(
                    lease_id,
                    int(fencing_token),
                    retry_body,
                    owner_pod_uid=self._owner_pod_uid,
                    claim_seconds=self._claim_seconds,
                )
            except Exception as exc:
                if getattr(exc, "code", "") == "pending_gpu_release_claimed_or_stale":
                    raise SchedulerError("RELEASE_CLAIM_CONFLICT", str(exc), 409) from exc
                raise
        else:
            with self._pending_lock:
                self._pending_releases[lease_id] = retry_body
                self._persist_pending_releases_locked()
        try:
            resp = self._request_raw("POST", f"/internal/v1/gpu-scheduler/leases/{lease_id}/release", body)
        except Exception as exc:
            if self._pending_backend == "postgres" and pg_claim_token:
                with contextlib.suppress(Exception):
                    self._require_pg_store().release_pending_gpu_claim(
                        lease_id,
                        claim_token=pg_claim_token,
                        payload_version=pg_payload_version,
                    )
            raise SchedulerError("RELEASE_TRANSPORT_FAILED", str(exc), 503) from exc
        if resp.status_code != 200:
            err = self._parse_error(resp)
            if resp.status_code == 400:
                logging.error(
                    "speech_gpu_release_quarantined_400 lease_id=%s code=%s message=%s",
                    lease_id,
                    err.code,
                    err.message,
                )
                if self._pending_backend == "postgres":
                    self._require_pg_store().quarantine_pending_gpu_release(
                        lease_id,
                        f"{err.code}:{err.message}",
                        claim_token=pg_claim_token,
                        payload_version=pg_payload_version,
                    )
                else:
                    with self._pending_lock:
                        cur = self._pending_releases.get(lease_id)
                        if isinstance(cur, dict):
                            upd = dict(cur)
                            upd["_quarantined_400"] = True
                            upd["_last_error"] = f"{err.code}:{err.message}"
                            self._pending_releases[lease_id] = upd
                            self._persist_pending_releases_locked()
            elif self._pending_backend == "postgres" and pg_claim_token:
                with contextlib.suppress(Exception):
                    self._require_pg_store().release_pending_gpu_claim(
                        lease_id,
                        claim_token=pg_claim_token,
                        payload_version=pg_payload_version,
                    )
            raise err
        try:
            data = resp.json() if resp.content else {}
        except ValueError:
            data = {}
        if not self._is_confirmed_release_receipt(lease_id, data):
            if self._pending_backend == "postgres" and pg_claim_token:
                with contextlib.suppress(Exception):
                    self._require_pg_store().release_pending_gpu_claim(
                        lease_id,
                        claim_token=pg_claim_token,
                        payload_version=pg_payload_version,
                    )
            state = data.get("state") if isinstance(data, dict) else None
            raise SchedulerError("RELEASE_NOT_TERMINAL", f"Unconfirmed release receipt for {lease_id}: state={state!r}", 502)
        if self._pending_backend == "postgres":
            self._require_pg_store().delete_pending_gpu_release(
                lease_id,
                claim_token=pg_claim_token,
                payload_version=pg_payload_version,
            )
        else:
            with self._pending_lock:
                self._pending_releases.pop(lease_id, None)
                self._persist_pending_releases_locked()
        return data

    def build_grant_headers(
        self,
        lease: dict | None,
        *,
        parent_task_id: str,
        target_identity: dict | None = None,
    ) -> dict[str, str]:
        if not self.enabled or not is_lease_granted(lease):
            return {}
        ident = target_identity or {}
        assigned = lease.get("gpu_uuids") or lease.get("assigned_gpu_uuids") or []
        gpu_uuids = [str(u) for u in assigned if str(u)] if isinstance(assigned, (list, tuple)) else []
        if not gpu_uuids and lease.get("gpu_uuid"):
            gpu_uuids = [str(lease["gpu_uuid"])]
        visible = ident.get("visible_gpu_uuids") or []
        if not gpu_uuids and visible:
            gpu_uuids = [str(visible[0])]
        headers = {
            "X-VF-GPU-Lease-ID": str(lease["lease_id"]),
            "X-VF-GPU-Fencing-Token": str(lease["fencing_token"]),
            "X-VF-GPU-Attempt-ID": str(lease.get("attempt_id") or "att-1"),
            "X-VF-GPU-Stage-ID": str(lease.get("stage_id") or "indextts2_inference"),
            "X-VF-GPU-Parent-Task-ID": str(parent_task_id),
            "X-VF-GPU-Consumer-Kind": "indextts2",
            "X-VF-GPU-Holder-App-ID": self.holder_app_id,
        }
        pod_uid = str(lease.get("pod_uid") or ident.get("pod_uid") or "")
        container_id = str(lease.get("container_id") or ident.get("container_id") or "")
        boot_id = str(lease.get("boot_id") or ident.get("boot_id") or "")
        if pod_uid and not pod_uid.startswith("unknown-"):
            headers["X-VF-GPU-Pod-UID"] = pod_uid
        if container_id and not container_id.startswith("unknown-"):
            headers["X-VF-GPU-Container-ID"] = container_id
        if boot_id and not boot_id.startswith("unknown-"):
            headers["X-VF-GPU-Boot-ID"] = boot_id
        if gpu_uuids:
            headers["X-VF-GPU-UUID"] = ",".join(gpu_uuids)
        expires_at = str(lease.get("expires_at") or "")
        if not expires_at:
            expires_at = format_time(datetime.now(timezone.utc) + timedelta(seconds=120))
        headers["X-VF-GPU-Expires-At"] = expires_at
        return headers

