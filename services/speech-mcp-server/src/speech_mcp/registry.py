import asyncio
import base64
from datetime import datetime, timezone
import os
from urllib.parse import urlsplit

import httpx

from . import __version__
from .domain import TOOLS, canonical


class Registry:
    def __init__(self):
        self.base = os.environ["ETCD_ENDPOINTS"].rstrip("/")
        address = urlsplit(self.base)
        if (address.scheme not in {"http", "https"} or not address.hostname
                or address.username is not None or address.password is not None
                or address.path or address.query or address.fragment
                or any(c.isspace() for c in self.base)):
            raise ValueError("invalid_etcd_endpoint")
        self.username = os.environ.get("ETCD_USERNAME", "")
        self.password = os.environ.get("ETCD_PASSWORD", "")
        if bool(self.username) != bool(self.password):
            raise ValueError("incomplete_etcd_credentials")
        self.token = None
        self.auth_lock = asyncio.Lock()
        self.endpoint = os.environ["MCP_SERVICE_ENDPOINT"]
        self.lease = None
        self.http = httpx.AsyncClient(timeout=5, trust_env=False, follow_redirects=False)

    async def request(self, path, value):
        async with self.auth_lock:
            for attempt in range(2):
                if self.username and self.token is None:
                    response = await self.http.post(self.base + "/v3/auth/authenticate",
                                                    json={"name": self.username, "password": self.password})
                    response.raise_for_status()
                    data = response.json()
                    token = data.get("token") if isinstance(data, dict) else None
                    if not isinstance(token, str) or not token or len(token) > 16384 or any(c.isspace() for c in token):
                        raise ValueError("invalid_etcd_auth_response")
                    self.token = token
                headers = {"Authorization": self.token} if self.token else {}
                response = await self.http.post(self.base + "/v3/" + path, json=value, headers=headers)
                try:
                    data = response.json()
                except ValueError:
                    data = None
                if self.username and (response.status_code == 401 or isinstance(data, dict) and data.get("code") in (16, "16")):
                    self.token = None
                    if attempt == 0:
                        continue
                response.raise_for_status()
                if not isinstance(data, dict) or "error" in data or "code" in data:
                    raise ValueError("invalid_etcd_response")
                return data

    async def register(self):
        grant = await self.request("lease/grant", {"TTL": 300})
        lease = str(grant["ID"])
        if not lease.lstrip("-").isdigit() or int(lease) == 0:
            raise ValueError("invalid_etcd_lease")
        self.lease = lease
        meta = {"domain": "speech", "endpoint": self.endpoint,
                "health_endpoint": self.endpoint.removesuffix("/mcp") + "/health",
                "version": __version__, "updated_at": datetime.now(timezone.utc).isoformat(), "tools": TOOLS}
        b64 = lambda s: base64.b64encode(s.encode()).decode()
        try:
            await self.request("kv/put", {"key": b64("/verdantflare/mcp/services/speech"),
                                          "value": b64(canonical(meta)), "lease": lease})
        except (httpx.HTTPError, ValueError, KeyError):
            self.lease = None
            try:
                await self.request("lease/revoke", {"ID": lease})
            except (httpx.HTTPError, ValueError):
                pass
            raise

    async def run(self):
        while True:
            try:
                if self.lease:
                    r = await self.request("lease/keepalive", {"ID": self.lease})
                    result = r.get("result", {})
                    if str(result.get("ID")) != self.lease or int(result.get("TTL", 0)) <= 0:
                        self.lease = None
                if not self.lease:
                    await self.register()
            except (httpx.HTTPError, ValueError, KeyError):
                self.lease = None
            await asyncio.sleep(60)

    async def close(self):
        try:
            if self.lease:
                await self.request("lease/revoke", {"ID": self.lease})
        except (httpx.HTTPError, ValueError):
            pass
        self.token = None
        await self.http.aclose()
