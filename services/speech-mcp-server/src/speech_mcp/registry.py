import asyncio
import base64
from datetime import datetime, timezone
import os

import httpx

from . import __version__
from .domain import TOOLS, canonical


class Registry:
    def __init__(self):
        self.base = os.environ["ETCD_ENDPOINTS"].rstrip("/")
        self.endpoint = os.environ["MCP_SERVICE_ENDPOINT"]
        self.lease = None
        self.http = httpx.AsyncClient(timeout=5, trust_env=False, follow_redirects=False)

    async def request(self, path, value):
        response = await self.http.post(self.base + "/v3/" + path, json=value)
        response.raise_for_status()
        return response.json()

    async def register(self):
        grant = await self.request("lease/grant", {"TTL": 300})
        self.lease = str(grant["ID"])
        meta = {"domain": "speech", "endpoint": self.endpoint,
                "health_endpoint": self.endpoint.removesuffix("/mcp") + "/health",
                "version": __version__, "updated_at": datetime.now(timezone.utc).isoformat(), "tools": TOOLS}
        b64 = lambda s: base64.b64encode(s.encode()).decode()
        await self.request("kv/put", {"key": b64("/verdantflare/mcp/services/speech"),
                                      "value": b64(canonical(meta)), "lease": self.lease})

    async def run(self):
        while True:
            try:
                if self.lease:
                    r = await self.request("lease/keepalive", {"ID": self.lease})
                    if int(r.get("result", {}).get("TTL", 0)) <= 0:
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
        except httpx.HTTPError:
            pass
        await self.http.aclose()
