import asyncio
from contextlib import nullcontext
import os
import signal
import sys

import uvicorn

from .clients import Peers, token
from .registry import Registry
from .server import applications
from .service import Service
from .store import Store


async def main():
    store = Store(os.environ["SPEECH_DATABASE_URL"])
    await asyncio.to_thread(store.migrate)
    service = Service(store, Peers.environment(), os.environ["SPEECH_MODEL_MANIFEST_SHA256"])
    apps = applications(service, token("SPEECH_MCP_TOKEN"), token("SPEECH_AUTHORITY_TOKEN"), Registry())
    servers = [uvicorn.Server(uvicorn.Config(apps[0], host="0.0.0.0", port=8000, access_log=False)),
               uvicorn.Server(uvicorn.Config(apps[1], host="0.0.0.0", port=8001, access_log=False,
                                            ssl_certfile=os.environ["SPEECH_TLS_CERT_FILE"],
                                            ssl_keyfile=os.environ["SPEECH_TLS_KEY_FILE"]))]
    for server in servers:
        server.capture_signals = nullcontext
    def stop():
        for server in servers:
            server.should_exit = True
    for sig in (signal.SIGINT, signal.SIGTERM):
        asyncio.get_running_loop().add_signal_handler(sig, stop)
    await asyncio.gather(*(s.serve() for s in servers))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception:
        # Startup connection exceptions may include protected configuration.
        print("speech_startup_failed", file=sys.stderr)
        raise SystemExit(1) from None
