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
    app = applications(service, token("SPEECH_MCP_TOKEN"), Registry())
    server = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=8000, access_log=False))
    server.capture_signals = nullcontext
    def stop():
        server.should_exit = True
    for sig in (signal.SIGINT, signal.SIGTERM):
        asyncio.get_running_loop().add_signal_handler(sig, stop)
    await server.serve()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception:
        # Startup connection exceptions may include protected configuration.
        print("speech_startup_failed", file=sys.stderr)
        raise SystemExit(1) from None
