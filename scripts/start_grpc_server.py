#!/usr/bin/env python3
"""Executable entrypoint script for APEX gRPC Recommendation Server."""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
import sys

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.grpc_server import serve_grpc


def main():
    parser = argparse.ArgumentParser(description="APEX gRPC Recommendation Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host address to bind to")
    parser.add_argument("--port", type=int, default=50051, help="Port to listen on")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    async def _run():
        server = await serve_grpc(host=args.host, port=args.port)
        logging.info(f"gRPC Server running on {args.host}:{args.port}. Press Ctrl+C to stop.")
        try:
            await server.wait_for_termination()
        except asyncio.CancelledError:
            await server.stop(grace=3.0)

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        logging.info("gRPC Server stopped cleanly.")


if __name__ == "__main__":
    main()
