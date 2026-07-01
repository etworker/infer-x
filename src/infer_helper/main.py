"""Inference Server Manager - HTTP API entry point."""

from __future__ import annotations

import argparse
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import ConfigManager
from .manager import InstanceManager
from .router import init_routes, router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("infer_helper")

_config: ConfigManager | None = None
_manager: InstanceManager | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _config, _manager
    _config = ConfigManager()
    _manager = InstanceManager(_config)
    init_routes(_config, _manager)
    logger.info(
        "Manager started: model_dir=%s, port_range=%d-%d",
        _config.config.model_dir,
        _config.config.port_range_start,
        _config.config.port_range_end,
    )
    yield
    logger.info("Shutting down...")
    await _manager.shutdown()


app = FastAPI(title="Inference Server Manager", version="1.0.0", lifespan=lifespan)
app.include_router(router)

STATIC_DIR = Path(__file__).parent.parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference Server Manager")
    parser.add_argument("--host", default="0.0.0.0", help="Listen host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8999, help="Listen port (default: 8999)")
    parser.add_argument("--config", default=None, help="Config file path")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
