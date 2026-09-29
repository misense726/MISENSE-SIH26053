"""
FastAPI Main Application and WebSocket Streaming Server.
Problem Statement: SIH 26053 - MI Sense
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path for universal execution
_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config.settings import settings
from backend.app.api.endpoints import router as api_router, websocket_stream_endpoint
from backend.app.simulation.engine import simulation_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mi_sense")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager starting and stopping the 15 Hz perception pipeline."""
    logger.info("Initializing MI Sense Perception Pipeline...")
    await simulation_engine.start()
    yield
    logger.info("Shutting down MI Sense Perception Pipeline...")
    await simulation_engine.stop()


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Adaptive Variable Resolution 2.5D LiDAR Mapping for Dynamic Perception (SIH 26053)",
    lifespan=lifespan,
)

# Enable CORS for frontend Vite dev server and production builds
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount REST API router (/api/health, /api/scenes, /api/config, etc.)
app.include_router(api_router, prefix="/api")


@app.get("/")
async def root():
    return {
        "project": settings.PROJECT_NAME,
        "problem_statement": "SIH 26053",
        "team": "MI Sense",
        "status": "active",
        "api_docs": "/docs",
        "stream_endpoint": "/ws/stream",
    }


# Forward top-level /ws/stream directly to the stream handler
@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    await websocket_stream_endpoint(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
