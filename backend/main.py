"""
Main FastAPI application entry point.
Orchestrates all routers and middleware.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
import logging
import uvicorn

from .config import Config

# Reduce noisy access logs
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)

# Create FastAPI app
app = FastAPI(
    title="Robodimm API",
    description="REST/WebSocket API for robot actuator sizing and motion simulation",
    version="1.0.0",
)

# Serve static meshes for Three.js visualization
_meshes_dir = Path(__file__).resolve().parent.parent / "meshes"
app.mount("/meshes", StaticFiles(directory=str(_meshes_dir)), name="meshes")

# Create station directory for imported geometries
_station_dir = Path(__file__).resolve().parent.parent / "station"
_station_dir.mkdir(exist_ok=True)
app.mount("/station", StaticFiles(directory=str(_station_dir)), name="station")

# CORS middleware - use origins from config
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and mount routers
from .auth import login
from .models import LoginRequest
from . import routers


# Authentication endpoint
@app.post("/login")
def login_endpoint(creds: LoginRequest):
    return login(creds)


# Register routers
app.include_router(routers.robot_info.router, prefix="", tags=["Robot Info"])
app.include_router(routers.jog.router, prefix="", tags=["Jog Control"])
app.include_router(routers.programming.router, prefix="", tags=["Programming"])
app.include_router(routers.execution.router, prefix="", tags=["Execution"])
app.include_router(routers.config.router, prefix="", tags=["Configuration"])
app.include_router(routers.station.router, prefix="", tags=["Station Geometry"])
app.include_router(routers.actuators.router, prefix="", tags=["Actuators"])

# Mount frontend at the end
# Note: Don't use StaticFiles at "/" as it intercepts WebSocket requests
_frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if _frontend_dir.exists():
    # Mount static assets separately
    app.mount("/app", StaticFiles(directory=str(_frontend_dir)), name="app")

    # Serve specific pages
    @app.get("/")
    async def serve_landing(request: Request):
        """Serve landing page at root."""
        return FileResponse(_frontend_dir / "landing.html")

    @app.get("/simulator")
    async def serve_simulator(request: Request):
        """Serve simulator page."""
        return FileResponse(_frontend_dir / "simulator.html")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str, request: Request):
        """Serve frontend SPA - returns appropriate file for non-API routes."""
        if full_path == "actuators_library.json":
            return FileResponse(_frontend_dir / "actuators_library.json")

        # Check if it's a WebSocket upgrade request
        if request.headers.get("upgrade", "").lower() == "websocket":
            # Let FastAPI handle WebSocket routes (will 404 if not found)
            return JSONResponse(status_code=404, content={"detail": "Not found"})

        # Skip API routes
        if full_path.startswith(
            (
                "login",
                "robot",
                "targets",
                "program",
                "configure",
                "actuators",
                "station",
                "trajectory",
                "export",
                "ws/",
                "saved_programs",
                "saved_configs",
            )
        ):
            return JSONResponse(status_code=404, content={"detail": "Not found"})

        # Try to serve the requested file
        file_path = _frontend_dir / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)

        # For backwards compatibility, redirect old index.html references to landing
        if full_path == "index.html":
            return FileResponse(_frontend_dir / "landing.html")

        # Default to simulator for SPA routing (handles /simulator?mode=xxx)
        return FileResponse(_frontend_dir / "simulator.html")


# Entry point
if __name__ == "__main__":
    uvicorn.run(
        app,
        host=Config.BACKEND_HOST,
        port=Config.BACKEND_PORT,
        access_log=False,
        log_level="warning",
    )
