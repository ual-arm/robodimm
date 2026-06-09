import pinocchio as pin
from fastapi import APIRouter
from typing import Dict, Any

router = APIRouter()

# Keep list of allowed origins for the health check response metadata
ALLOWED_ORIGINS = [
    "https://customrobotics.es",
    "https://www.customrobotics.es",
    "http://localhost:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174"
]

@router.get("/health")
def health_check() -> Dict[str, Any]:
    return {
        "status": "ok",
        "version": "1.0.0",
        "pinocchio_version": pin.__version__,
        "allowed_origins": ALLOWED_ORIGINS
    }

@router.get("/capabilities")
def get_capabilities() -> Dict[str, Any]:
    return {
        "capabilities": {
            "CR4": { "closed_chain_kkt": True },
            "CR6": { "serial_rnea": True }
        },
        "license_status": "dev_valid"
    }
