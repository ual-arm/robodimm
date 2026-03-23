"""
station.py - Station Geometry Endpoints
======================================

Endpoints for managing external 3D station geometries.
"""

from fastapi import APIRouter, Depends, UploadFile, File
from pydantic import BaseModel
from typing import List
import uuid
import shutil
from pathlib import Path

from ..session import SimulationSession, get_session


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class StationGeometryUpdate(BaseModel):
    id: str
    position: List[float]  # [x, y, z]
    rotation: List[float]  # [rx, ry, rz] in radians
    scale: float = 1.0

class DeleteStationGeometry(BaseModel):
    """Delete a station geometry by ID."""
    id: str


# =============================================================================
# ROUTER SETUP
# =============================================================================

router = APIRouter()

# Station directory (must match backend/main.py mount: /app/station)
STATION_DIR = Path(__file__).resolve().parent.parent.parent / "station"


# =============================================================================
# STATION GEOMETRY ENDPOINTS (External 3D Models)
# =============================================================================

@router.get("/station_geometries")
def get_station_geometries(session: SimulationSession = Depends(get_session)):
    """Get list of all station geometries."""
    return {"geometries": session.station_geometries}


@router.post("/upload_station_geometry")
async def upload_station_geometry(file: UploadFile = File(...), session: SimulationSession = Depends(get_session)):
    """Upload a GLTF file for station geometry."""
    # Validate file type
    filename_lower = file.filename.lower()
    if not filename_lower.endswith(('.gltf', '.glb')):
        return {"ok": False, "error": "Only .gltf and .glb files are supported"}
    
    # Get station directory from backend main
    STATION_DIR.mkdir(exist_ok=True)
    
    # Generate unique ID
    geom_id = str(uuid.uuid4())[:8]
    safe_filename = f"{geom_id}_{file.filename.replace(' ', '_')}"
    file_path = STATION_DIR / safe_filename
    
    # Read file content
    content = await file.read()
    
    # For GLTF (text-based JSON), process buffer references
    if filename_lower.endswith('.gltf'):
        try:
            import json
            gltf_data = json.loads(content.decode('utf-8'))
            
            # Check for external buffer files and copy them
            if 'buffers' in gltf_data:
                for buffer in gltf_data['buffers']:
                    if 'uri' in buffer and not buffer['uri'].startswith('data:'):
                        bin_filename = buffer['uri']
                        # Try to find .bin in repository meshes directory first
                        _meshes_dir = Path(__file__).resolve().parent.parent.parent / "meshes"
                        src_bin = _meshes_dir / bin_filename
                        if src_bin.exists():
                            dst_bin = STATION_DIR / bin_filename
                            if not dst_bin.exists():
                                dst_bin.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copy2(src_bin, dst_bin)
                                import logging
                                logging.info(f"Copied {bin_filename} from meshes to station")

            # Copy external textures if they exist in repository meshes directory
            if 'images' in gltf_data:
                _meshes_dir = Path(__file__).resolve().parent.parent.parent / "meshes"
                for image in gltf_data['images']:
                    uri = image.get('uri')
                    if uri and not uri.startswith('data:'):
                        src_img = _meshes_dir / uri
                        dst_img = STATION_DIR / uri
                        if src_img.exists() and not dst_img.exists():
                            dst_img.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(src_img, dst_img)
                            import logging
                            logging.info(f"Copied texture {uri} from meshes to station")
            
            # Save the GLTF
            with open(file_path, "w", encoding='utf-8') as f:
                json.dump(gltf_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            import logging
            logging.error(f"Error processing GLTF: {e}")
            # Fall back to saving as-is
            with open(file_path, "wb") as f:
                f.write(content)
    else:
        # GLB binary format - save as-is
        with open(file_path, "wb") as f:
            f.write(content)
    
    # Create geometry entry with default position/rotation
    geom_entry = {
        "id": geom_id,
        "name": file.filename,
        "url": f"/station/{safe_filename}",
        "position": [0.0, 0.0, 0.0],
        "rotation": [0.0, 0.0, 0.0],
        "scale": 1.0
    }
    session.station_geometries.append(geom_entry)
    
    return {"ok": True, "geometry": geom_entry}


@router.post("/upload_station_bin")
async def upload_station_bin(file: UploadFile = File(...)):
    """Upload a .bin file associated with a GLTF."""
    if not file.filename.lower().endswith('.bin'):
        return {"ok": False, "error": "Only .bin files are supported"}
    
    # Get station directory from backend main
    STATION_DIR.mkdir(exist_ok=True)
    
    # Save with original filename (GLTF references it by name)
    file_path = STATION_DIR / file.filename
    
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    return {"ok": True, "filename": file.filename}


@router.post("/update_station_geometry")
def update_station_geometry(cmd: StationGeometryUpdate, session: SimulationSession = Depends(get_session)):
    """Update position/rotation/scale of a station geometry."""
    for geom in session.station_geometries:
        if geom["id"] == cmd.id:
            geom["position"] = cmd.position
            geom["rotation"] = cmd.rotation
            geom["scale"] = cmd.scale
            return {"ok": True, "geometry": geom}
    return {"ok": False, "error": "Geometry not found"}


@router.post("/delete_station_geometry")
def delete_station_geometry(cmd: DeleteStationGeometry, session: SimulationSession = Depends(get_session)):
    """Delete a station geometry."""
    for i, geom in enumerate(session.station_geometries):
        if geom["id"] == cmd.id:
            # Try to delete the file too
            try:
                file_path = STATION_DIR / geom["url"].split("/")[-1]
                if file_path.exists():
                    file_path.unlink()
            except Exception:
                pass
            session.station_geometries.pop(i)
            return {"ok": True}
    return {"ok": False, "error": "Geometry not found"}
