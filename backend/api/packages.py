import uuid
import json
import shutil
from typing import List
from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, File, Request

router = APIRouter()

PACKAGES_DIR = Path.home() / ".robodimm" / "packages"
PACKAGES_DIR.mkdir(parents=True, exist_ok=True)

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

def validate_package_relative_path(path: str) -> Path:
    normalized = path.replace("\\", "/")
    rel_path = Path(normalized)
    if rel_path.is_absolute() or any(part in {"", ".", ".."} for part in rel_path.parts):
        raise HTTPException(status_code=400, detail=f"Invalid package path: {path}")
    allowed_extensions = {".glb", ".gltf", ".bin", ".stl", ".json"}
    if rel_path.suffix.lower() not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File extension {rel_path.suffix} not allowed. Supported: {', '.join(allowed_extensions)}"
        )
    return rel_path

@router.post("/packages/upload")
async def upload_package(request: Request, files: List[UploadFile] = File(...)):
    package_id = str(uuid.uuid4())
    package_path = PACKAGES_DIR / package_id
    package_path.mkdir(parents=True, exist_ok=True)

    saved_files = []
    try:
        for file in files:
            filename = file.filename
            if not filename:
                continue
                
            rel_path = validate_package_relative_path(filename)

            target_file_path = package_path / rel_path
            target_file_path.parent.mkdir(parents=True, exist_ok=True)

            # Read and enforce file size limit
            content = await file.read()
            if len(content) > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=413, 
                    detail=f"File '{filename}' size ({len(content)} bytes) exceeds maximum limit of 50MB"
                )

            with open(target_file_path, "wb") as f:
                f.write(content)
            saved_files.append((rel_path.as_posix(), target_file_path))

        robot_json_path = None
        for rel_name, abs_path in saved_files:
            if rel_name.endswith("robot.json"):
                robot_json_path = abs_path
                break

        if not robot_json_path:
            raise HTTPException(status_code=400, detail="Package upload failed: No 'robot.json' found.")

        try:
            with open(robot_json_path, "r", encoding="utf-8") as f:
                package_data = json.load(f)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse robot.json: {str(e)}")

        if package_data.get("schema") != "robodimm.package.v1":
            raise HTTPException(status_code=400, detail="robot.json is not a valid robodimm.package.v1 package")

        robot_spec = package_data.get("robot")
        if not isinstance(robot_spec, dict):
            raise HTTPException(status_code=400, detail="robot.json package is missing a valid robot spec")

        assets = package_data.get("assets", {})
        meshes = assets.get("meshes", [])
        
        updated_visuals = []
        for vis in robot_spec.get("visuals", []):
            body = vis.get("body")
            mesh_asset = next((m for m in meshes if m.get("body") == body), None)
            if mesh_asset:
                rel_path = validate_package_relative_path(mesh_asset.get("path", ""))
                asset_path = package_path / rel_path
                if not asset_path.exists():
                    raise HTTPException(status_code=400, detail=f"Mesh asset not found: {rel_path.as_posix()}")
                
                static_path = f"{package_id}/{rel_path.as_posix()}"
                
                # Check for port 8001 vs port 8000 and schema routing
                # In FastAPI, request.url_for generates URLs based on current host/port.
                # Since we route through 'packages_static', we specify that name.
                static_url = str(request.url_for("packages_static", path=static_path))
                vis["kind"] = "mesh"
                vis["meshUrl"] = static_url
                vis["originM"] = [0.0, 0.0, 0.0]
                vis["rpyRad"] = [0.0, 0.0, 0.0]
                vis["scale"] = [1.0, 1.0, 1.0]
            updated_visuals.append(vis)

        robot_spec["visuals"] = updated_visuals
        
        with open(robot_json_path, "w", encoding="utf-8") as f:
            json.dump(package_data, f, indent=2)

        return robot_spec

    except HTTPException as e:
        shutil.rmtree(package_path, ignore_errors=True)
        raise e
    except Exception as e:
        shutil.rmtree(package_path, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
