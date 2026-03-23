"""
robot_info.py - Robot Information Endpoints
========================================

Endpoints for robot structure and placement queries.
"""

from fastapi import APIRouter, Depends
import numpy as np
import logging
import pinocchio as pin

from ..session import SimulationSession, get_session
from ..utils import get_frontend_nq
from robot_core import q_pink_to_real


# =============================================================================
# ROUTER SETUP
# =============================================================================

router = APIRouter()


# =============================================================================
# ROBOT INFO ENDPOINTS
# =============================================================================

@router.get("/robot_info")
def robot_info(session: SimulationSession = Depends(get_session)):
    """Export robot structure: links, joints, geometries for frontend rendering."""
    links = []
    joints = []
    
    model = session.model
    geom_model = session.geom_model

    # Iterate over model joints (skip root joint at index 0)
    for i in range(1, model.njoints):
        joint = model.joints[i]
        parent_id = model.parents[i]
        
        # Get placement relative to parent
        placement = model.jointPlacements[i]
        pos = placement.translation.tolist()
        rot = placement.rotation.flatten().tolist()  # row-major
        
        joints.append({
            "id": i,
            "name": model.names[i],
            "parent_id": parent_id,
            "position": pos,
            "rotation": rot,
            "type": str(joint)  # Joint type as string
        })
    
    # Get geometries (shape metadata only)
    geoms = []
    for i, geom_obj in enumerate(geom_model.geometryObjects):
        placement = geom_obj.placement
        # Pinocchio already uses Z-up (same as Three.js with camera.up=[0,0,1])
        # No conversion needed
        try:
            p = np.array(placement.translation)
            R = np.array(placement.rotation)
            p_three = p.tolist()
            R_three = R.flatten().tolist()
        except Exception as e:
            logging.warning(f"Error in /robot_info geom {i}: {e}")
            p_three = [0.0, 0.0, 0.0]
            R_three = [1,0,0,0,1,0,0,0,1]

        # Determine shape info safely
        shape_info = {"type": "unknown"}
        shape = getattr(geom_obj, 'geometry', None)
        try:
            if shape is not None and hasattr(shape, 'radius') and hasattr(shape, 'halfLength'):
                # treat as cylinder-like
                shape_info = {
                    "type": "cylinder",
                    "radius": float(getattr(shape, 'radius')),
                    "height": float(getattr(shape, 'halfLength')) * 2.0
                }
            elif shape is not None and hasattr(shape, 'radius') and not hasattr(shape, 'halfLength'):
                # treat as sphere
                shape_info = {
                    "type": "sphere",
                    "radius": float(getattr(shape, 'radius'))
                }
            elif shape is not None and hasattr(shape, 'halfSide'):
                hs = getattr(shape, 'halfSide')
                try:
                    hs_list = list(hs)
                except Exception:
                    hs_list = [float(hs), float(hs), float(hs)]
                shape_info = {"type": "box", "halfSide": hs_list}
        except Exception:
            shape_info = {"type": "unknown"}

        mesh_url = None
        

        # CR4 Mapping (Specific folder, .glb extensions)
        MESH_MAPPING_CR4 = {
            "static_base": "meshes/CR4/base.glb",
            "L1": "meshes/CR4/link1.glb", 
            "L2": "meshes/CR4/link2.glb",
            "L3": "meshes/CR4/link3.glb", 
            "L4": "meshes/CR4/link4.glb", 
            "Crank": "meshes/CR4/crank.glb",
            "Rod": "meshes/CR4/rod.glb",
        }
        
        # Redundant geometries to skip when using custom meshes (only markers/axes)
        MESH_IGNORE_4DOF = {"tcp_x", "tcp_y", "tcp_z", "base_x", "base_y", "base_z"}

        # CR6 Mapping (Specific folder, .glb extensions)
        MESH_MAPPING_CR6 = {
            "static_base": "meshes/CR6/base.glb",
            "link1": "meshes/CR6/link1.glb", 
            "link2": "meshes/CR6/link2.glb",
            "crank": "meshes/CR6/crank.glb",
            "rod": "meshes/CR6/rod.glb",
            "link3": "meshes/CR6/link3.glb", 
            "link4": "meshes/CR6/link4.glb", 
            "link5": "meshes/CR6/link5.glb",
        }
        
        MESH_IGNORE_6DOF = {"tcp_x", "tcp_y", "tcp_z", "base_x", "base_y", "base_z"}

        geom_name = getattr(geom_obj, 'name', '')
        
        # Check explicit mapping
        if session.current_robot_type == "CR4":
             if geom_name in MESH_IGNORE_4DOF:
                 continue 
             if geom_name in MESH_MAPPING_CR4:
                 mesh_url = MESH_MAPPING_CR4[geom_name]
        
        else: # Default CR6
             if geom_name in MESH_IGNORE_6DOF:
                 continue
             if geom_name in MESH_MAPPING_CR6:
                 mesh_url = MESH_MAPPING_CR6[geom_name]

        # Legacy/Fallback overrides
        if not mesh_url and geom_name in {"g_shoulder", "g_elbow"}:
            mesh_url = "meshes/prisma1.gltf"

        # Get color from meshColor attribute (RGBA)
        color = None
        if hasattr(geom_obj, 'meshColor') and geom_obj.meshColor is not None:
            mc = geom_obj.meshColor
            if len(mc) >= 3:
                color = [float(mc[0]), float(mc[1]), float(mc[2]), float(mc[3]) if len(mc) > 3 else 1.0]

        geoms.append({
            "id": i,
            "name": geom_name if geom_name else f'geom_{i}',
            "link_id": getattr(geom_obj, 'parentJoint', None),
            "position": p_three,
            "rotation": R_three,
            "shape": shape_info,
            "mesh_url": mesh_url,
            "color": color
        })
    
    return {
        "nq": get_frontend_nq(session.current_robot_type, session.model_pink),
        "num_joints": model.njoints,
        "joints": joints,
        "geometries": geoms,
        "robot_type": session.current_robot_type,
        "scale": session.current_scale,
        "ee_always_down": session.ee_always_down,
        "dimensions": session.core.get('dimensions', {})
    }


@router.get("/robot_placements")
def robot_placements(session: SimulationSession = Depends(get_session)):
    """Return world placements for each geometry using current configuration.q."""
    
    # Access session state
    configuration = session.configuration
    model = session.model
    data = session.data
    constraint_model = session.constraint_model
    geom_model = session.geom_model
    
    # ensure FK uses the latest configuration
    q_pink = configuration.q
    q_real = q_pink_to_real(session.current_robot_type, model, data, constraint_model, q_pink)
        
    pin.forwardKinematics(model, data, q_real)
    pin.updateFramePlacements(model, data)

    # Conversion matrix from Pinocchio to Three.js:
    # Pinocchio already uses Z-up (same as Three.js), so use identity (no conversion)
    C = np.eye(3)

    # GLTF mapping (must match /robot_info)
    MESH_IGNORE_4DOF = {"L3_back", "base_cyl"}

    placements = []
    for i, geom_obj in enumerate(geom_model.geometryObjects):
        geom_name = getattr(geom_obj, 'name', '')
        
        # Skip ignored geometries for 4DOF
        if session.current_robot_type == "CR4" and geom_name in MESH_IGNORE_4DOF:
            continue
            
        parent = geom_obj.parentJoint
        p_three = np.array([0.0, 0.0, 0.0])
        R_three = np.eye(3)
        
        if parent < len(data.oMi):
            try:
                # world placement in pin frame
                joint_placement = data.oMi[parent]
                world = joint_placement * geom_obj.placement
                p_three = np.array(world.translation)
                R_three = np.array(world.rotation)
            except Exception as e:
                logging.warning(f"Error computing world placement for geom {i}: {e}")
                try:
                    p_three = np.array(geom_obj.placement.translation)
                    R_three = np.array(geom_obj.placement.rotation)
                except Exception:
                    pass
        else:
            try:
                p_three = np.array(geom_obj.placement.translation)
                R_three = np.array(geom_obj.placement.rotation)
            except Exception:
                pass

        placements.append({
            "id": i,
            "name": geom_name,
            "position": p_three.tolist(),
            "rotation": R_three.flatten().tolist()
        })

    # Provide the actual TCP pose (end_effector frame) so the frontend can
    # place the TCP consistently (JOG, targets, and program playback).
    try:
        fe = data.oMf[model.getFrameId('end_effector')]
        ee_pos = np.array(fe.translation)
        ee_rot = np.array(fe.rotation)
        ee_pos_list = ee_pos.tolist()
        ee_rot_list = ee_rot.flatten().tolist()
    except Exception as e:
        logging.warning(f"Error computing end_effector pose: {e}")
        ee_pos_list = [0.0, 0.0, 0.0]
        ee_rot_list = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

    return {"placements": placements, "ee_pos": ee_pos_list, "ee_rot": ee_rot_list}
