"""
robot_core/builders/cr6.py - CR6 Robot Builders
===========================================

This module provides robot model builders for CR6 6-DOF robot with spherical wrist.
"""

import pinocchio as pin
import numpy as np
from .base import add_geom, get_rotation_matrices, create_tcp_axes, add_static_base, create_parallelogram_constraint, add_payload


def build_cr6_real(scale=1.0, payload_kg=0.0, payload_inertia=None):
    """
    Build CR6 real robot model with parallelogram mechanism.
    
    Parameters
    ----------
    scale : float
        Scaling factor for all dimensions
    payload_kg : float
        Payload mass in kg at TCP
    payload_inertia : dict, optional
        Custom inertia tensor for payload
        
    Returns
    -------
    model : pin.Model
        The robot model
    geom_model : pin.GeometryModel
        The geometry model
    constraint_model : pin.RigidConstraintModel
        The parallelogram constraint model
    """
    model = pin.Model()
    model.name = "CR6"
    geom_model = pin.GeometryModel()
    
    # Dimensions (CR6 Adapted - 1:1 Scale)
    # Base chain matches CR4 exactly to reuse components
    L1_X = 0.0 * scale          # CR4 is 0.0 (Concentric)
    L1_Z = 0.400 * scale        # CR4 Base Height
    L2_Z = 0.540 * scale        # CR4 Arm Length
    L_CRANK = 0.200 * scale     # CR4 Crank
    L_ROD = 0.540 * scale       # CR4 Rod
    D_PARA = 0.200 * scale      # Matches Crank
    
    # Upper arm (L3..L6) - Rounded Values
    L3_X = 0.200 * scale        # 197.6 -> 200
    L3_Z = 0.100 * scale        # 103.4 -> 100
    L4_X = 0.380 * scale        # 380.7 -> 380
    L5_X = 0.065 * scale        # 65.1 -> 65
    
    # Masses (Collaborative design)
    s3 = scale**3
    M_LINK1 = 10.0 * s3 
    M_LINK2 = 8.0 * s3
    M_LINK3 = 4.0 * s3
    M_LINK4 = 2.0 * s3
    M_LINK5 = 1.0 * s3
    M_LINK6 = 0.5 * s3
    M_CRANK = 1.0 * s3
    M_ROD = 1.0 * s3

    R = get_rotation_matrices()
    
    # --- J1: Base (Z) ---
    j1_id = model.addJoint(0, pin.JointModelRZ(), pin.SE3.Identity(), "J1")
    model.appendBodyToJoint(j1_id, pin.Inertia.FromBox(M_LINK1, 0.2*scale, 0.2*scale, L1_Z), pin.SE3(np.eye(3), np.array([0,0,L1_Z/2])))
    
    # Static floor base (fixed to universe)
    add_static_base(geom_model, scale)
    
    # Rotating column (Link 1)
    # Matching CR4 definition (origin at bottom) to reuse mesh
    add_geom("link1", j1_id, pin.SE3.Identity(), 
             pin.hppfcl.Cylinder(0.02*scale, L1_Z), [0.9, 0.47, 0.08, 1.0], geom_model)

    # --- J2: Shoulder (Y) ---
    j2_placement = pin.SE3(np.eye(3), np.array([L1_X, 0.0, L1_Z]))
    j2_id = model.addJoint(j1_id, pin.JointModelRY(), j2_placement, "J2")
    model.appendBodyToJoint(j2_id, pin.Inertia.FromBox(M_LINK2, 0.05*scale, 0.05*scale, L2_Z), pin.SE3(np.eye(3), np.array([0,0,L2_Z/2])))
    add_geom("link2", j2_id, pin.SE3(np.eye(3), np.array([0,0,L2_Z/2])), pin.hppfcl.Cylinder(0.015*scale, L2_Z), [0.9, 0.47, 0.08, 1.0], geom_model)

    # --- J3real: Elbow motor (Y) ---
    j3real_placement = pin.SE3(np.eye(3), np.array([L1_X, 0.0, L1_Z]))
    j3real_id = model.addJoint(j1_id, pin.JointModelRY(), j3real_placement, "J3real")
    model.appendBodyToJoint(j3real_id, pin.Inertia.FromBox(M_CRANK, 0.03*scale, 0.03*scale, L_CRANK), pin.SE3(np.eye(3), np.array([-L_CRANK/2, 0, 0])))
    add_geom("crank", j3real_id, pin.SE3(R['Rnx'], np.array([-L_CRANK/2,0,0])), pin.hppfcl.Cylinder(0.01*scale, L_CRANK), [0.2, 0.5, 1.0, 1.0], geom_model)

    # --- J1p: Crank (Y) ---
    j1p_placement = pin.SE3(np.eye(3), np.array([-L_CRANK, 0.0, 0.0]))
    j1p_id = model.addJoint(j3real_id, pin.JointModelRY(), j1p_placement, "J1p")
    model.appendBodyToJoint(j1p_id, pin.Inertia.FromBox(M_ROD, 0.02*scale, 0.02*scale, L_ROD), pin.SE3(np.eye(3), np.array([0,0,L_ROD/2])))
    add_geom("rod", j1p_id, pin.SE3(np.eye(3), np.array([0,0,L_ROD/2])), pin.hppfcl.Cylinder(0.01*scale, L_ROD), [0.2, 0.5, 1.0, 1.0], geom_model)

    # --- J3: Elbow (Y) ---
    j3_placement = pin.SE3(np.eye(3), np.array([0.0, 0.0, L2_Z]))
    j3_id = model.addJoint(j2_id, pin.JointModelRY(), j3_placement, "J3")
    model.appendBodyToJoint(j3_id, pin.Inertia.FromBox(M_LINK3, L3_X, 0.05*scale, L3_Z), pin.SE3(np.eye(3), np.array([L3_X/2, 0, L3_Z/2])))
    
    # Combined upper arm solid (L3 + Parallelogram top connection)
    # Origin at J3 joint (Identity placement) for direct CAD mesh import
    add_geom("link3", j3_id, pin.SE3.Identity(), 
             pin.hppfcl.Cylinder(0.02*scale, 0.04*scale), [1.0, 0.9, 0.0, 1.0], geom_model)

    # --- J2p: Connecting rod (Y) ---
    j2p_placement = pin.SE3(np.eye(3), np.array([-D_PARA, 0.0, 0.0]))
    j2p_id = model.addJoint(j3_id, pin.JointModelRY(), j2p_placement, "J2p")
    model.appendBodyToJoint(j2p_id, pin.Inertia.FromBox(0.1 * s3, 0.05*scale, 0.05*scale, 0.05*scale), pin.SE3.Identity())
    # Visual rod2 removed, now part of link3 solid design


    # --- J4: Wrist Rotate (X) ---
    j4_placement = pin.SE3(np.eye(3), np.array([L3_X, 0.0, L3_Z]))
    j4_id = model.addJoint(j3_id, pin.JointModelRX(), j4_placement, "J4")
    model.appendBodyToJoint(j4_id, pin.Inertia.FromBox(M_LINK4, L4_X, 0.04*scale, 0.04*scale), pin.SE3(np.eye(3), np.array([L4_X/2, 0, 0])))
    add_geom("link4", j4_id, pin.SE3(R['Rx'], np.array([L4_X/2, 0, 0])), pin.hppfcl.Cylinder(0.007*scale, L4_X), [0.9, 0.47, 0.08, 1.0], geom_model)

    # --- J5: Wrist Tilt (Y) ---
    j5_placement = pin.SE3(np.eye(3), np.array([L4_X, 0.0, 0.0]))
    j5_id = model.addJoint(j4_id, pin.JointModelRY(), j5_placement, "J5")
    model.appendBodyToJoint(j5_id, pin.Inertia.FromBox(M_LINK5, L5_X, 0.03*scale, 0.03*scale), pin.SE3(np.eye(3), np.array([L5_X/2, 0, 0])))
    add_geom("link5", j5_id, pin.SE3(R['Rx'], np.array([L5_X/2, 0, 0])), pin.hppfcl.Cylinder(0.006*scale, L5_X), [0.5, 0.5, 0.5, 1.0], geom_model)

    # --- J6: Tool Rotate (X) ---
    j6_placement = pin.SE3(np.eye(3), np.array([L5_X, 0.0, 0.0]))
    j6_id = model.addJoint(j5_id, pin.JointModelRX(), j6_placement, "J6")
    
    # Payload
    add_payload(model, j6_id, payload_kg, payload_inertia)

    # --- Frame TCP ---
    tool0_placement = pin.SE3(pin.utils.rotate('y', np.pi/2), np.array([0.0, 0.0, 0.0]))
    model.addFrame(pin.Frame('tool0', j6_id, tool0_placement, pin.FrameType.OP_FRAME))
    model.addFrame(pin.Frame('end_effector', j6_id, tool0_placement, pin.FrameType.OP_FRAME))

    # --- Parallelogram Constraint ---
    constraint_model = create_parallelogram_constraint(model, j1p_id, j2p_id, L_ROD)
    
    return model, geom_model, constraint_model


def build_cr6_pink(scale=1.0):
    """
    Build CR6 Pink model (simplified serial chain for IK).
    
    Parameters
    ----------
    scale : float
        Scaling factor for all dimensions
        
    Returns
    -------
    model : pin.Model
        The simplified robot model
    """
    model = pin.Model()
    model.name = "CR6_Pink"
    
    # Dimensions (CR6 Adapted - 1:1 Scale)
    L1_X = 0.0 * scale
    L1_Z = 0.400 * scale
    L2_Z = 0.540 * scale
    
    # Upper arm (L3..L6) - Rounded Values
    L3_X = 0.200 * scale        # 197.6 -> 200
    L3_Z = 0.100 * scale        # 103.4 -> 100
    L4_X = 0.380 * scale        # 380.7 -> 380
    L5_X = 0.065 * scale        # 65.1 -> 65
    
    # J1
    j1_id = model.addJoint(0, pin.JointModelRZ(), pin.SE3.Identity(), "J1")
    model.appendBodyToJoint(j1_id, pin.Inertia.FromBox(1.0, 0.1, 0.1, 0.1), pin.SE3.Identity())
    
    # J2
    j2_pl = pin.SE3(np.eye(3), np.array([L1_X, 0, L1_Z]))
    j2_id = model.addJoint(j1_id, pin.JointModelRY(), j2_pl, "J2")
    model.appendBodyToJoint(j2_id, pin.Inertia.FromBox(1.0, 0.1, 0.1, 0.1), pin.SE3.Identity())
    
    # J3 (Active)
    j3_pl = pin.SE3(np.eye(3), np.array([0,0,L2_Z]))
    j3_id = model.addJoint(j2_id, pin.JointModelRY(), j3_pl, "J3")
    model.appendBodyToJoint(j3_id, pin.Inertia.FromBox(1.0, 0.1, 0.1, 0.1), pin.SE3.Identity())
    
    # J4
    j4_pl = pin.SE3(np.eye(3), np.array([L3_X, 0, L3_Z]))
    j4_id = model.addJoint(j3_id, pin.JointModelRX(), j4_pl, "J4")
    model.appendBodyToJoint(j4_id, pin.Inertia.FromBox(1.0, 0.1, 0.1, 0.1), pin.SE3.Identity())
    
    # J5
    j5_pl = pin.SE3(np.eye(3), np.array([L4_X, 0, 0]))
    j5_id = model.addJoint(j4_id, pin.JointModelRY(), j5_pl, "J5")
    model.appendBodyToJoint(j5_id, pin.Inertia.FromBox(1.0, 0.1, 0.1, 0.1), pin.SE3.Identity())
    
    # J6
    j6_pl = pin.SE3(np.eye(3), np.array([L5_X, 0, 0]))
    j6_id = model.addJoint(j5_id, pin.JointModelRX(), j6_pl, "J6")
    model.appendBodyToJoint(j6_id, pin.Inertia.FromBox(1.0, 0.1, 0.1, 0.1), pin.SE3.Identity())
    
    # Frame TCP
    tool0_pl = pin.SE3(pin.utils.rotate('y', np.pi/2), np.array([0.0, 0.0, 0.0]))
    model.addFrame(pin.Frame('end_effector', j6_id, tool0_pl, pin.FrameType.OP_FRAME))
    
    return model
