"""
robot_core/conversions.py - Joint Configuration Conversions
====================================================

This module provides conversion functions between different joint coordinate systems:
- Real model (with parallelogram mechanism)
- Pink model (simplified serial chain for IK)
- RobotStudio format (for frontend display)
"""

import pinocchio as pin
import numpy as np
from typing import Tuple, Optional


def q_real_to_pink(robot_type, model_real, q_real):
    """
    Convert from Real model (with parallelogram) to Pink model (serial chain).
    
    For 6-DOF: Returns [J1, J2, J3real, J4, J5, J6] - 6 elements
               Pink model uses J3 as the ABSOLUTE motor angle (J3real)
    For 4-DOF: Returns [J1, J2, J3, J_aux, J4] - 5 elements
               J3 is the RELATIVE elbow angle
               J_aux is calculated to maintain horizontality
    """
    if robot_type == "CR4":
        # Extract for Pink model: [J1, J2, J3, J_aux, J4]
        # J3 is the RELATIVE elbow angle (not J3real)
        try:
            idx = {name: model_real.getJointId(name) - 1 for name in ["J1", "J2", "J3", "J_aux", "J4"]}
            # J_aux = -(J2 + J3) for horizontality
            j_aux = -(q_real[idx["J2"]] + q_real[idx["J3"]])
            return np.array([
                q_real[idx["J1"]], 
                q_real[idx["J2"]], 
                q_real[idx["J3"]],   # Relative elbow angle
                j_aux,               # Calculated for horizontality
                q_real[idx["J4"]]    # Already inverted in q_real
            ])
        except Exception as e:
            print(f"Error in q_real_to_pink (4dof): {e}")
            return np.zeros(5)
    else:
        # 6-DOF: Extract [J1, J2, J3, J4, J5, J6]
        # J3 in Pink model is the RELATIVE angle (chained to J2)
        # NOT: absolute motor angle J3real
        try:
            idx = {name: model_real.getJointId(name) - 1 for name in ["J1", "J2", "J3", "J4", "J5", "J6"]}
            return np.array([
                q_real[idx["J1"]], 
                q_real[idx["J2"]], 
                q_real[idx["J3"]],  # Relative angle (J3_pink = J3_relativo)
                q_real[idx["J4"]], 
                q_real[idx["J5"]], 
                q_real[idx["J6"]]
            ])
        except Exception as e:
            print(f"Error in q_real_to_pink (6dof): {e}")
            return np.zeros(6)


def q_real_to_robotstudio(robot_type, model_real, q_real):
    """
    Convert from Real model to RobotStudio format (for frontend display).
    
    For 6-DOF: Returns [J1, J2, J3real, J4, J5, J6] - 6 elements
    For 4-DOF: Returns [J1, J2, J3real, J4] - 4 elements
               Note: J4 is INVERTED back to RobotStudio convention
    """
    if robot_type == "CR4":
        try:
            idx = {name: model_real.getJointId(name) - 1 for name in ["J1", "J2", "J3real", "J4"]}
            return np.array([
                q_real[idx["J1"]], 
                q_real[idx["J2"]], 
                q_real[idx["J3real"]],  # Absolute motor angle
                -q_real[idx["J4"]]      # Invert back to RobotStudio convention
            ])
        except Exception as e:
            print(f"Error in q_real_to_robotstudio (4dof): {e}")
            return np.zeros(4)
    else:
        try:
            idx = {name: model_real.getJointId(name) - 1 for name in ["J1", "J2", "J3real", "J4", "J5", "J6"]}
            return np.array([
                q_real[idx["J1"]], 
                q_real[idx["J2"]], 
                q_real[idx["J3real"]],
                q_real[idx["J4"]], 
                q_real[idx["J5"]], 
                q_real[idx["J6"]]
            ])
        except Exception as e:
            print(f"Error in q_real_to_robotstudio (6dof): {e}")
            return np.zeros(6)


def q_pink_to_real(robot_type, model_real, data_real, constraint_model, q_pink):
    """
    Convert from Pink model to Real model (with parallelogram).
    
    For 4-DOF: Input q_pink can be either:
        - [J1, J2, J3real, J4] (RobotStudio format, 4 elements)
        - [J1, J2, J3, J_aux, J4] (Pink model format, 5 elements)
    
    For 6-DOF: Input q_pink is [J1, J2, J3, J4, J5, J6]
        where J3 is the RELATIVE angle (chained to J2)
    """
    from .constants import solve_constraints
    
    q_real = pin.neutral(model_real)
    
    # Determine robot type from model name if possible, to avoid mismatch with robot_type string
    is_4dof = "CR4" in model_real.name or robot_type == "CR4"
    
    if is_4dof:
        # Input q_pink: [J1, J2, J3_pink, J4] where J3_pink = J3real (absolute angle)
        # Note: For 4-DOF, J4 is INVERTED (Z-down TCP convention)
        # J_aux is calculated automatically to maintain horizontality
        try:
            idx = {name: model_real.getJointId(name) - 1 for name in ["J1", "J2", "J3real", "J1p", "J3", "J2p", "J_aux", "J4"]}
            
            # Check if indices are valid for q_real
            if any(i >= len(q_real) for i in idx.values()):
                print(f"Warning: q_pink_to_real mismatch. Model {model_real.name} vs Type {robot_type}")
                return q_real
            
            # Handle q_pink with either 4 or 5 elements
            # If 4 elements: [J1, J2, J3real, J4] (RobotStudio format)
            # If 5 elements: [J1, J2, J3, J_aux, J4] (Pink model format)
            if len(q_pink) == 4:
                # RobotStudio format: [J1, J2, J3real, J4]
                q_real[idx["J1"]] = q_pink[0]
                q_real[idx["J2"]] = q_pink[1]
                q_real[idx["J3real"]] = q_pink[2]  # J3_pink = J3real (absolute)
                q_real[idx["J4"]] = -q_pink[3]     # INVERT J4 for Z-down TCP
                
                # Calculate relative angles for parallelogram
                # J3 (relative) = J3real - J2
                q_real[idx["J3"]] = q_pink[2] - q_pink[1]
                q_real[idx["J1p"]] = q_pink[1] - q_pink[2]
                q_real[idx["J2p"]] = 0.0
                
            else:
                # Pink model format: [J1, J2, J3, J_aux, J4]
                # Here J3 is already relative
                q_real[idx["J1"]] = q_pink[0]
                q_real[idx["J2"]] = q_pink[1]
                q_real[idx["J3"]] = q_pink[2]
                q_real[idx["J_aux"]] = q_pink[3]
                q_real[idx["J4"]] = q_pink[4]  # Already inverted if coming from Pink
                
                q_real[idx["J3real"]] = q_pink[1] + q_pink[2]
                q_real[idx["J1p"]] = -q_pink[2]
                q_real[idx["J2p"]] = 0.0
            
            # Solve parallelogram constraints
            q_real, _ = solve_constraints(model_real, data_real, constraint_model, q_real)
            
            # Calculate J_aux to maintain horizontality: θ_J_aux = -(θ_J2 + θ_J3)
            q_real[idx["J_aux"]] = -(q_real[idx["J2"]] + q_real[idx["J3"]])
            
        except Exception as e:
            print(f"Error in q_pink_to_real (4dof): {e}")
            return q_real
        
    else:
        # Pink: J1, J2, J3, J4, J5, J6
        # IMPORTANTE: En el modelo Pink, J3 está encadenado a J2 como joint serial
        # Por lo tanto q_pink[2] (J3_pink) es un ángulo RELATIVO al frame de salida de J2
        # En el modelo Real: J3real es el ángulo ABSOLUTO del motor, J3 es relativo
        # Relación: J3_pink = J3 (relativo), y J3real = J2 + J3_pink
        try:
            idx = {name: model_real.getJointId(name) - 1 for name in ["J1", "J2", "J3real", "J1p", "J3", "J2p", "J4", "J5", "J6"]}
            
            if any(i >= len(q_real) for i in idx.values()):
                 print(f"Warning: q_pink_to_real mismatch. Model {model_real.name} vs Type {robot_type}")
                 return q_real

            # Articulaciones activas directamente desde q_pink
            q_real[idx["J1"]] = q_pink[0]
            q_real[idx["J2"]] = q_pink[1]
            
            # J3_pink es RELATIVO (igual que J3 en el modelo real)
            # J3real (absoluto) = J2 + J3_pink
            q_real[idx["J3"]] = q_pink[2]  # J3_pink = J3 relativo
            q_real[idx["J3real"]] = q_pink[1] + q_pink[2]  # J3real = J2 + J3_relativo
            
            q_real[idx["J4"]] = q_pink[3]
            q_real[idx["J5"]] = q_pink[4]
            q_real[idx["J6"]] = q_pink[5]
            
            # Articulaciones pasivas del paralelogramo
            # J1p es opuesto a J3 relativo para mantener el paralelogramo
            q_real[idx["J1p"]] = -q_pink[2]
            q_real[idx["J2p"]] = 0.0
            
            q_real, _ = solve_constraints(model_real, data_real, constraint_model, q_real)
            
        except Exception as e:
            print(f"Error in q_pink_to_real (6dof): {e}")
            return q_real
        
    return q_real
