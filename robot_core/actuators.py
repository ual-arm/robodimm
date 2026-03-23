"""
robot_core/actuators.py - Actuator Selection
==========================================

This module provides functions for analyzing trajectory requirements and
selecting optimal motor+gearbox combinations.
"""

import numpy as np
from typing import Dict, List, Optional
from .constants import ACTIVE_JOINTS_CR6, ACTIVE_JOINTS_CR4, PARALLELOGRAM_TORQUE_COMBINATION_CR6, PARALLELOGRAM_TORQUE_COMBINATION_CR4


def analyze_trajectory_requirements(dynamics_data: dict, robot_type: str = "CR6") -> dict:
    """
    Analyze trajectory dynamics to extract torque/speed requirements per ACTIVE joint.
    
    For parallelogram-connected joints like J3 in CR6, motor torque is the
    sum of "real" joint torque plus parallelogram constraint force contribution.
    This is because the motor physically drives both the link and the
    parallelogram mechanism.
    
    Args:
        dynamics_data: Dictionary with 't', 'q', 'v', 'a', 'tau' arrays
        robot_type: "CR6" or "CR4" to determine which joints are active
        
    Returns:
        Dictionary with requirements for each active joint (J1-J6 or J1-J4)
    """
    if not dynamics_data or 'tau' not in dynamics_data or 'v' not in dynamics_data:
        return {}
    
    tau_array = np.array(dynamics_data['tau'])
    v_array = np.array(dynamics_data['v'])
    
    nv = tau_array.shape[1] if len(tau_array.shape) > 1 else 0
    if nv == 0:
        return {}
    
    # Select appropriate active joints mapping and parallelogram combination
    if robot_type == "CR4":
        active_joints = ACTIVE_JOINTS_CR4
        parallelogram_combo = PARALLELOGRAM_TORQUE_COMBINATION_CR4
    else:
        active_joints = ACTIVE_JOINTS_CR6
        parallelogram_combo = PARALLELOGRAM_TORQUE_COMBINATION_CR6
    
    requirements = {}
    for idx_v, joint_name in active_joints.items():
        if idx_v >= nv:
            continue
            
        # Get base torque for this active joint
        # Note: Since we use compute_motor_inverse_dynamics upstream, 
        # torque at J3real (index 2) ALREADY includes the parallelogram load via virtual work mapping.
        # So we just take the value directly.
        tau_j = tau_array[:, idx_v].copy()
        
        v_j = v_array[:, idx_v]
        
        peak_tau = float(np.max(np.abs(tau_j)))
        rms_tau = float(np.sqrt(np.mean(tau_j ** 2)))
        peak_v_rad = float(np.max(np.abs(v_j)))
        peak_v_rpm = peak_v_rad * 60.0 / (2.0 * np.pi)
        mean_v_rad = float(np.mean(np.abs(v_j)))
        mean_v_rpm = mean_v_rad * 60.0 / (2.0 * np.pi)
        
        requirements[joint_name] = {
            'peak_torque_Nm': round(peak_tau, 3),
            'rms_torque_Nm': round(rms_tau, 3),
            'peak_velocity_rad_s': round(peak_v_rad, 4),
            'peak_velocity_rpm': round(peak_v_rpm, 2),
            'mean_velocity_rpm': round(mean_v_rpm, 2),
            'internal_idx_v': idx_v,  # Keep track for debugging
            'includes_parallelogram': idx_v in parallelogram_combo  # Flag for documentation
        }
    
    return requirements


def select_actuators(
    requirements: dict,
    motors: list,
    gearboxes: list,
    compatibility_matrix: dict,
    safety_factor_torque: float = 1.5,
    safety_factor_speed: float = 1.2
) -> dict:
    """
    Select optimal motor+gearbox combinations for each joint based on requirements.
    
    Parameters
    ----------
    requirements : dict
        Requirements per joint from analyze_trajectory_requirements
    motors : list
        List of motor dictionaries with 'id', 'rated_torque_Nm', 'rated_speed_rpm', etc.
    gearboxes : list
        List of gearbox dictionaries with 'id', 'ratios', 'efficiency', etc.
    compatibility_matrix : dict
        Matrix mapping motor IDs to compatible gearbox IDs
    safety_factor_torque : float
        Safety factor for torque requirements
    safety_factor_speed : float
        Safety factor for speed requirements
        
    Returns
    -------
    dict
        Selection results with candidates and recommendations per joint
    """
    results = {}
    
    for joint_key, req in requirements.items():
        req_torque = req['peak_torque_Nm'] * safety_factor_torque
        req_speed_rpm = req['peak_velocity_rpm'] * safety_factor_speed
        
        candidates = []
        rejected_samples = []
        
        for motor in motors:
            motor_id = motor['id']
            # Try 'rated_' first (from JSON), then 'nominal_' (legacy/code fallback)
            motor_torque = motor.get('rated_torque_Nm', motor.get('nominal_torque_Nm', 0))
            motor_speed = motor.get('rated_speed_rpm', motor.get('nominal_speed_rpm', 0))
            
            if motor_torque <= 0 or motor_speed <= 0:
                continue
            
            # Fallback for compatibility: either from motor object or from central matrix
            compatible_gb_ids = motor.get('compatible_gearboxes', [])
            if not compatible_gb_ids and motor_id in compatibility_matrix:
                compatible_gb_ids = compatibility_matrix[motor_id]
            
            for gearbox in gearboxes:
                gb_id = gearbox['id']
                
                # Check compatibility (handle both list of IDs and more complex structures)
                is_compatible = False
                natural_match = False
                
                if gb_id in compatible_gb_ids:
                    is_compatible = True
                    # If matrix entry for this motor is a dict, check natural_match
                    if isinstance(compatibility_matrix.get(motor_id), dict):
                        natural_match = compatibility_matrix[motor_id].get(gb_id, {}).get('natural_match', False)
                    else:
                        # If it's just a list of compatible IDs, assume natural match
                        natural_match = True
                
                if not is_compatible:
                    continue
                
                # Support both 'ratios' (list) and 'ratio' (single value)
                ratios_to_test = gearbox.get('ratios', [])
                if not ratios_to_test and 'ratio' in gearbox:
                    ratios_to_test = [gearbox['ratio']]
                
                for ratio in ratios_to_test:
                    efficiency = gearbox.get('efficiency', 0.85) or 0.85
                    output_torque = motor_torque * ratio * efficiency
                    output_speed_rpm = motor_speed / ratio
                    
                    if output_torque >= req_torque and output_speed_rpm >= req_speed_rpm:
                        margin_torque = ((output_torque / req_torque) - 1) * 100 if req_torque > 0 else 999
                        margin_speed = ((output_speed_rpm / req_speed_rpm) - 1) * 100 if req_speed_rpm > 0 else 999
                        
                        score = 0
                        score += 0 if natural_match else 100
                        score += motor_torque * 5  # Favor smaller motors if they work
                        score += ratio * 0.1
                        score -= min(margin_torque, 100) * 0.2
                        
                        candidates.append({
                            'motor_id': motor_id,
                            'motor_desc': motor.get('name', motor_id),
                            'gearbox_id': gb_id,
                            'gearbox_desc': gearbox.get('name', gb_id),
                            'ratio': ratio,
                            'natural_match': natural_match,
                            'output_torque_Nm': round(output_torque, 2),
                            'max_output_speed_rpm': round(output_speed_rpm, 2),
                            'margin_torque_pct': round(margin_torque, 1),
                            'margin_speed_pct': round(margin_speed, 1),
                            'score': round(score, 2),
                            'motor_mass_kg': motor.get('mass_kg'),
                            'gearbox_mass_kg': gearbox.get('mass_kg')
                        })
                    else:
                        if len(rejected_samples) < 3:
                            fail_reason = []
                            if output_torque < req_torque:
                                fail_reason.append(f"torque: {output_torque:.1f} < {req_torque:.1f} Nm")
                            if output_speed_rpm < req_speed_rpm:
                                fail_reason.append(f"speed: {output_speed_rpm:.1f} < {req_speed_rpm:.1f} rpm")
                            rejected_samples.append({
                                'motor_id': motor_id,
                                'gearbox_id': gb_id,
                                'ratio': ratio,
                                'output_torque_Nm': round(output_torque, 2),
                                'max_output_speed_rpm': round(output_speed_rpm, 2),
                                'fail_reason': ', '.join(fail_reason)
                            })
        
        candidates.sort(key=lambda x: x['score'])
        top_candidates = candidates[:5] if len(candidates) > 5 else candidates
        
        results[joint_key] = {
            'required': {
                'torque_Nm': round(req_torque, 3),
                'speed_rpm': round(req_speed_rpm, 2),
                'original_peak_torque_Nm': req['peak_torque_Nm'],
                'original_peak_speed_rpm': req['peak_velocity_rpm'],
                'mean_velocity_rpm': req.get('mean_velocity_rpm', 0)
            },
            'candidates': top_candidates,
            'rejected_samples': rejected_samples,
            'recommended': top_candidates[0] if top_candidates else None
        }
    
    return results


def get_actuator_masses(selection: dict, motors: list, gearboxes: list) -> dict:
    """
    Get masses of selected actuators.
    
    Parameters
    ----------
    selection : dict
        Selection results from select_actuators
    motors : list
        List of motor dictionaries
    gearboxes : list
        List of gearbox dictionaries
        
    Returns
    -------
    dict
        Masses per joint (motor, gearbox, total)
    """
    motor_dict = {m['id']: m for m in motors}
    gearbox_dict = {g['id']: g for g in gearboxes}
    
    masses = {}
    
    for joint_key, data in selection.items():
        rec = data.get('recommended')
        if not rec:
            masses[joint_key] = {'motor_mass_kg': None, 'gearbox_mass_kg': None, 'total_kg': None}
            continue
        
        motor = motor_dict.get(rec['motor_id'], {})
        gearbox = gearbox_dict.get(rec['gearbox_id'], {})
        
        motor_mass = motor.get('mass_kg')
        gearbox_mass = gearbox.get('mass_kg')
        
        total = None
        if motor_mass is not None and gearbox_mass is not None:
            total = motor_mass + gearbox_mass
        
        masses[joint_key] = {
            'motor_mass_kg': motor_mass,
            'gearbox_mass_kg': gearbox_mass,
            'total_kg': total
        }
    
    return masses
