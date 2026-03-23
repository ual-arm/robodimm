"""
robot_core/dynamics/trajectory.py - Trajectory Inverse Dynamics
========================================================

This module provides inverse dynamics analysis for full robot trajectories.
"""

import pinocchio as pin
import numpy as np
from typing import Dict, List, Optional
from .constrained import compute_constrained_inverse_dynamics, compute_motor_inverse_dynamics


def smooth_array(arr, window=5):
    """
    Apply moving average smoothing to an array.
    
    Parameters
    ----------
    arr : np.ndarray
        Input array to smooth
    window : int
        Window size for moving average
        
    Returns
    -------
    smoothed : np.ndarray
        Smoothed array
    """
    if len(arr) < window:
        return arr
    result = np.copy(arr)
    half = window // 2
    for i in range(half, len(arr) - half):
        result[i] = np.mean(arr[i-half:i+half+1], axis=0)
    return result


def compute_constrained_inverse_dynamics_trajectory(model, data, constraint_model, 
                                                      trajectory, dt, friction_coeffs=None,
                                                      return_analysis=False):
    """
    Compute inverse dynamics for a full trajectory using closed-loop constraint formulation.
    
    This is the main function for analyzing robot trajectories with parallelogram
    or other closed kinematic chain mechanisms. It properly accounts for the
    internal forces in closed loops using Lagrange multipliers.
    
    Parameters
    ----------
    model : pinocchio.Model
        The robot model with all joints (including passive ones)
    data : pinocchio.Data
        Data structure for model
    constraint_model : pinocchio.RigidConstraintModel or list
        The constraint model(s) defining closed loop(s)
    trajectory : list of np.ndarray
        List of joint configurations [q1, q2, ..., qn]
    dt : float
        Time step between trajectory points
    friction_coeffs : list or np.ndarray, optional
        Viscous friction coefficients per joint
    return_analysis : bool, optional
        If True, return additional analysis data including constraint forces
        
    Returns
    -------
    dict
        Dictionary containing:
        - 't': Time vector
        - 'q': Joint positions
        - 'v': Joint velocities  
        - 'a': Joint accelerations
        - 'tau': Joint torques (constrained inverse dynamics)
        - 'tau_friction': Friction torques
        - 'tau_rnea': Open-chain RNEA torques (for comparison)
        If return_analysis=True, also includes:
        - 'lambda': Constraint forces (Lagrange multipliers)
        - 'constraint_violation': Position constraint violations
        - 'torque_correction': Difference between constrained and RNEA torques
        
    Notes
    -----
    This function:
    1. Numerically differentiates positions to get velocities and accelerations
    2. Applies smoothing to reduce numerical noise
    3. Computes constrained inverse dynamics at each point using KKT formulation
    4. Optionally compares with standard RNEA to show the effect of constraints
    
    The difference between 'tau' and 'tau_rnea' represents the contribution of
    constraint forces to joint torques - this is what makes 
    closed-loop analysis physically meaningful.
    """
    n = len(trajectory)
    nv = model.nv
    
    if friction_coeffs is None:
        friction_coeffs = [0.0] * nv
    elif len(friction_coeffs) < nv:
        friction_coeffs = list(friction_coeffs) + [0.0] * (nv - len(friction_coeffs))
    
    friction_coeffs = np.array(friction_coeffs[:nv])
    
    # Convert trajectory to array and smooth
    q_array = np.array([np.array(q) for q in trajectory])
    q_smooth = smooth_array(q_array, window=5)
    
    # Numerical differentiation for velocities
    v_array = np.zeros((n, nv))
    for i in range(1, n - 1):
        v_array[i] = (q_smooth[i + 1] - q_smooth[i - 1]) / (2 * dt)
    if n > 1:
        v_array[0] = (q_smooth[1] - q_smooth[0]) / dt
        v_array[-1] = (q_smooth[-1] - q_smooth[-2]) / dt
    
    v_smooth = smooth_array(v_array, window=7)
    
    # Numerical differentiation for accelerations
    a_array = np.zeros((n, nv))
    for i in range(1, n - 1):
        a_array[i] = (v_smooth[i + 1] - v_smooth[i - 1]) / (2 * dt)
    if n > 1:
        a_array[0] = (v_smooth[1] - v_smooth[0]) / dt if n > 1 else np.zeros(nv)
        a_array[-1] = (v_smooth[-1] - v_smooth[-2]) / dt if n > 1 else np.zeros(nv)
    
    a_smooth = smooth_array(a_array, window=9)
    
    # Initialize output arrays
    times = []
    positions = []
    velocities = []
    accelerations = []
    torques = []
    friction_torques = []
    torques_rnea = []
    lambda_history = []
    violation_history = []
    
    # Determine constraint dimension
    if constraint_model is not None:
        if isinstance(constraint_model, list):
            nc = sum(cm.size() for cm in constraint_model if cm is not None)
        else:
            nc = constraint_model.size()
    else:
        nc = 0
    
    for i in range(n):
        t = i * dt
        q_curr = q_smooth[i]
        v_curr = v_smooth[i]
        a_curr = a_smooth[i]
        
        # Compute constrained inverse dynamics
        if constraint_model is not None and return_analysis:
            tau_constrained, lambda_, violation = compute_constrained_inverse_dynamics(
                model, data, constraint_model, q_curr, v_curr, a_curr,
                return_multipliers=True
            )
            lambda_history.append(lambda_.tolist() if len(lambda_) > 0 else [0.0] * nc)
            violation_history.append(violation)
        elif constraint_model is not None:
            tau_constrained = compute_constrained_inverse_dynamics(
                model, data, constraint_model, q_curr, v_curr, a_curr
            )
        else:
            tau_constrained = pin.rnea(model, data, q_curr, v_curr, a_curr)
        
        # Also compute standard RNEA for comparison
        tau_rnea_val = pin.rnea(model, data, q_curr, v_curr, a_curr)
        
        # Add friction
        tau_friction = friction_coeffs * v_curr
        tau_total = tau_constrained + tau_friction
        
        times.append(t)
        positions.append(q_curr.tolist())
        velocities.append(v_curr.tolist())
        accelerations.append(a_curr.tolist())
        torques.append(tau_total.tolist())
        friction_torques.append(tau_friction.tolist())
        torques_rnea.append(tau_rnea_val.tolist())
    
    # Smooth torques
    tau_array = np.array(torques)
    tau_smooth = smooth_array(tau_array, window=7)
    torques = tau_smooth.tolist()
    
    result = {
        't': times,
        'q': positions,
        'v': velocities,
        'a': accelerations,
        'tau': torques,
        'tau_friction': friction_torques,
        'tau_rnea': torques_rnea,  # Open-chain comparison
    }
    
    if return_analysis:
        tau_rnea_array = np.array(torques_rnea)
        result['lambda'] = lambda_history
        result['constraint_violation'] = violation_history
        result['torque_correction'] = (tau_smooth - tau_rnea_array).tolist()
    
    return result


def compute_inverse_dynamics_trajectory(model, data, trajectory, dt, friction_coeffs=None,
                                        constraint_model=None, use_constrained_dynamics=True):
    """
    Compute full inverse dynamics analysis for a joint trajectory.
    
    This function supports two modes:
    
    1. CONSTRAINED DYNAMICS (use_constrained_dynamics=True, default):
       Uses rigorous KKT formulation with Lagrange multipliers to properly
       account for internal forces in closed kinematic chains. This is the
       physically correct approach for parallelogram mechanisms.
        
    2. HEURISTIC CORRECTION (use_constrained_dynamics=False):
       Uses standard RNEA with a post-hoc correction that assigns the passive
       joint torque to the motor. This is faster but less accurate.
    
    Parameters
    ----------
    model : pinocchio.Model
        The robot model with all joints
    data : pinocchio.Data
        Model data structure
    trajectory : list of np.ndarray
        Joint configuration trajectory
    dt : float
        Time step
    friction_coeffs : list or np.ndarray, optional
        Viscous friction coefficients per joint
    constraint_model : pinocchio.RigidConstraintModel, optional
        Constraint model for closed loops. Required for constrained dynamics.
    use_constrained_dynamics : bool, optional
        If True (default), use rigorous KKT formulation with Lagrange multipliers.
        If False, use heuristic RNEA correction.
        
    Returns
    -------
    dict
        't': time, 'q': positions, 'v': velocities, 'a': accelerations,
        'tau': torques, 'tau_friction': friction torques
        If constrained dynamics: also 'tau_rnea' for comparison
    """
    n = len(trajectory)
    nv = model.nv
    
    if friction_coeffs is None:
        friction_coeffs = [0.0] * nv
    elif len(friction_coeffs) < nv:
        friction_coeffs = list(friction_coeffs) + [0.0] * (nv - len(friction_coeffs))
    
    friction_coeffs = np.array(friction_coeffs[:nv])
    
    q_array = np.array([np.array(q) for q in trajectory])
    q_smooth = smooth_array(q_array, window=5)
    
    v_array = np.zeros((n, nv))
    for i in range(1, n - 1):
        v_array[i] = (q_smooth[i + 1] - q_smooth[i - 1]) / (2 * dt)
    if n > 1:
        v_array[0] = (q_smooth[1] - q_smooth[0]) / dt
        v_array[-1] = (q_smooth[-1] - q_smooth[-2]) / dt
    
    v_smooth = smooth_array(v_array, window=7)
    
    a_array = np.zeros((n, nv))
    for i in range(1, n - 1):
        a_array[i] = (v_smooth[i + 1] - v_smooth[i - 1]) / (2 * dt)
    if n > 1:
        a_array[0] = (v_smooth[1] - v_smooth[0]) / dt if n > 1 else np.zeros(nv)
        a_array[-1] = (v_smooth[-1] - v_smooth[-2]) / dt if n > 1 else np.zeros(nv)
    
    a_smooth = smooth_array(a_array, window=9)
    
    times = []
    positions = []
    velocities = []
    accelerations = []
    torques = []
    friction_torques = []
    torques_rnea = []  # For comparison when using constrained dynamics
    
    for i in range(n):
        t = i * dt
        q_curr = q_smooth[i]
        v_curr = v_smooth[i]
        a_curr = a_smooth[i]
        
        if use_constrained_dynamics and constraint_model is not None:
            # Use rigorous constrained inverse dynamics AND mapping to motors
            # This follows the strategy: KKT -> Virtual Work
            tau_dynamics = compute_motor_inverse_dynamics(
                model, data, constraint_model, q_curr, v_curr, a_curr
            )
            # Also compute RNEA for comparison
            tau_rnea_val = pin.rnea(model, data, q_curr, v_curr, a_curr)
            torques_rnea.append(tau_rnea_val.tolist())
        else:
            # Use standard RNEA with heuristic correction
            tau_dynamics = pin.rnea(model, data, q_curr, v_curr, a_curr)
            tau_dynamics = compute_motor_inverse_dynamics(
                model, data, constraint_model, q_curr, v_curr, a_curr
            )
        
        tau_friction = friction_coeffs * v_curr
        tau_total = tau_dynamics + tau_friction
        
        times.append(t)
        positions.append(q_curr.tolist())
        velocities.append(v_curr.tolist())
        accelerations.append(a_curr.tolist())
        torques.append(tau_total.tolist())
        friction_torques.append(tau_friction.tolist())
    
    tau_array = np.array(torques)
    tau_smooth = smooth_array(tau_array, window=7)
    torques = tau_smooth.tolist()
    
    result = {
        't': times,
        'q': positions,
        'v': velocities,
        'a': accelerations,
        'tau': torques,
        'tau_friction': friction_torques
    }
    
    # Add RNEA comparison if using constrained dynamics
    if use_constrained_dynamics and constraint_model is not None and torques_rnea:
        result['tau_rnea'] = torques_rnea
        result['method'] = 'constrained_kkt'
    else:
        result['method'] = 'rnea_heuristic'
    
    return result
