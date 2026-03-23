"""
robot_core/dynamics/comparison.py - Dynamics Methods Comparison
========================================================

This module provides functions to compare constrained vs heuristic inverse
dynamics methods for validation and paper figures.
"""

import numpy as np
from typing import Dict
from .trajectory import compute_constrained_inverse_dynamics_trajectory, compute_inverse_dynamics_trajectory


def compare_dynamics_methods(model, data, constraint_model, trajectory, dt, friction_coeffs=None):
    """
    Compare constrained vs heuristic inverse dynamics for validation and paper figures.
    
    This function is useful for:
    1. Validating constrained dynamics implementation
    2. Generating comparison figures for publications
    3. Quantifying error of heuristic approach
    
    Parameters
    ----------
    model, data, constraint_model, trajectory, dt, friction_coeffs
        Same as compute_inverse_dynamics_trajectory
        
    Returns
    -------
    dict
        Comparison data including:
        - 'constrained': Full results from KKT method
        - 'heuristic': Full results from RNEA + heuristic
        - 'rnea_raw': Raw RNEA without any correction
        - 'torque_diff_constrained_heuristic': Difference between methods
        - 'torque_diff_constrained_rnea': Difference from raw RNEA
        - 'max_diff_per_joint': Maximum difference per joint
        - 'rms_diff_per_joint': RMS difference per joint
    """
    # Compute with constrained dynamics (KKT)
    result_constrained = compute_constrained_inverse_dynamics_trajectory(
        model, data, constraint_model, trajectory, dt, friction_coeffs, return_analysis=True
    )
    
    # Compute with heuristic correction
    result_heuristic = compute_inverse_dynamics_trajectory(
        model, data, trajectory, dt, friction_coeffs, 
        constraint_model=None, use_constrained_dynamics=False
    )
    
    # Extract torque arrays
    tau_constrained = np.array(result_constrained['tau'])
    tau_heuristic = np.array(result_heuristic['tau'])
    tau_rnea = np.array(result_constrained['tau_rnea'])
    
    # Compute differences
    diff_ch = tau_constrained - tau_heuristic
    diff_cr = tau_constrained - tau_rnea
    
    return {
        'constrained': result_constrained,
        'heuristic': result_heuristic,
        't': result_constrained['t'],
        'tau_constrained': tau_constrained.tolist(),
        'tau_heuristic': tau_heuristic.tolist(),
        'tau_rnea_raw': tau_rnea.tolist(),
        'lambda': result_constrained.get('lambda', []),
        'constraint_violation': result_constrained.get('constraint_violation', []),
        'torque_diff_constrained_heuristic': diff_ch.tolist(),
        'torque_diff_constrained_rnea': diff_cr.tolist(),
        'max_diff_per_joint': np.max(np.abs(diff_cr), axis=0).tolist(),
        'rms_diff_per_joint': np.sqrt(np.mean(diff_cr**2, axis=0)).tolist(),
        'summary': {
            'max_torque_correction_Nm': float(np.max(np.abs(diff_cr))),
            'mean_constraint_violation_m': float(np.mean(result_constrained.get('constraint_violation', [0]))),
            'method_comparison': 'KKT with Lagrange multipliers vs RNEA open-chain'
        }
    }
