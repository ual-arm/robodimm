"""
robot_core/kinematics.py - Forward and Inverse Kinematics
========================================================

This module provides forward kinematics and inverse kinematics functions.
"""

import pinocchio as pin
import numpy as np
from typing import Tuple


def get_end_effector_position(model, data, q):
    """
    Compute end-effector position in world frame.
    
    Parameters
    ----------
    model : pin.Model
        The robot model
    data : pin.Data
        The data structure associated with model
    q : np.ndarray
        Joint configuration [nq]
        
    Returns
    -------
    position : np.ndarray
        End-effector position [x, y, z]
    """
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    fe = data.oMf[model.getFrameId('end_effector')]
    return np.array(fe.translation)


def run_ik_linear(configuration, ee_task, posture_task, target_pos, dt=0.005, max_iter=1000, tol=5e-4, solver='quadprog', damping=1e-8):
    """
    Solve IK for position-only target (preserves current orientation).
    
    Parameters
    ----------
    configuration : pink.Configuration
        Pink configuration object
    ee_task : pink.tasks.FrameTask
        End-effector frame task
    posture_task : pink.tasks.PostureTask
        Posture task for regularization
    target_pos : np.ndarray
        Target position [x, y, z]
    dt : float
        Time step for IK integration
    max_iter : int
        Maximum number of iterations
    tol : float
        Convergence tolerance
    solver : str
        IK solver type ('quadprog' or 'qpoases')
    damping : float
        Damping factor for IK
        
    Returns
    -------
    success : bool
        Whether IK converged
    iterations : int
        Number of iterations used
    error : float
        Final position error
    """
    import pink
    from pink.tasks import FrameTask, PostureTask

    # Preserve current EE orientation to avoid jumps switching from orientation jog
    pin.forwardKinematics(configuration.model, configuration.data, configuration.q)
    pin.updateFramePlacements(configuration.model, configuration.data)
    fe = configuration.data.oMf[configuration.model.getFrameId('end_effector')]
    target_se3 = pin.SE3(fe.rotation.copy(), target_pos)
    ee_task.set_target(target_se3)

    for i in range(max_iter):
        v = pink.solve_ik(configuration, [ee_task, posture_task], dt, solver=solver, damping=damping)
        configuration.integrate_inplace(v, dt)
        err = ee_task.compute_error(configuration)
        if np.linalg.norm(err[:3]) < tol:
            return True, i, np.linalg.norm(err[:3])
    # not converged
    err = ee_task.compute_error(configuration)
    return False, max_iter, np.linalg.norm(err[:3])


def run_ik_pose(configuration, ee_task, posture_task, target_se3, dt=0.005, max_iter=800, tol=5e-4, solver='quadprog', damping=1e-8):
    """
    Solve IK for full pose target (position + orientation).
    
    Parameters
    ----------
    configuration : pink.Configuration
        Pink configuration object
    ee_task : pink.tasks.FrameTask
        End-effector frame task
    posture_task : pink.tasks.PostureTask
        Posture task for regularization
    target_se3 : pin.SE3
        Target pose (position + orientation)
    dt : float
        Time step for IK integration
    max_iter : int
        Maximum number of iterations
    tol : float
        Convergence tolerance
    solver : str
        IK solver type ('quadprog' or 'qpoases')
    damping : float
        Damping factor for IK
        
    Returns
    -------
    success : bool
        Whether IK converged
    iterations : int
        Number of iterations used
    error : float
        Final pose error
    """
    import pink

    ee_task.set_target(target_se3)

    for i in range(max_iter):
        v = pink.solve_ik(configuration, [ee_task, posture_task], dt, solver=solver, damping=damping)
        configuration.integrate_inplace(v, dt)
        err = ee_task.compute_error(configuration)
        if np.linalg.norm(err) < tol:
            return True, i, float(np.linalg.norm(err))

    err = ee_task.compute_error(configuration)
    return False, max_iter, float(np.linalg.norm(err))
