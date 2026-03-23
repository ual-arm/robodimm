"""
robot_core/interpolation.py - Trajectory Interpolation
================================================

This module provides trajectory interpolation functions for robot motion planning.
Includes trapezoidal velocity profiles for smooth motion matching industrial controllers.
"""

import pinocchio as pin
import numpy as np
from typing import List


def trapezoidal_profile(t: float, accel_fraction: float = 0.2) -> float:
    """
    Trapezoidal velocity profile for smooth motion.
    
    Acceleration phase → Constant velocity cruise → Deceleration phase
    
    Parameters
    ----------
    t : float
        Normalized time [0, 1]
    accel_fraction : float
        Fraction of time spent in acceleration/deceleration (default: 0.2)
        
    Returns
    -------
    s : float
        Normalized position [0, 1]
    """
    if t <= 0:
        return 0.0
    if t >= 1:
        return 1.0
    
    ta = accel_fraction
    td = 1.0 - accel_fraction
    
    # Max velocity during cruise (area under trapezoid must equal 1.0)
    # For symmetric trapezoid: area = 0.5*ta*vmax + (td-ta)*vmax + 0.5*(1-td)*vmax = 1.0
    # Simplifying: vmax * (1 - accel_fraction) = 1.0
    vmax = 1.0 / (1.0 - accel_fraction) if accel_fraction < 0.5 else 2.0
    
    if t < ta:
        # Acceleration phase: s = 0.5 * a * t², where a = vmax/ta
        return 0.5 * vmax * t * t / ta
    elif t < td:
        # Constant velocity phase
        return 0.5 * vmax * ta + vmax * (t - ta)
    else:
        # Deceleration phase
        dt = 1.0 - t
        return 1.0 - 0.5 * vmax * dt * dt / ta


def linear_profile(t: float) -> float:
    """
    Linear interpolation profile (for comparison/debugging).
    
    Parameters
    ----------
    t : float
        Normalized time [0, 1]
        
    Returns
    -------
    s : float
        Normalized position [0, 1]
    """
    return np.clip(t, 0.0, 1.0)


def interpolate_joint(q_start, q_end, num_steps, profile: str = 'trapezoidal', accel_fraction: float = 0.2):
    """
    Interpolation in joint space (for MoveJ instruction).
    
    Parameters
    ----------
    q_start : np.ndarray
        Starting joint configuration
    q_end : np.ndarray
        Ending joint configuration
    num_steps : int
        Number of interpolation steps
    profile : str
        Interpolation profile: 'trapezoidal' or 'linear' (default: 'trapezoidal')
    accel_fraction : float
        Fraction of time for accel/decel when using trapezoidal profile (default: 0.2)
        
    Returns
    -------
    trajectory : list of np.ndarray
        List of joint configurations from start to end
    """
    trajectory = []
    for i in range(num_steps + 1):
        t = i / num_steps if num_steps > 0 else 1.0
        
        if profile == 'trapezoidal':
            s = trapezoidal_profile(t, accel_fraction)
        else:
            s = linear_profile(t)
            
        q_interp = (1 - s) * q_start + s * q_end
        trajectory.append(q_interp.copy())
    return trajectory


def interpolate_circular(model, data, configuration, ee_task, post_task, start_pose, via_pose, end_pose, num_steps, ik_dt=0.005, ik_iters=200):
    """
    Circular interpolation in Cartesian space (for MoveC instruction).
    
    Parameters
    ----------
    model : pin.Model
        The robot model
    data : pin.Data
        The data structure associated with model
    configuration : pink.Configuration
        Pink configuration object
    ee_task : pink.tasks.FrameTask
        End-effector frame task
    post_task : pink.tasks.PostureTask
        Posture task for regularization
    start_pose : dict
        Starting pose with 'position' and 'rotation' keys
    via_pose : dict
        Via point pose with 'position' and 'rotation' keys
    end_pose : dict
        Ending pose with 'position' and 'rotation' keys
    num_steps : int
        Number of interpolation steps
    ik_dt : float
        Time step for IK integration
    ik_iters : int
        Maximum IK iterations per step
        
    Returns
    -------
    trajectory : list of np.ndarray
        List of joint configurations along circular path
    """
    from pink.tasks import FrameTask
    
    trajectory = []
    
    # Extract positions
    p_start = np.array(start_pose['position'])
    p_via = np.array(via_pose['position'])
    p_end = np.array(end_pose['position'])
    R_start = np.array(start_pose['rotation']).reshape(3, 3)
    R_end = np.array(end_pose['rotation']).reshape(3, 3)
    
    # Fit circle through 3 points in 3D
    v1 = p_via - p_start
    v2 = p_end - p_start
    
    normal = np.cross(v1, v2)
    normal_norm = np.linalg.norm(normal)
    
    if normal_norm < 1e-9:
        return interpolate_cartesian(model, data, configuration, ee_task, post_task, start_pose, end_pose, num_steps, ik_dt, ik_iters, use_orientation=True)
    
    normal = normal / normal_norm
    
    d1 = np.linalg.norm(v1)**2
    d2 = np.linalg.norm(v2)**2
    cross_norm_sq = normal_norm**2
    
    t = (d1 * np.cross(normal, v2) + d2 * np.cross(v1, normal)) / (2 * cross_norm_sq)
    center = p_start + t
    radius = np.linalg.norm(p_start - center)
    
    u_start = (p_start - center) / radius
    u_end = (p_end - center) / radius
    
    e1 = u_start
    e2 = np.cross(normal, e1)
    
    cos_theta = np.clip(np.dot(u_end, e1), -1, 1)
    sin_theta = np.dot(u_end, e2)
    theta_end = np.arctan2(sin_theta, np.dot(u_end, e1))
    
    u_via = (p_via - center) / radius
    theta_via = np.arctan2(np.dot(u_via, e2), np.dot(u_via, e1))
    
    if theta_via < 0 and theta_end > 0:
        theta_end = theta_end - 2 * np.pi
    elif theta_via > 0 and theta_end < 0:
        theta_end = theta_end + 2 * np.pi
    
    move_task = FrameTask('end_effector', position_cost=15.0, orientation_cost=10.0)
    
    for i in range(num_steps + 1):
        alpha = i / num_steps
        theta = alpha * theta_end
        
        p_interp = center + radius * (np.cos(theta) * e1 + np.sin(theta) * e2)
        
        R_rel = R_start.T.dot(R_end)
        aa = pin.AngleAxis(R_rel)
        if aa.angle < 1e-6:
            R_interp = R_start.copy()
        else:
            aa_interp = pin.AngleAxis(alpha * aa.angle, aa.axis)
            R_interp = R_start.dot(aa_interp.toRotationMatrix())
        
        target_se3 = pin.SE3(R_interp, p_interp)
        move_task.set_target(target_se3)
        
        import pink
        for _ in range(ik_iters):
            v = pink.solve_ik(configuration, [move_task, post_task], ik_dt, solver='quadprog', damping=1e-8)
            configuration.integrate_inplace(v, ik_dt)
            err = move_task.compute_error(configuration)
            pos_err = np.linalg.norm(err[:3])
            ori_err = np.linalg.norm(err[3:6]) if len(err) > 3 else 0
            if pos_err < 1e-4 and ori_err < 1e-3:
                break
        
        trajectory.append(configuration.q.copy())
    
    return trajectory


def interpolate_cartesian(model, data, configuration, ee_task, post_task, start_pose, end_pose, num_steps, ik_dt=0.005, ik_iters=200, use_orientation=False):
    """
    Linear interpolation in Cartesian space (for MoveL instruction).
    
    Parameters
    ----------
    model : pin.Model
        The robot model
    data : pin.Data
        The data structure associated with model
    configuration : pink.Configuration
        Pink configuration object
    ee_task : pink.tasks.FrameTask
        End-effector frame task
    post_task : pink.tasks.PostureTask
        Posture task for regularization
    start_pose : dict
        Starting pose with 'position' and 'rotation' keys
    end_pose : dict
        Ending pose with 'position' and 'rotation' keys
    num_steps : int
        Number of interpolation steps
    ik_dt : float
        Time step for IK integration
    ik_iters : int
        Maximum IK iterations per step
    use_orientation : bool
        Whether to interpolate orientation
        
    Returns
    -------
    trajectory : list of np.ndarray
        List of joint configurations along linear path
    """
    from pink.tasks import FrameTask
    
    trajectory = []
    p_start = np.array(start_pose['position'])
    p_end = np.array(end_pose['position'])
    R_start = np.array(start_pose['rotation']).reshape(3, 3)
    R_end = np.array(end_pose['rotation']).reshape(3, 3)
    
    if use_orientation:
        move_task = FrameTask('end_effector', position_cost=15.0, orientation_cost=10.0)
    else:
        move_task = ee_task

    for i in range(num_steps + 1):
        alpha = i / num_steps
        p_interp = (1 - alpha) * p_start + alpha * p_end
        R_rel = R_start.T.dot(R_end)
        aa = pin.AngleAxis(R_rel)
        if aa.angle < 1e-6:
            R_interp = R_start.copy()
        else:
            aa_interp = pin.AngleAxis(alpha * aa.angle, aa.axis)
            R_interp = R_start.dot(aa_interp.toRotationMatrix())

        target_se3 = pin.SE3(R_interp, p_interp)
        move_task.set_target(target_se3)

        import pink
        for _ in range(ik_iters):
            v = pink.solve_ik(configuration, [move_task, post_task], ik_dt, solver='quadprog', damping=1e-8)
            configuration.integrate_inplace(v, ik_dt)
            err = move_task.compute_error(configuration)
            pos_err = np.linalg.norm(err[:3])
            if use_orientation:
                ori_err = np.linalg.norm(err[3:6]) if len(err) > 3 else 0
                if pos_err < 1e-4 and ori_err < 1e-3:
                    break
            else:
                if pos_err < 1e-4:
                    break

        trajectory.append(configuration.q.copy())

    return trajectory
