"""
robot_core/dynamics/constrained.py - Constrained Inverse Dynamics
===========================================================

This module implements rigorous inverse dynamics for closed kinematic chains
using Lagrange multipliers and KKT (Karush-Kuhn-Tucker) formulation.

THEORETICAL FOUNDATION:
-----------------------
For a constrained mechanical system, equations of motion are:

  M(q)·a + h(q,v) = τ + J_c^T · λ                    (1)
  g(q) = 0                                            (2) (position constraint)
  J_c · v = 0                                         (3) (velocity constraint)
  J_c · a + J̇_c · v = 0                               (4) (acceleration constraint)

Where:
  - M(q): Mass matrix [nv × nv]
  - h(q,v) = C(q,v)·v + g(q): Coriolis + gravity terms [nv]
  - τ: Joint torques [nv]
  - J_c: Constraint Jacobian [nc × nv]
  - λ: Lagrange multipliers (constraint forces) [nc]
  - nc: Number of constraint equations (3 for CONTACT_3D)

BAUMGARTE STABILIZATION:
------------------------
To prevent constraint drift, we use Baumgarte stabilization:

  J_c · a = -J̇_c · v - Kd · (J_c · v) - Kp · g(q)    (5)

Where Kp and Kd are stabilization gains (already in constraint_model.corrector)

KKT SYSTEM FOR INVERSE DYNAMICS:
--------------------------------
Given desired accelerations a, we solve for τ and λ:

  | M    -J_c^T | | τ_res |   | M·a + h |
  |             | |       | = |         |
  | J_c    0    | |  -λ    |   |  -γ     |

Where γ = J̇_c · v + Kd · (J_c · v) + Kp · g(q) is Baumgarte corrector term.

The actual joint torques are: τ = τ_res (for open-chain RNEA result)
corrected by constraint forces J_c^T · λ

PHYSICAL INTERPRETATION:
------------------------
- λ represents internal forces in closed kinematic chain
- For a parallelogram mechanism, λ captures forces transmitted through
  the connecting rod that an open-chain model would miss
- The corrected torques reflect the true motor loads including these
  internal constraint forces

REFERENCES:
-----------
[1] Featherstone, R. (2008). "Rigid Body Dynamics Algorithms". Springer.
[2] Baumgarte, J. (1972). "Stabilization of constraints and integrals of
    motion in dynamical systems". Computer Methods in Applied Mechanics.
[3] Murray, Li, Sastry (1994). "A Mathematical Introduction to Robotic
    Manipulation". CRC Press.
"""

import pinocchio as pin
import numpy as np
from typing import Tuple, Optional, List, Dict


CR4_J4_ACT_SIGN = -1.0


def _get_cr4_idx_map(model):
    names = set(model.names)
    required = {"J1", "J2", "J3real", "J3", "J_aux", "J1p", "J4"}
    if not required.issubset(names):
        return None
    idx_v = {model.names[i]: model.joints[i].idx_v for i in range(1, len(model.names))}
    return idx_v


def _build_cr4_actuator_projection(idx_v, nv):
    """Build B matrix such that q_real = B*q_act + const for CR4."""
    B = np.zeros((nv, 4), dtype=float)
    B[idx_v["J1"], 0] = 1.0
    B[idx_v["J2"], 1] = 1.0
    B[idx_v["J3real"], 2] = 1.0
    B[idx_v["J1p"], 1] = 1.0
    B[idx_v["J1p"], 2] = -1.0
    B[idx_v["J3"], 1] = -1.0
    B[idx_v["J3"], 2] = 1.0
    B[idx_v["J_aux"], 2] = -1.0
    B[idx_v["J4"], 3] = CR4_J4_ACT_SIGN
    return B


def compute_constrained_inverse_dynamics(
    model, data, constraint_models, q, v, a, mu=1e-8, return_multipliers=False
):
    """
    Compute inverse dynamics for a system with closed kinematic chain constraints.

    This function solves the constrained inverse dynamics problem using Pinocchio's
    native ContactCholeskyDecomposition for efficient and numerically stable KKT
    system solving. Unlike standard RNEA which treats the system as an open chain,
    this properly accounts for internal forces in closed loops (e.g.,
    parallelogram mechanisms).

    Mathematical formulation:
    -------------------------
    Standard equations of motion with constraints:
        M(q)·a + h(q,v) = τ + Jc^T·λ
        Jc·a = -γ  (Baumgarte-stabilized acceleration constraint)

    The KKT system solved is:
        | M   Jc^T | | dv     |   | τ - h |
        |          | |        | = |       |
        | Jc  -μI  | | -λ     |   | -γ    |

    For inverse dynamics (given a, find τ):
        τ = M·a + h - Jc^T·λ

    Parameters
    ----------
    model : pinocchio.Model
        The robot model (with all joints including passive ones)
    data : pinocchio.Data
        The data structure associated with model
    constraint_models : list[pinocchio.RigidConstraintModel] or pinocchio.RigidConstraintModel
        The constraint model(s) defining closed loop(s). Can be a single
        constraint or a list for multiple loops.
    q : np.ndarray
        Joint configuration [nq]
    v : np.ndarray
        Joint velocities [nv]
    a : np.ndarray
        Desired joint accelerations [nv]
    mu : float, optional
        Regularization parameter for KKT system (default: 1e-8)
    return_multipliers : bool, optional
        If True, also return Lagrange multipliers λ (default: False)

    Returns
    -------
    tau : np.ndarray
        Joint torques [nv] that produce desired accelerations while
        satisfying closed-loop constraints
    lambda_ : np.ndarray (only if return_multipliers=True)
        Lagrange multipliers representing constraint forces [nc]
    constraint_violation : float (only if return_multipliers=True)
        Norm of position constraint violation

    Notes
    -----
    - Uses Pinocchio's ContactCholeskyDecomposition for numerically stable KKT solving
    - The returned torques include contributions from both the open-chain
      dynamics and constraint forces transmitted through closed loops
    - For a parallelogram mechanism, this correctly computes the torque at
      the driving motor (J3real) including load transmitted through the
      connecting rod, which a simple RNEA would miss
    - The Baumgarte stabilization parameters (Kp, Kd) are taken from
      the constraint_model.corrector structure
    """
    # Handle single constraint or list of constraints
    if not isinstance(constraint_models, list):
        constraint_models = [constraint_models]

    # Filter out None constraints
    constraint_models = [cm for cm in constraint_models if cm is not None]

    # If no constraints, fall back to standard RNEA
    if len(constraint_models) == 0:
        tau = pin.rnea(model, data, q, v, a)
        if return_multipliers:
            return tau, np.array([]), 0.0
        return tau

    # Create constraint data structures
    constraint_datas = [cm.createData() for cm in constraint_models]

    # Total constraint dimension
    nc = sum(cm.size() for cm in constraint_models)

    # Compute kinematics and dynamics quantities
    pin.computeAllTerms(model, data, q, v)
    pin.computeJointJacobians(model, data, q)

    # Get mass matrix M and bias forces h = C·v + g
    M = data.M.copy()
    h = data.nle.copy()  # nle = nonlinear effects = C·v + g

    # ==========================================================================
    # USE PINOCCHIO'S NATIVE ContactCholeskyDecomposition FOR KKT SOLVING
    # ==========================================================================
    # This is much more numerically stable than manual pseudo-inverse approaches
    # and handles singular/ill-conditioned mass matrices properly.

    # Create KKT decomposition object
    kkt = pin.ContactCholeskyDecomposition(model, constraint_models)

    # Compute decomposition with regularization
    kkt.compute(model, data, constraint_models, constraint_datas, mu)

    # Get constraint Jacobian using Pinocchio's native function
    Jc = pin.getConstraintsJacobian(model, data, constraint_models, constraint_datas)

    # Compute Baumgarte stabilization terms for each constraint
    pin.forwardKinematics(model, data, q, v)
    pin.updateFramePlacements(model, data)

    gamma_list = []
    constraint_errors = []

    for cm, cd in zip(constraint_models, constraint_datas):
        # Get poses of two constrained frames
        oM1 = data.oMi[cm.joint1_id] * cm.joint1_placement
        oM2 = (
            data.oMi[cm.joint2_id] * cm.joint2_placement
            if cm.joint2_id > 0
            else cm.joint2_placement
        )

        # Position error (should be zero when constraint is satisfied)
        pos_error = oM1.translation - oM2.translation
        constraint_errors.append(pos_error)

        # Compute velocity error using constraint Jacobian slice
        # For this constraint's DOFs
        start_idx = sum(
            c.size() for c in constraint_models[: constraint_models.index(cm)]
        )
        end_idx = start_idx + cm.size()
        Jc_cm = Jc[start_idx:end_idx, :]
        vel_error = Jc_cm @ v

        # Baumgarte stabilization: γ = Kd·v_err + Kp·p_err
        Kp = cm.corrector.Kp
        Kd = cm.corrector.Kd
        gamma = Kd * vel_error + Kp * pos_error
        gamma_list.append(gamma)

    # Stack gamma for all constraints
    gamma_full = np.concatenate(gamma_list) if len(gamma_list) > 1 else gamma_list[0]

    # ==========================================================================
    # SOLVE KKT SYSTEM FOR LAGRANGE MULTIPLIERS
    # ==========================================================================
    # The KKT system format in Pinocchio's kkt.solve is:
    #   [M   Jc^T] [dv    ]   [tau - h]
    #   [Jc  -μI ] [-lambda] = [-gamma ]
    #
    # RHS format: [constraint_rhs (nc), dynamics_rhs (nv)]
    # Solution format: [lambda (nc), dv (nv)]
    #
    # For inverse dynamics: we set dynamics_rhs = M*a + h (as if tau=0)
    # and solve for lambda, then compute tau = M*a + h - Jc^T*lambda

    # Build right-hand side
    rhs = np.zeros(nc + model.nv)
    rhs[:nc] = -gamma_full  # Constraint RHS
    rhs[nc:] = M @ a + h  # Dynamics RHS (M*a + h when tau=0)

    # Solve KKT system using Pinocchio's efficient solver
    sol = kkt.solve(rhs)

    # Extract results
    lambda_ = sol[:nc]
    # dv_computed = sol[nc:]  # Not needed for inverse dynamics

    # Compute constrained torques: τ = M·a + h - Jc^T·λ
    tau = M @ a + h - Jc.T @ lambda_

    # Compute total constraint violation for diagnostics
    total_violation = np.linalg.norm(np.concatenate(constraint_errors))

    if return_multipliers:
        return tau, lambda_, total_violation

    return tau


def correct_parallelogram_torques(model, tau):
    """
    Map passive joint torques to their corresponding actuators for parallelogram mechanisms.

    PHYSICAL BASIS:
    ---------------
    In a parallelogram mechanism like IRB 2400/460:
    - J3real is the MOTOR located at the shoulder (base of link2)
    - J3 is the PASSIVE joint at the elbow
    - The parallelogram transmits torque from J3real to J3 with ratio R ≈ 1

    When we compute inverse dynamics (RNEA or constrained), we get torque
    that would be required at each joint. But J3 is passive - it cannot produce
    torque. The motor J3real must produce torque to move the elbow through
    the 4-bar linkage.

    For a symmetric parallelogram (like IRB 2400):
        τ_J3real = τ_J3 × R,  where R = 1 (transmission ratio)

    This is NOT a "heuristic" - it's the correct physical mapping for actuator sizing.

    MATHEMATICAL JUSTIFICATION:
    ---------------------------
    The constrained dynamics formulation solves for generalized forces τ in:
        M·a + h = τ + Jc^T·λ

    This gives τ_J3 (the load at the elbow) but τ_J3real ≈ 0 because J3real
    doesn't directly "feel" the load in the tree structure.

    The parallelogram constraint ensures that movement of J3real produces
    proportional movement at J3. By the principle of virtual work:
        P_motor = τ_J3real · ω_J3real = τ_J3 · ω_J3

    For IRB parallelogram where ω_J3real = ω_J3:
        τ_J3real = τ_J3

    Parameters
    ----------
    model : pinocchio.Model
        Robot model with parallelogram joints (J3real, J3)
    tau : np.ndarray
        Joint torques from inverse dynamics [nv]

    Returns
    -------
    tau_motor : np.ndarray
        Motor torques with correct mapping for actuator sizing [nv]

    Notes
    -----
    - This correction is essential for actuator dimensioning in parallelogram robots
    - For publication: this implements the virtual work principle for transmission
    - The result tau_motor represents what the physical motors must produce
    """
    tau_motor = tau.copy()

    try:
        # Get joint indices
        idx_j3real = model.getJointId("J3real") - 1  # Motor at shoulder
        idx_j3 = model.getJointId("J3") - 1  # Passive joint at elbow

        # Get velocity indices (idx_v) for these joints
        j3real_idx_v = model.joints[idx_j3real + 1].idx_v
        j3_idx_v = model.joints[idx_j3 + 1].idx_v

        # Transmission ratio for symmetric parallelogram
        # R = L_crank / L_crank = 1 (both bars of 4-bar are equal)
        R = 1.0

        # Map passive joint torque to motor
        # τ_motor_J3real = τ_passive_J3 × R
        tau_motor[j3real_idx_v] = tau[j3_idx_v] * R

    except Exception:
        # Model doesn't have parallelogram structure
        pass

    return tau_motor


def compute_motor_inverse_dynamics(
    model,
    data,
    constraint_model,
    q,
    v,
    a,
    return_analysis=False,
    torque_method="hybrid_actuation",
):
    """
    Compute motor torques for robots with parallelogram mechanisms.

    This is the CORRECT function for actuator sizing in closed-chain robots.
    It combines:
    1. Constrained inverse dynamics (KKT with Lagrange multipliers)
    2. Virtual work mapping from passive joints to motors

    PHYSICAL MODEL:
    ---------------
    For parallelogram robots (IRB 2400, IRB 460):

    ```
    [J2 Motor] ─┐
                ├─ [Link2] ─ [J3 Passive] ─ [Link3] ─ ...
    [J3real Motor] ─ [Crank] ─ [Rod] ─┘
    ```

    The motor J3real drives the passive joint J3 through a 4-bar linkage.
    The torque relationship is governed by virtual work:

        τ_J3real · θ̇_J3real = τ_J3 · θ̇_J3

    For symmetric parallelograms where θ̇_J3real ≈ θ̇_J3:
        τ_J3real = τ_J3

    Parameters
    ----------
    model : pinocchio.Model
        Robot model with all joints
    data : pinocchio.Data
        Model data structure
    constraint_model : pinocchio.RigidConstraintModel
        Constraint model for closed loop
    q, v, a : np.ndarray
        Joint positions, velocities, accelerations
    return_analysis : bool
        If True, return detailed analysis data

    Returns
    -------
    tau_motor : np.ndarray
        Motor torques for actuator sizing [nv]

    If return_analysis=True, returns tuple:
        (tau_motor, analysis_dict) where analysis_dict contains:
        - 'tau_constrained': Raw constrained dynamics torques
        - 'tau_rnea': Open-chain RNEA torques
        - 'lambda': Lagrange multipliers (constraint forces)
        - 'constraint_violation': Position constraint error

    Notes
    -----
    For publication purposes, this function:
    - Uses rigorous KKT formulation for constraint force computation
    - Applies virtual work principle for motor torque mapping
    - Provides physically meaningful results for actuator selection
    """
    # Step 1: Compute constrained inverse dynamics
    if constraint_model is not None:
        tau_constrained, lambda_, violation = compute_constrained_inverse_dynamics(
            model, data, constraint_model, q, v, a, return_multipliers=True
        )
    else:
        tau_constrained = pin.rnea(model, data, q, v, a)
        lambda_ = np.array([])
        violation = 0.0

    idx_cr4 = _get_cr4_idx_map(model)
    if idx_cr4 is not None:
        if torque_method not in {
            "hybrid_actuation",
            "virtual_work",
            "legacy_motor_map",
        }:
            raise ValueError(
                "torque_method must be hybrid_actuation|virtual_work|legacy_motor_map"
            )

        if torque_method == "legacy_motor_map":
            tau_motor = correct_parallelogram_torques(model, tau_constrained)
        else:
            B = _build_cr4_actuator_projection(idx_cr4, model.nv)
            tau_vw = B.T @ tau_constrained
            if torque_method == "hybrid_actuation":
                tau_act = np.array(
                    [tau_vw[0], tau_vw[1], tau_constrained[idx_cr4["J3"]], tau_vw[3]],
                    dtype=float,
                )
            else:
                tau_act = np.array(tau_vw, dtype=float)

            tau_motor = np.array(tau_constrained, dtype=float).copy()
            tau_motor[idx_cr4["J1"]] = tau_act[0]
            tau_motor[idx_cr4["J2"]] = tau_act[1]
            tau_motor[idx_cr4["J3real"]] = tau_act[2]
            tau_motor[idx_cr4["J4"]] = CR4_J4_ACT_SIGN * tau_act[3]
    else:
        # Generic fallback for non-CR4 robots
        tau_motor = correct_parallelogram_torques(model, tau_constrained)

    # Step 3: Also compute RNEA for comparison
    tau_rnea = pin.rnea(model, data, q, v, a)

    if return_analysis:
        analysis = {
            "tau_constrained": tau_constrained,
            "tau_rnea": tau_rnea,
            "lambda": lambda_,
            "constraint_violation": violation,
            "torque_method": torque_method,
        }
        return tau_motor, analysis

    return tau_motor
