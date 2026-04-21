"""
robot_core/constants.py - Robot Constants and Definitions
===================================================

This module contains constants, dimensions, masses, and joint mappings
for CR4 and CR6 robot models.
"""

import pinocchio as pin
import numpy as np

# =============================================================================
# SOLVER DE RESTRICCIONES
# =============================================================================


def solve_constraints(model, data, constraint_model, q0, max_iter=100, eps=1e-10):
    """
    Resuelve la geometría inversa para satisfacer las restricciones de cierre.
    """
    if constraint_model is None:
        return q0, True

    constraint_data = constraint_model.createData()
    q = q0.copy()
    y = np.ones(3)
    rho = 1e-10
    mu = 1e-4
    data.M = np.eye(model.nv) * rho
    try:
        kkt = pin.ContactCholeskyDecomposition(
            model, data, [constraint_model], [constraint_data]
        )
    except Exception:
        kkt = pin.ContactCholeskyDecomposition(model, [constraint_model])

    for k in range(max_iter):
        pin.computeJointJacobians(model, data, q)
        kkt.compute(model, data, [constraint_model], [constraint_data], mu)
        err = constraint_data.c1Mc2.translation
        if np.linalg.norm(err, np.inf) < eps:
            return q, True
        rhs = np.concatenate([-err - y * mu, np.zeros(model.nv)])
        dz = kkt.solve(rhs)
        dq = dz[3:]
        q = pin.integrate(model, q, -dq)
        y -= -dz[:3] + y
    return q, False


# =============================================================================
# MAPPING FROM INTERNAL MODEL VELOCITY INDICES TO USER-FACING JOINT NAMES
# =============================================================================

# For 6-DOF CR6: internal model has 9 DOF (J1, J2, J3real, J1p, J3, J2p, J4, J5, J6)
ACTIVE_JOINTS_CR6 = {
    0: "J1",
    1: "J2",
    2: "J3",  # J3 motor drives J3real + J3 constraint
    6: "J4",
    7: "J5",
    8: "J6",
}

PARALLELOGRAM_TORQUE_COMBINATION_CR6 = {
    2: 4,  # J3real (idx 2) + J3 constraint (idx 4)
}

# For 4-DOF CR4: internal model has 8 DOF
ACTIVE_JOINTS_CR4 = {
    0: "J1",
    1: "J2",
    2: "J3",
    7: "J4",
}

PARALLELOGRAM_TORQUE_COMBINATION_CR4 = {
    2: 4,
}
