"""
conftest.py - Shared fixtures for the Robodimm test suite.

Fixtures that require Pinocchio/Pink are marked with the ``pinocchio`` mark
and skipped automatically when those libraries are not available.
"""

import pytest
import numpy as np


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "pinocchio: mark test as requiring Pinocchio and Pink",
    )


# ---------------------------------------------------------------------------
# Helper: skip module-level if pinocchio is missing
# ---------------------------------------------------------------------------

def _pinocchio_available():
    try:
        import pinocchio  # noqa: F401
        import pink       # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Session-scoped robot fixtures (built once, reused across all tests)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def cr4_robot():
    """Build a scale-1 CR4 robot (no visualisation) once per session."""
    if not _pinocchio_available():
        pytest.skip("Pinocchio/Pink not available")
    from robot_core import build_robot
    return build_robot(robot_type="CR4", scale=1.0, payload_kg=0.0, visualize=False)


@pytest.fixture(scope="session")
def cr6_robot():
    """Build a scale-1 CR6 robot (no visualisation) once per session."""
    if not _pinocchio_available():
        pytest.skip("Pinocchio/Pink not available")
    from robot_core import build_robot
    return build_robot(robot_type="CR6", scale=1.0, payload_kg=0.0, visualize=False)


@pytest.fixture(scope="session")
def cr4_robot_scaled():
    """Build a scale-1.5 CR4 robot for scaling tests."""
    if not _pinocchio_available():
        pytest.skip("Pinocchio/Pink not available")
    from robot_core import build_robot
    return build_robot(robot_type="CR4", scale=1.5, payload_kg=5.0, visualize=False)


@pytest.fixture(scope="session")
def cr6_robot_payload():
    """Build a CR6 robot with non-zero payload for dynamics tests."""
    if not _pinocchio_available():
        pytest.skip("Pinocchio/Pink not available")
    from robot_core import build_robot
    return build_robot(robot_type="CR6", scale=1.0, payload_kg=3.0, visualize=False)


# ---------------------------------------------------------------------------
# Synthetic dynamics data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_cr4_dynamics():
    """
    Minimal synthetic dynamics dict matching the shape expected by
    analyze_trajectory_requirements for a CR4 robot.

    CR4 real model has nv=8; ACTIVE_JOINTS_CR4 uses indices {0,1,2,7}.
    """
    rng = np.random.default_rng(42)
    n_steps = 50
    nv = 8
    tau = rng.uniform(-50.0, 50.0, size=(n_steps, nv))
    v   = rng.uniform(-2.0,   2.0, size=(n_steps, nv))
    q   = rng.uniform(-1.5,   1.5, size=(n_steps, nv))
    a   = rng.uniform(-5.0,   5.0, size=(n_steps, nv))
    t   = np.linspace(0.0, 1.0, n_steps)
    return {"t": t, "q": q, "v": v, "a": a, "tau": tau}


@pytest.fixture
def synthetic_cr6_dynamics():
    """
    Minimal synthetic dynamics dict for a CR6 robot.
    CR6 real model has nv=9; ACTIVE_JOINTS_CR6 uses indices {0,1,2,6,7,8}.
    """
    rng = np.random.default_rng(7)
    n_steps = 50
    nv = 9
    tau = rng.uniform(-80.0, 80.0, size=(n_steps, nv))
    v   = rng.uniform(-3.0,   3.0, size=(n_steps, nv))
    q   = rng.uniform(-1.5,   1.5, size=(n_steps, nv))
    a   = rng.uniform(-5.0,   5.0, size=(n_steps, nv))
    t   = np.linspace(0.0, 1.0, n_steps)
    return {"t": t, "q": q, "v": v, "a": a, "tau": tau}


# ---------------------------------------------------------------------------
# Minimal actuator library fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_actuator_library():
    """
    A minimal motor + gearbox library for testing select_actuators.
    Motor:   rated_torque=1 Nm, rated_speed=3000 rpm
    Gearbox: ratio=100, efficiency=0.90
    Combined output: 90 Nm, 30 rpm
    """
    motors = [
        {
            "id": "M_1NM_3000",
            "name": "TestMotor 1Nm/3000rpm",
            "rated_torque_Nm": 1.0,
            "rated_speed_rpm": 3000,
            "mass_kg": 0.5,
            "compatible_gearboxes": ["GB_100"],
        }
    ]
    gearboxes = [
        {
            "id": "GB_100",
            "name": "TestGearbox ratio-100",
            "ratios": [100],
            "efficiency": 0.90,
            "mass_kg": 1.0,
        }
    ]
    compatibility_matrix = {
        "M_1NM_3000": {"GB_100": {"natural_match": True}}
    }
    return {"motors": motors, "gearboxes": gearboxes, "compatibility_matrix": compatibility_matrix}
