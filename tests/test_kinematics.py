"""
test_kinematics.py
==================
Unit tests for ``robot_core.kinematics``.

Tests forward kinematics (FK) and inverse kinematics (IK) for both
CR4 (4-DOF) and CR6 (6-DOF) robot models.

Requires Pinocchio and Pink (skipped otherwise).
"""

import numpy as np
import pytest

pytestmark = pytest.mark.pinocchio


def _skip_if_missing():
    try:
        import pinocchio  # noqa: F401
        import pink       # noqa: F401
    except ImportError:
        pytest.skip("Pinocchio/Pink not available")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cr4_robot_km():
    _skip_if_missing()
    import pink
    from pink.tasks import FrameTask, PostureTask
    from robot_core import build_robot, q_real_to_pink
    robot = build_robot(robot_type="CR4", scale=1.0, visualize=False)
    q_pink = q_real_to_pink("CR4", robot["model"], robot["q"])
    configuration = pink.Configuration(robot["model_pink"], robot["data_pink"], q_pink)
    ee_task = FrameTask("end_effector", position_cost=10.0, orientation_cost=0.0)
    post_task = PostureTask(cost=1e-3)
    post_task.set_target(q_pink)
    robot["q_pink"] = q_pink
    robot["configuration"] = configuration
    robot["ee_task"] = ee_task
    robot["post_task"] = post_task
    return robot


@pytest.fixture(scope="module")
def cr6_robot_km():
    _skip_if_missing()
    import pink
    from pink.tasks import FrameTask, PostureTask
    from robot_core import build_robot, q_real_to_pink
    robot = build_robot(robot_type="CR6", scale=1.0, visualize=False)
    q_pink = q_real_to_pink("CR6", robot["model"], robot["q"])
    configuration = pink.Configuration(robot["model_pink"], robot["data_pink"], q_pink)
    ee_task = FrameTask("end_effector", position_cost=10.0, orientation_cost=0.0)
    post_task = PostureTask(cost=1e-3)
    post_task.set_target(q_pink)
    robot["q_pink"] = q_pink
    robot["configuration"] = configuration
    robot["ee_task"] = ee_task
    robot["post_task"] = post_task
    return robot


# ---------------------------------------------------------------------------
# get_end_effector_position
# ---------------------------------------------------------------------------

class TestGetEndEffectorPosition:
    """Tests for the FK position extraction function."""

    def test_cr4_returns_3d_position(self, cr4_robot_km):
        from robot_core.kinematics import get_end_effector_position
        pos = get_end_effector_position(
            cr4_robot_km["model_pink"],
            cr4_robot_km["data_pink"],
            cr4_robot_km["q_pink"],
        )
        assert pos.shape == (3,)

    def test_cr4_neutral_position_above_ground(self, cr4_robot_km):
        """At neutral configuration the TCP must be above the ground (z > 0)."""
        from robot_core.kinematics import get_end_effector_position
        pos = get_end_effector_position(
            cr4_robot_km["model_pink"],
            cr4_robot_km["data_pink"],
            cr4_robot_km["q_pink"],
        )
        assert pos[2] > 0.0, f"TCP z={pos[2]:.4f} is not above ground"

    def test_cr4_neutral_within_workspace(self, cr4_robot_km):
        """At neutral, TCP should be within a plausible workspace radius."""
        from robot_core.kinematics import get_end_effector_position
        pos = get_end_effector_position(
            cr4_robot_km["model_pink"],
            cr4_robot_km["data_pink"],
            cr4_robot_km["q_pink"],
        )
        radius = np.linalg.norm(pos[:2])  # XY reach
        # Workspace radius ~ 945 mm at scale=1
        assert radius < 1.5, f"TCP XY radius {radius:.4f} m exceeds 1.5 m"

    def test_cr6_returns_3d_position(self, cr6_robot_km):
        from robot_core.kinematics import get_end_effector_position
        pos = get_end_effector_position(
            cr6_robot_km["model_pink"],
            cr6_robot_km["data_pink"],
            cr6_robot_km["q_pink"],
        )
        assert pos.shape == (3,)

    def test_position_changes_with_joint_angle(self, cr4_robot_km):
        """Moving J1 (base rotation) must change the TCP XY position."""
        from robot_core.kinematics import get_end_effector_position
        import pinocchio as pin

        model = cr4_robot_km["model_pink"]
        data  = cr4_robot_km["data_pink"]

        q0 = cr4_robot_km["q_pink"].copy()
        p0 = get_end_effector_position(model, data, q0)

        q1 = q0.copy()
        q1[0] += np.pi / 4  # rotate J1 by 45°
        p1 = get_end_effector_position(model, data, q1)

        assert not np.allclose(p0, p1, atol=1e-3), (
            "TCP did not move after changing J1 by 45°"
        )


# ---------------------------------------------------------------------------
# run_ik_linear (position-only IK)
# ---------------------------------------------------------------------------

class TestRunIKLinear:
    """Tests for the Cartesian position IK solver."""

    def test_ik_converges_to_current_position(self, cr4_robot_km):
        """
        Solving IK toward the current TCP position should converge
        with very small residual error.
        """
        from robot_core.kinematics import get_end_effector_position, run_ik_linear
        import pink

        robot = cr4_robot_km
        current_pos = get_end_effector_position(
            robot["model_pink"], robot["data_pink"], robot["q_pink"]
        )
        # Fresh configuration to avoid state contamination between tests
        config = pink.Configuration(robot["model_pink"], robot["data_pink"], robot["q_pink"].copy())
        success, iters, error = run_ik_linear(
            config,
            robot["ee_task"],
            robot["post_task"],
            target_pos=current_pos,
        )
        assert success, f"IK failed to converge on current position (iters={iters}, error={error:.2e})"
        assert error < 1e-3

    def test_ik_returns_three_values(self, cr4_robot_km):
        from robot_core.kinematics import run_ik_linear, get_end_effector_position
        import pink

        robot = cr4_robot_km
        current_pos = get_end_effector_position(
            robot["model_pink"], robot["data_pink"], robot["q_pink"]
        )
        config = pink.Configuration(robot["model_pink"], robot["data_pink"], robot["q_pink"].copy())
        result = run_ik_linear(config, robot["ee_task"], robot["post_task"], current_pos)
        assert len(result) == 3  # (success, iters, error)

    def test_ik_small_displacement_converges(self, cr4_robot_km):
        """A small cartesian displacement should converge."""
        from robot_core.kinematics import get_end_effector_position, run_ik_linear
        import pink

        robot = cr4_robot_km
        current_pos = get_end_effector_position(
            robot["model_pink"], robot["data_pink"], robot["q_pink"]
        )
        target_pos = current_pos + np.array([0.02, 0.0, 0.0])  # 2 cm in X
        config = pink.Configuration(robot["model_pink"], robot["data_pink"], robot["q_pink"].copy())
        success, iters, error = run_ik_linear(
            config, robot["ee_task"], robot["post_task"], target_pos
        )
        assert success, (
            f"IK failed for small 2 cm displacement (iters={iters}, error={error:.2e})"
        )

    def test_ik_cr6_small_displacement(self, cr6_robot_km):
        """Same convergence test for CR6."""
        from robot_core.kinematics import get_end_effector_position, run_ik_linear
        import pink

        robot = cr6_robot_km
        current_pos = get_end_effector_position(
            robot["model_pink"], robot["data_pink"], robot["q_pink"]
        )
        target_pos = current_pos + np.array([0.0, 0.02, 0.0])  # 2 cm in Y
        config = pink.Configuration(robot["model_pink"], robot["data_pink"], robot["q_pink"].copy())
        success, iters, error = run_ik_linear(
            config, robot["ee_task"], robot["post_task"], target_pos
        )
        assert success, (
            f"CR6 IK failed for small 2 cm displacement (iters={iters}, error={error:.2e})"
        )

    def test_ik_error_non_negative(self, cr4_robot_km):
        from robot_core.kinematics import get_end_effector_position, run_ik_linear
        import pink

        robot = cr4_robot_km
        current_pos = get_end_effector_position(
            robot["model_pink"], robot["data_pink"], robot["q_pink"]
        )
        config = pink.Configuration(robot["model_pink"], robot["data_pink"], robot["q_pink"].copy())
        _, _, error = run_ik_linear(
            config, robot["ee_task"], robot["post_task"], current_pos
        )
        assert error >= 0.0


# ---------------------------------------------------------------------------
# build_robot (integration smoke tests)
# ---------------------------------------------------------------------------

class TestBuildRobot:
    """Smoke tests for the robot builder function."""

    def test_cr4_build_returns_required_keys(self, cr4_robot_km):
        required = {"model", "data", "constraint_model",
                    "model_pink", "data_pink", "q",
                    "robot_type", "scale", "payload_kg"}
        assert required <= set(cr4_robot_km.keys())

    def test_cr6_build_returns_required_keys(self, cr6_robot_km):
        required = {"model", "data", "constraint_model",
                    "model_pink", "data_pink", "q",
                    "robot_type", "scale", "payload_kg"}
        assert required <= set(cr6_robot_km.keys())

    def test_cr4_model_nq(self, cr4_robot_km):
        """CR4 real model should have 8 configuration variables."""
        assert cr4_robot_km["model"].nq == 8

    def test_cr4_pink_model_nq(self, cr4_robot_km):
        """CR4 Pink (serial-5) model should have 5 configuration variables."""
        assert cr4_robot_km["model_pink"].nq == 5

    def test_cr6_model_nq(self, cr6_robot_km):
        """CR6 real model should have 9 configuration variables."""
        assert cr6_robot_km["model"].nq == 9

    def test_cr6_pink_model_nq(self, cr6_robot_km):
        """CR6 Pink model should have 6 configuration variables."""
        assert cr6_robot_km["model_pink"].nq == 6

    def test_cr4_robot_type_stored(self, cr4_robot_km):
        assert cr4_robot_km["robot_type"] == "CR4"

    def test_cr6_robot_type_stored(self, cr6_robot_km):
        assert cr6_robot_km["robot_type"] == "CR6"

    def test_cr4_scale_stored(self, cr4_robot_km):
        assert cr4_robot_km["scale"] == pytest.approx(1.0)

    def test_cr4_initial_q_correct_size(self, cr4_robot_km):
        assert len(cr4_robot_km["q"]) == cr4_robot_km["model"].nq

    def test_cr4_with_payload(self):
        """build_robot with non-zero payload should not raise."""
        _skip_if_missing()
        from robot_core import build_robot
        robot = build_robot(robot_type="CR4", scale=1.0,
                            payload_kg=5.0, visualize=False)
        assert robot["payload_kg"] == pytest.approx(5.0)

    def test_cr4_scaled_model(self):
        """build_robot with scale≠1 should produce a correctly labelled model."""
        _skip_if_missing()
        from robot_core import build_robot
        robot = build_robot(robot_type="CR4", scale=1.5, visualize=False)
        assert robot["scale"] == pytest.approx(1.5)

    def test_cr6_with_friction_coeffs(self):
        """friction_coeffs (6 values) should be accepted without error."""
        _skip_if_missing()
        from robot_core import build_robot
        robot = build_robot(
            robot_type="CR6", scale=1.0,
            friction_coeffs=[0.1, 0.1, 0.05, 0.05, 0.02, 0.02],
            visualize=False,
        )
        assert len(robot["friction_coeffs"]) == robot["model"].nv
