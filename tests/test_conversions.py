"""
test_conversions.py
===================
Unit tests for ``robot_core.conversions``.

Tests the coordinate-system conversions between the three joint
representations used in Robodimm:

  1. **Real** model (with parallelogram mechanism — full DOF vector)
  2. **Pink** model (simplified serial chain used for IK)
  3. **RobotStudio / Frontend** format (4 or 6 user-facing angles)

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
def cr4_models():
    _skip_if_missing()
    from robot_core import build_robot
    robot = build_robot(robot_type="CR4", scale=1.0, visualize=False)
    return robot


@pytest.fixture(scope="module")
def cr6_models():
    _skip_if_missing()
    from robot_core import build_robot
    robot = build_robot(robot_type="CR6", scale=1.0, visualize=False)
    return robot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _neutral_q_pink_cr4():
    return np.zeros(5)  # [J1, J2, J3_rel, J_aux, J4]


def _neutral_q_pink_cr6():
    return np.zeros(6)  # [J1..J6]


# ---------------------------------------------------------------------------
# CR4 conversions
# ---------------------------------------------------------------------------

class TestCR4Conversions:
    """Conversion tests for the CR4 (4-DOF parallelogram) robot."""

    def test_q_pink_to_real_neutral_no_exception(self, cr4_models):
        from robot_core.conversions import q_pink_to_real
        q_pink = _neutral_q_pink_cr4()
        q_real = q_pink_to_real(
            "CR4",
            cr4_models["model"],
            cr4_models["data"],
            cr4_models["constraint_model"],
            q_pink,
        )
        assert q_real is not None
        assert len(q_real) == cr4_models["model"].nq

    def test_q_real_to_pink_neutral(self, cr4_models):
        from robot_core.conversions import q_pink_to_real, q_real_to_pink
        q_pink0 = _neutral_q_pink_cr4()
        q_real  = q_pink_to_real(
            "CR4",
            cr4_models["model"], cr4_models["data"],
            cr4_models["constraint_model"], q_pink0,
        )
        q_pink_back = q_real_to_pink("CR4", cr4_models["model"], q_real)
        np.testing.assert_allclose(q_pink_back, q_pink0, atol=1e-6,
                                   err_msg="q_real→q_pink round-trip failed at neutral")

    @pytest.mark.parametrize("q_rs_input", [
        np.array([0.0,  0.0,  0.0,  0.0]),
        np.array([0.5,  0.3,  0.8, -1.571]),
        np.array([-0.9, 0.7, -0.2,  0.5]),
    ])
    def test_q_pink_to_real_then_robotstudio_roundtrip(self, cr4_models, q_rs_input):
        """
        RobotStudio→real→RobotStudio must recover the original 4-DOF vector.
        """
        from robot_core.conversions import q_pink_to_real, q_real_to_robotstudio
        # Convert from RobotStudio (4-DOF) to real via pink_to_real (accepts 4-elem input)
        q_real = q_pink_to_real(
            "CR4",
            cr4_models["model"], cr4_models["data"],
            cr4_models["constraint_model"], q_rs_input,
        )
        q_rs_back = q_real_to_robotstudio("CR4", cr4_models["model"], q_real)
        np.testing.assert_allclose(q_rs_back, q_rs_input, atol=1e-5,
                                   err_msg=f"RobotStudio round-trip failed for {q_rs_input}")

    def test_q_real_to_robotstudio_length(self, cr4_models):
        """q_real_to_robotstudio must return a 4-element array for CR4."""
        from robot_core.conversions import q_real_to_robotstudio
        q_real = cr4_models["q"]
        result = q_real_to_robotstudio("CR4", cr4_models["model"], q_real)
        assert len(result) == 4

    def test_q_real_to_pink_length(self, cr4_models):
        """q_real_to_pink must return a 5-element array for CR4."""
        from robot_core.conversions import q_real_to_pink
        q_real  = cr4_models["q"]
        q_pink  = q_real_to_pink("CR4", cr4_models["model"], q_real)
        assert len(q_pink) == 5

    def test_j_aux_horizontality_constraint(self, cr4_models):
        """
        After conversion pink→real, the J_aux angle must satisfy:
        J_aux = -(J2 + J3_rel).
        """
        from robot_core.conversions import q_pink_to_real, q_real_to_pink
        q_pink = np.array([0.3, 0.4, 0.2, -(0.4 + 0.2), -0.5])  # consistent aux
        q_real = q_pink_to_real(
            "CR4",
            cr4_models["model"], cr4_models["data"],
            cr4_models["constraint_model"], q_pink,
        )
        q_pink_back = q_real_to_pink("CR4", cr4_models["model"], q_real)
        j_aux_expected = -(q_pink_back[1] + q_pink_back[2])
        assert q_pink_back[3] == pytest.approx(j_aux_expected, abs=1e-5)


# ---------------------------------------------------------------------------
# CR6 conversions
# ---------------------------------------------------------------------------

class TestCR6Conversions:
    """Conversion tests for the CR6 (6-DOF spherical wrist) robot."""

    def test_q_pink_to_real_neutral_no_exception(self, cr6_models):
        from robot_core.conversions import q_pink_to_real
        q_pink = _neutral_q_pink_cr6()
        q_real = q_pink_to_real(
            "CR6",
            cr6_models["model"], cr6_models["data"],
            cr6_models["constraint_model"], q_pink,
        )
        assert q_real is not None
        assert len(q_real) == cr6_models["model"].nq

    def test_q_real_to_pink_length(self, cr6_models):
        from robot_core.conversions import q_real_to_pink
        q_real = cr6_models["q"]
        q_pink = q_real_to_pink("CR6", cr6_models["model"], q_real)
        assert len(q_pink) == 6

    def test_q_real_to_robotstudio_length(self, cr6_models):
        from robot_core.conversions import q_real_to_robotstudio
        q_real  = cr6_models["q"]
        q_rs    = q_real_to_robotstudio("CR6", cr6_models["model"], q_real)
        assert len(q_rs) == 6

    @pytest.mark.parametrize("q_pink_input", [
        np.zeros(6),
        np.array([0.5, -0.5,  0.3,  0.2, -0.1,  1.0]),
        np.array([-0.8, 0.6, -0.4, -0.3,  0.2, -0.5]),
    ])
    def test_pink_to_real_and_back_roundtrip(self, cr6_models, q_pink_input):
        """q_pink → q_real → q_pink must recover the original configuration."""
        from robot_core.conversions import q_pink_to_real, q_real_to_pink
        q_real = q_pink_to_real(
            "CR6",
            cr6_models["model"], cr6_models["data"],
            cr6_models["constraint_model"], q_pink_input,
        )
        q_pink_back = q_real_to_pink("CR6", cr6_models["model"], q_real)
        np.testing.assert_allclose(q_pink_back, q_pink_input, atol=1e-5,
                                   err_msg=f"CR6 round-trip failed for {q_pink_input}")

    def test_cr6_no_j4_inversion(self, cr6_models):
        """For CR6, q_real_to_robotstudio should NOT invert J4 (unlike CR4)."""
        import pinocchio as pin
        from robot_core.conversions import q_pink_to_real, q_real_to_robotstudio
        q_pink = np.array([0.0, 0.3, -0.2, 1.0, 0.5, -0.3])
        q_real = q_pink_to_real(
            "CR6",
            cr6_models["model"], cr6_models["data"],
            cr6_models["constraint_model"], q_pink,
        )
        q_rs = q_real_to_robotstudio("CR6", cr6_models["model"], q_real)
        # J4 index is 3; must not be inverted
        idx_j4_real = cr6_models["model"].getJointId("J4") - 1
        j4_real = q_real[idx_j4_real]
        assert q_rs[3] == pytest.approx(j4_real, abs=1e-6)
