"""
test_interpolation.py
=====================
Unit tests for ``robot_core.interpolation``.

All tests in this module are pure-Python / NumPy and do **not** require
Pinocchio or Pink.
"""

import numpy as np
import pytest

# robot_core.__init__ imports pinocchio at module level; skip the whole file
# gracefully when the robodimm_env conda environment is not active.
pytest.importorskip("pinocchio", reason="Pinocchio not available — activate robodimm_env")

from robot_core.interpolation import (
    trapezoidal_profile,
    linear_profile,
    interpolate_joint,
)


# ---------------------------------------------------------------------------
# trapezoidal_profile
# ---------------------------------------------------------------------------

class TestTrapezoidalProfile:
    """Tests for the trapezoidal velocity profile function."""

    def test_returns_zero_at_start(self):
        assert trapezoidal_profile(0.0) == pytest.approx(0.0)

    def test_returns_one_at_end(self):
        assert trapezoidal_profile(1.0) == pytest.approx(1.0)

    def test_clamps_below_zero(self):
        assert trapezoidal_profile(-0.5) == pytest.approx(0.0)

    def test_clamps_above_one(self):
        assert trapezoidal_profile(1.5) == pytest.approx(1.0)

    def test_midpoint_is_half(self):
        """For a symmetric trapezoidal profile, s(0.5) must equal 0.5."""
        assert trapezoidal_profile(0.5) == pytest.approx(0.5, abs=1e-12)

    def test_monotonically_non_decreasing(self):
        """s must never decrease as t increases from 0 to 1."""
        times = np.linspace(0.0, 1.0, 200)
        values = [trapezoidal_profile(t) for t in times]
        for i in range(1, len(values)):
            assert values[i] >= values[i - 1] - 1e-14, (
                f"Profile not monotone at t={times[i]:.4f}: "
                f"s={values[i]:.6f} < s_prev={values[i-1]:.6f}"
            )

    def test_acceleration_phase_slower_than_linear(self):
        """
        During the acceleration phase the robot is still speeding up,
        so the normalised position must lag behind the linear case.
        """
        # Default accel_fraction=0.2 => acceleration phase is t in (0, 0.2)
        t_accel = 0.1
        assert trapezoidal_profile(t_accel) < linear_profile(t_accel)

    def test_deceleration_phase_ahead_of_linear(self):
        """
        During the deceleration phase the robot is slowing down after having
        travelled faster than average, so s > linear interpolation.
        """
        # Default accel_fraction=0.2 => deceleration starts at t=0.8
        t_decel = 0.9
        assert trapezoidal_profile(t_decel) > linear_profile(t_decel)

    def test_custom_accel_fraction_endpoints(self):
        """Endpoints must be 0 and 1 regardless of accel_fraction."""
        for af in [0.1, 0.2, 0.3, 0.4]:
            assert trapezoidal_profile(0.0, af) == pytest.approx(0.0)
            assert trapezoidal_profile(1.0, af) == pytest.approx(1.0)

    def test_continuity_at_phase_boundaries(self):
        """
        Profile must be continuous at the acceleration/deceleration boundaries.
        Evaluate just below and just above the phase transition t=ta and t=td.
        """
        af = 0.2
        eps = 1e-7
        for boundary in [af, 1.0 - af]:
            s_left  = trapezoidal_profile(boundary - eps, af)
            s_right = trapezoidal_profile(boundary + eps, af)
            assert abs(s_right - s_left) < 1e-5, (
                f"Discontinuity at t={boundary}: Δs={abs(s_right-s_left):.2e}"
            )


# ---------------------------------------------------------------------------
# linear_profile
# ---------------------------------------------------------------------------

class TestLinearProfile:
    """Tests for the linear interpolation profile."""

    def test_returns_zero_at_start(self):
        assert linear_profile(0.0) == pytest.approx(0.0)

    def test_returns_one_at_end(self):
        assert linear_profile(1.0) == pytest.approx(1.0)

    def test_midpoint_is_half(self):
        assert linear_profile(0.5) == pytest.approx(0.5)

    def test_clamps_below_zero(self):
        assert linear_profile(-1.0) == pytest.approx(0.0)

    def test_clamps_above_one(self):
        assert linear_profile(2.0) == pytest.approx(1.0)

    def test_identity_inside_domain(self):
        """For t in [0, 1], linear_profile(t) == t."""
        for t in [0.0, 0.1, 0.33, 0.5, 0.75, 1.0]:
            assert linear_profile(t) == pytest.approx(t)


# ---------------------------------------------------------------------------
# interpolate_joint
# ---------------------------------------------------------------------------

class TestInterpolateJoint:
    """Tests for joint-space interpolation (MoveJ)."""

    def test_returns_num_steps_plus_one_points(self):
        """interpolate_joint(n) returns n+1 configs (t=0 … t=1)."""
        q_s = np.zeros(4)
        q_e = np.ones(4)
        traj = interpolate_joint(q_s, q_e, num_steps=10)
        assert len(traj) == 11

    def test_first_point_equals_start(self):
        q_s = np.array([0.1, -0.2, 0.3, -0.4])
        q_e = np.array([1.0,  1.0, 1.0,  1.0])
        traj = interpolate_joint(q_s, q_e, num_steps=20)
        np.testing.assert_allclose(traj[0], q_s, atol=1e-14)

    def test_last_point_equals_end(self):
        q_s = np.array([0.1, -0.2, 0.3, -0.4])
        q_e = np.array([1.0,  1.0, 1.0,  1.0])
        traj = interpolate_joint(q_s, q_e, num_steps=20)
        np.testing.assert_allclose(traj[-1], q_e, atol=1e-14)

    def test_zero_motion_trajectory(self):
        """If start == end, every point must equal start."""
        q = np.array([0.5, -0.3, 1.2, 0.0])
        traj = interpolate_joint(q, q, num_steps=15)
        for pt in traj:
            np.testing.assert_allclose(pt, q, atol=1e-14)

    def test_midpoint_close_to_midpoint_for_linear(self):
        """
        With a linear profile and an even number of steps, the central
        waypoint should be the arithmetic mean of start and end.
        """
        q_s = np.array([0.0, 0.0])
        q_e = np.array([2.0, 4.0])
        traj = interpolate_joint(q_s, q_e, num_steps=10, profile="linear")
        mid = traj[5]  # t = 0.5 exactly
        np.testing.assert_allclose(mid, (q_s + q_e) / 2.0, atol=1e-14)

    def test_trapezoidal_slower_than_linear_at_start(self):
        """
        Near the beginning, the trapezoidal profile lags behind the linear one.
        """
        q_s = np.array([0.0])
        q_e = np.array([1.0])
        traj_trap = interpolate_joint(q_s, q_e, num_steps=20)
        traj_lin  = interpolate_joint(q_s, q_e, num_steps=20, profile="linear")
        # Second waypoint (i=1, t=0.05) should be smaller for trapezoidal
        assert traj_trap[1][0] < traj_lin[1][0]

    def test_all_points_within_bounds(self):
        """Every interpolated point must lie between start and end for each DOF."""
        q_s = np.array([-1.0, 0.0,  0.5])
        q_e = np.array([ 1.0, 2.0, -0.5])
        traj = interpolate_joint(q_s, q_e, num_steps=30)
        for pt in traj:
            for j in range(len(q_s)):
                lo, hi = min(q_s[j], q_e[j]), max(q_s[j], q_e[j])
                assert lo - 1e-12 <= pt[j] <= hi + 1e-12, (
                    f"DOF {j}: value {pt[j]:.4f} outside [{lo:.4f}, {hi:.4f}]"
                )

    def test_six_dof_trajectory(self):
        """Interpolation must work for 6-DOF arrays (CR6 use case)."""
        q_s = np.array([0.0, 0.5, -0.3, 0.2, -0.1, 1.0])
        q_e = np.array([1.0, -0.5, 0.7, -0.4, 0.6, -1.0])
        traj = interpolate_joint(q_s, q_e, num_steps=50)
        assert len(traj) == 51
        np.testing.assert_allclose(traj[0],  q_s, atol=1e-14)
        np.testing.assert_allclose(traj[-1], q_e, atol=1e-14)
