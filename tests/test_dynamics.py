"""
test_dynamics.py
================
Unit tests for ``robot_core.dynamics``.

Tests:
- ``compute_constrained_inverse_dynamics`` — single time-step KKT solver
- ``correct_parallelogram_torques`` — virtual-work torque correction
- ``compute_constrained_inverse_dynamics_trajectory`` — full trajectory
- ``compare_dynamics_methods`` — KKT vs heuristic comparison

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
# Module-level skip guard
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def _require_pinocchio():
    _skip_if_missing()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cr4_dyn():
    from robot_core import build_robot
    return build_robot(robot_type="CR4", scale=1.0, payload_kg=0.0, visualize=False)


@pytest.fixture(scope="module")
def cr6_dyn():
    from robot_core import build_robot
    return build_robot(robot_type="CR6", scale=1.0, payload_kg=0.0, visualize=False)


# ---------------------------------------------------------------------------
# compute_constrained_inverse_dynamics  (single step)
# ---------------------------------------------------------------------------

class TestComputeConstrainedInverseDynamics:
    """Tests for the KKT single-step inverse dynamics solver."""

    def test_cr4_returns_nv_torques(self, cr4_dyn):
        from robot_core.dynamics import compute_constrained_inverse_dynamics
        model = cr4_dyn["model"]
        data  = cr4_dyn["data"]
        cm    = cr4_dyn["constraint_model"]
        q     = cr4_dyn["q"].copy()
        v     = np.zeros(model.nv)
        a     = np.zeros(model.nv)

        tau = compute_constrained_inverse_dynamics(model, data, cm, q, v, a)
        assert tau.shape == (model.nv,)

    def test_cr4_static_non_zero_gravity(self, cr4_dyn):
        """At rest (v=0, a=0), the KKT solver should report non-zero
        gravity-compensating torques for a non-vertical configuration."""
        from robot_core.dynamics import compute_constrained_inverse_dynamics
        from robot_core.conversions import q_pink_to_real

        model = cr4_dyn["model"]
        data  = cr4_dyn["data"]
        cm    = cr4_dyn["constraint_model"]

        # Use a non-trivial configuration
        q_pink = np.array([0.5, 0.4, 0.3, -(0.4 + 0.3), -0.2])
        q = q_pink_to_real("CR4", model, data, cm, q_pink)
        v = np.zeros(model.nv)
        a = np.zeros(model.nv)

        tau = compute_constrained_inverse_dynamics(model, data, cm, q, v, a)
        # At least one torque must be non-zero (gravity)
        assert not np.allclose(tau, 0.0, atol=1e-6)

    def test_cr4_returns_multipliers_on_request(self, cr4_dyn):
        """With return_multipliers=True, function returns a 3-tuple."""
        from robot_core.dynamics import compute_constrained_inverse_dynamics
        model = cr4_dyn["model"]
        data  = cr4_dyn["data"]
        cm    = cr4_dyn["constraint_model"]
        q     = cr4_dyn["q"].copy()
        v     = np.zeros(model.nv)
        a     = np.zeros(model.nv)

        result = compute_constrained_inverse_dynamics(
            model, data, cm, q, v, a, return_multipliers=True
        )
        assert len(result) == 3, "Expected (tau, lambda_, violation_norm)"
        tau, lambda_, viol = result
        assert tau.shape    == (model.nv,)
        assert lambda_ is not None
        assert viol    >= 0.0

    def test_cr6_returns_nv_torques(self, cr6_dyn):
        from robot_core.dynamics import compute_constrained_inverse_dynamics
        model = cr6_dyn["model"]
        data  = cr6_dyn["data"]
        cm    = cr6_dyn["constraint_model"]
        q     = cr6_dyn["q"].copy()
        v     = np.zeros(model.nv)
        a     = np.zeros(model.nv)

        tau = compute_constrained_inverse_dynamics(model, data, cm, q, v, a)
        assert tau.shape == (model.nv,)


# ---------------------------------------------------------------------------
# correct_parallelogram_torques
# ---------------------------------------------------------------------------

class TestCorrectParallelogramTorques:
    """Tests for the virtual-work parallelogram torque correction."""

    def test_output_shape_matches_input(self, cr4_dyn):
        from robot_core.dynamics import correct_parallelogram_torques
        model = cr4_dyn["model"]
        tau_in = np.ones(model.nv)
        tau_out = correct_parallelogram_torques(model, tau_in)
        assert tau_out.shape == (model.nv,)

    def test_zero_tau_gives_zero_output(self, cr4_dyn):
        """If all input torques are zero, output must also be zero."""
        from robot_core.dynamics import correct_parallelogram_torques
        model  = cr4_dyn["model"]
        tau_in = np.zeros(model.nv)
        tau_out = correct_parallelogram_torques(model, tau_in)
        np.testing.assert_allclose(tau_out, np.zeros(model.nv), atol=1e-14)

    def test_cr6_output_shape(self, cr6_dyn):
        from robot_core.dynamics import correct_parallelogram_torques
        model  = cr6_dyn["model"]
        tau_in = np.ones(model.nv)
        tau_out = correct_parallelogram_torques(model, tau_in)
        assert tau_out.shape == (model.nv,)


# ---------------------------------------------------------------------------
# compute_constrained_inverse_dynamics_trajectory
# ---------------------------------------------------------------------------

class TestComputeConstrainedIDTrajectory:
    """Tests for the full-trajectory inverse dynamics function."""

    def _build_short_trajectory(self, cr4_dyn, n_steps=20):
        """Generate a short joint-space trajectory for CR4."""
        from robot_core.interpolation import interpolate_joint
        q_start = cr4_dyn["q"].copy()
        q_end   = cr4_dyn["q"].copy()
        q_end[0] += 0.3  # J1 motion
        return interpolate_joint(q_start, q_end, num_steps=n_steps)

    def test_cr4_trajectory_returns_dict_with_required_keys(self, cr4_dyn):
        from robot_core.dynamics import compute_constrained_inverse_dynamics_trajectory
        traj = self._build_short_trajectory(cr4_dyn)
        dt   = 0.05
        result = compute_constrained_inverse_dynamics_trajectory(
            cr4_dyn["model"], cr4_dyn["data"],
            cr4_dyn["constraint_model"], traj, dt,
        )
        for key in ("t", "q", "v", "a", "tau"):
            assert key in result, f"Missing key '{key}' in dynamics result"

    def test_cr4_tau_shape(self, cr4_dyn):
        from robot_core.dynamics import compute_constrained_inverse_dynamics_trajectory
        n = 20
        traj = self._build_short_trajectory(cr4_dyn, n_steps=n)
        result = compute_constrained_inverse_dynamics_trajectory(
            cr4_dyn["model"], cr4_dyn["data"],
            cr4_dyn["constraint_model"], traj, dt=0.05,
        )
        tau_arr = np.array(result["tau"])
        # Number of timesteps close to n+1, nv=8
        assert tau_arr.ndim == 2
        assert tau_arr.shape[1] == cr4_dyn["model"].nv

    def test_cr4_time_array_monotone(self, cr4_dyn):
        from robot_core.dynamics import compute_constrained_inverse_dynamics_trajectory
        traj   = self._build_short_trajectory(cr4_dyn)
        result = compute_constrained_inverse_dynamics_trajectory(
            cr4_dyn["model"], cr4_dyn["data"],
            cr4_dyn["constraint_model"], traj, dt=0.05,
        )
        t_arr = np.array(result["t"])
        assert np.all(np.diff(t_arr) >= 0.0), "Time array is not monotonically non-decreasing"

    def test_cr4_with_friction(self, cr4_dyn):
        """Providing friction coefficients must not raise and returns 'tau_friction'."""
        from robot_core.dynamics import compute_constrained_inverse_dynamics_trajectory
        traj = self._build_short_trajectory(cr4_dyn)
        frict = [0.05] * cr4_dyn["model"].nv
        result = compute_constrained_inverse_dynamics_trajectory(
            cr4_dyn["model"], cr4_dyn["data"],
            cr4_dyn["constraint_model"], traj, dt=0.05,
            friction_coeffs=frict,
        )
        assert "tau_friction" in result


# ---------------------------------------------------------------------------
# compare_dynamics_methods
# ---------------------------------------------------------------------------

class TestCompareDynamicsMethods:
    """Tests for the DEMO vs PRO dynamics comparison."""

    def _short_traj(self, robot, n=15):
        from robot_core.interpolation import interpolate_joint
        q0 = robot["q"].copy()
        q1 = q0.copy()
        q1[0] += 0.2
        return interpolate_joint(q0, q1, num_steps=n)

    def test_cr4_returns_two_methods(self, cr4_dyn):
        from robot_core.dynamics import compare_dynamics_methods
        traj   = self._short_traj(cr4_dyn)
        result = compare_dynamics_methods(
            cr4_dyn["model"], cr4_dyn["data"],
            cr4_dyn["constraint_model"],
            traj, dt=0.05,
        )
        assert "constrained" in result
        assert "heuristic"   in result

    def test_cr4_summary_present(self, cr4_dyn):
        from robot_core.dynamics import compare_dynamics_methods
        traj   = self._short_traj(cr4_dyn)
        result = compare_dynamics_methods(
            cr4_dyn["model"], cr4_dyn["data"],
            cr4_dyn["constraint_model"],
            traj, dt=0.05,
        )
        assert "summary" in result

    def test_cr4_constrained_tau_shape(self, cr4_dyn):
        from robot_core.dynamics import compare_dynamics_methods
        traj   = self._short_traj(cr4_dyn)
        result = compare_dynamics_methods(
            cr4_dyn["model"], cr4_dyn["data"],
            cr4_dyn["constraint_model"],
            traj, dt=0.05,
        )
        tau_c = np.array(result["tau_constrained"])
        assert tau_c.ndim == 2
        assert tau_c.shape[1] == cr4_dyn["model"].nv

    def test_cr6_returns_results(self, cr6_dyn):
        from robot_core.dynamics import compare_dynamics_methods
        from robot_core.interpolation import interpolate_joint
        q0 = cr6_dyn["q"].copy()
        q1 = q0.copy()
        q1[0] += 0.2
        traj   = interpolate_joint(q0, q1, num_steps=10)
        result = compare_dynamics_methods(
            cr6_dyn["model"], cr6_dyn["data"],
            cr6_dyn["constraint_model"],
            traj, dt=0.05,
        )
        assert "constrained" in result
        assert "heuristic"   in result
