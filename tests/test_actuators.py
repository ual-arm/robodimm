"""
test_actuators.py
=================
Unit tests for ``robot_core.actuators``.

Covers:
- ``analyze_trajectory_requirements`` — extracts per-joint torque/speed peaks
- ``select_actuators`` — selects optimal motor+gearbox combinations
- ``get_actuator_masses`` — retrieves mass of selected hardware

All tests use synthetic data and do **not** require Pinocchio.
"""

import numpy as np
import pytest

# robot_core.__init__ imports pinocchio at module level; skip gracefully.
pytest.importorskip("pinocchio", reason="Pinocchio not available — activate robodimm_env")

from robot_core.actuators import (
    analyze_trajectory_requirements,
    select_actuators,
    get_actuator_masses,
)
from robot_core.constants import ACTIVE_JOINTS_CR4, ACTIVE_JOINTS_CR6


# ---------------------------------------------------------------------------
# Helper: build deterministic synthetic dynamics dict
# ---------------------------------------------------------------------------

def _make_dynamics(n_steps: int, nv: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    tau = rng.uniform(-30.0, 30.0, size=(n_steps, nv))
    v   = rng.uniform(-2.0,   2.0, size=(n_steps, nv))
    q   = rng.uniform(-1.5,   1.5, size=(n_steps, nv))
    a   = rng.uniform(-5.0,   5.0, size=(n_steps, nv))
    t   = np.linspace(0.0, 1.0, n_steps)
    return {"t": t, "q": q, "v": v, "a": a, "tau": tau}


# ---------------------------------------------------------------------------
# analyze_trajectory_requirements
# ---------------------------------------------------------------------------

class TestAnalyzeTrajectoryRequirements:
    """Tests for requirement extraction from dynamics data."""

    # ---- CR4 ---------------------------------------------------------------

    def test_cr4_returns_four_active_joints(self, synthetic_cr4_dynamics):
        reqs = analyze_trajectory_requirements(synthetic_cr4_dynamics, "CR4")
        assert set(reqs.keys()) == set(ACTIVE_JOINTS_CR4.values())

    def test_cr4_required_fields_present(self, synthetic_cr4_dynamics):
        reqs = analyze_trajectory_requirements(synthetic_cr4_dynamics, "CR4")
        required_keys = {
            "peak_torque_Nm", "rms_torque_Nm",
            "peak_velocity_rad_s", "peak_velocity_rpm",
            "mean_velocity_rpm", "internal_idx_v", "includes_parallelogram",
        }
        for joint, data in reqs.items():
            assert required_keys <= set(data.keys()), (
                f"Joint {joint} missing keys: {required_keys - set(data.keys())}"
            )

    def test_cr4_peak_torque_non_negative(self, synthetic_cr4_dynamics):
        reqs = analyze_trajectory_requirements(synthetic_cr4_dynamics, "CR4")
        for joint, data in reqs.items():
            assert data["peak_torque_Nm"] >= 0.0, f"{joint}: negative peak torque"

    def test_cr4_rms_torque_leq_peak(self, synthetic_cr4_dynamics):
        reqs = analyze_trajectory_requirements(synthetic_cr4_dynamics, "CR4")
        for joint, data in reqs.items():
            assert data["rms_torque_Nm"] <= data["peak_torque_Nm"] + 1e-9, (
                f"{joint}: RMS torque exceeds peak torque"
            )

    def test_cr4_velocity_rpm_conversion(self, synthetic_cr4_dynamics):
        """peak_velocity_rpm == peak_velocity_rad_s * 60 / (2π)."""
        reqs = analyze_trajectory_requirements(synthetic_cr4_dynamics, "CR4")
        for joint, data in reqs.items():
            expected_rpm = data["peak_velocity_rad_s"] * 60.0 / (2.0 * np.pi)
            assert data["peak_velocity_rpm"] == pytest.approx(expected_rpm, rel=1e-3)

    # ---- CR6 ---------------------------------------------------------------

    def test_cr6_returns_six_active_joints(self, synthetic_cr6_dynamics):
        reqs = analyze_trajectory_requirements(synthetic_cr6_dynamics, "CR6")
        assert set(reqs.keys()) == set(ACTIVE_JOINTS_CR6.values())

    def test_cr6_peak_velocity_non_negative(self, synthetic_cr6_dynamics):
        reqs = analyze_trajectory_requirements(synthetic_cr6_dynamics, "CR6")
        for joint, data in reqs.items():
            assert data["peak_velocity_rad_s"] >= 0.0

    # ---- Edge cases --------------------------------------------------------

    def test_empty_dict_returns_empty(self):
        assert analyze_trajectory_requirements({}) == {}

    def test_missing_tau_returns_empty(self):
        dyn = {"t": np.array([0.0, 1.0]), "v": np.zeros((2, 8))}  # no 'tau'
        assert analyze_trajectory_requirements(dyn, "CR4") == {}

    def test_deterministic_peak_torque(self):
        """Seed-0 data for CR4: peak for J1 (col 0) must match manual calculation."""
        nv, n = 8, 50
        dyn = _make_dynamics(n_steps=n, nv=nv, seed=42)
        reqs = analyze_trajectory_requirements(dyn, "CR4")
        # J1 maps to col 0 — function applies round(..., 3) internally
        expected_peak = round(float(np.max(np.abs(dyn["tau"][:, 0]))), 3)
        assert reqs["J1"]["peak_torque_Nm"] == pytest.approx(expected_peak, rel=1e-6)

    def test_j3_flagged_as_parallelogram_cr4(self):
        """J3 in CR4 (vel index 2) is connected to the parallelogram."""
        dyn = _make_dynamics(50, 8)
        reqs = analyze_trajectory_requirements(dyn, "CR4")
        assert reqs["J3"]["includes_parallelogram"] is True

    def test_j1_not_flagged_as_parallelogram_cr4(self):
        dyn = _make_dynamics(50, 8)
        reqs = analyze_trajectory_requirements(dyn, "CR4")
        assert reqs["J1"]["includes_parallelogram"] is False


# ---------------------------------------------------------------------------
# select_actuators
# ---------------------------------------------------------------------------

class TestSelectActuators:
    """Tests for the motor+gearbox selection algorithm."""

    def test_sufficient_actuator_appears_in_candidates(self, simple_actuator_library):
        """An actuator that satisfies the load must appear in candidates."""
        lib = simple_actuator_library
        # Motor: 1Nm @ 3000rpm, Gearbox: ratio=100, eff=0.9
        # Output: 90 Nm, 30 rpm
        requirements = {
            "J1": {
                "peak_torque_Nm": 50.0, "rms_torque_Nm": 30.0,
                "peak_velocity_rpm": 20.0, "mean_velocity_rpm": 10.0,
                "peak_velocity_rad_s": 2.09, "includes_parallelogram": False,
            }
        }
        result = select_actuators(
            requirements,
            lib["motors"], lib["gearboxes"], lib["compatibility_matrix"],
            safety_factor_torque=1.0, safety_factor_speed=1.0,
        )
        assert "J1" in result
        assert len(result["J1"]["candidates"]) >= 1
        assert result["J1"]["recommended"] is not None

    def test_insufficient_actuator_yields_no_candidates(self, simple_actuator_library):
        """Requirements beyond actuator capability → empty candidates."""
        lib = simple_actuator_library
        requirements = {
            "J1": {
                "peak_torque_Nm": 5000.0,   # Way beyond 90 Nm
                "rms_torque_Nm": 3000.0,
                "peak_velocity_rpm": 5.0,
                "mean_velocity_rpm": 2.0,
                "peak_velocity_rad_s": 0.52,
                "includes_parallelogram": False,
            }
        }
        result = select_actuators(
            requirements,
            lib["motors"], lib["gearboxes"], lib["compatibility_matrix"],
        )
        assert result["J1"]["candidates"] == []
        assert result["J1"]["recommended"] is None

    def test_safety_factors_increase_required_torque(self, simple_actuator_library):
        """
        A combination that passes at sf=1.0 may fail at sf=2.0.
        Output torque is 90 Nm; require 50 Nm.
        - sf=1.0: need 50 Nm  → pass
        - sf=2.0: need 100 Nm → fail (90 < 100)
        """
        lib = simple_actuator_library
        requirements = {
            "J1": {
                "peak_torque_Nm": 50.0, "rms_torque_Nm": 25.0,
                "peak_velocity_rpm": 5.0, "mean_velocity_rpm": 2.0,
                "peak_velocity_rad_s": 0.52, "includes_parallelogram": False,
            }
        }
        result_ok   = select_actuators(requirements, lib["motors"], lib["gearboxes"],
                                       lib["compatibility_matrix"],
                                       safety_factor_torque=1.0, safety_factor_speed=1.0)
        result_fail = select_actuators(requirements, lib["motors"], lib["gearboxes"],
                                       lib["compatibility_matrix"],
                                       safety_factor_torque=2.0, safety_factor_speed=1.0)
        assert len(result_ok["J1"]["candidates"])   >= 1
        assert len(result_fail["J1"]["candidates"]) == 0

    def test_result_contains_required_fields(self, simple_actuator_library):
        lib = simple_actuator_library
        requirements = {
            "J1": {
                "peak_torque_Nm": 10.0, "rms_torque_Nm": 5.0,
                "peak_velocity_rpm": 10.0, "mean_velocity_rpm": 5.0,
                "peak_velocity_rad_s": 1.05, "includes_parallelogram": False,
            }
        }
        result = select_actuators(requirements, lib["motors"], lib["gearboxes"],
                                  lib["compatibility_matrix"])
        assert "candidates"      in result["J1"]
        assert "rejected_samples" in result["J1"]
        assert "recommended"     in result["J1"]
        assert "required"        in result["J1"]

    def test_candidate_output_torque_satisfies_requirement(self, simple_actuator_library):
        """Every candidate must have output_torque_Nm ≥ required torque."""
        lib = simple_actuator_library
        requirements = {
            "J1": {
                "peak_torque_Nm": 10.0, "rms_torque_Nm": 5.0,
                "peak_velocity_rpm": 5.0, "mean_velocity_rpm": 2.5,
                "peak_velocity_rad_s": 0.52, "includes_parallelogram": False,
            }
        }
        result = select_actuators(requirements, lib["motors"], lib["gearboxes"],
                                  lib["compatibility_matrix"],
                                  safety_factor_torque=1.0, safety_factor_speed=1.0)
        for cand in result["J1"]["candidates"]:
            assert cand["output_torque_Nm"] >= 10.0 - 1e-9

    def test_empty_requirements_returns_empty(self, simple_actuator_library):
        lib = simple_actuator_library
        result = select_actuators({}, lib["motors"], lib["gearboxes"],
                                  lib["compatibility_matrix"])
        assert result == {}

    def test_all_joints_analyzed(self, simple_actuator_library):
        """All joints passed in requirements must appear in the output."""
        lib = simple_actuator_library
        base_req = {
            "peak_torque_Nm": 10.0, "rms_torque_Nm": 5.0,
            "peak_velocity_rpm": 10.0, "mean_velocity_rpm": 5.0,
            "peak_velocity_rad_s": 1.05, "includes_parallelogram": False,
        }
        requirements = {f"J{i}": dict(base_req) for i in range(1, 5)}
        result = select_actuators(requirements, lib["motors"], lib["gearboxes"],
                                  lib["compatibility_matrix"])
        assert set(result.keys()) == {"J1", "J2", "J3", "J4"}


# ---------------------------------------------------------------------------
# get_actuator_masses
# ---------------------------------------------------------------------------

class TestGetActuatorMasses:
    """Tests for mass retrieval from selection results."""

    def _make_selection(self):
        """Build a mock selection result with a known recommendation."""
        return {
            "J1": {
                "recommended": {
                    "motor_id":   "M_1NM_3000",
                    "gearbox_id": "GB_100",
                    "ratio":      100,
                }
            }
        }

    def test_returns_mass_for_each_joint(self, simple_actuator_library):
        lib = simple_actuator_library
        selection = self._make_selection()
        masses = get_actuator_masses(selection, lib["motors"], lib["gearboxes"])
        assert "J1" in masses

    def test_motor_mass_matches_library(self, simple_actuator_library):
        lib = simple_actuator_library
        selection = self._make_selection()
        masses = get_actuator_masses(selection, lib["motors"], lib["gearboxes"])
        assert masses["J1"]["motor_mass_kg"] == pytest.approx(0.5, rel=1e-9)

    def test_gearbox_mass_matches_library(self, simple_actuator_library):
        lib = simple_actuator_library
        selection = self._make_selection()
        masses = get_actuator_masses(selection, lib["motors"], lib["gearboxes"])
        assert masses["J1"]["gearbox_mass_kg"] == pytest.approx(1.0, rel=1e-9)

    def test_no_recommendation_returns_none(self, simple_actuator_library):
        lib = simple_actuator_library
        selection = {"J1": {"recommended": None}}
        masses = get_actuator_masses(selection, lib["motors"], lib["gearboxes"])
        assert masses["J1"]["total_kg"] is None
