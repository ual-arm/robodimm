"""Data-independent tests for the CR4 KKT diagnostics contract."""

from __future__ import annotations

import sys
import unittest

import numpy as np

sys.path.insert(0, ".")

from backend.dynamics.cr4_kkt import (  # noqa: E402
    ACCELERATION_RESIDUAL_TARGET_M_S2,
    PASSIVE_TORQUE_RESIDUAL_TARGET_NM,
    POSITION_RESIDUAL_TARGET_M,
    VELOCITY_RESIDUAL_TARGET_M_S,
    _constraint_kinematics,
    closed_full_configuration,
    compute_cr4_kkt_dynamics,
    get_or_build_model,
)
from backend.dynamics.schemas import CR4DiagnosticsModel  # noqa: E402


class TestCr4KktDiagnostics(unittest.TestCase):
    @staticmethod
    def robot() -> dict:
        return {
            "geometry": {
                "A": [0.0, 0.0, 0.0],
                "O": [0.260, 0.0, 0.7425],
                "B": [-0.140, 0.0, 0.7425],
                "C": [0.260, 0.0, 1.6875],
                "D": [-0.00488, 0.0, 0.88334],
                "E": [-0.00488, 0.0, 1.82834],
                "F": [0.48991, 0.0, 1.88034],
                "G": [1.51481, 0.0, 1.88034],
                "H": [1.285, 0.0, 1.6875],
                "P": [-0.140, 0.0, 1.6875],
                "J4": [1.505, 0.0, 1.476],
                "EE": [1.505, 0.0, 1.6875],
                "TCP": [1.505, 0.0, 1.436],
            },
            "inertials": {},
            "payload": {"massKg": 0.0},
            "limits": [],
        }

    def evaluate(self, robot=None, options=None):
        return compute_cr4_kkt_dynamics(
            robot or self.robot(),
            [0.1, -0.2, 0.3, 0.4],
            [0.2, 0.3, -0.4, 0.1],
            [0.1, -0.2, 0.3, -0.1],
            options,
        )

    def test_position_metrics_are_not_the_old_tautology(self):
        robot = self.robot()
        # Deliberately make the user hardpoint P inconsistent with the
        # parallelogram.  The actual contact placements then disagree even
        # though the KKT torque reconstruction identity would still be zero.
        robot["geometry"]["P"][0] += 0.01
        _, _, diagnostics, _ = self.evaluate(robot, {"fd_step": 1e-4})

        self.assertEqual(diagnostics["constraint_residual_norm"], 0.0)
        self.assertGreater(diagnostics["position_residual_stacked_norm"], 1e-5)
        self.assertGreater(diagnostics["position_residual_max_norm"], 1e-5)
        self.assertEqual(len(diagnostics["position_residual_vectors"]), 3)

    def test_contact_dimensions_rank_tolerance_and_solved_condition(self):
        robot = self.robot()
        _, _, diagnostics, _ = self.evaluate(robot, {"fd_step": 1e-4})
        built = get_or_build_model(robot)
        q_closed = closed_full_configuration(
            built.geom, np.array([0.1, -0.2, 0.3, 0.4], dtype=float)
        )
        jacobian_c, _ = _constraint_kinematics(built, q_closed)
        jacobian_p = jacobian_c[:, [3, 4, 5, 6, 7, 8]]
        singular_values = np.linalg.svd(jacobian_p, compute_uv=False)

        self.assertEqual(jacobian_c.shape, (9, 10))
        self.assertEqual(jacobian_p.shape, (9, 6))
        self.assertEqual(len(diagnostics["singular_values"]), 6)
        np.testing.assert_allclose(
            diagnostics["singular_values"], singular_values, rtol=1e-12, atol=1e-12
        )
        rank_tolerance = 9 * np.finfo(np.float64).eps * singular_values[0]
        self.assertAlmostEqual(diagnostics["rank_tolerance"], rank_tolerance)
        self.assertEqual(
            diagnostics["rank"],
            int(np.count_nonzero(singular_values > rank_tolerance)),
        )
        self.assertEqual(diagnostics["rank"], 6)
        self.assertAlmostEqual(
            diagnostics["condition_number"], singular_values[0] / singular_values[-1]
        )

    def test_fd_options_change_the_computed_state_mapping(self):
        tau_large, _, diag_large, _ = self.evaluate(
            options={"mapping_fd_step": 1e-3, "directional_fd_step": 1e-3}
        )
        tau_small, _, diag_small, _ = self.evaluate(
            options={"mapping_fd_step": 1e-7, "directional_fd_step": 1e-7}
        )

        self.assertGreater(np.max(np.abs(np.asarray(tau_large) - tau_small)), 1e-6)
        self.assertNotEqual(
            diag_large["acceleration_closure_residual_norm"],
            diag_small["acceleration_closure_residual_norm"],
        )

    @staticmethod
    def quintic_state(q0, q1, duration, u=0.25):
        q0 = np.asarray(q0, dtype=float)
        delta = np.asarray(q1, dtype=float) - q0
        scale = 10 * u**3 - 15 * u**4 + 6 * u**5
        scale_d = (30 * u**2 - 60 * u**3 + 30 * u**4) / duration
        scale_dd = (60 * u - 180 * u**2 + 120 * u**3) / duration**2
        return q0 + scale * delta, scale_d * delta, scale_dd * delta

    def test_frozen_nominal_and_near_singular_samples_meet_targets(self):
        cases = [
            self.quintic_state(
                [0.0, 0.0, 0.0, 0.0],
                [1.5415536976287865, 0.898877593289879, 0.9990302960023246, -9.042258053426622e-10],
                2.8904131830539748,
            ),
            self.quintic_state(
                [0.0, -0.680676707480057, -0.646206414793055, 0.0],
                [0.0, -0.680676707480057, -0.680676707480057, 0.0],
                0.16289638212819277,
            ),
        ]
        for q, qd, qdd in cases:
            _, _, diagnostics, warnings = compute_cr4_kkt_dynamics(
                self.robot(), q.tolist(), qd.tolist(), qdd.tolist(), {"fd_step": 1e-4}
            )
            for key in (
                "position_residual_vectors",
                "position_residual_norms",
                "velocity_closure_residual",
                "acceleration_closure_residual",
                "passive_torque_residual",
                "singular_values",
            ):
                self.assertTrue(np.all(np.isfinite(diagnostics[key])), key)
            self.assertEqual(diagnostics["rank"], 6)
            self.assertIsInstance(diagnostics["condition_number"], float)
            self.assertLessEqual(diagnostics["position_residual_max_norm"], POSITION_RESIDUAL_TARGET_M)
            self.assertLessEqual(diagnostics["velocity_closure_residual_norm"], VELOCITY_RESIDUAL_TARGET_M_S)
            self.assertLessEqual(diagnostics["acceleration_closure_residual_norm"], ACCELERATION_RESIDUAL_TARGET_M_S2)
            self.assertLessEqual(diagnostics["passive_torque_residual_norm"], PASSIVE_TORQUE_RESIDUAL_TARGET_NM)
            self.assertTrue(diagnostics["diagnostics_pass"])
            self.assertEqual(diagnostics["diagnostic_failures"], [])
            self.assertEqual(warnings, [])

    def test_failed_target_is_reported_as_a_warning(self):
        robot = self.robot()
        robot["geometry"]["P"][0] += 0.01
        _, _, diagnostics, warnings = self.evaluate(robot, {"fd_step": 1e-4})
        self.assertFalse(diagnostics["diagnostics_pass"])
        self.assertTrue(any("position residual" in failure for failure in diagnostics["diagnostic_failures"]))
        self.assertTrue(any("position residual" in warning for warning in warnings))

    def test_infinite_condition_is_json_safe(self):
        _, _, diagnostics, _ = self.evaluate(options={"fd_step": 1e-4})
        diagnostics["rank"] = 5
        diagnostics["condition_number"] = "infinity"
        encoded = CR4DiagnosticsModel(**diagnostics).model_dump_json()
        self.assertIn('\"condition_number\":\"infinity\"', encoded)

    def test_direct_mapping_step_validation(self):
        from backend.dynamics.cr4_kkt import mapped_jacobian

        built = get_or_build_model(self.robot())
        with self.assertRaisesRegex(ValueError, "finite positive"):
            mapped_jacobian(
                lambda value: closed_full_configuration(built.geom, value),
                np.zeros(4),
                0.0,
            )


if __name__ == "__main__":
    unittest.main()
