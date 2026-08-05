"""Tests for the legacy program endpoint's source-matched trajectory timing."""

from __future__ import annotations

import unittest

from backend.api.dynamics import build_program_trajectory_py


class TestLegacyTrajectorySemantics(unittest.TestCase):
    @staticmethod
    def cr4_robot() -> dict:
        return {
            "kind": "CR4",
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
            "limits": [
                {"lowerLimitRad": -2.87979, "upperLimitRad": 2.87979, "maxVelocityRadS": 2.53073, "maxAccelerationRadS2": 12.0},
                {"lowerLimitRad": -0.69813, "upperLimitRad": 1.48353, "maxVelocityRadS": 1.91986, "maxAccelerationRadS2": 10.0},
                {"lowerLimitRad": -0.69813, "upperLimitRad": 2.09440, "maxVelocityRadS": 2.09440, "maxAccelerationRadS2": 10.0},
                {"lowerLimitRad": -5.23599, "upperLimitRad": 5.23599, "maxVelocityRadS": 6.98132, "maxAccelerationRadS2": 35.0},
            ],
        }

    def test_cr4_movel_uses_tcp_distance_and_joint_limits(self):
        start = [
            1.541553735733068,
            0.9609910004498775,
            1.039232676381714,
            -1.1129717281121999e-7,
        ]
        target = [
            1.5474723577497647,
            1.2991925307033179,
            0.6203535464305907,
            0.005918496578899646,
        ]
        program = {
            "targets": [{"name": "target", "q": target}],
            "instructions": [
                {"type": "MoveL", "target_name": "target", "tcp_speed_m_s": 1.0}
            ],
        }

        times, qs, qds, qdds = build_program_trajectory_py(
            start, program, self.cr4_robot(), 0.005
        )
        self.assertAlmostEqual(times[-1], 0.4917721684655212, places=12)
        self.assertEqual(len(times), 100)
        self.assertEqual(qs[-1], target)
        self.assertTrue(all(value == 0.0 for value in qds[0] + qdds[0]))

    def test_movej_obeys_acceleration_limit_even_with_large_command_speed(self):
        robot = self.cr4_robot()
        robot["limits"][1]["maxVelocityRadS"] = 100.0
        target = [0.0, 1.0, 0.0, 0.0]
        program = {
            "targets": [{"name": "target", "q": target}],
            "instructions": [
                {"type": "MoveJ", "target_name": "target", "speed_rad_s": 1000.0}
            ],
        }
        times, qs, _qds, _qdds = build_program_trajectory_py(
            [0.0, 0.0, 0.0, 0.0], program, robot, 0.005
        )
        self.assertAlmostEqual(times[-1], (5.7735 / 10.0) ** 0.5, places=12)
        self.assertEqual(qs[-1], target)


if __name__ == "__main__":
    unittest.main()
