"""
Utility functions for robot coordinate transformations and demo data.
"""

import numpy as np
import pinocchio as pin


def get_frontend_nq(robot_type, model_pink):
    """Get number of joints visible to frontend."""
    if robot_type == "CR4":
        return 4
    return model_pink.nq


def q_pink_to_frontend(q_pink, robot_type):
    """
    Convert from Pink model format to frontend/RobotStudio format.

    For 6-DOF: Pink [J1, J2, J3, J4, J5, J6] -> Frontend [J1, J2, J3, J4, J5, J6]
    For 4-DOF: Pink [J1, J2, J3_rel, J_aux, J4] -> Frontend [J1, J2, J3real, J4]
    """
    if robot_type == "CR4":
        j3real = q_pink[1] + q_pink[2]  # J3real = J2 + J3_relative
        j4_rs = -q_pink[4]  # Invert J4 back to RobotStudio
        return np.array([q_pink[0], q_pink[1], j3real, j4_rs])
    return q_pink


def q_frontend_to_pink(q_frontend, current_q_pink, robot_type):
    """
    Convert from frontend/RobotStudio format to Pink model format.

    For 6-DOF: Frontend [J1, J2, J3, J4, J5, J6] -> Pink [J1, J2, J3, J4, J5, J6]
    For 4-DOF: Frontend [J1, J2, J3real, J4] -> Pink [J1, J2, J3_rel, J_aux, J4]
    """
    if robot_type == "CR4":
        q_p = current_q_pink.copy()
        if len(q_frontend) == 4:
            q_p[0] = q_frontend[0]  # J1
            q_p[1] = q_frontend[1]  # J2
            q_p[2] = q_frontend[2] - q_frontend[1]  # J3_relative = J3real - J2
            q_p[3] = -(q_p[1] + q_p[2])  # J_aux = -(J2 + J3_relative)
            q_p[4] = -q_frontend[3]  # J4 inverted for Z-down TCP
        elif len(q_frontend) == 5:
            return np.array(q_frontend, dtype=float)
        return q_p
    return np.array(q_frontend, dtype=float)


def map_joint_index_frontend_to_pink(idx, robot_type):
    """Map frontend joint index to Pink model index."""
    if robot_type == "CR4":
        if idx == 3:
            return 4  # J4 -> J4
        if idx > 3:
            return -1  # Invalid
    return idx


def _q_cr4_frontend_to_pink(q_frontend):
    """
    Convert CR4 frontend 4-DOF format to Pink 5-DOF format.
    Frontend [J1, J2, J3real, J4] -> Pink [J1, J2, J3_rel, J_aux, J4]
    """
    q = np.zeros(5)
    q[0] = q_frontend[0]  # J1
    q[1] = q_frontend[1]  # J2
    q[2] = q_frontend[2] - q_frontend[1]  # J3_relative = J3real - J2
    q[3] = -(q[1] + q[2])  # J_aux = -(J2 + J3_relative)
    q[4] = -q_frontend[3]  # J4 inverted for Z-down TCP
    return q


def create_demo_targets_and_program(robot_type="CR4"):
    """
    Create standardized benchmark targets + program for DEMO/PRO comparison.

    These targets are designed to exercise all joints with significant motion
    for comparing dynamics results between Newton-Euler (DEMO) and Pinocchio (PRO).

    The targets match frontend/js/demo_robot_data.js BENCHMARK_TARGETS_* constants.
    """
    demo_targets = []
    demo_program = []

    if robot_type == "CR4":
        # Default palletizing cycle targets in radians (matching DEMO mode)
        targets_data = [
            {"name": "PALLET_HOME", "q": [0.0, 0.0, 0.0, 0.0]},
            {"name": "PICK_APPROACH_LEFT", "q": [0.9, 1.1, 1.2, 0.0]},
            {"name": "PICK_LEFT_FLOOR", "q": [0.9, 1.15, 1.25, 0.0]},
            {"name": "PICK_LIFT_LEFT", "q": [0.9, 1.0, 1.1, 0.0]},
            {"name": "PLACE_APPROACH_FRONT", "q": [0.0, 1.1, 1.2, -1.571]},
            {"name": "PLACE_FRONT_BASE", "q": [0.0, 1.15, 1.25, -1.571]},
            {"name": "PLACE_LIFT_FRONT", "q": [0.0, 1.0, 1.1, -1.571]},
        ]

        # Compute FK to get positions/rotations
        from robot_core import build_robot

        robot_data = build_robot(robot_type="CR4", scale=1.0)
        model_pink = robot_data["model_pink"]
        data_pink = robot_data["data_pink"]

        for t in targets_data:
            # Convert frontend q to pink format
            q_pink = _q_cr4_frontend_to_pink(np.array(t["q"]))
            pin.forwardKinematics(model_pink, data_pink, q_pink)
            pin.updateFramePlacements(model_pink, data_pink)

            # Get TCP frame placement
            tcp_id = model_pink.getFrameId("tcp")
            if tcp_id >= len(model_pink.frames):
                tcp_id = model_pink.nframes - 1

            tcp_placement = data_pink.oMf[tcp_id]
            pos = tcp_placement.translation.tolist()
            rot = tcp_placement.rotation.flatten().tolist()

            demo_targets.append(
                {
                    "name": t["name"],
                    "position": pos,
                    "rotation": rot,
                    "q": t["q"],  # Store original frontend q
                }
            )

        # Default palletizing cycle program (matches DEMO mode)
        demo_program = [
            {"type": "MoveJ", "target_name": "PALLET_HOME", "speed": 45.0, "zone": 0.0},
            {
                "type": "MoveJ",
                "target_name": "PICK_APPROACH_LEFT",
                "speed": 45.0,
                "zone": 0.0,
            },
            {
                "type": "MoveJ",
                "target_name": "PICK_LEFT_FLOOR",
                "speed": 25.0,
                "zone": 0.0,
            },
            {
                "type": "MoveJ",
                "target_name": "PICK_LIFT_LEFT",
                "speed": 30.0,
                "zone": 0.0,
            },
            {
                "type": "MoveJ",
                "target_name": "PLACE_APPROACH_FRONT",
                "speed": 45.0,
                "zone": 0.0,
            },
            {
                "type": "MoveJ",
                "target_name": "PLACE_FRONT_BASE",
                "speed": 25.0,
                "zone": 0.0,
            },
            {
                "type": "MoveJ",
                "target_name": "PLACE_LIFT_FRONT",
                "speed": 30.0,
                "zone": 0.0,
            },
            {"type": "MoveJ", "target_name": "PALLET_HOME", "speed": 45.0, "zone": 0.0},
        ]

    else:  # CR6
        # Benchmark targets in radians (matching DEMO mode)
        targets_data = [
            {"name": "BENCH_HOME", "q": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
            {"name": "BENCH_EXTEND", "q": [-0.785, 0.785, -0.785, 0.785, 0.785, 0.0]},
            {"name": "BENCH_MID", "q": [0.524, -0.524, 0.524, 0.524, -0.524, 1.571]},
            {"name": "BENCH_BACK", "q": [0.262, -0.785, 0.262, -0.262, 0.785, -0.785]},
        ]

        # Compute FK to get positions/rotations
        from robot_core import build_robot

        robot_data = build_robot(robot_type="CR6", scale=1.0)
        model = robot_data["model_pink"]  # CR6 Pink model has same DOF
        data = robot_data["data_pink"]

        for t in targets_data:
            q = np.array(t["q"])
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)

            # Get TCP frame placement
            tcp_id = model.getFrameId("tcp")
            if tcp_id >= len(model.frames):
                tcp_id = model.nframes - 1

            tcp_placement = data.oMf[tcp_id]
            pos = tcp_placement.translation.tolist()
            rot = tcp_placement.rotation.flatten().tolist()

            demo_targets.append(
                {"name": t["name"], "position": pos, "rotation": rot, "q": q.tolist()}
            )

        # Standardized benchmark program (matches DEMO mode)
        demo_program = [
            {"type": "MoveJ", "target_name": "BENCH_HOME", "speed": 50.0, "zone": 0.0},
            {
                "type": "MoveJ",
                "target_name": "BENCH_EXTEND",
                "speed": 50.0,
                "zone": 0.0,
            },
            {"type": "MoveJ", "target_name": "BENCH_MID", "speed": 50.0, "zone": 0.0},
            {"type": "MoveJ", "target_name": "BENCH_BACK", "speed": 50.0, "zone": 0.0},
            {"type": "MoveJ", "target_name": "BENCH_HOME", "speed": 50.0, "zone": 0.0},
        ]

    return demo_targets, demo_program
