"""Source-only CR4 finite-difference sensitivity verification.

This is deliberately independent of the regression fixtures and generated
benchmark artifacts.  It rebuilds the canonical IRB460 CR4 specification,
recreates the five frozen one-instruction trajectories, and evaluates the
current source KKT implementation at the protocol's common finite-difference
step ladder.  The output is source verification, not Simscape benchmark
evidence.
"""

from __future__ import annotations

import copy
import json
import math
import sys
import unittest
from pathlib import Path
from typing import Any

import numpy as np

# Make direct invocation work from the repository root without relying on the
# import path chosen by the caller.  Keep this script self-contained: in
# particular, do not import fixtures from another test module.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True

from backend.dynamics.cr4_kkt import (  # noqa: E402
    Cr4GeometryContext,
    closed_chain_points,
    compute_cr4_kkt_batch,
)


DT_S = 0.005
FD_STEPS = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7)

CANONICAL_GEOMETRY: dict[str, list[float]] = {
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
}

# These are the original IRB460 preset limits.  Tier scales are applied only
# to copies for duration calculation; the dynamics robot receives these
# unscaled limits, as required by protocol section 3.2.
ORIGINAL_LIMITS: tuple[dict[str, Any], ...] = (
    {
        "name": "J1",
        "lowerLimitRad": -2.87979,
        "upperLimitRad": 2.87979,
        "maxVelocityRadS": 2.53073,
        "maxAccelerationRadS2": 12.0,
        "frictionCoeffNmSPerRad": 0.5,
    },
    {
        "name": "J2",
        "lowerLimitRad": -0.69813,
        "upperLimitRad": 1.48353,
        "maxVelocityRadS": 1.91986,
        "maxAccelerationRadS2": 10.0,
        "frictionCoeffNmSPerRad": 0.5,
    },
    {
        "name": "J3",
        "lowerLimitRad": -0.69813,
        "upperLimitRad": 2.09440,
        "maxVelocityRadS": 2.09440,
        "maxAccelerationRadS2": 10.0,
        "frictionCoeffNmSPerRad": 0.5,
    },
    {
        "name": "J4",
        "lowerLimitRad": -5.23599,
        "upperLimitRad": 5.23599,
        "maxVelocityRadS": 6.98132,
        "maxAccelerationRadS2": 35.0,
        "frictionCoeffNmSPerRad": 0.5,
    },
)


FROZEN_SCENARIOS: tuple[dict[str, Any], ...] = (
    {
        "id": "CR4-NOM-MJ-NOM",
        "payload_kg": 50.0,
        "tier": "nominal",
        "velocity_scale": 0.75,
        "acceleration_scale": 0.75,
        "primitive": "MoveJ",
        "speed": 1.0,
        "start": [0.0, 0.0, 0.0, 0.0],
        "target": [
            1.5415536976287865,
            0.898877593289879,
            0.9990302960023246,
            -9.042258053426622e-10,
        ],
    },
    {
        "id": "CR4-ZERO-ML-FAST",
        "payload_kg": 0.0,
        "tier": "fast",
        "velocity_scale": 1.0,
        "acceleration_scale": 1.0,
        "primitive": "MoveL",
        "speed": 1.0,
        "start": [
            1.541553735733068,
            0.9609910004498775,
            1.039232676381714,
            -1.1129717281121999e-7,
        ],
        "target": [
            1.5474723577497647,
            1.2991925307033179,
            0.6203535464305907,
            0.005918496578899646,
        ],
    },
    {
        "id": "CR4-HIGH-ML-SLOW",
        "payload_kg": 110.0,
        "tier": "slow",
        "velocity_scale": 0.5,
        "acceleration_scale": 0.5,
        "primitive": "MoveL",
        "speed": 0.25,
        "start": [0.0, 0.8505420354364523, 1.5119773393180762, -0.012249350584953987],
        "target": [0.0, 1.161062789016279, 0.8593033159371732, -0.012249350584953987],
    },
    {
        "id": "CR4-NOM-MJ-EXT",
        "payload_kg": 50.0,
        "tier": "nominal",
        "velocity_scale": 0.75,
        "acceleration_scale": 0.75,
        "primitive": "MoveJ",
        "speed": 1.0,
        "start": [0.0, 0.0, 0.0, 0.0],
        "target": [0.0, 1.46607670748006, 0.00183508772257135, 0.0],
    },
    {
        "id": "CR4-NOM-ML-NS",
        "payload_kg": 50.0,
        "tier": "nominal",
        "velocity_scale": 0.75,
        "acceleration_scale": 0.75,
        "primitive": "MoveL",
        "speed": 0.5,
        "start": [0.0, -0.680676707480057, -0.646206414793055, 0.0],
        "target": [0.0, -0.680676707480057, -0.680676707480057, 0.0],
    },
)


def _payload_inertia(mass_kg: float) -> list[list[float]]:
    side = 0.03
    p_inertia = mass_kg * (side * side + side * side) / 12.0
    return [
        [p_inertia, 0.0, 0.0],
        [0.0, p_inertia, 0.0],
        [0.0, 0.0, p_inertia],
    ]


def canonical_robot(payload_kg: float) -> dict[str, Any]:
    """Build the solver input from the canonical IRB460 source constants."""
    return {
        "schema": "robodimm.robot.v1",
        "kind": "CR4",
        "name": "CR4 (IRB460 preset)",
        "units": "SI",
        "geometry": copy.deepcopy(CANONICAL_GEOMETRY),
        # Empty inertials intentionally selects the canonical backend body
        # defaults, BODY_MASSES, rather than a benchmark fixture.
        "inertials": {},
        "payload": {
            "body": "PAYLOAD",
            "massKg": payload_kg,
            "comM": [0.0, 0.0, 0.0],
            "inertiaKgM2": _payload_inertia(payload_kg),
            "frame": "link",
        },
        "limits": copy.deepcopy(list(ORIGINAL_LIMITS)),
    }


def tcp_position(q: np.ndarray, geometry: dict[str, list[float]]) -> np.ndarray:
    """Match PalletizerEngine.forwardKinematics TCP placement for MoveL."""
    limits = ORIGINAL_LIMITS
    q_clamped = np.asarray(q, dtype=float).copy()
    for index, limit in enumerate(limits):
        q_clamped[index] = np.clip(
            q_clamped[index], limit["lowerLimitRad"], limit["upperLimitRad"]
        )
    points = closed_chain_points(
        Cr4GeometryContext(geometry), q_clamped[1], q_clamped[2]
    )
    planar_tcp = points["TCP"]
    c = math.cos(q_clamped[0])
    s = math.sin(q_clamped[0])
    return np.asarray(
        [c * planar_tcp[0] - s * planar_tcp[1],
         s * planar_tcp[0] + c * planar_tcp[1],
         planar_tcp[2]],
        dtype=float,
    )


def _quintic_segment(
    start_q: np.ndarray,
    target_q: np.ndarray,
    duration_s: float,
    instruction_index: int,
    start_time_s: float,
    dt_s: float,
) -> list[dict[str, Any]]:
    """Use the exact source quintic sampling and ceil/short-last-step rule."""
    steps = max(math.ceil(duration_s / dt_s), 1)
    step_dt = duration_s / steps
    delta = target_q - start_q
    result: list[dict[str, Any]] = []
    for step in range(1, steps + 1):
        ti = step * step_dt
        u = ti / duration_s
        scale = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
        scale_d = (30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4) / duration_s
        scale_dd = (60.0 * u - 180.0 * u**2 + 120.0 * u**3) / (duration_s**2)
        result.append(
            {
                "time_s": start_time_s + ti,
                "q": (start_q + scale * delta).tolist(),
                "qd": (scale_d * delta).tolist(),
                "qdd": (scale_dd * delta).tolist(),
                "instruction_index": instruction_index,
            }
        )
    return result


def frozen_trajectory(scenario: dict[str, Any], dt_s: float = DT_S) -> list[dict[str, Any]]:
    """Recreate buildProgramDynamicsTrajectory for one frozen instruction."""
    start_q = np.asarray(scenario["start"], dtype=float)
    target_q = np.asarray(scenario["target"], dtype=float)
    delta = target_q - start_q
    limits = ORIGINAL_LIMITS
    velocity_scale = float(scenario["velocity_scale"])
    acceleration_scale = float(scenario["acceleration_scale"])
    speed = max(float(scenario["speed"]), 1e-3)

    if scenario["primitive"] == "MoveJ":
        duration_s = (1.875 * float(np.max(np.abs(delta)))) / speed
    elif scenario["primitive"] == "MoveL":
        start_tcp = tcp_position(start_q, CANONICAL_GEOMETRY)
        target_tcp = tcp_position(target_q, CANONICAL_GEOMETRY)
        distance_m = float(np.linalg.norm(target_tcp - start_tcp))
        duration_s = distance_m / max(float(scenario["speed"]), 1e-4)
    else:
        raise ValueError(f"Unsupported frozen primitive: {scenario['primitive']}")

    # Protocol 3.2 scales copies used for trajectory generation.  The source
    # duration semantics take the maximum of command, velocity, and
    # acceleration durations, then enforce the sample interval.
    for index, limit in enumerate(limits):
        delta_i = abs(float(delta[index]))
        if delta_i > 1e-6:
            velocity_limit = limit["maxVelocityRadS"] * velocity_scale
            acceleration_limit = limit["maxAccelerationRadS2"] * acceleration_scale
            t_velocity = (1.875 * delta_i) / velocity_limit
            t_acceleration = math.sqrt((5.7735 * delta_i) / acceleration_limit)
            duration_s = max(duration_s, t_velocity, t_acceleration)
    duration_s = max(duration_s, dt_s)

    points: list[dict[str, Any]] = [
        {
            "time_s": 0.0,
            "q": start_q.tolist(),
            "qd": [0.0] * 4,
            "qdd": [0.0] * 4,
            "instruction_index": -1,
        }
    ]
    points.extend(
        _quintic_segment(start_q, target_q, duration_s, 0, 0.0, dt_s)
    )
    if len(points) < 2 or not np.all(np.diff([p["time_s"] for p in points]) > 0.0):
        raise AssertionError(f"Non-increasing trajectory for {scenario['id']}")
    if points[-1]["time_s"] > duration_s + 1e-12:
        raise AssertionError(f"Trajectory exceeds source duration for {scenario['id']}")
    if max(np.diff([p["time_s"] for p in points])) > dt_s + 1e-12:
        raise AssertionError(f"Trajectory step exceeds dt for {scenario['id']}")
    np.testing.assert_allclose(points[-1]["q"], target_q, rtol=0.0, atol=2e-14)
    return points


def _raw_evaluation(
    robot: dict[str, Any], trajectory: list[dict[str, Any]], h: float
) -> dict[str, Any]:
    samples = [
        {"time_s": point["time_s"], "q": point["q"], "qd": point["qd"], "qdd": point["qdd"]}
        for point in trajectory
    ]
    output_samples, diagnostics, warnings = compute_cr4_kkt_batch(
        robot, samples, {"fd_step": h}
    )
    if len(output_samples) != len(trajectory) or len(diagnostics) != len(trajectory):
        raise AssertionError("CR4 batch returned a different sample count")
    tau = np.asarray([sample["tau"] for sample in output_samples], dtype=float)
    if tau.shape != (len(trajectory), 4) or not np.all(np.isfinite(tau)):
        raise AssertionError("Non-finite or malformed CR4 torque output")

    # Keep every Section 7 diagnostic in memory for the sensitivity comparison;
    # only aggregate summaries are emitted in the concise JSON report.
    for diagnostic in diagnostics:
        for key in (
            "position_residual_vectors",
            "position_residual_norms",
            "position_residual_stacked_norm",
            "position_residual_max_norm",
            "velocity_closure_residual",
            "velocity_closure_residual_norm",
            "acceleration_closure_residual",
            "acceleration_closure_residual_norm",
            "passive_torque_residual",
            "passive_torque_residual_norm",
            "singular_values",
            "rank",
            "rank_tolerance",
            "condition_number",
        ):
            if key not in diagnostic:
                raise AssertionError(f"Missing Section 7 diagnostic field: {key}")
        if not np.all(np.isfinite(np.asarray(diagnostic["singular_values"], dtype=float))):
            raise AssertionError("Non-finite CR4 singular value")
        if diagnostic["rank"] != 6:
            raise AssertionError(f"Unexpected CR4 passive rank: {diagnostic['rank']}")

    return {
        "tau": tau,
        "diagnostics": diagnostics,
        "warnings": warnings,
        "times": np.asarray([sample["time_s"] for sample in samples], dtype=float),
    }


RESIDUAL_FIELDS: tuple[tuple[str, str, float], ...] = (
    ("position", "position_residual_max_norm", 1e-8),
    ("velocity", "velocity_closure_residual_norm", 1e-8),
    ("acceleration", "acceleration_closure_residual_norm", 1e-6),
    ("passive_torque", "passive_torque_residual_norm", 1e-8),
)


def _condition_values(diagnostics: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [
            math.inf if diagnostic["condition_number"] == "infinity" else float(diagnostic["condition_number"])
            for diagnostic in diagnostics
        ],
        dtype=float,
    )


def _scenario_summary(raw: dict[str, Any]) -> dict[str, Any]:
    diagnostics = raw["diagnostics"]
    condition_values = _condition_values(diagnostics)
    singular_values = np.asarray(
        [d["singular_values"] for d in diagnostics], dtype=float
    )
    residual_maxima = {
        name: float(max(float(d[key]) for d in diagnostics))
        for name, key, _target in RESIDUAL_FIELDS
    }
    return {
        "samples": int(raw["tau"].shape[0]),
        "duration_s": float(raw["times"][-1]),
        "max_abs_torque_Nm": float(np.max(np.abs(raw["tau"]))),
        "rank_min": int(min(int(d["rank"]) for d in diagnostics)),
        "condition_max": "infinity" if np.any(np.isinf(condition_values)) else float(np.max(condition_values)),
        "rank_tolerance_max": float(max(float(d["rank_tolerance"]) for d in diagnostics)),
        "singular_value_min": float(np.min(singular_values)),
        "singular_value_max": float(np.max(singular_values)),
        "residual_maxima": residual_maxima,
        "diagnostic_pass_count": int(sum(bool(d["diagnostics_pass"]) for d in diagnostics)),
        "warning_count": int(len(raw["warnings"])),
    }


def _overall_summary(raw_by_scenario: dict[str, dict[str, Any]]) -> dict[str, Any]:
    all_tau = np.concatenate([raw["tau"] for raw in raw_by_scenario.values()], axis=0)
    all_diagnostics = [
        diagnostic
        for raw in raw_by_scenario.values()
        for diagnostic in raw["diagnostics"]
    ]
    condition_values = _condition_values(all_diagnostics)
    singular_values = np.asarray(
        [d["singular_values"] for d in all_diagnostics], dtype=float
    )
    return {
        "samples": int(all_tau.shape[0]),
        "max_abs_torque_Nm": float(np.max(np.abs(all_tau))),
        "rank_min": int(min(int(d["rank"]) for d in all_diagnostics)),
        "condition_max": "infinity" if np.any(np.isinf(condition_values)) else float(np.max(condition_values)),
        "rank_tolerance_max": float(max(float(d["rank_tolerance"]) for d in all_diagnostics)),
        "singular_value_min": float(np.min(singular_values)),
        "singular_value_max": float(np.max(singular_values)),
        "residual_maxima": {
            name: float(max(float(d[key]) for d in all_diagnostics))
            for name, key, _target in RESIDUAL_FIELDS
        },
        "diagnostic_pass_count": int(sum(bool(d["diagnostics_pass"]) for d in all_diagnostics)),
    }


def _flatten_residual(raw_by_scenario: dict[str, dict[str, Any]], key: str) -> np.ndarray:
    return np.concatenate(
        [np.asarray([float(d[key]) for d in raw["diagnostics"]]) for raw in raw_by_scenario.values()]
    )


def _adjacent_summary(
    larger_h: float,
    smaller_h: float,
    larger: dict[str, dict[str, Any]],
    smaller: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    torque_changes: list[np.ndarray] = []
    torque_denominator = 1.0
    rank_identical = True
    condition_changes: list[np.ndarray] = []
    condition_identical = True
    residual_changes: dict[str, dict[str, float]] = {}
    residual_growth_stable = True
    residual_growth_excess: dict[str, float] = {}

    for scenario in FROZEN_SCENARIOS:
        scenario_id = scenario["id"]
        tau_large = larger[scenario_id]["tau"]
        tau_small = smaller[scenario_id]["tau"]
        if tau_large.shape != tau_small.shape:
            raise AssertionError(f"Sample mismatch between adjacent h values for {scenario_id}")
        torque_changes.append(np.abs(tau_large - tau_small).reshape(-1))
        torque_denominator = max(torque_denominator, float(np.max(np.abs(tau_small))))
        ranks_large = np.asarray([d["rank"] for d in larger[scenario_id]["diagnostics"]])
        ranks_small = np.asarray([d["rank"] for d in smaller[scenario_id]["diagnostics"]])
        rank_identical = rank_identical and bool(np.array_equal(ranks_large, ranks_small))
        conditions_large = _condition_values(larger[scenario_id]["diagnostics"])
        conditions_small = _condition_values(smaller[scenario_id]["diagnostics"])
        condition_identical = condition_identical and bool(
            np.array_equal(conditions_large, conditions_small)
        )
        if np.any(~np.isfinite(conditions_large)) or np.any(~np.isfinite(conditions_small)):
            if not np.array_equal(np.isinf(conditions_large), np.isinf(conditions_small)):
                condition_identical = False
            condition_changes.append(np.zeros_like(conditions_large))
        else:
            condition_changes.append(np.abs(conditions_large - conditions_small))

    change_values = np.concatenate(torque_changes)
    relative_changes = change_values / torque_denominator
    condition_change_values = np.concatenate(condition_changes)

    for name, key, target in RESIDUAL_FIELDS:
        large_values = _flatten_residual(larger, key)
        small_values = _flatten_residual(smaller, key)
        differences = np.abs(large_values - small_values)
        residual_changes[name] = {
            "median": float(np.median(differences)),
            "max": float(np.max(differences)),
        }
        excess = float(np.max(small_values - 1.1 * large_values - target))
        residual_growth_excess[name] = excess
        if excess > 0.0:
            residual_growth_stable = False

    d_h = float(np.max(change_values))
    r_h = d_h / torque_denominator
    torque_stable = d_h <= 1e-6 or r_h <= 1e-6
    stable = torque_stable and rank_identical and residual_growth_stable
    return {
        "larger_h": larger_h,
        "smaller_h": smaller_h,
        "median_D_h_Nm": float(np.median(change_values)),
        "D_h_Nm": d_h,
        "median_R_h": float(np.median(relative_changes)),
        "R_h": r_h,
        "torque_stable": torque_stable,
        "rank_identical": rank_identical,
        "condition_identical": condition_identical,
        "condition_changes": {
            "median": float(np.median(condition_change_values)),
            "max": float(np.max(condition_change_values)),
        },
        "residual_growth_stable": residual_growth_stable,
        "residual_growth_excess": residual_growth_excess,
        "residual_changes": residual_changes,
        "stable": stable,
    }


def run_campaign() -> dict[str, Any]:
    trajectories = {
        scenario["id"]: frozen_trajectory(scenario, DT_S)
        for scenario in FROZEN_SCENARIOS
    }
    robots = {
        scenario["id"]: canonical_robot(float(scenario["payload_kg"]))
        for scenario in FROZEN_SCENARIOS
    }

    evaluations: dict[float, dict[str, dict[str, Any]]] = {}
    per_h: list[dict[str, Any]] = []
    for h in FD_STEPS:
        by_scenario: dict[str, dict[str, Any]] = {}
        for scenario in FROZEN_SCENARIOS:
            scenario_id = scenario["id"]
            by_scenario[scenario_id] = _raw_evaluation(
                robots[scenario_id], trajectories[scenario_id], h
            )
        evaluations[h] = by_scenario
        per_h.append(
            {
                "h": h,
                "scenarios": {
                    scenario["id"]: _scenario_summary(by_scenario[scenario["id"]])
                    for scenario in FROZEN_SCENARIOS
                },
                "overall": _overall_summary(by_scenario),
            }
        )

    adjacent = [
        _adjacent_summary(FD_STEPS[index], FD_STEPS[index + 1], evaluations[FD_STEPS[index]], evaluations[FD_STEPS[index + 1]])
        for index in range(len(FD_STEPS) - 1)
    ]
    stable_adjacent_pairs = [
        index for index, pair in enumerate(adjacent) if pair["stable"]
    ]
    plateau_starts = [
        index for index in stable_adjacent_pairs
        if index + 1 in stable_adjacent_pairs
    ]
    if not plateau_starts:
        raise AssertionError("No two consecutive stable finite-difference pairs")
    selected_h = FD_STEPS[plateau_starts[0]]
    if selected_h != 1e-4:
        raise AssertionError(
            f"Expected selected largest h 1e-4, got {selected_h:g}; "
            f"stable adjacent pairs={stable_adjacent_pairs}"
        )

    return {
        "verification": "source verification only; not Simscape benchmark evidence",
        "simscape_benchmark_evidence": False,
        "robot": "canonical IRB460 CR4",
        "damping_protocol": "E2E-VM05-v1 (0.5 Nm/(rad/s))",
        "dt_s": DT_S,
        "trajectory_semantics": "source quintic duration semantics; MoveL uses TCP distance",
        "scenario_count": len(FROZEN_SCENARIOS),
        "scenario_sample_counts": {
            scenario["id"]: len(trajectories[scenario["id"]])
            for scenario in FROZEN_SCENARIOS
        },
        "H": list(FD_STEPS),
        "per_h": per_h,
        "adjacent": adjacent,
        "stable_pair_indices": stable_adjacent_pairs,
        "plateau_start_indices": plateau_starts,
        "selected_h": selected_h,
    }


class TestCr4FiniteDifferenceSensitivity(unittest.TestCase):
    """The same assertion is available to ordinary unittest discovery."""

    def test_selected_largest_h_is_1e_minus_4(self) -> None:
        summary = run_campaign()
        self.assertEqual(summary["selected_h"], 1e-4)


if __name__ == "__main__":
    # Direct execution is intentionally a machine-readable, single-line
    # report rather than unittest's verbose human-oriented test output.
    print(json.dumps(run_campaign(), sort_keys=True, separators=(",", ":")))
