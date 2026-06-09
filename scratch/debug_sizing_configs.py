import sys
import json
import math
import numpy as np
from pathlib import Path

# Add directories to path
sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append("/home/user/code/kineforge_ws/kineforge")

from backend.dynamics.cr4_kkt import compute_cr4_kkt_dynamics
from backend.dynamics.cr6_serial import compute_cr6_serial_dynamics
from scratch.debug_j2_torque import (
    get_cr6_default_program,
    get_cr4_default_program,
    build_trajectory,
    cr6_spec,
    cr4_spec
)


def compute_demands(traj, spec_kind, robot_spec):
    taus = []
    speeds = []
    for p in traj:
        if spec_kind == 'CR6':
            tau, _, _ = compute_cr6_serial_dynamics(robot_spec, p["q"], p["qd"], p["qdd"])
        else:
            tau, _, _, _ = compute_cr4_kkt_dynamics(robot_spec, p["q"], p["qd"], p["qdd"])
        taus.append(tau[1])
        speeds.append(p["qd"][1])  # J2 joint speed

    taus = np.array(taus)
    speeds = np.array(speeds)
    return {
        "tau_rms": float(np.sqrt(np.mean(taus**2))),
        "tau_peak": float(np.max(np.abs(taus))),
        "speed_peak": float(np.max(np.abs(speeds)))
    }


def evaluate_candidate(motor, gearbox, demand, margins):
    efficiency = gearbox["efficiency"]
    ratio = gearbox["ratio"]
    motor_peak_factor = margins.get("motorPeakFactor", 5.0)

    tau_out_cont = motor["rated_torque_Nm"] * ratio * efficiency
    tau_out_peak = motor["rated_torque_Nm"] * motor_peak_factor * ratio * efficiency
    omega_out_max = (motor["no_load_speed_rpm"] / ratio) * (2.0 * math.pi / 60.0)

    required_cont = demand["tau_rms"] * margins["continuous"]
    required_peak = demand["tau_peak"] * margins["peak"]
    required_speed = demand["speed_peak"] * margins["speed"]

    speed_peak_rpm = demand["speed_peak"] * (60.0 / (2.0 * math.pi))
    required_gb_input_speed_rpm = speed_peak_rpm * ratio * margins["speed"]

    failure_reasons = []
    if tau_out_cont < required_cont:
        failure_reasons.append(f"ContTorque({tau_out_cont:.1f}<{required_cont:.1f})")
    if tau_out_peak < required_peak:
        failure_reasons.append(f"PeakTorque({tau_out_peak:.1f}<{required_peak:.1f})")
    if omega_out_max < required_speed:
        failure_reasons.append(f"Speed({omega_out_max:.2f}<{required_speed:.2f})")
    if gearbox["max_continuous_torque_Nm"] < required_cont:
        failure_reasons.append(f"GBContTorque({gearbox['max_continuous_torque_Nm']:.1f}<{required_cont:.1f})")
    if gearbox["max_intermittent_torque_Nm"] < required_peak:
        failure_reasons.append(f"GBIntermTorque({gearbox['max_intermittent_torque_Nm']:.1f}<{required_peak:.1f})")
    if gearbox["max_input_speed_rpm"] < required_gb_input_speed_rpm:
        failure_reasons.append(f"GBInputSpeed({gearbox['max_input_speed_rpm']:.1f}<{required_gb_input_speed_rpm:.1f})")

    passes = len(failure_reasons) == 0

    def safe_margin(available, demanded):
        return available / demanded if demanded > 1e-6 else float('inf')

    min_margin = min(
        safe_margin(tau_out_cont, demand["tau_rms"]),
        safe_margin(tau_out_peak, demand["tau_peak"]),
        safe_margin(omega_out_max, demand["speed_peak"]),
        safe_margin(gearbox["max_continuous_torque_Nm"], demand["tau_rms"]),
        safe_margin(gearbox["max_intermittent_torque_Nm"], demand["tau_peak"])
    )

    return {
        "passes": passes,
        "failure_reasons": failure_reasons,
        "min_margin": min_margin,
        "total_mass": motor["mass_kg"] + gearbox["mass_kg"],
        "motor_kW": motor["rated_power_W"] / 1000.0,
        "motor_torque": motor["rated_torque_Nm"],
        "ratio": ratio,
        "gearbox_type": gearbox["type"],
        "gearbox_name": gearbox["name"],
        "motor_name": motor["name"],
        "gb_cont": gearbox["max_continuous_torque_Nm"],
        "gb_interm": gearbox["max_intermittent_torque_Nm"]
    }


def rank_candidates(candidates, objective):
    def sort_key(c):
        passes_first = 0 if c["passes"] else 1
        if objective == "min_power":
            return (passes_first, c["motor_kW"], c["total_mass"], c["ratio"])
        elif objective == "min_gearbox":
            return (passes_first, c["total_mass"], c["motor_kW"], c["ratio"])
        elif objective == "max_margin":
            margin_key = -c["min_margin"] if c["min_margin"] != float('inf') else -1e9
            return (passes_first, margin_key, c["total_mass"], c["motor_kW"])
        else:  # min_mass
            return (passes_first, c["total_mass"], c["motor_kW"], c["ratio"])
    return sorted(candidates, key=sort_key)


def run_sizing_analysis(robot_name, demand, library, configs):
    motors = {m["id"]: m for m in library["motors"]}
    gearboxes = {g["id"]: g for g in library["gearboxes"]}
    matrix = library["compatibility_matrix"]

    print(f"\n# Robot: {robot_name}")
    print(f"Demands: RMS={demand['tau_rms']:.1f} Nm  Peak={demand['tau_peak']:.1f} Nm  Speed_peak={demand['speed_peak']:.3f} rad/s")

    for label, conf in configs.items():
        margins = {
            "continuous": conf["continuous"],
            "peak": conf.get("peak", 1.2),
            "speed": conf["speed"],
            "motorPeakFactor": 5.0
        }
        objective = conf["objective"]

        raw_candidates = []
        for motor_id, gb_ids in matrix.items():
            if motor_id not in motors:
                continue
            motor = motors[motor_id]
            for gb_id in gb_ids:
                if gb_id not in gearboxes:
                    continue
                raw_candidates.append(evaluate_candidate(motor, gearboxes[gb_id], demand, margins))

        ranked = rank_candidates(raw_candidates, objective)

        print(f"\n## Config {label}: cont={conf['continuous']}, speed={conf['speed']}, objective={conf['objective']}")
        print("| Rk | Pass | Motor kW | Motor Nm | Gearbox | Ratio | GB Cont Nm | GB Peak Nm | Mass kg | MinMargin | Failure |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for idx, c in enumerate(ranked[:20]):
            reasons = ", ".join(c["failure_reasons"]) if c["failure_reasons"] else "—"
            margin_str = f"{c['min_margin']:.2f}x" if c['min_margin'] != float('inf') else "∞"
            print(f"| {idx+1} | {'YES' if c['passes'] else 'NO'} | {c['motor_kW']:.2f} | "
                  f"{c['motor_torque']:.1f} | {c['gearbox_name']} | {c['ratio']} | "
                  f"{c['gb_cont']} | {c['gb_interm']} | {c['total_mass']:.1f} | {margin_str} | {reasons} |")


def main():
    lib_path = Path("/home/user/code/kineforge_ws/robodimm/public/actuators_library.json")
    with open(lib_path, "r") as f:
        library = json.load(f)

    configs = {
        "A": {"continuous": 1.3, "speed": 1.15, "objective": "min_mass",  "peak": 1.2},
        "B": {"continuous": 1.1, "speed": 1.05, "objective": "min_power", "peak": 1.2},
        "C": {"continuous": 1.0, "speed": 1.0,  "objective": "min_power", "peak": 1.1},
    }

    # CR6
    cr6_prog = get_cr6_default_program()
    cr6_traj = build_trajectory(cr6_prog, cr6_spec)
    run_sizing_analysis("CR6 (IRB4600-45/2.05)",
                        compute_demands(cr6_traj, 'CR6', cr6_spec), library, configs)

    # CR4  — note: cr4_spec imported from debug_j2_torque already has corrected masses
    cr4_prog = get_cr4_default_program()
    cr4_traj = build_trajectory(cr4_prog, cr4_spec)
    run_sizing_analysis("CR4 (IRB460)",
                        compute_demands(cr4_traj, 'CR4', cr4_spec), library, configs)


if __name__ == "__main__":
    main()
