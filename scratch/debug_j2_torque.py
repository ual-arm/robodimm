import sys
import json
import math
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append("/home/user/code/kineforge_ws/kineforge")

from backend.dynamics.cr4_kkt import compute_cr4_kkt_dynamics
from backend.dynamics.cr6_serial import compute_cr6_serial_dynamics

def get_cr6_default_program():
    targets = [
        {"name": "Target_10", "q": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
        {"name": "Target_20", "q": [0.2, 0.4, -0.3, 0.5, -0.6, 0.8]},
        {"name": "Target_30", "q": [0.5, -0.2, 0.4, -0.8, 0.9, -1.2]},
        {"name": "Target_40", "q": [-0.3, 0.5, -0.5, 1.2, -1.0, 1.5]},
        {"name": "Target_50", "q": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
        {"name": "Target_60", "q": [0.6, -0.4, 0.2, 0.8, -0.5, 0.0]},
        {"name": "Target_70", "q": [-0.5, 0.3, -0.2, -0.6, 0.8, -0.5]},
        {"name": "Target_80", "q": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
    ]
    instructions = [
        {"type": "MoveJ", "target_name": "Target_10", "speed_rad_s": 1.305, "zone_m": 0.1},
        {"type": "MoveJ", "target_name": "Target_20", "speed_rad_s": 1.305, "zone_m": 0.1},
        {"type": "MoveJ", "target_name": "Target_30", "speed_rad_s": 1.305, "zone_m": 0.1},
        {"type": "MoveL", "target_name": "Target_40", "tcp_speed_m_s": 1.0, "zone_m": 0.1},
        {"type": "MoveL", "target_name": "Target_50", "tcp_speed_m_s": 1.0, "zone_m": 0.1},
        {"type": "MoveJ", "target_name": "Target_60", "speed_rad_s": 1.305, "zone_m": 0.1},
        {"type": "MoveL", "target_name": "Target_70", "tcp_speed_m_s": 1.0, "zone_m": 0.1},
        {"type": "MoveJ", "target_name": "Target_10", "speed_rad_s": 1.305, "zone_m": 0.1}
    ]
    return {"targets": targets, "instructions": instructions}

def get_cr4_default_program():
    targets = [
        {"name": "Target_10", "q": [0.0, 0.0, 0.0, 0.0]},
        {"name": "Target_20", "q": [1.54155, 0.89888, 0.99903, 0.0]},
        {"name": "Target_30", "q": [1.54155, 0.96099, 1.03923, 0.0]},
        {"name": "Target_40", "q": [1.54747, 1.29919, 0.62035, 0.00592]},
        {"name": "Target_50", "q": [1.54747, 0.86525, -0.00149, 0.00592]},
        {"name": "Target_60", "q": [1.52930, -0.22091, 0.30493, -0.01225]},
        {"name": "Target_70", "q": [0.0, -0.22091, 0.30493, -0.01225]},
        {"name": "Target_80", "q": [0.0, -0.22091, 0.30493, -0.01225]},
        {"name": "Target_90", "q": [0.0, 0.85054, 1.51198, -0.01225]},
        {"name": "Target_100", "q": [0.0, 1.16106, 0.85930, -0.01225]}
    ]
    instructions = [
        {"type": "MoveJ", "target_name": "Target_10", "speed_rad_s": 1.0, "zone_m": 0.1},
        {"type": "MoveJ", "target_name": "Target_20", "speed_rad_s": 1.0, "zone_m": 0.1},
        {"type": "MoveL", "target_name": "Target_30", "tcp_speed_m_s": 1.0, "zone_m": 0.1},
        {"type": "MoveL", "target_name": "Target_40", "tcp_speed_m_s": 1.0, "zone_m": 0.1},
        {"type": "MoveL", "target_name": "Target_50", "tcp_speed_m_s": 1.0, "zone_m": 0.1},
        {"type": "MoveL", "target_name": "Target_60", "tcp_speed_m_s": 1.0, "zone_m": 0.1},
        {"type": "MoveJ", "target_name": "Target_70", "speed_rad_s": 1.0, "zone_m": 0.1},
        {"type": "MoveL", "target_name": "Target_90", "tcp_speed_m_s": 1.0, "zone_m": 0.1},
        {"type": "MoveL", "target_name": "Target_100", "tcp_speed_m_s": 1.0, "zone_m": 0.1},
        {"type": "MoveJ", "target_name": "Target_10", "speed_rad_s": 1.0, "zone_m": 0.1}
    ]
    return {"targets": targets, "instructions": instructions}

def append_quintic_segment(points, q0, q1, duration_s, dt_s):
    steps = max(int(math.ceil(duration_s / dt_s)), 1)
    step_dt = duration_s / steps
    last_time = points[-1]["time_s"] if len(points) > 0 else 0.0
    
    q0 = np.array(q0)
    q1 = np.array(q1)
    delta = q1 - q0
    
    for step in range(1, steps + 1):
        ti = step * step_dt
        u = ti / duration_s
        s = 10.0 * (u**3) - 15.0 * (u**4) + 6.0 * (u**5)
        sd = (30.0 * (u**2) - 60.0 * (u**3) + 30.0 * (u**4)) / duration_s
        sdd = (60.0 * u - 180.0 * (u**2) + 120.0 * (u**3)) / (duration_s**2)
        
        q = q0 + s * delta
        qd = sd * delta
        qdd = sdd * delta
        
        points.append({
            "time_s": last_time + ti,
            "q": q.tolist(),
            "qd": qd.tolist(),
            "qdd": qdd.tolist()
        })

def build_trajectory(program, robot_spec, dt_s=0.005):
    points = [{"time_s": 0.0, "q": program["targets"][0]["q"], "qd": [0.0]*len(program["targets"][0]["q"]), "qdd": [0.0]*len(program["targets"][0]["q"])}]
    current_q = program["targets"][0]["q"]
    
    for inst in program["instructions"]:
        target = next(t for t in program["targets"] if t["name"] == inst["target_name"])
        target_q = target["q"]
        
        # Determine duration
        if inst["type"] == "MoveJ":
            speed = max(inst["speed_rad_s"], 1e-3)
            max_delta = max(abs(t - c) for t, c in zip(target_q, current_q))
            duration_s = (1.875 * max_delta) / speed
        else:
            speed = max(inst.get("tcp_speed_m_s", 1.0), 1e-3)
            max_delta = max(abs(t - c) for t, c in zip(target_q, current_q))
            duration_s = (1.875 * max_delta) / speed
            
        # Limit by joint velocity & acceleration
        for i in range(len(current_q)):
            delta_qi = abs(target_q[i] - current_q[i])
            if delta_qi > 1e-6:
                limit = robot_spec["limits"][i]
                v_lim = limit.get("maxVelocityRadS", 3.0)
                a_lim = limit.get("maxAccelerationRadS2", 15.0)
                t_vel = (1.875 * delta_qi) / v_lim
                t_acc = math.sqrt((5.7735 * delta_qi) / a_lim)
                duration_s = max(duration_s, t_vel, t_acc)
                
        duration_s = max(duration_s, dt_s)
        append_quintic_segment(points, current_q, target_q, duration_s, dt_s)
        current_q = target_q
        
    return points

# Specs
cr6_spec = {
    "kind": "CR6",
    "limits": [
        {"name": "J1", "maxVelocityRadS": 3.054, "maxAccelerationRadS2": 12.0},
        {"name": "J2", "maxVelocityRadS": 3.054, "maxAccelerationRadS2": 12.0},
        {"name": "J3", "maxVelocityRadS": 3.054, "maxAccelerationRadS2": 12.0},
        {"name": "J4", "maxVelocityRadS": 4.363, "maxAccelerationRadS2": 25.0},
        {"name": "J5", "maxVelocityRadS": 4.363, "maxAccelerationRadS2": 25.0},
        {"name": "J6", "maxVelocityRadS": 6.283, "maxAccelerationRadS2": 25.0}
    ],
    "inertials": {
        "LINK1": {"massKg": 45.0, "comM": [0.175, 0.0, 0.495], "inertiaKgM2": [[0.920, 0.0, 0.0], [0.0, 0.920, 0.0], [0.0, 0.0, 0.004593]]},
        "LINK2": {"massKg": 65.0, "comM": [0.0, 0.0, 0.5475], "inertiaKgM2": [[6.498, 0.0, 0.0], [0.0, 6.498, 0.0], [0.0, 0.0, 0.006636]]},
        "LINK3": {"massKg": 40.0, "comM": [0.0, 0.0, 0.0875], "inertiaKgM2": [[0.104, 0.0, 0.0], [0.0, 0.104, 0.0], [0.0, 0.0, 0.004083]]},
        "LINK4": {"massKg": 28.0, "comM": [0.61525, 0.0, 0.0], "inertiaKgM2": [[0.002858, 0.0, 0.0], [0.0, 3.534399, 0.0], [0.0, 0.0, 3.534399]]},
        "LINK5": {"massKg": 17.0, "comM": [0.0, 0.0, 0.0], "inertiaKgM2": [[0.002499, 0.0, 0.0], [0.0, 0.002499, 0.0], [0.0, 0.0, 0.002499]]},
        "LINK6": {"massKg": 5.0,  "comM": [0.0425, 0.0, 0.0], "inertiaKgM2": [[0.000510, 0.0, 0.0], [0.0, 0.003265, 0.0], [0.0, 0.0, 0.003265]]}
    },
    "payload": {
        "massKg": 15.0, "comM": [0.0, 0.0, 0.0], "inertiaKgM2": [[0.00225, 0.0, 0.0], [0.0, 0.00225, 0.0], [0.0, 0.0, 0.00225]]
    },
    "geometry": {
        "joints": [
            {"name": "J1", "a_m": 0.175, "alpha_rad": -math.pi / 2.0, "d_m": 0.495, "theta_offset_rad": 0.0},
            {"name": "J2", "a_m": 1.095, "alpha_rad": 0.0, "d_m": 0.0, "theta_offset_rad": -math.pi / 2.0},
            {"name": "J3", "a_m": 0.175, "alpha_rad": -math.pi / 2.0, "d_m": 0.0, "theta_offset_rad": 0.0},
            {"name": "J4", "a_m": 0.0, "alpha_rad": math.pi / 2.0, "d_m": 1.2305, "theta_offset_rad": 0.0},
            {"name": "J5", "a_m": 0.0, "alpha_rad": math.pi / 2.0, "d_m": 0.0, "theta_offset_rad": math.pi},
            {"name": "J6", "a_m": 0.0, "alpha_rad": 0.0, "d_m": 0.085, "theta_offset_rad": 0.0}
        ],
        "tool_transform": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    }
}

cr4_spec = {
    "kind": "CR4",
    "limits": [
        {"name": "J1", "maxVelocityRadS": 2.53073, "maxAccelerationRadS2": 12.0},
        {"name": "J2", "maxVelocityRadS": 1.91986, "maxAccelerationRadS2": 10.0},
        {"name": "J3", "maxVelocityRadS": 2.09440, "maxAccelerationRadS2": 10.0},
        {"name": "J4", "maxVelocityRadS": 6.98132, "maxAccelerationRadS2": 35.0}
    ],
    "inertials": {
        "SWING":      {"massKg": 90.0},
        "P_ARM":      {"massKg": 35.0},
        "LOWER_ARM":  {"massKg": 75.0},
        "P_LINK":     {"massKg": 25.0},
        "UPPER_ARM":  {"massKg": 40.0},
        "LOWER_LINK": {"massKg": 20.0},
        "LINK_PLATE": {"massKg": 15.0},
        "UPPER_LINK": {"massKg": 15.0},
        "TILT":       {"massKg": 15.0},
        "DISK":       {"massKg": 10.0}
    },
    "payload": {
        "massKg": 50.0, "comM": [0.0, 0.0, 0.0], "inertiaKgM2": [[0.0075, 0.0, 0.0], [0.0, 0.0075, 0.0], [0.0, 0.0, 0.0075]]
    },
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
        "O_B": 0.4,
        "O_C": 0.945,
        "B_P": 0.945,
        "P_H": 1.425,
        "P_C": 0.4,
        "C_H": 1.025,
        "D_E": 0.945,
        "C_E": 0.3,
        "C_F": 0.3,
        "E_F": 0.4975,
        "F_G": 1.0249,
        "H_G": 0.3,
        "H_EE": 0.22,
        "G_EE": 0.3166,
        "EE_TCP": 0.2515
    }
}

print("=== DECOMPOSING CR6 J2 TORQUES ===")
cr6_prog = get_cr6_default_program()
cr6_traj = build_trajectory(cr6_prog, cr6_spec)
cr6_tau_total = []
cr6_tau_grav = []
for p in cr6_traj:
    tau_t, _, _ = compute_cr6_serial_dynamics(cr6_spec, p["q"], p["qd"], p["qdd"])
    tau_g, _, _ = compute_cr6_serial_dynamics(cr6_spec, p["q"], [0]*6, [0]*6)
    cr6_tau_total.append(tau_t[1])
    cr6_tau_grav.append(tau_g[1])

cr6_tau_total = np.array(cr6_tau_total)
cr6_tau_grav = np.array(cr6_tau_grav)
cr6_tau_dyn = cr6_tau_total - cr6_tau_grav

print(f"J2 RMS total torque: {np.sqrt(np.mean(cr6_tau_total**2)):.2f} Nm")
print(f"J2 RMS gravity torque: {np.sqrt(np.mean(cr6_tau_grav**2)):.2f} Nm")
print(f"J2 RMS dynamic torque: {np.sqrt(np.mean(cr6_tau_dyn**2)):.2f} Nm")
print(f"J2 Peak total torque: {np.max(np.abs(cr6_tau_total)):.2f} Nm")
print(f"J2 Peak gravity torque: {np.max(np.abs(cr6_tau_grav)):.2f} Nm")
print(f"J2 Peak dynamic torque: {np.max(np.abs(cr6_tau_dyn)):.2f} Nm")

print("\n=== DECOMPOSING CR4 J2 TORQUES ===")
cr4_prog = get_cr4_default_program()
cr4_traj = build_trajectory(cr4_prog, cr4_spec)
cr4_tau_total = []
cr4_tau_grav = []
for p in cr4_traj:
    tau_t, _, _, _ = compute_cr4_kkt_dynamics(cr4_spec, p["q"], p["qd"], p["qdd"])
    tau_g, _, _, _ = compute_cr4_kkt_dynamics(cr4_spec, p["q"], [0]*4, [0]*4)
    cr4_tau_total.append(tau_t[1])
    cr4_tau_grav.append(tau_g[1])

cr4_tau_total = np.array(cr4_tau_total)
cr4_tau_grav = np.array(cr4_tau_grav)
cr4_tau_dyn = cr4_tau_total - cr4_tau_grav

print(f"J2 RMS total torque: {np.sqrt(np.mean(cr4_tau_total**2)):.2f} Nm")
print(f"J2 RMS gravity torque: {np.sqrt(np.mean(cr4_tau_grav**2)):.2f} Nm")
print(f"J2 RMS dynamic torque: {np.sqrt(np.mean(cr4_tau_dyn**2)):.2f} Nm")
print(f"J2 Peak total torque: {np.max(np.abs(cr4_tau_total)):.2f} Nm")
print(f"J2 Peak gravity torque: {np.max(np.abs(cr4_tau_grav)):.2f} Nm")
print(f"J2 Peak dynamic torque: {np.max(np.abs(cr4_tau_dyn)):.2f} Nm")
