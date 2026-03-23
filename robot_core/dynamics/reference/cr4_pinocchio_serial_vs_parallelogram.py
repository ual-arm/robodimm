"""
Comparacion Pinocchio:
1) CR4 serial puro (5 GDL tipo DH)
2) CR4 con q3act + paralelogramo (modelo real de robodimm)

Se usan las mismas masas/inercias principales del ensayo serial (cr4_params.mat),
y para manivela/biela se emplean inercias ligeras.
"""

import os
import sys
import numpy as np

try:
    import pinocchio as pin
except Exception as exc:
    raise RuntimeError(
        "No se pudo importar pinocchio. Ejecuta en robot_env con pinocchio instalado."
    ) from exc

try:
    import scipy.io as sio
except Exception as exc:
    raise RuntimeError(
        "No se pudo importar scipy.io (necesario para cargar cr4_params.mat)."
    ) from exc

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

import cr4_newton_euler_motors as ne


ROBODIMM_PATH_DEFAULT = r"C:\Users\josel\Documents\GitHub\robodimm"
G = 9.81

# Offsets DH usados en los ensayos previos (q_theta = q_user + offset)
CR4_DH_OFFSETS = np.array([0.0, -np.pi / 2, np.pi / 2, np.pi, 0.0], dtype=float)

# Masas muy bajas para paralelogramo secundario
CRANK_MASS = 0.001
ROD_MASS = 0.001
CONNECTOR_MASS = 0.02
LIGHT_INERTIA_DIAG = np.array([1e-5, 1e-5, 1e-5], dtype=float)

# Parametros de motor por defecto (mismo criterio usado en ensayos seriales)
M_MOT_DEFAULT = 1.0
I_MOT_DEFAULT = 5.0e-7
GR_MOT_DEFAULT = 100.0
# Convencion de actuador para J4 (RobotStudio/Z-down): q_real(J4) = -q_act(J4)
J4_ACT_SIGN = -1.0


def _results_dir():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(p, exist_ok=True)
    return p


def _signal_metrics(ref, est):
    ref = np.asarray(ref, dtype=float).reshape(-1)
    est = np.asarray(est, dtype=float).reshape(-1)
    err = est - ref

    corr = float(np.corrcoef(est, ref)[0, 1]) if np.std(est) > 0 and np.std(ref) > 0 else np.nan
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))

    A = np.vstack([est, np.ones_like(est)]).T
    slope, intercept = np.linalg.lstsq(A, ref, rcond=None)[0]

    return {
        "corr": corr,
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "slope": float(slope),
        "intercept": float(intercept),
    }


def _load_csv_columns(csv_file):
    data = np.genfromtxt(csv_file, delimiter=",", names=True)
    if data.size == 0:
        raise ValueError(f"CSV vacio o no legible: {csv_file}")

    names = list(data.dtype.names or [])
    required = [
        "q1",
        "q2",
        "q3",
        "q4",
        "q5",
        "qd1",
        "qd2",
        "qd3",
        "qd4",
        "qd5",
        "qdd1",
        "qdd2",
        "qdd3",
        "qdd4",
        "qdd5",
    ]
    missing = [c for c in required if c not in names]
    if missing:
        raise ValueError(f"Faltan columnas en CSV: {missing}")

    out = {"time": data["time"] if "time" in names else np.arange(data.shape[0], dtype=float)}
    for col in required:
        out[col] = data[col]
    return out


def _as_array3(x):
    a = np.array(x, dtype=float).reshape(-1)
    if a.size != 3:
        raise ValueError(f"Se esperaban 3 componentes, recibido {a}")
    return a


def _append_point_mass_at_placement(model, host_joint_id, mass, placement):
    if mass <= 0.0:
        return
    # Minima inercia rotacional para evitar singularidades numericas.
    I_eps = np.diag([1e-12, 1e-12, 1e-12])
    motor_inertia = pin.Inertia(float(mass), np.zeros(3), I_eps)
    model.appendBodyToJoint(host_joint_id, motor_inertia, placement)


def _append_stator_on_parent_side_of_joint(model, target_joint_id, mass):
    if mass <= 0.0:
        return
    parent_joint_id = model.parents[target_joint_id]
    if parent_joint_id < 0:
        return
    placement_parent_to_target = model.jointPlacements[target_joint_id]
    _append_point_mass_at_placement(model, parent_joint_id, mass, placement_parent_to_target)


def _static_dh_se3(a, alpha, d):
    ca, sa = np.cos(alpha), np.sin(alpha)
    R = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, ca, -sa],
            [0.0, sa, ca],
        ],
        dtype=float,
    )
    t = np.array([a, 0.0, d], dtype=float)
    return pin.SE3(R, t)


def load_params_from_mat(mat_file="cr4_params.mat", n_links=5):
    # Reutiliza el mismo mapeo legacy->serial5 del modulo NE para evitar
    # inconsistencias de COM/inercia entre scripts.
    return ne.load_params_from_mat(mat_file=mat_file, n_links=n_links)


def build_serial_model(params):
    model = pin.Model()
    model.gravity.linear = np.array([0.0, 0.0, -G], dtype=float)

    dh_a = params["dh_a"]
    dh_alpha = params["dh_alpha"]
    dh_d = params["dh_d"]
    masses = params["masses"]
    coms = params["coms"]
    inertias = params["inertias"]

    parent = 0
    joint_ids = []
    for i in range(5):
        joint_placement = pin.SE3.Identity() if i == 0 else _static_dh_se3(dh_a[i - 1], dh_alpha[i - 1], dh_d[i - 1])
        jid = model.addJoint(parent, pin.JointModelRZ(), joint_placement, f"joint{i+1}")
        joint_ids.append(jid)

        body_inertia = pin.Inertia(masses[i], coms[i], inertias[i])
        body_placement = _static_dh_se3(dh_a[i], dh_alpha[i], dh_d[i])
        model.appendBodyToJoint(jid, body_inertia, body_placement)
        parent = jid

    ee_placement = _static_dh_se3(dh_a[-1], dh_alpha[-1], dh_d[-1])
    ee_frame = pin.Frame("ee", joint_ids[-1], joint_ids[-1], ee_placement, pin.FrameType.OP_FRAME)
    ee_frame_id = model.addFrame(ee_frame)
    return model, ee_frame_id, joint_ids


def build_real_parallelogram_model(params, robodimm_path=ROBODIMM_PATH_DEFAULT, robot_scale=1.0):
    if not os.path.isdir(robodimm_path):
        raise FileNotFoundError(f"No existe robodimm en: {robodimm_path}")

    if robodimm_path not in sys.path:
        sys.path.insert(0, robodimm_path)

    try:
        from robot_core.builders.cr4 import build_cr4_real
    except Exception as exc:
        raise RuntimeError("No se pudo importar build_cr4_real desde robodimm.") from exc

    model, _, constraint_model = build_cr4_real(scale=float(robot_scale))
    model.gravity.linear = np.array([0.0, 0.0, -G], dtype=float)

    idx_v = {model.names[i]: model.joints[i].idx_v for i in range(1, len(model.names))}
    return model, constraint_model, idx_v


def _apply_light_parallelogram_inertias(real_model):
    name_to_joint_id = {real_model.names[i]: i for i in range(1, len(real_model.names))}
    light_I = np.diag(LIGHT_INERTIA_DIAG)
    for jn, mass in (("J3real", CRANK_MASS), ("J1p", ROD_MASS), ("J2p", CONNECTOR_MASS)):
        jid = name_to_joint_id[jn]
        real_model.inertias[jid] = pin.Inertia(mass, np.zeros(3), light_I)


def apply_serial_motor_stator_masses(serial_model, serial_joint_ids, m_axis4):
    """
    Replica el mismo criterio validado con Simscape para modelo serial:
    - masa de J2 en Link1
    - masa de J3 en Link2
    - masa de J4 (wrist Z) en Link4
    - masa de J1 en base fija (no se modela en cadena movil)
    """
    m_axis4 = np.asarray(m_axis4, dtype=float).reshape(4)
    if np.all(m_axis4 <= 0.0):
        return

    # Estatores en lado BASE de joints actuados:
    # J2, J3, J5(q5=J4 wrist Z). J1 cae en base fija y no entra en la cadena movil.
    _append_stator_on_parent_side_of_joint(serial_model, serial_joint_ids[1], m_axis4[1])  # J2
    _append_stator_on_parent_side_of_joint(serial_model, serial_joint_ids[2], m_axis4[2])  # J3act
    _append_stator_on_parent_side_of_joint(serial_model, serial_joint_ids[4], m_axis4[3])  # J4_TCP


def apply_parallel_motor_stator_masses(real_model, m_axis4, layout="concentric_j2_j3act"):
    """
    Aplica masas de estator en el modelo real.
    layouts:
      - concentric_j2_j3act: J2 y J3act concentricos -> ambas masas aguas arriba (J1)
      - serial_like: referencia sin concentricidad (J2 aguas arriba en J1, motor3 en codo via J3)
    En ambos casos la masa de J4 se coloca aguas arriba del wrist (J_aux).
    """
    m_axis4 = np.asarray(m_axis4, dtype=float).reshape(4)
    if np.all(m_axis4 <= 0.0):
        return

    if layout not in {"concentric_j2_j3act", "serial_like"}:
        raise ValueError("layout debe ser 'concentric_j2_j3act' o 'serial_like'")

    name_to_joint_id = {real_model.names[i]: i for i in range(1, len(real_model.names))}

    # Motor de J4 siempre en lado BASE de J4.
    target_joint_names = [("J4", m_axis4[3])]

    if layout == "concentric_j2_j3act":
        # J2 y J3act concentricos: ambos estatores en lado BASE de J2/J3real.
        target_joint_names += [("J2", m_axis4[1]), ("J3real", m_axis4[2])]
    else:
        # Referencia "tipo serial": J2 en lado BASE de J2 y motor3 desplazado al codo (BASE de J3).
        target_joint_names += [("J2", m_axis4[1]), ("J3", m_axis4[2])]

    for name, mass in target_joint_names:
        target_jid = name_to_joint_id[name]
        _append_stator_on_parent_side_of_joint(real_model, target_jid, mass)


def apply_payload_to_serial_model(serial_model, serial_joint_ids, params, payload):
    p_mass = float(payload.get("mass", 0.0))
    if p_mass <= 0.0:
        return

    p_com_tcp = np.array(payload.get("com_from_tcp", np.zeros(3)), dtype=float).reshape(3)
    p_inertia_tcp = np.array(payload.get("inertia", np.zeros((3, 3))), dtype=float).reshape(3, 3)

    a = float(params["dh_a"][-1])
    alpha = float(params["dh_alpha"][-1])
    d = float(params["dh_d"][-1])
    j5_to_tcp = _static_dh_se3(a, alpha, d)
    tcp_to_payload = pin.SE3(np.eye(3), p_com_tcp)
    j5_to_payload = j5_to_tcp * tcp_to_payload

    I_payload = pin.Inertia(p_mass, np.zeros(3), p_inertia_tcp)
    serial_model.appendBodyToJoint(serial_joint_ids[-1], I_payload, j5_to_payload)


def apply_payload_to_real_model(real_model, payload, tool_frame_name="tool0"):
    p_mass = float(payload.get("mass", 0.0))
    if p_mass <= 0.0:
        return

    p_com_tcp = np.array(payload.get("com_from_tcp", np.zeros(3)), dtype=float).reshape(3)
    p_inertia_tcp = np.array(payload.get("inertia", np.zeros((3, 3))), dtype=float).reshape(3, 3)

    tool_id = real_model.getFrameId(tool_frame_name)
    tool_frame = real_model.frames[tool_id]
    parent_joint = tool_frame.parentJoint
    j_to_tcp = tool_frame.placement
    tcp_to_payload = pin.SE3(np.eye(3), p_com_tcp)
    j_to_payload = j_to_tcp * tcp_to_payload

    I_payload = pin.Inertia(p_mass, np.zeros(3), p_inertia_tcp)
    real_model.appendBodyToJoint(parent_joint, I_payload, j_to_payload)


def apply_inertia_mapping_direct_link_copy(real_model, params):
    masses = params["masses"]
    coms = params["coms"]
    inertias = params["inertias"]
    name_to_joint_id = {real_model.names[i]: i for i in range(1, len(real_model.names))}

    main_map = {
        "J1": 0,
        "J2": 1,
        "J3": 2,
        "J_aux": 3,
        "J4": 4,
    }
    for jn, li in main_map.items():
        jid = name_to_joint_id[jn]
        real_model.inertias[jid] = pin.Inertia(masses[li], coms[li], inertias[li])

    _apply_light_parallelogram_inertias(real_model)


def apply_inertia_mapping_serial_joint_transfer(
    serial_model,
    serial_data,
    serial_joint_ids,
    real_model,
    real_data,
    constraint_model,
):
    try:
        from robot_core.conversions import q_pink_to_real
    except Exception as exc:
        raise RuntimeError("No se pudo importar q_pink_to_real para transferencia de inercias.") from exc

    # Home consistente con los ensayos: q_theta = offsets
    q_serial_home = CR4_DH_OFFSETS.copy()
    q5_user_home = np.zeros(5, dtype=float)
    q_real_home = q_pink_to_real("CR4", real_model, real_data, constraint_model, q5_user_home)

    pin.forwardKinematics(serial_model, serial_data, q_serial_home)
    pin.updateFramePlacements(serial_model, serial_data)
    pin.forwardKinematics(real_model, real_data, q_real_home)
    pin.updateFramePlacements(real_model, real_data)

    name_to_joint_id = {real_model.names[i]: i for i in range(1, len(real_model.names))}
    pairs = [
        ("J1", serial_joint_ids[0]),
        ("J2", serial_joint_ids[1]),
        ("J3", serial_joint_ids[2]),
        ("J_aux", serial_joint_ids[3]),
        ("J4", serial_joint_ids[4]),
    ]

    for real_name, serial_jid in pairs:
        real_jid = name_to_joint_id[real_name]
        I_serial = serial_model.inertias[serial_jid]

        # Pose del frame serial visto desde el frame real.
        T_rs = real_data.oMi[real_jid].inverse() * serial_data.oMi[serial_jid]
        R = T_rs.rotation
        t = T_rs.translation

        com_real = R @ I_serial.lever + t
        inertia_real = R @ I_serial.inertia @ R.T
        real_model.inertias[real_jid] = pin.Inertia(I_serial.mass, com_real, inertia_real)

    _apply_light_parallelogram_inertias(real_model)


def theta_to_user5(q_theta):
    return np.asarray(q_theta, dtype=float).reshape(5) - CR4_DH_OFFSETS


def build_real_state_from_user5(q5_user, qd5_user, qdd5_user, idx_v, nq, nv):
    q_real = np.zeros(nq, dtype=float)
    v_real = np.zeros(nv, dtype=float)
    a_real = np.zeros(nv, dtype=float)

    # q5_user = [J1, J2, J3_rel, J_aux, J4]
    # q3act = J2 + J3_rel
    # J_aux = -q3act (se fuerza por consistencia del mecanismo)
    j1 = q5_user[0]
    j2 = q5_user[1]
    j3_rel = q5_user[2]
    j4 = q5_user[4]
    j3act = j2 + j3_rel
    jaux = -j3act

    jd1 = qd5_user[0]
    jd2 = qd5_user[1]
    jd3_rel = qd5_user[2]
    jd4 = qd5_user[4]
    jd3act = jd2 + jd3_rel
    jdaux = -jd3act

    jdd1 = qdd5_user[0]
    jdd2 = qdd5_user[1]
    jdd3_rel = qdd5_user[2]
    jdd4 = qdd5_user[4]
    jdd3act = jdd2 + jdd3_rel
    jddaux = -jdd3act

    q_real[idx_v["J1"]] = j1
    q_real[idx_v["J2"]] = j2
    q_real[idx_v["J3"]] = j3_rel
    q_real[idx_v["J_aux"]] = jaux
    # Convencion CR4 real: J4 interno va invertido respecto al actuador exportado.
    q_real[idx_v["J4"]] = J4_ACT_SIGN * j4
    q_real[idx_v["J3real"]] = j3act
    q_real[idx_v["J1p"]] = j2 - j3act
    q_real[idx_v["J2p"]] = 0.0

    v_real[idx_v["J1"]] = jd1
    v_real[idx_v["J2"]] = jd2
    v_real[idx_v["J3"]] = jd3_rel
    v_real[idx_v["J_aux"]] = jdaux
    v_real[idx_v["J4"]] = J4_ACT_SIGN * jd4
    v_real[idx_v["J3real"]] = jd3act
    v_real[idx_v["J1p"]] = jd2 - jd3act
    v_real[idx_v["J2p"]] = 0.0

    a_real[idx_v["J1"]] = jdd1
    a_real[idx_v["J2"]] = jdd2
    a_real[idx_v["J3"]] = jdd3_rel
    a_real[idx_v["J_aux"]] = jddaux
    a_real[idx_v["J4"]] = J4_ACT_SIGN * jdd4
    a_real[idx_v["J3real"]] = jdd3act
    a_real[idx_v["J1p"]] = jdd2 - jdd3act
    a_real[idx_v["J2p"]] = 0.0

    return q_real, v_real, a_real


def build_actuator_projection_matrix(idx_v, nv):
    """
    Matriz B para mapear coordenadas actuadas qa=[J1,J2,J3act,J4]
    a coordenadas generalizadas del modelo real:
      q_real = B * qa + const
    """
    B = np.zeros((nv, 4), dtype=float)
    # J1
    B[idx_v["J1"], 0] = 1.0
    # J2
    B[idx_v["J2"], 1] = 1.0
    # J3act
    B[idx_v["J3real"], 2] = 1.0
    # J1p = J2 - J3act
    B[idx_v["J1p"], 1] = 1.0
    B[idx_v["J1p"], 2] = -1.0
    # J3 = J3act - J2
    B[idx_v["J3"], 1] = -1.0
    B[idx_v["J3"], 2] = 1.0
    # J_aux = -J3act
    B[idx_v["J_aux"], 2] = -1.0
    # J4 (convencion actuador vs coordenada interna del modelo real)
    B[idx_v["J4"], 3] = J4_ACT_SIGN
    return B


def serial_tau5_to_actuator_tau4(tau5):
    """
    Mapea torques generalizados seriales q=[J1,J2,J3_rel,J_aux,J4]
    a torques de actuadores [J1,J2,J3act,J4] con:
      J3_rel = J3act - J2
      J_aux  = -J3act
    """
    t = np.asarray(tau5, dtype=float).reshape(5)
    return np.array(
        [
            t[0],
            t[1] - t[2],
            t[2],
            t[4],
        ],
        dtype=float,
    )


def compute_constrained_inverse_dynamics_native(model, data, constraint_model, q, v, a, mu=1e-8):
    """
    Inversa dinamica con cadena cerrada usando API nativa de Pinocchio.
    Basado en formulacion KKT con ContactCholeskyDecomposition.
    """
    if constraint_model is None:
        return pin.rnea(model, data, q, v, a)

    constraint_models = [constraint_model]
    constraint_datas = [constraint_model.createData()]
    nc = constraint_model.size()

    pin.computeAllTerms(model, data, q, v)
    pin.computeJointJacobians(model, data, q)

    M = data.M.copy()
    h = data.nle.copy()

    kkt = pin.ContactCholeskyDecomposition(model, constraint_models)
    kkt.compute(model, data, constraint_models, constraint_datas, mu)

    Jc = pin.getConstraintsJacobian(model, data, constraint_models, constraint_datas)

    pin.forwardKinematics(model, data, q, v)
    pin.updateFramePlacements(model, data)

    cm = constraint_model
    cd = constraint_datas[0]
    oM1 = data.oMi[cm.joint1_id] * cm.joint1_placement
    oM2 = data.oMi[cm.joint2_id] * cm.joint2_placement if cm.joint2_id > 0 else cm.joint2_placement
    pos_error = oM1.translation - oM2.translation
    vel_error = Jc @ v
    gamma = cm.corrector.Kd * vel_error + cm.corrector.Kp * pos_error

    rhs = np.zeros(nc + model.nv, dtype=float)
    rhs[:nc] = -gamma
    rhs[nc:] = M @ a + h

    sol = kkt.solve(rhs)
    lambda_ = sol[:nc]
    tau = M @ a + h - Jc.T @ lambda_
    return tau


def _save_results_csv(out_csv, time, tau_serial, tau_real, ee_serial, ee_real, q3act):
    arr = np.column_stack([time, q3act, tau_serial, tau_real, ee_serial, ee_real])
    header = ",".join(
        [
            "time",
            "q3act",
            "tau1_serial",
            "tau2_serial",
            "tau3_serial",
            "tau4_serial",
            "tau1_para",
            "tau2_para",
            "tau3act_para",
            "tau4_para",
            "Xee_serial",
            "Yee_serial",
            "Zee_serial",
            "Xee_para",
            "Yee_para",
            "Zee_para",
        ]
    )
    np.savetxt(out_csv, arr, delimiter=",", header=header, comments="")


def compare_serial_vs_parallelogram(
    csv_file=None,
    mat_file="cr4_params.mat",
    motor_mat_file="cr4_motor_params.mat",
    sim_params_csv=None,
    prefer_sim_params_csv=True,
    robodimm_path=ROBODIMM_PATH_DEFAULT,
    save_csv=True,
    save_plot=True,
    show_plot=True,
    torque_method="hybrid_actuation",
    solver_method="native_pinocchio",
    inertia_mapping="serial_joint_transfer",
    use_motor_mat=True,
    m_mot=0.0,
    I_mot=0.0,
    gr_mot=0.0,
    b_visc_axis4=None,
    payload=None,
    parallel_motor_layout="concentric_j2_j3act",
):
    if csv_file is None:
        csv_file = ne._default_simscape_results_csv()

    sim_csv = ne._default_sim_params_csv() if sim_params_csv is None else sim_params_csv
    robot_scale, robot_scale_source = ne.resolve_robot_scale(sim_csv, default=1.0)

    cols = _load_csv_columns(csv_file)
    base_params = load_params_from_mat(mat_file=mat_file, n_links=5)
    params = ne.scale_rigid_body_params(base_params, robot_scale)

    # Caso dinamico de motores/damping/payload
    if use_motor_mat:
        case = ne.load_motor_payload_case(
            motor_mat_file=motor_mat_file,
            sim_params_csv=sim_csv,
            prefer_sim_params_csv=prefer_sim_params_csv,
        )
    else:
        # fallback legacy con escalares
        if b_visc_axis4 is None:
            b_visc_axis4 = np.zeros(4, dtype=float)
        if payload is None:
            payload = {"mass": 0.0, "com_from_tcp": np.zeros(3), "inertia": np.zeros((3, 3))}

        i_ref_scalar = (gr_mot**2) * I_mot
        case = {
            "source": "scalar_args",
            "study_profile": "scalar_args",
            "stator_link_masses": np.array([m_mot, m_mot, 0.0, m_mot, 0.0], dtype=float),
            "I_ref_joint": np.array([i_ref_scalar, i_ref_scalar, i_ref_scalar, 0.0, i_ref_scalar], dtype=float),
            "b_visc_joint": np.array([b_visc_axis4[0], b_visc_axis4[1], b_visc_axis4[2], 0.0, b_visc_axis4[3]], dtype=float),
            "tau_coulomb_joint": np.zeros(5, dtype=float),
            "payload": payload,
            "motor_ids": [],
            "gear_ids": [],
        }

    # Mapas por actuador [J1, J2, J3act, J4]
    stator_link = np.asarray(case["stator_link_masses"], dtype=float).reshape(5)
    m_axis4 = np.array([0.0, stator_link[0], stator_link[1], stator_link[3]], dtype=float)
    iref5 = np.asarray(case["I_ref_joint"], dtype=float).reshape(5)
    iref_axis4 = np.array([iref5[0], iref5[1], iref5[2], iref5[4]], dtype=float)
    b5 = np.asarray(case["b_visc_joint"], dtype=float).reshape(5)
    b_axis4 = np.array([b5[0], b5[1], b5[2], b5[4]], dtype=float)
    c5 = np.asarray(case["tau_coulomb_joint"], dtype=float).reshape(5)
    c_axis4 = np.array([c5[0], c5[1], c5[2], c5[4]], dtype=float)
    payload_case = case["payload"]

    serial_model, serial_ee_frame_id, serial_joint_ids = build_serial_model(params)
    serial_data = serial_model.createData()

    real_model, constraint_model, idx_v = build_real_parallelogram_model(
        params, robodimm_path=robodimm_path, robot_scale=robot_scale
    )
    real_data = real_model.createData()
    real_tool0_id = real_model.getFrameId("tool0")

    if inertia_mapping not in {"serial_joint_transfer", "direct_link_copy"}:
        raise ValueError("inertia_mapping debe ser 'serial_joint_transfer' o 'direct_link_copy'")
    if inertia_mapping == "serial_joint_transfer":
        apply_inertia_mapping_serial_joint_transfer(
            serial_model,
            serial_data,
            serial_joint_ids,
            real_model,
            real_data,
            constraint_model,
        )
    else:
        apply_inertia_mapping_direct_link_copy(real_model, params)

    # Motores: masa de estator en lado BASE de cada joint.
    apply_serial_motor_stator_masses(serial_model, serial_joint_ids, m_axis4)
    apply_parallel_motor_stator_masses(real_model, m_axis4, layout=parallel_motor_layout)
    # Payload en TCP.
    apply_payload_to_serial_model(serial_model, serial_joint_ids, params, payload_case)
    apply_payload_to_real_model(real_model, payload_case, tool_frame_name="tool0")

    if torque_method not in {"virtual_work", "legacy_motor_map", "hybrid_actuation"}:
        raise ValueError("torque_method debe ser 'virtual_work', 'legacy_motor_map' o 'hybrid_actuation'")
    if solver_method not in {"native_pinocchio", "robodimm_helper"}:
        raise ValueError("solver_method debe ser 'native_pinocchio' o 'robodimm_helper'")

    compute_motor_inverse_dynamics = None
    compute_constrained_inverse_dynamics = None
    if torque_method == "legacy_motor_map" or torque_method == "hybrid_actuation" or solver_method == "robodimm_helper":
        try:
            from robot_core.dynamics.constrained import compute_motor_inverse_dynamics as _cmid
            from robot_core.dynamics.constrained import compute_constrained_inverse_dynamics as _ccid
        except Exception as exc:
            raise RuntimeError("No se pudo importar dinamica constrained desde robodimm.") from exc
        compute_motor_inverse_dynamics = _cmid
        compute_constrained_inverse_dynamics = _ccid

    B = build_actuator_projection_matrix(idx_v, real_model.nv)

    q_theta = np.column_stack([cols["q1"], cols["q2"], cols["q3"], cols["q4"], cols["q5"]])
    qd_theta = np.column_stack([cols["qd1"], cols["qd2"], cols["qd3"], cols["qd4"], cols["qd5"]])
    qdd_theta = np.column_stack([cols["qdd1"], cols["qdd2"], cols["qdd3"], cols["qdd4"], cols["qdd5"]])

    # Este modelo paralelo asume q_aux = -(q2 + q3_rel). Si el CSV trae q4
    # independiente (serial 5DOF puro), la FK/torques no pueden coincidir.
    passive_err = q_theta[:, 3] + q_theta[:, 1] + q_theta[:, 2] - np.pi
    passive_max_abs = float(np.max(np.abs(passive_err)))
    if passive_max_abs > 1e-3:
        print(
            "[WARN] Trayectoria incompatible con la restriccion del paralelogramo: "
            "q4 + q2 + q3 - pi != 0. "
            f"max|error|={passive_max_abs:.6f} rad. "
            "Se esperan discrepancias en EE y torques para el modelo closed-loop."
        )

    n = q_theta.shape[0]
    tau_serial_4 = np.zeros((n, 4))
    tau_para_4 = np.zeros((n, 4))
    ee_serial = np.zeros((n, 3))
    ee_para = np.zeros((n, 3))
    q3act = np.zeros(n)

    for i in range(n):
        q_s = q_theta[i, :]
        qd_s = qd_theta[i, :]
        qdd_s = qdd_theta[i, :]
        sign_serial = np.where(np.abs(qd_s) > 1e-9, np.sign(qd_s), 0.0)

        tau5 = pin.rnea(serial_model, serial_data, q_s, qd_s, qdd_s)
        # Rotor en serial: J1, J2, J3, J4(wrist) -> q1,q2,q3,q5
        tau5 = tau5 + np.array(
            [
                iref_axis4[0] * qdd_s[0],
                iref_axis4[1] * qdd_s[1],
                iref_axis4[2] * qdd_s[2],
                0.0,
                iref_axis4[3] * qdd_s[4],
            ],
            dtype=float,
        )
        # Damping viscoso serial
        tau5 = tau5 + np.array(
            [
                b_axis4[0] * qd_s[0],
                b_axis4[1] * qd_s[1],
                b_axis4[2] * qd_s[2],
                0.0,
                b_axis4[3] * qd_s[4],
            ],
            dtype=float,
        )
        tau5 = tau5 + np.array(
            [
                c_axis4[0] * sign_serial[0],
                c_axis4[1] * sign_serial[1],
                c_axis4[2] * sign_serial[2],
                0.0,
                c_axis4[3] * sign_serial[4],
            ],
            dtype=float,
        )
        tau_serial_4[i, :] = serial_tau5_to_actuator_tau4(tau5)

        pin.forwardKinematics(serial_model, serial_data, q_s)
        pin.updateFramePlacements(serial_model, serial_data)
        ee_serial[i, :] = serial_data.oMf[serial_ee_frame_id].translation

        q5_user = theta_to_user5(q_s)
        qd5_user = qd_s.copy()
        qdd5_user = qdd_s.copy()
        q_real, v_real, a_real = build_real_state_from_user5(
            q5_user, qd5_user, qdd5_user, idx_v, real_model.nq, real_model.nv
        )

        if torque_method == "virtual_work" or torque_method == "hybrid_actuation":
            if solver_method == "native_pinocchio":
                tau_general = compute_constrained_inverse_dynamics_native(
                    real_model, real_data, constraint_model, q_real, v_real, a_real
                )
            else:
                tau_general = compute_constrained_inverse_dynamics(
                    real_model, real_data, constraint_model, q_real, v_real, a_real
                )
            tau_vw = B.T @ tau_general
            if torque_method == "virtual_work":
                # Trabajo virtual completo: tau_act = B^T * tau_general
                tau_para_4[i, :] = tau_vw
            else:
                # Hibrido: J2 desde trabajo virtual y J3act desde carga de codo pasivo J3.
                tau_para_4[i, :] = np.array(
                    [
                        tau_vw[0],
                        tau_vw[1],
                        tau_general[idx_v["J3"]],
                        tau_vw[3],
                    ],
                    dtype=float,
                )
        else:
            tau_motor = compute_motor_inverse_dynamics(
                real_model, real_data, constraint_model, q_real, v_real, a_real, return_analysis=False
            )
            tau_para_4[i, :] = np.array(
                [
                    tau_motor[idx_v["J1"]],
                    tau_motor[idx_v["J2"]],
                    tau_motor[idx_v["J3real"]],
                    J4_ACT_SIGN * tau_motor[idx_v["J4"]],
                ],
                dtype=float,
            )

        # Rotor en paralelo para actuadores [J1, J2, J3act, J4]
        jdd3act = qdd5_user[1] + qdd5_user[2]
        qd3act = qd5_user[1] + qd5_user[2]
        sign_qd3act = float(np.sign(qd3act)) if abs(qd3act) > 1e-9 else 0.0
        sign_para = np.array(
            [
                np.sign(qd5_user[0]) if abs(qd5_user[0]) > 1e-9 else 0.0,
                np.sign(qd5_user[1]) if abs(qd5_user[1]) > 1e-9 else 0.0,
                sign_qd3act,
                np.sign(qd5_user[4]) if abs(qd5_user[4]) > 1e-9 else 0.0,
            ],
            dtype=float,
        )
        tau_para_4[i, :] = tau_para_4[i, :] + np.array(
            [
                iref_axis4[0] * qdd5_user[0],
                iref_axis4[1] * qdd5_user[1],
                iref_axis4[2] * jdd3act,
                iref_axis4[3] * qdd5_user[4],
            ],
            dtype=float,
        )
        # Damping viscoso paralelo
        tau_para_4[i, :] = tau_para_4[i, :] + np.array(
            [
                b_axis4[0] * qd5_user[0],
                b_axis4[1] * qd5_user[1],
                b_axis4[2] * qd3act,
                b_axis4[3] * qd5_user[4],
            ],
            dtype=float,
        )
        tau_para_4[i, :] = tau_para_4[i, :] + (c_axis4 * sign_para)

        pin.forwardKinematics(real_model, real_data, q_real)
        pin.updateFramePlacements(real_model, real_data)
        ee_para[i, :] = real_data.oMf[real_tool0_id].translation

        q3act[i] = q5_user[1] + q5_user[2]

    print("Comparacion Pinocchio: Serial puro vs Paralelogramo (q3act)")
    print(f"  robot_scale: {robot_scale:.6f} (origen: {robot_scale_source})")
    print(f"  dh.a escalado [m]: {np.round(params['dh_a'], 6).tolist()}")
    print(f"  dh.d escalado [m]: {np.round(params['dh_d'], 6).tolist()}")
    print(f"  Mapeo de inercias: {inertia_mapping}")
    print(f"  Fuente caso motores: {case['source']}")
    print(f"  Perfil estudio: {case['study_profile']}")
    if case["motor_ids"]:
        print(f"  Motor IDs: {case['motor_ids']}")
    if case["gear_ids"]:
        print(f"  Gear IDs : {case['gear_ids']}")
    print(f"  m_axis4 [kg] [J1,J2,J3act,J4] = {np.round(m_axis4, 6).tolist()}")
    print(f"  I_ref_axis4 [kg*m^2] = {np.round(iref_axis4, 6).tolist()}")
    print(f"  b_visc_axis4 [N*m*s/rad] = {np.round(b_axis4, 6).tolist()}")
    print(f"  tau_coulomb_axis4 [N*m] = {np.round(c_axis4, 6).tolist()}")
    print(
        "  payload: "
        f"m={float(payload_case.get('mass', 0.0)):.3f} kg, "
        f"com_tcp={np.round(payload_case.get('com_from_tcp', np.zeros(3)), 4).tolist()}"
    )
    print(f"  Layout motores paralelo: {parallel_motor_layout}")
    print(f"  Metodo de torque paralelogramo: {torque_method}")
    if torque_method in {"virtual_work", "hybrid_actuation"}:
        print(f"  Solver cadena cerrada: {solver_method}")
    print("  Torques comparados en actuadores: [J1, J2, J3act, J4]")
    print("  Nota: comparacion justa -> tau3act(paralelo) vs tau3(serial)")
    print("  Formato: corr | RMSE | bias | fit_ref=(a*PARA+b)")

    tau_names = ["tau1", "tau2", "tau3(serial vs q3act)", "tau4"]
    ee_names = ["Xee", "Yee", "Zee"]

    for i, name in enumerate(tau_names):
        m = _signal_metrics(tau_serial_4[:, i], tau_para_4[:, i])
        print(
            f"  {name:7s} "
            f"{m['corr']:.4f} | {m['rmse']:.6f} | {m['bias']:.6f} | "
            f"a={m['slope']:.4f}, b={m['intercept']:.4f}"
        )

    print("\n  FK (Serial vs Paralelogramo):")
    for i, name in enumerate(ee_names):
        m = _signal_metrics(ee_serial[:, i], ee_para[:, i])
        print(
            f"  {name:4s} "
            f"{m['corr']:.4f} | {m['rmse']:.6e} | {m['bias']:.6e} | "
            f"a={m['slope']:.4f}, b={m['intercept']:.4e}"
        )

    motors_active = (
        np.any(np.abs(m_axis4) > 0.0)
        or np.any(np.abs(iref_axis4) > 0.0)
        or np.any(np.abs(b_axis4) > 0.0)
        or np.any(np.abs(c_axis4) > 0.0)
        or float(payload_case.get("mass", 0.0)) > 0.0
    )

    if save_csv:
        out_csv_name = (
            f"cr4_pinocchio_serial_vs_parallelogram_motors_{parallel_motor_layout}_results.csv"
            if motors_active
            else "cr4_pinocchio_serial_vs_parallelogram_results.csv"
        )
        out_csv = os.path.join(_results_dir(), out_csv_name)
        _save_results_csv(out_csv, cols["time"], tau_serial_4, tau_para_4, ee_serial, ee_para, q3act)
        print(f"\n  CSV guardado en: {out_csv}")

    if plt is not None:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        ax = axes.ravel()

        for i, name in enumerate(tau_names):
            m = _signal_metrics(tau_serial_4[:, i], tau_para_4[:, i])
            ax[i].plot(cols["time"], tau_serial_4[:, i], "-", linewidth=1.8, label="Serial puro")
            ax[i].plot(cols["time"], tau_para_4[:, i], "--", linewidth=1.8, label="Paralelogramo q3act")
            ax[i].set_title(f"{name} | corr={m['corr']:.3f}, RMSE={m['rmse']:.3f}")
            ax[i].set_xlabel("Tiempo [s]")
            ax[i].set_ylabel("Torque [Nm]")
            ax[i].grid(True, alpha=0.3)
            ax[i].legend()

        title = "CR4 Pinocchio: Serial puro vs Paralelogramo (q3act)"
        if motors_active:
            title += " + motores"
        fig.suptitle(title, fontsize=12)
        fig.tight_layout()

        if save_plot:
            out_png_name = (
                f"cr4_pinocchio_serial_vs_parallelogram_motors_{parallel_motor_layout}_comparacion.png"
                if motors_active
                else "cr4_pinocchio_serial_vs_parallelogram_comparacion.png"
            )
            out_png = os.path.join(_results_dir(), out_png_name)
            fig.savefig(out_png, dpi=150, bbox_inches="tight")
            print(f"  Figura guardada en: {out_png}")

        if show_plot:
            plt.show()
        else:
            plt.close(fig)
    else:
        print("  Matplotlib no disponible: se omite figura.")

    return {
        "tau_serial": tau_serial_4,
        "tau_parallelogram": tau_para_4,
        "ee_serial": ee_serial,
        "ee_parallelogram": ee_para,
    }


if __name__ == "__main__":
    print("CR4 Pinocchio - Serial vs Paralelogramo")
    print("=" * 60)
    compare_serial_vs_parallelogram(
        show_plot=False,
        save_plot=True,
        save_csv=True,
        use_motor_mat=True,
        parallel_motor_layout="concentric_j2_j3act",
    )
