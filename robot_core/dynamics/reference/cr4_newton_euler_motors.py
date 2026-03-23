"""
CR4 Newton-Euler serial 5DOF con motores + reductora + damping + payload.

Comparacion Simscape (serial) vs Newton-Euler (serial) con:
- masas de estator por articulacion
- inercia reflejada por articulacion
- damping viscoso articular
- payload en TCP

Prioridad de fuentes del caso dinamico:
1) CSV fuente de simulacion (`results/*_sim_params.csv`), si se habilita.
2) cr4_motor_params.mat.
3) defaults legacy.
"""

import csv
import os
import numpy as np
from copy import deepcopy

try:
    import scipy.io as sio
except Exception as exc:
    raise RuntimeError("Se requiere scipy para cargar .mat") from exc

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# Defaults legacy (se usan como fallback si no hay cr4_motor_params.mat)
m_mot = 1.0
I_mot = 5.0e-7
gr_mot = 100.0
motor_rotor_joint_mask = np.array([1, 1, 1, 0, 1], dtype=float)       # q1..q5
motor_stator_mass_link_mask = np.array([1, 1, 0, 1, 0], dtype=float)  # Link1..Link5
G = 9.81


def _project_dir():
    return os.path.dirname(os.path.abspath(__file__))


def _results_dir():
    p = os.path.join(_project_dir(), "results")
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


def _as_array3(x):
    a = np.array(x, dtype=float).reshape(-1)
    if a.size != 3:
        raise ValueError(f"Se esperaban 3 componentes, recibido {a}")
    return a


def _as_1d(x, n=None):
    a = np.array(x, dtype=float).reshape(-1)
    if n is not None and a.size != n:
        raise ValueError(f"Tamano invalido: esperado {n}, recibido {a.size}")
    return a


def _map_links_legacy6_to_serial5_dh(links, dh_a, dh_d):
    """
    Convierte el esquema legacy (6 links) a serial5 para el CR4 actual:
    [Base, UpperArm, Forearm, WristAux, TCP] en frame DH.
    """
    arr = np.asarray(links, dtype=object).reshape(-1)
    if arr.size < 6:
        return arr

    out = arr[[0, 2, 3, 4, 5]].copy()

    # Base: eje cilindrico en +Y (en vez de +Z).
    try:
        c0 = _as_array3(out[0].com)
        I0 = np.array(out[0].inertia, dtype=float).reshape(3, 3)
        out[0].com = np.array([0.0, 0.5 * float(dh_d[0]), 0.0], dtype=float)
        d0 = np.diag(I0)
        # Solo swap si venia legacy con COM dominante en Z.
        if abs(c0[2]) > abs(c0[1]) + 1e-12:
            out[0].inertia = np.diag([d0[0], d0[2], d0[1]])
    except Exception:
        pass

    # Centros geometricos en el frame local de los STEP Part2..Part5.
    try:
        y_offset = -0.025
        x_tcp_offset = -0.025
        out[1].com = np.array([-0.5 * float(dh_a[1]), y_offset, 0.0], dtype=float)
        out[2].com = np.array([-0.5 * float(dh_a[2]), y_offset, 0.0], dtype=float)
        out[3].com = np.array([-0.5 * float(dh_a[3]), y_offset, 0.0], dtype=float)
        out[4].com = np.array([x_tcp_offset, 0.0, -0.5 * float(dh_d[4])], dtype=float)
    except Exception:
        pass

    return out


def _default_simscape_results_csv():
    return os.path.join(_results_dir(), "cr4_simscape_results.csv")


def _point_mass_inertia_about_new_com(m, d):
    d = np.asarray(d, dtype=float).reshape(3)
    return m * ((np.dot(d, d) * np.eye(3)) - np.outer(d, d))


def _combine_two_bodies(m1, c1, I1, m2, c2, I2):
    c1 = np.asarray(c1, dtype=float).reshape(3)
    I1 = np.asarray(I1, dtype=float).reshape(3, 3)
    c2 = np.asarray(c2, dtype=float).reshape(3)
    I2 = np.asarray(I2, dtype=float).reshape(3, 3)
    m = float(m1) + float(m2)
    if m <= 0.0:
        return 0.0, np.zeros(3), np.zeros((3, 3))
    c = (m1 * c1 + m2 * c2) / m
    I = (
        I1
        + _point_mass_inertia_about_new_com(m1, c1 - c)
        + I2
        + _point_mass_inertia_about_new_com(m2, c2 - c)
    )
    return m, c, I


def _combine_body_with_point_mass(m1, c1, I1, m2, c2=None):
    if c2 is None:
        c2 = np.zeros(3, dtype=float)
    return _combine_two_bodies(m1, c1, I1, m2, c2, np.zeros((3, 3)))


def _rot_x(alpha):
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    return np.array([[1.0, 0.0, 0.0], [0.0, ca, -sa], [0.0, sa, ca]], dtype=float)


def load_params_from_mat(mat_file="cr4_params.mat", n_links=5):
    m = sio.loadmat(mat_file, squeeze_me=True, struct_as_record=False)
    dh = m["dh"]
    links = m["links"]
    dh_a = np.array(dh.a, dtype=float).reshape(-1)[:n_links]
    dh_alpha = np.array(dh.alpha, dtype=float).reshape(-1)[:n_links]
    dh_d = np.array(dh.d, dtype=float).reshape(-1)[:n_links]
    links = _map_links_legacy6_to_serial5_dh(links, dh_a, dh_d)

    masses, coms, inertias = [], [], []
    for i in range(n_links):
        li = links[i]
        masses.append(float(li.mass))
        coms.append(_as_array3(li.com))
        inertias.append(np.array(li.inertia, dtype=float))

    return {
        "dh_a": dh_a,
        "dh_alpha": dh_alpha,
        "dh_d": dh_d,
        "masses": masses,
        "coms": coms,
        "inertias": inertias,
    }


def _default_sim_params_csv():
    return os.path.join(_results_dir(), "cr4_serial5_source_sim_params.csv")


def resolve_robot_scale(sim_params_csv=None, default=1.0):
    sim_csv = _default_sim_params_csv() if sim_params_csv is None else sim_params_csv
    csv_params = _load_sim_params_kv_csv(sim_csv)
    scale = _to_float_or_default(csv_params.get("robot_scale", default), default)
    if scale <= 0.0:
        raise ValueError(f"robot_scale invalido ({scale}) en {sim_csv}. Debe ser > 0.")
    source = sim_csv if csv_params else "default(1.0)"
    return float(scale), source


def scale_rigid_body_params(params, robot_scale):
    s = float(robot_scale)
    if s <= 0.0:
        raise ValueError(f"robot_scale debe ser > 0, recibido {s}")

    mass_scale = s**3
    inertia_scale = s**5

    masses_in = np.asarray(params["masses"], dtype=float).reshape(5)
    dh_a_in = np.asarray(params["dh_a"], dtype=float).reshape(5)
    dh_d_in = np.asarray(params["dh_d"], dtype=float).reshape(5)

    coms_out = [np.asarray(c, dtype=float).reshape(3) * s for c in params["coms"]]
    inertias_out = [np.asarray(I, dtype=float).reshape(3, 3) * inertia_scale for I in params["inertias"]]

    return {
        "dh_a": dh_a_in * s,
        "dh_alpha": np.asarray(params["dh_alpha"], dtype=float).reshape(5),
        "dh_d": dh_d_in * s,
        "masses": (masses_in * mass_scale).tolist(),
        "coms": coms_out,
        "inertias": inertias_out,
    }


def _to_float_or_default(x, default=0.0):
    try:
        v = float(x)
        if np.isnan(v):
            return float(default)
        return v
    except Exception:
        return float(default)


def _load_sim_params_kv_csv(csv_path):
    if (csv_path is None) or (not os.path.isfile(csv_path)):
        return {}

    def _norm_token(x):
        s = str(x).replace("\ufeff", "").strip().strip('"').strip("'")
        return s

    out = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as fh:
        rd = csv.DictReader(fh)
        fields = [_norm_token(f).lower() for f in (rd.fieldnames or [])]
        if ("key" not in fields) or ("value" not in fields):
            return {}

        for row in rd:
            k = _norm_token(row.get("key", ""))
            if (k == "") or (k.lower() == "nan"):
                continue
            out[k] = _norm_token(row.get("value", ""))
    return out


def _build_case_from_sim_params_kv(params, source_label):
    def _parse_id_tokens(raw):
        s = str(raw).strip()
        if s == "":
            return []
        for sep in [";", "|"]:
            s = s.replace(sep, ",")
        return [tok.strip() for tok in s.split(",") if tok.strip() != ""]

    def _collect_q_ids(prefix):
        out = []
        for j in range(1, 6):
            key = f"{prefix}_q{j}"
            v = str(params.get(key, "")).strip()
            if v != "":
                out.append(f"q{j}:{v}")
        if out:
            return out

        # Fallback opcional: lista en un unico campo (ej. motor_ids="id1,id2,...")
        return _parse_id_tokens(params.get(f"{prefix}s", params.get(f"{prefix}_ids", "")))

    # Base: todo a cero salvo lo que venga en CSV.
    case = {
        "source": source_label,
        "study_profile": str(params.get("profile_name", "sim_params_csv")),
        "iref_model_mode": str(params.get("iref_model_mode", "q5_physical")),
        "stator_link_masses": np.zeros(5, dtype=float),
        "I_ref_joint": np.zeros(5, dtype=float),
        "b_visc_joint": np.zeros(5, dtype=float),
        "tau_coulomb_joint": np.zeros(5, dtype=float),
        "payload": {"mass": 0.0, "com_from_tcp": np.zeros(3), "inertia": np.zeros((3, 3))},
        "motor_ids": [],
        "gear_ids": [],
    }
    case["motor_ids"] = _collect_q_ids("motor_id")
    case["gear_ids"] = _collect_q_ids("gear_id")

    # Inercias reflejadas por joint serial q1..q5 [kg*m^2].
    case["I_ref_joint"] = np.array(
        [
            _to_float_or_default(params.get("rotor_inertia_reflected_q1", params.get("I_ref_J1", 0.0))),
            _to_float_or_default(params.get("rotor_inertia_reflected_q2", params.get("I_ref_J2", 0.0))),
            _to_float_or_default(params.get("rotor_inertia_reflected_q3", params.get("I_ref_J3act", 0.0))),
            _to_float_or_default(params.get("rotor_inertia_reflected_q4", 0.0)),
            _to_float_or_default(params.get("rotor_inertia_reflected_q5", params.get("I_ref_J4TCP", 0.0))),
        ],
        dtype=float,
    )

    # Friccion viscosa en rad/s [N*m*s/rad].
    b1 = _to_float_or_default(params.get("friction_viscous_q1", 0.0))
    b2 = _to_float_or_default(params.get("friction_viscous_q2", 0.0))
    b3 = _to_float_or_default(params.get("friction_viscous_q3", 0.0))
    b4 = _to_float_or_default(params.get("friction_viscous_q4", 0.0))
    b5 = _to_float_or_default(params.get("friction_viscous_q5", 0.0))

    # Fallback si solo vienen coeficientes en deg/s.
    if np.allclose([b1, b2, b3, b4, b5], 0.0):
        b1 = _to_float_or_default(params.get("b_visc_J1_deg", 0.0)) * (180.0 / np.pi)
        b2 = _to_float_or_default(params.get("b_visc_J2_deg", 0.0)) * (180.0 / np.pi)
        b3 = _to_float_or_default(params.get("b_visc_J3act_deg", 0.0)) * (180.0 / np.pi)
        b4 = _to_float_or_default(params.get("b_visc_J4_deg", 0.0)) * (180.0 / np.pi)
        b5 = _to_float_or_default(params.get("b_visc_J4TCP_deg", 0.0)) * (180.0 / np.pi)

    case["b_visc_joint"] = np.array([b1, b2, b3, b4, b5], dtype=float)

    case["tau_coulomb_joint"] = np.array(
        [
            _to_float_or_default(params.get("friction_coulomb_q1", 0.0)),
            _to_float_or_default(params.get("friction_coulomb_q2", 0.0)),
            _to_float_or_default(params.get("friction_coulomb_q3", 0.0)),
            _to_float_or_default(params.get("friction_coulomb_q4", 0.0)),
            _to_float_or_default(params.get("friction_coulomb_q5", 0.0)),
        ],
        dtype=float,
    )

    # Masas de estator en links [Link1..Link5] (opcionales).
    # Convencion antigua axis4: m_mot_J2->Link1, m_mot_J3act->Link2, m_mot_J4TCP->Link4.
    stator = np.zeros(5, dtype=float)
    if "stator_link_mass_l1_kg" in params:
        stator = np.array(
            [
                _to_float_or_default(params.get("stator_link_mass_l1_kg", 0.0)),
                _to_float_or_default(params.get("stator_link_mass_l2_kg", 0.0)),
                _to_float_or_default(params.get("stator_link_mass_l3_kg", 0.0)),
                _to_float_or_default(params.get("stator_link_mass_l4_kg", 0.0)),
                _to_float_or_default(params.get("stator_link_mass_l5_kg", 0.0)),
            ],
            dtype=float,
        )
    elif any(f"motor_mass_q{j}_kg" in params for j in range(1, 6)):
        # Mapeo serial 5GDL:
        # - q1 en base fija (no entra en cadena movil)
        # - q2..q5 en lado BASE de joints 2..5 -> Link1..Link4.
        motor_mass_q = np.array(
            [
                _to_float_or_default(params.get("motor_mass_q1_kg", 0.0)),
                _to_float_or_default(params.get("motor_mass_q2_kg", 0.0)),
                _to_float_or_default(params.get("motor_mass_q3_kg", 0.0)),
                _to_float_or_default(params.get("motor_mass_q4_kg", 0.0)),
                _to_float_or_default(params.get("motor_mass_q5_kg", 0.0)),
            ],
            dtype=float,
        )
        stator[0] = motor_mass_q[1]
        stator[1] = motor_mass_q[2]
        stator[2] = motor_mass_q[3]
        stator[3] = motor_mass_q[4]
    else:
        stator[0] = _to_float_or_default(params.get("m_mot_J2", 0.0))
        stator[1] = _to_float_or_default(params.get("m_mot_J3act", 0.0))
        stator[3] = _to_float_or_default(params.get("m_mot_J4TCP", 0.0))
    case["stator_link_masses"] = stator

    payload_mass = _to_float_or_default(params.get("payload_mass_kg", 0.0))
    payload_com = np.array(
        [
            _to_float_or_default(params.get("payload_com_x_m", 0.0)),
            _to_float_or_default(params.get("payload_com_y_m", 0.0)),
            _to_float_or_default(params.get("payload_com_z_m", 0.0)),
        ],
        dtype=float,
    )
    payload_inertia = np.diag(
        [
            _to_float_or_default(params.get("payload_Ixx_kgm2", 0.0)),
            _to_float_or_default(params.get("payload_Iyy_kgm2", 0.0)),
            _to_float_or_default(params.get("payload_Izz_kgm2", 0.0)),
        ]
    )
    case["payload"] = {"mass": payload_mass, "com_from_tcp": payload_com, "inertia": payload_inertia}

    return case


def load_motor_payload_case(
    motor_mat_file="cr4_motor_params.mat",
    sim_params_csv=None,
    prefer_sim_params_csv=False,
):
    """
    Devuelve un caso dinamico unificado con:
    - stator_link_masses (5)
    - I_ref_joint (5)
    - b_visc_joint (5)
    - tau_coulomb_joint (5)
    - payload dict {mass, com_from_tcp, inertia}
    """
    case = {
        "source": "legacy_defaults",
        "study_profile": "legacy_defaults",
        "iref_model_mode": "q5_physical",
        "stator_link_masses": motor_stator_mass_link_mask * m_mot,
        "I_ref_joint": motor_rotor_joint_mask * ((gr_mot**2) * I_mot),
        "b_visc_joint": np.zeros(5, dtype=float),
        "tau_coulomb_joint": np.zeros(5, dtype=float),
        "payload": {"mass": 0.0, "com_from_tcp": np.zeros(3), "inertia": np.zeros((3, 3))},
        "motor_ids": [],
        "gear_ids": [],
    }

    sim_csv = _default_sim_params_csv() if sim_params_csv is None else sim_params_csv

    # Si se pide, prioriza CSV de fuente unica para garantizar coherencia con Simscape.
    if prefer_sim_params_csv and os.path.isfile(sim_csv):
        csv_params = _load_sim_params_kv_csv(sim_csv)
        if csv_params:
            return _build_case_from_sim_params_kv(csv_params, sim_csv)

    if not os.path.isfile(motor_mat_file):
        # Fallback a CSV si existe (aunque no se pidio prioridad explicita).
        if os.path.isfile(sim_csv):
            csv_params = _load_sim_params_kv_csv(sim_csv)
            if csv_params:
                return _build_case_from_sim_params_kv(csv_params, sim_csv)
        return case

    m = sio.loadmat(motor_mat_file, squeeze_me=True, struct_as_record=False)
    motor = m.get("motor", None)
    payload = m.get("payload", None)
    if motor is None:
        return case

    case["source"] = motor_mat_file
    if hasattr(motor, "study_profile"):
        case["study_profile"] = str(motor.study_profile)

    if hasattr(motor, "motor_ids"):
        ids = np.array(motor.motor_ids, dtype=object).reshape(-1).tolist()
        case["motor_ids"] = [str(x) for x in ids]
    if hasattr(motor, "gear_ids"):
        ids = np.array(motor.gear_ids, dtype=object).reshape(-1).tolist()
        case["gear_ids"] = [str(x) for x in ids]

    if hasattr(motor, "m_mot_axis"):
        m_axis = _as_1d(motor.m_mot_axis, 4)  # [J1, J2, J3act, J4_TCP]
        # Lado BASE del joint en serial:
        # J2 -> Link1, J3 -> Link2, J4_TCP(q5) -> Link4, J1 en base fija (no movil)
        case["stator_link_masses"] = np.array([m_axis[1], m_axis[2], 0.0, m_axis[3], 0.0], dtype=float)

    if hasattr(motor, "I_ref_axis"):
        iref_axis = _as_1d(motor.I_ref_axis, 4)  # [J1, J2, J3act, J4_TCP]
        case["I_ref_joint"] = np.array([iref_axis[0], iref_axis[1], iref_axis[2], 0.0, iref_axis[3]], dtype=float)
    elif hasattr(motor, "I_mot_axis") and hasattr(motor, "gr_axis"):
        i_m = _as_1d(motor.I_mot_axis, 4)
        gr = _as_1d(motor.gr_axis, 4)
        iref_axis = (gr**2) * i_m
        case["I_ref_joint"] = np.array([iref_axis[0], iref_axis[1], iref_axis[2], 0.0, iref_axis[3]], dtype=float)

    if hasattr(motor, "b_visc_rad"):
        b_axis = _as_1d(motor.b_visc_rad, 4)
        case["b_visc_joint"] = np.array([b_axis[0], b_axis[1], b_axis[2], 0.0, b_axis[3]], dtype=float)

    if payload is not None and hasattr(payload, "mass"):
        p_mass = float(payload.mass)
        p_com = _as_array3(payload.com_from_tcp) if hasattr(payload, "com_from_tcp") else np.zeros(3)
        if hasattr(payload, "Ixx") and hasattr(payload, "Iyy") and hasattr(payload, "Izz"):
            p_I = np.diag([float(payload.Ixx), float(payload.Iyy), float(payload.Izz)])
        elif hasattr(payload, "I"):
            p_I = np.array(payload.I, dtype=float).reshape(3, 3)
        else:
            p_I = np.zeros((3, 3), dtype=float)
        case["payload"] = {"mass": p_mass, "com_from_tcp": p_com, "inertia": p_I}

    return case


def apply_motor_lumped_inertia(params, stator_link_masses=None, payload=None):
    masses = [float(x) for x in params["masses"]]
    coms = [np.array(c, dtype=float).reshape(3) for c in params["coms"]]
    inertias = [np.array(I, dtype=float).reshape(3, 3) for I in params["inertias"]]

    if stator_link_masses is None:
        stator_link_masses = motor_stator_mass_link_mask * m_mot
    stator_link_masses = _as_1d(stator_link_masses, 5)

    for i in range(5):
        m_add = float(stator_link_masses[i])
        if m_add <= 0.0:
            continue
        m_new, c_new, I_new = _combine_body_with_point_mass(masses[i], coms[i], inertias[i], m_add, np.zeros(3))
        masses[i], coms[i], inertias[i] = m_new, c_new, I_new

    if payload is not None:
        p_mass = float(payload.get("mass", 0.0))
        if p_mass > 0.0:
            p_com_tcp = _as_array3(payload.get("com_from_tcp", np.zeros(3)))
            p_I_tcp = np.array(payload.get("inertia", np.zeros((3, 3))), dtype=float).reshape(3, 3)

            # En CR4.slx, payload.com_from_tcp se aplica en el mismo nodo mecanico
            # donde se conecta DH-5/Link5 (frame TCP del ultimo eslabon).
            # Para mantener equivalencia con Simscape, no se suma d5 aqui.
            p_com_5 = p_com_tcp
            p_I_5 = p_I_tcp

            i = 4
            m_new, c_new, I_new = _combine_two_bodies(
                masses[i], coms[i], inertias[i], p_mass, p_com_5, p_I_5
            )
            masses[i], coms[i], inertias[i] = m_new, c_new, I_new

    return {
        "dh_a": np.array(params["dh_a"], dtype=float),
        "dh_alpha": np.array(params["dh_alpha"], dtype=float),
        "dh_d": np.array(params["dh_d"], dtype=float),
        "masses": masses,
        "coms": coms,
        "inertias": inertias,
    }


def apply_reflected_inertia_mode(params, i_ref_joint, mode):
    """
    Controla como modelar I_ref:
    - diag: torque adicional I_ref*qdd (legacy)
    - q5_physical: I_ref de q5 como inercia fisica en Link5 (equivalente a Inertia4 en CR4.slx)
    - q3_q5_physical: idem q5 y tambien q3 como inercia fisica en Link3 (Inertia5 en CR4.slx)
    """
    mode_norm = str(mode).strip().lower()
    if mode_norm not in {"diag", "q5_physical", "q3_q5_physical"}:
        raise ValueError(f"iref_model_mode invalido: {mode}. Use diag|q5_physical|q3_q5_physical")

    i_ref_joint = np.asarray(i_ref_joint, dtype=float).reshape(5)
    params_out = deepcopy(params)
    i_ref_diag = i_ref_joint.copy()

    def _add_to_link_izz(link_idx, value):
        if abs(value) <= 0.0:
            return
        I = np.array(params_out["inertias"][link_idx], dtype=float).reshape(3, 3)
        I[2, 2] += float(value)
        params_out["inertias"][link_idx] = I

    if mode_norm in {"q5_physical", "q3_q5_physical"}:
        # q5 -> Link5 (index 4)
        _add_to_link_izz(4, i_ref_joint[4])
        i_ref_diag[4] = 0.0

    if mode_norm == "q3_q5_physical":
        # q3 -> Link3 (index 2)
        _add_to_link_izz(2, i_ref_joint[2])
        i_ref_diag[2] = 0.0

    return params_out, i_ref_diag, mode_norm


def newton_euler_serial5(q, qd, qdd, params):
    dh_a = params["dh_a"]
    dh_alpha = params["dh_alpha"]
    dh_d = params["dh_d"]
    masses = params["masses"]
    coms = params["coms"]
    inertias = params["inertias"]

    q = np.asarray(q, dtype=float).reshape(-1)
    qd = np.asarray(qd, dtype=float).reshape(-1)
    qdd = np.asarray(qdd, dtype=float).reshape(-1)
    n = 5
    if q.size != n or qd.size != n or qdd.size != n:
        raise ValueError("q, qd, qdd deben tener 5 elementos")

    p, R = [], []
    for i in range(n):
        ct, st = np.cos(q[i]), np.sin(q[i])
        ca, sa = np.cos(dh_alpha[i]), np.sin(dh_alpha[i])
        R_i = np.array([[ct, -ca * st, sa * st], [st, ca * ct, -sa * ct], [0.0, sa, ca]], dtype=float)
        R.append(R_i)
        p.append(np.array([dh_a[i], dh_d[i] * np.sin(dh_alpha[i]), dh_d[i] * np.cos(dh_alpha[i])], dtype=float))

    z0 = np.array([0.0, 0.0, 1.0], dtype=float)
    w = [np.zeros(3)]
    dw = [np.zeros(3)]
    dv = [np.array([0.0, 0.0, G])]

    for i in range(n):
        w_i = R[i].T @ (w[i] + z0 * qd[i])
        dw_i = R[i].T @ (dw[i] + z0 * qdd[i] + np.cross(w[i], z0 * qd[i]))
        dv_i = np.cross(dw_i, p[i]) + np.cross(w_i, np.cross(w_i, p[i])) + R[i].T @ dv[i]
        w.append(w_i)
        dw.append(dw_i)
        dv.append(dv_i)

    a_com = []
    for i in range(n):
        a_i = np.cross(dw[i + 1], coms[i]) + np.cross(w[i + 1], np.cross(w[i + 1], coms[i])) + dv[i + 1]
        a_com.append(a_i)

    f = [np.zeros(3) for _ in range(n + 1)]
    n_moment = [np.zeros(3) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        f[i] = (R[i + 1] @ f[i + 1] + masses[i] * a_com[i]) if i < n - 1 else (masses[i] * a_com[i])
        N_i = inertias[i] @ dw[i + 1] + np.cross(w[i + 1], inertias[i] @ w[i + 1])
        if i < n - 1:
            rec = np.cross(R[i + 1].T @ p[i], f[i + 1])
            n_moment[i] = R[i + 1] @ (n_moment[i + 1] + rec) + np.cross(p[i] + coms[i], masses[i] * a_com[i]) + N_i
        else:
            n_moment[i] = np.cross(p[i] + coms[i], masses[i] * a_com[i]) + N_i

    tau = np.zeros(n)
    for i in range(n):
        tau[i] = n_moment[i] @ (R[i].T @ z0)
    return tau


def forward_kinematics_ee(q, params):
    dh_a = params["dh_a"]
    dh_alpha = params["dh_alpha"]
    dh_d = params["dh_d"]
    q = np.asarray(q, dtype=float).reshape(-1)
    if q.size != 5:
        raise ValueError("q debe tener 5 elementos")
    T = np.eye(4)
    for i in range(5):
        ct, st = np.cos(q[i]), np.sin(q[i])
        ca, sa = np.cos(dh_alpha[i]), np.sin(dh_alpha[i])
        A = np.array(
            [[ct, -ca * st, sa * st, dh_a[i] * ct], [st, ca * ct, -sa * ct, dh_a[i] * st], [0.0, sa, ca, dh_d[i]], [0.0, 0.0, 0.0, 1.0]],
            dtype=float,
        )
        T = T @ A
    return T[:3, 3]


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
        "tau1",
        "tau2",
        "tau3",
        "tau4",
        "tau5",
        "Xee",
        "Yee",
        "Zee",
    ]
    missing = [c for c in required if c not in names]
    if missing:
        raise ValueError(f"Faltan columnas en CSV: {missing}")
    out = {"time": data["time"] if "time" in names else np.arange(data.shape[0], dtype=float)}
    for col in required:
        out[col] = data[col]
    return out


def _save_results_csv(out_csv, time, tau_ne, tau_sc, ee_ne, ee_sc):
    arr = np.column_stack([time, tau_ne, tau_sc, ee_ne, ee_sc])
    header = ",".join(
        [
            "time",
            "tau1_ne",
            "tau2_ne",
            "tau3_ne",
            "tau4_ne",
            "tau5_ne",
            "tau1_sc",
            "tau2_sc",
            "tau3_sc",
            "tau4_sc",
            "tau5_sc",
            "Xee_ne",
            "Yee_ne",
            "Zee_ne",
            "Xee_sc",
            "Yee_sc",
            "Zee_sc",
        ]
    )
    np.savetxt(out_csv, arr, delimiter=",", header=header, comments="")


def validate_serial_with_motors(
    csv_file=None,
    mat_file="cr4_params.mat",
    motor_mat_file="cr4_motor_params.mat",
    sim_params_csv=None,
    prefer_sim_params_csv=True,
    save_plot=True,
    show_plot=True,
    save_csv=True,
):
    if csv_file is None:
        csv_file = _default_simscape_results_csv()

    sim_csv = _default_sim_params_csv() if sim_params_csv is None else sim_params_csv
    robot_scale, robot_scale_source = resolve_robot_scale(sim_csv, default=1.0)
    mass_scale = robot_scale**3
    inertia_scale = robot_scale**5

    case = load_motor_payload_case(
        motor_mat_file=motor_mat_file,
        sim_params_csv=sim_csv,
        prefer_sim_params_csv=prefer_sim_params_csv,
    )
    base_params = load_params_from_mat(mat_file=mat_file, n_links=5)
    scaled_params = scale_rigid_body_params(base_params, robot_scale)
    params = apply_motor_lumped_inertia(
        scaled_params,
        stator_link_masses=case["stator_link_masses"],
        payload=case["payload"],
    )
    cols = _load_csv_columns(csv_file)

    I_ref_vec = np.asarray(case["I_ref_joint"], dtype=float).reshape(5)
    iref_mode = str(case.get("iref_model_mode", "q5_physical"))
    params, I_ref_diag_vec, iref_mode = apply_reflected_inertia_mode(params, I_ref_vec, iref_mode)
    b_visc_vec = np.asarray(case["b_visc_joint"], dtype=float).reshape(5)
    tau_coulomb_vec = np.asarray(case["tau_coulomb_joint"], dtype=float).reshape(5)

    q = np.column_stack([cols["q1"], cols["q2"], cols["q3"], cols["q4"], cols["q5"]])
    qd = np.column_stack([cols["qd1"], cols["qd2"], cols["qd3"], cols["qd4"], cols["qd5"]])
    qdd = np.column_stack([cols["qdd1"], cols["qdd2"], cols["qdd3"], cols["qdd4"], cols["qdd5"]])
    tau_sc = np.column_stack([cols["tau1"], cols["tau2"], cols["tau3"], cols["tau4"], cols["tau5"]])
    ee_sc = np.column_stack([cols["Xee"], cols["Yee"], cols["Zee"]])

    n = q.shape[0]
    tau_ne = np.zeros((n, 5))
    ee_ne = np.zeros((n, 3))

    for i in range(n):
        tau_rigid = newton_euler_serial5(q[i], qd[i], qdd[i], params)
        sign_qd = np.where(np.abs(qd[i]) > 1e-9, np.sign(qd[i]), 0.0)
        tau_ne[i, :] = tau_rigid + (I_ref_diag_vec * qdd[i]) + (b_visc_vec * qd[i]) + (tau_coulomb_vec * sign_qd)
        ee_ne[i, :] = forward_kinematics_ee(q[i], params)

    print("Validacion Serial (Simscape vs NE): motores + damping + payload")
    print(f"  robot_scale: {robot_scale:.6f} (origen: {robot_scale_source})")
    print(f"  Escalado dinamico: mass={mass_scale:.6f}, inertia={inertia_scale:.6f}")
    print(f"  dh.a escalado [m]: {np.round(params['dh_a'], 6).tolist()}")
    print(f"  dh.d escalado [m]: {np.round(params['dh_d'], 6).tolist()}")
    print(f"  Fuente caso: {case['source']}")
    print(f"  Perfil: {case['study_profile']}")
    if case["motor_ids"]:
        print(f"  Motor IDs: {case['motor_ids']}")
    if case["gear_ids"]:
        print(f"  Gear IDs : {case['gear_ids']}")
    print(f"  stator_link_masses (Link1..Link5) = {np.round(case['stator_link_masses'], 6).tolist()}")
    print(f"  I_ref_joint_raw (q1..q5) = {np.round(I_ref_vec, 6).tolist()}")
    print(f"  iref_model_mode = {iref_mode}")
    print(f"  I_ref_joint_diag_after_mode (q1..q5) = {np.round(I_ref_diag_vec, 6).tolist()}")
    print(f"  b_visc_joint [N*m*s/rad] (q1..q5) = {np.round(b_visc_vec, 6).tolist()}")
    print(f"  tau_coulomb_joint [N*m] (q1..q5) = {np.round(tau_coulomb_vec, 6).tolist()}")
    print(
        "  payload: "
        f"m={case['payload']['mass']:.3f} kg, "
        f"com_tcp={np.round(case['payload']['com_from_tcp'], 4).tolist()}"
    )
    print("  Formato: corr | RMSE | bias | fit_ref=(a*NE+b)")

    tau_names = [f"tau{i}" for i in range(1, 6)]
    ee_names = ["Xee", "Yee", "Zee"]
    for i, name in enumerate(tau_names):
        m = _signal_metrics(tau_sc[:, i], tau_ne[:, i])
        print(f"  {name:4s} {m['corr']:.4f} | {m['rmse']:.6f} | {m['bias']:.6f} | a={m['slope']:.4f}, b={m['intercept']:.4f}")
    for i, name in enumerate(ee_names):
        m = _signal_metrics(ee_sc[:, i], ee_ne[:, i])
        print(f"  {name:4s} {m['corr']:.4f} | {m['rmse']:.6e} | {m['bias']:.6e} | a={m['slope']:.4f}, b={m['intercept']:.4e}")

    if save_csv:
        out_csv = os.path.join(_results_dir(), "cr4_ne_motors_results.csv")
        _save_results_csv(out_csv, cols["time"], tau_ne, tau_sc, ee_ne, ee_sc)
        print(f"\n  CSV guardado en: {out_csv}")

    if plt is not None:
        fig, axes = plt.subplots(4, 2, figsize=(14, 13))
        ax = axes.ravel()
        for i, name in enumerate(tau_names):
            m = _signal_metrics(tau_sc[:, i], tau_ne[:, i])
            ax[i].plot(cols["time"], tau_sc[:, i], "-", linewidth=1.8, label="Simscape")
            ax[i].plot(cols["time"], tau_ne[:, i], "--", linewidth=1.8, label="NE")
            ax[i].set_title(f"{name} | corr={m['corr']:.3f}, RMSE={m['rmse']:.3f}")
            ax[i].set_xlabel("Tiempo [s]")
            ax[i].set_ylabel("Torque [Nm]")
            ax[i].grid(True, alpha=0.3)
            ax[i].legend()

        for j, name in enumerate(ee_names):
            m = _signal_metrics(ee_sc[:, j], ee_ne[:, j])
            a = ax[5 + j]
            a.plot(cols["time"], ee_sc[:, j], "-", linewidth=1.8, label=f"Simscape {name}")
            a.plot(cols["time"], ee_ne[:, j], "--", linewidth=1.8, label=f"NE {name}")
            a.set_title(f"{name} | corr={m['corr']:.3f}, RMSE={m['rmse']:.2e}")
            a.set_xlabel("Tiempo [s]")
            a.set_ylabel("m")
            a.grid(True, alpha=0.3)
            a.legend()

        fig.suptitle("CR4 Serial: Simscape vs NE (motores + damping + payload)", fontsize=12)
        fig.tight_layout()
        if save_plot:
            out_png = os.path.join(_results_dir(), "cr4_ne_motors_vs_simscape.png")
            fig.savefig(out_png, dpi=150, bbox_inches="tight")
            print(f"  Figura guardada en: {out_png}")
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
    else:
        print("  Matplotlib no disponible: se omite figura.")

    return tau_ne, ee_ne


if __name__ == "__main__":
    print("CR4 Newton-Euler serial 5DOF (motores + damping + payload)")
    print("=" * 68)
    validate_serial_with_motors(show_plot=False, save_plot=True, save_csv=True)
