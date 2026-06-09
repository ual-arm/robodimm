# Actuator Sizing Methodology

Robodimm's actuator sizing is **passive and deterministic**: given a
`TorqueLog` and an `ActuatorLibrary`, it evaluates every compatible
motor–gearbox pair against six hard pass/fail constraints, then ranks the
passing candidates by the chosen `SizingObjective`. There are **no
iterative loops** that modify the robot spec.

All logic lives in `src/math/actuators.ts` (393 lines); the input shapes
and output envelope are declared in `src/model/schemas.ts`.

---

## 1. Inputs

### 1.1 `TorqueLog`

Produced by either the DEMO frontend (`generateSimulationTorques`,
`src/math/recording.ts`) or the PRO backend (`/api/dynamics/batch`):

```ts
interface TorqueLog {
  joint_names: string[];
  samples: Array<{
    time_s: number;
    q: number[];
    velocity: number[];          // or joint_velocity in legacy logs
    acceleration: number[];      // or joint_acceleration in legacy logs
    joint_velocity?: number[];
    joint_acceleration?: number[];
    tau: number[];
  }>;
  dt_s: number;
  engine_used?: string;          // 'demo_frontend' | 'pro_cr4_kkt' | 'pro_cr6_serial'
  model_id?: string;
  manifest?: any;                // returned by PRO; identifies hashes + q-space
}
```

### 1.2 `ActuatorLibrary`

Loaded once at app start from `/actuators_library.json` (the static
catalog served by nginx with a 1 h `Cache-Control` header). The schema:

```ts
interface ActuatorLibrary {
  motors: MotorSpec[];
  gearboxes: GearboxSpec[];
  compatibility_matrix: Record<string, string[]>;   // motor_id → gearbox_ids
  metadata: { version: string; last_updated: string; description: string };
}
```

`compatibility_matrix` is the *strict* allowlist: a motor not present in
the matrix produces **zero** candidates for that joint
(`actuators.ts:307-309`). This is the *strict mode* of candidate
generation; it is intentional and matches real catalog
manufacturer-pairing rules.

`MotorSpec` carries `rated_power_W`, `rated_torque_Nm`, `no_load_speed_rpm`,
`stall_torque_Nm`, `mass_kg`, and rotor inertia. `GearboxSpec` carries
`ratio`, `stages`, `efficiency`, `max_continuous_torque_Nm`,
`max_intermittent_torque_Nm`, `max_input_speed_rpm`, `backlash_arcmin`,
`mass_kg`, `for_servo_mm`, and the `type` discriminator (`'harmonic'` or
`'cycloidal'`).

### 1.3 `SizingMargins`

The safety factors applied uniformly per joint:

```ts
interface SizingMargins {
  continuous: number;           // ≥ 1.0; default 1.5
  peak: number;                 // ≥ 1.0; default 2.0
  speed: number;                // ≥ 1.0; default 1.2
  power: number;                // ≥ 1.0; default 1.1
  motorPeakFactor: number;      // 5.0 in the standard policy
  enforcePowerLimit?: boolean;  // false → power is a warning, not a fail
  sizingObjective?: SizingObjective;   // default 'min_mass'
}
```

`motorPeakFactor = 5.0` encodes the "5× rated torque" peak-overload policy
that is standard for short robotic transients; see § 2.2.

### 1.4 The `SizingObjective` enum

```ts
type SizingObjective = 'min_mass' | 'min_power' | 'min_gearbox' | 'max_margin';
```

The objective changes **only the ranking**, not the hard constraints.

---

## 2. Joint demands

`computeJointDemands(torqueLog)` (`actuators.ts:8-69`) computes per-joint
demands from the log. RMS quantities are integrated with **time-weighted
trapezoidal accumulation** to handle non-uniform sampling
(`actuators.ts:24-48`):

$$
\tau_{\mathrm{rms}} = \sqrt{\dfrac{\int \tau^2\,dt}{T}}, \quad
\omega_{\mathrm{rms}} = \sqrt{\dfrac{\int \dot q^2\,dt}{T}}, \quad
P_{\mathrm{rms}} = \sqrt{\dfrac{\int P^2\,dt}{T}}
$$

with $P = \tau\,\dot q$ and $T$ the total sample time (computed as
$\sum dt$). Sample intervals with $dt \le 0$ are skipped to defend
against backwards or zero-length jumps. Peak quantities are
$\max(|x|)$ over the log. The peak regenerative power (negative-power
events) is tracked separately as `regen_peak_W` for future regen-resistor
sizing.

The `JointDemand` object returned per joint is:

```ts
interface JointDemand {
  joint_name: string;
  tau_rms_Nm: number;
  tau_peak_Nm: number;
  speed_rms_rad_s: number;
  speed_peak_rad_s: number;
  power_rms_W: number;
  power_peak_W: number;
  regen_peak_W: number;
  cycle_time_s: number;
}
```

---

## 3. The six hard pass/fail constraints

For each motor+gearbox pair, the candidate passes if **all** of the
following hold (`evaluateMotorGearboxCandidate`, `actuators.ts:75-199`).

Let $\eta$ be the gearbox efficiency, $r$ the ratio, $b_p$ the
`motorPeakFactor` (default $5.0$). Then:

### 3.1 Rule 1 — Output continuous torque

$$
\tau_{\mathrm{out,cont}} \;=\; \tau_{\mathrm{motor,rated}} \cdot r \cdot \eta
$$

Pass if
$\tau_{\mathrm{out,cont}} \;\ge\; \tau_{\mathrm{rms}} \cdot m_{\mathrm{cont}}$.

### 3.2 Rule 2 — Output peak torque (5× overload policy)

$$
\tau_{\mathrm{out,peak}} \;=\; \tau_{\mathrm{motor,rated}} \cdot b_p \cdot r \cdot \eta
$$

Pass if
$\tau_{\mathrm{out,peak}} \;\ge\; \tau_{\mathrm{peak}} \cdot m_{\mathrm{peak}}$.

The `5× rated torque for short robotic transients` policy is encoded
verbatim in the produced `ActuatorSizingReport.motor_peak_policy`
(`actuators.ts:363`).

### 3.3 Rule 3 — Output maximum speed

$$
\omega_{\mathrm{out,max}} \;=\; \frac{n_{0,\mathrm{motor}}}{r} \cdot \frac{2\pi}{60}
$$

(with $n_0$ the no-load speed in rpm). Pass if
$\omega_{\mathrm{out,max}} \;\ge\; \omega_{\mathrm{peak}} \cdot m_{\mathrm{speed}}$.

### 3.4 Rule 4 — Gearbox continuous torque

Pass if
$T_{\mathrm{gb,cont}} \;\ge\; \tau_{\mathrm{rms}} \cdot m_{\mathrm{cont}}$.

### 3.5 Rule 5 — Gearbox intermittent (peak) torque

Pass if
$T_{\mathrm{gb,int}} \;\ge\; \tau_{\mathrm{peak}} \cdot m_{\mathrm{peak}}$.

### 3.6 Rule 6 — Gearbox input speed

The demanded input speed (motor-side) is
$\omega_{\mathrm{peak}} \cdot r$, converted to rpm. Pass if

$$
n_{\mathrm{gb,in,max}} \;\ge\; \frac{\omega_{\mathrm{peak}}\,r}{2\pi/60} \cdot m_{\mathrm{speed}}.
$$

### 3.7 Power (warning, by default)

A seventh check compares
$P_{\mathrm{motor,rated}} \cdot \eta$ against
$P_{\mathrm{rms}} / \eta \cdot m_{\mathrm{power}}$. By default this
generates a warning, not a failure (`enforcePowerLimit = false`); set the
flag in `SizingMargins` to convert it to a hard fail.

A candidate is considered a *pass* if and only if it has zero
`failure_reasons` after the six (or seven) checks.

---

## 4. Per-candidate margin metrics

For ranking and inspection, every candidate also exposes
(`actuators.ts:162-179`):

| Metric | Definition |
|---|---|
| `continuous_margin` | $\tau_{\mathrm{out,cont}} / \tau_{\mathrm{rms}}$ (∞ if demand is 0) |
| `peak_margin` | $\tau_{\mathrm{out,peak}} / \tau_{\mathrm{peak}}$ |
| `speed_margin` | $\omega_{\mathrm{out,max}} / \omega_{\mathrm{peak}}$ |
| `gearbox_continuous_margin` | $T_{\mathrm{gb,cont}} / \tau_{\mathrm{rms}}$ |
| `gearbox_peak_margin` | $T_{\mathrm{gb,int}} / \tau_{\mathrm{peak}}$ |
| `power_margin` | $P_{\mathrm{motor,rated}} \cdot \eta / P_{\mathrm{rms}}$ |
| `min_margin` | $\min$ of the five mechanical margins (excludes power) |

`Infinity` is used to represent "demand is 0" so the candidate naturally
ranks first.

---

## 5. Ranking objectives

`rankCandidates(candidates, objective)` (`actuators.ts:209-280`) is a
total-order sort with explicit tie-breakers. The structure is:

1. **Passes first.** All `passes === true` candidates precede all
   `passes === false` candidates.
2. **Objective-dependent primary key** (see below).
3. **Common tie-breakers** (in this order):
   - Lower gear ratio first (`a.ratio - b.ratio`).
   - For objectives other than `max_margin`: higher `min_margin` first
     (∞ first).

The four objectives choose their primary key as:

| Objective | Primary key | Secondary | Tertiary |
|---|---|---|---|
| `min_mass` *(default)* | `total_mass_kg` ↓ | `motor.rated_power_W` ↓ | — |
| `min_power` | `motor.rated_power_W` ↓ | `total_mass_kg` ↓ | — |
| `min_gearbox` | `gearbox.mass_kg` ↓ | `total_mass_kg` ↓ | `motor.rated_power_W` ↓ |
| `max_margin` | `min_margin` ↑ | `total_mass_kg` ↓ | `motor.rated_power_W` ↓ |

All `> 1e-5` differences trigger a comparison; smaller differences fall
through to the next tie-breaker, ensuring deterministic ordering for
ties in floating-point keys.

---

## 6. The selection entrypoint

`selectActuatorsForLog(torqueLog, library, margins, robotKind, robotName, filters?)`
(`actuators.ts:286-368`) is the top-level function the rest of the app
calls. It:

1. Computes the per-joint `JointDemand[]`.
2. For each joint, iterates the catalog **only over pairs allowed by
   `compatibility_matrix`**. The reducer-type filter (`'harmonic'`,
   `'cycloidal'`, or `'any'`) further restricts `GearboxSpec.type`.
3. Evaluates and ranks every candidate.
4. Picks the **first** `passes === true` candidate as `best`. If no
   candidate passes, `best = undefined`.
5. Returns the full `ActuatorSizingReport` envelope.

The envelope is the canonical audit document:

```ts
interface ActuatorSizingReport {
  schema: "robodimm.actuator_sizing_report.v1";
  robot_kind: string;
  robot_name: string;
  dynamics_source: string;     // e.g. 'pro_cr4_kkt'
  torque_log_hash: string;     // 32-bit djb2 of JSON(tau samples)
  catalog_version: string;
  catalog_anonymized: boolean;
  motor_peak_policy: "rated_torque_x_5_for_short_robotic_transients";
  margins: SizingMargins;
  joints: JointActuatorSelection[];
  complete: boolean;           // every joint has a best candidate
}
```

`complete === true` is the green light for "this robot can be built from
the catalog" — a hard precondition before procurement.

---

## 7. Worked example

Suppose a CR4 J2 demand is
$\tau_{\mathrm{rms}} = 1100\,\mathrm{Nm}$, $\tau_{\mathrm{peak}} = 2300\,\mathrm{Nm}$,
$\omega_{\mathrm{peak}} = 1.2\,\mathrm{rad/s}$, with margins
$(m_{\mathrm{cont}}, m_{\mathrm{peak}}, m_{\mathrm{speed}}) = (1.5, 2.0, 1.2)$.

A motor `AC_7500W_1500` ($\tau_{\mathrm{rated}} = 47.7\,\mathrm{Nm}$,
$n_0 = 3000\,\mathrm{rpm}$) paired with a harmonic reducer
`HD65_160` ($r = 160$, $\eta = 0.85$, $T_{\mathrm{gb,cont}} = 2400\,\mathrm{Nm}$,
$T_{\mathrm{gb,int}} = 4500\,\mathrm{Nm}$, $n_{\mathrm{gb,in,max}} = 3500\,\mathrm{rpm}$):

| Rule | Required | Available | Pass? |
|---|---:|---:|---|
| 1. Output cont. | $1100 \cdot 1.5 = 1650$ | $47.7 \cdot 160 \cdot 0.85 = 6487$ | ✓ |
| 2. Output peak  | $2300 \cdot 2.0 = 4600$ | $47.7 \cdot 5 \cdot 160 \cdot 0.85 = 32436$ | ✓ |
| 3. Output speed | $1.2 \cdot 1.2 = 1.44$ | $3000/160 \cdot 0.105 = 1.96$ | ✓ |
| 4. GB cont.     | $1650$ | $2400$ | ✓ |
| 5. GB int.      | $4600$ | $4500$ | ✗ |

The candidate fails Rule 5 only. Ranking will demote it behind any pair
that satisfies all six (or even just rules 1–4 with a higher-rated
gearbox), and the *best* selected for J2 will be the first such pair in
the `min_mass`-ordered list.

---

## 8. Why passive, not iterative?

The deprecated `runIterativeSizingAlgorithm` (`src/math/recording.ts:65-140`)
modified the robot's inertials to account for the selected actuator
masses, re-ran dynamics, and re-selected. The current code path is
*passive* because:

- The inertial contribution of a harmonic or cycloidal reducer at the
  joint is at most a few percent of the gearbox-rated torque / radius²,
  and a system engineer must verify the full mass budget before
  procurement. Iterating on the spec silently to make the candidate
  *fit* would mask that verification step.
- A deterministic, single-pass selection is **reproducible** — the
  report's `torque_log_hash` and `catalog_version` are sufficient to
  reproduce the same `best` for the same `TorqueLog`.

The legacy function is retained for backwards compatibility with
existing scripts (`@deprecated` since the deterministic selection was
introduced).
