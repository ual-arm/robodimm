# Mathematical Foundations

This document describes every mathematical model that Robodimm implements
directly. All quantities are in SI units (m, rad, s, kg, N·m). The frontend
solver (`src/math/serial6.ts`, `src/math/palletizer.ts`) and the PRO backend
solvers (`backend/dynamics/cr6_serial.py`, `backend/dynamics/cr4_kkt.py`) share
the same kinematic and dynamic conventions, so the formulas below apply to
both DEMO and PRO modes.

The two robot families supported by Robodimm are:

- **CR6** — 6-DoF serial articulated arm (IRB 4600-class), `nx = 6`,
  generalized coordinates $q = (q_1, \ldots, q_6)$.
- **CR4** — 4-DoF parallel palletizer (IRB 460-class), `nx = 4`,
  actuated coordinates $q = (q_1, q_2, q_3, q_4)$ for joints J1 (column
  rotation), J2 (lower arm pitch), J3 (upper arm pitch via a parallelogram),
  and J4 (end-effector roll). Six more bodies are passive (driven by the
  closed-chain kinematic constraints).

---

## 1. Coordinate frames and conventions

- The world frame is **Z-up**, with the gravity vector
  $g = (0,\, 0,\, -9.80665)\,\mathrm{m/s^2}$ (`G = 9.80665` in
  `backend/dynamics/cr4_kkt.py:25`).
- The 3D viewer compensates for Three.js being Y-up by applying a
  $-\pi/2$ rotation about the X-axis to a `robotGroup` that owns the active
  robot (`src/viewer/scene.ts:22-25`). The kinematic conventions below are
  *unaware* of this display transform; the same FK applies in both modes.
- **CAD vs link frames.** Each link exposes a *user-facing* CAD frame and an
  *internal* DH (or constraint-tree) frame. The CAD frame is anchored at the
  home configuration; during motion it rotates relative to the world. The
  helper `getCadAlignedTransform(T, T_home)` returns a transform whose
  rotation equals $R(T) \cdot R(T_\text{home})^\top$ and whose translation
  matches the live FK, so user-authored CAD meshes stay aligned visually
  (`src/viewer/RobotViewer.tsx:328-331`, `src/viewer/frameHelpers.ts`).

---

## 2. CR6 — Denavit–Hartenberg forward kinematics

The CR6 uses the **standard (Craig) DH convention** with the
$q$–$\theta$–$d$ transformation:

$$
T_{\text{DH}}(a, \alpha, d, \theta) \;=\;
\mathrm{Rot}_z(\theta)\;\mathrm{Trans}_z(d)\;\mathrm{Trans}_x(a)\;\mathrm{Rot}_x(\alpha)
$$

implemented in pure TypeScript at `src/math/matrix.ts:standardDH` and in
Python at `backend/dynamics/cr6_serial.py:_standard_dh` (lines 250-265).
Expanded:

$$
T_{\text{DH}} \;=\;
\begin{bmatrix}
c_\theta & -s_\theta c_\alpha &  s_\theta s_\alpha & a\,c_\theta \\
s_\theta &  c_\theta c_\alpha & -c_\theta s_\alpha & a\,s_\theta \\
0        &  s_\alpha          &  c_\alpha          & d          \\
0        &  0                 &  0                 & 1
\end{bmatrix}
$$

A per-joint `theta_offset_rad` is added to the input $q_i$ before the DH
rotation (`backend/dynamics/cr6_serial.py:126`). A 4×4 `tool_transform` is
right-multiplied onto the chain to obtain the TCP frame.

### Jacobian

The geometric Jacobian columns for body $b$ (centred at the COM) are
(`backend/dynamics/cr6_serial.py:186-209`):

$$
J_{v,b}^{(i)} = \begin{cases}
z_{i-1} \times (p_b - o_{i-1}), & i \le \text{link}(b) \\
0, & \text{otherwise}
\end{cases},
\qquad
J_{\omega,b}^{(i)} = \begin{cases}
z_{i-1}, & i \le \text{link}(b) \\
0, & \text{otherwise}
\end{cases}
$$

where $z_{i-1}$ and $o_{i-1}$ are the world-space axis and origin of joint
$i$ from the FK pass.

### Viscous friction

A scalar viscous-friction coefficient $b_i$ (units N·m·s/rad) is added per
joint:

$$
\tau_{f,i} = b_i\,\dot q_i
$$

`Serial6Template.inverse_dynamics` and the frontend `Serial6Engine` both add
this term in user space (`backend/dynamics/cr6_serial.py:217-218`,
`src/math/serial6.ts`).

---

## 3. CR4 — closed-chain kinematics and the KKT dynamics

### 3.1 The four actuated joints

The CR4 user space has four actuated revolute joints (after the swing
revolute about world Y, the geometry is planar in the XZ plane):

- $q_1$ — J1 swing, about world $Y$ axis through point $A$.
- $q_2$ — J2, rotates the **lower arm** $OC$ about $O$ in the XZ plane.
- $q_3$ — J3, rotates the **crank** $OB$ about $O$ in the XZ plane.
- $q_4$ — J4, rotates the **end-effector disk** about an axis fixed to the
  tilt linkage; its $z$-axis is reconstructed from the home configuration
  in `j4_vertical_axis_frame_in_hgee` (`cr4_kkt.py:109-120`).

The other six bodies (P_ARM, LOWER_ARM, P_LINK, UPPER_ARM, LOWER_LINK,
LINK_PLATE, UPPER_LINK, TILT) are passive and follow rigidly from the four
actuated coordinates through the loop-closure constraints.

### 3.2 Hardpoint closure constraints

The geometry is described by twelve planar hardpoints in the XZ plane
(`src/model/schemas.ts:Cr4Geometry`):

$$
\{A, O, B, C, D, E, F, G, H, P, J_4, EE, TCP\}
$$

with the following physical invariants enforced by
`backend/dynamics/validation.py:validate_cr4_geometry`:

1. $B.z = O.z$  (crank horizontal at home)
2. $C.x = O.x$  (lower arm vertical at home)
3. $P = B + C - O$  (parallelogram closure)
4. $E = D + C - O$  (lower parallelogram closure)
5. $H.z = C.z$  (extension horizontal)
6. $EE.z = H.z$  (tool mount horizontal)
7. $TCP.x = EE.x$  (tool tip vertical)
8. $TCP.z < EE.z$  (tool tip points down)
9. All critical segment lengths $> 10^{-4}\,\mathrm{m}$
10. The circles centred at $F$ and $H$ with radii $\overline{FG}$ and
    $\overline{HG}$ must intersect (triangle inequality on $FH$).

The user-facing `updateCr4HardpointXZ` and `updateCr4DesignParameter` actions
in the Zustand store re-derive the dependent hardpoints (P, E) on every edit
to preserve the parallelogram invariant in real time
(`src/model/state.ts:239-273`).

### 3.3 Forward kinematics of the closed chain

Given $q = (q_1, q_2, q_3, q_4)$, the FK is computed in two stages.

**Stage 1 — planar chain.** A 2D rotation about world $Y$ is applied to
points $B$ and $C$:

$$
R_y(\theta) =
\begin{bmatrix}
\cos\theta & 0 & \sin\theta \\
0          & 1 & 0          \\
-\sin\theta& 0 & \cos\theta
\end{bmatrix}
$$

giving $B = R_y(q_3)(B_0 - O) + O$ and $C = R_y(q_2)(C_0 - O) + O$. The
parallelogram closure then yields $P = B + (C - O)$ and $E = D + (C - O)$
(`cr4_kkt.py:266-300`).

**Stage 2 — linkage circle intersection.** Point $G$ is the intersection of
two circles in the XZ plane — one of radius $\overline{FG}$ centred at $F$,
and one of radius $\overline{HG}$ centred at $H$ — disambiguated by the
*signed* area preference of the home configuration (`circle_intersection_xz`,
`cr4_kkt.py:303-329`):

$$
G \;=\; F + a\,\hat e \;+\; h\,\hat e_\perp,
\quad
a = \frac{r_a^2 - r_b^2 + d^2}{2d}, \quad
h = \sqrt{\,r_a^2 - a^2\,}
$$

where $\hat e = (H - F)/d$ in XZ. The two candidate intersections are
filtered by the home-side sign, and the closer one to the home $G$ is
chosen.

### 3.4 State mapping to the Pinocchio cut tree

The PRO backend builds a **cut tree** with one rotational DoF per body
(10 in total for CR4 — see `cr4_kkt.py:128-256`). The mapping from user
space $q^{\text{user}} \in \mathbb{R}^4$ to the cut-tree configuration
$q^{\text{cut}} \in \mathbb{R}^{10}$ is

$$
q^{\text{cut}}_i = -\,\text{atan2}\!\big(z_j - z_i,\; x_j - x_i\big)
\qquad
(\text{closed\_full\_configuration})
$$

with the actuated cut indices $(0, 1, 2, 9)$ being $q^{\text{user}}$ itself
(`cr4_kkt.py:337-359`). The Jacobian $J = \partial q^{\text{cut}}/\partial
q^{\text{user}}$ is computed by central differences with $\varepsilon = 10^{-6}$
and angle-unwrapping to keep the cut tree continuous through $2\pi$
wraps (`mapped_jacobian`, `cr4_kkt.py:362-372`).

The velocity and acceleration mappings are

$$
\dot q^{\text{cut}} = J\,\dot q^{\text{user}}, \qquad
\ddot q^{\text{cut}} = J\,\ddot q^{\text{user}} + \dot J\,\dot q^{\text{user}},
$$

with $\dot J \dot q$ computed by a symmetric difference of Jacobians at
$q^{\text{user}} \pm \varepsilon \dot q^{\text{user}}$ (`mapped_state`,
`cr4_kkt.py:375-385`).

### 3.5 Open-loop RNEA on the cut tree

With the cut-tree state, Pinocchio's Recursive Newton–Euler Algorithm
(`pin.rnea`) is evaluated on the **unconstrained** cut tree:

$$
\tau^{\text{open}} = M(q^{\text{cut}})\,\ddot q^{\text{cut}} \;+\; C(q^{\text{cut}}, \dot q^{\text{cut}})\,\dot q^{\text{cut}} \;+\; g(q^{\text{cut}})
$$

(`cr4_kkt.py:430`). The four actuators at indices $\mathcal{A} = \{0, 1, 2,
\text{idx}_v(J_4)\}$ are present; the other six are passive and must carry
zero torque.

### 3.6 KKT system: closing the loops

The loop-closure constraints are encoded as three Pinocchio 3D contact
constraints (`cr4_kkt.py:249-254`):

- $g_1$: PCH at $P + \vec{PC}$ to OC at $C + \vec{OC}$
- $g_2$: CEF at $E$ to DE at $D + \vec{DE}$
- $g_3$: HGEE at $G$ to FG at $F + \vec{FG}$

The full constraint Jacobian
$J_c = \partial g / \partial q^{\text{cut}} \in \mathbb{R}^{9 \times 10}$
is obtained from `pin.getConstraintsJacobian` and rows of magnitude below
$10^{-10}$ are dropped (`cr4_kkt.py:440-441`).

The KKT condition for the closed chain is that **all passive joint torques
must vanish**:

$$
\tau^{\text{restored}}_p = \tau^{\text{open}}_p + (J_c^\top \lambda)_p = 0,
\qquad p \in \mathcal{P}
$$

Solving for the Lagrange multipliers in the least-squares sense
(`cr4_kkt.py:447`):

$$
\lambda \;=\; \mathrm{lstsq}\!\big(J_{c,p}^\top,\; -\tau^{\text{open}}_p\big)
$$

and the restored actuated torques are

$$
\tau^{\text{restored}} = \tau^{\text{open}} + J_c^\top \lambda,
\qquad
\tau^{\text{act}} = \tau^{\text{restored}}_{\mathcal{A}}
$$

### 3.7 Mapping back to user space

Torques are projected back to user space through the transpose Jacobian of
the actuated cut configuration (`closed_torque_to_user`,
`cr4_kkt.py:388-397`):

$$
\tau^{\text{user}} = \left(\frac{\partial q^{\text{actuated-cut}}}{\partial q^{\text{user}}}\right)^{\!\!\!\top}\;\tau^{\text{act}}
$$

A fixed sign flip is applied to J4 (`tau_user[3] *= -1.0`) so the reported
torque matches the Simscape convention for the vertical re-aligned axis
(`cr4_kkt.py:396`).

### 3.8 Viscous friction

After user-space mapping, the viscous friction term

$$
\tau^{\text{user}}_i \;+=\; b_i \cdot \dot q^{\text{user}}_i
$$

is added per actuated joint using the
`frictionCoeffNmSPerRad` field of the joint's `LimitSpec`
(`cr4_kkt.py:457-461`).

### 3.9 KKT diagnostics

For each sample the backend reports
(`cr4_kkt.py:466-477`):

- `constraint_residual_norm`:
  $\lVert J_{c,\mathcal{A}}^\top \lambda + \tau^{\text{open}}_{\mathcal{A}} -
  \tau^{\text{restored}}_{\mathcal{A}} \rVert_2$
- `passive_torque_residual_norm`:
  $\lVert \tau^{\text{restored}}_{\mathcal{P}} \rVert_2$
- `condition_number`: $\sigma_{\max}(J_c)/\sigma_{\min}(J_c)$ from the SVD

These are the metrics asserted against thresholds in
`backend/test_regression.py` (e.g. J1, J4 RMSE < 0.02 Nm; total RMSE
< 0.3 Nm).

---

## 4. Inertial helpers and defaults

The PRO backend provides geometric constructors for typical link shapes
(`backend/dynamics/pinocchio_utils.py`):

- `rod_x_inertia(m, L, t)` — thin rod along X with thickness $t$ (default
  10 mm). Tensor diagonal:
  $(m(t^2 + t^2)/12,\; m(L^2 + t^2)/12,\; m(L^2 + t^2)/12)$.
- `cylinder_z_tensor(m, r, l)` — cylinder along Z:
  $(m(3r^2 + l^2)/12,\; m(3r^2 + l^2)/12,\; mr^2/2)$.
- `triangle_plate_inertia_xz(m, vertices, t)` — triangular plate in XZ.
- `payload_inertia(m, com, I)` — payload; defaults to a 30 mm cube if
  `inertiaKgM2` is omitted.

For CR4, the moderated body masses
(`backend/dynamics/cr4_kkt.py:28-39` and
`backend/test_regression.py:78-89`) are:

| Body | Mass (kg) |
|---|---:|
| `SWING`     |  90 |
| `P_ARM`     |  35 |
| `LOWER_ARM` |  75 |
| `P_LINK`    |  25 |
| `UPPER_ARM` |  40 |
| `LOWER_LINK`|  20 |
| `LINK_PLATE`|  15 |
| `UPPER_LINK`|  15 |
| `TILT`      |  15 |
| `DISK`      |  10 |
| **Σ moving** | **340** |

For CR6, the default IRB 4600-45/2.05 masses (45/65/40/28/17/5 kg for
LINK1..LINK6, 200 kg total moving) are documented in the top-level
`README.md`. **The shipping weight of 425 kg is not rescaled into the link
inertias** — the fixed base, covers, and ballast are intentionally excluded.

---

## 5. Trajectory generation

Both the frontend (`src/math/trajectory.ts`) and the legacy backend
trajectory builder
(`backend/api/dynamics.py:append_quintic_segment`, lines 214-237) use the
same quintic blend for each `MoveJ` segment:

$$
s(u) = 10u^3 - 15u^4 + 6u^5
$$

with derivatives

$$
\dot s = (30u^2 - 60u^3 + 30u^4) / T, \quad
\ddot s = (60u - 180u^2 + 120u^3) / T^2
$$

for $u \in [0, 1]$ and segment duration $T$. These blend functions are
$C^2$-continuous in $q$ (continuous position, velocity, and acceleration).
The duration is sized by the maximum per-joint delta and the user-supplied
`speed_rad_s` (with the legacy `1.875` factor that yields a peak
dimensionless velocity of $1.875$ at $u = 0.5$). `MoveL` extends this with
a Cartesian duration floor based on `tcp_speed_m_s`.

---

## 6. References

- Featherstone, *Rigid Body Dynamics Algorithms*, Springer 2008.
- Craig, *Introduction to Robotics*, 3rd ed., Pearson 2005 (DH convention).
- Carpentier et al., *The Pinocchio C++ library*, IEEE-RAM 2019
  (used via `pin.rnea` and `pin.RigidConstraintModel`).
- Bruyninckx, *Closed-form inverse dynamics of closed-chain mechanisms*, 1996
  (KKT projection for parallel manipulators).
