# Validation & Benchmarks

This document explains how Robodimm's dynamics solvers are validated
against an independent ground-truth model (Simscape Multibody™), reports
the joint-level and total RMSE achieved on both supported robot
families, and describes how to reproduce the comparison.

The reference suite is a stand-alone script rather than a normal fast unit
test. It loads a reproducibility manifest and Simscape-generated CSV from the
sibling `robodimm_paper/experiments/` repository, runs the PRO backend on every
sample, and reports residuals. It never generates or modifies evidence.

---

## 1. Simscape ground-truth methodology

The reference data is produced by a MATLAB + Simulink + Simscape
Multibody™ pipeline. For each robot family:

1. **Geometry export.** The same `RobotSpec` JSON that drives the
   frontend is consumed by a Simscape builder that constructs a
   `smimport` compatible URDF, attaches rigid bodies with matching
   inertias, and defines the same joint tree (with passive joints for
   CR4's parallelogram).
2. **Reference trajectories.** A representative industrial trajectory
   (≈ 1 cycle of a palletizing pick-and-place for CR4, a multi-pose
   welding path for CR6) is generated offline as a CSV with columns
   `time, q_J{i} | q{i}, qd_J{i} | qd{i}, qdd_J{i} | qdd{i}, tau_J{i} | tau{i}`.
   Two column conventions are auto-detected by the regression loader
   (`backend/test_regression.py:36-49`):
   - `q_J1, qd_J1, ...` (Simscape-style, 1-indexed and explicitly
     prefixed)
   - `q1, qd1, ...` (plain)
3. **Simscape simulation.** The Simscape model is integrated with a
   fixed-step solver at the same `dt_s` as the frontend trajectory
   builder. The recorded `tau_*` columns are the reference torques.
4. **Manifest.** The exact `RobotSpec` used by Simscape is stored under
   `robodimm_paper/experiments/archive/submitted-validation-f33a676/`
   `E2E-VM05-v1/robodimm_cr{4,6}/robodimm_output/{demo,pro}/`.
   The loader prefers `demo/` then falls back to `pro/`.
5. **Damping protocol.** The default `E2E-VM05-v1` verification requires and
   preserves `0.5 N m/(rad/s)` on every joint. `REG-ZD-v1` explicitly zeros
   damping, but reads only from
   `experiments/regression/REG-ZD-v1/`; it never reinterprets E2E files as
   zero-damping evidence.
6. **Inertial override.** The CR4 manifest's `inertials` may be sparse;
   the script applies the moderated Simscape masses
   (`CR4_SIMSCAPE_MASSES`, `test_regression.py:78-89`) to any body
   with `massKg == 0.0` before evaluation. This is the same
   `BODY_MASSES` table used inside `cr4_kkt.py:28-39`.

The reference CSVs and manifests live outside `robodimm/` on purpose: they are
independent evidence versioned in the paper repository.

---

## 2. Reproducing the suite

### 2.1 Prerequisites

- The `robodimm-pro-backend` conda env active.
- The two repositories cloned side by side as `robodimm/` and
  `robodimm_paper/`. Override the latter with
  `ROBODIMM_PAPER_EXPERIMENTS=/path/to/experiments` if needed.
- For the optional re-generation of the Simscape CSVs, MATLAB R2026a
  with Simulink and Simscape Multibody™.

### 2.2 Run

```bash
mamba run -n robodimm-pro-backend python backend/test_regression.py \
  --protocol E2E-VM05-v1

# Available after the Phase 3 zero-damping capsule is exposed:
mamba run -n robodimm-pro-backend python backend/test_regression.py \
  --protocol REG-ZD-v1
```

Exit code `0` means both robot comparisons ran and passed. Missing files are a
hard error rather than a skip, so a no-data run can never print success.

Sample output (numbers reproduced from `test_regression.py:140-153`):

```
--- Running CR4 KKT regression vs Simscape (E2E-VM05-v1) ---
  Using manifest: ../robodimm_paper/experiments/archive/submitted-validation-f33a676/E2E-VM05-v1/robodimm_cr4/robodimm_output/demo/..._manifest.json
  Comparing N samples...
  CR4 Joint-level RMSE (Nm):
    J1: 0.000912 Nm  (max abs: 0.004201 Nm)
    J2: 0.172053 Nm  (max abs: 0.731802 Nm)
    J3: 0.180219 Nm  (max abs: 0.821005 Nm)
    J4: 0.001004 Nm  (max abs: 0.003872 Nm)
  Total RMSE: 0.245102 Nm
  ✅ CR4 KKT regression passed!
  ✅ CR4 zero-payload computation passed!
  --- CR4 specific feature tests ---
  ✅ SWING COM offset correct!
  ✅ Payload mass fusion at J4 correct!
  ✅ J4 sign convention test passed!
  ✅ Sparse inertials correctly fall back to BODY_MASSES!
  ✅ Custom geometry build and compute passed!

--- Running CR6 serial regression vs Simscape (E2E-VM05-v1) ---
  Comparing N samples...
  CR6 Joint-level RMSE (Nm):
    J1: 3.71e-13 Nm  (max abs: 1.18e-12 Nm)
    J2: 8.92e-13 Nm  (max abs: 2.04e-12 Nm)
    J3: 1.40e-12 Nm  (max abs: 3.77e-12 Nm)
    J4: 2.81e-13 Nm  (max abs: 9.94e-13 Nm)
    J5: 1.07e-12 Nm  (max abs: 2.39e-12 Nm)
    J6: 3.22e-13 Nm  (max abs: 8.01e-13 Nm)
  Total RMSE: 9.21e-13 Nm
  ✅ CR6 serial regression passed!
  --- CR6 specific feature tests ---
  ✅ frame='cad' inertial conversion produces expected torques!
  ✅ Identity tool_transform has no effect!
  ✅ Payload contribution test passed!
  ✅ DH theta_offset not duplicated!

🎉 ALL REGRESSION TESTS PASSED SUCCESSFULLY!
```

### 2.3 Protocol isolation

The `--protocol` flag selects a fixed data location and damping rule. Do not
copy, rename, or reuse one protocol's CSVs under the other protocol. To point at
a different paper checkout, set `ROBODIMM_PAPER_EXPERIMENTS`; expected relative
paths and filenames remain unchanged.

---

## 3. RMSE results and acceptance thresholds

### 3.1 CR4 — closed-chain KKT

The KKT solver reaches **sub-Nm RMSE on J1 and J4** (the swing and disk
joints, which are isolated single-DoF bodies) and **~0.18 Nm on J2 and
J3** (the lower and upper arm pitches that couple through the
parallelogram). The total RMSE is **0.245 Nm**, which is well under the
**0.3 Nm acceptance threshold** stated in the project plan.

The CR4 acceptance criteria enforced by `test_regression.py:148-152`:

| Joint | Threshold | Source |
|---|---:|---|
| J1 | 0.02 Nm | The single-DoF swing revolute; expected to be near floating-point. |
| J2 | 1.0 Nm  | Couples the lower arm to the parallelogram; small KKT residual. |
| J3 | 1.0 Nm  | Symmetric counterpart of J2. |
| J4 | 0.02 Nm | Single-DoF disk revolute; expected near floating-point. |
| Total | 0.3 Nm | Aggregate KKT accuracy across the 4-DoF user space. |

The residuals come from the KKT least-squares solve and the user-to-cut-tree
central differences. The frozen five-step sensitivity campaign selects
`1e-4 rad` for both mapping operations; see
`backend/test_cr4_fd_sensitivity.py` and `docs/math_foundations.md` § 3.9.

### 3.2 CR6 — open-chain Newton–Euler

The CR6 RNEA is a direct port of Pinocchio's `pin.rnea` applied to a
serial 6R. The comparison against Simscape's RNEA-equivalent
multibody integrator gives **machine-precision agreement**:

| Joint | RMSE (Nm) |
|---:|---:|
| J1 | 3.7 × 10⁻¹³ |
| J2 | 8.9 × 10⁻¹³ |
| J3 | 1.4 × 10⁻¹² |
| J4 | 2.8 × 10⁻¹³ |
| J5 | 1.1 × 10⁻¹² |
| J6 | 3.2 × 10⁻¹³ |
| **Total** | **9.2 × 10⁻¹³** |

All six joints pass the 0.01 Nm threshold by 10 orders of magnitude,
demonstrating that the two algorithms are mathematically equivalent on
this serial chain to floating-point precision
(`test_regression.py:278-280`).

### 3.3 Why the CR4 / CR6 gap

CR6 has no passive joints; the RNEA output already *is* the actuated
torque. CR4 has six passive joints that must be eliminated via the KKT
projection. Two numerical steps contribute to the ~10⁻¹ Nm residual:

- The cut-tree mapping uses the selected `1e-4 rad` common step for
  $\partial q^{\text{cut}}/\partial q^{\text{user}}$ and its directional
  derivative.
- The KKT linear solve
  $J_{c,p}^\top \lambda = -\tau_p^{\text{open}}$ is a least-squares
  solve over nine constraints in six passive coordinates (rank
  6); the null space is geometrically exact but numerically
  $\mathcal O(10^{-13})$ in double precision.

Both contributions are bounded well below the 0.3 Nm target.

---

## 4. Targeted unit tests in the regression script

Beyond the RMSE comparison, `test_regression.py` exercises properties that
guard model construction. They run after the relevant manifest and CSV have
been loaded; missing reference inputs fail the command before feature tests.

### 4.1 CR4 feature tests (`_test_cr4_specific_features`)

| Test | Assertion | What it protects |
|---|---|---|
| SWING COM offset | $\mathrm{lever} = (O - A) + \mathrm{com}_{\mathrm{SWING}}$ | Coordinate-frame shift on the swing body. |
| Payload fusion at J4 | $m_{J_4} = m_{\mathrm{DISK}} + m_{\mathrm{payload}}$ | End-effector mass concatenation. |
| J4 sign convention | $\tau_{J_4}(0) \approx 0$ at home; pure-spin torque matches the selected viscous coefficient. | Vertical-axis sign and damping protocol. |
| Sparse inertials | $\tau_{J_3} > 100\,\mathrm{Nm}$ with empty `inertials` | `BODY_MASSES` fallback (no zero-mass bodies). |
| Custom geometry | Custom hardpoint offsets still build and produce 4 torques. | Generic geometry build path. |

### 4.2 CR6 feature tests (`_test_cr6_specific_features`)

| Test | Assertion | What it protects |
|---|---|---|
| `frame = 'cad'` inertials | $\tau_{J_2} > 100\,\mathrm{Nm}$ | CAD-to-link inertial frame conversion. |
| Identity tool transform | $\lVert \tau - \tau_{\mathrm{eye}} \rVert_\infty < 10^{-9}$ | `tool_transform` defaulting. |
| Payload contribution | $\lVert \tau - \tau_{\mathrm{no\;pay}} \rVert_\infty > 0.1$ (when payload > 0). | Payload COM and inertia are added to J6. |
| `theta_offset` not duplicated | $\tau_{J_2} > 100\,\mathrm{Nm}$ at $q = 0$ with $\theta_{\mathrm{offset}} = -\pi/2$ | FK does not double-apply the joint offset. |

The feature assertions are mathematically independent of reference torques but
share the loaded manifest. The RMSE comparison remains the heavier data-gated
release check.

---

## 5. Continuous-integration wiring

There is no GitHub Actions workflow checked into the repository
(`.github/` is not present). To wire the suite into CI, add a job that:

1. Sets up the `robodimm-pro-backend` conda env from `environment.yml`.
2. Checks out the sibling `robodimm_paper` reference data.
3. Runs `python backend/test_regression.py --protocol E2E-VM05-v1` and fails
   on non-zero exit.

The script returns exit code `0` on success and `1` on any
`AssertionError` (`test_regression.py:351-353`).

---

## 6. Summary

The table below summarizes the archived submitted comparison and is not the
final 2026 revision benchmark. The revision protocol records a known CR4
geometry discrepancy in that archive, so final evidence must be regenerated
after the candidate source commit is frozen.

| Metric | CR4 (KKT) | CR6 (RNEA) |
|---|---:|---:|
| Solver | Pinocchio RNEA on cut tree + KKT projection | Pinocchio RNEA |
| Mean per-joint RMSE | J1/J4 ~10⁻³ Nm, J2/J3 ~0.18 Nm | ~10⁻¹² Nm |
| **Total RMSE** | **0.245 Nm** | **9.2 × 10⁻¹³ Nm** |
| Acceptance threshold | 0.3 Nm | 0.01 Nm |
| Reference source | Simscape Multibody™ | Simscape Multibody™ |
| Reproducibility manifest schema | `*_reproducibility_manifest.json` (paper experiments) | same |
| Determinism | Single-pass; cached model, no iteration | Single-pass; pure function of $(q,\dot q,\ddot q)$ |

The archived CR4 comparison is within its retained source-regression threshold;
the archived CR6 comparison is at floating-point precision. Neither result by
itself validates actuator procurement or replaces the frozen revision matrix.
