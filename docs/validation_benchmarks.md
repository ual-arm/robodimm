# Validation & Benchmarks

This document explains Robodimm's cross-software verification against
author-developed Simscape Multibody™ reference models, reports protocol- and
engine-specific RMSE for both supported robot families, and describes how to
reproduce the comparisons. Simscape is a reference implementation here, not an
independent experimental ground truth.

The reference suite is a stand-alone script rather than a normal fast unit
test. It loads a reproducibility manifest and Simscape-generated CSV from the
sibling `robodimm_paper/experiments/` repository, runs the PRO backend on every
sample, and reports residuals. It never generates or modifies evidence.

---

## 1. Simscape reference-model methodology

The reference data is produced by a MATLAB + Simulink + Simscape
Multibody™ pipeline. For each robot family:

1. **Geometry export.** The same `RobotSpec` JSON that drives the
   frontend is consumed by a Simscape builder that constructs a
   `smimport` compatible URDF, attaches rigid bodies with matching
   inertias, and defines the same joint tree (with passive joints for
   CR4's parallelogram).
2. **Reference trajectories.** A representative industrial trajectory for
   each family is provided as a CSV with columns
   `time, q_J{i} | q{i}, qd_J{i} | qd{i}, qdd_J{i} | qdd{i}, tau_J{i} | tau{i}`.
   Two column conventions are auto-detected by the regression loader
   (`backend/test_regression.py:80-119`):
   - `q_J1, qd_J1, ...` (Simscape-style, 1-indexed and explicitly
     prefixed)
   - `q1, qd1, ...` (plain)
3. **Simscape simulation.** The Simscape model uses variable-step `ode15s`
   with specified output times equal to the source trajectory samples. The
   recorded `tau_*` columns are the reference torques.
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
   (`CR4_SIMSCAPE_MASSES`, `test_regression.py:128-139`) to any body
   with `massKg == 0.0` before evaluation. This is the same
   `BODY_MASSES` table used inside `cr4_kkt.py:28-39`.

The reference CSVs and manifests live outside `robodimm/` on purpose: they are
separately versioned cross-software evidence in the paper repository.

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

mamba run -n robodimm-pro-backend python backend/test_regression.py \
  --protocol REG-ZD-v1
```

Exit code `0` means both robot comparisons ran and passed. Missing files are a
hard error rather than a skip, so a no-data run can never print success.

At application commit `d4736c9`, both commands run 2,896 CR4 and 2,553 CR6
samples. The observed PRO-vs-Simscape totals are summarized below; Sections 3.2
and 3.3 give the per-joint values and provenance.

| Protocol | Damping on every tested joint | CR4 total RMSE | CR6 total RMSE |
|---|---:|---:|---:|
| `E2E-VM05-v1` archived application regression | 0.5 N m/(rad/s) | 0.244639 Nm | approximately 1.15 × 10⁻¹² Nm |
| `REG-ZD-v1` fresh code regression | 0 N m/(rad/s) | 0.0482381 Nm | 1.15198 × 10⁻¹² Nm |

### 2.3 Protocol isolation

The `--protocol` flag selects a fixed data location and damping rule. Do not
copy, rename, or reuse one protocol's CSVs under the other protocol. To point at
a different paper checkout, set `ROBODIMM_PAPER_EXPERIMENTS`; expected relative
paths and filenames remain unchanged.

---

## 3. Protocol-specific RMSE results and thresholds

For both protocols, error is PRO model minus Simscape reference and samples are
weighted uniformly. Total RMSE is computed over all joints and all common
samples. Protocol identifier, damping value, and engine pair are part of every
result below.

### 3.1 Acceptance thresholds

The retained CR4 criteria enforced by `test_regression.py:193-198` are:

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

CR6 requires every joint and the total to remain below `0.01 Nm`; these checks
are enforced at `test_regression.py:325-328`.

### 3.2 Archived `E2E-VM05-v1` application regression — 0.5 damping

The application-side `E2E-VM05-v1` command replays the archived Simscape CSVs
with `0.5 N m/(rad/s)` damping on every tested joint. At application commit
`d4736c9`, its PRO-vs-Simscape results are:

| Family | J1 | J2 | J3 | J4 | J5 | J6 | Total RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| CR4 (`E2E-VM05-v1`) | 0.004639 | 0.408171 | 0.269752 | 0.001426 | — | — | **0.244639 Nm** |
| CR6 (`E2E-VM05-v1`) | 3.58e-13 | 2.69e-12 | 7.64e-13 | 1.35e-14 | 2.16e-14 | 2.39e-15 | **approximately 1.15e-12 Nm** |

These are PRO solver regressions. The archived complete workflow also compares
the browser DEMO engine against Simscape under the same `E2E-VM05-v1`
0.5-damping protocol; its CR6 DEMO-vs-Simscape total RMSE is
`0.0014272910085280494 Nm`. Engine pair and damping protocol must therefore be
reported together.

### 3.3 Fresh `REG-ZD-v1` code regression — zero damping

`REG-ZD-v1` was generated from application commit
`d4736c9a867930a7ce3987fb945766618280d3a7` and paper execution commit
`c4f79a6720a674adb60c08050b233ed2b19008a8`. It uses exact zero damping,
fresh PRO torques, fresh Simscape output, and no archived torque/reference
fallback. Both family manifests and the root manifest are complete.

| Family | J1 | J2 | J3 | J4 | J5 | J6 | Total RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| CR4 (`REG-ZD-v1`) | 0.00485863 | 0.06566990 | 0.07049461 | 0.00142560 | — | — | **0.0482381188296853 Nm** |
| CR6 (`REG-ZD-v1`) | 3.59e-13 | 2.69e-12 | 7.67e-13 | 1.35e-14 | 2.16e-14 | 2.39e-15 | **1.1519804174808516e-12 Nm** |

The source of record is
`robodimm_paper/experiments/regression/REG-ZD-v1/manifest.json`, with per-family
metrics and checksums below that directory. The historical zero-damping CR6
figure of approximately `9.2e-13 Nm` is superseded by this fresh
`REG-ZD-v1` result.

### 3.4 Why CR4 and CR6 differ

CR6 has no passive joints; the RNEA output already *is* the actuated
torque. CR4 has six passive joints that must be eliminated via the KKT
projection. Two numerical steps contribute to the larger CR4 residual under
both `E2E-VM05-v1` and `REG-ZD-v1`:

The two CR4 totals are not a damping-only sensitivity experiment: the archived
`E2E-VM05-v1` input also retains its pre-revision CR4 geometry, whereas
`REG-ZD-v1` uses the frozen Gate 4 candidate robot. They must not be subtracted
to attribute the difference solely to viscous damping.

- The cut-tree mapping uses the selected `1e-4 rad` common step for
  $\partial q^{\text{cut}}/\partial q^{\text{user}}$ and its directional
  derivative.
- The KKT linear solve
  $J_{c,p}^\top \lambda = -\tau_p^{\text{open}}$ is a least-squares
  solve over nine constraints in six passive coordinates (rank
  6); the null space is geometrically exact but numerically
  $\mathcal O(10^{-13})$ in double precision.

Under `REG-ZD-v1`, both contributions remain below the 0.3 Nm total target;
the fresh CR4 total RMSE is `0.0482381188296853 Nm`.

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
3. Runs both `python backend/test_regression.py --protocol E2E-VM05-v1` and
   `python backend/test_regression.py --protocol REG-ZD-v1`, failing on either
   non-zero exit.

The script returns exit code `0` on success and `1` on any
`AssertionError` (`test_regression.py:410-412`).

---

## 6. Summary

The nominal regressions are intentionally reported as two experiments:

| Protocol | Purpose | Damping | CR4 PRO-vs-Simscape total RMSE | CR6 PRO-vs-Simscape total RMSE |
|---|---|---:|---:|---:|
| `E2E-VM05-v1` | Archived application/workflow regression | 0.5 N m/(rad/s) | 0.244639 Nm | approximately 1.15e-12 Nm |
| `REG-ZD-v1` | Fresh mathematical code regression | 0 N m/(rad/s) | 0.0482381188296853 Nm | 1.1519804174808516e-12 Nm |

The archived `E2E-VM05-v1` tree has a known CR4 geometry qualification and is
not relabelled as zero-damping evidence. The fresh `REG-ZD-v1` capsule is tied
to candidate `d4736c9` and its own checksummed manifest. Neither nominal
regression replaces the expanded eight-scenario revision matrix or validates
actuator procurement by itself.
