# Robodimm

> **Interactive Robot Trajectory Programming, Rigid-Body Dynamics, and
> Deterministic Actuator Sizing — in your browser.**

Robodimm is a web-based environment for designing and validating the
mechanical sizing of industrial robots. It provides a real-time 3D
visualiser (Three.js), a client-side kinematic and approximate-dynamic
solver, and an optional Python backend that runs **Pinocchio 4** for
high-fidelity inverse dynamics. The actuator-sizer is passive,
deterministic, and reproducible — given a `TorqueLog` and a catalog, it
returns the same `best` candidate every time.

The environment supports two robot families only:

1. **CR6** — 6-DoF serial articulated arm, IRB 4600-class, with
   decoupling spherical wrist. **Standard DH + RNEA** solver.
2. **CR4** — 4-DoF parallel palletizer, IRB 460-class, with a
   parallelogram linkage. **Closed-chain KKT** solver on a Pinocchio
   cut tree.

## Presentation Video

A short walkthrough of the full workflow — parametric editor, jog,
program, signal recording, and deterministic actuator sizing.

https://github.com/ual-arm/robodimm/blob/main/docs/video_robodimm.mp4?raw=true

<video src="docs/video_robodimm.mp4" controls width="100%"></video>

---

## Key Features

### Frontend (browser, React 18 + Three.js + Zustand)

- **Real-time 3D viewer.** Z-up world frame, slate-dark background,
  CAD-aligned axes, COM markers, trajectory path, optional world grid.
  CAD-authored GLB/GLTF and STL meshes are loaded on demand; primitive
  geometry is the safe fallback.
- **Two robot families.** Switch between CR6 and CR4 with full
  geometry, inertia, and limit editors. The CR4 editor enforces the
  parallelogram closure $P = B + C - O$ and $E = D + C - O$ in real
  time.
- **Jogging panel.** Joint-space sliders and Cartesian XYZ+yaw jog
  relative to either the **World** or **TCP** frame. Continuous
  mouse-hold jogging.
- **Duty-cycle sequencer.** `MoveJ`, `MoveL`, and `Pause` instructions
  with named targets. JSON / YAML program import-export.
- **Deterministic actuator sizing.** Six hard pass/fail constraints
  (continuous output torque, 5× peak, max speed, gearbox continuous,
  gearbox intermittent, gearbox input speed) and four ranking
  objectives (`min_mass`, `min_power`, `min_gearbox`, `max_margin`).
  Full audit manifest: `robodimm.actuator_sizing_report.v1`.
- **Station objects.** Drop GLB environment meshes (tables, fences,
  fixtures) into the world frame without affecting the kinematic
  tree; the loader is a cancellable reconciler that prevents WebGL
  leaks.

### Backend (Python, FastAPI + Pinocchio 4)

- **CR4 closed-chain KKT.** A cut tree is built with
  `pin.JointModelRY`/`RZ` bodies for all ten links and three
  `pin.RigidConstraintModel` 3D contact constraints close the loops.
  Lagrange multipliers are recovered by `lstsq` of the passive-joint
  columns of $J_c^\top$.
- **CR6 Newton–Euler.** A direct call to `pin.rnea` on the six-DoF
  serial chain, with CAD-frame → link-frame inertial conversion when
  the user spec is provided in CAD coordinates.
- **Pinocchio model cache.** SHA-256 keyed on the canonical
  `json.dumps(robot, sort_keys=True)`. Cold start ~30 s; subsequent
  calls are sub-millisecond on cached models.
- **Viscous friction model.** A scalar $b_i\,\dot q_i$ term per joint,
  configured by `frictionCoeffNmSPerRad` on the joint limit. Default
  $b_i = 0$ to match the Simscape reference.
- **Validation against Simscape.** CR4 KKT reaches a total RMSE of
  **0.245 Nm** on a representative palletizing trajectory; CR6 RNEA
  matches to **9.2 × 10⁻¹³ Nm** (floating-point precision).

---

## Technology Stack

| Layer | Technologies |
|---|---|
| Frontend | React 18, TypeScript ~5.6, Vite 5.4, Three.js 0.184, Zustand 5, Tailwind CSS 3 (via PostCSS), Lucide-React, Recharts |
| Frontend tests | Vitest 1.6 |
| Backend | Python 3.9–3.10, FastAPI ≥ 0.100, Uvicorn, Pydantic ≥ 2.0 |
| Dynamics | Pinocchio ≥ 3.0 (Conda), NumPy ≥ 1.22, SciPy ≥ 1.8 |
| Packaging | Docker / Docker Compose, nginx 1.27 |
| Validation ground truth | MATLAB R2026a, Simulink, Simscape Multibody™ |

---

## Quick Links

| Document | What it covers |
|---|---|
| 🏁 [`docs/getting_started.md`](./docs/getting_started.md) | Prerequisites, install, DEMO and PRO modes, Docker, env vars, five-minute walkthrough, how to run the test suite |
| 📐 [`docs/math_foundations.md`](./docs/math_foundations.md) | DH convention, CR4 hardpoint invariants, the cut-tree mapping, the KKT matrix system, the closed-loop J4 sign convention, viscous friction model, trajectory blending |
| 🔌 [`docs/api_reference.md`](./docs/api_reference.md) | Every FastAPI endpoint, full JSON request/response payloads, the SHA-256 model cache, the trajectory hash, the CORS allowlist |
| 🎨 [`docs/frontend_guide.md`](./docs/frontend_guide.md) | Zustand store slices, Three.js scene factory, the CAD-aligned frame helper, the GLB station-object reconciler, the cancellable loader |
| ⚙️ [`docs/sizing_methodology.md`](./docs/sizing_methodology.md) | The six pass/fail constraints (with formulas), per-candidate margin metrics, the four ranking objectives, a worked example |
| 🧪 [`docs/validation_benchmarks.md`](./docs/validation_benchmarks.md) | Simscape comparison methodology, per-joint RMSE tables, the 0.245 Nm CR4 / 9.2e-13 Nm CR6 results, how to reproduce |

---

## Quick Start (60 seconds)

### DEMO mode (browser only, no Python)

```bash
npm install
npm run dev          # → http://localhost:5173
```

The full feature set works in DEMO mode: parametric editor, jog,
program editor, signal recording, and actuator sizing. The
approximate CR4 dynamics in the browser are not as accurate as the PRO
backend's KKT solver, but they are sufficient for design exploration.

### PRO mode (with Pinocchio)

```bash
# Backend (one-time setup + launch)
./releases/setup_backend.sh

# Frontend (in another terminal)
npm run dev
```

The header engine switcher lights the **PRO (Python API)** pill in
green when the backend advertises both `CR4.closed_chain_kkt = true`
and `CR6.serial_rnea = true`. The first PRO batch is slow (~30 s cold
start); subsequent calls are sub-millisecond thanks to the
SHA-256-keyed Pinocchio model cache.

### Docker

```bash
docker compose up --build                  # frontend on :8080
docker build -f Dockerfile.backend -t robodimm/backend-pro .
docker run -p 127.0.0.1:8001:8001 robodimm/backend-pro
```

---

## Running Tests

```bash
# Frontend (Vitest)
npx vitest run
npx vitest run -t "CR4"                    # name filter
npx vitest run src/math/actuators.test.ts  # single file

# Backend (regression vs Simscape)
mamba run -n robodimm-pro-backend python backend/test_regression.py
```

The Python regression script is **not** collected by `pytest` — it is
invoked directly because it loads the Simscape CSVs and reproducibility
manifests from the sibling `../ensayos/robodimm_cr{4,6}/` directory in
the workspace.

---

## Repository Layout

```
robodimm/
├── README.md                    ← this file (the portal)
├── AGENTS.md                    ← agent quick-start (commands, gotchas)
├── docs/                        ← SoftwareX-grade technical documentation
│   ├── getting_started.md
│   ├── math_foundations.md
│   ├── api_reference.md
│   ├── frontend_guide.md
│   ├── sizing_methodology.md
│   └── validation_benchmarks.md
├── src/                         ← React + Three.js frontend
│   ├── main.tsx, App.tsx
│   ├── api/backend.ts           ← PRO fetch wrappers (800 ms / 120 s timeouts)
│   ├── math/                    ← pure-TS FK/IK/dynamics/sizing
│   ├── model/                   ← Zustand store + schemas
│   ├── ui/                      ← Editor, Jog, Program, Sizing tabs
│   └── viewer/                  ← Three.js scene, meshLoaders, reconciler
├── backend/                     ← FastAPI + Pinocchio PRO backend
│   ├── main.py                  ← CORS, PNA preflight, /api/packages/static
│   ├── api/                     ← health, dynamics, packages routers
│   ├── dynamics/                ← cr4_kkt, cr6_serial, schemas, validation
│   └── test_regression.py       ← stand-alone Simscape regression
├── public/
│   └── actuators_library.json   ← static catalog served by nginx
├── packages/                    ← local robot packages (robot.json + meshes)
├── releases/                    ← setup_backend.sh / .bat
├── Dockerfile, Dockerfile.backend, docker-compose.yml
├── environment.yml              ← conda env 'robodimm-pro-backend'
├── nginx.conf                   ← 1 h cache for /actuators_library.json
├── package.json, vite.config.ts, tsconfig*.json, eslint.config.js
└── tailwind.config.js, postcss.config.js
```

---

## License

Robodimm is released under the **MIT License**.

```
MIT License

Copyright (c) 2024–2026 Custom Robotics
```

See [`LICENSE`](./LICENSE) for the full text.

---

## Citation

If you use Robodimm in academic work, please cite the SoftwareX paper:

```bibtex
@article{robodimm2026softwarex,
  author    = {{Custom Robotics}},
  title     = {{Robodimm}: Interactive Robot Trajectory Programming and
               Deterministic Actuator Sizing in the Browser with
               {Pinocchio}-backed Inverse Dynamics},
  journal   = {SoftwareX},
  volume    = {XX},
  pages     = {XXXXXX},
  year      = {2026},
  publisher = {Elsevier},
  doi       = {10.1016/j.softx.2026.XXXXXX},
  url       = {https://github.com/customrobotics/robodimm}
}
```

Software and accompanying Simscape reference data are versioned
together; please pin a specific release tag (e.g. `v1.0.0`) when
citing, and include the `dynamics_source` field from the
`actuator_sizing_report.v1` envelope to identify which engine
(`demo_frontend`, `pro_cr4_kkt`, or `pro_cr6_serial`) produced the
results.
