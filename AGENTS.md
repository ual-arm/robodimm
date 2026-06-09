# Robodimm Agent Instructions

Web-based robot trajectory programming, kinematic/dynamic analysis, and deterministic actuator sizing. Two robot families only: **CR6** (6-DoF serial, IRB 4600-class) and **CR4** (4-DoF parallel palletizer, IRB 460-class).

Workspace context (sibling projects, conventions, models): see `../AGENTS.md`. Sibling Python project with shared CR4/Pinocchio code: `../kineforge/`.

## Run modes

- **DEMO**: pure client-side, browser-only. Frontend math, frontend dynamics approximation, frontend sizing. No Python needed.
- **PRO**: frontend + local Python FastAPI backend at `127.0.0.1:8001` (Pinocchio for high-fidelity ID via closed-chain KKT for CR4 and RNEA for CR6). The `App.tsx` engine switcher (line 70–107) gates PRO behind a successful `hasBackend` health poll; cold start can take 30+ s, so `pingBackendUrl` uses 800 ms per attempt with 10 s polling.

## Frontend (Node + Vite + React 18 + TS + Three.js + Zustand)

Dev server (port 5173):
```bash
npm install
npm run dev
```

Build, preview, lint:
```bash
npm run build       # tsc -b && vite build  (strict, must pass)
npm run preview
npm run lint        # eslint .
```

Type check (CI parity):
```bash
npx tsc -p tsconfig.app.json --noEmit
```

### Frontend layout
```
src/
├── App.tsx              # Tabs: editor | jog | program | actuators. Engine switcher DEMO/PRO.
├── main.tsx             # React 18 createRoot, StrictMode.
├── api/backend.ts       # fetch wrappers: health, capabilities, /api/dynamics, /api/dynamics/batch, /api/packages/upload. 800 ms health timeout, 120 s batch.
├── math/                # Pure TS FK/IK/dynamics, no React. Tests live here next to code.
│   ├── serial6.ts       # CR6 (DH, standard convention)
│   ├── palletizer.ts    # CR4
│   ├── palletizerGeometry.ts
│   ├── matrix.ts        # 4×4 helpers, rotx/roty/rotz, standardDH
│   ├── trajectory.ts
│   ├── recording.ts     # generateSimulationTorques, runIterativeSizingAlgorithm
│   └── actuators.ts     # selectActuatorsForLog (sizing)
├── model/
│   ├── state.ts         # Zustand store: 943 lines. ALL app state lives here.
│   └── schemas.ts       # RobotSpec, ProgramSpec, TorqueLog, ActuatorLibrary, etc.
├── ui/                  # Tabs (ActuatorsTab, EditorTab, JogTab, ProgramTab) + InspectorPanel
├── viewer/              # Three.js: RobotViewer, scene, frameHelpers, meshLoaders, trajectoryLayer, robotVisuals
└── index.css
```

### Frontend tests (Vitest 1.6, no vitest.config — uses Vite config)
```bash
npx vitest run                            # all
npx vitest run src/math/serial6.test.ts   # single file (if it exists)
npx vitest run -t "CR4"                   # name filter
```
Test files: `src/math/math.test.ts` (CR6 + CR4 + reference cases from `reference_tests.json`), `actuators.test.ts`, `palletizerGeometry.test.ts`, `model/state.test.ts`.

### Frontend conventions
- **SI internally** (m, rad, s, N·m). UI shows deg/mm and must convert at the boundary — never inline ad-hoc conversions.
- **q-space**: the CR4 parallel palletizer has internal q (actuated) vs. dependent coordinates vs. TCP. Stick to one convention per call site; prefer the helpers in `palletizer.ts` over recomputing.
- **Zustand store is canonical state** — UI components read from `useRobodimmStore()`, do not duplicate state locally for things that are already there.
- Backend URL is mutable via `setApiBackendUrl` (used by health probe to switch primary → fallback).
- No comment-only edits; lint passes `react-refresh/only-export-components` (warn-level).

## Backend (Python 3.9–3.10, FastAPI, Pinocchio)

Pinocchio needs conda/mamba — pip is not enough. Use the project-provided script:
```bash
# Linux/macOS
./releases/setup_backend.sh
# Windows
.\releases\setup_backend.bat
```
This creates conda env `robodimm-pro-backend` from `environment.yml` and boots on `127.0.0.1:8001`.

Manual (Docker / debugging):
```bash
mamba env create -f environment.yml
mamba run -n robodimm-pro-backend python backend/main.py
# or
docker build -f Dockerfile.backend -t robodimm/backend-pro .
docker compose up --build      # frontend image; backend via setup_backend.sh
```

Override loopback bind for containers: `ROBODIMM_HOST=0.0.0.0` (already set in `Dockerfile.backend`).

### Backend layout
```
backend/
├── main.py              # FastAPI app, CORS allowlist (customrobotics.es + localhost:5173/5174), PNA preflight, mounts /api/packages/static
├── api/
│   ├── health.py        # GET /api/health, /api/capabilities (must return CR4 + CR6)
│   ├── dynamics.py      # /api/dynamics, /api/dynamics/batch, /api/dynamics/validate
│   └── packages.py      # /api/packages/upload
├── dynamics/
│   ├── cr4_kkt.py       # CR4 closed-chain inverse dynamics via KKT
│   ├── cr6_serial.py    # CR6 RNEA
│   ├── pinocchio_utils.py
│   ├── schemas.py       # Pydantic request/response models, manifest schema
│   └── validation.py
└── test_regression.py   # pytest-less regression suite; reads ../ensayos/robodimm_cr{4,6}
```

### Backend tests
The regression script is **not** collected by pytest. Run directly from project root with the env active; it loads manifest CSVs from `../ensayos/robodimm_cr4/robodimm_output/{demo,pro}/` and `../ensayos/robodimm_cr6/...` (sibling dirs in the workspace). Path bootstrap in `main.py` and `test_regression.py` adds `../kineforge` to `sys.path` — keep that working.

### Backend conventions
- **Pinocchio model cache** lives in-process in `cr4_kkt._MODEL_CACHE` keyed by robot hash. Cold first-sample is 30+ s; subsequent are fast.
- **Manifest schema** for reproducibility: `robodimm.actuator_sizing_report.v1` (frontend) and `robodimm.dynamics.v1` (batch API).
- CORS allowlist is hard-coded in `main.py:22` and `health.py:8` — keep in sync. PNA preflight is added by a custom middleware (`main.py:42`).
- Static meshes served from `~/.robodimm/packages/` (auto-created on startup).

## Environment / config

Vite env vars (prefix `VITE_`): `VITE_APP_MODE`, `VITE_ENABLE_PRO_BACKEND`, `VITE_PRO_BACKEND_URL_PRIMARY`, `VITE_PRO_BACKEND_URL_FALLBACK`. Templates: `.env.example`, `.env.local.example`, `.env.production.example`. Copy one to `.env` — never commit real `.env` (gitignored).

## Common tasks

| Task | Command |
|---|---|
| Add a new actuator library entry | edit `public/actuators_library.json` (served by nginx with 1 h cache) |
| Add a robot package | drop into `packages/<name>/` with `robot.json` + `meshes/`; upload via `/api/packages/upload` to materialize in `~/.robodimm/packages/` |
| Run a single sizing objective on trajectory | Editor tab → Set → Program tab → Program → Actuators tab → Sizing Objective |
| Inspect a dynamics mismatch (PRO vs DEMO) | see `scratch/debug_j2_torque.py`, `scratch/debug_sizing_configs.py` (throwaway scripts) |
| Inertial defaults | use **moving-body-only** masses (README §"Inertial Parameters"). Datasheet shipping weight ≠ link masses; do not rescale. |

## Gotchas an agent will hit

- `vitest` is pinned to `1.6` while `vite` is `5.4` — do not bump one without the other.
- `tailwindcss` is `^3.4.19` but `@tailwindcss/vite` is `^4.3.0`; `postcss.config.js` uses the v3 plugin pipeline. Don't "modernize" to `@tailwindcss/postcss` without checking.
- `tsconfig.app.json` uses `noEmit: true` — type check is separate from build, both run in `npm run build` via `tsc -b`.
- `package-lock.json` is tracked; `Dockerfile` uses `npm ci`. Do not delete the lockfile.
- CORS in `main.py` only allows the dev ports 5173 and 5174 plus production domains; any new frontend port requires editing the allowlist.
- `KINEFORGE_MIGRATION_REFERENCE.md` and `PLAN_ROBODIMM_FRONTEND_KINEFORGE.md` are historical/architectural context for the port from the desktop Kineforge project — useful for understanding CR4/CR6 invariants, but they are not normative for the current code.
