# Getting Started

This guide takes a reviewer from a fresh clone to a running Robodimm stack
(frontend + optional PRO backend) in under five minutes, and walks through a
complete first sizing study on either the CR4 parallel palletizer or the CR6
serial articulated arm.

---

## 1. Prerequisites

| Component | Required version | Notes |
|---|---|---|
| **Node.js** | ≥ 18 (tested with 20) | `node -v` |
| **npm** | ≥ 9 (bundled with Node 20) | `npm -v` |
| **Python** | 3.9 or 3.10 | Only needed for **PRO mode** (Pinocchio). |
| **conda / mamba** | mamba ≥ 1.5 recommended | Pinocchio is **not** pip-installable. |
| **Docker / Docker Compose** *(optional)* | ≥ 24 | Single-command full stack. |
| **WebGL-capable browser** | Chrome 120+, Firefox 120+, Safari 17+ | Three.js renderer. |

The default DEMO mode (browser-only) only requires Node.js. The PRO mode adds
the optional Python backend.

---

## 2. Quick start — DEMO mode (browser-only, no Python)

```bash
# 1. Clone and install
git clone <repo-url> robodimm
cd robodimm
npm install

# 2. Run the development server
npm run dev
# → open http://localhost:5173
```

That's it. The frontend runs entirely in the browser, performs approximate
inverse dynamics for CR4/CR6, and produces a complete actuator sizing report
from the bundled actuator catalog (`public/actuators_library.json`).

Useful scripts:

| Command | Purpose |
|---|---|
| `npm run dev` | Vite dev server on port 5173 with HMR |
| `npm run build` | `tsc -b && vite build` (strict TS, must pass) |
| `npm run preview` | Serve the production build locally |
| `npm run lint` | ESLint over the project |
| `npx tsc -p tsconfig.app.json --noEmit` | Type-check only (CI parity) |
| `npx vitest run` | Run all Vitest unit tests |

---

## 3. PRO mode — start the optional Python backend

The PRO backend exposes the same dynamics through a local FastAPI service bound
to `127.0.0.1:8001`, using **Pinocchio 4** as the rigid-body engine. CR4
trajectories are solved via closed-chain KKT and CR6 via Newton–Euler RNEA.

### 3a. Recommended (conda / mamba)

```bash
# Linux / macOS
./releases/setup_backend.sh
# Windows
.\releases\setup_backend.bat
```

The script:
1. Detects `mamba` (or `conda` as fallback).
2. Creates the `robodimm-pro-backend` environment from `environment.yml`.
3. Launches `python backend/main.py` on `127.0.0.1:8001`.

Manual equivalent:

```bash
mamba env create -f environment.yml
mamba run -n robodimm-pro-backend python backend/main.py
```

### 3b. Docker

```bash
# Frontend image (nginx serving the static build)
docker build -t robodimm/frontend .
docker run -p 8080:80 robodimm/frontend

# Backend image (micromamba + Pinocchio)
docker build -f Dockerfile.backend -t robodimm/backend-pro .
docker run -p 127.0.0.1:8001:8001 robodimm/backend-pro
```

The provided `docker-compose.yml` orchestrates the frontend on port 8080. The
backend container expects `ROBODIMM_HOST=0.0.0.0` (already set in
`Dockerfile.backend`) so it can be reached by the host browser.

### 3c. Production deployment behind Apache reverse proxy

This is the recommended setup for serving Robodimm over HTTPS on a public
domain (e.g. `https://customrobotics.es`).

**1. Create `.env` before building** (Vite bakes variables at compile time):

```bash
cp .env.production.example .env
# Edit .env — set VITE_PRO_BACKEND_URL_PRIMARY to your public domain:
# VITE_PRO_BACKEND_URL_PRIMARY=https://your-domain.com
# VITE_PRO_BACKEND_URL_FALLBACK=https://your-domain.com
```

**2. Build and start the frontend container:**

```bash
docker compose up --build -d   # nginx on :8080
```

**3. (Optional) Start the PRO backend container:**

```bash
docker build -f Dockerfile.backend -t robodimm/backend-pro .
docker run -d --name robodimm-backend -p 127.0.0.1:8001:8001 robodimm/backend-pro
```

**4. Apache VirtualHost** (`/etc/apache2/sites-enabled/your-site.conf`):

```apache
<VirtualHost *:443>
    ServerName your-domain.com
    SSLEngine on
    SSLCertificateFile    /etc/letsencrypt/live/your-domain.com/fullchain.pem
    SSLCertificateKeyFile /etc/letsencrypt/live/your-domain.com/privkey.pem
    ProxyPreserveHost On

    # PRO backend (optional — omit if not running the backend container)
    ProxyPass /api http://127.0.0.1:8001/api
    ProxyPassReverse /api http://127.0.0.1:8001/api

    # Frontend (nginx)
    ProxyPass / http://127.0.0.1:8080/
    ProxyPassReverse / http://127.0.0.1:8080/
</VirtualHost>
```

```bash
sudo apache2ctl configtest && sudo systemctl reload apache2
```

### 3d. Verify the backend

```bash
curl http://127.0.0.1:8001/api/health
# → {"status":"ok","version":"1.0.0","pinocchio_version":"...","allowed_origins":[...]}

curl http://127.0.0.1:8001/api/capabilities
# → {"capabilities":{"CR4":{"closed_chain_kkt":true},"CR6":{"serial_rnea":true}},"license_status":"dev_valid"}
```

In the running frontend, the header engine switcher (top right) lights the
**PRO (Python API)** pill in green when both `CR4` and `CR6` capabilities are
advertised. The first PRO batch is slow (cold start ~30 s while the
closed-chain Pinocchio model is built and cached); subsequent calls are
fast.

---

## 4. Environment variables

Copy one of the templates to `.env`:

```bash
cp .env.example .env          # default
cp .env.local.example .env     # local development
cp .env.production.example .env  # production-style
```

| Variable | Default | Description |
|---|---|---|
| `VITE_APP_MODE` | `production` | `production` or `development`. |
| `VITE_ENABLE_PRO_BACKEND` | `true` | Set to `false` to disable health polling (pure offline demo). |
| `VITE_PRO_BACKEND_URL_PRIMARY` | `http://127.0.0.1:8001` | Primary PRO backend URL. |
| `VITE_PRO_BACKEND_URL_FALLBACK` | `http://localhost:8001` | Fallback URL used after consecutive failures. |

`.env` is git-ignored.

---

## 5. Five-minute walkthrough

Once the frontend is open at `http://localhost:5173`:

1. **Pick a robot family.** The default is **CR6 (IRB 4600-class)**. Click
   `Change Robot` and select **CR4 (Palletizer)** to load the IRB 460-class
   parallel palletizer instead.
2. **Inspect the editor.** The *Editor Spec* tab shows the active geometry
   (DH table for CR6, planar hardpoints for CR4), inertial parameters, and
   joint limits. Defaults follow a *moving-body-only* model — datasheet
   shipping weight is intentionally not rescaled into link inertias (see the
   `Inertial Parameters` section of the top-level README).
3. **Lock the configuration.** Click **Lock Robot (Set)**. The viewer
   becomes interactive and the *Jog Panel*, *Program*, and *Sizing* tabs
   unlock.
4. **Jog the robot.** Switch to *Jog Panel*. Use the joint sliders or
   Cartesian XYZ + yaw jog. Hold a slider for continuous motion.
5. **Build a small program.** Switch to *Program*. Add a `MoveJ` target and
   instruction, optionally a `MoveL`, and a `Pause`. The frontend builds a
   smooth quintic-blend trajectory internally (see
   `docs/math_foundations.md` § *Trajectory generation*).
6. **Record dynamics.** Click **Signal Recording** in the *Sizing* tab. The
   frontend solves the trajectory in DEMO mode (browser) and POSTs the same
   samples to `/api/dynamics/batch` if the PRO backend is online. The
   resulting `TorqueLog` and `ActuatorSizingReport` are shown in the
   *Sizing* panel and can be exported as JSON or CSV.
7. **Switch engine.** Toggle the **DEMO / PRO** switch in the header to
   compare. The report is recomputed using the new engine; the manifest
   includes a `dynamics_source` field (`demo_frontend`, `pro_cr4_kkt`, or
   `pro_cr6_serial`).
8. **Change the sizing objective.** In the *Sizing* tab, open the **Sizing
   Objective** dropdown and pick `min_mass` (default), `min_power`,
   `min_gearbox`, or `max_margin`. The same six hard pass/fail criteria
   apply; only the ranking changes (see `docs/sizing_methodology.md`).

---

## 6. Running the test suite

### Frontend (Vitest)

```bash
npx vitest run                       # all unit tests
npx vitest run -t "CR4"              # name filter
npx vitest run src/math/actuators.test.ts   # single file
```

### Backend (regression vs Simscape)

The Python regression script is **not** collected by `pytest`; it must be
invoked directly with the `robodimm-pro-backend` env active:

```bash
mamba run -n robodimm-pro-backend python backend/test_regression.py
```

It loads Simscape-generated reference CSVs and the corresponding
reproducibility manifests from the sibling `../ensayos/robodimm_cr{4,6}/`
directories in the workspace. See `docs/validation_benchmarks.md` for the
expected RMSE thresholds and the methodology.

---

## 7. Where to go next

| I want to understand… | Read |
|---|---|
| The math behind CR4 closed-chain KKT and CR6 RNEA | [`docs/math_foundations.md`](./math_foundations.md) |
| The full FastAPI surface area, JSON schemas, and SHA-256 caching | [`docs/api_reference.md`](./api_reference.md) |
| How the React + Three.js viewer works, state model, GLB reconciler | [`docs/frontend_guide.md`](./frontend_guide.md) |
| The six hard sizing constraints and the four ranking objectives | [`docs/sizing_methodology.md`](./sizing_methodology.md) |
| The Simscape validation methodology and measured RMSE | [`docs/validation_benchmarks.md`](./validation_benchmarks.md) |
