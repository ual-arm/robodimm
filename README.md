# Robodimm: Interactive Robot Sizing & Motion Programming Environment

Robodimm is an interactive environment for robot trajectory programming, kinematic/dynamic analysis, and deterministic actuator sizing (motors and gearboxes). Evolved from the `kineforge` rigid body dynamics libraries, Robodimm features a real-time 3D visualizer (Three.js), client-side kinematics, and server-side dynamics solvers (FastAPI + Pinocchio) wrapped in an industrial-grade simulation interface.

The environment supports two main robot families:
1. **CR6 (Articulated Serial 6-DoF)**: Articulated serial arm with a decoupling spherical wrist (similar to ABB IRB 4600).
2. **CR4 (Parallel Palletizer 4-DoF)**: Parallel parallelogram linkage structure with 4 degrees of freedom (similar to ABB IRB 460).

---

## Overview

Robodimm can be deployed in two modes:
- **DEMO Mode**: A fully client-side standalone web application. Runs approximate dynamics in the browser and sizing based on those torques.
- **PRO Mode**: Integrates with a local Python backend running **Pinocchio** via loopback `127.0.0.1:8001`, enabling high-fidelity inverse dynamics computations (closed-chain KKT for CR4 and full RNEA for CR6).

---

## Features

* **3D Visualizer**: Interactive robot viewer built on Three.js featuring CAD axes, center of mass (COM) indicators, trajectory path visualizer, and natural Y-Up camera controls.
* **Jogging Panel**: Support for joint-space sliders and Cartesian coordinates (TCP) jogging relative to either the **World** or tool (**TCP**) frames. Continuous mouse-hold jogging is fully supported.
* **Duty Cycle Sequencer**:
  * Sequence commands including joint motions (`MoveJ`), linear tool motions (`MoveL`), and waits (`Pause`).
  * Pre-configured presets matching real industrial cycles.
  * Load programs directly using a robust JSON/YAML importer.
* **Deterministic Actuator Sizing**:
  * Passive candidate selection based on trajectory `TorqueLog` demands (does not modify robot spec or run iterative loops).
  * Evaluates candidates against 6 criteria (continuous output torque, peak output torque using a 5x rated motor limit policy, maximum speed, gearbox continuous torque, gearbox intermittent torque, and gearbox input speed limit).
  * Rated power check generates warnings instead of hard filtering by default.
  * User-friendly reducer type filtering (Strain wave / Harmonic and Cycloidal).
* **Auditing & Manifests**: Expose dynamic logs to CSV and complete reproducibility manifests to JSON matching schema `robodimm.actuator_sizing_report.v1`.

---

## Running Locally

To run the frontend development server:

1. Install Node.js dependencies:
   ```bash
   npm install
   ```
2. Start the development server:
   ```bash
   npm run dev
   ```
3. Open your browser at `http://localhost:5173/`.

---

## Building Production Assets

To package and minify the static web assets for production deployment:

```bash
npm run build
```

The generated `dist/` directory is fully static and can be deployed with any static hosting provider, reverse proxy, or inside a generic web container.

---

## Optional PRO Backend

To enable high-fidelity Pinocchio calculations, you can run the PRO backend locally on your machine:

### Linux/macOS
```bash
./releases/setup_backend.sh
```

### Windows
```bash
./releases/setup_backend.bat
```

The script automatically checks for a Conda/Mamba installation, creates the required environment from `environment.yml`, and boots the server bound strictly to loopback `127.0.0.1:8001`.

---

## Environment Variables

Robodimm uses Vite-style environment variables to configure backend connectivity. You can copy the provided templates to configure your environment:

```bash
cp .env.example .env
```

Available variables:
- `VITE_APP_MODE`: Set to `production` or `development`.
- `VITE_ENABLE_PRO_BACKEND`: Set to `false` to disable loopback health polling entirely (ideal for pure offline demo deployments).
- `VITE_PRO_BACKEND_URL_PRIMARY`: The primary URL of the local PRO backend (default: `http://127.0.0.1:8001`).
- `VITE_PRO_BACKEND_URL_FALLBACK`: The fallback URL (default: `http://localhost:8001`).

---

## Actuator Sizing

The sizing panel evaluates catalog parameters from `public/actuators_library.json`. Selections are deterministic:
- Integrates joint demands using sample-specific $dt$ time intervals.
- Matches pairs strictly according to the compatibility matrix.
- Evaluates six constraints per candidate: continuous output torque, peak output torque (5× rated motor factor), maximum joint speed, gearbox continuous torque, gearbox intermittent torque, and gearbox input speed limit.

### Sizing Objective

The **Sizing Objective** dropdown (Actuators tab) controls how passing candidates are ranked:

| Objective | Ranking priority |
|---|---|
| `min_mass` *(default)* | Lightest motor+gearbox assembly first |
| `min_power` | Smallest motor rated power first |
| `min_gearbox` | Smallest gearbox (by assembly mass) first |
| `max_margin` | Widest safety margin first |

All objectives still apply the same six hard pass/fail constraints — only the ranking among passing candidates changes.

---

## Validation

To run kinematics, dynamics, and sizing integration tests:

```bash
npx vitest run
```

To run TypeScript compiler verification:

```bash
npx tsc -p tsconfig.app.json --noEmit
```

## Inertial Parameters

The inertial parameters editor (Editor tab) lets users inspect and override the mass, centre of mass (COM), and inertia tensor for every robot body. Default values for each robot preset follow a **moving-body-only** model:

> **Important:** The datasheet total weight of a robot includes the fixed base casting, J1 motor and reducer housing, covers, cabling, and ballast — none of which load J2/J3 gravitationally. Only the moving mass **downstream** of each joint should be included in the link inertials. Do **not** scale link masses to match the datasheet total weight without a proper fixed-vs-moving breakdown.

### CR6 defaults (IRB 4600-45/2.05, 425 kg shipping weight)

| Link | Mass | Notes |
|---|---|---|
| LINK1 | 45 kg | Column structure rotating on J1 (J1 motor housing stays fixed) |
| LINK2 | 65 kg | Upper arm — main contributor to J2 gravity torque |
| LINK3 | 40 kg | Forearm with J4 motor housing |
| LINK4 | 28 kg | Wrist roll body |
| LINK5 | 17 kg | Wrist pitch body |
| LINK6 |  5 kg | Wrist output flange |
| **Total moving** | **200 kg** | Fixed base accounts for remaining ~225 kg of shipping weight |
| Payload *(default)* | 15 kg | Representative application load — override as needed |

### CR4 defaults (IRB 460, ~925 kg shipping weight)

| Link | Mass | Notes |
|---|---|---|
| SWING | 90 kg | Rotating column, loads J1 only |
| P\_ARM | 35 kg | Proximal arm link |
| LOWER\_ARM | 75 kg | Main lower arm beam — main contributor to J2 gravity torque |
| P\_LINK | 25 kg | Proximal parallel link |
| UPPER\_ARM | 40 kg | Upper arm / forearm |
| LOWER\_LINK | 20 kg | Lower parallel link |
| LINK\_PLATE | 15 kg | Link plate / coupler |
| UPPER\_LINK | 15 kg | Upper parallel link |
| TILT | 15 kg | Tilt / wrist body |
| DISK | 10 kg | J4 output disk / flange |
| **Total moving** | **340 kg** | Fixed base accounts for remaining ~585 kg of shipping weight |
| Payload *(default)* | 50 kg | Override as needed |

---

## License

This software is released under the MIT License.
