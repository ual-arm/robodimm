# Backend API Reference

The PRO backend is a FastAPI service bound to `127.0.0.1:8001`. It serves
three purposes:

1. Health and capability discovery (`/api/health`, `/api/capabilities`).
2. Inverse dynamics for one or many samples
   (`/api/dynamics`, `/api/dynamics/inverse`, `/api/dynamics/batch`).
3. Robot package upload and mesh asset serving
   (`/api/packages/upload`, `/api/packages/static/...`).

This document lists every endpoint, the verbatim JSON schemas, and the
caching / optimisation strategy used to make repeated requests cheap.

The full OpenAPI specification is generated automatically by FastAPI and is
available at `http://127.0.0.1:8001/docs` once the server is running. The
schemas below are extracted from `backend/dynamics/schemas.py`.

---

## 1. Conventions

- **Content type**: every request and response is `application/json` except
  `/api/packages/upload` (`multipart/form-data`).
- **CORS allowlist** (`backend/main.py:22-29` and
  `backend/api/health.py:8-15`):
  - `https://customrobotics.es`, `https://www.customrobotics.es`
  - `http://localhost:5173`, `http://localhost:5174`
  - `http://127.0.0.1:5173`, `http://127.0.0.1:5174`
  A custom middleware in `backend/main.py:42-47` adds the
  `Access-Control-Allow-Private-Network: true` header on `OPTIONS` for PNA
  preflight.
- **Joint counts**: `CR4` expects 4 elements for `q`, `qd`, `qdd`; `CR6`
  expects 6. The endpoints return `400` with an explanatory message on
  mismatch.
- **Units**: SI (m, rad, rad/s, rad/s², kg, N·m, W).

---

## 2. Endpoints

### 2.1 `GET /api/health`

Liveness probe. No body. Returns Pinocchio version and the CORS allowlist.

**Response 200**

```json
{
  "status": "ok",
  "version": "1.0.0",
  "pinocchio_version": "3.6.0",
  "allowed_origins": [
    "https://customrobotics.es",
    "https://www.customrobotics.es",
    "http://localhost:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174"
  ]
}
```

### 2.2 `GET /api/capabilities`

Used by the frontend engine-switcher to decide whether to enable PRO mode.
Both `CR4.closed_chain_kkt` and `CR6.serial_rnea` must be `true`; otherwise
the PRO button stays disabled and shows the "Offline" badge
(`src/api/backend.ts:62-63`).

**Response 200**

```json
{
  "capabilities": {
    "CR4": { "closed_chain_kkt": true },
    "CR6": { "serial_rnea": true }
  },
  "license_status": "dev_valid"
}
```

### 2.3 `POST /api/dynamics/inverse`

Single-sample inverse dynamics. Accepts a full `RobotSpecModel` and a
single `q, qd, qdd` triple.

**Request body**

```json
{
  "robot": {
    "kind": "CR4",
    "name": "irb460",
    "geometry": { "A": [0,0,0], "O": [0,0,0.7], "B": [0.4,0,0.7], "C": [0,0,1.4], "...": "..." },
    "inertials": { "SWING": { "massKg": 90, "comM": [0.18,0,0.25] }, "...": "..." },
    "payload":    { "massKg": 50, "comM": [0,0,-0.2] },
    "limits": [
      { "name": "J1", "lowerLimitRad": -3.14, "upperLimitRad": 3.14, "frictionCoeffNmSPerRad": 0.0 },
      "..."
    ]
  },
  "q":   [0.0, 0.5, -0.3, 0.0],
  "qd":  [0.0, 0.1, -0.05, 0.0],
  "qdd": [0.0, 0.2, -0.10, 0.0],
  "schema_version": "robodimm.dynamics.v1",
  "options": {}
}
```

**Response 200 — `DynamicsResponse`**

```json
{
  "tauNm":   [12.3, 1450.7, -820.4, 4.1],
  "powerW":  [0.0, 145.07, 41.02, 0.0],
  "engine_used": "pro_cr4_kkt",
  "model_id":    "cr4_pinocchio_kkt.v1",
  "manifest": {
    "model_id": "cr4_pinocchio_kkt.v1",
    "backend_version": "1.0.0",
    "pinocchio_version": "3.6.0",
    "robot_hash": "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
    "trajectory_hash": null,
    "q_space_convention": "standard_dh_with_offsets",
    "timestamp": "2026-06-09T12:34:56Z"
  },
  "warnings": []
}
```

`engine_used` is `pro_cr4_kkt` or `pro_cr6_serial` depending on `robot.kind`.
For CR4, `Diagnostics` are *not* included on the single-sample endpoint;
use `/api/dynamics/batch` to retrieve them.

### 2.4 `POST /api/dynamics/batch`

Batched inverse dynamics. Same `robot` body; samples are an explicit list.

**Request body — `DynamicsBatchRequest`**

```json
{
  "robot":  { "...see above...": "..." },
  "schema_version": "robodimm.dynamics.v1",
  "options": {},
  "samples": [
    { "time_s": 0.000, "q": [0,0,0,0], "qd": [0,0,0,0], "qdd": [0,0,0,0] },
    { "time_s": 0.005, "q": [0,0.01,0,0], "qd": [0,2,0,0], "qdd": [0,0,0,0] }
  ]
}
```

The samples list is bounded to 1..10 000 by a Pydantic field validator
(`backend/dynamics/schemas.py:96-103`).

**Response 200 — `DynamicsBatchResponse`**

```json
{
  "joint_names": ["J1", "J2", "J3", "J4"],
  "samples": [
    {
      "time_s": 0.0,
      "q": [0,0,0,0],
      "velocity": [0,0,0,0],
      "acceleration": [0,0,0,0],
      "tau":  [0.0, 0.0, 0.0, 0.0],
      "power": [0.0, 0.0, 0.0, 0.0]
    }
  ],
  "dt_s": 0.005,
  "engine_used": "pro_cr4_kkt",
  "model_id": "cr4_pinocchio_kkt.v1",
  "manifest": {
    "model_id": "cr4_pinocchio_kkt.v1",
    "backend_version": "1.0.0",
    "pinocchio_version": "3.6.0",
    "robot_hash": "9f86d0...0a08",
    "trajectory_hash": "3b4c5d...ee01",
    "q_space_convention": "standard_dh_with_offsets",
    "timestamp": "2026-06-09T12:34:56Z"
  },
  "diagnostics": [
    { "constraint_residual_norm": 1.4e-12, "passive_torque_residual_norm": 3.2e-13, "condition_number": 14.7 }
  ],
  "warnings": []
}
```

`diagnostics` is only populated for CR4 (CR6 returns `null` and a
`warnings` array).

### 2.5 `POST /api/dynamics` *(legacy)*

Convenience endpoint that takes a full `ProgramSpec` (named targets +
instructions), builds a quintic-blended trajectory on the server, and
returns the torque log in one round trip. Internally it just calls
`build_program_trajectory_py` then dispatches to the CR4 or CR6 batch
solver (`backend/api/dynamics.py:239-365`).

**Request body — `LegacyDynamicsRequest`**

```json
{
  "robot":   { "...": "RobotSpecModel" },
  "program": {
    "name": "default",
    "targets":     [{ "name": "P1", "q": [0.0, 0.5, -0.3, 0.0] }],
    "instructions":[
      { "type": "MoveJ", "target_name": "P1", "speed_rad_s": 1.0, "zone_m": 0.0 }
    ]
  }
}
```

**Response 200**

```json
{
  "joint_names": ["J1","J2","J3","J4"],
  "samples":     [{ "time_s": 0.0, "q": [0,0,0,0], "velocity":[0,0,0,0],
                    "acceleration":[0,0,0,0], "joint_velocity":[0,0,0,0],
                    "joint_acceleration":[0,0,0,0], "tau":[0,0,0,0] }],
  "dt_s": 0.005
}
```

### 2.6 `POST /api/dynamics/validate`

Geometry-only validation. No dynamics. Useful for editor UIs.

**Request body** — a single `RobotSpecModel`.

**Response 200 — `ValidationResponse`**

```json
{ "valid": true,  "errors": [], "warnings": [] }
```
or
```json
{ "valid": false, "errors": ["Crank B must be horizontal with Pivot O: B.z=0.70000 != O.z=0.70001"], "warnings": [] }
```

### 2.7 `POST /api/packages/upload`

Upload a robot package as `multipart/form-data` (a `robot.json` plus its
mesh assets). Returns the resolved `RobotSpec` with `meshUrl` fields
re-pointed to the backend's static file server.

**Form fields**

- One or more file parts under the form name `files`. Each part uses the
  browser's `webkitRelativePath` (when present) to preserve the directory
  structure.

**Constraints** (`backend/api/packages.py:13-26`)

- Maximum file size: **50 MB** per file (`413` if exceeded).
- Allowed extensions: `.glb`, `.gltf`, `.bin`, `.stl`, `.json`.
- Paths must be relative; absolute paths and `..` segments are rejected
  with `400`.

**Response 200** — the resolved `RobotSpecModel`:

```json
{
  "kind": "CR6",
  "name": "my_robot",
  "geometry": { "...": "..." },
  "inertials": { "...": "..." },
  "payload":   { "...": "..." },
  "limits":    [ "..." ],
  "visuals": [
    {
      "body": "LINK3",
      "kind": "mesh",
      "meshUrl": "http://127.0.0.1:8001/api/packages/static/<uuid>/meshes/link3.glb",
      "originM": [0,0,0],
      "rpyRad":  [0,0,0],
      "scale":   [1,1,1]
    }
  ]
}
```

### 2.8 `GET /api/packages/static/{package_id}/{path}`

Static mesh server. Mounted on `~/.robodimm/packages/`
(`backend/main.py:50-52`). The directory is created on startup.

---

## 3. Pydantic schemas

The full set of request and response models lives in
`backend/dynamics/schemas.py` (154 lines). The most important shapes are
reproduced here for reference.

### 3.1 `InertialSpecModel`

```python
class InertialSpecModel(BaseModel):
    massKg: float = Field(ge=0.0)                              # kg
    comM: Optional[List[float]] = None                         # [x,y,z] m
    inertiaKgM2: Optional[List[List[float]]] = None             # 3x3 tensor
    frame: Optional[str] = None                                # "cad" | "link"
```

Field validators enforce `len(comM) == 3`, `shape(inertiaKgM2) == (3,3)`,
and non-negative diagonal entries
(`backend/dynamics/schemas.py:13-29`).

### 3.2 `RobotSpecModel`

```python
class RobotSpecModel(BaseModel):
    kind: Literal['CR4', 'CR6']
    name: str
    geometry: Dict[str, Any]   # schema depends on kind (see validation.py)
    inertials: Dict[str, InertialSpecModel]
    payload: InertialSpecModel
    limits: List[LimitSpecModel]
```

### 3.3 `TrajectorySampleModel`, `DynamicsBatchRequest`

```python
class TrajectorySampleModel(BaseModel):
    time_s: float
    q:   List[float]   # 4 (CR4) or 6 (CR6)
    qd:  List[float]
    qdd: List[float]

class DynamicsBatchRequest(BaseModel):
    robot: RobotSpecModel
    samples: List[TrajectorySampleModel]   # 1..10000
    schema_version: str                    # e.g. "robodimm.dynamics.v1"
    options: Optional[Dict[str, Any]] = {}
```

### 3.4 `DynamicsResponse` and `DynamicsBatchResponse`

```python
class DynamicsManifestModel(BaseModel):
    model_id: str
    backend_version: str
    pinocchio_version: str
    robot_hash: str
    trajectory_hash: Optional[str] = None
    q_space_convention: str
    timestamp: str

class CR4DiagnosticsModel(BaseModel):
    constraint_residual_norm: float
    passive_torque_residual_norm: float
    condition_number: float

class BatchSampleResponse(BaseModel):
    time_s: float
    q: List[float]
    velocity: List[float]
    acceleration: List[float]
    tau: List[float]
    power: List[float]
```

---

## 4. SHA-256 caching strategy

Two caches make repeated requests cheap.

### 4.1 Per-robot model cache (in-process)

`backend/dynamics/cr4_kkt.py:_MODEL_CACHE` is a module-level
`Dict[str, BuiltClosedModel]` keyed by the SHA-256 hash of the *sorted-keys
JSON* of the `robot` dict (`cr4_kkt.py:71-73`, `259-263`):

```python
def get_robot_hash(robot: dict) -> str:
    serialized = json.dumps(robot, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
```

The cache stores the **already-built** Pinocchio model, constraint models,
and constraint datas. Building the model is by far the most expensive step
(cold start ~30 s on a typical workstation because of the CR4 closed-chain
constraint assembly); caching it means subsequent requests for the *same*
robot are nearly free. The cache is process-local, lives for the lifetime
of the FastAPI process, and is invalidated on restart.

In the batch endpoint, the cache is checked once at the top of
`compute_cr4_kkt_batch` and then the per-sample loop reuses the cached
model (`cr4_kkt.py:482-514`).

### 4.2 Per-batch trajectory hash (in the response manifest)

`backend/api/dynamics.py:get_trajectory_hash` (lines 33-51) computes a
SHA-256 over the **cleaned** sample list (each sample reduced to
`time_s, q, qd, qdd` as plain floats, sorted JSON) and stores it on the
returned manifest:

```python
manifest = make_manifest(robot_dict, "cr4_pinocchio_kkt.v1",
                         get_trajectory_hash(samples))
```

This hash is the audit trail of *exactly which* trajectory was evaluated.
It is included on the `actuator_sizing_report.v1` envelope so reviewers
can reproduce a sizing report from its `torque_log_hash` plus the
`trajectory_hash` field of the original dynamics manifest.

### 4.3 Frontend-side cache

`src/api/backend.ts:fetchWithTimeout` uses a single
`AbortController` per request with an 800 ms budget for health probes
(`src/api/backend.ts:14-32`). PRO `/api/dynamics/batch` requests use a
**120 s** budget because the first call may pay the 30 s cold-start cost
(`src/api/backend.ts:111`). The active URL is mutable via
`setApiBackendUrl` and falls back from `127.0.0.1` to `localhost` after
`consecutiveBackendFailures` exceeds the threshold tracked in
`src/model/state.ts`.
