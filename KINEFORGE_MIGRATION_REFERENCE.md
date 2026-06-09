# Referencia Tecnica Kineforge Para Migracion A Robodimm

Fecha: 2026-06-04

Este documento resume el estado funcional de Kineforge relevante para portar una parte a Robodimm frontend. Rutas relativas desde `robodimm/`: `../kineforge/...`.

## Principios Que Hay Que Conservar

- SI internamente: metros, radianes, segundos, N m.
- UI puede mostrar grados o mm, pero debe convertir al estado SI.
- Frames CAD para usuario; frames DH/Pinocchio internos ocultos.
- Payload como cuerpo extra soldado al EE/TCP.
- Separar `Play` visual de trayectoria oficial de dimensionamiento.
- Par RMS/continuo y par pico/transitorio son metricas diferentes.
- Conversiones q-space centralizadas, nunca inline.

## Mapa De Carpetas Kineforge Relevantes

- `../kineforge/core/templates/serial6.py`: template CR6/IRB4600, FK DH, IK, Jacobiano, inverse dynamics, payload.
- `../kineforge/core/templates/palletizer.py`: template CR4/IRB460, hardpoints, FK, payload.
- `../kineforge/core/kinematics/serial6_template_backend.py`: frames CAD `LINK1..LINK6`, frames internos `link_1..link_6`, snapshots UI.
- `../kineforge/core/kinematics/palletizer_template_backend.py`: backend FK del template palletizer.
- `../kineforge/core/kinematics/cr4_palletizer_backend.py`: backend CR4 geometrico.
- `../kineforge/core/kinematics/ik_solvers/cr4_palletizer.py`: IK geometrica CR4.
- `../kineforge/core/dynamics/torque.py`: `TorqueLog`, inverse dynamics trajectory, dispatch serial6 template, metricas.
- `../kineforge/core/dynamics/trajectory.py`: trayectoria suave de programa para sizing.
- `../kineforge/core/dynamics/inertial_view.py`: lectura/escritura de inertiales y payload.
- `../kineforge/core/actuators/schema.py`: schema de motores/reductoras.
- `../kineforge/core/actuators/selection.py`: seleccion de actuadores.
- `../kineforge/core/program/models.py`: targets e instrucciones.
- `../kineforge/core/program/playback.py`: play visual, MoveJ/MoveL/Pause, zone.
- `../kineforge/core/model_loading/visuals_overrides.py`: overrides de meshes, scale, frame_name.
- `../kineforge/app/ui/main_window.py`: UI Qt actual, inspector, docks, CAD frames, COM markers.
- `../kineforge/app/ui/body_mesh_dialog.py`: dialogo de mesh, escala y origin.
- `../kineforge/app/controllers/main_window_controller.py`: orquestacion UI, Jog, Program, torque recording, actuators.
- `../kineforge/viz/pyvista_view/widget.py`: viewport PyVista, equivalente conceptual para Three.js.

## CR6 / IRB4600: Serial6Template

Ruta principal: `../kineforge/core/templates/serial6.py`

### Datos

Clases importantes:

- `DHJointSpec`: nombre, `a_m`, `alpha_rad`, `d_m`, `theta_offset_rad`, limites, inertial.
- `SerialInertialSpec`: `mass_kg`, `com_m`, `inertia_kg_m2`, `frame`.
- `Serial6Template`: seis joints DH, tool transform, gravedad y payload.
- `Serial6FK`: `joint_origins`, `joint_axes`, `joint_body_transforms`, `link_transforms`, `tcp_transform`.

Schema actual en `template.json`:

- `schema_version`: `serial6_dh.v1`
- `joints`: seis joints con DH e inertial.
- `payload`: inertial extra, por defecto 15 kg en IRB4600.

### FK

Funcion: `Serial6Template.forward_kinematics(q)`.

Convencion DH implementada por `_standard_dh(a_m, alpha_rad, d_m, theta_rad)`:

```text
T = RotZ(theta) * TransX(a) * RotX(alpha) * TransZ(d)
```

En cada joint:

- `origin`: posicion del eje articular antes de aplicar el joint actual.
- `axis`: eje Z actual en mundo.
- `joint_body_transform`: frame tras `RotZ(theta)`, usado para CAD/inertial conversion.
- `link_transform`: frame tras DH completo.

Para Robodimm: portar literalmente el algebra de `_standard_dh`, `forward_kinematics`, `geometric_jacobian` y helpers de rotacion.

### CAD Frames

Ruta: `../kineforge/core/kinematics/serial6_template_backend.py`

En `forward_kinematics()`:

- `LINK1..LINK6`: frames CAD user-facing.
- `link_1..link_6`: frames internos/link/DH.
- `dh_1..dh_6`: alias debug.
- `EE_frame`, `payload_link`, `PAYLOAD`.

Funcion clave:

```python
def _cad_aligned_frame(transform, home_transform):
    cad_transform = transform.copy()
    cad_transform[:3, :3] = transform[:3, :3] @ home_transform[:3, :3].T
    return cad_transform
```

Efecto: en home, el frame CAD queda alineado con mundo; durante movimiento rota relativo a home. Esta es la semantica que debe ver el usuario para disenar CAD.

### Jacobiano

Funcion: `Serial6Template.geometric_jacobian(q, point=None)`.

Formula:

- `Jv[:, i] = axis_i x (tip - origin_i)`
- `Jw[:, i] = axis_i`

Usos:

- Jog cartesiano.
- Velocidad TCP cinematica.
- Inverse dynamics por Jacobianos.

### IK

Funcion principal: `Serial6Template.solve_spherical_wrist_ik(...)`.

Notas:

- Pensada para geometria de muneca esferica.
- Usa validaciones de geometria soportada.
- Devuelve `Serial6IKResult`: success, q, errores, branch, message.

Para Robodimm: portar primero FK/Jacobiano y dejar IK geometrica como modulo CR6 dedicado. Si hay dudas, usar backend Python opcional mientras se valida el port.

### Inverse Dynamics CR6

Funcion: `Serial6Template.inverse_dynamics(q, qd, qdd)`.

Puntos importantes:

- Usa FK DH real, no URDF.
- Usa gravedad `gravity_m_s2`.
- Construye Jacobianos por COM de cada cuerpo.
- Incluye inercia rotacional.
- Convierte inertials CAD a link frame para cuerpos serial6 cuando toca.
- Incluye `payload` como body extra en `fk.tcp_transform`.

Bug corregido en Kineforge: `simulate_inverse_dynamics_trajectory` no debe usar `pin.rnea` con el URDF template serial6, porque ese URDF no tiene geometria DH real. Ahora despacha a `Serial6Template.inverse_dynamics` si detecta `serial6_dh.v1`.

Ruta relacionada: `../kineforge/core/dynamics/torque.py`.

## CR4 / IRB460: PalletizerTemplate

Rutas:

- `../kineforge/core/templates/palletizer.py`
- `../kineforge/core/kinematics/palletizer_template_backend.py`
- `../kineforge/core/kinematics/cr4_palletizer_backend.py`
- `../kineforge/core/kinematics/ik_solvers/cr4_palletizer.py`

### Datos

Elementos relevantes:

- Hardpoints en plano XZ, Y=0.
- `PalletizerInertialSpec` y payload.
- `PalletizerTemplate.forward_kinematics(...)`.
- `irb460_palletizer_template()` para defaults.
- `cube_payload_inertial(mass_kg, side_m=0.03)`.

### FK/IK

Backends:

- `PalletizerTemplateKinematicsBackend.forward_kinematics` genera snapshot compatible con UI.
- `Cr4PalletizerKinematicsBackend.forward_kinematics` implementa backend geometrico CR4.
- `Cr4PalletizerIK.solve_ik` resuelve IK especifica.

Para Robodimm: portar CR4 como motor separado. No intentar meter CR4 y CR6 en un modelo generico excesivo al inicio.

## Programacion: Targets, MoveJ, MoveL, Pause

Rutas:

- `../kineforge/core/program/models.py`
- `../kineforge/core/program/playback.py`
- `../kineforge/core/program/storage.py`

Datos:

- `ProgramTarget`: `name`, `q`, `joint_values`, `tcp_pose` opcional.
- `MoveJInstruction`: target, `speed_rad_s`, `zone_m`.
- `MoveLInstruction`: target, `tcp_speed_m_s`, `zone_m`.
- `PauseInstruction`: `duration_s`.
- `RobotProgram`: targets e instructions.

Playback:

- `ProgramPlayer.build_playback(...)` produce samples para visualizacion.
- `MoveJ`: interpolacion lineal q-space por velocidad articular.
- `MoveL`: interpola cartesiano si hay backend+IK; si no, fallback joint-space.
- `zone_m`: aplicado como distancia cartesiana en runs MoveL; no aplicar como radio articular.

Para Robodimm: `Play` debe usar esta idea, pero no debe ser la fuente oficial de torques.

## Trajectory Recording Para Dynamics

Ruta: `../kineforge/core/dynamics/trajectory.py`

Funciones:

- `build_program_dynamics_trajectory(...)`
- `build_program_sizing_trajectory(...)`
- `write_joint_trajectory_csv(...)`

Semantica:

- Trayectoria oficial para dimensionamiento.
- Usa time scaling quintico para suavidad.
- Produce `q`, `qd`, `qdd`, `time_s`, `instruction_indices`.
- `MoveL` usa duracion por distancia TCP si hay descriptor/backend.

Para Robodimm: implementar `Signal Recording` sobre esta semantica, no sobre el Play visual.

## TorqueLog Y Metrics

Ruta: `../kineforge/core/dynamics/torque.py`

Clases:

- `TorqueSample`: time, q, velocity, acceleration, joint_velocity, joint_acceleration, tau.
- `TorqueLog`: samples, joint_names, dt.

Metodos:

- `time_vector()`.
- `torque_matrix()`.
- `velocity_matrix()`.
- `peak_abs_velocity_by_joint()`.
- `peak_abs_by_joint()`.
- `sizing_metrics(peak_fraction=0.8)`.

Metricas de sizing:

- `torque_rms_Nm`
- `torque_peak_Nm`
- `peak_duration_s`
- `speed_peak_rad_s`
- `power_peak_abs_W`
- `power_rms_W`
- `mechanical_energy_abs_J`

Para Robodimm: replicar estas metricas en frontend para Actuators.

## Inertials Y Payload

Ruta: `../kineforge/core/dynamics/inertial_view.py`

Funciones:

- `get_inertial_view(descriptor)`.
- `updated_inertial_value(row, field, index, value)`.
- `update_inertial(row, package_root)`.

Conceptos:

- Lee inertials desde template o URDF.
- Para serial6, inertials pueden estar en `frame: "cad"`.
- Al escribir a URDF dinamico, convierte CAD a link cuando corresponde.
- Payload aparece como row `PAYLOAD`.
- Payload serial6 se guarda en `template.json.payload`.
- Payload URDF se refleja como `payload_link` fijo hijo de `EE_frame`.

Defaults relevantes:

- CR6/IRB4600: payload 15 kg, cubo 30 mm, inertia diagonal 0.00225 kg m2.
- CR4/IRB460: payload 50 kg, cubo 30 mm, inertia diagonal 0.0075 kg m2.

Para Robodimm: no hace falta URDF para frontend, pero si mantener `payload` como cuerpo dinamico extra soldado al TCP.

## Actuators

Rutas:

- `../kineforge/core/actuators/schema.py`
- `../kineforge/core/actuators/storage.py`
- `../kineforge/core/actuators/selection.py`

Schema:

- `MotorSpec`: potencia, par nominal, par stall, velocidad, constantes electricas, inercia rotor, masa, voltage, dimensiones.
- `GearboxSpec`: ratio, torque continuo/intermitente, eficiencia, input speed, backlash, masa, inercia.
- `ActuatorLibrary`: motors, gearboxes, compatibility matrix, metadata.

Seleccion:

- `select_actuators_for_torque_log(library, torque_log, ...)`.
- Demanda por joint: RMS como continuo, max abs como pico, max velocidad.
- Ranking por masa total, potencia motor, ratio.
- Margenes: continuo, pico, velocidad.
- Overload reductora configurable.

Para Robodimm: la tabla editable de actuadores puede usar el mismo schema en JSON.

## Visuals, Meshes Y CAD Frames

Rutas:

- `../kineforge/core/model_loading/visuals_overrides.py`
- `../kineforge/core/model_loading/urdf_visuals.py`
- `../kineforge/app/ui/body_mesh_dialog.py`
- `../kineforge/viz/pyvista_view/widget.py`

Estado actual:

- `visuals.yaml` guarda override por `link_name`.
- `VisualElement` incluye `link_name` y `frame_name` opcional.
- Para IRB4600, override de body `LINK2` conserva `link_name="link_2"` para validacion, pero `frame_name="LINK2"` para render CAD.
- Viewport usa `visual.frame_name` si existe, si no usa `visual.link_name`.
- `suggest_default_mesh_scale` detecta STL binario grande y propone escala `(0.001, 0.001, 0.001)`.

Leccion importante:

- GLB en metros y frame CAD es el formato recomendado.
- STL puede venir en mm y en frame de pieza SolidWorks, no en frame CAD.
- No asumir que STL + 0.001 corrige origen/orientacion.

## UI Qt Actual Como Referencia Conceptual

Ruta principal: `../kineforge/app/ui/main_window.py`

Funciones/zonas a consultar:

- `IRB460_SOLIDS` y `IRB4600_SOLIDS`: nombres user-facing de solidos.
- `_populate_inertials`: tabla de inerciales.
- `_populate_inspector`: inspector con solidos, COM, CAD frames.
- `_com_marker_entries`: marker de COM en frame correcto.
- `_display_frame_for_inertial_row`: mapeo inertial row a frame visual.
- `_build_solids_tree_item`: visibilidad y override de meshes.
- `_build_frame_tree_item`: arbol de frames.

Ruta controller: `../kineforge/app/controllers/main_window_controller.py`

Funciones/zonas:

- `load_body_mesh`, `update_body_mesh_override`, `remove_body_mesh_override`.
- `update_inertial_value`.
- Program save/load/add target/instructions.
- Recording de torques alrededor de `build_program_sizing_trajectory` y `simulate_inverse_dynamics_trajectory`.
- Actuator selection.

Para Robodimm: no portar Qt, pero si copiar la estructura conceptual de paneles y flujos.

## Viewport Actual Como Referencia Para Three.js

Ruta: `../kineforge/viz/pyvista_view/widget.py`

Conceptos:

- `_initialize_scene`: crea actores visuales, puntos joint, frame axis, COM markers.
- `_update_scene`: aplica `frame_pose.transform @ visual.origin_transform`.
- `_build_visual_mesh`: primitivas o mesh.
- `_apply_visual_visibility` y `_apply_com_visibility`.
- `_add_all_frame_axes`: ejes de frames.

Para Three.js:

- Mesh Object3D por visual.
- Parent lógico por frame o actualizar matrixWorld por frame pose.
- CAD frames como `AxesHelper` o custom triads.
- COM markers como small spheres.
- Solidos con material PBR sobrio.

## Archivos De Robot Kineforge Actuales

CR6:

- `../kineforge/robot_library/irb4600_serial6_template/template.json`
- `../kineforge/robot_library/irb4600_serial6_template/model.urdf`
- `../kineforge/robot_library/irb4600_serial6_template/semantics.yaml`
- `../kineforge/robot_library/irb4600_serial6_template/meshes/`

CR4:

- `../kineforge/robot_library/irb460_palletizer_template/template.json`
- `../kineforge/robot_library/irb460_palletizer_template/model.urdf`
- `../kineforge/robot_library/irb460_palletizer_template/semantics.yaml`

Actuators:

- Buscar biblioteca en `../kineforge` con `core/actuators/storage.py` y tests. El schema reutilizable esta en `core/actuators/schema.py`.

## Validacion Externa

Kineforge usa scripts en `../ensayos/` para comparacion con Simscape/Pinocchio. Robodimm debe exportar un manifiesto suficiente para que un script externo replique:

- Robot kind y parametros.
- Hardpoints/DH.
- Inertials y payload.
- Programa/trajectory.
- Senales q/qd/qdd/tau.
- Engine usado.

No hacer que Robodimm dependa de MATLAB. Robodimm exporta datos; scripts externos validan.

## Pitfalls Ya Detectados En Kineforge

- `pin.rnea` con URDF serial6 template daba torques de gravedad cero porque el URDF no codifica geometria DH real.
- El payload debe afectar inverse dynamics; comprobar cambiando de 15 kg a 115 kg en CR6 y verificando aumento de J2/J3/J5.
- `np.gradient(position, time)` para velocidad TCP puede meter ruido; usar velocidad cinematica `J[:3] @ qd` o senales directas.
- `zone_m` es distancia cartesiana, no radio articular.
- PySide6 puede segfaultear con `model.setData(index, float, Qt.EditRole)`; irrelevante para Robodimm pero documenta que UI frameworks pueden tener trampas de tipo.
- STL puede venir 1000x grande y con origen de pieza, no CAD frame.

## Checklist De Port Inicial

- Portar tipos `RobotSpec`, `InertialSpec`, `VisualSpec`, `ProgramSpec`, `TorqueLog`, `ActuatorLibrary`.
- Portar CR6 FK/Jacobian/inverse dynamics y payload.
- Portar CR4 FK/IK.
- Implementar CAD frame map para CR4 y CR6.
- Implementar viewer con primitives primero.
- Implementar Editor con CR6 default y boton `Set`.
- Implementar Jog articular y Save Target.
- Implementar Program + Play.
- Implementar Signal Recording y export CSV/JSON.
- Implementar Actuators desde TorqueLog.
- Agregar backend Python opcional solo despues de tener frontend validado.
