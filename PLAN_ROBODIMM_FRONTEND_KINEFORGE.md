# Plan De Trabajo: Robodimm Frontend A Partir De Kineforge

Fecha: 2026-06-04

Objetivo: evolucionar parte de Kineforge hacia una version Robodimm con fuerte componente frontend, centrada solo en dos familias de robots configurables:

- CR4: robot paletizador 4 DoF basado en la plantilla IRB460 de Kineforge.
- CR6: robot serial 6 DoF basado en la plantilla IRB4600 de Kineforge.

No se pretende portar todo Kineforge. Quedan fuera control hardware generico, otros robots, mecanismos 2D, adaptadores ROS/hardware no necesarios, Qt/PyVista y funcionalidades de laboratorio que no aporten al simulador Robodimm.

## Resultado Esperado

Robodimm debe ser una aplicacion web con:

- Frontend JS/TypeScript + Three.js como experiencia principal.
- Motores FK, IK e inverse dynamics disponibles en frontend para CR4 y CR6.
- Backend Python opcional para calculos locales con Pinocchio/Pink y validacion avanzada.
- Flujo de configuracion explicito: el usuario edita/carga un robot y debe pulsar `Set` para fijarlo como robot activo.
- Registro de senales reproducible: `Play` solo visualiza; `Signal Recording` genera trayectoria suave, pares, CSV y manifiesto JSON/YAML para validacion externa.

## Alcance Funcional

### Incluido

- Editor parametrico de CR4 y CR6.
- Hardpoints/plano XZ para CR4 y CR6 cuando proceda.
- Parametros inerciales editables: masa, COM, matriz de inercia y payload.
- Geometria basica y mesh CAD por eslabon.
- Elementos de estacion: mesas, estanterias, fixtures u otros meshes estaticos.
- CAD frames user-facing para diseno y visualizacion.
- Jog articular y cartesiano.
- Programacion con targets, `MoveJ`, `MoveL`, `Pause`, velocidades y zonas.
- Play visual.
- Signal recording con CSV y manifiesto de validacion.
- Dimensionamiento de actuadores desde pares RMS/continuos y pico/transitorios.
- Tabla editable de actuadores basada en `actuators.json`.

### Excluido En La Primera Version

- Control hardware real.
- Robots genericos fuera de CR4/CR6.
- Pink como motor frontend. Pink queda para backend Python opcional.
- PyVista/Qt. El visor sera Three.js.
- Export ROS/CODESYS salvo que se defina posteriormente.
- Emulacion exacta RobotStudio/RAPID. La prioridad es dimensionamiento robusto.

## Arquitectura Propuesta

### Capas

1. `frontend/model`
Datos canonicos del robot, programa, actuadores, meshes y estacion.

2. `frontend/math`
FK, IK, Jacobianos, trayectoria, inverse dynamics, conversiones de unidades y q-space.

3. `frontend/viewer`
Three.js: robot, meshes, CAD frames, COM markers, trayectoria, estacion y controles de camara.

4. `frontend/ui`
Panel derecho con pestanas, panel izquierdo inspector y estado global.

5. `backend-python` opcional
Servicios Pinocchio/Pink/validacion externa. Debe ser opcional, nunca requisito para usar el simulador basico.

### Estado Canonico

Mantener un unico estado activo equivalente a:

```ts
type ActiveRobot = {
  kind: "CR4" | "CR6";
  robotSpec: RobotSpec;
  inertials: InertialSpec[];
  payload: InertialSpec;
  visuals: VisualSpec[];
  station: StationObject[];
  engines: EngineSelection;
  q: number[];
  frames: FramePoseMap;
  comMarkers: ComMarker[];
};
```

Regla importante: todos los calculos internos en SI: metros, radianes, segundos, N m. La UI puede mostrar grados y mm, pero debe convertir al escribir al estado.

## Layout De Producto

### Centro: Visor 3D Three.js

Objetivo visual: limpio, moderno, cercano en lenguaje visual a Isaac Sim pero simple.

Requisitos:

- Fondo oscuro sobrio, grid configurable, luces suaves, materiales no estridentes.
- Visualizacion de robot activo con meshes o primitivas.
- CAD frames visibles/ocultables.
- COM markers visibles/ocultables.
- Solidos visibles/ocultables por eslabon.
- Elementos de estacion visibles/ocultables.
- Trayectoria/programa opcionalmente dibujado.
- Gizmos de ejes con X derecha, Z arriba, Y profundidad.

### Lateral Izquierdo: Inspector

El inspector debe mostrar solo informacion util para el usuario y CAD.

Contenidos:

- Arbol de solidos: `BASE`, `LINK1`...`LINK6`, `PAYLOAD` o equivalentes CR4.
- Checkboxes de visibilidad por solido.
- Arbol de CAD frames: mostrar/ocultar cada frame.
- Coordenadas y orientacion de cada CAD frame.
- Centros de masa: visibilidad global y por cuerpo.
- Elementos de estacion.
- Estado resumido del robot activo: tipo, DoF, motores de calculo activos.

Regla critica: la UI debe hablar siempre en frames CAD/user-facing. Los frames DH/Pinocchio/backend quedan ocultos salvo modo debug.

### Lateral Derecho: Pestanas

#### 1. Editor

Debe abrirse por defecto. Debe haber un robot por defecto, recomendado `CR6` con parametros IRB4600 actuales de Kineforge.

Secciones:

1. Tipo de robot
- Select: `CR4` o `CR6`.
- Al cambiar, recargar defaults del tipo seleccionado.

2. Hardpoints/dimensiones
- Para CR4: hardpoints paletizador en plano XZ, Y=0 por simetria.
- Para CR6: parametros DH/geom equivalentes al template IRB4600; se pueden presentar como hardpoints CAD si resulta mas amigable.
- Defaults cargados desde Kineforge.
- Campos editables con unidades claras.

3. Inerciales
- Tabla por body/link: masa, COM, Ixx, Iyy, Izz, Ixy, Ixz, Iyz.
- Payload como cuerpo extra soldado al TCP/EE.
- Defaults de Kineforge para CR4/IRB460 y CR6/IRB4600.

4. Geometrias del robot
- Modo primitivas: cilindros, esferas, prismas.
- Modo CAD mesh por eslabon.
- Meshes en frame CAD. Preferir GLB en metros y frame correcto.
- STL permitido solo con controles explicitos de escala y offset, porque puede venir en mm y/o frame de pieza SolidWorks.

5. Estacion
- Subida de meshes estaticos: mesa, estanteria, fixture.
- Transform global editable: posicion, orientacion, escala.

6. Motores de calculo
- Frontend CR4 FK/IK/ID.
- Frontend CR6 FK/IK/ID.
- Backend Python opcional: Pinocchio/Pink si esta disponible.
- Indicador de motor activo y fallback.

7. Persistencia
- `Load`: cargar robot/especificacion.
- `Save`: guardar robot actual.
- `Set`: fija el robot activo. Obligatorio para habilitar Jog, Program, Actuators y Recording.

Regla de UX: hasta pulsar `Set`, mostrar banner claro: `Robot no fijado. Pulsa Set para simular y registrar senales.`

#### 2. Jog

Contenidos:

- Sliders articulares en grados.
- Campos numericos sincronizados.
- Botones de jog articular incremental.
- Movimiento cartesiano del TCP.
- Reorientacion del TCP.
- Referencia `World/Base` o `TCP` para movimientos avanzados.
- Boton `Save Target` para programacion rapida.

Reglas:

- Internamente q en radianes.
- Conversiones q-space centralizadas, nunca inline.
- CR4 y CR6 pueden tener IK geometrica dedicada en frontend.
- Si el IK falla, no mover estado y mostrar causa.

#### 3. Program

Contenidos:

- Tabla/lista de targets.
- Editor de instrucciones: `MoveJ`, `MoveL`, `Pause`.
- Argumentos: target, speed, zone.
- `Load Program`, `Save Program`.
- `Play`: solo visualizacion, no dimensionamiento.
- `Signal Recording`: trayectoria suave, pares, metricas y export.

`Signal Recording` debe generar:

1. CSV de senales
- `time_s`
- `q1..qn`
- `qd1..qdn`
- `qdd1..qddn`
- `tau1..taun`
- opcional: TCP position, TCP velocity, instruction_index

2. Manifiesto JSON/YAML
- Tipo robot y version de schema.
- Hardpoints/parametros DH.
- Inertiales y payload.
- Programa y targets.
- Configuracion de motores de calculo.
- Configuracion visual/estacion si aporta a reproducibilidad.
- Unidades.
- Version de Robodimm.
- Hash/metadata de meshes si procede.

3. Estado interno para graficas y Actuators
- `TorqueLog` equivalente.
- RMS/continuo, pico/transitorio, duracion de pico, velocidad pico, potencia, energia.

#### 4. Actuators

Contenidos:

- Carga por defecto desde `actuators.json`.
- Tabla editable de motores, reductoras y matriz de compatibilidad.
- Botones: seleccionar, recalcular, guardar biblioteca, importar/exportar.
- Resultados por articulacion: demanda continua, demanda pico, velocidad pico, candidatos, margenes.

Reglas:

- No dimensionar desde `Play`. Dimensionar desde el ultimo `Signal Recording` valido.
- Separar par RMS/continuo y par pico/transitorio.
- Margenes configurables: continuo, pico, velocidad y overload de reductora.

## Modelo De Datos Recomendado

### RobotSpec

```ts
type RobotSpec = {
  schema: "robodimm.robot.v1";
  kind: "CR4" | "CR6";
  name: string;
  units: "SI";
  geometry: Cr4Geometry | Cr6Geometry;
  inertials: Record<string, InertialSpec>;
  payload: InertialSpec;
  visuals: VisualSpec[];
  station: StationObject[];
  limits: JointLimit[];
};
```

### InertialSpec

```ts
type InertialSpec = {
  body: string;
  massKg: number;
  comM: [number, number, number];
  inertiaKgM2: [[number, number, number], [number, number, number], [number, number, number]];
  frame: "cad" | "link" | "tcp";
};
```

### VisualSpec

```ts
type VisualSpec = {
  body: string;
  frameName: string;
  kind: "primitive" | "mesh";
  meshUrl?: string;
  primitive?: PrimitiveSpec;
  originM: [number, number, number];
  rpyRad: [number, number, number];
  scale: [number, number, number];
  visible: boolean;
};
```

### ProgramSpec

```ts
type ProgramSpec = {
  schema: "robodimm.program.v1";
  name: string;
  targets: ProgramTarget[];
  instructions: ProgramInstruction[];
};
```

## CR4: Linea Tecnica

Base en Kineforge:

- `../kineforge/core/templates/palletizer.py`
- `../kineforge/core/kinematics/palletizer_template_backend.py`
- `../kineforge/core/kinematics/cr4_palletizer_backend.py`
- `../kineforge/core/kinematics/ik_solvers/cr4_palletizer.py`

Tareas:

1. Definir schema JS de hardpoints CR4 en plano XZ.
2. Portar FK geometrica y frames CAD.
3. Portar IK geometrica CR4.
4. Portar Jacobiano si se necesita Jog cartesiano diferencial.
5. Portar inverse dynamics si se usa modelo aproximado frontend.
6. Mantener q-space claro: frontend/user, motor matematico, backend opcional.

## CR6: Linea Tecnica

Base en Kineforge:

- `../kineforge/core/templates/serial6.py`
- `../kineforge/core/kinematics/serial6_template_backend.py`
- `../kineforge/core/dynamics/torque.py`

Tareas:

1. Portar `DHJointSpec`, `SerialInertialSpec`, `Serial6Template` a TS/JS.
2. Portar FK standard DH.
3. Portar CAD frames: `LINK1..LINK6` con orientacion user-facing.
4. Portar Jacobiano geometrico.
5. Portar IK de muneca esferica si se conserva la geometria compatible.
6. Portar inverse dynamics por Jacobianos que incluye payload.
7. No usar URDF cero-geometria para dynamics frontend.

Nota critica: en Kineforge se corrigio que `simulate_inverse_dynamics_trajectory` use `Serial6Template.inverse_dynamics` para templates serial6, porque el URDF template no codifica la geometria DH real.

## Frames CAD Y Meshes

Regla central: lo que ve el usuario y lo que se documenta para SolidWorks son CAD frames.

Para CR6:

- `LINK1..LINK6` son frames CAD/user-facing.
- En home, su orientacion debe ser mundo-alineada: X derecha, Z arriba, Y profundidad.
- Internamente pueden existir `link_1..link_6` o `dh_1..dh_6`, pero no deben contaminar UI normal.

Para meshes:

- Preferir GLB en metros y ya alineado con CAD frame.
- STL suele venir en mm. Detectar escala 0.001 si el bounding box es grande.
- STL exportado desde SolidWorks puede venir en frame de pieza, no en frame CAD. No asumir que `STL + scale 0.001` basta.
- Ofrecer controles de offset/orientacion por mesh.
- Documentar para el usuario el frame esperado antes de importar CAD.

## Signal Recording: Semantica

`Play`:

- Interpola y visualiza.
- Puede usar camino mas simple o tiempo de UI.
- No genera torques oficiales.

`Signal Recording`:

- Construye trayectoria suave reproducible, preferiblemente quintica.
- Calcula `q`, `qd`, `qdd` analiticos o por formula estable.
- Calcula inverse dynamics con robot activo y payload actual.
- Guarda CSV y manifiesto.
- Actualiza graficas y datos de dimensionamiento.

## Plan Por Fases

### Fase 0: Inventario Y Decision De Stack

- Confirmar stack: Vite/React/Three.js o framework actual Robodimm.
- Definir si Robodimm sera TypeScript estricto.
- Crear schemas `robodimm.robot.v1`, `robodimm.program.v1`, `robodimm.recording.v1`, `robodimm.actuators.v1`.
- Decidir formato de persistencia: JSON principal; YAML solo si aporta legibilidad.

### Fase 1: Core JS CR6

- Portar FK DH, CAD frames, Jacobiano e IK CR6.
- Portar inverse dynamics CR6 con payload.
- Tests numericos contra Kineforge para home, varios q y payloads.

### Fase 2: Core JS CR4

- Portar hardpoints y FK/IK CR4.
- Tests numericos contra Kineforge.
- Definir inverse dynamics frontend o puente backend.

### Fase 3: Viewer Three.js

- Robot con primitivas.
- CAD frames y COM markers.
- Visibilidad por solido.
- Mesh import por link y station objects.

### Fase 4: Editor Y Set Robot

- Defaults CR6 al iniciar.
- Cambio CR4/CR6.
- Tablas hardpoints, inertials, payload, visuals, station.
- Load/Save/Set.

### Fase 5: Jog

- Sliders articulares.
- Jog cartesiano y orientacion.
- Save target.
- World/TCP reference.

### Fase 6: Program Y Recording

- Targets e instrucciones.
- Play visual.
- Signal Recording con CSV, manifiesto y datos internos.
- Graficas basicas.

### Fase 7: Actuators

- Biblioteca editable.
- Seleccion por torque log.
- Reporte de margenes.

### Fase 8: Backend Python Opcional

- Endpoint FK/IK/ID Pinocchio/Pink.
- Comparacion frontend vs backend.
- Export para Simulink/Matlab fuera de Robodimm.

## Criterios De Aceptacion

- Al abrir, CR6 default queda visible con parametros IRB4600.
- Sin pulsar `Set`, Jog/Program/Actuators no operan o muestran aviso.
- Tras `Set`, Jog articular mueve robot y frames correctamente.
- `Save Target` crea target reproducible.
- Program con `MoveJ` y `MoveL` reproduce trayectorias.
- `Signal Recording` cambia torques al cambiar payload.
- Actuators solo usa el ultimo recording valido.
- Meshes GLB en CAD frame se colocan correctamente.
- STL grande se escala automaticamente a 0.001, pero conserva controles de offset.
- Inspector oculta internals DH por defecto.

## Riesgos Tecnicos

- Confusion grados/radianes.
- Confusion metros/mm en CAD imports.
- Confusion CAD frame vs DH frame vs Pinocchio frame.
- q-space CR4/CR6 si existen convenciones de Robodimm previas.
- Inverse dynamics incorrecta si se usa un modelo visual en vez del modelo matematico real.
- Program playback no debe confundirse con trajectory recording para dimensionamiento.
- Meshes de SolidWorks pueden estar en frame de pieza, no frame CAD.

## Decisiones Iniciales Recomendadas

- TypeScript para core frontend.
- SI en estado interno.
- CAD frames como unico contrato con usuario.
- `Signal Recording` como fuente oficial de torques y actuadores.
- Defaults CR6/IRB4600 cargados al inicio.
- CR4 y CR6 implementados como motores separados, no como robot generico abstracto excesivo.
