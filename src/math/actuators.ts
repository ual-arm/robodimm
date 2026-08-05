import { ActuatorLibrary, TorqueLog, SizingMargins, JointDemand, ActuatorCandidate, MotorSpec, GearboxSpec, ActuatorSizingReport, GearboxType, JointActuatorSelection, RobotActuatorSelection, SizingObjective } from '../model/schemas';
import { canonicalSha256 } from './provenance';

type MechanicalConstraint = 'M1' | 'M2' | 'M3' | 'M4' | 'M5' | 'M6';

const RAD_S_TO_RPM = 60.0 / (2.0 * Math.PI);
const MECHANICAL_CONSTRAINTS: MechanicalConstraint[] = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6'];

/**
 * Computes time-weighted RMS and peak joint demands from the simulation TorqueLog.
 * RMS values are calculated using non-uniform time delta intervals to ensure physical correctness:
 * RMS = sqrt( integral(x^2 dt) / T )
 */
export function computeJointDemands(torqueLog: TorqueLog): JointDemand[] {
  const demands: JointDemand[] = [];
  const jointNames = torqueLog.joint_names;
  const numSamples = torqueLog.samples.length;
  if (numSamples === 0) return [];

  // The first sample is an endpoint, not a synthetic interval.  Invalid time
  // bases invalidate the complete demand rather than silently dropping an
  // interval and producing a procurement-like result from partial data.
  const timeValid = numSamples >= 2 && torqueLog.samples.every((sample, index) => {
    if (!Number.isFinite(sample.time_s)) return false;
    return index === 0 || sample.time_s > torqueLog.samples[index - 1].time_s;
  });

  for (let j = 0; j < jointNames.length; j++) {
    let sumTauSq = 0;
    let sumSpeedSq = 0;
    let sumPowerSq = 0;
    let peakTau = 0;
    let peakSpeed = 0;
    let peakPower = 0;
    let peakRegen = 0;
    let valuesValid = true;
    const valueAt = (sampleIndex: number) => {
      const sample = torqueLog.samples[sampleIndex];
      const tauValue = sample.tau[j];
      const velocityValue = (sample.joint_velocity ?? sample.velocity ?? [])[j];
      const valid = Number.isFinite(tauValue) && Number.isFinite(velocityValue);
      valuesValid = valuesValid && valid;
      const tau = valid ? tauValue : 0;
      const velocity = valid ? velocityValue : 0;
      return { tau, velocity, power: tau * velocity };
    };

    // Peaks are retained even when the time base is incomplete or invalid.
    for (let s = 0; s < numSamples; s++) {
      const { tau, velocity, power } = valueAt(s);
      peakTau = Math.max(peakTau, Math.abs(tau));
      peakSpeed = Math.max(peakSpeed, Math.abs(velocity));
      peakPower = Math.max(peakPower, Math.abs(power));
      if (power < 0) peakRegen = Math.max(peakRegen, -power);
    }

    // Evaluate every sample before deciding whether RMS integration is valid.
    const sampleValues = torqueLog.samples.map((_sample, index) => valueAt(index));

    let cycleTime = 0;
    if (timeValid && valuesValid) {
      cycleTime = torqueLog.samples[numSamples - 1].time_s - torqueLog.samples[0].time_s;
      for (let s = 1; s < numSamples; s++) {
        const dt = torqueLog.samples[s].time_s - torqueLog.samples[s - 1].time_s;
        const previous = sampleValues[s - 1];
        const current = sampleValues[s];

        // Time-weighted trapezoidal integration of x^2, including both
        // endpoints of each interval.
        sumTauSq += 0.5 * (previous.tau ** 2 + current.tau ** 2) * dt;
        sumSpeedSq += 0.5 * (previous.velocity ** 2 + current.velocity ** 2) * dt;
        sumPowerSq += 0.5 * (previous.power ** 2 + current.power ** 2) * dt;
      }
    }

    const tau_rms_Nm = cycleTime > 0 ? Math.sqrt(sumTauSq / cycleTime) : 0;
    const speed_rms_rad_s = cycleTime > 0 ? Math.sqrt(sumSpeedSq / cycleTime) : 0;
    const power_rms_W = cycleTime > 0 ? Math.sqrt(sumPowerSq / cycleTime) : 0;

    demands.push({
      joint_name: jointNames[j],
      tau_rms_Nm,
      tau_peak_Nm: peakTau,
      speed_rms_rad_s,
      speed_peak_rad_s: peakSpeed,
      power_rms_W,
      power_peak_W: peakPower,
      regen_peak_W: peakRegen,
      cycle_time_s: cycleTime,
      complete: timeValid && valuesValid,
      time_valid: timeValid,
      values_valid: valuesValid,
      sample_count: numSamples
    });
  }

  return demands;
}

/**
 * Verifies a motor and gearbox combination against joint demands using multi-criteria margins.
 * Checks output capacity limits and gearbox physical limits, generating warnings for power.
 */
export function evaluateMotorGearboxCandidate(
  demand: JointDemand,
  motor: MotorSpec,
  gearbox: GearboxSpec,
  margins: SizingMargins
): ActuatorCandidate {
  const failure_reasons: string[] = [];
  const warnings: string[] = [];
  const motorPeakFactor = margins.motorPeakFactor ?? 5.0;
  const capacityValues = [
    motor.rated_power_W,
    motor.rated_torque_Nm,
    motor.no_load_speed_rpm,
    motor.mass_kg,
    gearbox.ratio,
    gearbox.efficiency,
    gearbox.max_continuous_torque_Nm,
    gearbox.max_intermittent_torque_Nm,
    gearbox.max_input_speed_rpm,
    gearbox.mass_kg,
    margins.continuous,
    margins.peak,
    margins.speed,
    margins.power,
    motorPeakFactor
  ];
  if (!capacityValues.every(value => Number.isFinite(value) && value > 0)) {
    failure_reasons.push('Candidate contains a non-finite or non-positive catalog capacity or sizing factor.');
  }

  if (demand.complete === false) {
    failure_reasons.push('Torque demand is incomplete: finite torque/velocity values and at least two samples with strictly increasing times are required.');
  }
  const demandValues = [
    demand.tau_rms_Nm,
    demand.tau_peak_Nm,
    demand.speed_rms_rad_s,
    demand.speed_peak_rad_s,
    demand.power_rms_W,
    demand.power_peak_W,
    demand.regen_peak_W,
    demand.cycle_time_s
  ];
  if (!demandValues.every(value => Number.isFinite(value) && value >= 0)) {
    failure_reasons.push('Torque demand contains a non-finite or negative derived value.');
  }

  // 1. Output capacities
  const tau_out_cont = motor.rated_torque_Nm * gearbox.ratio * gearbox.efficiency;

  // Peak torque is a configurable generic benchmark assumption, not a
  // manufacturer-backed capability or a statement about allowable duration.
  const tau_out_peak = motor.rated_torque_Nm * motorPeakFactor * gearbox.ratio * gearbox.efficiency;
  const omega_out_max = (motor.no_load_speed_rpm / gearbox.ratio) * (2.0 * Math.PI / 60.0);

  // Demand speeds at the gearbox input shaft
  const speed_peak_rpm = demand.speed_peak_rad_s * RAD_S_TO_RPM;
  const max_input_speed_demanded_rpm = speed_peak_rpm * gearbox.ratio;

  // 2. Verify all sizing criteria
  
  // Rule 1: Output continuous torque
  const required_cont = demand.tau_rms_Nm * margins.continuous;
  if (tau_out_cont < required_cont) {
    failure_reasons.push(
      `Output continuous torque (${tau_out_cont.toFixed(1)} Nm) is below required safety limit (${required_cont.toFixed(1)} Nm).`
    );
  }

  // Rule 2: Output peak torque
  const required_peak = demand.tau_peak_Nm * margins.peak;
  if (tau_out_peak < required_peak) {
    failure_reasons.push(
      `Output peak torque (${tau_out_peak.toFixed(1)} Nm) is below required safety limit (${required_peak.toFixed(1)} Nm).`
    );
  }

  // Rule 3: Output maximum speed
  const required_speed = demand.speed_peak_rad_s * margins.speed;
  if (omega_out_max < required_speed) {
    const omega_out_max_rpm = motor.no_load_speed_rpm / gearbox.ratio;
    const required_speed_rpm = required_speed * RAD_S_TO_RPM;
    failure_reasons.push(
      `Maximum output speed (${omega_out_max_rpm.toFixed(0)} RPM) is below required speed limit (${required_speed_rpm.toFixed(0)} RPM).`
    );
  }

  // Rule 4: Gearbox continuous torque limit
  const required_gb_cont = demand.tau_rms_Nm * margins.continuous;
  if (gearbox.max_continuous_torque_Nm < required_gb_cont) {
    failure_reasons.push(
      `Gearbox max continuous torque rating (${gearbox.max_continuous_torque_Nm} Nm) is below required safety limit (${required_gb_cont.toFixed(1)} Nm).`
    );
  }

  // Rule 5: Gearbox intermittent/peak torque limit
  const required_gb_peak = demand.tau_peak_Nm * margins.peak;
  if (gearbox.max_intermittent_torque_Nm < required_gb_peak) {
    failure_reasons.push(
      `Gearbox max intermittent torque rating (${gearbox.max_intermittent_torque_Nm} Nm) is below required safety limit (${required_gb_peak.toFixed(1)} Nm).`
    );
  }

  // Rule 6: Gearbox input speed limit
  const required_gb_input_speed_rpm = max_input_speed_demanded_rpm * margins.speed;
  if (gearbox.max_input_speed_rpm < required_gb_input_speed_rpm) {
    failure_reasons.push(
      `Gearbox max input speed limit (${gearbox.max_input_speed_rpm} RPM) is below required input speed (${required_gb_input_speed_rpm.toFixed(0)} RPM).`
    );
  }

  // Rule 7: Power rating limit (warning by default, blocks only if enforcePowerLimit is checked)
  const required_power_rms = (demand.power_rms_W / gearbox.efficiency) * margins.power;
  const power_margin = required_power_rms > 1e-6 ? motor.rated_power_W / required_power_rms : Infinity;
  let power_warning: string | undefined;

  if (power_margin < 1) {
    power_warning = `Motor rated power (${motor.rated_power_W} W) is below required RMS power limit (${required_power_rms.toFixed(0)} W).`;
    warnings.push(power_warning);
    if (margins.enforcePowerLimit) {
      failure_reasons.push(power_warning);
    }
  }

  // Margin calculation helper with division-by-zero protection (caps at Infinity if demand is 0)
  const getMargin = (available: number, demanded: number) => {
    if (!Number.isFinite(available) || !Number.isFinite(demanded) || demanded < 0) return 0;
    return demanded === 0 ? Infinity : available / demanded;
  };

  // Six safety-adjusted mechanical margins.  The object insertion order is
  // deliberate: it is also the fixed M1..M6 tie order for the limiter.
  const mechanical_margins = {
    M1: getMargin(tau_out_cont, demand.tau_rms_Nm * margins.continuous),
    M2: getMargin(tau_out_peak, demand.tau_peak_Nm * margins.peak),
    M3: getMargin(omega_out_max, demand.speed_peak_rad_s * margins.speed),
    M4: getMargin(gearbox.max_continuous_torque_Nm, demand.tau_rms_Nm * margins.continuous),
    M5: getMargin(gearbox.max_intermittent_torque_Nm, demand.tau_peak_Nm * margins.peak),
    M6: getMargin(gearbox.max_input_speed_rpm, max_input_speed_demanded_rpm * margins.speed)
  };
  const min_margin = Math.min(...MECHANICAL_CONSTRAINTS.map(constraint => mechanical_margins[constraint]));
  let limiting_constraint: MechanicalConstraint = 'M1';
  for (const constraint of MECHANICAL_CONSTRAINTS.slice(1)) {
    if (mechanical_margins[constraint] < mechanical_margins[limiting_constraint]) {
      limiting_constraint = constraint;
    }
  }
  if (!MECHANICAL_CONSTRAINTS.every(constraint => {
    const margin = mechanical_margins[constraint];
    return margin === Infinity || Number.isFinite(margin);
  })) {
    failure_reasons.push('Candidate contains a non-finite mechanical capacity or margin.');
  }
  const passes = failure_reasons.length === 0;

  return {
    motor_id: motor.id,
    gearbox_id: gearbox.id,
    gearbox_type: gearbox.type as GearboxType,
    ratio: gearbox.ratio,
    passes,
    failure_reasons,
    // Legacy names are retained, now with the safety-adjusted definitions.
    continuous_margin: mechanical_margins.M1,
    peak_margin: mechanical_margins.M2,
    speed_margin: mechanical_margins.M3,
    gearbox_continuous_margin: mechanical_margins.M4,
    gearbox_peak_margin: mechanical_margins.M5,
    gearbox_input_speed_margin: mechanical_margins.M6,
    power_margin,
    mechanical_margins,
    limiting_constraint,
    limiting_margin: min_margin,
    warnings,
    power_warning,
    min_margin,
    total_mass_kg: motor.mass_kg + gearbox.mass_kg,
    motor,
    gearbox
  };
}

/**
 * Sorts and ranks candidates:
 * 1. Passing combinations first.
 * 2. Higher min_margin first (Infinity values placed first).
 * 3. Lower total mass next.
 * 4. Smaller motor power next.
 * 5. Lower ratio next.
 */
export function rankCandidates(candidates: ActuatorCandidate[], objective: SizingObjective = 'min_mass'): ActuatorCandidate[] {
  const candidateMinMargin = (candidate: ActuatorCandidate): number => {
    // Recompute from the normalized six-margin set when available so a stale
    // legacy min_margin cannot affect max_margin ranking.
    if (candidate.mechanical_margins) {
      return Math.min(...MECHANICAL_CONSTRAINTS.map(constraint => candidate.mechanical_margins[constraint]));
    }
    return candidate.min_margin;
  };

  const compareDescendingMargin = (a: number, b: number): number => {
    if (a === b) return 0;
    if (a === Infinity) return -1;
    if (b === Infinity) return 1;
    if (Math.abs(a - b) <= 1e-5) return 0;
    return b - a;
  };

  return [...candidates].sort((a, b) => {
    // 1. Passes first (true first)
    if (a.passes !== b.passes) {
      return a.passes ? -1 : 1;
    }
    
    if (objective === 'min_power') {
      // 2. Smaller motor power first
      if (Math.abs(a.motor.rated_power_W - b.motor.rated_power_W) > 1e-5) {
        return a.motor.rated_power_W - b.motor.rated_power_W;
      }
      // 3. Lower total mass next
      if (Math.abs(a.total_mass_kg - b.total_mass_kg) > 1e-5) {
        return a.total_mass_kg - b.total_mass_kg;
      }
    } else if (objective === 'min_gearbox') {
      // 2. Smaller gearbox mass first
      if (Math.abs(a.gearbox.mass_kg - b.gearbox.mass_kg) > 1e-5) {
        return a.gearbox.mass_kg - b.gearbox.mass_kg;
      }
      // 3. Lower total mass next
      if (Math.abs(a.total_mass_kg - b.total_mass_kg) > 1e-5) {
        return a.total_mass_kg - b.total_mass_kg;
      }
      // 4. Smaller motor power next
      if (Math.abs(a.motor.rated_power_W - b.motor.rated_power_W) > 1e-5) {
        return a.motor.rated_power_W - b.motor.rated_power_W;
      }
    } else if (objective === 'max_margin') {
      // 2. Higher min_margin first (Infinity placed first)
      const aMinMargin = candidateMinMargin(a);
      const bMinMargin = candidateMinMargin(b);
      const marginComparison = compareDescendingMargin(aMinMargin, bMinMargin);
      if (marginComparison !== 0) return marginComparison;
      // 3. Lower total mass next
      if (Math.abs(a.total_mass_kg - b.total_mass_kg) > 1e-5) {
        return a.total_mass_kg - b.total_mass_kg;
      }
      // 4. Smaller motor power next
      if (Math.abs(a.motor.rated_power_W - b.motor.rated_power_W) > 1e-5) {
        return a.motor.rated_power_W - b.motor.rated_power_W;
      }
    } else {
      // Default: min_mass
      // 2. Lower total mass first
      if (Math.abs(a.total_mass_kg - b.total_mass_kg) > 1e-5) {
        return a.total_mass_kg - b.total_mass_kg;
      }
      // 3. Smaller motor power next
      if (Math.abs(a.motor.rated_power_W - b.motor.rated_power_W) > 1e-5) {
        return a.motor.rated_power_W - b.motor.rated_power_W;
      }
    }

    // Common tie-breakers:
    // Ratio next (lower first)
    if (Math.abs(a.ratio - b.ratio) > 1e-5) {
      return a.ratio - b.ratio;
    }

    // Safety margin next (if not already sorted by max_margin)
    if (objective !== 'max_margin') {
      const aMinMargin = candidateMinMargin(a);
      const bMinMargin = candidateMinMargin(b);
      const marginComparison = compareDescendingMargin(aMinMargin, bMinMargin);
      if (marginComparison !== 0) return marginComparison;
    }

    // Do not rely on engine sort stability for catalog ties.
    const compareCodePoints = (first: string, second: string): number => first < second ? -1 : first > second ? 1 : 0;
    const motorOrder = compareCodePoints(a.motor_id, b.motor_id);
    if (motorOrder !== 0) return motorOrder;
    return compareCodePoints(a.gearbox_id, b.gearbox_id);
  });
}

/**
 * Selection entrypoint for TorqueLog.
 * Maps candidates based strictly on the library's compatibility_matrix.
 */
export function selectActuatorsForLog(
  torqueLog: TorqueLog,
  library: ActuatorLibrary,
  margins: SizingMargins,
  robotKind: string,
  robotName: string,
  filters?: {
    gearboxType?: 'harmonic' | 'cycloidal' | 'any';
  }
): ActuatorSizingReport {
  const demands = computeJointDemands(torqueLog);
  const joints: JointActuatorSelection[] = [];

  const gearboxTypeFilter = filters?.gearboxType || 'any';
  const gearboxesMap = new Map(library.gearboxes.map(g => [g.id, g]));

  for (let j = 0; j < demands.length; j++) {
    const demand = demands[j];
    const candidates: ActuatorCandidate[] = [];

    for (const motor of library.motors) {
      const compatibleGearboxIds = library.compatibility_matrix[motor.id];
      // If the motor is not found in the matrix, it generates zero candidates (strict mode)
      if (!compatibleGearboxIds) continue;

      for (const gbId of compatibleGearboxIds) {
        const gearbox = gearboxesMap.get(gbId);
        if (!gearbox) continue;

        // Apply reducer type filter
        if (gearboxTypeFilter !== 'any' && gearbox.type !== gearboxTypeFilter) {
          continue;
        }

        const candidate = evaluateMotorGearboxCandidate(demand, motor, gearbox, margins);
        candidates.push(candidate);
      }
    }

    const ranked = rankCandidates(candidates, margins.sizingObjective);
    const best = ranked.find(c => c.passes);

    joints.push({
      demand,
      candidates: ranked,
      best
    });
  }

  const complete = joints.length > 0 && joints.every(j => j.demand.complete && j.best !== undefined);

  const provenance_hash_algorithm = 'sha256-canonical-json-v1' as const;
  const torque_log_hash = canonicalSha256(torqueLog);
  const catalog_hash = canonicalSha256(library);

  const dynamics_source = torqueLog.engine_used || 'demo_frontend';
  const buildSourceCommit = import.meta.env.VITE_ROBODIMM_SOURCE_COMMIT || null;
  const catalogFileSha256 = import.meta.env.VITE_ACTUATOR_CATALOG_SHA256 || null;

  return {
    schema: "robodimm.actuator_sizing_report.v2",
    robot_kind: robotKind,
    robot_name: robotName,
    dynamics_source,
    torque_log_hash,
    catalog_hash,
    provenance_hash_algorithm,
    torque_log_hash_scope: 'parsed_canonical_json',
    catalog_hash_scope: 'parsed_canonical_json',
    catalog_file_sha256: catalogFileSha256,
    dynamics_manifest: torqueLog.manifest ?? null,
    source_commit: torqueLog.manifest?.source_commit ?? buildSourceCommit,
    catalog_version: library.metadata.version || "2.0",
    catalog_anonymized: true,
    motor_peak_policy: "generic_rated_torque_factor_benchmark_assumption",
    motor_peak_assumption: {
      factor: margins.motorPeakFactor ?? 5.0,
      basis: 'generic_benchmark_assumption',
      manufacturer_backed: false,
      peak_duration_s: null,
      peak_duration_known: false
    },
    screening_scope: 'preliminary_actuator_screening',
    recommendation_type: 'design_support_recommendation',
    procurement_validated: false,
    warnings: [
      'Preliminary actuator screening and candidate ranking only; this is not procurement validation.',
      'The generic motor peak factor is not manufacturer-backed and allowable peak duration and thermal duty are unknown.'
    ],
    margins,
    joints,
    complete
  };
}

export function getCr4LinkForJoint(jointIdx: number): string {
  const cr4Links = ['SWING', 'LOWER_ARM', 'UPPER_ARM', 'DISK'];
  return cr4Links[jointIdx] || 'FOOT';
}

/**
 * @deprecated Use selectActuatorsForLog instead. For backwards compatibility with legacy code.
 */
export function selectActuators(
  library: ActuatorLibrary,
  torqueLog: TorqueLog,
  margins: {
    continuous: number;
    peak: number;
    speed: number;
    power?: number;
    motorPeakFactor?: number;
    enforcePowerLimit?: boolean;
  }
): RobotActuatorSelection {
  const fullMargins: SizingMargins = {
    continuous: margins.continuous,
    peak: margins.peak,
    speed: margins.speed,
    power: margins.power ?? 1.1,
    motorPeakFactor: margins.motorPeakFactor ?? 5.0,
    enforcePowerLimit: margins.enforcePowerLimit ?? false
  };
  return selectActuatorsForLog(torqueLog, library, fullMargins, 'unknown', 'unknown');
}
