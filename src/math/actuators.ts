import { ActuatorLibrary, TorqueLog, SizingMargins, JointDemand, ActuatorCandidate, MotorSpec, GearboxSpec, ActuatorSizingReport, GearboxType, JointActuatorSelection, RobotActuatorSelection, SizingObjective } from '../model/schemas';

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

  for (let j = 0; j < jointNames.length; j++) {
    let sumTauSq = 0;
    let sumSpeedSq = 0;
    let sumPowerSq = 0;
    let peakTau = 0;
    let peakSpeed = 0;
    let peakPower = 0;
    let peakRegen = 0;
    let totalTime = 0;

    for (let s = 0; s < numSamples; s++) {
      const sample = torqueLog.samples[s];
      const prevTime = s > 0 ? torqueLog.samples[s - 1].time_s : sample.time_s - torqueLog.dt_s;
      const dt = sample.time_s - prevTime;
      
      // Prevent backward jumps or zero intervals
      if (dt <= 0) continue;

      const tau = sample.tau[j];
      const velocity = sample.joint_velocity ? sample.joint_velocity[j] : (sample.velocity ? sample.velocity[j] : 0);
      const p = tau * velocity;

      sumTauSq += tau * tau * dt;
      sumSpeedSq += velocity * velocity * dt;
      sumPowerSq += p * p * dt;

      peakTau = Math.max(peakTau, Math.abs(tau));
      peakSpeed = Math.max(peakSpeed, Math.abs(velocity));
      peakPower = Math.max(peakPower, Math.abs(p));

      // Regenerative power check (negative power)
      if (p < 0) {
        peakRegen = Math.max(peakRegen, Math.abs(p));
      }
      totalTime += dt;
    }

    const tau_rms_Nm = totalTime > 0 ? Math.sqrt(sumTauSq / totalTime) : 0;
    const speed_rms_rad_s = totalTime > 0 ? Math.sqrt(sumSpeedSq / totalTime) : 0;
    const power_rms_W = totalTime > 0 ? Math.sqrt(sumPowerSq / totalTime) : 0;

    demands.push({
      joint_name: jointNames[j],
      tau_rms_Nm,
      tau_peak_Nm: peakTau,
      speed_rms_rad_s,
      speed_peak_rad_s: peakSpeed,
      power_rms_W,
      power_peak_W: peakPower,
      regen_peak_W: peakRegen,
      cycle_time_s: totalTime
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
  const motorPeakFactor = margins.motorPeakFactor ?? 5.0;

  // 1. Output capacities
  const tau_out_cont = motor.rated_torque_Nm * gearbox.ratio * gearbox.efficiency;
  
  // Peak torque available based on 5x rated motor torque policy
  const tau_out_peak = motor.rated_torque_Nm * motorPeakFactor * gearbox.ratio * gearbox.efficiency;
  const omega_out_max = (motor.no_load_speed_rpm / gearbox.ratio) * (2.0 * Math.PI / 60.0);

  // Demand speeds at the gearbox input shaft
  const speed_peak_rpm = demand.speed_peak_rad_s * (60.0 / (2.0 * Math.PI));
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
    const required_speed_rpm = required_speed * (60.0 / (2.0 * Math.PI));
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
  const power_margin = demand.power_rms_W > 1e-6 ? (motor.rated_power_W * gearbox.efficiency) / demand.power_rms_W : Infinity;
  
  if (motor.rated_power_W < required_power_rms) {
    if (margins.enforcePowerLimit) {
      failure_reasons.push(
        `Motor rated power (${motor.rated_power_W} W) is below required RMS power limit (${required_power_rms.toFixed(0)} W).`
      );
    }
  }

  const passes = failure_reasons.length === 0;

  // Margin calculation helper with division-by-zero protection (caps at Infinity if demand is 0)
  const getMargin = (available: number, demanded: number) => {
    return demanded < 1e-6 ? Infinity : available / demanded;
  };

  const continuous_margin = getMargin(tau_out_cont, demand.tau_rms_Nm);
  const peak_margin = getMargin(tau_out_peak, demand.tau_peak_Nm);
  const speed_margin = getMargin(omega_out_max, demand.speed_peak_rad_s);
  const gearbox_continuous_margin = getMargin(gearbox.max_continuous_torque_Nm, demand.tau_rms_Nm);
  const gearbox_peak_margin = getMargin(gearbox.max_intermittent_torque_Nm, demand.tau_peak_Nm);

  // Compute minimum safety margin among critical rules (excluding power warning)
  const min_margin = Math.min(
    continuous_margin,
    peak_margin,
    speed_margin,
    gearbox_continuous_margin,
    gearbox_peak_margin
  );

  return {
    motor_id: motor.id,
    gearbox_id: gearbox.id,
    gearbox_type: gearbox.type as GearboxType,
    ratio: gearbox.ratio,
    passes,
    failure_reasons,
    continuous_margin,
    peak_margin,
    speed_margin,
    gearbox_continuous_margin,
    gearbox_peak_margin,
    power_margin,
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
      if (a.min_margin !== b.min_margin) {
        if (a.min_margin === Infinity) return -1;
        if (b.min_margin === Infinity) return 1;
        return b.min_margin - a.min_margin;
      }
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
    if (objective !== 'max_margin' && a.min_margin !== b.min_margin) {
      if (a.min_margin === Infinity) return -1;
      if (b.min_margin === Infinity) return 1;
      return b.min_margin - a.min_margin;
    }

    return 0;
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

  const complete = joints.every(j => j.best !== undefined);

  // Deterministic catalog/log hashes for manifest
  const getSimpleHash = (str: string): string => {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      hash = (hash << 5) - hash + str.charCodeAt(i);
      hash |= 0;
    }
    return Math.abs(hash).toString(16);
  };

  const torque_log_hash = getSimpleHash(JSON.stringify(torqueLog.samples.map(s => s.tau)));
  const library_hash = getSimpleHash(JSON.stringify({
    motors: library.motors.map(m => m.id),
    gearboxes: library.gearboxes.map(g => g.id)
  }));

  const dynamics_source = (torqueLog as any).engine_used || 'demo_frontend';

  return {
    schema: "robodimm.actuator_sizing_report.v1",
    robot_kind: robotKind,
    robot_name: robotName,
    dynamics_source,
    torque_log_hash,
    catalog_version: library.metadata.version || "2.0",
    catalog_anonymized: true,
    motor_peak_policy: "rated_torque_x_5_for_short_robotic_transients",
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
  margins: { continuous: number; peak: number; speed: number; power?: number }
): RobotActuatorSelection {
  const fullMargins: SizingMargins = {
    continuous: margins.continuous,
    peak: margins.peak,
    speed: margins.speed,
    power: margins.power ?? 1.1,
    motorPeakFactor: 5.0,
    enforcePowerLimit: false
  };
  return selectActuatorsForLog(torqueLog, library, fullMargins, 'unknown', 'unknown');
}

