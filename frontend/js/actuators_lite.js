/**
 * Frontend-only actuator sizing helpers for DEMO/lite mode.
 * Mirrors backend/robot_core selection logic.
 */

const ACTIVE_JOINTS_CR4 = { 0: 'J1', 1: 'J2', 2: 'J3', 3: 'J4' };
const ACTIVE_JOINTS_CR6 = { 0: 'J1', 1: 'J2', 2: 'J3', 3: 'J4', 4: 'J5', 5: 'J6' };

export function analyzeTrajectoryRequirementsLite(dynamicsData, robotType = 'CR4') {
  if (!dynamicsData || !Array.isArray(dynamicsData.tau) || !Array.isArray(dynamicsData.v)) return {};
  const tau = dynamicsData.tau;
  const vel = dynamicsData.v;
  const nv = tau[0]?.length || 0;
  const active = robotType === 'CR4' ? ACTIVE_JOINTS_CR4 : ACTIVE_JOINTS_CR6;
  const out = {};

  for (const [idxStr, name] of Object.entries(active)) {
    const idx = Number(idxStr);
    if (idx >= nv) continue;
    const tauJ = tau.map(r => Number(r[idx] || 0));
    const vJ = vel.map(r => Number(r[idx] || 0));
    const peakTau = Math.max(...tauJ.map(Math.abs));
    const rmsTau = Math.sqrt(tauJ.reduce((s, x) => s + x * x, 0) / Math.max(tauJ.length, 1));
    const peakVRad = Math.max(...vJ.map(Math.abs));
    const peakVRpm = peakVRad * 60 / (2 * Math.PI);
    const meanVRpm = (vJ.reduce((s, x) => s + Math.abs(x), 0) / Math.max(vJ.length, 1)) * 60 / (2 * Math.PI);

    out[name] = {
      peak_torque_Nm: Number(peakTau.toFixed(3)),
      rms_torque_Nm: Number(rmsTau.toFixed(3)),
      peak_velocity_rad_s: Number(peakVRad.toFixed(4)),
      peak_velocity_rpm: Number(peakVRpm.toFixed(2)),
      mean_velocity_rpm: Number(meanVRpm.toFixed(2)),
      internal_idx_v: idx,
    };
  }
  return out;
}

export function selectActuatorsLite(requirements, motors, gearboxes, compatibilityMatrix, sfTorque = 1.5, sfSpeed = 1.2) {
  const result = {};
  for (const [joint, req] of Object.entries(requirements || {})) {
    const reqTorque = Number(req.peak_torque_Nm || 0) * sfTorque;
    const reqSpeed = Number(req.peak_velocity_rpm || 0) * sfSpeed;
    const candidates = [];
    const rejected = [];

    for (const motor of motors || []) {
      const motorId = motor.id;
      const motorTorque = Number(motor.rated_torque_Nm ?? motor.nominal_torque_Nm ?? 0);
      const motorSpeed = Number(motor.rated_speed_rpm ?? motor.nominal_speed_rpm ?? 0);
      if (motorTorque <= 0 || motorSpeed <= 0) continue;

      let compatible = motor.compatible_gearboxes || [];
      if ((!compatible || compatible.length === 0) && compatibilityMatrix?.[motorId]) {
        compatible = compatibilityMatrix[motorId];
      }

      for (const gb of gearboxes || []) {
        const gbId = gb.id;
        if (!compatible?.includes?.(gbId)) continue;
        const naturalMatch = typeof compatibilityMatrix?.[motorId] === 'object'
          ? Boolean(compatibilityMatrix[motorId]?.[gbId]?.natural_match)
          : true;

        let ratios = gb.ratios || [];
        if ((!ratios || ratios.length === 0) && gb.ratio) ratios = [gb.ratio];

        for (const ratioRaw of ratios) {
          const ratio = Number(ratioRaw || 0);
          if (ratio <= 0) continue;
          const efficiency = Number(gb.efficiency ?? 0.85) || 0.85;
          const outTorque = motorTorque * ratio * efficiency;
          const outSpeed = motorSpeed / ratio;
          if (outTorque >= reqTorque && outSpeed >= reqSpeed) {
            const marginTau = reqTorque > 0 ? ((outTorque / reqTorque) - 1) * 100 : 999;
            const marginW = reqSpeed > 0 ? ((outSpeed / reqSpeed) - 1) * 100 : 999;
            let score = 0;
            score += naturalMatch ? 0 : 100;
            score += motorTorque * 5;
            score += ratio * 0.1;
            score -= Math.min(marginTau, 100) * 0.2;
            candidates.push({
              motor_id: motorId,
              motor_desc: motor.name || motor.description || motorId,
              gearbox_id: gbId,
              gearbox_desc: gb.name || gb.description || gbId,
              ratio,
              natural_match: naturalMatch,
              output_torque_Nm: Number(outTorque.toFixed(2)),
              max_output_speed_rpm: Number(outSpeed.toFixed(2)),
              margin_torque_pct: Number(marginTau.toFixed(1)),
              margin_speed_pct: Number(marginW.toFixed(1)),
              score: Number(score.toFixed(2)),
              motor_mass_kg: motor.mass_kg,
              gearbox_mass_kg: gb.mass_kg,
            });
          } else if (rejected.length < 3) {
            const fail = [];
            if (outTorque < reqTorque) fail.push(`torque: ${outTorque.toFixed(1)} < ${reqTorque.toFixed(1)} Nm`);
            if (outSpeed < reqSpeed) fail.push(`speed: ${outSpeed.toFixed(1)} < ${reqSpeed.toFixed(1)} rpm`);
            rejected.push({
              motor_id: motorId,
              gearbox_id: gbId,
              ratio,
              output_torque_Nm: Number(outTorque.toFixed(2)),
              output_speed_rpm: Number(outSpeed.toFixed(2)),
              fail_reason: fail.join(', '),
            });
          }
        }
      }
    }

    candidates.sort((a, b) => a.score - b.score);
    const top = candidates.slice(0, 5);
    result[joint] = {
      required: {
        torque_Nm: Number(reqTorque.toFixed(3)),
        speed_rpm: Number(reqSpeed.toFixed(2)),
        original_peak_torque_Nm: Number(req.peak_torque_Nm || 0),
        original_peak_speed_rpm: Number(req.peak_velocity_rpm || 0),
        mean_velocity_rpm: Number(req.mean_velocity_rpm || 0),
      },
      candidates: top,
      rejected_samples: rejected,
      recommended: top[0] || null,
    };
  }
  return result;
}

export function getActuatorMassesLite(selection, motors, gearboxes) {
  const md = Object.fromEntries((motors || []).map(m => [m.id, m]));
  const gd = Object.fromEntries((gearboxes || []).map(g => [g.id, g]));
  const out = {};
  for (const [joint, data] of Object.entries(selection || {})) {
    const rec = data?.recommended;
    if (!rec) {
      out[joint] = { motor_mass_kg: null, gearbox_mass_kg: null, total_kg: null };
      continue;
    }
    const mm = md[rec.motor_id]?.mass_kg ?? null;
    const gm = gd[rec.gearbox_id]?.mass_kg ?? null;
    out[joint] = { motor_mass_kg: mm, gearbox_mass_kg: gm, total_kg: (mm != null && gm != null) ? mm + gm : null };
  }
  return out;
}

export function validateSelectionLite(round1Req, round1Sel, motors, gearboxes, compatibilityMatrix, sfTorque = 1.5, sfSpeed = 1.2, robotType = 'CR4', payloadKg = 0) {
  const md = Object.fromEntries((motors || []).map(m => [m.id, m]));
  const gd = Object.fromEntries((gearboxes || []).map(g => [g.id, g]));

  const jointKeys = Object.keys(round1Req || {});
  const additional = new Array(jointKeys.length).fill(0);
  const idx = (k) => k.startsWith('J') ? (parseInt(k.slice(1), 10) - 1) : 0;

  for (const [joint, selData] of Object.entries(round1Sel || {})) {
    const rec = selData?.recommended;
    if (!rec) continue;
    const totalMass = Number(md[rec.motor_id]?.mass_kg || 0) + Number(gd[rec.gearbox_id]?.mass_kg || 0);
    if (robotType === 'CR4') {
      if (joint === 'J2' || joint === 'J3') additional[0] += totalMass;
      else if (joint === 'J4') {
        for (let i = 0; i < 3; i++) additional[i] += totalMass;
      }
    } else {
      const j = idx(joint);
      for (let i = 0; i < j; i++) additional[i] += totalMass;
    }
  }

  const req2 = {};
  for (const [joint, req] of Object.entries(round1Req || {})) {
    const j = idx(joint);
    const extra = additional[j] || 0;
    const baseMass = robotType === 'CR4' ? 15.0 : 8.0;
    const mult = 1.0 + (extra / Math.max(baseMass + Number(payloadKg || 0), 1.0));
    req2[joint] = {
      ...req,
      peak_torque_Nm: Number(req.peak_torque_Nm || 0) * mult,
      rms_torque_Nm: Number(req.rms_torque_Nm || 0) * mult,
    };
  }

  const sel2 = selectActuatorsLite(req2, motors, gearboxes, compatibilityMatrix, sfTorque, sfSpeed);
  const changes = [];
  for (const joint of Object.keys(round1Sel || {})) {
    const r1 = round1Sel[joint]?.recommended;
    const r2 = sel2[joint]?.recommended;
    if (r1 && r2) {
      if (r1.motor_id !== r2.motor_id || r1.gearbox_id !== r2.gearbox_id) {
        changes.push({ joint, round1: `${r1.motor_id} + ${r1.gearbox_id}`, round2: `${r2.motor_id} + ${r2.gearbox_id}`, reason: 'Actuator mass effect' });
      }
    } else if (r1 && !r2) {
      changes.push({ joint, round1: `${r1.motor_id} + ${r1.gearbox_id}`, round2: 'NO SOLUTION', reason: 'Actuator masses exceed capacity' });
    }
  }

  const hasMissing = Object.values(round1Sel || {}).some(v => !v?.recommended);
  return {
    validated: changes.length === 0 && !hasMissing,
    round2_selection: sel2,
    changes_needed: changes,
    additional_masses_kg: Object.fromEntries(additional.map((m, i) => [`joint_${i + 1}`, Number(m.toFixed(2))])),
    total_actuator_contribution_kg: Number((additional[0] || 0).toFixed(2)),
  };
}
