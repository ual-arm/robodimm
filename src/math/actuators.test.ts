import { describe, it, expect } from 'vitest';
import {
  computeJointDemands,
  evaluateMotorGearboxCandidate,
  rankCandidates,
  selectActuatorsForLog
} from './actuators';
import { TorqueLog, ActuatorLibrary, SizingMargins, MotorSpec, GearboxSpec } from '../model/schemas';

describe('Deterministic Actuator Sizing Engine', () => {
  // 1. Non-uniform time-weighted integrations
  it('should compute time-weighted RMS and peak joint demands correctly with non-uniform time delta intervals', () => {
    const mockLog: TorqueLog = {
      dt_s: 0.1,
      joint_names: ['Joint1'],
      samples: [
        { time_s: 0.0, tau: [10.0], joint_velocity: [2.0], velocity: [2.0], q: [0] },
        // dt = 0.2
        { time_s: 0.2, tau: [20.0], joint_velocity: [4.0], velocity: [4.0], q: [0] },
        // dt = 0.3
        { time_s: 0.5, tau: [0.0], joint_velocity: [0.0], velocity: [0.0], q: [0] },
      ]
    } as any;
    
    // dt for index 0: 0.0 - (0.0 - 0.1) = 0.1s. tau = 10, vel = 2. power = 20.
    // dt for index 1: 0.2 - 0.0 = 0.2s. tau = 20, vel = 4. power = 80.
    // dt for index 2: 0.5 - 0.2 = 0.3s. tau = 0, vel = 0. power = 0.
    // Total time = 0.1 + 0.2 + 0.3 = 0.6s.
    // Integral(tau^2 dt) = 10^2 * 0.1 + 20^2 * 0.2 + 0 * 0.3 = 10 + 80 = 90.
    // RMS tau = sqrt(90 / 0.6) = sqrt(150) = 12.2474.
    // Integral(speed^2 dt) = 2^2 * 0.1 + 4^2 * 0.2 + 0 * 0.3 = 0.4 + 3.2 = 3.6.
    // RMS speed = sqrt(3.6 / 0.6) = sqrt(6) = 2.4495.
    // Integral(power^2 dt) = 20^2 * 0.1 + 80^2 * 0.2 + 0 * 0.3 = 40 + 1280 = 1320.
    // RMS power = sqrt(1320 / 0.6) = sqrt(2200) = 46.904.
    
    const demands = computeJointDemands(mockLog);
    expect(demands.length).toBe(1);
    const d = demands[0];
    expect(d.joint_name).toBe('Joint1');
    expect(d.tau_rms_Nm).toBeCloseTo(Math.sqrt(150), 4);
    expect(d.speed_rms_rad_s).toBeCloseTo(Math.sqrt(6), 4);
    expect(d.power_rms_W).toBeCloseTo(Math.sqrt(2200), 4);
    expect(d.tau_peak_Nm).toBe(20.0);
    expect(d.speed_peak_rad_s).toBe(4.0);
    expect(d.power_peak_W).toBe(80.0);
    expect(d.cycle_time_s).toBeCloseTo(0.6, 6);
  });

  // Regen peak peak check (negative power)
  it('should compute regenerative peak power correctly', () => {
    const mockLog: TorqueLog = {
      dt_s: 0.1,
      joint_names: ['Joint1'],
      samples: [
        { time_s: 0.1, tau: [-10.0], joint_velocity: [2.0], velocity: [2.0], q: [0] }, // power = -20
        { time_s: 0.2, tau: [15.0], joint_velocity: [2.0], velocity: [2.0], q: [0] },  // power = 30
        { time_s: 0.3, tau: [10.0], joint_velocity: [-3.0], velocity: [-3.0], q: [0] } // power = -30
      ]
    } as any;
    const demands = computeJointDemands(mockLog);
    expect(demands[0].regen_peak_W).toBe(30); // max of |-20| and |-30|
  });

  // Zero-demand handling
  it('should handle zero demand correctly without NaN or division by zero, yielding Infinity margins', () => {
    const mockLog: TorqueLog = {
      dt_s: 0.1,
      joint_names: ['Joint1'],
      samples: [
        { time_s: 0.1, tau: [0.0], joint_velocity: [0.0], velocity: [0.0], q: [0] }
      ]
    } as any;
    const demands = computeJointDemands(mockLog);
    expect(demands[0].tau_rms_Nm).toBe(0);
    
    const motor: MotorSpec = {
      id: 'motor1', name: 'M1', manufacturer: 'Generic', series: 'S1',
      rated_power_W: 100, rated_torque_Nm: 1, rated_speed_rpm: 1000,
      stall_torque_Nm: 3, no_load_speed_rpm: 1500, mass_kg: 1
    } as any;
    const gearbox: GearboxSpec = {
      id: 'gb1', name: 'G1', manufacturer: 'Generic', series: 'S1',
      ratio: 50, stages: 1, max_continuous_torque_Nm: 50, max_intermittent_torque_Nm: 100,
      efficiency: 0.9, max_input_speed_rpm: 3000, type: 'harmonic', mass_kg: 0.5
    } as any;
    const margins: SizingMargins = { continuous: 1.0, peak: 1.0, speed: 1.0, power: 1.0, motorPeakFactor: 5.0, enforcePowerLimit: false };
    
    const candidate = evaluateMotorGearboxCandidate(demands[0], motor, gearbox, margins);
    expect(candidate.passes).toBe(true);
    expect(candidate.continuous_margin).toBe(Infinity);
    expect(candidate.peak_margin).toBe(Infinity);
    expect(candidate.speed_margin).toBe(Infinity);
    expect(candidate.power_margin).toBe(Infinity);
    expect(candidate.min_margin).toBe(Infinity);
  });

  // Sizing margin rule validations
  describe('Sizing Margin Checks', () => {
    const defaultMotor: MotorSpec = {
      id: 'm1', name: 'Motor1', manufacturer: 'Generic', series: 'S',
      rated_power_W: 500, rated_torque_Nm: 2.0, rated_speed_rpm: 3000,
      stall_torque_Nm: 99.0, // Should be ignored under 5x rated policy
      no_load_speed_rpm: 4000, mass_kg: 2.0
    } as any;

    const defaultGearbox: GearboxSpec = {
      id: 'g1', name: 'Gearbox1', manufacturer: 'Generic', series: 'S',
      ratio: 50, stages: 1, max_continuous_torque_Nm: 120, max_intermittent_torque_Nm: 250,
      efficiency: 0.9, max_input_speed_rpm: 5000, type: 'harmonic', mass_kg: 1.0
    } as any;

    const defaultMargins: SizingMargins = {
      continuous: 1.3,
      peak: 1.2,
      speed: 1.15,
      power: 1.1,
      motorPeakFactor: 5.0,
      enforcePowerLimit: false
    };

    const baseDemand = {
      joint_name: 'J1',
      tau_rms_Nm: 50.0,
      tau_peak_Nm: 200.0,
      speed_rms_rad_s: 1.0,
      speed_peak_rad_s: 2.0,
      power_rms_W: 100.0,
      power_peak_W: 300.0,
      regen_peak_W: 50.0,
      cycle_time_s: 1.0
    };

    it('should pass if all requirements are met within margins', () => {
      const candidate = evaluateMotorGearboxCandidate(baseDemand, defaultMotor, defaultGearbox, defaultMargins);
      expect(candidate.passes).toBe(true);
      expect(candidate.failure_reasons.length).toBe(0);
    });

    it('should fail continuous torque limit if demanded is too high', () => {
      const demand = { ...baseDemand, tau_rms_Nm: 80.0 };
      const candidate = evaluateMotorGearboxCandidate(demand, defaultMotor, defaultGearbox, defaultMargins);
      expect(candidate.passes).toBe(false);
      expect(candidate.failure_reasons.some(r => r.includes('continuous torque'))).toBe(true);
    });

    it('should fail peak torque limit based strictly on 5x rated torque policy (ignoring stall_torque_Nm)', () => {
      const demand = { ...baseDemand, tau_peak_Nm: 400.0 };
      const candidate = evaluateMotorGearboxCandidate(demand, defaultMotor, defaultGearbox, defaultMargins);
      expect(candidate.passes).toBe(false);
      expect(candidate.failure_reasons.some(r => r.includes('peak torque'))).toBe(true);
    });

    it('should fail output speed limit when motor no-load speed is insufficient', () => {
      const demand = { ...baseDemand, speed_peak_rad_s: 8.0 };
      const candidate = evaluateMotorGearboxCandidate(demand, defaultMotor, defaultGearbox, defaultMargins);
      expect(candidate.passes).toBe(false);
      expect(candidate.failure_reasons.some(r => r.includes('output speed'))).toBe(true);
    });

    it('should pass motor power check but generate warning if enforcePowerLimit is false', () => {
      const lowPowerMotor = { ...defaultMotor, rated_power_W: 50 } as any;
      const candidate = evaluateMotorGearboxCandidate(baseDemand, lowPowerMotor, defaultGearbox, defaultMargins);
      expect(candidate.passes).toBe(true);
      expect(candidate.failure_reasons.length).toBe(0);
      expect(candidate.power_margin).toBeLessThan(1.0);
    });

    it('should fail motor power check if enforcePowerLimit is true', () => {
      const lowPowerMotor = { ...defaultMotor, rated_power_W: 50 } as any;
      const margins = { ...defaultMargins, enforcePowerLimit: true };
      const candidate = evaluateMotorGearboxCandidate(baseDemand, lowPowerMotor, defaultGearbox, margins);
      expect(candidate.passes).toBe(false);
      expect(candidate.failure_reasons.some(r => r.includes('rated power'))).toBe(true);
    });

    it('should fail gearbox continuous torque rating check', () => {
      const lowContinuousGb = { ...defaultGearbox, max_continuous_torque_Nm: 60.0 } as any;
      const candidate = evaluateMotorGearboxCandidate(baseDemand, defaultMotor, lowContinuousGb, defaultMargins);
      expect(candidate.passes).toBe(false);
      expect(candidate.failure_reasons.some(r => r.includes('Gearbox max continuous torque'))).toBe(true);
    });

    it('should fail gearbox intermittent torque rating check', () => {
      const lowIntermittentGb = { ...defaultGearbox, max_intermittent_torque_Nm: 230.0 } as any;
      const candidate = evaluateMotorGearboxCandidate(baseDemand, defaultMotor, lowIntermittentGb, defaultMargins);
      expect(candidate.passes).toBe(false);
      expect(candidate.failure_reasons.some(r => r.includes('Gearbox max intermittent torque'))).toBe(true);
    });

    it('should fail gearbox input speed limit check', () => {
      const lowSpeedGb = { ...defaultGearbox, max_input_speed_rpm: 1000 } as any;
      const candidate = evaluateMotorGearboxCandidate(baseDemand, defaultMotor, lowSpeedGb, defaultMargins);
      expect(candidate.passes).toBe(false);
      expect(candidate.failure_reasons.some(r => r.includes('Gearbox max input speed limit'))).toBe(true);
    });
  });

  // Filter and catalog test cases
  describe('Select Actuators For Log API', () => {
    const library: ActuatorLibrary = {
      motors: [
        { id: 'M_SMALL', name: 'Small Motor', manufacturer: 'Generic', series: 'S', rated_power_W: 200, rated_torque_Nm: 0.5, rated_speed_rpm: 3000, stall_torque_Nm: 1.5, no_load_speed_rpm: 4000, mass_kg: 1.0 },
        { id: 'M_LARGE', name: 'Large Motor', manufacturer: 'Generic', series: 'L', rated_power_W: 1000, rated_torque_Nm: 3.0, rated_speed_rpm: 3000, stall_torque_Nm: 9.0, no_load_speed_rpm: 4000, mass_kg: 3.0 },
        { id: 'M_UNMAPPED', name: 'Unmapped Motor', manufacturer: 'Generic', series: 'U', rated_power_W: 500, rated_torque_Nm: 1.5, rated_speed_rpm: 3000, stall_torque_Nm: 4.5, no_load_speed_rpm: 4000, mass_kg: 2.0 }
      ] as any,
      gearboxes: [
        { id: 'GB_HARMONIC_SMALL', name: 'Harmonic Small', manufacturer: 'Generic', series: 'HS', ratio: 50, stages: 1, max_continuous_torque_Nm: 30, max_intermittent_torque_Nm: 60, efficiency: 0.9, max_input_speed_rpm: 6000, type: 'harmonic', mass_kg: 0.5 },
        { id: 'GB_CYCLOIDAL_LARGE', name: 'Cycloidal Large', manufacturer: 'Generic', series: 'CL', ratio: 80, stages: 2, max_continuous_torque_Nm: 300, max_intermittent_torque_Nm: 600, efficiency: 0.85, max_input_speed_rpm: 4000, type: 'cycloidal', mass_kg: 4.0 }
      ] as any,
      compatibility_matrix: {
        'M_SMALL': ['GB_HARMONIC_SMALL'],
        'M_LARGE': ['GB_CYCLOIDAL_LARGE']
      },
      metadata: {
        version: '2.0',
        last_updated: '2026-06-07',
        description: 'Mock library'
      } as any
    };

    const mockLog: TorqueLog = {
      dt_s: 0.1,
      joint_names: ['Joint1'],
      samples: [
        { time_s: 0.1, tau: [20.0], joint_velocity: [1.0], velocity: [1.0], q: [0] }
      ]
    } as any;

    const margins: SizingMargins = { continuous: 1.0, peak: 1.0, speed: 1.0, power: 1.0, motorPeakFactor: 5.0, enforcePowerLimit: false };

    it('should respect compatibility_matrix strictly and ignore unmapped motors', () => {
      const report = selectActuatorsForLog(mockLog, library, margins, 'cr4', 'CR4_Robot');
      const candidates = report.joints[0].candidates;
      
      expect(candidates.some(c => c.motor_id === 'M_SMALL')).toBe(true);
      expect(candidates.some(c => c.motor_id === 'M_LARGE')).toBe(true);
      expect(candidates.some(c => c.motor_id === 'M_UNMAPPED')).toBe(false);
    });

    it('should filter harmonic (strain wave) gearboxes and exclude cycloidal', () => {
      const report = selectActuatorsForLog(mockLog, library, margins, 'cr4', 'CR4_Robot', { gearboxType: 'harmonic' });
      const candidates = report.joints[0].candidates;
      
      expect(candidates.every(c => c.gearbox_type === 'harmonic')).toBe(true);
      expect(candidates.some(c => c.gearbox_id === 'GB_HARMONIC_SMALL')).toBe(true);
      expect(candidates.some(c => c.gearbox_id === 'GB_CYCLOIDAL_LARGE')).toBe(false);
    });

    it('should filter cycloidal gearboxes and exclude harmonic', () => {
      const report = selectActuatorsForLog(mockLog, library, margins, 'cr4', 'CR4_Robot', { gearboxType: 'cycloidal' });
      const candidates = report.joints[0].candidates;
      
      expect(candidates.every(c => c.gearbox_type === 'cycloidal')).toBe(true);
      expect(candidates.some(c => c.gearbox_id === 'GB_CYCLOIDAL_LARGE')).toBe(true);
      expect(candidates.some(c => c.gearbox_id === 'GB_HARMONIC_SMALL')).toBe(false);
    });

    it('should ensure the exported report is clean of commercial brand names', () => {
      const report = selectActuatorsForLog(mockLog, library, margins, 'cr4', 'CR4_Robot');
      const jsonStr = JSON.stringify(report);
      expect(jsonStr.toLowerCase()).not.toContain('nabtesco');
    });

    it('should rank candidates by smallest passing option first (lower mass, then lower power, then higher margin)', () => {
      const c1 = {
        motor_id: 'm1', gearbox_id: 'g1', gearbox_type: 'harmonic' as any, ratio: 50,
        passes: false, failure_reasons: ['failed'], continuous_margin: 2.0, peak_margin: 2.0, speed_margin: 2.0,
        gearbox_continuous_margin: 2.0, gearbox_peak_margin: 2.0, power_margin: 2.0, min_margin: 2.0, total_mass_kg: 1.0,
        motor: { id: 'm1', name: 'M1', rated_power_W: 100 } as any, gearbox: { id: 'g1', name: 'G1' } as any
      };
      // c2 has smaller mass and lower power, but lower margin than c3
      const c2 = {
        motor_id: 'm2', gearbox_id: 'g2', gearbox_type: 'harmonic' as any, ratio: 50,
        passes: true, failure_reasons: [], continuous_margin: 1.5, peak_margin: 1.5, speed_margin: 1.5,
        gearbox_continuous_margin: 1.5, gearbox_peak_margin: 1.5, power_margin: 1.5, min_margin: 1.5, total_mass_kg: 1.0,
        motor: { id: 'm2', name: 'M2', rated_power_W: 100 } as any, gearbox: { id: 'g2', name: 'G2' } as any
      };
      // c3 has larger mass and power, but higher margin
      const c3 = {
        motor_id: 'm3', gearbox_id: 'g3', gearbox_type: 'harmonic' as any, ratio: 50,
        passes: true, failure_reasons: [], continuous_margin: 2.5, peak_margin: 2.5, speed_margin: 2.5,
        gearbox_continuous_margin: 2.5, gearbox_peak_margin: 2.5, power_margin: 2.5, min_margin: 2.5, total_mass_kg: 2.0,
        motor: { id: 'm3', name: 'M3', rated_power_W: 300 } as any, gearbox: { id: 'g3', name: 'G3' } as any
      };

      const ranked = rankCandidates([c1, c3, c2]);
      
      // c2 should be first because it passes and has lower mass (1.0kg vs 2.0kg)
      expect(ranked[0].motor_id).toBe('m2');
      expect(ranked[1].motor_id).toBe('m3');
      expect(ranked[2].motor_id).toBe('m1');
    });
  });
});
