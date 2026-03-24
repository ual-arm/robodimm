"""
test_backend_models.py
======================
Unit tests for ``backend.models`` Pydantic schemas.

These tests confirm that valid payloads are accepted and invalid ones are
rejected, providing a contract for the REST API surface.
"""

import pytest
from pydantic import ValidationError
from backend.models import (
    LoginRequest,
    JointCommand,
    MoveLinearCommand,
    JogJointCommand,
    JogCartesianCommand,
    JogOrientationCommand,
    SaveTargetCommand,
    AddInstructionCommand,
    DeleteTargetCommand,
    DeleteInstructionCommand,
    ExecuteProgramCommand,
    RobotConfigCommand,
    MotorSpec,
    GearboxSpec,
    ActuatorLibraryUpdate,
    ActuatorSelectionRequest,
)


# ---------------------------------------------------------------------------
# LoginRequest
# ---------------------------------------------------------------------------

class TestLoginRequest:
    def test_valid(self):
        req = LoginRequest(username="admin", password="robotics")
        assert req.username == "admin"
        assert req.password == "robotics"

    def test_missing_username_raises(self):
        with pytest.raises(ValidationError):
            LoginRequest(password="robotics")

    def test_missing_password_raises(self):
        with pytest.raises(ValidationError):
            LoginRequest(username="admin")


# ---------------------------------------------------------------------------
# JointCommand
# ---------------------------------------------------------------------------

class TestJointCommand:
    def test_valid_cr4(self):
        cmd = JointCommand(q=[0.1, -0.2, 0.3, -0.4])
        assert len(cmd.q) == 4

    def test_valid_cr6(self):
        cmd = JointCommand(q=[0.0] * 6)
        assert len(cmd.q) == 6

    def test_missing_q_raises(self):
        with pytest.raises(ValidationError):
            JointCommand()


# ---------------------------------------------------------------------------
# MoveLinearCommand
# ---------------------------------------------------------------------------

class TestMoveLinearCommand:
    def test_valid_with_defaults(self):
        cmd = MoveLinearCommand(target=[0.5, 0.0, 0.8])
        assert cmd.dt        == pytest.approx(0.005)
        assert cmd.max_iter  == 1000
        assert cmd.tol       == pytest.approx(5e-4)

    def test_custom_params(self):
        cmd = MoveLinearCommand(target=[0.3, -0.1, 0.6], dt=0.01, max_iter=500, tol=1e-3)
        assert cmd.dt       == pytest.approx(0.01)
        assert cmd.max_iter == 500
        assert cmd.tol      == pytest.approx(1e-3)

    def test_missing_target_raises(self):
        with pytest.raises(ValidationError):
            MoveLinearCommand()


# ---------------------------------------------------------------------------
# JogJointCommand
# ---------------------------------------------------------------------------

class TestJogJointCommand:
    def test_valid(self):
        cmd = JogJointCommand(index=2, delta=0.05)
        assert cmd.index == 2
        assert cmd.delta == pytest.approx(0.05)

    def test_negative_delta(self):
        cmd = JogJointCommand(index=0, delta=-0.1)
        assert cmd.delta == pytest.approx(-0.1)

    def test_missing_fields_raise(self):
        with pytest.raises(ValidationError):
            JogJointCommand(index=1)
        with pytest.raises(ValidationError):
            JogJointCommand(delta=0.01)


# ---------------------------------------------------------------------------
# JogCartesianCommand
# ---------------------------------------------------------------------------

class TestJogCartesianCommand:
    def test_valid_base_frame(self):
        cmd = JogCartesianCommand(delta=[0.01, 0.0, -0.02], frame="base")
        assert cmd.frame == "base"
        assert len(cmd.delta) == 3

    def test_default_frame(self):
        cmd = JogCartesianCommand(delta=[0.0, 0.0, 0.01])
        assert cmd.frame == "base"

    def test_ee_frame(self):
        cmd = JogCartesianCommand(delta=[0.0, 0.01, 0.0], frame="ee")
        assert cmd.frame == "ee"


# ---------------------------------------------------------------------------
# JogOrientationCommand
# ---------------------------------------------------------------------------

class TestJogOrientationCommand:
    def test_valid(self):
        import math
        cmd = JogOrientationCommand(delta=[0.0, math.pi / 36, 0.0])
        assert len(cmd.delta) == 3

    def test_default_frame_ee(self):
        cmd = JogOrientationCommand(delta=[0.0, 0.0, 0.01])
        assert cmd.frame == "ee"


# ---------------------------------------------------------------------------
# SaveTargetCommand
# ---------------------------------------------------------------------------

class TestSaveTargetCommand:
    def test_valid(self):
        cmd = SaveTargetCommand(name="PICK_LEFT")
        assert cmd.name == "PICK_LEFT"

    def test_missing_name_raises(self):
        with pytest.raises(ValidationError):
            SaveTargetCommand()


# ---------------------------------------------------------------------------
# AddInstructionCommand
# ---------------------------------------------------------------------------

class TestAddInstructionCommand:
    def test_movej_defaults(self):
        cmd = AddInstructionCommand(type="MoveJ", target_name="HOME")
        assert cmd.type        == "MoveJ"
        assert cmd.speed       == pytest.approx(100.0)
        assert cmd.zone        == pytest.approx(50.0)
        assert cmd.pause_time  == pytest.approx(1.0)
        assert cmd.via_target_name == ""

    def test_pause_instruction(self):
        cmd = AddInstructionCommand(type="Pause", pause_time=2.5)
        assert cmd.type       == "Pause"
        assert cmd.pause_time == pytest.approx(2.5)

    def test_movec_instruction(self):
        cmd = AddInstructionCommand(type="MoveC", target_name="END", via_target_name="VIA")
        assert cmd.via_target_name == "VIA"

    def test_missing_type_raises(self):
        with pytest.raises(ValidationError):
            AddInstructionCommand()


# ---------------------------------------------------------------------------
# DeleteTargetCommand / DeleteInstructionCommand
# ---------------------------------------------------------------------------

class TestDeleteCommands:
    def test_delete_target(self):
        cmd = DeleteTargetCommand(name="PICK_LEFT")
        assert cmd.name == "PICK_LEFT"

    def test_delete_instruction(self):
        cmd = DeleteInstructionCommand(index=3)
        assert cmd.index == 3

    def test_delete_target_missing_raises(self):
        with pytest.raises(ValidationError):
            DeleteTargetCommand()

    def test_delete_instruction_missing_raises(self):
        with pytest.raises(ValidationError):
            DeleteInstructionCommand()


# ---------------------------------------------------------------------------
# ExecuteProgramCommand
# ---------------------------------------------------------------------------

class TestExecuteProgramCommand:
    def test_default_speed_factor(self):
        cmd = ExecuteProgramCommand()
        assert cmd.speed_factor == pytest.approx(1.0)

    def test_custom_speed_factor(self):
        cmd = ExecuteProgramCommand(speed_factor=0.5)
        assert cmd.speed_factor == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# RobotConfigCommand
# ---------------------------------------------------------------------------

class TestRobotConfigCommand:
    def test_defaults(self):
        cmd = RobotConfigCommand()
        assert cmd.robot_type  == "CR6"
        assert cmd.scale       == pytest.approx(1.0)
        assert cmd.payload_kg  == pytest.approx(0.0)
        assert cmd.payload_inertia is None

    def test_cr4_config(self):
        cmd = RobotConfigCommand(robot_type="CR4", scale=1.5, payload_kg=10.0)
        assert cmd.robot_type == "CR4"
        assert cmd.scale      == pytest.approx(1.5)

    def test_with_friction_coeffs(self):
        cmd = RobotConfigCommand(friction_coeffs=[0.1, 0.1, 0.05, 0.05, 0.02, 0.02])
        assert len(cmd.friction_coeffs) == 6

    def test_with_payload_inertia_dict(self):
        pi = {"box_size_xyz_m": [0.1, 0.1, 0.05], "com_from_tcp": [0, 0, -0.025]}
        cmd = RobotConfigCommand(payload_inertia=pi)
        assert cmd.payload_inertia == pi


# ---------------------------------------------------------------------------
# MotorSpec
# ---------------------------------------------------------------------------

class TestMotorSpec:
    def test_minimal_valid(self):
        m = MotorSpec(id="M_200W", nominal_torque_Nm=0.637, nominal_speed_rpm=3000)
        assert m.id                 == "M_200W"
        assert m.nominal_torque_Nm  == pytest.approx(0.637)
        assert m.nominal_speed_rpm  == pytest.approx(3000)
        assert m.compatible_gearboxes == []

    def test_with_optional_fields(self):
        m = MotorSpec(
            id="M_400W",
            nominal_torque_Nm=1.27,
            nominal_speed_rpm=3000,
            mass_kg=1.2,
            compatible_gearboxes=["HD_20", "HD_32"],
        )
        assert m.mass_kg == pytest.approx(1.2)
        assert "HD_20" in m.compatible_gearboxes

    def test_missing_required_fields_raise(self):
        with pytest.raises(ValidationError):
            MotorSpec(id="M_x")  # missing torque and speed


# ---------------------------------------------------------------------------
# GearboxSpec
# ---------------------------------------------------------------------------

class TestGearboxSpec:
    def test_minimal_valid(self):
        g = GearboxSpec(id="HD_20", ratios=[50, 100, 160])
        assert g.id      == "HD_20"
        assert g.ratios  == [50, 100, 160]
        assert g.efficiency is None

    def test_with_efficiency(self):
        g = GearboxSpec(id="HD_32", ratios=[100], efficiency=0.85, mass_kg=2.0)
        assert g.efficiency == pytest.approx(0.85)
        assert g.mass_kg    == pytest.approx(2.0)

    def test_missing_id_raises(self):
        with pytest.raises(ValidationError):
            GearboxSpec(ratios=[50])


# ---------------------------------------------------------------------------
# ActuatorLibraryUpdate
# ---------------------------------------------------------------------------

class TestActuatorLibraryUpdate:
    def test_all_none_by_default(self):
        upd = ActuatorLibraryUpdate()
        assert upd.motors   is None
        assert upd.gearboxes is None
        assert upd.compatibility_matrix is None

    def test_partial_update(self):
        upd = ActuatorLibraryUpdate(motors=[{"id": "M1", "rated_torque_Nm": 1.0}])
        assert len(upd.motors) == 1
        assert upd.gearboxes   is None


# ---------------------------------------------------------------------------
# ActuatorSelectionRequest
# ---------------------------------------------------------------------------

class TestActuatorSelectionRequest:
    def test_defaults(self):
        req = ActuatorSelectionRequest()
        assert req.safety_factor_torque == pytest.approx(1.5)
        assert req.safety_factor_speed  == pytest.approx(1.2)

    def test_custom_factors(self):
        req = ActuatorSelectionRequest(safety_factor_torque=2.0, safety_factor_speed=1.5)
        assert req.safety_factor_torque == pytest.approx(2.0)
        assert req.safety_factor_speed  == pytest.approx(1.5)
