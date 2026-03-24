"""
test_backend_utils.py
=====================
Unit tests for ``backend.utils``.

The coordinate-conversion helpers and demo-target generators are pure
Python / NumPy: they don't require the web server to be running, but the
``create_demo_targets_and_program`` function needs Pinocchio to compute FK.
"""

import numpy as np
import pytest

# backend.utils imports pinocchio at module level; skip gracefully.
pytest.importorskip("pinocchio", reason="Pinocchio not available — activate robodimm_env")

from backend.utils import (
    get_frontend_nq,
    q_pink_to_frontend,
    q_frontend_to_pink,
    map_joint_index_frontend_to_pink,
)


# ---------------------------------------------------------------------------
# get_frontend_nq
# ---------------------------------------------------------------------------

class TestGetFrontendNq:
    """Tests for frontend DOF count helper."""

    def test_cr4_returns_four(self):
        # CR4 always returns 4 regardless of model
        assert get_frontend_nq("CR4", None) == 4

    def test_cr6_returns_model_nq(self):
        """For CR6, the function returns model_pink.nq."""
        class _FakeModel:
            nq = 6
        assert get_frontend_nq("CR6", _FakeModel()) == 6


# ---------------------------------------------------------------------------
# q_pink_to_frontend  (CR4)
# ---------------------------------------------------------------------------

class TestQPinkToFrontendCR4:
    """
    CR4 Pink format: [J1, J2, J3_rel, J_aux, J4]
    Frontend format: [J1, J2, J3real, J4_rs]

    Relations:
        J3real  = J2 + J3_rel
        J4_rs   = -J4
    """

    def test_zero_config(self):
        q_pink = np.zeros(5)
        result = q_pink_to_frontend(q_pink, "CR4")
        np.testing.assert_allclose(result, np.zeros(4), atol=1e-14)

    def test_j3real_computed_correctly(self):
        q_pink = np.array([0.0, 0.5, 0.3, -0.8, 0.0])  # J2=0.5, J3_rel=0.3
        result = q_pink_to_frontend(q_pink, "CR4")
        assert result[2] == pytest.approx(0.8, abs=1e-14)  # J3real = 0.5+0.3

    def test_j4_inverted(self):
        q_pink = np.array([0.0, 0.0, 0.0, 0.0, 1.571])
        result = q_pink_to_frontend(q_pink, "CR4")
        assert result[3] == pytest.approx(-1.571, abs=1e-14)

    def test_j1_passthrough(self):
        q_pink = np.array([0.9, 0.0, 0.0, 0.0, 0.0])
        result = q_pink_to_frontend(q_pink, "CR4")
        assert result[0] == pytest.approx(0.9, abs=1e-14)

    def test_output_length_four(self):
        result = q_pink_to_frontend(np.zeros(5), "CR4")
        assert len(result) == 4

    def test_known_values(self):
        # Pink [J1=0.1, J2=0.2, J3_rel=0.3, J_aux=-0.5, J4=0.4]
        q_pink = np.array([0.1, 0.2, 0.3, -0.5, 0.4])
        result = q_pink_to_frontend(q_pink, "CR4")
        expected = np.array([0.1, 0.2, 0.5, -0.4])  # J3real=0.2+0.3, J4=-0.4
        np.testing.assert_allclose(result, expected, atol=1e-14)


class TestQPinkToFrontendCR6:
    """CR6 Pink format == Frontend format (passthrough)."""

    def test_identity_passthrough(self):
        q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        result = q_pink_to_frontend(q, "CR6")
        np.testing.assert_allclose(result, q, atol=1e-14)


# ---------------------------------------------------------------------------
# q_frontend_to_pink  (CR4)
# ---------------------------------------------------------------------------

class TestQFrontendToPinkCR4:
    """
    Frontend format: [J1, J2, J3real, J4]
    Pink format:     [J1, J2, J3_rel, J_aux, J4_pink]

    Relations:
        J3_rel  = J3real - J2
        J_aux   = -(J2 + J3_rel) = -J3real
        J4_pink = -J4
    """

    def test_zero_config(self):
        q_fe = np.zeros(4)
        q_p0 = np.zeros(5)
        result = q_frontend_to_pink(q_fe, q_p0, "CR4")
        np.testing.assert_allclose(result, np.zeros(5), atol=1e-14)

    def test_j3_rel_computed_correctly(self):
        q_fe = np.array([0.0, 0.5, 0.8, 0.0])  # J2=0.5, J3real=0.8
        q_p0 = np.zeros(5)
        result = q_frontend_to_pink(q_fe, q_p0, "CR4")
        assert result[2] == pytest.approx(0.3, abs=1e-14)  # J3_rel = 0.8-0.5

    def test_j_aux_computed_correctly(self):
        q_fe = np.array([0.0, 0.5, 0.8, 0.0])
        q_p0 = np.zeros(5)
        result = q_frontend_to_pink(q_fe, q_p0, "CR4")
        # J_aux = -(J2 + J3_rel) = -(0.5+0.3) = -0.8
        assert result[3] == pytest.approx(-0.8, abs=1e-14)

    def test_j4_inverted(self):
        q_fe = np.array([0.0, 0.0, 0.0, 1.571])
        q_p0 = np.zeros(5)
        result = q_frontend_to_pink(q_fe, q_p0, "CR4")
        assert result[4] == pytest.approx(-1.571, abs=1e-14)

    def test_output_length_five(self):
        result = q_frontend_to_pink(np.zeros(4), np.zeros(5), "CR4")
        assert len(result) == 5


class TestRoundTripCR4:
    """
    pink → frontend → pink must be an identity transformation.
    """

    @pytest.mark.parametrize("q_pink_input", [
        np.array([0.0,  0.0,  0.0,  0.0,  0.0]),
        np.array([0.5,  0.3,  0.2, -0.5,  1.2]),
        np.array([-0.9, 0.8, -0.5,  -0.3, -1.0]),
        np.array([1.0, -0.5,  0.7,  -0.2,  0.5]),
    ])
    def test_pink_frontend_pink_roundtrip(self, q_pink_input):
        q_fe = q_pink_to_frontend(q_pink_input, "CR4")
        q_recovered = q_frontend_to_pink(q_fe, q_pink_input.copy(), "CR4")
        np.testing.assert_allclose(q_recovered, q_pink_input, atol=1e-14,
                                   err_msg=f"Round-trip failed for q_pink={q_pink_input}")


class TestRoundTripCR6:
    """CR6 conversion is a passthrough, so round-trip is trivially exact."""

    def test_cr6_roundtrip(self):
        q = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6])
        q_fe = q_pink_to_frontend(q, "CR6")
        q_recov = q_frontend_to_pink(q_fe, q.copy(), "CR6")
        np.testing.assert_allclose(q_recov, q, atol=1e-14)


# ---------------------------------------------------------------------------
# map_joint_index_frontend_to_pink
# ---------------------------------------------------------------------------

class TestMapJointIndexFrontendToPink:
    """Tests for joint index mapping between frontend and Pink model."""

    @pytest.mark.parametrize("idx,expected", [
        (0, 0),   # J1
        (1, 1),   # J2
        (2, 2),   # J3
        (3, 4),   # J4 → Pink index 4
    ])
    def test_cr4_mapping(self, idx, expected):
        assert map_joint_index_frontend_to_pink(idx, "CR4") == expected

    def test_cr4_invalid_index(self):
        """Index ≥4 has no corresponding DOF in the CR4 model."""
        assert map_joint_index_frontend_to_pink(4, "CR4") == -1
        assert map_joint_index_frontend_to_pink(5, "CR4") == -1

    @pytest.mark.parametrize("idx", [0, 1, 2, 3, 4, 5])
    def test_cr6_passthrough(self, idx):
        """CR6 joints map 1-to-1."""
        assert map_joint_index_frontend_to_pink(idx, "CR6") == idx


# ---------------------------------------------------------------------------
# create_demo_targets_and_program  (requires Pinocchio)
# ---------------------------------------------------------------------------

class TestCreateDemoTargetsAndProgram:
    """
    Tests for the demo target/program generators.
    These functions call build_robot internally, so they require Pinocchio.
    """

    @pytest.fixture(autouse=True)
    def _skip_without_pinocchio(self):
        try:
            import pinocchio  # noqa: F401
            import pink       # noqa: F401
        except ImportError:
            pytest.skip("Pinocchio/Pink not available")

    def test_cr4_returns_seven_targets(self):
        from backend.utils import create_demo_targets_and_program
        targets, _ = create_demo_targets_and_program("CR4")
        assert len(targets) == 7

    def test_cr4_program_instructions_count(self):
        from backend.utils import create_demo_targets_and_program
        _, program = create_demo_targets_and_program("CR4")
        assert len(program) == 8  # 7 moves + return to HOME

    def test_cr6_returns_four_targets(self):
        from backend.utils import create_demo_targets_and_program
        targets, _ = create_demo_targets_and_program("CR6")
        assert len(targets) == 4

    def test_cr4_target_fields(self):
        from backend.utils import create_demo_targets_and_program
        targets, _ = create_demo_targets_and_program("CR4")
        for t in targets:
            assert "name"     in t
            assert "position" in t
            assert "rotation" in t
            assert "q"        in t
            # position is 3-element list, rotation is 9-element (flattened 3×3)
            assert len(t["position"]) == 3
            assert len(t["rotation"]) == 9
            assert len(t["q"])        == 4  # CR4 frontend q has 4 DOF

    def test_cr4_program_all_movej(self):
        from backend.utils import create_demo_targets_and_program
        _, program = create_demo_targets_and_program("CR4")
        for instr in program:
            assert instr["type"] == "MoveJ"
            assert "target_name" in instr
            assert "speed"       in instr
