"""
test_inertial_params.py
=======================
Unit tests for ``robot_core.inertial_params``.

All tests are pure NumPy and do **not** require Pinocchio.
"""

import numpy as np
import pytest

# robot_core.__init__ imports pinocchio at module level; skip gracefully.
pytest.importorskip("pinocchio", reason="Pinocchio not available — activate robodimm_env")

from robot_core.inertial_params import (
    box_inertia,
    cylinder_inertia,
    hollow_cylinder_inertia,
    get_cr4_serial5_reference_params,
    get_cr4_closed_loop_joint_inertial_params,
    CR4_SERIAL5_REF,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def assert_diagonal(I, msg=""):
    """Assert that a 3×3 matrix is diagonal (off-diagonal entries ≈ 0)."""
    off_diag = I - np.diag(np.diag(I))
    assert np.allclose(off_diag, 0.0), f"Matrix not diagonal{': ' if msg else ''}{msg}\n{I}"


def assert_positive_definite_diagonal(I, msg=""):
    """Assert that all diagonal elements are strictly positive."""
    diag = np.diag(I)
    assert np.all(diag > 0), (
        f"Non-positive diagonal element{': ' if msg else ''}{msg}: {diag}"
    )


# ---------------------------------------------------------------------------
# box_inertia
# ---------------------------------------------------------------------------

class TestBoxInertia:
    """Tests for the rectangular box inertia tensor."""

    def test_unit_cube_result(self):
        """For a 1 kg unit cube (1×1×1 m), all principal moments equal 1/6 kg·m²."""
        I = box_inertia(mass=1.0, width=1.0, height=1.0, depth=1.0)
        expected = 1.0 / 6.0
        np.testing.assert_allclose(np.diag(I), [expected, expected, expected], rtol=1e-12)

    def test_shape_is_3x3(self):
        I = box_inertia(1.0, 0.2, 0.3, 0.4)
        assert I.shape == (3, 3)

    def test_result_is_diagonal(self):
        I = box_inertia(2.0, 0.1, 0.2, 0.3)
        assert_diagonal(I)

    def test_all_positive_principal_moments(self):
        I = box_inertia(5.0, 0.4, 0.3, 0.2)
        assert_positive_definite_diagonal(I)

    def test_scaling_with_mass(self):
        """Inertia scales linearly with mass."""
        I1 = box_inertia(1.0, 0.2, 0.3, 0.5)
        I3 = box_inertia(3.0, 0.2, 0.3, 0.5)
        np.testing.assert_allclose(I3, 3.0 * I1, rtol=1e-12)

    def test_thin_plate_dominant_axis(self):
        """
        For a thin plate (depth ≈ 0), Izz ≈ m*(w²+h²)/12 dominates,
        and Iz should be much larger than Ixx ≈ m*h²/12.
        """
        I = box_inertia(mass=1.0, width=1.0, height=1.0, depth=1e-6)
        # Izz = m*(w^2+h^2)/12 = 1/6, Ixx = m*h^2/12 = 1/12
        assert np.diag(I)[2] > np.diag(I)[0]

    def test_known_asymmetric_values(self):
        """Verify explicit formula for a non-symmetric box."""
        m, w, h, d = 2.0, 0.3, 0.4, 0.5
        Ixx = m / 12.0 * (h**2 + d**2)
        Iyy = m / 12.0 * (w**2 + d**2)
        Izz = m / 12.0 * (w**2 + h**2)
        I = box_inertia(m, w, h, d)
        np.testing.assert_allclose(np.diag(I), [Ixx, Iyy, Izz], rtol=1e-12)


# ---------------------------------------------------------------------------
# cylinder_inertia
# ---------------------------------------------------------------------------

class TestCylinderInertia:
    """Tests for the solid cylinder inertia tensor."""

    def test_shape_is_3x3(self):
        I = cylinder_inertia(1.0, 0.1, 0.5)
        assert I.shape == (3, 3)

    def test_result_is_diagonal(self):
        I = cylinder_inertia(2.0, 0.05, 0.3, "z")
        assert_diagonal(I)

    def test_all_positive_principal_moments(self):
        I = cylinder_inertia(1.0, 0.1, 0.4, "z")
        assert_positive_definite_diagonal(I)

    def test_z_axis_symmetry(self):
        """For axis='z', Ixx == Iyy (transverse symmetry)."""
        I = cylinder_inertia(1.0, 0.1, 0.5, "z")
        assert np.diag(I)[0] == pytest.approx(np.diag(I)[1], rel=1e-12)

    def test_y_axis_symmetry(self):
        """For axis='y', Ixx == Izz (transverse symmetry)."""
        I = cylinder_inertia(1.0, 0.1, 0.5, "y")
        assert np.diag(I)[0] == pytest.approx(np.diag(I)[2], rel=1e-12)

    def test_x_axis_symmetry(self):
        """For axis='x', Iyy == Izz (transverse symmetry)."""
        I = cylinder_inertia(1.0, 0.1, 0.5, "x")
        assert np.diag(I)[1] == pytest.approx(np.diag(I)[2], rel=1e-12)

    def test_known_values_z_axis(self):
        """Verify formula: I_axial = m*r²/2, I_perp = m*(3r²+h²)/12."""
        m, r, h = 1.0, 0.1, 0.5
        I_perp  = m / 12.0 * (3 * r**2 + h**2)
        I_axial = m / 2.0  * r**2
        I = cylinder_inertia(m, r, h, "z")
        np.testing.assert_allclose(np.diag(I), [I_perp, I_perp, I_axial], rtol=1e-12)

    def test_scaling_with_mass(self):
        I1 = cylinder_inertia(1.0, 0.05, 0.2, "z")
        I4 = cylinder_inertia(4.0, 0.05, 0.2, "z")
        np.testing.assert_allclose(I4, 4.0 * I1, rtol=1e-12)


# ---------------------------------------------------------------------------
# hollow_cylinder_inertia
# ---------------------------------------------------------------------------

class TestHollowCylinderInertia:
    """Tests for the hollow cylinder (tube) inertia tensor."""

    def test_shape_is_3x3(self):
        I = hollow_cylinder_inertia(1.0, 0.1, 0.05, 0.3)
        assert I.shape == (3, 3)

    def test_result_is_diagonal(self):
        I = hollow_cylinder_inertia(1.0, 0.1, 0.05, 0.3, "z")
        assert_diagonal(I)

    def test_zero_inner_radius_matches_solid_cylinder(self):
        """With r_inner=0, hollow cylinder must equal a solid cylinder."""
        m, r, h = 2.0, 0.08, 0.4
        I_hollow = hollow_cylinder_inertia(m, r, 0.0, h, "z")
        I_solid  = cylinder_inertia(m, r, h, "z")
        np.testing.assert_allclose(I_hollow, I_solid, rtol=1e-10)

    def test_larger_hole_gives_larger_axial_inertia(self):
        """
        For the same outer radius and mass, a larger inner radius (thinner
        wall at larger radius) increases the axial moment of inertia.
        """
        m, r_out, h = 1.0, 0.1, 0.3
        I_small_hole = hollow_cylinder_inertia(m, r_out, 0.02, h, "z")
        I_large_hole = hollow_cylinder_inertia(m, r_out, 0.08, h, "z")
        assert np.diag(I_large_hole)[2] > np.diag(I_small_hole)[2]

    def test_positive_principal_moments(self):
        I = hollow_cylinder_inertia(3.0, 0.12, 0.06, 0.5, "y")
        assert_positive_definite_diagonal(I)


# ---------------------------------------------------------------------------
# get_cr4_serial5_reference_params
# ---------------------------------------------------------------------------

class TestCR4Serial5Params:
    """Tests for the CR4 reference inertial parameter function."""

    def test_scale_one_matches_reference(self):
        """At scale=1.0, masses and inertias must equal the reference constants."""
        params = get_cr4_serial5_reference_params(scale=1.0)
        for i, ref_mass in enumerate(CR4_SERIAL5_REF["masses"]):
            assert params["masses"][i] == pytest.approx(ref_mass, rel=1e-12)
        for i, ref_I in enumerate(CR4_SERIAL5_REF["inertias"]):
            np.testing.assert_allclose(params["inertias"][i], ref_I, rtol=1e-12)

    def test_five_links(self):
        """CR4 serial-5 model must have exactly 5 links."""
        params = get_cr4_serial5_reference_params()
        assert len(params["masses"])  == 5
        assert len(params["inertias"]) == 5
        assert len(params["coms"])     == 5

    def test_mass_scales_cubically_by_default(self):
        """Default exponent is 3 → masses scale as s³."""
        params_s2 = get_cr4_serial5_reference_params(scale=2.0)
        params_s1 = get_cr4_serial5_reference_params(scale=1.0)
        for m2, m1 in zip(params_s2["masses"], params_s1["masses"]):
            assert m2 == pytest.approx(8.0 * m1, rel=1e-12)

    def test_inertia_scales_as_s5_by_default(self):
        """Default inertia exponent is m_exp+2=5 → inertias scale as s⁵."""
        params_s2 = get_cr4_serial5_reference_params(scale=2.0)
        params_s1 = get_cr4_serial5_reference_params(scale=1.0)
        for I2, I1 in zip(params_s2["inertias"], params_s1["inertias"]):
            np.testing.assert_allclose(I2, 32.0 * I1, rtol=1e-12)

    def test_com_scales_linearly(self):
        """Centre-of-mass positions scale linearly with the scale factor."""
        s = 3.0
        params_s = get_cr4_serial5_reference_params(scale=s)
        params_1 = get_cr4_serial5_reference_params(scale=1.0)
        for c_s, c_1 in zip(params_s["coms"], params_1["coms"]):
            np.testing.assert_allclose(c_s, s * c_1, rtol=1e-12)

    def test_invalid_scale_raises(self):
        with pytest.raises(ValueError):
            get_cr4_serial5_reference_params(scale=0.0)
        with pytest.raises(ValueError):
            get_cr4_serial5_reference_params(scale=-1.0)

    def test_custom_mass_exponent(self):
        """Custom mass_scale_exp changes mass scaling accordingly."""
        params_exp4 = get_cr4_serial5_reference_params(scale=2.0, structural_mass_scale_exp=4.0)
        params_exp3 = get_cr4_serial5_reference_params(scale=2.0, structural_mass_scale_exp=3.0)
        # exp=4 → masses *= 16; exp=3 → masses *= 8 → ratio = 2
        for m4, m3 in zip(params_exp4["masses"], params_exp3["masses"]):
            assert m4 == pytest.approx(2.0 * m3, rel=1e-12)

    def test_all_masses_positive(self):
        for s in [0.5, 1.0, 2.0]:
            params = get_cr4_serial5_reference_params(scale=s)
            for m in params["masses"]:
                assert m > 0.0

    def test_all_inertias_positive_definite(self):
        params = get_cr4_serial5_reference_params(scale=1.0)
        for I in params["inertias"]:
            assert_positive_definite_diagonal(I)


# ---------------------------------------------------------------------------
# get_cr4_closed_loop_joint_inertial_params
# ---------------------------------------------------------------------------

class TestCR4ClosedLoopParams:
    """Tests for the closed-loop joint inertial parameter mapping."""

    EXPECTED_JOINTS = {"J1", "J2", "J3", "J_aux", "J4", "J3real", "J1p", "J2p"}

    def test_returns_all_expected_joints(self):
        params = get_cr4_closed_loop_joint_inertial_params(scale=1.0)
        assert set(params.keys()) == self.EXPECTED_JOINTS

    def test_each_joint_has_required_fields(self):
        params = get_cr4_closed_loop_joint_inertial_params(scale=1.0)
        for joint, data in params.items():
            assert "mass"    in data, f"Missing 'mass' for {joint}"
            assert "com"     in data, f"Missing 'com' for {joint}"
            assert "inertia" in data, f"Missing 'inertia' for {joint}"

    def test_all_masses_positive(self):
        params = get_cr4_closed_loop_joint_inertial_params(scale=1.0)
        for joint, data in params.items():
            assert data["mass"] > 0.0, f"Non-positive mass for {joint}"

    def test_inertia_tensors_are_3x3(self):
        params = get_cr4_closed_loop_joint_inertial_params(scale=1.0)
        for joint, data in params.items():
            assert data["inertia"].shape == (3, 3), (
                f"Wrong inertia shape for {joint}: {data['inertia'].shape}"
            )

    def test_main_link_masses_match_serial5(self):
        """J1..J4 masses in closed-loop model must match serial-5 reference."""
        cl_params = get_cr4_closed_loop_joint_inertial_params(scale=1.0)
        s5_params  = get_cr4_serial5_reference_params(scale=1.0)
        joint_order = ["J1", "J2", "J3", "J_aux", "J4"]
        for i, jname in enumerate(joint_order):
            assert cl_params[jname]["mass"] == pytest.approx(
                s5_params["masses"][i], rel=1e-12
            ), f"Mass mismatch for {jname}"

    def test_passive_links_have_tiny_mass(self):
        """J3real, J1p, J2p are passive/virtual links and should be very light."""
        params = get_cr4_closed_loop_joint_inertial_params(scale=1.0)
        for virtual in ["J3real", "J1p"]:
            assert params[virtual]["mass"] < 0.01, (
                f"Virtual link {virtual} has unexpectedly large mass: {params[virtual]['mass']}"
            )

    def test_scaling_affects_all_joints(self):
        """With scale=2.0 and default exponent=3, all masses should be 8× larger."""
        p1 = get_cr4_closed_loop_joint_inertial_params(scale=1.0)
        p2 = get_cr4_closed_loop_joint_inertial_params(scale=2.0)
        for joint in self.EXPECTED_JOINTS:
            assert p2[joint]["mass"] == pytest.approx(
                8.0 * p1[joint]["mass"], rel=1e-10
            ), f"Mass scaling failed for {joint}"
