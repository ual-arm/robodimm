"""
Session management and simulation state.
"""

import numpy as np
import pinocchio as pin
import pink
from pink.tasks import FrameTask, PostureTask
from typing import Optional, Any
from robot_core import build_robot, q_real_to_pink, q_pink_to_real


class SimulationSession:
    """Encapsulates robot state and configuration for a single user session."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        import time

        self.created_at = time.time()

        # Robot configuration state
        self.current_robot_type = "CR4"
        self.current_scale = 1.0
        self.current_payload_kg = 0.0
        self.current_payload_inertia: Optional[dict] = None
        self.current_friction_coeffs: Optional[list] = None
        self.current_reflected_inertia: Optional[list] = None
        self.current_coulomb_friction: Optional[list] = None
        self.current_motor_masses: Optional[list] = None
        self.current_motor_layout: str = "concentric_j2_j3act"
        self.current_iref_model_mode: str = "diag"
        self.current_structural_mass_scale_exp: float = 3.0
        self.current_structural_inertia_scale_exp: Optional[float] = None

        # Pinocchio/Pink model storage
        self.core: Optional[dict[str, Any]] = None
        self.model: Optional[Any] = None
        self.data: Optional[Any] = None
        self.constraint_model: Optional[Any] = None
        self.model_pink: Optional[Any] = None
        self.data_pink: Optional[Any] = None
        self.geom_model: Optional[Any] = None
        self.geom_data: Optional[Any] = None
        self.viz: Optional[Any] = None
        self.q: Optional[np.ndarray] = None
        self.ee_always_down = False
        self.robot_friction_coeffs = []

        # Pink configuration
        self.configuration: Optional[Any] = None
        self.ee_task: Optional[Any] = None
        self.post_task: Optional[Any] = None

        # Program State
        self.last_trajectory_data: Optional[dict[str, Any]] = None
        self.targets = []
        self.program = []

        # Animation state
        self.is_animating = False

        # Station geometries
        self.station_geometries = []

        # Initialize robot
        self.rebuild_robot()
        self.load_defaults()

    def rebuild_robot(self):
        """Build or rebuild the robot model based on current params."""
        self.core = build_robot(
            robot_type=self.current_robot_type,
            scale=self.current_scale,
            payload_kg=self.current_payload_kg,
            payload_inertia=self.current_payload_inertia,
            friction_coeffs=self.current_friction_coeffs,
            reflected_inertia=self.current_reflected_inertia,
            coulomb_friction=self.current_coulomb_friction,
            motor_masses=self.current_motor_masses,
            motor_layout=self.current_motor_layout,
            iref_model_mode=self.current_iref_model_mode,
            structural_mass_scale_exp=self.current_structural_mass_scale_exp,
            structural_inertia_scale_exp=self.current_structural_inertia_scale_exp,
        )
        self.model = self.core["model"]
        self.data = self.core["data"]
        self.constraint_model = self.core["constraint_model"]
        self.model_pink = self.core["model_pink"]
        self.data_pink = self.core["data_pink"]
        self.geom_model = self.core["geom_model"]
        self.geom_data = self.core["geom_data"]
        self.viz = self.core["viz"]
        self.q = self.core["q"]
        self.ee_always_down = self.core.get("ee_always_down", False)
        self.robot_friction_coeffs = self.core.get("friction_coeffs", [])
        self.robot_reflected_inertia = self.core.get("reflected_inertia", [])
        self.robot_coulomb_friction = self.core.get("coulomb_friction", [])
        self.robot_motor_masses = self.core.get("motor_masses", [])

        # Re-init Pink
        q_pink_init = q_real_to_pink(self.current_robot_type, self.model, self.q)
        self.configuration = pink.Configuration(
            self.model_pink, self.data_pink, q_pink_init
        )

        self.ee_task = FrameTask(
            "end_effector", position_cost=10.0, orientation_cost=0.0
        )
        self.post_task = PostureTask(cost=1e-3)
        self.post_task.set_target(q_pink_init)

    def load_defaults(self):
        """Load demo targets."""
        from .utils import create_demo_targets_and_program, q_frontend_to_pink

        t, p = create_demo_targets_and_program(self.current_robot_type)
        self.targets = t
        self.program = p

        # Start session at first program target for deterministic DEMO/PRO parity.
        if self.targets:
            q0_frontend = np.array(self.targets[0].get("q", []), dtype=float)
            if q0_frontend.size > 0:
                q0_pink = q_frontend_to_pink(
                    q0_frontend, self.configuration.q.copy(), self.current_robot_type
                )
                self.configuration.q = q0_pink
                self.post_task.set_target(self.configuration.q)
                self.q[:] = q_pink_to_real(
                    self.current_robot_type,
                    self.model,
                    self.data,
                    self.constraint_model,
                    q0_pink,
                )


def get_session(session_id: str = "default"):
    """
    FastAPI dependency to get or create a simulation session.

    Parameters
    ----------
    session_id : str
        Session identifier (default: "default")

    Returns
    -------
    SimulationSession
        The session for the given ID
    """
    if session_id not in sessions:
        sessions[session_id] = SimulationSession(session_id)
    return sessions[session_id]


# In-memory session store
sessions = {}
