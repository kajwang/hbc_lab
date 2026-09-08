from __future__ import annotations

import torch
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg, VecEnvStepReturn
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import math as math_utils

from hbc_lab.assets.objects import APPLE_OBJECT_FRAME_OFFSET_Z, OBJECT_PLATFORM_HEIGHT, OBJECT_PLATFORM_SIZE
from hbc_lab.assets.robots.unitree import G1_29DOF_BODY_JOINT_NAMES

from hbc_lab.tasks.manager_based.skill.contact_labels import ContactLabel, ContactMode
from ..mdp.contact_progress import (
    compute_active_hand_grasp_progress,
    symmetric_parallel_gripper_orientation_error,
)
from ..mdp.drc_math import compute_drc_weights, update_ema
from ..mdp.domain_randomization_curriculum import advance_curriculum_level
from ..mdp.gripper import Dex1GripperController
from ..mdp.grasp_references import (
    accessible_candidate_weights,
    assign_candidate_indices,
    candidate_evaluation_yaws,
    compose_object_grasp_pose,
    load_grasp_reference_library,
    select_candidate_indices,
)
from ..mdp.high_level_actions import HighLevelActionLimits, HighLevelCommandState, decode_high_level_action
from ..mdp.low_level_observations import G1SphericalPostureLowLevelObsBuilder
from ..mdp.low_level_policy import LowLevelPolicyWrapper
from ..mdp.object_mass_curriculum import balanced_active_hand_progress, update_monotonic_curriculum_level
from ..mdp.object_shape_bps import OBJECT_SHAPE_SPECS, decode_shape_scale_variant_indices, load_directional_bps_data
from ..mdp.scenes import (
    DEX1_LINK_CONTACT_SENSOR_NAMES,
    HAND_CENTER_FRAME_NAME,
    LEFT_GRIPPER_CONTACT_SENSOR_NAMES,
    RIGHT_GRIPPER_CONTACT_SENSOR_NAMES,
    SHALLOW_TRAY_SIZE,
    SHALLOW_TRAY_WALL_HEIGHT,
    SHALLOW_TRAY_WALL_THICKNESS,
)
from ..mdp.safety import nested_nonfinite_ratio, nonfinite_ratio, sanitize_nested_tensors, sanitize_tensor
from ...pose_motion import flatten_motion_frame_markers


TARGET_OBJECT_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/G1Dex1HierDrc/object_goal",
    markers={
        "target": sim_utils.SphereCfg(
            radius=0.055,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 1.0)),
        ),
    },
)
LEFT_CONTACT_TARGET_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/G1Dex1HierDrc/left_contact_target",
    markers={
        "target": sim_utils.SphereCfg(
            radius=0.035,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.35, 1.0)),
        ),
    },
)
RIGHT_CONTACT_TARGET_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/G1Dex1HierDrc/right_contact_target",
    markers={
        "target": sim_utils.SphereCfg(
            radius=0.035,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.55, 0.05)),
        ),
    },
)


def _motion_frame_marker(scale: float):
    marker = FRAME_MARKER_CFG.markers["frame"].copy()
    marker.scale = (scale, scale, scale)
    return marker


MOTION_FRAME_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/G1Dex1HierDrc/motion_frames",
    markers={
        "current": _motion_frame_marker(0.08),
        "keyframe_0": _motion_frame_marker(0.12),
        "keyframe_1": _motion_frame_marker(0.16),
    },
)

GRASP_REFERENCE_DIAGNOSTIC_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/G1Dex1HierDrc/grasp_reference_diagnostics",
    markers={
        "reference": _motion_frame_marker(0.16),
        "actual": _motion_frame_marker(0.08),
    },
)

PLATFORM_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/G1Dex1HierDrc/support_platforms",
    markers={
        "platform": sim_utils.CuboidCfg(
            size=(1.0, 1.0, 1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.24, 0.26, 0.28)),
        ),
    },
)


class G1Dex1HierDrcEnv(ManagerBasedRLEnv):
    cfg: ManagerBasedRLEnvCfg

    body_joint_names = G1_29DOF_BODY_JOINT_NAMES

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        if hasattr(cfg, "apply_object_shape_encoding"):
            cfg.apply_object_shape_encoding()
        cfg.apply_debug_visualization()
        self.object_shape_names = tuple(getattr(cfg, "object_shape_names", ()))
        self.object_shape_bps_basis_o = None
        self.object_stable_quat_wxyz = None
        self.object_shape_surface_offsets_o = None
        self.object_geometry_center_offsets_o = None
        self.object_geometry_bounds_min_o = None
        self.object_geometry_bounds_max_o = None
        self.object_principal_axes_o = None
        self.grasp_reference_positions_o = None
        self.grasp_reference_quaternions_o = None
        self.grasp_reference_scores = None
        self.grasp_reference_valid = None
        self.grasp_reference_modes = None
        self.selected_grasp_reference_index = torch.zeros(
            cfg.scene.num_envs,
            device=cfg.sim.device,
            dtype=torch.long,
        )
        self._grasp_candidate_mapping_logged = torch.zeros(
            cfg.scene.num_envs,
            device=cfg.sim.device,
            dtype=torch.bool,
        )
        if self.object_shape_names:
            shape_data = load_directional_bps_data(
                cfg.object_shape_bps_data_path,
                device=cfg.sim.device,
            )
            data_names = shape_data["shape_names"]
            missing_names = set(self.object_shape_names) - set(data_names)
            if missing_names:
                raise ValueError(f"Missing BPS metadata for object shapes: {sorted(missing_names)}")
            data_indices = torch.tensor(
                [data_names.index(name) for name in self.object_shape_names],
                device=cfg.sim.device,
                dtype=torch.long,
            )
            self.object_shape_surface_offsets_o = shape_data["surface_offsets_o"][data_indices]
            self.object_shape_bps_basis_o = shape_data["basis_points_o"]
            self.object_geometry_center_offsets_o = shape_data["geometry_center_offsets_o"][data_indices]
            self.object_geometry_bounds_min_o = shape_data["bounds_min_o"][data_indices]
            self.object_geometry_bounds_max_o = shape_data["bounds_max_o"][data_indices]
            self.object_principal_axes_o = shape_data["principal_axes_o"][data_indices]
            self.object_stable_quat_wxyz = torch.tensor(
                [OBJECT_SHAPE_SPECS[name].stable_quat_wxyz for name in self.object_shape_names],
                device=cfg.sim.device,
            )
        if getattr(cfg, "grasp_reference_enabled", False):
            if not self.object_shape_names:
                raise ValueError("Grasp references require object_shape_names")
            grasp_library = load_grasp_reference_library(
                cfg.grasp_reference_data_path,
                device=cfg.sim.device,
            )
            missing_names = set(self.object_shape_names) - set(grasp_library.shape_names)
            if missing_names:
                raise ValueError(f"Missing grasp references for object shapes: {sorted(missing_names)}")
            grasp_indices = torch.tensor(
                [grasp_library.shape_names.index(name) for name in self.object_shape_names],
                device=cfg.sim.device,
                dtype=torch.long,
            )
            self.grasp_reference_positions_o = grasp_library.positions_o[grasp_indices]
            self.grasp_reference_quaternions_o = grasp_library.quaternions_o[grasp_indices]
            self.grasp_reference_scores = grasp_library.scores[grasp_indices]
            self.grasp_reference_valid = grasp_library.valid[grasp_indices]
            self.grasp_reference_modes = grasp_library.modes[grasp_indices]
            if torch.any(self.grasp_reference_valid.sum(dim=-1) == 0):
                invalid_shapes = [
                    name
                    for name, has_valid in zip(
                        self.object_shape_names,
                        self.grasp_reference_valid.any(dim=-1).tolist(),
                    )
                    if not has_valid
                ]
                raise ValueError(f"Object shapes have no valid grasp references: {invalid_shapes}")
        self.d_active_hand = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.d_goal = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.c_contact = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.c_couple = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.c_grasp = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.c_physical_grasp = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.c_opposition = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.c_pinch = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.active_grip = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.active_left_finger_contact = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.active_right_finger_contact = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.active_opposition = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.left_hand_contact = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.right_hand_contact = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.active_left_finger_force = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.active_right_finger_force = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.active_contact_cos_sim = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.active_pinch_score = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.active_orientation_error = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.active_position_gate = torch.ones(cfg.scene.num_envs, device=cfg.sim.device)
        self.active_orientation_gate = torch.ones(cfg.scene.num_envs, device=cfg.sim.device)
        self.W_app = torch.ones(cfg.scene.num_envs, device=cfg.sim.device)
        self.W_couple = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.W_manip = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.object_initial_pos_w = torch.zeros(cfg.scene.num_envs, 3, device=cfg.sim.device)
        self.object_initial_root_z_w = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.object_target_pos_w = torch.zeros(cfg.scene.num_envs, 3, device=cfg.sim.device)
        self.object_fallen = torch.zeros(cfg.scene.num_envs, dtype=torch.bool, device=cfg.sim.device)
        self.active_object_index = torch.zeros(cfg.scene.num_envs, dtype=torch.long, device=cfg.sim.device)
        self.object_size_scale = torch.ones(cfg.scene.num_envs, device=cfg.sim.device)
        self.domain_randomization_curriculum_level = torch.full(
            (),
            cfg.domain_randomization_start_level,
            device=cfg.sim.device,
        )
        self.domain_randomization_grasp_rate = torch.zeros((), device=cfg.sim.device)
        self.domain_randomization_grasp_rate_ema = torch.zeros((), device=cfg.sim.device)
        self.domain_randomization_left_grasp_rate = torch.zeros((), device=cfg.sim.device)
        self.domain_randomization_right_grasp_rate = torch.zeros((), device=cfg.sim.device)
        self.object_mass = torch.full(
            (cfg.scene.num_envs,),
            cfg.object_mass_start_mass,
            device=cfg.sim.device,
        )
        self.object_mass_w_manip_ema = torch.full((), cfg.object_mass_start_w, device=cfg.sim.device)
        self.object_mass_curriculum_level = torch.full((), cfg.object_mass_start_w, device=cfg.sim.device)
        self.object_mass_w_manip_left_mean = torch.full((), cfg.object_mass_start_w, device=cfg.sim.device)
        self.object_mass_w_manip_right_mean = torch.full((), cfg.object_mass_start_w, device=cfg.sim.device)
        self.object_mass_w_manip_balanced = torch.full((), cfg.object_mass_start_w, device=cfg.sim.device)
        self.success_proximity_count = torch.zeros(cfg.scene.num_envs, dtype=torch.long, device=cfg.sim.device)
        self.task_succeeded = torch.zeros(cfg.scene.num_envs, dtype=torch.bool, device=cfg.sim.device)
        self.last_high_level_action = torch.zeros(cfg.scene.num_envs, cfg.action_dim, device=cfg.sim.device)
        self.prev_high_level_action = torch.zeros(cfg.scene.num_envs, cfg.action_dim, device=cfg.sim.device)
        self.command_rate = torch.zeros(cfg.scene.num_envs, 17, device=cfg.sim.device)
        self.previous_command_rate = torch.zeros_like(self.command_rate)
        self.command_acceleration = torch.zeros_like(self.command_rate)
        self.previous_command_acceleration = torch.zeros_like(self.command_rate)
        self.command_jerk = torch.zeros_like(self.command_rate)
        self._high_level_action_saturation_ratio = torch.zeros((), device=cfg.sim.device)
        self._high_level_action_abs_mean = torch.zeros((), device=cfg.sim.device)
        self._nonfinite_high_level_action_ratio = torch.zeros((), device=cfg.sim.device)
        self._nonfinite_low_level_action_ratio = torch.zeros((), device=cfg.sim.device)
        self._nonfinite_reward_ratio = torch.zeros((), device=cfg.sim.device)
        self._nonfinite_observation_ratio = torch.zeros((), device=cfg.sim.device)
        self._last_low_level_action = torch.zeros(cfg.scene.num_envs, len(self.body_joint_names), device=cfg.sim.device)
        self.active_hand = torch.zeros(cfg.scene.num_envs, dtype=torch.long, device=cfg.sim.device)
        self.contact_label = ContactLabel.from_active_hand(
            self.active_hand,
            contact_mode=ContactMode.GRASP,
        )
        super().__init__(cfg, render_mode, **kwargs)

        if self.object_shape_names:
            size_factors = tuple(cfg.object_size_scale_factors)
            shape_indices, scale_indices = decode_shape_scale_variant_indices(
                self.scene["object"].data.default_mass,
                num_scale_factors=len(size_factors),
            )
            if int(shape_indices.min()) < 0 or int(shape_indices.max()) >= len(self.object_shape_names):
                raise ValueError(
                    f"Decoded shape index outside [0, {len(self.object_shape_names) - 1}]: "
                    f"{int(shape_indices.min())}..{int(shape_indices.max())}"
                )
            self.active_object_index.copy_(shape_indices)
            size_factor_tensor = torch.tensor(size_factors, device=self.device)
            self.object_size_scale.copy_(size_factor_tensor[scale_indices])

        robot = self.scene["robot"]
        self.upper_body_joint_ids = list(
            robot.find_joints(G1_29DOF_BODY_JOINT_NAMES[12:], preserve_order=True)[0]
        )
        if len(self.upper_body_joint_ids) != 17:
            raise RuntimeError(f"Expected 17 upper-body joints, got {len(self.upper_body_joint_ids)}")
        self.left_shoulder_body_id = robot.find_bodies("left_shoulder_roll_link")[0][0]
        self.right_shoulder_body_id = robot.find_bodies("right_shoulder_roll_link")[0][0]
        self.torso_body_id = robot.find_bodies("torso_link")[0][0]
        self.high_level_command = self.command_manager.get_term("high_level")
        self.active_hand = self.high_level_command.active_hand.clone()
        self.command_state = HighLevelCommandState(
            base_velocity=self.high_level_command.base_velocity.clone(),
            posture_command=self.high_level_command.posture_command.clone(),
            left_hand_center_pose_a=self.high_level_command.left_hand_center_pose_a.clone(),
            right_hand_center_pose_a=self.high_level_command.right_hand_center_pose_a.clone(),
            left_grip=self.high_level_command.left_grip.clone(),
            right_grip=self.high_level_command.right_grip.clone(),
        )
        self.action_limits = HighLevelActionLimits()
        self.low_level_policy = None
        if getattr(cfg, "low_level_policy_path", ""):
            self.low_level_policy = LowLevelPolicyWrapper(cfg.low_level_policy_path, self.device)
        elif not getattr(cfg, "allow_missing_low_level_policy", False):
            raise ValueError(
                "A frozen low-level policy is required for G1Dex1HierDrcEnv. "
                "Set env.low_level_policy_path=/path/to/exported_low_level_policy.pt, "
                "or set env.allow_missing_low_level_policy=True only for zero-action debugging."
            )
        self.low_level_obs_builder = G1SphericalPostureLowLevelObsBuilder(
            self,
            history_length=cfg.low_level_obs_history_length,
            finite_obs_clip=cfg.finite_obs_clip,
        )
        self.gripper_controller = Dex1GripperController(robot, self.device)
        self._joint_ids = None
        self._default_joint_pos = None
        self._reset_contact_accumulators()
        self.target_pose_visualizer = None
        self.left_contact_target_visualizer = None
        self.right_contact_target_visualizer = None
        self.motion_frame_visualizer = None
        self.grasp_reference_pose_visualizer = None
        self.platform_visualizer = None

    def _reset_contact_accumulators(self) -> None:
        self._step_left_contact = torch.zeros(self.num_envs, device=self.device)
        self._step_right_contact = torch.zeros(self.num_envs, device=self.device)
        self._step_left_pinch = torch.zeros(self.num_envs, device=self.device)
        self._step_right_pinch = torch.zeros(self.num_envs, device=self.device)
        self._step_left_left_force_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._step_left_right_force_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._step_right_left_force_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._step_right_right_force_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._step_link_contact = {
            f"{side}_{link_name}": torch.zeros(self.num_envs, device=self.device)
            for side, link_name, _ in DEX1_LINK_CONTACT_SENSOR_NAMES
        }

    def _get_low_level_joint_info(self):
        if self._joint_ids is None:
            robot = self.scene["robot"]
            joint_ids, joint_names = robot.find_joints(self.body_joint_names, preserve_order=False)
            if len(joint_ids) != len(self.body_joint_names):
                raise RuntimeError(
                    f"Expected {len(self.body_joint_names)} low-level action joints, got {len(joint_ids)}: {joint_names}"
                )
            self._joint_ids = list(joint_ids)
            self._default_joint_pos = robot.data.default_joint_pos[:, self._joint_ids]
        return self._joint_ids, self._default_joint_pos

    def _sum_sensor_force(self, sensor_name: str) -> torch.Tensor:
        sensor = self.scene.sensors[sensor_name]
        if hasattr(sensor.data, "force_matrix_w") and sensor.data.force_matrix_w is not None:
            force = sensor.data.force_matrix_w[..., :3]
        else:
            force = sensor.data.net_forces_w[..., :3]
        while force.ndim > 2:
            force = force.sum(dim=1)
        return force

    def _dex1_gripper_finger_forces(self, side: str) -> tuple[torch.Tensor, torch.Tensor]:
        if side == "left":
            sensor_names = LEFT_GRIPPER_CONTACT_SENSOR_NAMES
        elif side == "right":
            sensor_names = RIGHT_GRIPPER_CONTACT_SENSOR_NAMES
        else:
            raise ValueError(f"Unsupported hand side: {side!r}")
        if len(sensor_names) != 2:
            raise RuntimeError(f"Dex1 {side} gripper expects two finger contact sensors, got {sensor_names}")
        return self._sum_sensor_force(sensor_names[0]), self._sum_sensor_force(sensor_names[1])

    def _contact_confidence_from_force(self, force_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        force = torch.norm(force_w, dim=-1)
        contact = 1.0 - torch.exp(-force / self.cfg.hand_contact_force_threshold)
        return contact.clamp(0.0, 1.0), force

    def _accumulate_link_contact_diagnostics(self) -> None:
        for side, link_name, sensor_name in DEX1_LINK_CONTACT_SENSOR_NAMES:
            key = f"{side}_{link_name}"
            force_w = self._sum_sensor_force(sensor_name)
            contact, _ = self._contact_confidence_from_force(force_w)
            self._step_link_contact[key] = torch.maximum(self._step_link_contact[key], contact)

    def _accumulate_contact_components(self) -> None:
        left_left_force_w, left_right_force_w = self._dex1_gripper_finger_forces("left")
        right_left_force_w, right_right_force_w = self._dex1_gripper_finger_forces("right")
        left_progress = compute_active_hand_grasp_progress(
            left_gripper_left_force_w=left_left_force_w,
            left_gripper_right_force_w=left_right_force_w,
            right_gripper_left_force_w=right_left_force_w,
            right_gripper_right_force_w=right_right_force_w,
            active_hand=torch.zeros(self.num_envs, dtype=torch.long, device=self.device),
            left_grip=self.command_state.left_grip,
            right_grip=self.command_state.right_grip,
            left_distance=torch.zeros(self.num_envs, device=self.device),
            right_distance=torch.zeros(self.num_envs, device=self.device),
            force_threshold=self.cfg.hand_contact_force_threshold,
        )
        right_progress = compute_active_hand_grasp_progress(
            left_gripper_left_force_w=left_left_force_w,
            left_gripper_right_force_w=left_right_force_w,
            right_gripper_left_force_w=right_left_force_w,
            right_gripper_right_force_w=right_right_force_w,
            active_hand=torch.ones(self.num_envs, dtype=torch.long, device=self.device),
            left_grip=self.command_state.left_grip,
            right_grip=self.command_state.right_grip,
            left_distance=torch.zeros(self.num_envs, device=self.device),
            right_distance=torch.zeros(self.num_envs, device=self.device),
            force_threshold=self.cfg.hand_contact_force_threshold,
        )
        left_active = left_progress.contact > self._step_left_contact
        right_active = right_progress.contact > self._step_right_contact
        self._step_left_contact = torch.maximum(self._step_left_contact, left_progress.contact)
        self._step_right_contact = torch.maximum(self._step_right_contact, right_progress.contact)
        self._step_left_pinch = torch.maximum(self._step_left_pinch, left_progress.pinch)
        self._step_right_pinch = torch.maximum(self._step_right_pinch, right_progress.pinch)
        self._step_left_left_force_w = torch.where(left_active.unsqueeze(-1), left_left_force_w, self._step_left_left_force_w)
        self._step_left_right_force_w = torch.where(left_active.unsqueeze(-1), left_right_force_w, self._step_left_right_force_w)
        self._step_right_left_force_w = torch.where(right_active.unsqueeze(-1), right_left_force_w, self._step_right_left_force_w)
        self._step_right_right_force_w = torch.where(right_active.unsqueeze(-1), right_right_force_w, self._step_right_right_force_w)
        self._accumulate_link_contact_diagnostics()

    def _apply_low_level_action(self, low_action: torch.Tensor) -> torch.Tensor:
        robot = self.scene["robot"]
        joint_ids, default_joint_pos = self._get_low_level_joint_info()
        low_action = sanitize_tensor(low_action, finite_clip=self.cfg.low_level_action_clip)
        joint_target = default_joint_pos + self.cfg.low_level_action_scale * low_action
        robot.set_joint_position_target(joint_target, joint_ids=joint_ids)
        return low_action

    def _sanitize_buffer(
        self,
        tensor: torch.Tensor,
        finite_clip: float | None = None,
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> None:
        tensor.copy_(sanitize_tensor(tensor, finite_clip=finite_clip))
        if min_value is not None or max_value is not None:
            tensor.clamp_(min=min_value, max=max_value)

    def _sanitize_progress_buffers(self) -> None:
        distance_clip = self.cfg.progress_distance_clip
        force_clip = self.cfg.contact_force_clip
        for name in ("d_active_hand", "d_goal"):
            self._sanitize_buffer(getattr(self, name), distance_clip, 0.0, distance_clip)
        for name in (
            "c_contact",
            "c_couple",
            "c_grasp",
            "c_physical_grasp",
            "c_opposition",
            "c_pinch",
            "active_grip",
            "active_left_finger_contact",
            "active_right_finger_contact",
            "active_opposition",
            "active_pinch_score",
            "active_position_gate",
            "active_orientation_gate",
            "left_hand_contact",
            "right_hand_contact",
            "W_app",
            "W_couple",
            "W_manip",
        ):
            self._sanitize_buffer(getattr(self, name), 1.0, 0.0, 1.0)
        for name in ("active_left_finger_force", "active_right_finger_force"):
            self._sanitize_buffer(getattr(self, name), force_clip, 0.0, force_clip)
        self._sanitize_buffer(self.active_contact_cos_sim, 1.0, -1.0, 1.0)
        self._sanitize_buffer(self.active_orientation_error, torch.pi, 0.0, torch.pi)
        for tensor in self._step_link_contact.values():
            self._sanitize_buffer(tensor, 1.0, 0.0, 1.0)

    def _sanitize_rollout_outputs(self) -> None:
        self._nonfinite_reward_ratio = nonfinite_ratio(self.reward_buf).detach()
        self._nonfinite_observation_ratio = nested_nonfinite_ratio(self.obs_buf).detach()
        self.reward_buf.copy_(sanitize_tensor(self.reward_buf, finite_clip=self.cfg.finite_reward_clip))
        self.obs_buf = sanitize_nested_tensors(self.obs_buf, finite_clip=self.cfg.finite_obs_clip)
        if "observations" in self.extras:
            self.extras["observations"] = sanitize_nested_tensors(
                self.extras["observations"],
                finite_clip=self.cfg.finite_obs_clip,
            )

    def _apply_gripper_command(self):
        self.gripper_controller.apply(self.command_state.left_grip, self.command_state.right_grip)

    def _apply_debug_gripper_override(self):
        if not getattr(self.cfg, "debug_fixed_gripper", False):
            return
        self.command_state.left_grip[:] = self.cfg.debug_fixed_left_grip
        self.command_state.right_grip[:] = self.cfg.debug_fixed_right_grip

    def _active_hand_center_pose_a(self) -> torch.Tensor:
        left_active = (self.active_hand == 0).unsqueeze(-1)
        return torch.where(
            left_active,
            self.command_state.left_hand_center_pose_a,
            self.command_state.right_hand_center_pose_a,
        )

    def _object_frame_pos_w(self) -> torch.Tensor:
        if self.object_geometry_center_offsets_o is not None:
            center_offset_o = (
                self.object_geometry_center_offsets_o[self.active_object_index]
                * self.object_size_scale.unsqueeze(-1)
            )
            return self._object_root_pos_w() + math_utils.quat_apply(self._object_root_quat_w(), center_offset_o)
        target_pos_w = self.scene["object_frame"].data.target_pos_w
        if target_pos_w.shape[1] == 1:
            return target_pos_w[:, 0, :]
        env_ids = torch.arange(self.num_envs, device=self.device)
        return target_pos_w[env_ids, self.active_object_index]

    def _object_root_pos_w(self) -> torch.Tensor:
        obj = self.scene["object"]
        if hasattr(obj.data, "object_pos_w"):
            env_ids = torch.arange(self.num_envs, device=self.device)
            return obj.data.object_pos_w[env_ids, self.active_object_index]
        return obj.data.root_pos_w

    def _object_root_quat_w(self) -> torch.Tensor:
        obj = self.scene["object"]
        if hasattr(obj.data, "object_quat_w"):
            env_ids = torch.arange(self.num_envs, device=self.device)
            return obj.data.object_quat_w[env_ids, self.active_object_index]
        return obj.data.root_quat_w

    def _object_fallen(self) -> torch.Tensor:
        object_root_z = self._object_root_pos_w()[:, 2]
        if self.object_geometry_center_offsets_o is not None:
            initial_root_z = self.object_initial_root_z_w
            platform_height = self.cfg.multishape_support_height
        else:
            initial_root_z = self.object_initial_pos_w[:, 2] - APPLE_OBJECT_FRAME_OFFSET_Z * self.object_size_scale
            platform_height = OBJECT_PLATFORM_HEIGHT
        fall_threshold = initial_root_z - 0.5 * platform_height
        return object_root_z < fall_threshold

    def _sample_grasp_reference_indices(self, env_ids: torch.Tensor) -> None:
        if self.grasp_reference_valid is None:
            return
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)
        fixed_candidate_indices = self.cfg.grasp_reference_candidate_indices
        if fixed_candidate_indices:
            self.selected_grasp_reference_index[env_ids] = assign_candidate_indices(
                env_ids,
                fixed_candidate_indices,
            )
            return
        shape_indices = self.active_object_index[env_ids]
        grasp_pos_o = self.grasp_reference_positions_o[shape_indices] * self.object_size_scale[env_ids, None, None]
        grasp_quat_o = self.grasp_reference_quaternions_o[shape_indices]
        num_candidates = grasp_pos_o.shape[1]
        object_root_pos_w = self._object_root_pos_w()[env_ids]
        object_root_quat_w = self._object_root_quat_w()[env_ids]
        candidate_positions_w, candidate_quaternions_w = compose_object_grasp_pose(
            object_root_pos_w[:, None, :].expand(-1, num_candidates, -1).reshape(-1, 3),
            object_root_quat_w[:, None, :].expand(-1, num_candidates, -1).reshape(-1, 4),
            grasp_pos_o.reshape(-1, 3),
            grasp_quat_o.reshape(-1, 4),
        )
        candidate_positions_w = candidate_positions_w.reshape(-1, num_candidates, 3)
        candidate_quaternions_w = candidate_quaternions_w.reshape(-1, num_candidates, 4)
        support_top_z = self.scene.env_origins[env_ids, 2] + self.cfg.multishape_support_height
        robot = self.scene["robot"]
        left_shoulder_positions_w = robot.data.body_pos_w[env_ids, self.left_shoulder_body_id]
        right_shoulder_positions_w = robot.data.body_pos_w[env_ids, self.right_shoulder_body_id]
        active_shoulder_positions_w = torch.where(
            (self.active_hand[env_ids] == 0).unsqueeze(-1),
            left_shoulder_positions_w,
            right_shoulder_positions_w,
        )
        candidate_weights = accessible_candidate_weights(
            candidate_positions_w=candidate_positions_w,
            candidate_quaternions_w=candidate_quaternions_w,
            candidate_scores=self.grasp_reference_scores[shape_indices],
            candidate_valid=self.grasp_reference_valid[shape_indices],
            object_positions_w=self._object_frame_pos_w()[env_ids],
            robot_positions_w=self.scene["robot"].data.root_pos_w[env_ids],
            robot_quaternions_w=math_utils.yaw_quat(self.scene["robot"].data.root_quat_w[env_ids]),
            active_hand=self.active_hand[env_ids],
            support_top_z=support_top_z,
            active_shoulder_positions_w=active_shoulder_positions_w,
        )
        selected = select_candidate_indices(
            candidate_weights,
            deterministic=self.cfg.grasp_reference_select_highest_weight,
        )
        self.selected_grasp_reference_index[env_ids] = selected
        self._log_candidate_evaluation_selection(env_ids, shape_indices, selected, candidate_weights)

    def _log_candidate_evaluation_selection(
        self,
        env_ids: torch.Tensor,
        shape_indices: torch.Tensor,
        selected: torch.Tensor,
        candidate_weights: torch.Tensor,
    ) -> None:
        if self.cfg.grasp_candidate_yaw_count < 1:
            return
        unlogged = ~self._grasp_candidate_mapping_logged[env_ids]
        if not bool(unlogged.any()):
            return
        local_rows = torch.nonzero(unlogged, as_tuple=False).flatten()
        normalized_weights = candidate_weights / candidate_weights.sum(dim=-1, keepdim=True)
        yaw_deg = torch.rad2deg(
            candidate_evaluation_yaws(env_ids, self.cfg.grasp_candidate_yaw_count)
        )
        for local_row in local_rows.tolist():
            env_id = int(env_ids[local_row])
            shape_index = int(shape_indices[local_row])
            candidate_index = int(selected[local_row])
            score = float(self.grasp_reference_scores[shape_index, candidate_index])
            mode = int(self.grasp_reference_modes[shape_index, candidate_index])
            probability = float(normalized_weights[local_row, candidate_index])
            print(
                f"[Grasp Candidate Play] env {env_id:02d}: yaw={float(yaw_deg[local_row]):5.1f} deg, "
                f"candidate={candidate_index:02d}, probability={probability:.3f}, "
                f"score={score:.3f}, mode={mode}",
                flush=True,
            )
        self._grasp_candidate_mapping_logged[env_ids[local_rows]] = True

    def _selected_grasp_reference_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.grasp_reference_positions_o is None:
            raise RuntimeError("Selected grasp pose requested without a grasp reference library")
        grasp_pos_o = self.grasp_reference_positions_o[
            self.active_object_index,
            self.selected_grasp_reference_index,
        ] * self.object_size_scale.unsqueeze(-1)
        grasp_quat_o = self.grasp_reference_quaternions_o[
            self.active_object_index,
            self.selected_grasp_reference_index,
        ]
        return compose_object_grasp_pose(
            self._object_root_pos_w(),
            self._object_root_quat_w(),
            grasp_pos_o,
            grasp_quat_o,
        )

    def _log_shape_diagnostics(self, grasp_progress: torch.Tensor) -> None:
        if not self.object_shape_names:
            return
        for shape_index, shape_name in enumerate(self.object_shape_names):
            mask = self.active_object_index == shape_index
            if not bool(mask.any()):
                continue
            stable_grasp = grasp_progress[mask] >= 0.5
            self.extras["log"][f"Shape/{shape_name}_grasp_rate"] = stable_grasp.float().mean()

    def _active_target_tracking_diagnostics(self) -> tuple[torch.Tensor, torch.Tensor]:
        left_active = (self.active_hand == 0).unsqueeze(-1)
        left_target_w = self.low_level_obs_builder._target_pos_w(
            self.command_state.left_hand_center_pose_a,
            "left",
            self.command_state.posture_command,
        )
        right_target_w = self.low_level_obs_builder._target_pos_w(
            self.command_state.right_hand_center_pose_a,
            "right",
            self.command_state.posture_command,
        )
        active_target_w = torch.where(left_active, left_target_w, right_target_w)

        hand_center_pos_w = self.scene[HAND_CENTER_FRAME_NAME].data.target_pos_w
        active_hand_center_w = torch.where(left_active, hand_center_pos_w[:, 0, :], hand_center_pos_w[:, 1, :])
        target_object_dist = torch.norm(active_target_w - self._object_frame_pos_w(), dim=-1)
        hand_center_tracking_error = torch.norm(active_hand_center_w - active_target_w, dim=-1)
        return target_object_dist, hand_center_tracking_error

    def _active_grasp_pose_diagnostics(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        zeros = torch.zeros_like(self.d_active_hand)
        if not self.cfg.grasp_reference_controls_target:
            return zeros, zeros, zeros

        left_active = (self.active_hand == 0).unsqueeze(-1)
        left_command_pos_w, left_command_quat_w = self.low_level_obs_builder._target_pose_w(
            self.command_state.left_hand_center_pose_a,
            "left",
            self.command_state.posture_command,
        )
        right_command_pos_w, right_command_quat_w = self.low_level_obs_builder._target_pose_w(
            self.command_state.right_hand_center_pose_a,
            "right",
            self.command_state.posture_command,
        )
        del left_command_pos_w, right_command_pos_w
        command_quat_w = torch.where(left_active, left_command_quat_w, right_command_quat_w)

        hand_center_quat_w = self.scene[HAND_CENTER_FRAME_NAME].data.target_quat_w
        actual_quat_w = torch.where(
            left_active,
            hand_center_quat_w[:, 0],
            hand_center_quat_w[:, 1],
        )
        reference_quat_w = torch.where(
            left_active,
            self.contact_label.target_orientation[:, 0],
            self.contact_label.target_orientation[:, 1],
        )
        return (
            symmetric_parallel_gripper_orientation_error(command_quat_w, reference_quat_w),
            symmetric_parallel_gripper_orientation_error(actual_quat_w, command_quat_w),
            symmetric_parallel_gripper_orientation_error(actual_quat_w, reference_quat_w),
        )

    def _log_high_level_diagnostics(self):
        left_active = self.active_hand == 0
        left_grip = self.command_state.left_grip.squeeze(-1)
        right_grip = self.command_state.right_grip.squeeze(-1)
        active_grip = torch.where(left_active, left_grip, right_grip)
        target_object_dist, hand_center_tracking_error = self._active_target_tracking_diagnostics()
        self.extras["log"]["HL/active_grip_mean"] = active_grip.mean()
        self.extras["log"]["HL/active_hand_center_target_object_dist"] = target_object_dist.mean()
        self.extras["log"]["HL/active_hand_center_tracking_error"] = hand_center_tracking_error.mean()

    def _compute_progress(self):
        self._update_contact_target_regions()
        object_pos_w = self._object_frame_pos_w()
        hand_center_pos_w = self.scene[HAND_CENTER_FRAME_NAME].data.target_pos_w
        left_pos_w = hand_center_pos_w[:, 0, :]
        right_pos_w = hand_center_pos_w[:, 1, :]
        target_region = self.contact_label.target_region
        target_orientation = self.contact_label.target_orientation
        left_distance = torch.norm(left_pos_w - target_region[:, 0, :], dim=-1)
        right_distance = torch.norm(right_pos_w - target_region[:, 1, :], dim=-1)
        if not self.cfg.grasp_reference_controls_target:
            left_orientation_error = torch.zeros_like(left_distance)
            right_orientation_error = torch.zeros_like(right_distance)
        else:
            hand_center_quat_w = self.scene[HAND_CENTER_FRAME_NAME].data.target_quat_w
            left_orientation_error = symmetric_parallel_gripper_orientation_error(
                hand_center_quat_w[:, 0],
                target_orientation[:, 0],
            )
            right_orientation_error = symmetric_parallel_gripper_orientation_error(
                hand_center_quat_w[:, 1],
                target_orientation[:, 1],
            )
        progress = compute_active_hand_grasp_progress(
            left_gripper_left_force_w=self._step_left_left_force_w,
            left_gripper_right_force_w=self._step_left_right_force_w,
            right_gripper_left_force_w=self._step_right_left_force_w,
            right_gripper_right_force_w=self._step_right_right_force_w,
            active_hand=self.active_hand,
            left_grip=self.command_state.left_grip,
            right_grip=self.command_state.right_grip,
            left_distance=left_distance,
            right_distance=right_distance,
            force_threshold=self.cfg.hand_contact_force_threshold,
            close_distance=self.cfg.close_distance,
            close_gate_width=self.cfg.close_gate_width,
            left_orientation_error=left_orientation_error,
            right_orientation_error=right_orientation_error,
            close_orientation_zero_error=self.cfg.close_orientation_zero_error,
            close_orientation_gate_width=self.cfg.close_orientation_gate_width,
        )
        self.d_active_hand = progress.distance
        self.active_grip = progress.grip
        self.active_left_finger_contact = progress.left_contact
        self.active_right_finger_contact = progress.right_contact
        self.active_left_finger_force = progress.left_force
        self.active_right_finger_force = progress.right_force
        self.active_opposition = progress.pinch
        self.active_pinch_score = progress.pinch_score
        self.active_contact_cos_sim = progress.cos_sim
        self.active_orientation_error = progress.orientation_error
        self.active_position_gate = progress.close_position_gate
        self.active_orientation_gate = progress.close_orientation_gate
        self.d_goal = torch.norm(object_pos_w - self.object_target_pos_w, dim=-1)
        self.left_hand_contact = self._step_left_contact
        self.right_hand_contact = self._step_right_contact
        self.c_contact = update_ema(self.c_contact, progress.contact, alpha=0.2)
        self.c_opposition = update_ema(self.c_opposition, progress.pinch, alpha=0.2)
        self.c_pinch = update_ema(self.c_pinch, progress.pinch, alpha=0.2)
        physical_grasp = progress.contact * progress.grip
        self.c_grasp = update_ema(self.c_grasp, physical_grasp, alpha=0.2)
        self.c_physical_grasp = update_ema(self.c_physical_grasp, physical_grasp, alpha=0.2)
        self.c_couple = update_ema(self.c_couple, physical_grasp, alpha=0.2)
        self.extras["log"]["GraspPose/pose_aligned_grasp_mean"] = progress.grasp.detach().mean()
        self.object_fallen = self._object_fallen()
        weights = compute_drc_weights(self.d_active_hand, self.c_couple, alpha=5.0)
        self.W_app = weights[:, 0]
        self.W_couple = weights[:, 1]
        self.W_manip = weights[:, 2]

    def _update_contact_target_regions(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if not self.cfg.grasp_reference_controls_target:
            object_pos_w = self._object_frame_pos_w()
            target_region = object_pos_w.unsqueeze(1).expand(-1, 2, -1)
            self.contact_label.set_target_region(env_ids, target_region[env_ids])
            return

        grasp_pos_w, grasp_quat_w = self._selected_grasp_reference_pose_w()
        target_region = grasp_pos_w.unsqueeze(1).expand(-1, 2, -1)
        target_orientation = grasp_quat_w.unsqueeze(1).expand(-1, 2, -1)
        self.contact_label.set_target_region_pose(
            env_ids,
            target_region[env_ids],
            target_orientation[env_ids],
        )

    def _update_object_mass_curriculum(self) -> None:
        if not getattr(self.cfg, "object_mass_curriculum_enabled", False):
            return
        if hasattr(self.scene["object"].data, "object_pos_w"):
            self.object_mass_curriculum_level.copy_(
                self.domain_randomization_curriculum_level * self.cfg.object_mass_anchor_w
            )
            return
        alpha = self.cfg.object_mass_w_manip_ema_alpha
        curriculum_progress = (
            self.c_physical_grasp.detach()
            if self.cfg.object_mass_use_physical_grasp
            else self.W_manip.detach()
        )
        left_mean, right_mean, balanced = balanced_active_hand_progress(
            curriculum_progress,
            self.active_hand,
        )
        self.object_mass_w_manip_left_mean.copy_(left_mean)
        self.object_mass_w_manip_right_mean.copy_(right_mean)
        self.object_mass_w_manip_balanced.copy_(balanced)
        next_ema, next_level = update_monotonic_curriculum_level(
            self.object_mass_curriculum_level,
            self.object_mass_w_manip_ema,
            balanced,
            alpha=alpha,
        )
        self.object_mass_w_manip_ema.copy_(next_ema)
        self.object_mass_curriculum_level.copy_(next_level)

    def _update_domain_randomization_curriculum(self) -> None:
        grasped = (self.c_grasp.detach() >= self.cfg.domain_randomization_grasp_threshold).float()
        left_rate, right_rate, balanced_rate = balanced_active_hand_progress(grasped, self.active_hand)
        self.domain_randomization_left_grasp_rate.copy_(left_rate)
        self.domain_randomization_right_grasp_rate.copy_(right_rate)
        self.domain_randomization_grasp_rate.copy_(balanced_rate)
        alpha = self.cfg.domain_randomization_grasp_ema_alpha
        self.domain_randomization_grasp_rate_ema.mul_(1.0 - alpha).add_(alpha * balanced_rate)
        if not getattr(self.cfg, "domain_randomization_curriculum_enabled", False):
            return
        if self.common_step_counter % self.cfg.domain_randomization_update_interval != 0:
            return

        next_level = advance_curriculum_level(
            float(self.domain_randomization_curriculum_level),
            float(self.domain_randomization_grasp_rate_ema),
            threshold=self.cfg.domain_randomization_advance_threshold,
            step=self.cfg.domain_randomization_level_step,
        )
        if next_level > float(self.domain_randomization_curriculum_level):
            self.domain_randomization_curriculum_level.fill_(next_level)
            self.domain_randomization_grasp_rate_ema.zero_()

    def _check_success(self):
        success_condition = (self.c_couple > self.cfg.success_couple_threshold) & (self.d_goal < self.cfg.success_distance)
        self.success_proximity_count[success_condition] += 1
        self.success_proximity_count[~success_condition] = 0
        self.task_succeeded = self.success_proximity_count >= self.cfg.success_steps

    def _reset_command_state(self, env_ids: torch.Tensor):
        self.command_state.base_velocity[env_ids] = self.high_level_command.base_velocity[env_ids]
        self.command_state.posture_command[env_ids] = self.high_level_command.posture_command[env_ids]
        self.command_state.left_hand_center_pose_a[env_ids] = self.high_level_command.left_hand_center_pose_a[env_ids]
        self.command_state.right_hand_center_pose_a[env_ids] = self.high_level_command.right_hand_center_pose_a[env_ids]
        self.command_state.left_grip[env_ids] = self.high_level_command.left_grip[env_ids]
        self.command_state.right_grip[env_ids] = self.high_level_command.right_grip[env_ids]

    def _reset_hier_buffers(self, env_ids: torch.Tensor):
        self.d_active_hand[env_ids] = 0.0
        self.d_goal[env_ids] = 0.0
        self.c_contact[env_ids] = 0.0
        self.c_couple[env_ids] = 0.0
        self.c_grasp[env_ids] = 0.0
        self.c_physical_grasp[env_ids] = 0.0
        self.c_opposition[env_ids] = 0.0
        self.c_pinch[env_ids] = 0.0
        self.active_grip[env_ids] = 0.0
        self.active_left_finger_contact[env_ids] = 0.0
        self.active_right_finger_contact[env_ids] = 0.0
        self.active_left_finger_force[env_ids] = 0.0
        self.active_right_finger_force[env_ids] = 0.0
        self.active_opposition[env_ids] = 0.0
        self.active_pinch_score[env_ids] = 0.0
        self.active_contact_cos_sim[env_ids] = 0.0
        self.active_orientation_error[env_ids] = 0.0
        self.active_position_gate[env_ids] = 1.0
        self.active_orientation_gate[env_ids] = 1.0
        self.left_hand_contact[env_ids] = 0.0
        self.right_hand_contact[env_ids] = 0.0
        self.object_fallen[env_ids] = False
        self.W_app[env_ids] = 1.0
        self.W_couple[env_ids] = 0.0
        self.W_manip[env_ids] = 0.0
        self.success_proximity_count[env_ids] = 0
        self.task_succeeded[env_ids] = False
        self.last_high_level_action[env_ids] = 0.0
        self.prev_high_level_action[env_ids] = 0.0
        self.command_rate[env_ids] = 0.0
        self.previous_command_rate[env_ids] = 0.0
        self.command_acceleration[env_ids] = 0.0
        self.previous_command_acceleration[env_ids] = 0.0
        self.command_jerk[env_ids] = 0.0
        self._last_low_level_action[env_ids] = 0.0
        self.low_level_obs_builder.reset(env_ids)
        self.active_hand[env_ids] = self.high_level_command.active_hand[env_ids]
        self._reset_command_state(env_ids)

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._reset_hier_buffers(env_ids)
        self._sample_grasp_reference_indices(env_ids)
        self._update_contact_target_regions(env_ids)

    def _update_target_pose_visualization(self) -> None:
        if getattr(self.cfg, "target_pose_debug_vis", False):
            if self.target_pose_visualizer is None:
                self.target_pose_visualizer = VisualizationMarkers(TARGET_OBJECT_MARKER_CFG)
                self.left_contact_target_visualizer = VisualizationMarkers(LEFT_CONTACT_TARGET_MARKER_CFG)
                self.right_contact_target_visualizer = VisualizationMarkers(RIGHT_CONTACT_TARGET_MARKER_CFG)
                self.target_pose_visualizer.set_visibility(True)
                self.left_contact_target_visualizer.set_visibility(True)
                self.right_contact_target_visualizer.set_visibility(True)

            left_target_region = self.contact_label.target_region[:, 0, :]
            right_target_region = self.contact_label.target_region[:, 1, :]
            left_active = self.contact_label.effector_mask[:, 0]
            right_active = self.contact_label.effector_mask[:, 1]
            left_target_scales = left_active.to(dtype=left_target_region.dtype).unsqueeze(-1).expand(-1, 3)
            right_target_scales = right_active.to(dtype=right_target_region.dtype).unsqueeze(-1).expand(-1, 3)
            self.target_pose_visualizer.visualize(self.object_target_pos_w)
            self.left_contact_target_visualizer.visualize(left_target_region, scales=left_target_scales)
            self.right_contact_target_visualizer.visualize(right_target_region, scales=right_target_scales)

        self._update_platform_pose_visualization()
        self._update_grasp_reference_pose_visualization()
        self._update_motion_frame_visualization()

    def _update_grasp_reference_pose_visualization(self) -> None:
        enabled = getattr(self.cfg, "grasp_reference_pose_debug_vis", False)
        if not enabled or self.grasp_reference_positions_o is None:
            if self.grasp_reference_pose_visualizer is not None:
                self.grasp_reference_pose_visualizer.set_visibility(False)
            return
        if self.grasp_reference_pose_visualizer is None:
            self.grasp_reference_pose_visualizer = VisualizationMarkers(
                GRASP_REFERENCE_DIAGNOSTIC_MARKER_CFG
            )

        left_active = (self.active_hand == 0).unsqueeze(-1)
        reference_pos_w, reference_quat_w = self._selected_grasp_reference_pose_w()

        hand_center = self.scene[HAND_CENTER_FRAME_NAME]
        actual_pos_w = torch.where(
            left_active,
            hand_center.data.target_pos_w[:, 0],
            hand_center.data.target_pos_w[:, 1],
        )
        actual_quat_w = torch.where(
            left_active,
            hand_center.data.target_quat_w[:, 0],
            hand_center.data.target_quat_w[:, 1],
        )

        positions = torch.cat((reference_pos_w, actual_pos_w), dim=0)
        orientations = torch.cat((reference_quat_w, actual_quat_w), dim=0)
        marker_indices = torch.cat(
            (
                torch.zeros(self.num_envs, dtype=torch.long, device=self.device),
                torch.ones(self.num_envs, dtype=torch.long, device=self.device),
            )
        )
        self.grasp_reference_pose_visualizer.set_visibility(True)
        self.grasp_reference_pose_visualizer.visualize(
            translations=positions,
            orientations=orientations,
            marker_indices=marker_indices,
        )

    def _update_platform_pose_visualization(self) -> None:
        if not getattr(self.cfg, "platform_pose_debug_vis", False):
            return
        if self.platform_visualizer is None:
            self.platform_visualizer = VisualizationMarkers(PLATFORM_MARKER_CFG)
            self.platform_visualizer.set_visibility(True)

        if self.object_shape_names:
            roots = torch.cat((self.object_initial_pos_w, self.object_target_pos_w), dim=0).clone()
            roots[:, 2] = torch.cat((self.scene.env_origins[:, 2], self.scene.env_origins[:, 2]))
            roots[:, 2] += 0.5 * self.cfg.multishape_support_height
            if self.cfg.multishape_platform_has_walls:
                half_x = 0.5 * SHALLOW_TRAY_SIZE[0]
                half_y = 0.5 * SHALLOW_TRAY_SIZE[1]
                wall_z = 0.5 * self.cfg.multishape_support_height + 0.5 * SHALLOW_TRAY_WALL_HEIGHT
                offsets = torch.tensor(
                    (
                        (0.0, 0.0, 0.0),
                        (0.0, half_y - 0.5 * SHALLOW_TRAY_WALL_THICKNESS, wall_z),
                        (0.0, -half_y + 0.5 * SHALLOW_TRAY_WALL_THICKNESS, wall_z),
                        (-half_x + 0.5 * SHALLOW_TRAY_WALL_THICKNESS, 0.0, wall_z),
                        (half_x - 0.5 * SHALLOW_TRAY_WALL_THICKNESS, 0.0, wall_z),
                    ),
                    device=roots.device,
                    dtype=roots.dtype,
                )
                positions = (roots.unsqueeze(1) + offsets.unsqueeze(0)).reshape(-1, 3)
                component_scales = torch.tensor(
                    (
                        SHALLOW_TRAY_SIZE,
                        (SHALLOW_TRAY_SIZE[0], SHALLOW_TRAY_WALL_THICKNESS, SHALLOW_TRAY_WALL_HEIGHT),
                        (SHALLOW_TRAY_SIZE[0], SHALLOW_TRAY_WALL_THICKNESS, SHALLOW_TRAY_WALL_HEIGHT),
                        (
                            SHALLOW_TRAY_WALL_THICKNESS,
                            SHALLOW_TRAY_SIZE[1] - 2.0 * SHALLOW_TRAY_WALL_THICKNESS,
                            SHALLOW_TRAY_WALL_HEIGHT,
                        ),
                        (
                            SHALLOW_TRAY_WALL_THICKNESS,
                            SHALLOW_TRAY_SIZE[1] - 2.0 * SHALLOW_TRAY_WALL_THICKNESS,
                            SHALLOW_TRAY_WALL_HEIGHT,
                        ),
                    ),
                    device=roots.device,
                    dtype=roots.dtype,
                )
                scales = component_scales.unsqueeze(0).expand(roots.shape[0], -1, -1).reshape(-1, 3)
            else:
                positions = roots
                scales = torch.tensor(SHALLOW_TRAY_SIZE, device=roots.device, dtype=roots.dtype).expand(
                    roots.shape[0], -1
                )
            orientations = torch.zeros(positions.shape[0], 4, device=positions.device, dtype=positions.dtype)
            orientations[:, 0] = 1.0
        else:
            init_platform = self.scene["object_init_platform"]
            target_platform = self.scene["object_target_platform"]
            positions = torch.cat((init_platform.data.root_pos_w, target_platform.data.root_pos_w), dim=0)
            orientations = torch.cat((init_platform.data.root_quat_w, target_platform.data.root_quat_w), dim=0)
            platform_height = self.cfg.domain_randomization_support_height
            scales = torch.tensor(
                OBJECT_PLATFORM_SIZE,
                device=positions.device,
                dtype=positions.dtype,
            ).expand(positions.shape[0], -1)
        self.platform_visualizer.visualize(
            translations=positions,
            orientations=orientations,
            scales=scales,
        )

    def _update_motion_frame_visualization(self) -> None:
        if not getattr(self.cfg, "motion_keyframe_debug_vis", False):
            return
        required_fields = (
            "motion_current_pos_w",
            "motion_current_quat_w",
            "motion_target_pos_w",
            "motion_target_quat_w",
            "motion_guide_valid",
        )
        if not all(hasattr(self, field) for field in required_fields):
            return
        if self.motion_target_pos_w.shape[1] != 2:
            raise ValueError("motion frame visualization currently requires exactly two keyframes")
        if self.motion_frame_visualizer is None:
            self.motion_frame_visualizer = VisualizationMarkers(MOTION_FRAME_MARKER_CFG)

        valid = self.motion_guide_valid
        if not torch.any(valid):
            self.motion_frame_visualizer.set_visibility(False)
            return

        positions, orientations, marker_indices = flatten_motion_frame_markers(
            self.motion_current_pos_w[valid],
            self.motion_current_quat_w[valid],
            self.motion_target_pos_w[valid],
            self.motion_target_quat_w[valid],
        )
        self.motion_frame_visualizer.set_visibility(True)
        self.motion_frame_visualizer.visualize(
            translations=positions,
            orientations=orientations,
            marker_indices=marker_indices,
        )

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        self.prev_high_level_action = self.last_high_level_action.clone()
        raw_high_level_action = action.to(self.device)
        self._nonfinite_high_level_action_ratio = nonfinite_ratio(raw_high_level_action).detach()
        self.last_high_level_action = sanitize_tensor(
            raw_high_level_action,
            finite_clip=self.cfg.finite_action_clip,
        )
        self._high_level_action_saturation_ratio = (
            torch.abs(self.last_high_level_action) >= self.cfg.finite_action_clip
        ).float().mean().detach()
        self._high_level_action_abs_mean = torch.abs(self.last_high_level_action).mean().detach()
        self.previous_command_rate.copy_(self.command_rate)
        self.previous_command_acceleration.copy_(self.command_acceleration)
        self.command_state, command_rate_state = decode_high_level_action(
            self.last_high_level_action,
            self.command_state,
            self.action_limits,
            dt=self.step_dt,
        )
        self.command_rate.copy_(command_rate_state.as_tensor())
        self.command_acceleration.copy_((self.command_rate - self.previous_command_rate) / self.step_dt)
        self.command_jerk.copy_(
            (self.command_acceleration - self.previous_command_acceleration) / self.step_dt
        )
        self._apply_debug_gripper_override()
        self.high_level_command.set_command(
            self.command_state.base_velocity,
            self.command_state.posture_command,
            self.command_state.left_hand_center_pose_a,
            self.command_state.right_hand_center_pose_a,
            self.command_state.left_grip,
            self.command_state.right_grip,
        )

        self.recorder_manager.record_pre_step()
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        self._reset_contact_accumulators()
        self._nonfinite_low_level_action_ratio.zero_()
        for _ in range(self.cfg.high_level_decimation):
            low_obs = self.low_level_obs_builder.build(self.command_state, self._last_low_level_action)
            if self.low_level_policy is None:
                raw_low_action = torch.zeros(self.num_envs, len(self.body_joint_names), device=self.device)
            else:
                raw_low_action = self.low_level_policy(low_obs)
            self._nonfinite_low_level_action_ratio = torch.maximum(
                self._nonfinite_low_level_action_ratio,
                nonfinite_ratio(raw_low_action).detach(),
            )
            raw_low_action = sanitize_tensor(raw_low_action, finite_clip=self.cfg.low_level_action_clip)
            for _ in range(self.cfg.low_level_decimation):
                self._sim_step_counter += 1
                self._apply_low_level_action(raw_low_action)
                self._apply_gripper_command()
                self.scene.write_data_to_sim()
                self.sim.step(render=False)
                self.recorder_manager.record_post_physics_decimation_step()
                if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                    self.sim.render()
                self.scene.update(dt=self.physics_dt)
                self._accumulate_contact_components()
            self._last_low_level_action = raw_low_action.detach()

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self._compute_progress()
        self._sanitize_progress_buffers()
        self._update_domain_randomization_curriculum()
        self._update_object_mass_curriculum()
        self._check_success()
        step_success_count = self.task_succeeded.sum().float().detach()
        step_active_hand = self.active_hand.clone()
        step_d_active_hand = self.d_active_hand.clone()
        step_c_contact = self.c_contact.clone()
        step_c_couple = self.c_couple.clone()
        step_c_grasp = self.c_grasp.clone()
        step_c_physical_grasp = self.c_physical_grasp.clone()
        (
            step_reference_command_orientation_error,
            step_command_actual_orientation_error,
            step_reference_actual_orientation_error,
        ) = self._active_grasp_pose_diagnostics()
        step_position_gate = self.active_position_gate.clone()
        step_orientation_gate = self.active_orientation_gate.clone()
        step_d_goal = self.d_goal.clone()
        step_object_fallen = self.object_fallen.clone()
        step_W_app = self.W_app.clone()
        step_W_couple = self.W_couple.clone()
        step_W_manip = self.W_manip.clone()

        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated | self.task_succeeded
        self.reset_time_outs = self.termination_manager.time_outs
        self.reset_buf = self.reset_buf | self.task_succeeded
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.recorder_manager.record_pre_reset(reset_env_ids)
            self._reset_idx(reset_env_ids)
            self.scene.write_data_to_sim()
            self.sim.forward()
            self.scene.update(dt=0.0)
            self.recorder_manager.record_post_reset(reset_env_ids)

        self.command_manager.compute(dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        self.obs_buf = self.observation_manager.compute(update_history=True)
        self._sanitize_rollout_outputs()
        self._update_target_pose_visualization()
        self.extras["log"]["DRC/d_active_hand_mean"] = step_d_active_hand.mean()
        self.extras["log"]["DRC/c_contact_mean"] = step_c_contact.mean()
        self.extras["log"]["DRC/c_couple_mean"] = step_c_couple.mean()
        self.extras["log"]["DRC/c_grasp_mean"] = step_c_grasp.mean()
        self.extras["log"]["DRC/c_physical_grasp_mean"] = step_c_physical_grasp.mean()
        self.extras["log"]["DRC/d_goal_mean"] = step_d_goal.mean()
        self.extras["log"]["DRC/object_fall_mean"] = step_object_fallen.float().mean()
        self.extras["log"]["DRC/object_mass_curriculum_level"] = self.object_mass_curriculum_level
        self.extras["log"]["DRC/object_mass_mean"] = self.object_mass.mean()
        self.extras["log"]["Curriculum/grasp_rate_ema"] = self.domain_randomization_grasp_rate_ema
        self.extras["log"]["Curriculum/domain_randomization_level"] = self.domain_randomization_curriculum_level
        self.extras["log"]["Curriculum/object_size_scale_min"] = self.object_size_scale.min()
        self.extras["log"]["Curriculum/object_size_scale_max"] = self.object_size_scale.max()
        self.extras["log"]["DRC/W_app_mean"] = step_W_app.mean()
        self.extras["log"]["DRC/W_couple_mean"] = step_W_couple.mean()
        self.extras["log"]["DRC/W_manip_mean"] = step_W_manip.mean()
        self.extras["log"]["Contact/active_left_ratio"] = (step_active_hand == 0).float().mean()
        self.extras["log"]["GraspPose/reference_command_orientation_error_mean"] = (
            step_reference_command_orientation_error.mean()
        )
        self.extras["log"]["GraspPose/command_actual_orientation_error_mean"] = (
            step_command_actual_orientation_error.mean()
        )
        self.extras["log"]["GraspPose/reference_actual_orientation_error_mean"] = (
            step_reference_actual_orientation_error.mean()
        )
        self.extras["log"]["GraspPose/position_gate_mean"] = step_position_gate.mean()
        self.extras["log"]["GraspPose/orientation_gate_mean"] = step_orientation_gate.mean()
        self.extras["log"]["HL/action_saturation_ratio"] = self._high_level_action_saturation_ratio
        self.extras["log"]["Safety/nonfinite_high_level_action_ratio"] = self._nonfinite_high_level_action_ratio
        self.extras["log"]["Safety/nonfinite_low_level_action_ratio"] = self._nonfinite_low_level_action_ratio
        self.extras["log"]["Safety/nonfinite_reward_ratio"] = self._nonfinite_reward_ratio
        self.extras["log"]["Safety/nonfinite_observation_ratio"] = self._nonfinite_observation_ratio
        self._log_high_level_diagnostics()
        self._log_shape_diagnostics(step_c_physical_grasp)
        self.extras["log"]["Task/success_count"] = step_success_count
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
