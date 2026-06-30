from __future__ import annotations

import torch
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg, VecEnvStepReturn
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG

from hbc_lab.assets.objects import OBJECT_PLATFORM_HEIGHT
from hbc_lab.assets.robots.unitree import G1_29DOF_BODY_JOINT_NAMES

from ..mdp.contact_progress import (
    compute_active_hand_grasp_progress,
)
from ..mdp.drc_math import compute_drc_weights, update_ema
from ..mdp.gripper import Dex1GripperController
from ..mdp.high_level_actions import HighLevelActionLimits, HighLevelCommandState, decode_high_level_action
from ..mdp.low_level_observations import G1SphericalPostureLowLevelObsBuilder
from ..mdp.low_level_policy import LowLevelPolicyWrapper
from ..mdp.scenes import (
    DEX1_LINK_CONTACT_SENSOR_NAMES,
    HAND_CENTER_FRAME_NAME,
    LEFT_GRIPPER_CONTACT_SENSOR_NAMES,
    RIGHT_GRIPPER_CONTACT_SENSOR_NAMES,
)


TARGET_OBJECT_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/G1Dex1HierDrc/object_goal",
    markers={
        "target": sim_utils.SphereCfg(
            radius=0.055,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 1.0)),
        ),
    },
)
TARGET_OBJECT_FRAME_MARKER_CFG = FRAME_MARKER_CFG.replace(prim_path="/Visuals/G1Dex1HierDrc/object_goal_frame")
TARGET_OBJECT_FRAME_MARKER_CFG.markers["frame"].scale = (0.08, 0.08, 0.08)

OBJECT_INITIAL_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/G1Dex1HierDrc/object_initial",
    markers={
        "initial": sim_utils.SphereCfg(
            radius=0.045,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.75, 1.0)),
        ),
    },
)
OBJECT_INITIAL_FRAME_MARKER_CFG = FRAME_MARKER_CFG.replace(prim_path="/Visuals/G1Dex1HierDrc/object_initial_frame")
OBJECT_INITIAL_FRAME_MARKER_CFG.markers["frame"].scale = (0.06, 0.06, 0.06)


class G1Dex1HierDrcEnv(ManagerBasedRLEnv):
    cfg: ManagerBasedRLEnvCfg

    body_joint_names = G1_29DOF_BODY_JOINT_NAMES

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        self.d_active_hand = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.d_goal = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.c_contact = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.c_couple = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.c_grasp = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
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
        self.W_app = torch.ones(cfg.scene.num_envs, device=cfg.sim.device)
        self.W_couple = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.W_manip = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.object_initial_pos_w = torch.zeros(cfg.scene.num_envs, 3, device=cfg.sim.device)
        self.object_target_pos_w = torch.zeros(cfg.scene.num_envs, 3, device=cfg.sim.device)
        self.object_fallen = torch.zeros(cfg.scene.num_envs, dtype=torch.bool, device=cfg.sim.device)
        self.success_proximity_count = torch.zeros(cfg.scene.num_envs, dtype=torch.long, device=cfg.sim.device)
        self.task_succeeded = torch.zeros(cfg.scene.num_envs, dtype=torch.bool, device=cfg.sim.device)
        self.last_high_level_action = torch.zeros(cfg.scene.num_envs, cfg.action_dim, device=cfg.sim.device)
        self.prev_high_level_action = torch.zeros(cfg.scene.num_envs, cfg.action_dim, device=cfg.sim.device)
        self._last_low_level_action = torch.zeros(cfg.scene.num_envs, len(self.body_joint_names), device=cfg.sim.device)
        self.active_hand = torch.zeros(cfg.scene.num_envs, dtype=torch.long, device=cfg.sim.device)
        super().__init__(cfg, render_mode, **kwargs)

        robot = self.scene["robot"]
        self.left_wrist_body_id = robot.find_bodies("left_wrist_yaw_link")[0][0]
        self.right_wrist_body_id = robot.find_bodies("right_wrist_yaw_link")[0][0]
        self.high_level_command = self.command_manager.get_term("high_level")
        self.active_hand = self.high_level_command.active_hand.clone()
        self.command_state = HighLevelCommandState(
            base_velocity=self.high_level_command.base_velocity.clone(),
            posture_command=self.high_level_command.posture_command.clone(),
            left_wrist_pose_b=self.high_level_command.left_wrist_pose_b.clone(),
            right_wrist_pose_b=self.high_level_command.right_wrist_pose_b.clone(),
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
        self.target_pose_frame_visualizer = None
        self.object_initial_pose_visualizer = None
        self.object_initial_pose_frame_visualizer = None

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
        self._step_link_force = {
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
            contact, force = self._contact_confidence_from_force(force_w)
            self._step_link_contact[key] = torch.maximum(self._step_link_contact[key], contact)
            self._step_link_force[key] = torch.maximum(self._step_link_force[key], force)

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
        if self.cfg.low_level_action_clip is not None:
            low_action = torch.nan_to_num(
                low_action,
                nan=0.0,
                posinf=self.cfg.low_level_action_clip,
                neginf=-self.cfg.low_level_action_clip,
            )
            low_action = torch.clamp(low_action, -self.cfg.low_level_action_clip, self.cfg.low_level_action_clip)
        else:
            low_action = torch.nan_to_num(low_action, nan=0.0)
        joint_target = default_joint_pos + self.cfg.low_level_action_scale * low_action
        robot.set_joint_position_target(joint_target, joint_ids=joint_ids)
        return low_action

    def _apply_gripper_command(self):
        self.gripper_controller.apply(self.command_state.left_grip, self.command_state.right_grip)

    def _apply_debug_gripper_override(self):
        if not getattr(self.cfg, "debug_fixed_gripper", False):
            return
        self.command_state.left_grip[:] = self.cfg.debug_fixed_left_grip
        self.command_state.right_grip[:] = self.cfg.debug_fixed_right_grip

    def _active_wrist_pose_and_limits(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        left_active = (self.active_hand == 0).unsqueeze(-1)
        pose = torch.where(left_active, self.command_state.left_wrist_pose_b, self.command_state.right_wrist_pose_b)
        left_lower = torch.tensor(self.action_limits.left_workspace_min, device=self.device, dtype=pose.dtype)
        left_upper = torch.tensor(self.action_limits.left_workspace_max, device=self.device, dtype=pose.dtype)
        right_lower = torch.tensor(self.action_limits.right_workspace_min, device=self.device, dtype=pose.dtype)
        right_upper = torch.tensor(self.action_limits.right_workspace_max, device=self.device, dtype=pose.dtype)
        lower = torch.where(left_active, left_lower.unsqueeze(0), right_lower.unsqueeze(0))
        upper = torch.where(left_active, left_upper.unsqueeze(0), right_upper.unsqueeze(0))
        return pose, lower, upper

    def _object_frame_pos_w(self) -> torch.Tensor:
        return self.scene["object_frame"].data.target_pos_w[:, 0, :]

    def _object_fallen(self) -> torch.Tensor:
        object_root_z = self.scene["object"].data.root_pos_w[:, 2]
        fall_threshold = self.scene.env_origins[:, 2] + 0.5 * OBJECT_PLATFORM_HEIGHT
        return object_root_z < fall_threshold

    def _active_target_tracking_diagnostics(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        robot = self.scene["robot"]
        left_active = (self.active_hand == 0).unsqueeze(-1)
        left_target_w = self.low_level_obs_builder._target_pos_w(
            self.command_state.left_wrist_pose_b,
            "left",
            self.command_state.posture_command,
        )
        right_target_w = self.low_level_obs_builder._target_pos_w(
            self.command_state.right_wrist_pose_b,
            "right",
            self.command_state.posture_command,
        )
        active_target_w = torch.where(left_active, left_target_w, right_target_w)

        left_wrist_pos_w = robot.data.body_pos_w[:, self.left_wrist_body_id]
        right_wrist_pos_w = robot.data.body_pos_w[:, self.right_wrist_body_id]
        active_wrist_pos_w = torch.where(left_active, left_wrist_pos_w, right_wrist_pos_w)

        hand_center_pos_w = self.scene[HAND_CENTER_FRAME_NAME].data.target_pos_w
        active_hand_center_w = torch.where(left_active, hand_center_pos_w[:, 0, :], hand_center_pos_w[:, 1, :])
        target_object_dist = torch.norm(active_target_w - self._object_frame_pos_w(), dim=-1)
        wrist_tracking_error = torch.norm(active_wrist_pos_w - active_target_w, dim=-1)
        hand_center_to_target_dist = torch.norm(active_hand_center_w - active_target_w, dim=-1)
        return target_object_dist, wrist_tracking_error, hand_center_to_target_dist

    def _log_high_level_diagnostics(self):
        active_wrist_pose_b, lower, upper = self._active_wrist_pose_and_limits()
        active_wrist_pos_b = active_wrist_pose_b[:, :3]
        left_active = self.active_hand == 0
        left_grip = self.command_state.left_grip.squeeze(-1)
        right_grip = self.command_state.right_grip.squeeze(-1)
        active_grip = torch.where(left_active, left_grip, right_grip)
        inactive_grip = torch.where(left_active, right_grip, left_grip)
        target_object_dist, wrist_tracking_error, hand_center_to_target_dist = self._active_target_tracking_diagnostics()
        eps = 1.0e-4
        self.extras["log"]["HL/root_height_cmd_mean"] = self.command_state.posture_command[:, 0].mean()
        self.extras["log"]["HL/torso_pitch_cmd_mean"] = self.command_state.posture_command[:, 1].mean()
        self.extras["log"]["HL/active_wrist_cmd_x_mean"] = active_wrist_pos_b[:, 0].mean()
        self.extras["log"]["HL/active_wrist_cmd_y_mean"] = active_wrist_pos_b[:, 1].mean()
        self.extras["log"]["HL/active_wrist_cmd_z_mean"] = active_wrist_pos_b[:, 2].mean()
        self.extras["log"]["HL/active_grip_mean"] = active_grip.mean()
        self.extras["log"]["HL/inactive_grip_mean"] = inactive_grip.mean()
        self.extras["log"]["HL/active_grip_closed_ratio"] = (active_grip > 0.5).float().mean()
        self.extras["log"]["HL/active_wrist_target_object_dist"] = target_object_dist.mean()
        self.extras["log"]["HL/active_wrist_tracking_error"] = wrist_tracking_error.mean()
        self.extras["log"]["HL/active_hand_center_to_wrist_target_dist"] = hand_center_to_target_dist.mean()
        self.extras["log"]["HL/active_wrist_cmd_x_min_ratio"] = (active_wrist_pos_b[:, 0] <= lower[:, 0] + eps).float().mean()
        self.extras["log"]["HL/active_wrist_cmd_x_max_ratio"] = (active_wrist_pos_b[:, 0] >= upper[:, 0] - eps).float().mean()
        self.extras["log"]["HL/active_wrist_cmd_y_min_ratio"] = (active_wrist_pos_b[:, 1] <= lower[:, 1] + eps).float().mean()
        self.extras["log"]["HL/active_wrist_cmd_y_max_ratio"] = (active_wrist_pos_b[:, 1] >= upper[:, 1] - eps).float().mean()
        self.extras["log"]["HL/active_wrist_cmd_z_min_ratio"] = (active_wrist_pos_b[:, 2] <= lower[:, 2] + eps).float().mean()
        self.extras["log"]["HL/active_wrist_cmd_z_max_ratio"] = (active_wrist_pos_b[:, 2] >= upper[:, 2] - eps).float().mean()

    def _log_link_contact_diagnostics(self) -> None:
        for side, link_name, _ in DEX1_LINK_CONTACT_SENSOR_NAMES:
            key = f"{side}_{link_name}"
            self.extras["log"][f"ContactLink/{key}_mean"] = self._step_link_contact[key].mean()
            self.extras["log"][f"ContactLink/{key}_force"] = self._step_link_force[key].mean()

        left_active = self.active_hand == 0
        for link_name in ("Link1_2", "Link1_3", "Link2_2", "Link2_3"):
            left_key = f"left_{link_name}"
            right_key = f"right_{link_name}"
            active_contact = torch.where(left_active, self._step_link_contact[left_key], self._step_link_contact[right_key])
            active_force = torch.where(left_active, self._step_link_force[left_key], self._step_link_force[right_key])
            self.extras["log"][f"ContactLink/active_{link_name}_mean"] = active_contact.mean()
            self.extras["log"][f"ContactLink/active_{link_name}_force"] = active_force.mean()

    def _compute_progress(self):
        object_pos_w = self._object_frame_pos_w()
        hand_center_pos_w = self.scene[HAND_CENTER_FRAME_NAME].data.target_pos_w
        left_pos_w = hand_center_pos_w[:, 0, :]
        right_pos_w = hand_center_pos_w[:, 1, :]
        left_distance = torch.norm(left_pos_w - object_pos_w, dim=-1)
        right_distance = torch.norm(right_pos_w - object_pos_w, dim=-1)
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
        self.d_goal = torch.norm(object_pos_w - self.object_target_pos_w, dim=-1)
        self.left_hand_contact = self._step_left_contact
        self.right_hand_contact = self._step_right_contact
        self.c_contact = update_ema(self.c_contact, progress.contact, alpha=0.2)
        self.c_opposition = update_ema(self.c_opposition, progress.pinch, alpha=0.2)
        self.c_pinch = update_ema(self.c_pinch, progress.pinch, alpha=0.2)
        self.c_grasp = update_ema(self.c_grasp, progress.grasp, alpha=0.2)
        # Baseline A: enter manipulation only after two-finger pinch.
        # self.c_couple = update_ema(self.c_couple, progress.pinch, alpha=0.2)
        self.c_couple = update_ema(self.c_couple, progress.grasp, alpha=0.2)
        self.object_fallen = self._object_fallen()
        weights = compute_drc_weights(self.d_active_hand, self.c_couple, alpha=5.0)
        self.W_app = weights[:, 0]
        self.W_couple = weights[:, 1]
        self.W_manip = weights[:, 2]

    def _check_success(self):
        success_condition = (self.c_couple > self.cfg.success_couple_threshold) & (self.d_goal < self.cfg.success_distance)
        self.success_proximity_count[success_condition] += 1
        self.success_proximity_count[~success_condition] = 0
        self.task_succeeded = self.success_proximity_count >= self.cfg.success_steps

    def _reset_command_state(self, env_ids: torch.Tensor):
        self.command_state.base_velocity[env_ids] = self.high_level_command.base_velocity[env_ids]
        self.command_state.posture_command[env_ids] = self.high_level_command.posture_command[env_ids]
        self.command_state.left_wrist_pose_b[env_ids] = self.high_level_command.left_wrist_pose_b[env_ids]
        self.command_state.right_wrist_pose_b[env_ids] = self.high_level_command.right_wrist_pose_b[env_ids]
        self.command_state.left_grip[env_ids] = self.high_level_command.left_grip[env_ids]
        self.command_state.right_grip[env_ids] = self.high_level_command.right_grip[env_ids]

    def _reset_hier_buffers(self, env_ids: torch.Tensor):
        self.d_active_hand[env_ids] = 0.0
        self.d_goal[env_ids] = 0.0
        self.c_contact[env_ids] = 0.0
        self.c_couple[env_ids] = 0.0
        self.c_grasp[env_ids] = 0.0
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
        self._last_low_level_action[env_ids] = 0.0
        self.low_level_obs_builder.reset(env_ids)
        self.active_hand[env_ids] = self.high_level_command.active_hand[env_ids]
        self._reset_command_state(env_ids)

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._reset_hier_buffers(env_ids)

    def _update_target_pose_visualization(self) -> None:
        if not getattr(self.cfg, "target_pose_debug_vis", False):
            return
        if self.target_pose_visualizer is None:
            self.target_pose_visualizer = VisualizationMarkers(TARGET_OBJECT_MARKER_CFG)
            self.target_pose_frame_visualizer = VisualizationMarkers(TARGET_OBJECT_FRAME_MARKER_CFG)
            self.object_initial_pose_visualizer = VisualizationMarkers(OBJECT_INITIAL_MARKER_CFG)
            self.object_initial_pose_frame_visualizer = VisualizationMarkers(OBJECT_INITIAL_FRAME_MARKER_CFG)
            self.target_pose_visualizer.set_visibility(True)
            self.target_pose_frame_visualizer.set_visibility(True)
            self.object_initial_pose_visualizer.set_visibility(True)
            self.object_initial_pose_frame_visualizer.set_visibility(True)
        target_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        target_quat_w[:, 0] = 1.0
        self.object_initial_pose_visualizer.visualize(self.object_initial_pos_w)
        self.object_initial_pose_frame_visualizer.visualize(self.object_initial_pos_w, target_quat_w)
        self.target_pose_visualizer.visualize(self.object_target_pos_w)
        self.target_pose_frame_visualizer.visualize(self.object_target_pos_w, target_quat_w)

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        self.prev_high_level_action = self.last_high_level_action.clone()
        self.last_high_level_action = torch.clamp(action.to(self.device), -1.0, 1.0)
        self.command_state = decode_high_level_action(self.last_high_level_action, self.command_state, self.action_limits)
        self._apply_debug_gripper_override()
        self.high_level_command.set_command(
            self.command_state.base_velocity,
            self.command_state.posture_command,
            self.command_state.left_wrist_pose_b,
            self.command_state.right_wrist_pose_b,
            self.command_state.left_grip,
            self.command_state.right_grip,
        )

        self.recorder_manager.record_pre_step()
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        self._reset_contact_accumulators()
        for _ in range(self.cfg.high_level_decimation):
            low_obs = self.low_level_obs_builder.build(self.command_state, self._last_low_level_action)
            if self.low_level_policy is None:
                raw_low_action = torch.zeros(self.num_envs, len(self.body_joint_names), device=self.device)
            else:
                raw_low_action = self.low_level_policy(low_obs)
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
        self._check_success()

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
        self.obs_buf = self.observation_manager.compute()
        self._update_target_pose_visualization()
        self.extras["log"]["DRC/d_active_hand_mean"] = self.d_active_hand.mean()
        self.extras["log"]["DRC/c_contact_mean"] = self.c_contact.mean()
        self.extras["log"]["DRC/c_couple_mean"] = self.c_couple.mean()
        self.extras["log"]["DRC/c_grasp_mean"] = self.c_grasp.mean()
        self.extras["log"]["DRC/c_opposition_mean"] = self.c_opposition.mean()
        self.extras["log"]["DRC/c_pinch_mean"] = self.c_pinch.mean()
        self.extras["log"]["DRC/d_goal_mean"] = self.d_goal.mean()
        self.extras["log"]["DRC/object_fall_mean"] = self.object_fallen.float().mean()
        self.extras["log"]["DRC/W_app_mean"] = self.W_app.mean()
        self.extras["log"]["DRC/W_couple_mean"] = self.W_couple.mean()
        self.extras["log"]["DRC/W_manip_mean"] = self.W_manip.mean()
        self.extras["log"]["Contact/left_hand_mean"] = self.left_hand_contact.mean()
        self.extras["log"]["Contact/right_hand_mean"] = self.right_hand_contact.mean()
        self.extras["log"]["Contact/active_left_ratio"] = (self.active_hand == 0).float().mean()
        self.extras["log"]["Contact/active_left_finger_mean"] = self.active_left_finger_contact.mean()
        self.extras["log"]["Contact/active_right_finger_mean"] = self.active_right_finger_contact.mean()
        self.extras["log"]["Contact/active_left_finger_force"] = self.active_left_finger_force.mean()
        self.extras["log"]["Contact/active_right_finger_force"] = self.active_right_finger_force.mean()
        self.extras["log"]["Contact/active_opposition_mean"] = self.active_opposition.mean()
        self.extras["log"]["Contact/active_pinch_score_mean"] = self.active_pinch_score.mean()
        self.extras["log"]["Contact/active_force_cos_sim_mean"] = self.active_contact_cos_sim.mean()
        self._log_link_contact_diagnostics()
        self.extras["log"]["HL/left_grip_mean"] = self.command_state.left_grip.mean()
        self.extras["log"]["HL/right_grip_mean"] = self.command_state.right_grip.mean()
        self._log_high_level_diagnostics()
        self.extras["log"]["Task/success_count"] = self.task_succeeded.sum().float()
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
