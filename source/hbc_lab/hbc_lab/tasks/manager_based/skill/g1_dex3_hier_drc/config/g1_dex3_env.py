from __future__ import annotations

import torch
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg, VecEnvStepReturn
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG

from hbc_lab.assets.robots.unitree import G1_29DOF_BODY_JOINT_NAMES

from ..mdp.contact_progress import (
    compute_active_hand_grasp_progress,
    compute_hand_contact_confidence,
)
from ..mdp.drc_math import compute_drc_weights, update_ema
from ..mdp.gripper import Dex3GripperController
from ..mdp.high_level_actions import HighLevelActionLimits, HighLevelCommandState, decode_high_level_action
from ..mdp.low_level_observations import G1SphericalPostureLowLevelObsBuilder
from ..mdp.low_level_policy import LowLevelPolicyWrapper
from ..mdp.scenes import HAND_CENTER_FRAME_NAME, LEFT_HAND_CONTACT_SENSOR_NAMES, RIGHT_HAND_CONTACT_SENSOR_NAMES


TARGET_OBJECT_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/G1Dex3HierDrc/object_goal",
    markers={
        "target": sim_utils.SphereCfg(
            radius=0.055,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 1.0)),
        ),
    },
)
TARGET_OBJECT_FRAME_MARKER_CFG = FRAME_MARKER_CFG.replace(prim_path="/Visuals/G1Dex3HierDrc/object_goal_frame")
TARGET_OBJECT_FRAME_MARKER_CFG.markers["frame"].scale = (0.08, 0.08, 0.08)

OBJECT_INITIAL_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/G1Dex3HierDrc/object_initial",
    markers={
        "initial": sim_utils.SphereCfg(
            radius=0.045,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.75, 1.0)),
        ),
    },
)
OBJECT_INITIAL_FRAME_MARKER_CFG = FRAME_MARKER_CFG.replace(prim_path="/Visuals/G1Dex3HierDrc/object_initial_frame")
OBJECT_INITIAL_FRAME_MARKER_CFG.markers["frame"].scale = (0.06, 0.06, 0.06)


class G1Dex3HierDrcEnv(ManagerBasedRLEnv):
    cfg: ManagerBasedRLEnvCfg

    body_joint_names = G1_29DOF_BODY_JOINT_NAMES

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        self.d_active_hand = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.d_goal = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.c_contact = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.c_couple = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.c_grasp = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.active_grip = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.left_hand_contact = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.right_hand_contact = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.left_hand_force = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.right_hand_force = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.W_app = torch.ones(cfg.scene.num_envs, device=cfg.sim.device)
        self.W_couple = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.W_manip = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.object_initial_pos_w = torch.zeros(cfg.scene.num_envs, 3, device=cfg.sim.device)
        self.object_target_pos_w = torch.zeros(cfg.scene.num_envs, 3, device=cfg.sim.device)
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
                "A frozen low-level policy is required for G1Dex3HierDrcEnv. "
                "Set env.low_level_policy_path=/path/to/exported_low_level_policy.pt, "
                "or set env.allow_missing_low_level_policy=True only for zero-action debugging."
            )
        self.low_level_obs_builder = G1SphericalPostureLowLevelObsBuilder(
            self,
            history_length=cfg.low_level_obs_history_length,
            finite_obs_clip=cfg.finite_obs_clip,
        )
        self.gripper_controller = Dex3GripperController(robot, self.device)
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
        self._step_left_force = torch.zeros(self.num_envs, device=self.device)
        self._step_right_force = torch.zeros(self.num_envs, device=self.device)

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

    def _sum_sensor_group_force(self, sensor_names: tuple[str, ...]) -> torch.Tensor:
        force_mag = torch.zeros(self.num_envs, device=self.device)
        for sensor_name in sensor_names:
            force_mag += torch.norm(self._sum_sensor_force(sensor_name), dim=-1)
        force_w = torch.zeros(self.num_envs, 3, device=self.device)
        force_w[:, 0] = force_mag
        return force_w

    def _accumulate_contact_components(self) -> None:
        left_force_w = self._sum_sensor_group_force(LEFT_HAND_CONTACT_SENSOR_NAMES)
        right_force_w = self._sum_sensor_group_force(RIGHT_HAND_CONTACT_SENSOR_NAMES)
        left_components = compute_hand_contact_confidence(left_force_w, force_threshold=self.cfg.hand_contact_force_threshold)
        right_components = compute_hand_contact_confidence(
            right_force_w,
            force_threshold=self.cfg.hand_contact_force_threshold,
        )
        self._step_left_contact = torch.maximum(self._step_left_contact, left_components.contact)
        self._step_right_contact = torch.maximum(self._step_right_contact, right_components.contact)
        self._step_left_force = torch.maximum(self._step_left_force, left_components.force)
        self._step_right_force = torch.maximum(self._step_right_force, right_components.force)

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

    def _compute_progress(self):
        obj = self.scene["object"]
        object_pos_w = obj.data.root_pos_w
        hand_center_pos_w = self.scene[HAND_CENTER_FRAME_NAME].data.target_pos_w
        left_pos_w = hand_center_pos_w[:, 0, :]
        right_pos_w = hand_center_pos_w[:, 1, :]
        left_distance = torch.norm(left_pos_w - object_pos_w, dim=-1)
        right_distance = torch.norm(right_pos_w - object_pos_w, dim=-1)
        left_components = compute_hand_contact_confidence(
            self._sum_sensor_group_force(LEFT_HAND_CONTACT_SENSOR_NAMES),
            force_threshold=self.cfg.hand_contact_force_threshold,
        )
        right_components = compute_hand_contact_confidence(
            self._sum_sensor_group_force(RIGHT_HAND_CONTACT_SENSOR_NAMES),
            force_threshold=self.cfg.hand_contact_force_threshold,
        )
        left_components.contact = torch.maximum(left_components.contact, self._step_left_contact)
        right_components.contact = torch.maximum(right_components.contact, self._step_right_contact)
        progress = compute_active_hand_grasp_progress(
            left_components=left_components,
            right_components=right_components,
            active_hand=self.active_hand,
            left_grip=self.command_state.left_grip,
            right_grip=self.command_state.right_grip,
            left_distance=left_distance,
            right_distance=right_distance,
            close_distance=self.cfg.close_distance,
            close_gate_width=self.cfg.close_gate_width,
        )
        self.d_active_hand = progress.distance
        self.active_grip = progress.grip
        self.d_goal = torch.norm(object_pos_w - self.object_target_pos_w, dim=-1)
        self.left_hand_contact = self._step_left_contact
        self.right_hand_contact = self._step_right_contact
        self.left_hand_force = self._step_left_force
        self.right_hand_force = self._step_right_force
        self.c_contact = update_ema(self.c_contact, progress.contact, alpha=0.2)
        self.c_grasp = update_ema(self.c_grasp, progress.grasp, alpha=0.2)
        self.c_couple = update_ema(self.c_couple, progress.grasp, alpha=0.2)
        weights = compute_drc_weights(self.d_active_hand, self.c_couple, alpha=5.0)
        self.W_app = weights[:, 0]
        self.W_couple = weights[:, 1]
        self.W_manip = weights[:, 2]

    def _check_success(self):
        success_condition = (self.c_couple > self.cfg.success_couple_threshold) & (self.d_goal < self.cfg.success_distance)
        self.success_proximity_count[success_condition] += 1
        self.success_proximity_count[~success_condition] = 0
        self.task_succeeded = self.success_proximity_count >= self.cfg.success_steps

    def _reset_hier_buffers(self, env_ids: torch.Tensor):
        self.d_active_hand[env_ids] = 0.0
        self.d_goal[env_ids] = 0.0
        self.c_contact[env_ids] = 0.0
        self.c_couple[env_ids] = 0.0
        self.c_grasp[env_ids] = 0.0
        self.active_grip[env_ids] = 0.0
        self.left_hand_contact[env_ids] = 0.0
        self.right_hand_contact[env_ids] = 0.0
        self.left_hand_force[env_ids] = 0.0
        self.right_hand_force[env_ids] = 0.0
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
        self.extras["log"]["DRC/d_goal_mean"] = self.d_goal.mean()
        self.extras["log"]["DRC/W_app_mean"] = self.W_app.mean()
        self.extras["log"]["DRC/W_couple_mean"] = self.W_couple.mean()
        self.extras["log"]["DRC/W_manip_mean"] = self.W_manip.mean()
        self.extras["log"]["Contact/left_hand_mean"] = self.left_hand_contact.mean()
        self.extras["log"]["Contact/right_hand_mean"] = self.right_hand_contact.mean()
        self.extras["log"]["Contact/active_left_ratio"] = (self.active_hand == 0).float().mean()
        self.extras["log"]["HL/left_grip_mean"] = self.command_state.left_grip.mean()
        self.extras["log"]["HL/right_grip_mean"] = self.command_state.right_grip.mean()
        self.extras["log"]["Task/success_count"] = self.task_succeeded.sum().float()
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
