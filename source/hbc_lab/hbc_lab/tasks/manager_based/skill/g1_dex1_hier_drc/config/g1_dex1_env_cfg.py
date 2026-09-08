from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from hbc_lab.assets.objects import OBJECT_PLATFORM_HEIGHT
from hbc_lab.tasks.locomotion import mdp

from ..mdp.actions import G1Dex1HierDrcActionsCfg
from ..mdp.commands import G1Dex1HierDrcCommandsCfg
from ..mdp.curriculums import G1Dex1HierDrcCurriculumCfg
from ..mdp.events import G1Dex1HierDrcEventCfg
from ..mdp.observations import G1Dex1HierDrcObservationsCfg
from ..mdp.rewards import G1Dex1HierDrcRewardsCfg
from ..mdp.scenes import G1Dex1HierDrcSceneCfg, LEFT_GRIPPER_CONTACT_SENSOR_NAMES, RIGHT_GRIPPER_CONTACT_SENSOR_NAMES
from ..mdp.grasp_references import GRASP_REFERENCE_DATA_PATH


def joint_vel_explosion(
    env: ManagerBasedRLEnv,
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    is_exploded = (torch.abs(joint_vel) > threshold) | torch.isnan(joint_vel) | torch.isinf(joint_vel)
    return torch.any(is_exploded, dim=1)


def nonfinite_sim_state(env: ManagerBasedRLEnv) -> torch.Tensor:
    invalid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for asset_name in ("robot", "object"):
        asset = env.scene[asset_name]
        attributes = (
            ("object_pos_w", "object_quat_w", "object_lin_vel_w", "object_ang_vel_w")
            if hasattr(asset.data, "object_pos_w")
            else ("root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w", "joint_pos", "joint_vel")
        )
        for attribute in attributes:
            tensor = getattr(asset.data, attribute, None)
            if tensor is not None:
                invalid |= ~torch.isfinite(tensor.reshape(env.num_envs, -1)).all(dim=1)
    return invalid


@configclass
class G1Dex1HierDrcTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.25})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.2})
    joint_vel_explosion = DoneTerm(func=joint_vel_explosion, params={"threshold": 1000.0})
    nonfinite_sim_state = DoneTerm(func=nonfinite_sim_state)


@configclass
class G1Dex1HierDrcEnvCfg(ManagerBasedRLEnvCfg):
    scene: G1Dex1HierDrcSceneCfg = G1Dex1HierDrcSceneCfg(num_envs=4096, env_spacing=6.0)
    actions: G1Dex1HierDrcActionsCfg = G1Dex1HierDrcActionsCfg()
    commands: G1Dex1HierDrcCommandsCfg = G1Dex1HierDrcCommandsCfg()
    observations: G1Dex1HierDrcObservationsCfg = G1Dex1HierDrcObservationsCfg()
    rewards: G1Dex1HierDrcRewardsCfg = G1Dex1HierDrcRewardsCfg()
    terminations: G1Dex1HierDrcTerminationsCfg = G1Dex1HierDrcTerminationsCfg()
    events: G1Dex1HierDrcEventCfg = G1Dex1HierDrcEventCfg()
    curriculum: G1Dex1HierDrcCurriculumCfg = G1Dex1HierDrcCurriculumCfg()

    low_level_policy_path: str = ""
    allow_missing_low_level_policy: bool = False
    enable_debug_visualization: bool = False
    target_pose_debug_vis: bool = False
    motion_keyframe_debug_vis: bool = False
    platform_pose_debug_vis: bool = False
    debug_fixed_gripper: bool = False
    debug_fixed_left_grip: float = 1.0
    debug_fixed_right_grip: float = 1.0
    high_level_decimation: int = 5
    low_level_decimation: int = 4
    action_dim: int = 19
    low_level_obs_history_length: int = 5
    low_level_action_scale: float = 0.25
    low_level_action_clip: float | None = None
    finite_action_clip: float = 1.0
    finite_obs_clip: float = 100.0
    finite_reward_clip: float = 1000.0
    contact_force_clip: float = 1000.0
    progress_distance_clip: float = 20.0
    hand_contact_force_threshold: float = 2.0
    close_distance: float = 0.20
    close_gate_width: float = 0.10
    close_orientation_zero_error: float = 1.0471975512
    close_orientation_gate_width: float = 0.6981317008
    success_distance: float = 0.08
    success_steps: int = 25
    success_couple_threshold: float = 0.45
    object_goal_radius_range: tuple[float, float] = (1.5, 2.5)
    object_mass_curriculum_enabled: bool = True
    object_mass_w_manip_ema_alpha: float = 0.01
    object_mass_start_w: float = 0.0
    object_mass_ref_w: float = 0.10
    object_mass_start_mass: float = 10.0
    object_mass_ref_mass: float = 5.0
    object_mass_anchor_w: float = 0.20
    object_mass_anchor_mass: float = 1.0
    object_mass_final_mass: float = 0.5
    object_mass_ref_log_std: float = 0.15
    object_mass_final_log_std: float = 0.45
    object_mass_min: float = 0.15
    object_mass_max: float = 25.0
    object_mass_use_physical_grasp: bool = False
    domain_randomization_curriculum_enabled: bool = False
    domain_randomization_start_level: float = 0.0
    domain_randomization_grasp_ema_alpha: float = 0.01
    domain_randomization_grasp_threshold: float = 0.35
    domain_randomization_advance_threshold: float = 0.55
    domain_randomization_update_interval: int = 500
    domain_randomization_level_step: float = 0.10
    domain_randomization_start_init_x: tuple[float, float] = (1.5, 2.0)
    domain_randomization_final_init_x: tuple[float, float] = (1.0, 2.7)
    domain_randomization_start_init_y: tuple[float, float] = (-0.35, 0.35)
    domain_randomization_final_init_y: tuple[float, float] = (-1.0, 1.0)
    domain_randomization_start_init_z: tuple[float, float] = (0.5, 0.5)
    domain_randomization_final_init_z: tuple[float, float] = (0.0, 0.7)
    domain_randomization_support_height: float = 0.7
    object_shape_names: tuple[str, ...] = ()
    object_size_scale_factors: tuple[float, ...] = (1.0,)
    object_shape_bps_data_path: str = ""
    object_shape_bps_enabled: bool = False
    grasp_reference_enabled: bool = False
    grasp_reference_controls_target: bool = False
    grasp_reference_data_path: str = str(GRASP_REFERENCE_DATA_PATH)
    grasp_candidate_asset_id: int = -1
    grasp_reference_candidate_indices: tuple[int, ...] = ()
    grasp_candidate_yaw_count: int = 0
    grasp_reference_select_highest_weight: bool = False
    grasp_reference_pose_debug_vis: bool = False
    grasp_pose_guidance_enabled: bool = False
    grasp_pose_position_scale: float = 0.15
    grasp_pose_position_reward_weight: float = 2.0
    grasp_pose_orientation_reward_weight: float = 2.0
    grasp_pose_release_threshold: float = 0.50
    grasp_pose_release_width: float = 0.08
    multishape_support_height: float = OBJECT_PLATFORM_HEIGHT
    multishape_platform_has_walls: bool = True
    domain_randomization_start_goal_radius: tuple[float, float] = (1.5, 2.5)
    domain_randomization_final_goal_radius: tuple[float, float] = (1.5, 4.0)
    motion_release_radius: float = 0.60
    motion_release_width: float = 0.06
    motion_near_posture_floor: float = 0.20
    motion_workspace_radius_range: tuple[float, float] = (0.20, 0.58)
    motion_workspace_scale: float = 0.10
    motion_position_deadzone: float = 0.05
    motion_position_scale: float = 0.10
    motion_orientation_deadzone: float = 0.20
    motion_orientation_scale: float = 0.50
    motion_root_height_deadzone: float = 0.03
    motion_root_height_scale: float = 0.10
    motion_torso_pitch_deadzone: float = 0.08
    motion_torso_pitch_scale: float = 0.25
    motion_root_tilt_deadzone: float = 0.10
    motion_root_tilt_scale: float = 0.35
    motion_joint_limit_margin: float = 0.10
    motion_command_position_scale: float = 0.25
    motion_command_orientation_scale: float = 0.80
    motion_arm_position_scale: float = 0.60
    motion_arm_velocity_scale: float = 4.0
    motion_command_smoothing_time: float = 0.20
    motion_quality_coefficient: float = 0.10
    motion_quality_loss_cap: float = 2.0

    def apply_debug_visualization(self) -> None:
        enabled = self.enable_debug_visualization
        self.commands.high_level.debug_vis = enabled
        self.target_pose_debug_vis = enabled
        getattr(self.scene, "object_frame").debug_vis = enabled
        getattr(self.scene, "hand_center_frame").debug_vis = enabled

    def __post_init__(self):
        self.decimation = self.high_level_decimation * self.low_level_decimation
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.low_level_decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 40 * 2**15
        self.scene.contact_forces.update_period = self.sim.dt
        for sensor_name in LEFT_GRIPPER_CONTACT_SENSOR_NAMES + RIGHT_GRIPPER_CONTACT_SENSOR_NAMES:
            getattr(self.scene, sensor_name).update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.low_level_decimation * self.sim.dt
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = True
