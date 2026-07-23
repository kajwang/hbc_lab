from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from hbc_lab.tasks.locomotion import mdp

from ..mdp.actions import G1Dex1HierDrcActionsCfg
from ..mdp.commands import G1Dex1HierDrcCommandsCfg
from ..mdp.curriculums import G1Dex1HierDrcCurriculumCfg
from ..mdp.events import G1Dex1HierDrcEventCfg
from ..mdp.observations import G1Dex1HierDrcObservationsCfg
from ..mdp.rewards import G1Dex1HierDrcRewardsCfg
from ..mdp.scenes import G1Dex1HierDrcSceneCfg, LEFT_GRIPPER_CONTACT_SENSOR_NAMES, RIGHT_GRIPPER_CONTACT_SENSOR_NAMES


def joint_vel_explosion(
    env: ManagerBasedRLEnv,
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    is_exploded = (torch.abs(joint_vel) > threshold) | torch.isnan(joint_vel) | torch.isinf(joint_vel)
    return torch.any(is_exploded, dim=1)


@configclass
class G1Dex1HierDrcTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.25})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.2})
    joint_vel_explosion = DoneTerm(func=joint_vel_explosion, params={"threshold": 1000.0})


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
    target_pose_debug_vis: bool = False
    debug_fixed_gripper: bool = False
    debug_fixed_left_grip: float = 1.0
    debug_fixed_right_grip: float = 1.0
    high_level_decimation: int = 5
    low_level_decimation: int = 4
    action_dim: int = 19
    low_level_obs_history_length: int = 5
    low_level_action_scale: float = 0.25
    low_level_action_clip: float | None = None
    finite_obs_clip: float = 100.0
    hand_contact_force_threshold: float = 2.0
    close_distance: float = 0.20
    close_gate_width: float = 0.10
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

    def __post_init__(self):
        self.decimation = self.high_level_decimation * self.low_level_decimation
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.low_level_decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.scene.contact_forces.update_period = self.sim.dt
        for sensor_name in LEFT_GRIPPER_CONTACT_SENSOR_NAMES + RIGHT_GRIPPER_CONTACT_SENSOR_NAMES:
            getattr(self.scene, sensor_name).update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.low_level_decimation * self.sim.dt
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = True
