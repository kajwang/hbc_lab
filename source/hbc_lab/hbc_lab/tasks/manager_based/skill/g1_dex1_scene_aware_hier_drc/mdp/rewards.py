from __future__ import annotations

import torch
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from hbc_lab.tasks.manager_based.skill.g1_dex1_hier_drc.mdp.rewards import G1Dex1HierDrcRewardsCfg

from .observations import local_environment_scan_obs
from .geometry_shaping import geometry_conditioned_posture_reward
from .multi_geometry import GEOMETRY_FAMILY_NAMES


def environment_collision_penalty(
    env,
    threshold: float,
    force_scale: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize non-foot/non-hand body contacts with the scene using deployable contact sensing."""
    sensor = env.scene[sensor_cfg.name]
    body_ids = sensor_cfg.body_ids
    force_history = sensor.data.net_forces_w_history[:, :, body_ids]
    max_force = torch.linalg.norm(force_history, dim=-1).amax(dim=(1, 2))
    penalty = torch.tanh(torch.clamp(max_force - threshold, min=0.0) / force_scale)

    env.extras["log"]["Environment/body_collision_ratio"] = (max_force > threshold).float().mean()
    env.extras["log"]["Environment/body_collision_force_mean"] = max_force.mean()
    family_id = getattr(env, "geometry_family_id", None)
    if family_id is not None:
        scan = local_environment_scan_obs(env).detach()
        for family_index, name in enumerate(GEOMETRY_FAMILY_NAMES):
            mask = family_id == family_index
            if bool(mask.any()):
                env.extras["log"][f"Environment/{name}_body_collision_ratio"] = (
                    max_force[mask] > threshold
                ).float().mean()
                env.extras["log"][f"Environment/{name}_scan_proximity_mean"] = scan[mask].mean()
                env.extras["log"][f"Environment/{name}_d_active_hand_mean"] = env.d_active_hand[mask].mean()
                env.extras["log"][f"Environment/{name}_physical_grasp_mean"] = (
                    env.c_physical_grasp[mask].mean()
                )
                env.extras["log"][f"Environment/{name}_success_ratio"] = (
                    env.task_succeeded[mask].float().mean()
                )
    return penalty


def distance_relaxed_hand_collision_penalty(
    env,
    threshold: float,
    force_scale: float,
    relax_distance: float,
    left_sensor_cfg: SceneEntityCfg,
    right_sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Keep both hands clear, but release only the active hand near its target."""
    sensor = env.scene[left_sensor_cfg.name]

    def maximum_force(body_ids: list[int]) -> torch.Tensor:
        history = sensor.data.net_forces_w_history[:, :, body_ids]
        return torch.linalg.norm(history, dim=-1).amax(dim=(1, 2))

    left_force = maximum_force(left_sensor_cfg.body_ids)
    right_force = maximum_force(right_sensor_cfg.body_ids)
    left_penalty = torch.tanh(torch.clamp(left_force - threshold, min=0.0) / force_scale)
    right_penalty = torch.tanh(torch.clamp(right_force - threshold, min=0.0) / force_scale)

    active_weight = torch.tanh(env.d_active_hand.detach() / max(relax_distance, 1.0e-6))
    left_active = env.active_hand == 0
    left_weight = torch.where(left_active, active_weight, torch.ones_like(active_weight))
    right_weight = torch.where(~left_active, active_weight, torch.ones_like(active_weight))
    penalty = left_weight * left_penalty + right_weight * right_penalty

    log = env.extras["log"]
    log["Environment/active_hand_contact_weight"] = active_weight.mean()
    log["Environment/left_hand_collision_force_mean"] = left_force.mean()
    log["Environment/right_hand_collision_force_mean"] = right_force.mean()
    log["Environment/hand_collision_penalty_mean"] = penalty.mean()
    return penalty


@configclass
class G1Dex1SceneAwareRewardsCfg(G1Dex1HierDrcRewardsCfg):
    environment_collision = RewTerm(
        func=environment_collision_penalty,
        weight=-5.0,
        params={
            "threshold": 5.0,
            "force_scale": 40.0,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["(?!.*ankle.*|.*foot.*|.*hand.*|.*wrist.*).*"],
            ),
        },
    )
    hand_environment_collision = RewTerm(
        func=distance_relaxed_hand_collision_penalty,
        weight=-2.0,
        params={
            "threshold": 5.0,
            "force_scale": 40.0,
            "relax_distance": 0.25,
            "left_sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["left_.*(?:hand|wrist).*"],
            ),
            "right_sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["right_.*(?:hand|wrist).*"],
            ),
        },
    )


@configclass
class G1Dex1SceneAwareGeometryRewardsCfg(G1Dex1SceneAwareRewardsCfg):
    geometry_posture = RewTerm(
        func=geometry_conditioned_posture_reward,
        weight=1.0,
        params={
            "safe_margin": 0.08,
            "transition_width": 0.06,
            "relative_coefficient": 0.05,
            "active_hand_relax_distance": 0.25,
        },
    )
