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

    constrained = getattr(env, "scene_is_constrained", None)
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
                if constrained is not None:
                    for condition_name, condition_mask in (
                        ("open", mask & ~constrained),
                        ("constrained", mask & constrained),
                    ):
                        if bool(condition_mask.any()):
                            prefix = f"Environment/{name}_{condition_name}"
                            env.extras["log"][f"{prefix}_d_active_hand_mean"] = (
                                env.d_active_hand[condition_mask].mean()
                            )
                            env.extras["log"][f"{prefix}_physical_grasp_mean"] = (
                                env.c_physical_grasp[condition_mask].mean()
                            )
                            env.extras["log"][f"{prefix}_success_ratio"] = (
                                env.task_succeeded[condition_mask].float().mean()
                            )
    elif constrained is not None:
        scan = local_environment_scan_obs(env).detach()
        env.extras["log"]["Environment/table_ratio"] = constrained.float().mean()
        for name, mask in (("open", ~constrained), ("table", constrained)):
            if bool(mask.any()):
                env.extras["log"][f"Environment/{name}_body_collision_ratio"] = (
                    max_force[mask] > threshold
                ).float().mean()
                env.extras["log"][f"Environment/{name}_scan_proximity_mean"] = scan[mask].mean()
                env.extras["log"][f"Environment/{name}_d_active_hand_mean"] = env.d_active_hand[mask].mean()
                env.extras["log"][f"Environment/{name}_physical_grasp_mean"] = env.c_physical_grasp[mask].mean()
                env.extras["log"][f"Environment/{name}_success_ratio"] = env.task_succeeded[mask].float().mean()
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


@configclass
class G1Dex1SceneAwareGeometryRewardsCfg(G1Dex1SceneAwareRewardsCfg):
    geometry_posture = RewTerm(
        func=geometry_conditioned_posture_reward,
        weight=1.0,
        params={
            "safe_margin": 0.08,
            "transition_width": 0.06,
            "relative_coefficient": 0.05,
        },
    )
