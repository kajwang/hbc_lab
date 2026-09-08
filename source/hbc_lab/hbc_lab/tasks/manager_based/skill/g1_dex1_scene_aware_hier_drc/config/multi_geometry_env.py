from __future__ import annotations

import torch

from hbc_lab.tasks.manager_based.skill.g1_dex1_hier_drc.mdp.object_mass_curriculum import (
    balanced_active_hand_progress,
    update_monotonic_curriculum_level,
)

from ..mdp.geometry_shaping import body_obstacle_clearance
from ..mdp.multi_geometry import (
    GEOMETRY_FAMILY_NAMES,
    paired_active_hand,
    paired_family_and_constraint,
)
from .scene_aware_env import G1Dex1SceneAwareEnv


def _condition_and_hand_balanced_progress(
    values: torch.Tensor,
    active_hand: torch.Tensor,
    constrained: torch.Tensor,
) -> torch.Tensor:
    """Return the weakest present open/constrained and left/right stratum."""
    condition_progress = []
    for condition in (False, True):
        mask = constrained == condition
        if bool(mask.any()):
            _, _, progress = balanced_active_hand_progress(values[mask], active_hand[mask])
            condition_progress.append(progress)
    if not condition_progress:
        return values.new_zeros(())
    return torch.stack(condition_progress).amin()


class G1Dex1MultiGeometryEnv(G1Dex1SceneAwareEnv):
    """Counterfactually paired scene-aware PnP with geometry and mass curricula."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        family_count = len(GEOMETRY_FAMILY_NAMES)
        env_ids = torch.arange(cfg.scene.num_envs, device=cfg.sim.device)
        if getattr(cfg, "geometry_ood_mode", False):
            family = torch.full_like(env_ids, family_count - 1)
            constrained = torch.ones_like(env_ids, dtype=torch.bool)
        elif getattr(cfg, "play_env_id", -1) >= 0:
            play_env_id = int(cfg.play_env_id)
            if not 0 <= play_env_id < family_count:
                raise ValueError(
                    f"play_env_id must be in [0, {family_count - 1}], got {play_env_id}."
                )
            family = torch.full_like(env_ids, play_env_id)
            if getattr(cfg, "geometry_preview_sweep", False):
                constrained = torch.remainder(env_ids, 2) == 1
            else:
                constrained = torch.ones_like(env_ids, dtype=torch.bool)
        elif getattr(cfg, "train_family_id", -1) >= 0:
            train_family_id = int(cfg.train_family_id)
            if not 0 <= train_family_id < family_count:
                raise ValueError(
                    f"train_family_id must be in [0, {family_count - 1}], got {train_family_id}."
                )
            family = torch.full_like(env_ids, train_family_id)
            constrained = torch.remainder(env_ids, 2) == 1
        else:
            family, constrained = paired_family_and_constraint(env_ids, family_count)
        self.geometry_family_id = family
        self.scene_is_constrained = constrained
        self.geometry_family_level = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)

        self.geometry_curriculum_levels = torch.full(
            (family_count,), cfg.geometry_curriculum_start_level, device=cfg.sim.device
        )
        self.geometry_safe_reach_streak = torch.zeros(
            cfg.scene.num_envs, dtype=torch.long, device=cfg.sim.device
        )
        self.geometry_safe_reach_rate = torch.zeros(family_count, device=cfg.sim.device)
        self.geometry_safe_reach_rate_ema = torch.zeros(family_count, device=cfg.sim.device)

        self.object_mass_curriculum_levels = torch.full(
            (family_count,), cfg.object_mass_family_start_level, device=cfg.sim.device
        )
        self.object_mass_family_progress = torch.zeros(family_count, device=cfg.sim.device)
        self.object_mass_family_progress_ema = torch.full(
            (family_count,), cfg.object_mass_family_start_level, device=cfg.sim.device
        )

        super().__init__(cfg, render_mode, **kwargs)

    def _reset_hier_buffers(self, env_ids: torch.Tensor):
        play_active_id = int(getattr(self.cfg, "play_active_id", -1))
        if play_active_id >= 0:
            if play_active_id not in (0, 1):
                raise ValueError(f"play_active_id must be 0 (left) or 1 (right), got {play_active_id}.")
            active_hand = torch.full_like(env_ids, play_active_id)
        else:
            active_hand = paired_active_hand(env_ids, len(GEOMETRY_FAMILY_NAMES))
        self.high_level_command.active_hand[env_ids] = active_hand
        self.high_level_command.contact_label.set_single_active_hand(env_ids, active_hand)
        super()._reset_hier_buffers(env_ids)
        self.geometry_safe_reach_streak[env_ids] = 0

    def _update_geometry_curriculum(self) -> None:
        constrained = self.scene_is_constrained
        minimum_clearance = body_obstacle_clearance(self).amin(dim=-1)
        instant_safe_reach = (
            constrained
            & (self.d_active_hand <= self.cfg.geometry_safe_reach_distance)
            & (minimum_clearance >= self.cfg.geometry_safe_clearance)
            & ~self.object_fallen
        )
        self.geometry_safe_reach_streak = torch.where(
            instant_safe_reach,
            self.geometry_safe_reach_streak + 1,
            torch.zeros_like(self.geometry_safe_reach_streak),
        )
        stable_safe_reach = (
            self.geometry_safe_reach_streak >= self.cfg.geometry_safe_reach_steps
        ).float()

        alpha = self.cfg.geometry_curriculum_safe_reach_ema_alpha
        for family_index in range(len(GEOMETRY_FAMILY_NAMES)):
            mask = constrained & (self.geometry_family_id == family_index)
            if not bool(mask.any()):
                continue
            _, _, rate = balanced_active_hand_progress(
                stable_safe_reach[mask], self.active_hand[mask]
            )
            self.geometry_safe_reach_rate[family_index] = rate
            self.geometry_safe_reach_rate_ema[family_index].mul_(1.0 - alpha).add_(alpha * rate)

        if not getattr(self.cfg, "geometry_curriculum_enabled", False):
            return
        if self.common_step_counter % self.cfg.geometry_curriculum_update_interval != 0:
            return
        can_advance = (
            self.geometry_safe_reach_rate_ema
            >= self.cfg.geometry_curriculum_advance_threshold
        )
        next_levels = torch.clamp(
            self.geometry_curriculum_levels + self.cfg.geometry_curriculum_level_step,
            max=1.0,
        )
        advanced = can_advance & (self.geometry_curriculum_levels < 1.0)
        self.geometry_curriculum_levels.copy_(
            torch.where(advanced, next_levels, self.geometry_curriculum_levels)
        )
        self.geometry_safe_reach_rate_ema[advanced] = 0.0

    def _update_object_mass_curriculum(self) -> None:
        self._update_geometry_curriculum()
        if not getattr(self.cfg, "object_mass_curriculum_enabled", False):
            return
        if hasattr(self.scene["object"].data, "object_pos_w"):
            return super()._update_object_mass_curriculum()

        progress = self.c_physical_grasp.detach()
        for family_index in range(len(GEOMETRY_FAMILY_NAMES)):
            mask = self.geometry_family_id == family_index
            if not bool(mask.any()):
                continue
            self.object_mass_family_progress[family_index] = (
                _condition_and_hand_balanced_progress(
                    progress[mask],
                    self.active_hand[mask],
                    self.scene_is_constrained[mask],
                )
            )

        next_ema, next_levels = update_monotonic_curriculum_level(
            self.object_mass_curriculum_levels,
            self.object_mass_family_progress_ema,
            self.object_mass_family_progress,
            alpha=self.cfg.object_mass_w_manip_ema_alpha,
        )
        self.object_mass_family_progress_ema.copy_(next_ema)
        self.object_mass_curriculum_levels.copy_(next_levels)

        # Retain the generic scalar diagnostics as family means.
        self.object_mass_w_manip_balanced.copy_(self.object_mass_family_progress.mean())
        self.object_mass_w_manip_ema.copy_(self.object_mass_family_progress_ema.mean())
        self.object_mass_curriculum_level.copy_(self.object_mass_curriculum_levels.mean())

    def _log_high_level_diagnostics(self) -> None:
        super()._log_high_level_diagnostics()
        log = self.extras["log"]
        for family_index, family_name in enumerate(GEOMETRY_FAMILY_NAMES):
            log[f"Curriculum/geometry_{family_name}_level"] = self.geometry_curriculum_levels[
                family_index
            ]
            log[f"Curriculum/geometry_{family_name}_safe_reach_rate"] = (
                self.geometry_safe_reach_rate[family_index]
            )
            log[f"Curriculum/geometry_{family_name}_safe_reach_rate_ema"] = (
                self.geometry_safe_reach_rate_ema[family_index]
            )
            log[f"Curriculum/object_mass_{family_name}_level"] = (
                self.object_mass_curriculum_levels[family_index]
            )
            log[f"Curriculum/object_mass_{family_name}_progress"] = (
                self.object_mass_family_progress[family_index]
            )
