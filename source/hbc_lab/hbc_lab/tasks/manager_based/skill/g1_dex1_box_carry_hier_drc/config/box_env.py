from __future__ import annotations

import torch

from hbc_lab.assets.objects import BOX_CUBE_CENTER_Z
from hbc_lab.tasks.manager_based.skill.g1_dex1_hier_drc.config.g1_dex1_env import (
    G1Dex1HierDrcEnv,
)
from hbc_lab.tasks.manager_based.skill.g1_dex1_hier_drc.mdp.drc_math import compute_drc_weights, update_ema

from ..mdp.contact_progress import compute_bimanual_support_progress
from ..mdp.scenes import BOX_SUPPORT_CONTACT_KEYS, HAND_CENTER_FRAME_NAME


class G1Dex1BoxCarryEnv(G1Dex1HierDrcEnv):
    """Bimanual ground-box transport task using center-pose guidance."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        self.left_hand_object_distance = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.right_hand_object_distance = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.left_support_contact = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.right_support_contact = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.bimanual_support_contact = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.box_lift_height = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        super().__init__(cfg, render_mode, **kwargs)

    def _reset_contact_accumulators(self) -> None:
        super()._reset_contact_accumulators()
        for side in ("left", "right"):
            self._step_link_contact[f"{side}_palm"] = torch.zeros(self.num_envs, device=self.device)
            self._step_link_force[f"{side}_palm"] = torch.zeros(self.num_envs, device=self.device)

    def _accumulate_contact_components(self) -> None:
        self._accumulate_link_contact_diagnostics()
        for side in ("left", "right"):
            force_w = self._sum_sensor_force(f"{side}_palm_contact")
            contact, force = self._contact_confidence_from_force(force_w)
            key = f"{side}_palm"
            self._step_link_contact[key] = torch.maximum(self._step_link_contact[key], contact)
            self._step_link_force[key] = torch.maximum(self._step_link_force[key], force)

    def _support_region_contacts(self, side: str) -> torch.Tensor:
        return torch.stack(
            [self._step_link_contact[f"{side}_{region}"] for region in BOX_SUPPORT_CONTACT_KEYS],
            dim=-1,
        )

    def _support_region_forces(self, side: str) -> torch.Tensor:
        return torch.stack(
            [self._step_link_force[f"{side}_{region}"] for region in BOX_SUPPORT_CONTACT_KEYS],
            dim=-1,
        )

    def _object_fallen(self) -> torch.Tensor:
        object_root_z = self.scene["object"].data.root_pos_w[:, 2]
        return object_root_z < self.scene.env_origins[:, 2] - 0.02

    def _compute_progress(self):
        object_pos_w = self._object_frame_pos_w()
        hand_center_pos_w = self.scene[HAND_CENTER_FRAME_NAME].data.target_pos_w
        left_distance = torch.norm(hand_center_pos_w[:, 0, :] - object_pos_w, dim=-1)
        right_distance = torch.norm(hand_center_pos_w[:, 1, :] - object_pos_w, dim=-1)
        progress = compute_bimanual_support_progress(
            left_distance=left_distance,
            right_distance=right_distance,
            left_region_contacts=self._support_region_contacts("left"),
            right_region_contacts=self._support_region_contacts("right"),
        )

        self.left_hand_object_distance = progress.left_distance
        self.right_hand_object_distance = progress.right_distance
        self.d_active_hand = progress.distance
        self.left_support_contact = progress.left_support
        self.right_support_contact = progress.right_support
        self.bimanual_support_contact = progress.bimanual_support
        self.left_hand_contact = progress.left_support
        self.right_hand_contact = progress.right_support
        self._step_left_contact = progress.left_support
        self._step_right_contact = progress.right_support
        self.active_left_finger_contact = progress.left_support
        self.active_right_finger_contact = progress.right_support
        self.active_left_finger_force = torch.amax(self._support_region_forces("left"), dim=-1)
        self.active_right_finger_force = torch.amax(self._support_region_forces("right"), dim=-1)
        self.active_grip = 0.5 * (
            self.command_state.left_grip.squeeze(-1) + self.command_state.right_grip.squeeze(-1)
        )

        self.d_goal = torch.norm(object_pos_w - self.object_target_pos_w, dim=-1)
        mean_support = 0.5 * (progress.left_support + progress.right_support)
        self.c_contact = update_ema(self.c_contact, mean_support, alpha=0.2)
        self.c_grasp = update_ema(self.c_grasp, progress.bimanual_support, alpha=0.2)
        self.c_couple = update_ema(self.c_couple, progress.bimanual_support, alpha=0.2)
        zeros = torch.zeros_like(self.c_couple)
        self.c_opposition = update_ema(self.c_opposition, zeros, alpha=0.2)
        self.c_pinch = update_ema(self.c_pinch, zeros, alpha=0.2)
        self.object_fallen = self._object_fallen()
        self.box_lift_height = self.scene["object"].data.root_pos_w[:, 2] - (
            self.scene.env_origins[:, 2] + BOX_CUBE_CENTER_Z
        )

        weights = compute_drc_weights(self.d_active_hand, self.c_couple, alpha=5.0)
        self.W_app = weights[:, 0]
        self.W_couple = weights[:, 1]
        self.W_manip = weights[:, 2]
        self._update_object_mass_curriculum()

    def _reset_hier_buffers(self, env_ids: torch.Tensor):
        super()._reset_hier_buffers(env_ids)
        self.left_hand_object_distance[env_ids] = 0.0
        self.right_hand_object_distance[env_ids] = 0.0
        self.left_support_contact[env_ids] = 0.0
        self.right_support_contact[env_ids] = 0.0
        self.bimanual_support_contact[env_ids] = 0.0
        self.box_lift_height[env_ids] = 0.0

    def _log_link_contact_diagnostics(self) -> None:
        super()._log_link_contact_diagnostics()
        for side in ("left", "right"):
            self.extras["log"][f"ContactLink/{side}_palm_mean"] = self._step_link_contact[
                f"{side}_palm"
            ].mean()
            self.extras["log"][f"ContactLink/{side}_palm_force"] = self._step_link_force[f"{side}_palm"].mean()
        self.extras["log"]["BoxCarry/left_support_mean"] = self.left_support_contact.mean()
        self.extras["log"]["BoxCarry/right_support_mean"] = self.right_support_contact.mean()
        self.extras["log"]["BoxCarry/bimanual_support_mean"] = self.bimanual_support_contact.mean()
        self.extras["log"]["BoxCarry/left_center_distance"] = self.left_hand_object_distance.mean()
        self.extras["log"]["BoxCarry/right_center_distance"] = self.right_hand_object_distance.mean()
        self.extras["log"]["BoxCarry/lift_height_mean"] = self.box_lift_height.mean()
        self.extras["log"]["BoxCarry/left_effector_required"] = self.contact_label.effector_mask[:, 0].mean()
        self.extras["log"]["BoxCarry/right_effector_required"] = self.contact_label.effector_mask[:, 1].mean()
