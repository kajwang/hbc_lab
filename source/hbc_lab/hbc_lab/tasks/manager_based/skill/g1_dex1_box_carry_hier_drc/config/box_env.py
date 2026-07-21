from __future__ import annotations

import torch
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from hbc_lab.assets.objects import BOX_CUBE_CENTER_Z, BOX_CUBE_SIZE
from hbc_lab.tasks.manager_based.skill.g1_dex1_hier_drc.config.g1_dex1_env import (
    G1Dex1HierDrcEnv,
)
from hbc_lab.tasks.manager_based.skill.g1_dex1_hier_drc.mdp.drc_math import compute_drc_weights, update_ema

from ..mdp.contact_progress import (
    compute_bimanual_distance_gated_couple,
    compute_bimanual_support_progress,
    compute_independent_support_reward_terms,
)
from ..mdp.face_targets import (
    choose_left_positive_assignment,
    compute_long_axis_face_centers,
    select_assigned_face_targets,
)
from ..mdp.scenes import BOX_SUPPORT_CONTACT_KEYS, HAND_CENTER_FRAME_NAME


LEFT_FACE_TARGET_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/G1Dex1BoxCarry/left_face_target",
    markers={
        "target": sim_utils.SphereCfg(
            radius=0.035,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.35, 1.0)),
        ),
    },
)
RIGHT_FACE_TARGET_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/G1Dex1BoxCarry/right_face_target",
    markers={
        "target": sim_utils.SphereCfg(
            radius=0.035,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.55, 0.05)),
        ),
    },
)


class G1Dex1BoxCarryEnv(G1Dex1HierDrcEnv):
    """Bimanual ground-box transport guided by opposing long-axis face centers."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        self.left_hand_object_distance = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.right_hand_object_distance = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.left_support_contact = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.right_support_contact = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.bimanual_support_contact = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.bimanual_contact_gate = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.left_support_distance_gate = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.right_support_distance_gate = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.bimanual_distance_gate = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.distance_gated_couple = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.gated_support = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.early_contact = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.box_lift_height = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.d_goal_xy = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self.left_face_uses_positive = torch.zeros(
            cfg.scene.num_envs,
            dtype=torch.bool,
            device=cfg.sim.device,
        )
        self.face_assignment_pending = torch.ones(
            cfg.scene.num_envs,
            dtype=torch.bool,
            device=cfg.sim.device,
        )
        self.left_face_target_visualizer = None
        self.right_face_target_visualizer = None
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

    def _update_contact_target_regions(self, env_ids: torch.Tensor | None = None) -> None:
        obj = self.scene["object"]
        positive_w, negative_w = compute_long_axis_face_centers(
            obj.data.root_pos_w,
            obj.data.root_quat_w,
            half_extent=0.5 * BOX_CUBE_SIZE[0],
        )
        hand_center_pos_w = self.scene[HAND_CENTER_FRAME_NAME].data.target_pos_w
        pending = self.face_assignment_pending
        if torch.any(pending):
            left_positive = choose_left_positive_assignment(
                positive_w,
                negative_w,
                hand_center_pos_w[:, 0, :],
                hand_center_pos_w[:, 1, :],
            )
            self.left_face_uses_positive[pending] = left_positive[pending]
            self.face_assignment_pending[pending] = False

        left_target_w, right_target_w = select_assigned_face_targets(
            positive_w,
            negative_w,
            self.left_face_uses_positive,
        )
        target_region = torch.stack((left_target_w, right_target_w), dim=1)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self.contact_label.set_target_region(env_ids, target_region[env_ids])

    def _compute_progress(self):
        self._update_contact_target_regions()
        object_pos_w = self._object_frame_pos_w()
        hand_center_pos_w = self.scene[HAND_CENTER_FRAME_NAME].data.target_pos_w
        target_region = self.contact_label.target_region
        left_distance = torch.norm(
            hand_center_pos_w[:, 0, :] - target_region[:, 0, :],
            dim=-1,
        )
        right_distance = torch.norm(
            hand_center_pos_w[:, 1, :] - target_region[:, 1, :],
            dim=-1,
        )
        progress = compute_bimanual_support_progress(
            left_distance=left_distance,
            right_distance=right_distance,
            left_region_contacts=self._support_region_contacts("left"),
            right_region_contacts=self._support_region_contacts("right"),
        )
        left_distance_gate, right_distance_gate, gated_support, early_contact = (
            compute_independent_support_reward_terms(
                left_distance=progress.left_distance,
                right_distance=progress.right_distance,
                left_support=progress.left_support,
                right_support=progress.right_support,
                distance_scale=0.15,
            )
        )
        self.left_support_distance_gate = left_distance_gate
        self.right_support_distance_gate = right_distance_gate
        self.bimanual_distance_gate = torch.minimum(
            self.left_support_distance_gate,
            self.right_support_distance_gate,
        )
        self.gated_support = gated_support
        self.early_contact = early_contact
        self.distance_gated_couple = compute_bimanual_distance_gated_couple(
            left_gate=self.left_support_distance_gate,
            right_gate=self.right_support_distance_gate,
            raw_couple=progress.couple_gate,
        )

        self.left_hand_object_distance = progress.left_distance
        self.right_hand_object_distance = progress.right_distance
        self.d_active_hand = progress.distance
        self.left_support_contact = progress.left_support
        self.right_support_contact = progress.right_support
        self.bimanual_support_contact = progress.support_density
        self.bimanual_contact_gate = progress.couple_gate
        self.left_hand_contact = progress.left_contact_gate
        self.right_hand_contact = progress.right_contact_gate
        self._step_left_contact = progress.left_contact_gate
        self._step_right_contact = progress.right_contact_gate
        self.active_left_finger_contact = progress.left_contact_gate
        self.active_right_finger_contact = progress.right_contact_gate
        self.active_left_finger_force = torch.amax(self._support_region_forces("left"), dim=-1)
        self.active_right_finger_force = torch.amax(self._support_region_forces("right"), dim=-1)
        self.active_grip = 0.5 * (
            self.command_state.left_grip.squeeze(-1) + self.command_state.right_grip.squeeze(-1)
        )

        self.d_goal = torch.norm(object_pos_w - self.object_target_pos_w, dim=-1)
        self.d_goal_xy = torch.norm((object_pos_w - self.object_target_pos_w)[:, :2], dim=-1)
        self.c_contact = update_ema(self.c_contact, progress.support_density, alpha=0.2)
        self.c_grasp = update_ema(self.c_grasp, progress.support_density, alpha=0.2)
        self.c_couple = update_ema(self.c_couple, self.distance_gated_couple, alpha=0.2)
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
        self.bimanual_contact_gate[env_ids] = 0.0
        self.left_support_distance_gate[env_ids] = 0.0
        self.right_support_distance_gate[env_ids] = 0.0
        self.bimanual_distance_gate[env_ids] = 0.0
        self.distance_gated_couple[env_ids] = 0.0
        self.gated_support[env_ids] = 0.0
        self.early_contact[env_ids] = 0.0
        self.box_lift_height[env_ids] = 0.0
        self.d_goal_xy[env_ids] = 0.0
        self.left_face_uses_positive[env_ids] = False
        self.face_assignment_pending[env_ids] = True

    def _update_target_pose_visualization(self) -> None:
        super()._update_target_pose_visualization()
        if not getattr(self.cfg, "target_pose_debug_vis", False):
            return
        self._update_contact_target_regions()
        if self.left_face_target_visualizer is None:
            self.left_face_target_visualizer = VisualizationMarkers(LEFT_FACE_TARGET_MARKER_CFG)
            self.right_face_target_visualizer = VisualizationMarkers(RIGHT_FACE_TARGET_MARKER_CFG)
            self.left_face_target_visualizer.set_visibility(True)
            self.right_face_target_visualizer.set_visibility(True)
        self.left_face_target_visualizer.visualize(self.contact_label.target_region[:, 0, :])
        self.right_face_target_visualizer.visualize(self.contact_label.target_region[:, 1, :])

    def _log_link_contact_diagnostics(self, active_hand: torch.Tensor | None = None) -> None:
        super()._log_link_contact_diagnostics(active_hand)
        for side in ("left", "right"):
            self.extras["log"][f"ContactLink/{side}_palm_mean"] = self._step_link_contact[
                f"{side}_palm"
            ].mean()
            self.extras["log"][f"ContactLink/{side}_palm_force"] = self._step_link_force[f"{side}_palm"].mean()
        self.extras["log"]["BoxCarry/left_support_mean"] = self.left_support_contact.mean()
        self.extras["log"]["BoxCarry/right_support_mean"] = self.right_support_contact.mean()
        self.extras["log"]["BoxCarry/bimanual_support_mean"] = self.bimanual_support_contact.mean()
        self.extras["log"]["BoxCarry/support_density_mean"] = self.bimanual_support_contact.mean()
        self.extras["log"]["BoxCarry/couple_gate_mean"] = self.bimanual_contact_gate.mean()
        self.extras["log"]["BoxCarry/bimanual_distance_gate_mean"] = self.bimanual_distance_gate.mean()
        self.extras["log"]["BoxCarry/distance_gated_couple_mean"] = self.distance_gated_couple.mean()
        for value, log_name in (
            (self.left_support_distance_gate, "BoxCarry/left_support_distance_gate"),
            (self.right_support_distance_gate, "BoxCarry/right_support_distance_gate"),
            (self.gated_support, "BoxCarry/gated_support_mean"),
            (self.early_contact, "BoxCarry/early_contact_mean"),
        ):
            self.extras["log"][log_name] = value.mean()
        self.extras["log"]["BoxCarry/left_face_target_error"] = self.left_hand_object_distance.mean()
        self.extras["log"]["BoxCarry/right_face_target_error"] = self.right_hand_object_distance.mean()
        self.extras["log"]["BoxCarry/left_positive_assignment_ratio"] = (
            self.left_face_uses_positive.float().mean()
        )
        self.extras["log"]["BoxCarry/lift_height_mean"] = self.box_lift_height.mean()
        self.extras["log"]["BoxCarry/goal_distance_xy_mean"] = self.d_goal_xy.mean()
        self.extras["log"]["BoxCarry/left_effector_required"] = self.contact_label.effector_mask[:, 0].mean()
        self.extras["log"]["BoxCarry/right_effector_required"] = self.contact_label.effector_mask[:, 1].mean()
