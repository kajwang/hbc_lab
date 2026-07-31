from __future__ import annotations

import torch

from hbc_lab.tasks.manager_based.skill.contact_labels import ContactMode
from hbc_lab.tasks.manager_based.skill.g1_dex1_hier_drc.config.g1_dex1_env import (
    G1Dex1HierDrcEnv,
)
from hbc_lab.tasks.manager_based.skill.g1_dex1_hier_drc.mdp.contact_progress import (
    compute_gripper_contact_components,
)
from hbc_lab.tasks.manager_based.skill.g1_dex1_hier_drc.mdp.drc_math import compute_drc_weights, update_ema
from hbc_lab.tasks.manager_based.skill.g1_dex1_hier_drc.mdp.scenes import HAND_CENTER_FRAME_NAME

from ..mdp.progress import (
    CartManipulationProgress,
    bimanual_approach_distance,
    bimanual_grasp_confidence,
    compute_cart_goal,
    compute_handle_targets,
    compute_transport_progress,
)


class G1Dex1CartPushEnv(G1Dex1HierDrcEnv):
    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        num_envs = cfg.scene.num_envs
        device = cfg.sim.device
        self.left_hand_object_distance = torch.zeros(num_envs, device=device)
        self.right_hand_object_distance = torch.zeros(num_envs, device=device)
        self.left_grip = torch.zeros(num_envs, device=device)
        self.right_grip = torch.zeros(num_envs, device=device)
        self.left_contact = torch.zeros(num_envs, device=device)
        self.right_contact = torch.zeros(num_envs, device=device)
        self.left_grasp = torch.zeros(num_envs, device=device)
        self.right_grasp = torch.zeros(num_envs, device=device)
        self.independent_contact = torch.zeros(num_envs, device=device)
        self.independent_grasp = torch.zeros(num_envs, device=device)
        self.bimanual_contact = torch.zeros(num_envs, device=device)
        self.bimanual_grasp = torch.zeros(num_envs, device=device)
        self.transport_progress = torch.zeros(num_envs, device=device)
        self.cart_goal_forward = torch.zeros(num_envs, device=device)
        self.cart_goal_lateral = torch.zeros(num_envs, device=device)
        self.cart_initial_quat_w = torch.zeros(num_envs, 4, device=device)
        self.cart_initial_quat_w[:, 0] = 1.0
        self.cart_goal_pending = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.cart_goal_update_delay = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.cart_frame_refresh_pending = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.cart_manipulation_progress = CartManipulationProgress(self.transport_progress)
        super().__init__(cfg, render_mode, **kwargs)
        self.object_mass.copy_(self.scene["object"].data.default_mass.sum(dim=-1))

    def _cart_handle_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        frame = self.scene["object_frame"].data
        return frame.target_pos_w[:, 0, :], frame.target_quat_w[:, 0, :]

    def _object_frame_pos_w(self) -> torch.Tensor:
        return self._cart_handle_pose_w()[0]

    def _object_fallen(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _update_object_mass_curriculum(self) -> None:
        return

    def _update_cart_goal_while_settling(self, handle_pos_w: torch.Tensor) -> None:
        settling_ids = self.cart_goal_pending.nonzero(as_tuple=False).squeeze(-1)
        if settling_ids.numel() == 0:
            return
        self.object_initial_pos_w[settling_ids] = handle_pos_w[settling_ids]
        self.object_target_pos_w[settling_ids] = compute_cart_goal(
            handle_pos_w[settling_ids],
            self.cart_initial_quat_w[settling_ids],
            self.cart_goal_forward[settling_ids],
            self.cart_goal_lateral[settling_ids],
        )

    def _update_contact_target_regions(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        if torch.any(self.cart_frame_refresh_pending[env_ids]):
            self.cart_frame_refresh_pending[env_ids] = False
            # Accessing the FrameTransformer here would cache the pre-forward pose.
            return

        handle_pos_w, handle_quat_w = self._cart_handle_pose_w()
        left_target_w, right_target_w = compute_handle_targets(
            handle_pos_w,
            handle_quat_w,
            half_width=self.cfg.cart_handle_target_half_width,
        )
        target_region = torch.stack((left_target_w, right_target_w), dim=1)
        self.contact_label.set_target_region(env_ids, target_region[env_ids])
        self._update_cart_goal_while_settling(handle_pos_w)

    def _compute_progress(self):
        self._update_contact_target_regions()
        settling_ids = (
            self.cart_goal_pending & (self.cart_goal_update_delay > 0)
        ).nonzero(as_tuple=False).squeeze(-1)
        if settling_ids.numel() > 0:
            self.cart_goal_update_delay[settling_ids] -= 1
            finished_ids = settling_ids[self.cart_goal_update_delay[settling_ids] == 0]
            self.cart_goal_pending[finished_ids] = False

        hand_center_pos_w = self.scene[HAND_CENTER_FRAME_NAME].data.target_pos_w
        target_region = self.contact_label.target_region
        left_distance = torch.norm(hand_center_pos_w[:, 0, :] - target_region[:, 0, :], dim=-1)
        right_distance = torch.norm(hand_center_pos_w[:, 1, :] - target_region[:, 1, :], dim=-1)

        left_progress = compute_gripper_contact_components(
            left_finger_force_w=self._step_left_left_force_w,
            right_finger_force_w=self._step_left_right_force_w,
            grip=self.command_state.left_grip,
            distance=left_distance,
            force_threshold=self.cfg.hand_contact_force_threshold,
            close_distance=self.cfg.close_distance,
            close_gate_width=self.cfg.close_gate_width,
        )
        right_progress = compute_gripper_contact_components(
            left_finger_force_w=self._step_right_left_force_w,
            right_finger_force_w=self._step_right_right_force_w,
            grip=self.command_state.right_grip,
            distance=right_distance,
            force_threshold=self.cfg.hand_contact_force_threshold,
            close_distance=self.cfg.close_distance,
            close_gate_width=self.cfg.close_gate_width,
        )

        self.left_hand_object_distance = left_distance
        self.right_hand_object_distance = right_distance
        self.d_active_hand = bimanual_approach_distance(left_distance, right_distance)
        self.left_grip = self.command_state.left_grip.squeeze(-1)
        self.right_grip = self.command_state.right_grip.squeeze(-1)
        self.left_contact = left_progress.contact
        self.right_contact = right_progress.contact
        self.left_grasp = left_progress.grasp
        self.right_grasp = right_progress.grasp
        self.independent_contact = 0.5 * (self.left_contact + self.right_contact)
        self.independent_grasp = 0.5 * (self.left_grasp + self.right_grasp)
        self.bimanual_contact = bimanual_grasp_confidence(self.left_contact, self.right_contact)
        self.bimanual_grasp = bimanual_grasp_confidence(self.left_grasp, self.right_grasp)

        self.active_grip = 0.5 * (self.left_grip + self.right_grip)
        self.active_left_finger_contact = 0.5 * (
            left_progress.left_contact + right_progress.left_contact
        )
        self.active_right_finger_contact = 0.5 * (
            left_progress.right_contact + right_progress.right_contact
        )
        self.active_left_finger_force = 0.5 * (left_progress.left_force + right_progress.left_force)
        self.active_right_finger_force = 0.5 * (
            left_progress.right_force + right_progress.right_force
        )
        self.active_opposition = 0.5 * (left_progress.pinch + right_progress.pinch)
        self.active_pinch_score = 0.5 * (
            left_progress.pinch_score + right_progress.pinch_score
        )
        self.active_contact_cos_sim = 0.5 * (left_progress.cos_sim + right_progress.cos_sim)
        self.left_hand_contact = self._step_left_contact
        self.right_hand_contact = self._step_right_contact

        handle_pos_w = self._object_frame_pos_w()
        self.d_goal = torch.norm(
            (handle_pos_w - self.object_target_pos_w)[:, :2],
            dim=-1,
        )
        initial_distance = torch.norm(
            (self.object_initial_pos_w - self.object_target_pos_w)[:, :2],
            dim=-1,
        )
        self.transport_progress = compute_transport_progress(initial_distance, self.d_goal)
        self.cart_manipulation_progress = CartManipulationProgress(self.transport_progress)

        self.c_contact = update_ema(self.c_contact, self.bimanual_contact, alpha=0.2)
        mean_pinch = 0.5 * (left_progress.pinch + right_progress.pinch)
        self.c_opposition = update_ema(self.c_opposition, mean_pinch, alpha=0.2)
        self.c_pinch = update_ema(self.c_pinch, mean_pinch, alpha=0.2)
        self.c_grasp = update_ema(self.c_grasp, self.bimanual_grasp, alpha=0.2)
        self.c_couple = update_ema(self.c_couple, self.bimanual_grasp, alpha=0.2)
        self.object_fallen.zero_()
        weights = compute_drc_weights(self.d_active_hand, self.c_couple, alpha=5.0)
        self.W_app = weights[:, 0]
        self.W_couple = weights[:, 1]
        self.W_manip = weights[:, 2]

    def _check_success(self):
        success_condition = (self.d_goal < self.cfg.success_distance) & (
            self.c_couple > self.cfg.success_couple_threshold
        )
        self.success_proximity_count[success_condition] += 1
        self.success_proximity_count[~success_condition] = 0
        self.task_succeeded = self.success_proximity_count >= self.cfg.success_steps

    def _reset_hier_buffers(self, env_ids: torch.Tensor):
        super()._reset_hier_buffers(env_ids)
        self.cart_goal_pending[env_ids] = True
        self.cart_goal_update_delay[env_ids] = self.cfg.cart_goal_settle_steps
        self.cart_frame_refresh_pending[env_ids] = True
        for buffer in (
            self.left_hand_object_distance,
            self.right_hand_object_distance,
            self.left_grip,
            self.right_grip,
            self.left_contact,
            self.right_contact,
            self.left_grasp,
            self.right_grasp,
            self.independent_contact,
            self.independent_grasp,
            self.bimanual_contact,
            self.bimanual_grasp,
            self.transport_progress,
        ):
            buffer[env_ids] = 0.0
        mask = torch.ones(env_ids.numel(), 2, device=self.device)
        self.contact_label.set_effector_mask(env_ids, mask)
        self.contact_label.set_contact_mode(env_ids, ContactMode.GRASP)

    def _log_link_contact_diagnostics(self, active_hand: torch.Tensor | None = None) -> None:
        super()._log_link_contact_diagnostics(active_hand)
        self.extras["log"]["Cart/left_target_distance_mean"] = self.left_hand_object_distance.mean()
        self.extras["log"]["Cart/right_target_distance_mean"] = self.right_hand_object_distance.mean()
        self.extras["log"]["Cart/left_contact_mean"] = self.left_contact.mean()
        self.extras["log"]["Cart/right_contact_mean"] = self.right_contact.mean()
        self.extras["log"]["Cart/left_grasp_mean"] = self.left_grasp.mean()
        self.extras["log"]["Cart/right_grasp_mean"] = self.right_grasp.mean()
        self.extras["log"]["Cart/independent_contact_mean"] = self.independent_contact.mean()
        self.extras["log"]["Cart/independent_grasp_mean"] = self.independent_grasp.mean()
        self.extras["log"]["Cart/bimanual_contact_mean"] = self.bimanual_contact.mean()
        self.extras["log"]["Cart/bimanual_grasp_mean"] = self.bimanual_grasp.mean()
        self.extras["log"]["Cart/transport_progress_mean"] = self.transport_progress.mean()
        self.extras["log"]["Cart/goal_distance_xy_mean"] = self.d_goal.mean()
        self.extras["log"]["Cart/goal_pending_ratio"] = self.cart_goal_pending.float().mean()
