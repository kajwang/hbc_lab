from __future__ import annotations

import torch

from hbc_lab.tasks.manager_based.skill.contact_labels import ContactMode, RIGHT_HAND
from hbc_lab.tasks.manager_based.skill.g1_dex1_hier_drc.config.g1_dex1_env import (
    G1Dex1HierDrcEnv,
)
from hbc_lab.tasks.manager_based.skill.g1_dex1_hier_drc.mdp.contact_progress import (
    compute_active_hand_grasp_progress,
)
from hbc_lab.tasks.manager_based.skill.g1_dex1_hier_drc.mdp.drc_math import compute_drc_weights, update_ema
from hbc_lab.tasks.manager_based.skill.g1_dex1_hier_drc.mdp.scenes import HAND_CENTER_FRAME_NAME


class G1Dex1DoorOpenEnv(G1Dex1HierDrcEnv):
    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        num_envs = cfg.scene.num_envs
        device = cfg.sim.device
        self.handle_angle = torch.zeros(num_envs, device=device)
        self.hinge_angle = torch.zeros(num_envs, device=device)
        self.inactive_left_contact = torch.zeros(num_envs, device=device)
        self.latch_released = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.door_initial_pose_pending = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.door_target_update_delay = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._door_frame_indices: dict[str, int] = {}
        super().__init__(cfg, render_mode, **kwargs)

        door = self.scene["object"]
        hinge_ids, hinge_names = door.find_joints("joint_1")
        handle_ids, handle_names = door.find_joints("joint_2")
        if len(hinge_ids) != 1 or len(handle_ids) != 1:
            raise RuntimeError(
                f"Door requires joint_1 and joint_2, resolved hinge={hinge_names}, handle={handle_names}"
            )
        self.hinge_joint_id = int(hinge_ids[0])
        self.handle_joint_id = int(handle_ids[0])

    def _get_door_frame_index(self, frame_name: str) -> int:
        if frame_name not in self._door_frame_indices:
            frame_names = list(self.scene["object_frame"].data.target_frame_names)
            if frame_name not in frame_names:
                raise RuntimeError(f"Door frame {frame_name!r} not found in {frame_names}")
            self._door_frame_indices[frame_name] = frame_names.index(frame_name)
        return self._door_frame_indices[frame_name]

    def _door_handle_pos_w(self) -> torch.Tensor:
        index = self._get_door_frame_index("door_handle")
        return self.scene["object_frame"].data.target_pos_w[:, index, :]

    def _door_goal_pos_w(self) -> torch.Tensor:
        index = self._get_door_frame_index("door_handle_goal")
        return self.scene["object_frame"].data.target_pos_w[:, index, :]

    def _object_frame_pos_w(self) -> torch.Tensor:
        return self._door_handle_pos_w()

    def _object_fallen(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _update_object_mass_curriculum(self) -> None:
        return

    def _update_contact_target_regions(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        delayed = self.door_target_update_delay[env_ids] > 0
        if torch.any(delayed):
            self.door_target_update_delay[env_ids[delayed]] -= 1
            # Accessing the FrameTransformer here would cache the pre-forward pose.
            return

        handle_pos_w = self._door_handle_pos_w()
        target_region = torch.zeros(self.num_envs, 2, 3, device=self.device)
        target_region[:, RIGHT_HAND, :] = handle_pos_w
        self.contact_label.set_target_region(env_ids, target_region[env_ids])
        self.object_target_pos_w[env_ids] = self._door_goal_pos_w()[env_ids]

        ready = self.door_initial_pose_pending[env_ids]
        ready_ids = env_ids[ready]
        if ready_ids.numel() > 0:
            self.object_initial_pos_w[ready_ids] = handle_pos_w[ready_ids]
            self.door_initial_pose_pending[ready_ids] = False

    def _simulate_door_latch(self) -> None:
        door = self.scene["object"]
        hinge_pos = door.data.joint_pos[:, self.hinge_joint_id]
        hinge_vel = door.data.joint_vel[:, self.hinge_joint_id]
        handle_pos = door.data.joint_pos[:, self.handle_joint_id]
        self.latch_released = (torch.abs(handle_pos) > self.cfg.door_latch_handle_threshold) | (
            torch.abs(hinge_pos) > self.cfg.door_latch_hinge_release_threshold
        )
        locked_torque = -self.cfg.door_latch_stiffness * hinge_pos - self.cfg.door_latch_damping * hinge_vel
        applied_torque = torch.where(self.latch_released, torch.zeros_like(locked_torque), locked_torque)
        door.set_joint_effort_target(applied_torque.unsqueeze(-1), joint_ids=[self.hinge_joint_id])

    def _apply_low_level_action(self, low_action: torch.Tensor) -> torch.Tensor:
        applied_action = super()._apply_low_level_action(low_action)
        self._simulate_door_latch()
        return applied_action

    def _compute_progress(self):
        self._update_contact_target_regions()
        hand_center_pos_w = self.scene[HAND_CENTER_FRAME_NAME].data.target_pos_w
        target_region = self.contact_label.target_region
        left_distance = torch.norm(hand_center_pos_w[:, 0, :] - target_region[:, 0, :], dim=-1)
        right_distance = torch.norm(hand_center_pos_w[:, 1, :] - target_region[:, 1, :], dim=-1)
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
        self.left_hand_contact = self._step_left_contact
        self.right_hand_contact = self._step_right_contact
        self.inactive_left_contact = torch.stack(
            (
                self._step_link_contact["left_Link1_2"],
                self._step_link_contact["left_Link1_3"],
                self._step_link_contact["left_Link2_2"],
                self._step_link_contact["left_Link2_3"],
            ),
            dim=-1,
        ).amax(dim=-1)

        door = self.scene["object"]
        self.hinge_angle = door.data.joint_pos[:, self.hinge_joint_id]
        self.handle_angle = door.data.joint_pos[:, self.handle_joint_id]
        self.d_goal = torch.norm(self._door_handle_pos_w() - self.object_target_pos_w, dim=-1)
        self.c_contact = update_ema(self.c_contact, progress.contact, alpha=0.2)
        self.c_opposition = update_ema(self.c_opposition, progress.pinch, alpha=0.2)
        self.c_pinch = update_ema(self.c_pinch, progress.pinch, alpha=0.2)
        self.c_grasp = update_ema(self.c_grasp, progress.grasp, alpha=0.2)
        self.c_couple = update_ema(self.c_couple, progress.grasp, alpha=0.2)
        self.object_fallen.zero_()
        weights = compute_drc_weights(self.d_active_hand, self.c_couple, alpha=5.0)
        self.W_app = weights[:, 0]
        self.W_couple = weights[:, 1]
        self.W_manip = weights[:, 2]

    def _check_success(self):
        success_condition = (self.hinge_angle >= self.cfg.door_hinge_target) & (
            self.c_couple > self.cfg.success_couple_threshold
        )
        self.success_proximity_count[success_condition] += 1
        self.success_proximity_count[~success_condition] = 0
        self.task_succeeded = self.success_proximity_count >= self.cfg.success_steps

    def _reset_hier_buffers(self, env_ids: torch.Tensor):
        super()._reset_hier_buffers(env_ids)
        self.door_initial_pose_pending[env_ids] = True
        self.door_target_update_delay[env_ids] = 1
        self.handle_angle[env_ids] = 0.0
        self.hinge_angle[env_ids] = 0.0
        self.inactive_left_contact[env_ids] = 0.0
        self.latch_released[env_ids] = False
        self.active_hand[env_ids] = RIGHT_HAND
        mask = torch.tensor((0.0, 1.0), device=self.device).repeat(env_ids.numel(), 1)
        self.contact_label.set_effector_mask(env_ids, mask)
        self.contact_label.set_contact_mode(env_ids, ContactMode.GRASP)

    def _log_link_contact_diagnostics(self, active_hand: torch.Tensor | None = None) -> None:
        super()._log_link_contact_diagnostics(active_hand)
        self.extras["log"]["Door/handle_angle_mean"] = self.handle_angle.mean()
        self.extras["log"]["Door/hinge_angle_mean"] = self.hinge_angle.mean()
        self.extras["log"]["Door/latch_released_ratio"] = self.latch_released.float().mean()
        self.extras["log"]["Door/inactive_left_contact_mean"] = self.inactive_left_contact.mean()
