from __future__ import annotations

import torch

from hbc_lab.tasks.manager_based.skill.contact_labels import ContactMode
from hbc_lab.tasks.manager_based.skill.g1_dex1_hier_drc.config.g1_dex1_env import (
    G1Dex1HierDrcEnv,
)
from hbc_lab.tasks.manager_based.skill.g1_dex1_hier_drc.mdp.contact_progress import (
    compute_active_hand_grasp_progress,
)
from hbc_lab.tasks.manager_based.skill.g1_dex1_hier_drc.mdp.drc_math import compute_drc_weights, update_ema
from hbc_lab.tasks.manager_based.skill.g1_dex1_hier_drc.mdp.scenes import HAND_CENTER_FRAME_NAME
from hbc_lab.tasks.manager_based.skill.pose_motion import (
    compose_local_axis_pose,
    compute_cumulative_keyframe_progress,
    compute_masked_pose_error,
    gather_keyframe,
    quaternion_apply,
    quaternion_conjugate,
)


class G1Dex1DoorOpenEnv(G1Dex1HierDrcEnv):
    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        num_envs = cfg.scene.num_envs
        device = cfg.sim.device
        self.handle_angle = torch.zeros(num_envs, device=device)
        self.hinge_angle = torch.zeros(num_envs, device=device)
        self.inactive_hand_contact = torch.zeros(num_envs, device=device)
        self.latch_released = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.door_initial_pose_pending = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.door_target_update_delay = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._door_frame_indices: dict[str, int] = {}
        self.handle_body_id: int | None = None
        self.motion_target_pos_w = torch.zeros(num_envs, 2, 3, device=device)
        self.motion_target_quat_w = torch.zeros(num_envs, 2, 4, device=device)
        self.motion_target_quat_w[..., 0] = 1.0
        self.motion_position_mask = torch.ones(num_envs, 2, 3, device=device)
        self.motion_rotation_mask = torch.ones(num_envs, 2, 3, device=device)
        self.motion_keyframe_index = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.motion_keyframe_stable_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.motion_guide_valid = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.motion_sequence_complete = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.motion_phase_initial_error = torch.zeros(num_envs, device=device)
        self.motion_phase_initial_error_valid = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.motion_cumulative_progress = torch.zeros(num_envs, device=device)
        self.motion_progress = torch.zeros(num_envs, device=device)
        self.motion_position_error = torch.zeros(num_envs, device=device)
        self.motion_orientation_error = torch.zeros(num_envs, device=device)
        self.motion_keyframe_reached = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.motion_current_pos_w = torch.zeros(num_envs, 3, device=device)
        self.motion_current_quat_w = torch.zeros(num_envs, 4, device=device)
        self.motion_current_quat_w[:, 0] = 1.0
        super().__init__(cfg, render_mode, **kwargs)

        door = self.scene["object"]
        hinge_ids, hinge_names = door.find_joints("joint_1")
        handle_ids, handle_names = door.find_joints("joint_2")
        handle_body_ids, handle_body_names = door.find_bodies("link_2")
        if len(hinge_ids) != 1 or len(handle_ids) != 1 or len(handle_body_ids) != 1:
            raise RuntimeError(
                "Door requires joint_1, joint_2, and link_2, "
                f"resolved hinge={hinge_names}, handle={handle_names}, body={handle_body_names}"
            )
        self.hinge_joint_id = int(hinge_ids[0])
        self.handle_joint_id = int(handle_ids[0])
        self.handle_body_id = int(handle_body_ids[0])

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

    def _door_handle_quat_w(self) -> torch.Tensor:
        index = self._get_door_frame_index("door_handle")
        return self.scene["object_frame"].data.target_quat_w[:, index, :]

    def _door_handle_pivot_pos_w(self) -> torch.Tensor:
        if self.handle_body_id is None:
            handle_body_ids, handle_body_names = self.scene["object"].find_bodies("link_2")
            if len(handle_body_ids) != 1:
                raise RuntimeError(f"Door requires link_2, resolved body={handle_body_names}")
            self.handle_body_id = int(handle_body_ids[0])
        return self.scene["object"].data.body_pos_w[:, self.handle_body_id, :]

    def _door_goal_pos_w(self) -> torch.Tensor:
        index = self._get_door_frame_index("door_handle_goal")
        return self.scene["object_frame"].data.target_pos_w[:, index, :]

    def _door_goal_quat_w(self) -> torch.Tensor:
        index = self._get_door_frame_index("door_handle_goal")
        return self.scene["object_frame"].data.target_quat_w[:, index, :]

    def _initialize_motion_guide(
        self,
        env_ids: torch.Tensor,
        handle_pos_w: torch.Tensor,
        handle_quat_w: torch.Tensor,
        handle_pivot_pos_w: torch.Tensor,
        goal_pos_w: torch.Tensor,
        goal_quat_w: torch.Tensor,
    ) -> None:
        axis_local = torch.tensor(
            self.cfg.motion_unlock_axis_local,
            device=self.device,
            dtype=handle_quat_w.dtype,
        ).unsqueeze(0).expand(env_ids.numel(), -1)
        unlock_angle = torch.full(
            (env_ids.numel(),),
            self.cfg.motion_unlock_angle,
            device=self.device,
            dtype=handle_quat_w.dtype,
        )
        handle_offset_local = quaternion_apply(
            quaternion_conjugate(handle_quat_w),
            handle_pos_w - handle_pivot_pos_w,
        )
        goal_pivot_pos_w = goal_pos_w - quaternion_apply(goal_quat_w, handle_offset_local)
        unlock_pos_w, unlock_quat_w = compose_local_axis_pose(
            handle_pos_w,
            handle_quat_w,
            handle_pivot_pos_w,
            axis_local,
            unlock_angle,
        )
        open_pos_w, open_quat_w = compose_local_axis_pose(
            goal_pos_w,
            goal_quat_w,
            goal_pivot_pos_w,
            axis_local,
            unlock_angle,
        )

        self.motion_target_pos_w[env_ids, 0] = unlock_pos_w
        self.motion_target_pos_w[env_ids, 1] = open_pos_w
        self.motion_target_quat_w[env_ids, 0] = unlock_quat_w
        self.motion_target_quat_w[env_ids, 1] = open_quat_w
        self.motion_position_mask[env_ids] = 1.0
        self.motion_rotation_mask[env_ids] = 1.0
        self.motion_keyframe_index[env_ids] = 0
        self.motion_keyframe_stable_count[env_ids] = 0
        self.motion_guide_valid[env_ids] = True
        self.motion_sequence_complete[env_ids] = False
        self.motion_phase_initial_error_valid[env_ids] = False
        self.motion_cumulative_progress[env_ids] = 0.0
        self.motion_progress[env_ids] = 0.0
        self.motion_current_pos_w[env_ids] = handle_pos_w
        self.motion_current_quat_w[env_ids] = handle_quat_w

    def _update_motion_guide(self) -> None:
        current_pos_w = self._door_handle_pos_w()
        current_quat_w = self._door_handle_quat_w()
        self.motion_current_pos_w.copy_(current_pos_w)
        self.motion_current_quat_w.copy_(current_quat_w)
        self.motion_keyframe_reached.zero_()

        valid_ids = torch.nonzero(self.motion_guide_valid, as_tuple=False).squeeze(-1)
        if valid_ids.numel() == 0:
            self.motion_position_error.zero_()
            self.motion_orientation_error.zero_()
            self.motion_cumulative_progress.zero_()
            self.motion_progress.zero_()
            return

        target_pos_w = gather_keyframe(self.motion_target_pos_w, self.motion_keyframe_index)
        target_quat_w = gather_keyframe(self.motion_target_quat_w, self.motion_keyframe_index)
        position_mask = gather_keyframe(self.motion_position_mask, self.motion_keyframe_index)
        rotation_mask = gather_keyframe(self.motion_rotation_mask, self.motion_keyframe_index)
        total_error, position_error, orientation_error = compute_masked_pose_error(
            current_pos=current_pos_w,
            current_quat=current_quat_w,
            target_pos=target_pos_w,
            target_quat=target_quat_w,
            position_mask=position_mask,
            rotation_mask=rotation_mask,
            position_scale=self.cfg.motion_position_scale,
            rotation_scale=self.cfg.motion_rotation_scale,
        )
        self.motion_position_error.copy_(position_error * self.cfg.motion_position_scale)
        self.motion_orientation_error.copy_(orientation_error * self.cfg.motion_rotation_scale)

        initialize_phase = self.motion_guide_valid & ~self.motion_phase_initial_error_valid
        self.motion_phase_initial_error[initialize_phase] = torch.clamp(
            total_error[initialize_phase],
            min=torch.finfo(total_error.dtype).eps,
        )
        self.motion_phase_initial_error_valid[initialize_phase] = True
        cumulative, normalized = compute_cumulative_keyframe_progress(
            current_error=total_error,
            phase_initial_error=self.motion_phase_initial_error,
            keyframe_index=self.motion_keyframe_index,
            num_keyframes=self.motion_target_pos_w.shape[1],
        )
        self.motion_cumulative_progress.copy_(
            torch.where(self.motion_guide_valid, cumulative, torch.zeros_like(cumulative))
        )
        self.motion_progress.copy_(
            torch.where(self.motion_guide_valid, normalized, torch.zeros_like(normalized))
        )
        within_target = (
            (self.motion_position_error <= self.cfg.motion_position_tolerance)
            & (self.motion_orientation_error <= self.cfg.motion_rotation_tolerance)
            & self.motion_guide_valid
        )
        self.motion_keyframe_stable_count = torch.where(
            within_target,
            self.motion_keyframe_stable_count + 1,
            torch.zeros_like(self.motion_keyframe_stable_count),
        )
        stable = self.motion_keyframe_stable_count >= self.cfg.motion_keyframe_stable_steps
        last_keyframe = self.motion_keyframe_index == self.motion_target_pos_w.shape[1] - 1
        advance = stable & ~last_keyframe
        finish = stable & last_keyframe & ~self.motion_sequence_complete
        self.motion_keyframe_reached = advance | finish
        self.motion_sequence_complete |= finish
        self.motion_keyframe_index[advance] += 1
        self.motion_keyframe_stable_count[advance] = 0

        self.motion_phase_initial_error_valid[advance] = False

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
        handle_quat_w = self._door_handle_quat_w()
        handle_pivot_pos_w = self._door_handle_pivot_pos_w()
        goal_pos_w = self._door_goal_pos_w()
        goal_quat_w = self._door_goal_quat_w()
        target_region = torch.zeros(self.num_envs, 2, 3, device=self.device)
        active_mask = self.contact_label.effector_mask.to(dtype=torch.bool)
        target_region = torch.where(active_mask.unsqueeze(-1), handle_pos_w.unsqueeze(1), target_region)
        target_orientation = handle_quat_w.unsqueeze(1).expand(-1, 2, -1)
        self.contact_label.set_target_region_pose(
            env_ids,
            target_region[env_ids],
            target_orientation[env_ids],
        )
        self.object_target_pos_w[env_ids] = goal_pos_w[env_ids]

        ready = self.door_initial_pose_pending[env_ids]
        ready_ids = env_ids[ready]
        if ready_ids.numel() > 0:
            self.object_initial_pos_w[ready_ids] = handle_pos_w[ready_ids]
            self._initialize_motion_guide(
                ready_ids,
                handle_pos_w[ready_ids],
                handle_quat_w[ready_ids],
                handle_pivot_pos_w[ready_ids],
                goal_pos_w[ready_ids],
                goal_quat_w[ready_ids],
            )
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
        self._update_motion_guide()
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
        left_contact = torch.stack(
            (
                self._step_link_contact["left_Link1_2"],
                self._step_link_contact["left_Link1_3"],
                self._step_link_contact["left_Link2_2"],
                self._step_link_contact["left_Link2_3"],
            ),
            dim=-1,
        ).amax(dim=-1)
        right_contact = torch.stack(
            (
                self._step_link_contact["right_Link1_2"],
                self._step_link_contact["right_Link1_3"],
                self._step_link_contact["right_Link2_2"],
                self._step_link_contact["right_Link2_3"],
            ),
            dim=-1,
        ).amax(dim=-1)
        self.inactive_hand_contact = torch.where(self.active_hand == 0, right_contact, left_contact)

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
        self.inactive_hand_contact[env_ids] = 0.0
        self.latch_released[env_ids] = False
        self.motion_keyframe_index[env_ids] = 0
        self.motion_keyframe_stable_count[env_ids] = 0
        self.motion_guide_valid[env_ids] = False
        self.motion_sequence_complete[env_ids] = False
        self.motion_phase_initial_error[env_ids] = 0.0
        self.motion_phase_initial_error_valid[env_ids] = False
        self.motion_cumulative_progress[env_ids] = 0.0
        self.motion_progress[env_ids] = 0.0
        self.motion_position_error[env_ids] = 0.0
        self.motion_orientation_error[env_ids] = 0.0
        self.motion_keyframe_reached[env_ids] = False
        fixed_mask = self.cfg.commands.high_level.fixed_effector_mask
        if fixed_mask is None or sum(value > 0.0 for value in fixed_mask) != 1:
            raise ValueError("DoorOpen requires exactly one active effector in fixed_effector_mask")
        mask = torch.tensor(fixed_mask, device=self.device).repeat(env_ids.numel(), 1)
        self.active_hand[env_ids] = torch.argmax(mask, dim=-1)
        self.contact_label.set_effector_mask(env_ids, mask)
        self.contact_label.set_contact_mode(env_ids, ContactMode.GRASP)

    def _log_link_contact_diagnostics(self, active_hand: torch.Tensor | None = None) -> None:
        super()._log_link_contact_diagnostics(active_hand)
        self.extras["log"]["Door/handle_angle_mean"] = self.handle_angle.mean()
        self.extras["log"]["Door/hinge_angle_mean"] = self.hinge_angle.mean()
        self.extras["log"]["Door/latch_released_ratio"] = self.latch_released.float().mean()
        self.extras["log"]["Door/inactive_hand_contact_mean"] = self.inactive_hand_contact.mean()
        self.extras["log"]["Motion/keyframe_index_mean"] = self.motion_keyframe_index.float().mean()
        self.extras["log"]["Motion/position_error_mean"] = self.motion_position_error.mean()
        self.extras["log"]["Motion/orientation_error_mean"] = self.motion_orientation_error.mean()
        self.extras["log"]["Motion/progress_mean"] = self.motion_progress.mean()
        self.extras["log"]["Motion/cumulative_progress_mean"] = self.motion_cumulative_progress.mean()
        self.extras["log"]["Motion/keyframe_reached_ratio"] = self.motion_keyframe_reached.float().mean()
        self.extras["log"]["Motion/sequence_complete_ratio"] = self.motion_sequence_complete.float().mean()
