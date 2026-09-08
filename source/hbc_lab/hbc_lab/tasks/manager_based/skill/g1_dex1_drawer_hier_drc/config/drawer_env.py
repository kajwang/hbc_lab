from __future__ import annotations

import torch
from isaaclab.utils.math import quat_apply

from hbc_lab.tasks.manager_based.skill.contact_labels import ContactMode
from hbc_lab.tasks.manager_based.skill.contact_sensor_filters import select_contact_filter_force
from hbc_lab.tasks.manager_based.skill.g1_dex1_hier_drc.config.g1_dex1_env import G1Dex1HierDrcEnv
from hbc_lab.tasks.manager_based.skill.g1_dex1_hier_drc.mdp.contact_progress import (
    compute_active_hand_grasp_progress,
)
from hbc_lab.tasks.manager_based.skill.g1_dex1_hier_drc.mdp.drc_math import compute_drc_weights, update_ema
from hbc_lab.tasks.manager_based.skill.g1_dex1_hier_drc.mdp.scenes import HAND_CENTER_FRAME_NAME
from hbc_lab.tasks.manager_based.skill.pose_motion import (
    compute_cumulative_keyframe_progress,
    compute_masked_pose_error,
    gather_keyframe,
)


TOP_DRAWER = 0
BOTTOM_DRAWER = 1


class G1Dex1DrawerEnv(G1Dex1HierDrcEnv):
    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        num_envs = cfg.scene.num_envs
        device = cfg.sim.device
        self.selected_drawer = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.selected_drawer_joint_pos = torch.zeros(num_envs, device=device)
        self.selected_drawer_joint_vel = torch.zeros(num_envs, device=device)
        self.inactive_hand_contact = torch.zeros(num_envs, device=device)
        self.drawer_initial_pose_pending = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.drawer_target_update_delay = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._drawer_frame_indices: dict[str, int] = {}

        self.motion_target_pos_w = torch.zeros(num_envs, 2, 3, device=device)
        self.motion_target_quat_w = torch.zeros(num_envs, 2, 4, device=device)
        self.motion_target_quat_w[..., 0] = 1.0
        self.motion_position_mask = torch.ones(num_envs, 2, 3, device=device)
        self.motion_rotation_mask = torch.zeros(num_envs, 2, 3, device=device)
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
        self.motion_open_direction_w = torch.zeros(num_envs, 3, device=device)
        super().__init__(cfg, render_mode, **kwargs)

        cabinet = self.scene["object"]
        top_ids, top_names = cabinet.find_joints("drawer_top_joint")
        bottom_ids, bottom_names = cabinet.find_joints("drawer_bottom_joint")
        if len(top_ids) != 1 or len(bottom_ids) != 1:
            raise RuntimeError(
                "Sektion cabinet requires drawer_top_joint and drawer_bottom_joint, "
                f"resolved top={top_names}, bottom={bottom_names}"
            )
        self.drawer_joint_ids = torch.tensor(
            [int(top_ids[0]), int(bottom_ids[0])],
            dtype=torch.long,
            device=self.device,
        )

    def _get_drawer_frame_index(self, frame_name: str) -> int:
        if frame_name not in self._drawer_frame_indices:
            frame_names = list(self.scene["object_frame"].data.target_frame_names)
            if frame_name not in frame_names:
                raise RuntimeError(f"Drawer frame {frame_name!r} not found in {frame_names}")
            self._drawer_frame_indices[frame_name] = frame_names.index(frame_name)
        return self._drawer_frame_indices[frame_name]

    def _all_handle_poses_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        frame = self.scene["object_frame"].data
        frame_indices = [
            self._get_drawer_frame_index("drawer_handle_top"),
            self._get_drawer_frame_index("drawer_handle_bottom"),
        ]
        return frame.target_pos_w[:, frame_indices, :], frame.target_quat_w[:, frame_indices, :]

    def _selected_handle_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        positions_w, quaternions_w = self._all_handle_poses_w()
        batch_ids = torch.arange(self.num_envs, device=self.device)
        return positions_w[batch_ids, self.selected_drawer], quaternions_w[batch_ids, self.selected_drawer]

    def _selected_handle_pos_w(self) -> torch.Tensor:
        return self._selected_handle_pose_w()[0]

    def _selected_handle_quat_w(self) -> torch.Tensor:
        return self._selected_handle_pose_w()[1]

    def _selected_drawer_joint_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        cabinet = self.scene["object"]
        batch_ids = torch.arange(self.num_envs, device=self.device)
        joint_pos = cabinet.data.joint_pos[:, self.drawer_joint_ids]
        joint_vel = cabinet.data.joint_vel[:, self.drawer_joint_ids]
        return joint_pos[batch_ids, self.selected_drawer], joint_vel[batch_ids, self.selected_drawer]

    def _sum_sensor_force(self, sensor_name: str) -> torch.Tensor:
        sensor = self.scene.sensors[sensor_name]
        force_matrix_w = getattr(sensor.data, "force_matrix_w", None)
        if force_matrix_w is None:
            return super()._sum_sensor_force(sensor_name)
        return select_contact_filter_force(
            force_matrix_w[..., :3],
            filter_indices=self.selected_drawer,
        )

    def _initialize_motion_guide(
        self,
        env_ids: torch.Tensor,
        closed_pos_w: torch.Tensor,
        closed_quat_w: torch.Tensor,
    ) -> None:
        cabinet = self.scene["object"]
        open_axis_local = torch.tensor(
            self.cfg.drawer_open_axis_local,
            device=self.device,
            dtype=closed_pos_w.dtype,
        ).expand(env_ids.numel(), -1)
        open_direction_w = quat_apply(cabinet.data.root_quat_w[env_ids], open_axis_local)
        open_direction_w = torch.nn.functional.normalize(open_direction_w, dim=-1)
        open_delta_w = open_direction_w * self.cfg.drawer_open_distance

        self.motion_target_pos_w[env_ids, 0] = closed_pos_w + open_delta_w
        self.motion_target_pos_w[env_ids, 1] = closed_pos_w
        self.motion_target_quat_w[env_ids, 0] = closed_quat_w
        self.motion_target_quat_w[env_ids, 1] = closed_quat_w
        self.motion_position_mask[env_ids] = 1.0
        self.motion_rotation_mask[env_ids] = 0.0
        self.motion_keyframe_index[env_ids] = 0
        self.motion_keyframe_stable_count[env_ids] = 0
        self.motion_guide_valid[env_ids] = True
        self.motion_sequence_complete[env_ids] = False
        self.motion_phase_initial_error_valid[env_ids] = False
        self.motion_cumulative_progress[env_ids] = 0.0
        self.motion_progress[env_ids] = 0.0
        self.motion_current_pos_w[env_ids] = closed_pos_w
        self.motion_current_quat_w[env_ids] = closed_quat_w
        self.motion_open_direction_w[env_ids] = open_direction_w
        self.object_initial_pos_w[env_ids] = closed_pos_w
        self.object_target_pos_w[env_ids] = self.motion_target_pos_w[env_ids, 0]

    def _update_motion_guide(self) -> None:
        current_pos_w, current_quat_w = self._selected_handle_pose_w()
        self.motion_current_pos_w.copy_(current_pos_w)
        self.motion_current_quat_w.copy_(current_quat_w)
        self.motion_keyframe_reached.zero_()

        if not torch.any(self.motion_guide_valid):
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
        self.object_target_pos_w.copy_(gather_keyframe(self.motion_target_pos_w, self.motion_keyframe_index))

    def _object_frame_pos_w(self) -> torch.Tensor:
        return self._selected_handle_pos_w()

    def _object_fallen(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _update_object_mass_curriculum(self) -> None:
        return

    def _update_contact_target_regions(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        delayed = self.drawer_target_update_delay[env_ids] > 0
        if torch.any(delayed):
            self.drawer_target_update_delay[env_ids[delayed]] -= 1
            return

        handle_pos_w, handle_quat_w = self._selected_handle_pose_w()
        target_region = torch.zeros(self.num_envs, 2, 3, device=self.device)
        active_mask = self.contact_label.effector_mask.to(dtype=torch.bool)
        target_region = torch.where(active_mask.unsqueeze(-1), handle_pos_w.unsqueeze(1), target_region)
        target_orientation = handle_quat_w.unsqueeze(1).expand(-1, 2, -1)
        self.contact_label.set_target_region_pose(
            env_ids,
            target_region[env_ids],
            target_orientation[env_ids],
        )

        ready = self.drawer_initial_pose_pending[env_ids]
        ready_ids = env_ids[ready]
        if ready_ids.numel() > 0:
            self._initialize_motion_guide(
                ready_ids,
                handle_pos_w[ready_ids],
                handle_quat_w[ready_ids],
            )
            self.drawer_initial_pose_pending[ready_ids] = False

        valid = self.motion_guide_valid[env_ids]
        valid_ids = env_ids[valid]
        if valid_ids.numel() > 0:
            current_target = gather_keyframe(self.motion_target_pos_w, self.motion_keyframe_index)
            self.object_target_pos_w[valid_ids] = current_target[valid_ids]

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

        self.selected_drawer_joint_pos, self.selected_drawer_joint_vel = self._selected_drawer_joint_state()
        current_target = gather_keyframe(self.motion_target_pos_w, self.motion_keyframe_index)
        self.d_goal = torch.norm(self._selected_handle_pos_w() - current_target, dim=-1)
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
        success_condition = self.motion_sequence_complete & (
            self.c_couple > self.cfg.success_couple_threshold
        )
        self.success_proximity_count[success_condition] += 1
        self.success_proximity_count[~success_condition] = 0
        self.task_succeeded = self.success_proximity_count >= self.cfg.success_steps

    def _reset_hier_buffers(self, env_ids: torch.Tensor):
        super()._reset_hier_buffers(env_ids)
        top_selected = torch.rand(env_ids.numel(), device=self.device) < self.cfg.top_drawer_probability
        self.selected_drawer[env_ids] = torch.where(
            top_selected,
            torch.full_like(env_ids, TOP_DRAWER),
            torch.full_like(env_ids, BOTTOM_DRAWER),
        )
        self.drawer_initial_pose_pending[env_ids] = True
        self.drawer_target_update_delay[env_ids] = 1
        self.selected_drawer_joint_pos[env_ids] = 0.0
        self.selected_drawer_joint_vel[env_ids] = 0.0
        self.inactive_hand_contact[env_ids] = 0.0
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
        self.motion_open_direction_w[env_ids] = 0.0

        fixed_mask = self.cfg.commands.high_level.fixed_effector_mask
        if fixed_mask is None or sum(value > 0.0 for value in fixed_mask) != 1:
            raise ValueError("Drawer requires exactly one active effector in fixed_effector_mask")
        mask = torch.tensor(fixed_mask, device=self.device).repeat(env_ids.numel(), 1)
        self.active_hand[env_ids] = torch.argmax(mask, dim=-1)
        self.contact_label.set_effector_mask(env_ids, mask)
        self.contact_label.set_contact_mode(env_ids, ContactMode.GRASP)

    def _log_link_contact_diagnostics(self, active_hand: torch.Tensor | None = None) -> None:
        super()._log_link_contact_diagnostics(active_hand)
        self.extras["log"]["Drawer/top_selected_ratio"] = (
            self.selected_drawer == TOP_DRAWER
        ).float().mean()
        self.extras["log"]["Drawer/selected_joint_pos_mean"] = self.selected_drawer_joint_pos.mean()
        self.extras["log"]["Drawer/selected_joint_vel_mean"] = self.selected_drawer_joint_vel.mean()
        self.extras["log"]["Drawer/inactive_hand_contact_mean"] = self.inactive_hand_contact.mean()
        self.extras["log"]["Motion/keyframe_index_mean"] = self.motion_keyframe_index.float().mean()
        self.extras["log"]["Motion/position_error_mean"] = self.motion_position_error.mean()
        self.extras["log"]["Motion/orientation_error_mean"] = self.motion_orientation_error.mean()
        self.extras["log"]["Motion/progress_mean"] = self.motion_progress.mean()
        self.extras["log"]["Motion/cumulative_progress_mean"] = self.motion_cumulative_progress.mean()
        self.extras["log"]["Motion/keyframe_reached_ratio"] = self.motion_keyframe_reached.float().mean()
        self.extras["log"]["Motion/sequence_complete_ratio"] = self.motion_sequence_complete.float().mean()
