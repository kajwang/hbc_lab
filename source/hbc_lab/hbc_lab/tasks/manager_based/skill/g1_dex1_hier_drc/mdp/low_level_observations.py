from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.utils import math as math_utils

from hbc_lab.assets.robots.unitree import G1_29DOF_BODY_JOINT_NAMES
from hbc_lab.tasks.locomotion.mdp.pose_transforms import posture_anchor_pose_w

from .high_level_actions import HighLevelCommandState
from .scenes import HAND_CENTER_FRAME_NAME


class G1SphericalPostureLowLevelObsBuilder:
    """Build the frozen G1 spherical-posture low-level policy observation."""

    def __init__(
        self,
        env,
        history_length: int = 5,
        height_scan_offset: float = 0.5,
        finite_obs_clip: float | None = 100.0,
    ):
        self.env = env
        self.history_length = history_length
        self.height_scan_offset = height_scan_offset
        self.anchor_height_offset = 0.43
        self.finite_obs_clip = finite_obs_clip
        self.device = env.device
        self.joint_names = G1_29DOF_BODY_JOINT_NAMES
        self.joint_ids: list[int] | None = None
        self.left_anchor_body_id: int | None = None
        self.right_anchor_body_id: int | None = None
        self.torso_body_id: int | None = None
        self._term_history: list[torch.Tensor] | None = None
        self._history_valid: torch.Tensor | None = None

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if self._term_history is None:
            return
        if env_ids is None:
            self._history_valid[:] = False
        else:
            self._history_valid[env_ids] = False

    def _resolve_ids(self):
        robot: Articulation = self.env.scene["robot"]
        if self.joint_ids is None:
            joint_ids, joint_names = robot.find_joints(self.joint_names, preserve_order=False)
            if len(joint_ids) != len(self.joint_names):
                raise RuntimeError(f"Expected {len(self.joint_names)} low-level joints, got {len(joint_ids)}: {joint_names}")
            self.joint_ids = list(joint_ids)
            self.left_anchor_body_id = robot.find_bodies("left_shoulder_pitch_link")[0][0]
            self.right_anchor_body_id = robot.find_bodies("right_shoulder_pitch_link")[0][0]
            self.torso_body_id = robot.find_bodies("torso_link")[0][0]
        return (
            self.joint_ids,
            self.left_anchor_body_id,
            self.right_anchor_body_id,
            self.torso_body_id,
        )

    def _frame_pose_in_root_frame(self, frame_index: int) -> torch.Tensor:
        robot: Articulation = self.env.scene["robot"]
        frame_sensor = self.env.scene[HAND_CENTER_FRAME_NAME]
        pos_b, quat_b = math_utils.subtract_frame_transforms(
            robot.data.root_pos_w,
            robot.data.root_quat_w,
            frame_sensor.data.target_pos_w[:, frame_index],
            frame_sensor.data.target_quat_w[:, frame_index],
        )
        return torch.cat((pos_b, quat_b), dim=-1)

    def _height_scan(self) -> torch.Tensor:
        if "height_scanner" not in self.env.scene.sensors:
            return torch.zeros(self.env.num_envs, 0, device=self.device)
        sensor = self.env.scene.sensors["height_scanner"]
        height_scan = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - self.height_scan_offset
        return height_scan.clip(-1.0, 5.0)

    def _posture_error(self, command_state: HighLevelCommandState, torso_body_id: int) -> torch.Tensor:
        robot: Articulation = self.env.scene["robot"]
        root_height = robot.data.root_pos_w[:, 2] - self.env.scene.env_origins[:, 2]
        _, torso_pitch, _ = math_utils.euler_xyz_from_quat(robot.data.body_quat_w[:, torso_body_id])
        current_posture = torch.stack((root_height, torso_pitch), dim=-1)
        return command_state.posture_command - current_posture

    def _anchor_pose_w(self, side: str, posture_command: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        robot: Articulation = self.env.scene["robot"]
        _, left_anchor_id, right_anchor_id, _ = self._resolve_ids()
        anchor_id = left_anchor_id if side == "left" else right_anchor_id
        assert anchor_id is not None

        return posture_anchor_pose_w(
            robot.data.root_pos_w,
            robot.data.root_quat_w,
            robot.data.body_pos_w[:, anchor_id],
            self.env.scene.env_origins,
            posture_command,
            self.anchor_height_offset,
        )

    def _target_pose_w(
        self,
        pose_b: torch.Tensor,
        side: str,
        posture_command: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return math_utils.combine_frame_transforms(
            *self._anchor_pose_w(side, posture_command),
            pose_b[:, :3],
            pose_b[:, 3:],
        )

    def _target_pos_w(self, pose_b: torch.Tensor, side: str, posture_command: torch.Tensor) -> torch.Tensor:
        target_pos_w, _ = self._target_pose_w(pose_b, side, posture_command)
        return target_pos_w

    def _orientation_error_b(
        self,
        current_pos_w: torch.Tensor,
        current_quat_w: torch.Tensor,
        target_pos_w: torch.Tensor,
        target_quat_w: torch.Tensor,
    ) -> torch.Tensor:
        robot: Articulation = self.env.scene["robot"]
        _, rot_error_w = math_utils.compute_pose_error(current_pos_w, current_quat_w, target_pos_w, target_quat_w)
        return math_utils.quat_apply_inverse(robot.data.root_quat_w, rot_error_w)

    def _single_frame_terms(
        self,
        command_state: HighLevelCommandState,
        last_low_level_action: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        robot: Articulation = self.env.scene["robot"]
        joint_ids, _, _, torso_body_id = self._resolve_ids()
        assert torso_body_id is not None

        joint_pos_rel = robot.data.joint_pos[:, joint_ids] - robot.data.default_joint_pos[:, joint_ids]
        joint_vel_rel = robot.data.joint_vel[:, joint_ids]
        left_current = self._frame_pose_in_root_frame(frame_index=0)
        right_current = self._frame_pose_in_root_frame(frame_index=1)
        left_current = left_current.clip(-2.0, 2.0)
        right_current = right_current.clip(-2.0, 2.0)
        left_target_w, left_target_quat_w = self._target_pose_w(
            command_state.left_wrist_pose_b,
            "left",
            command_state.posture_command,
        )
        right_target_w, right_target_quat_w = self._target_pose_w(
            command_state.right_wrist_pose_b,
            "right",
            command_state.posture_command,
        )
        left_target_b, _ = math_utils.subtract_frame_transforms(
            robot.data.root_pos_w, robot.data.root_quat_w, left_target_w
        )
        right_target_b, _ = math_utils.subtract_frame_transforms(
            robot.data.root_pos_w, robot.data.root_quat_w, right_target_w
        )
        left_error = left_target_b - left_current[:, :3]
        right_error = right_target_b - right_current[:, :3]
        left_error = left_error.clip(-1.0, 1.0)
        right_error = right_error.clip(-1.0, 1.0)
        frame_sensor = self.env.scene[HAND_CENTER_FRAME_NAME]
        left_orientation_error = self._orientation_error_b(
            frame_sensor.data.target_pos_w[:, 0],
            frame_sensor.data.target_quat_w[:, 0],
            left_target_w,
            left_target_quat_w,
        ).clip(-torch.pi, torch.pi)
        right_orientation_error = self._orientation_error_b(
            frame_sensor.data.target_pos_w[:, 1],
            frame_sensor.data.target_quat_w[:, 1],
            right_target_w,
            right_target_quat_w,
        ).clip(-torch.pi, torch.pi)
        posture_error = self._posture_error(command_state, torso_body_id).clip(-1.0, 1.0)

        return (
            robot.data.root_ang_vel_b * 0.2,
            robot.data.projected_gravity_b,
            command_state.base_velocity,
            joint_pos_rel,
            joint_vel_rel * 0.05,
            last_low_level_action,
            self._height_scan(),
            command_state.left_wrist_pose_b,
            command_state.right_wrist_pose_b,
            left_current,
            right_current,
            left_error,
            right_error,
            command_state.posture_command,
            posture_error,
            left_orientation_error,
            right_orientation_error,
        )

    def _single_frame(self, command_state: HighLevelCommandState, last_low_level_action: torch.Tensor) -> torch.Tensor:
        obs = torch.cat(self._single_frame_terms(command_state, last_low_level_action), dim=-1)
        if self.finite_obs_clip is not None:
            obs = torch.nan_to_num(obs, nan=0.0, posinf=self.finite_obs_clip, neginf=-self.finite_obs_clip)
            obs = torch.clamp(obs, -self.finite_obs_clip, self.finite_obs_clip)
        return obs

    def _initialize_term_history(self, term_values: tuple[torch.Tensor, ...]) -> None:
        self._term_history = [
            torch.zeros(
                self.env.num_envs,
                self.history_length,
                term.shape[-1],
                device=self.device,
                dtype=term.dtype,
            )
            for term in term_values
        ]
        self._history_valid = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.bool)

    def _push_term_history(self, term_values: tuple[torch.Tensor, ...]) -> None:
        if self._term_history is None:
            self._initialize_term_history(term_values)
        assert self._term_history is not None and self._history_valid is not None
        invalid_env_ids = torch.nonzero(~self._history_valid, as_tuple=False).squeeze(-1)
        if invalid_env_ids.numel() > 0:
            for history, term in zip(self._term_history, term_values):
                history[invalid_env_ids, :] = term[invalid_env_ids].unsqueeze(1)
            self._history_valid[invalid_env_ids] = True
        for history, term in zip(self._term_history, term_values):
            history[:] = torch.roll(history, shifts=-1, dims=1)
            history[:, -1] = term

    def _flatten_term_major_history(self) -> torch.Tensor:
        assert self._term_history is not None
        term_history_values = [
            history.reshape(self.env.num_envs, self.history_length * history.shape[-1])
            for history in self._term_history
        ]
        return torch.cat(term_history_values, dim=-1)

    def build(self, command_state: HighLevelCommandState, last_low_level_action: torch.Tensor) -> torch.Tensor:
        term_values = self._single_frame_terms(command_state, last_low_level_action)
        if self.finite_obs_clip is not None:
            term_values = tuple(
                torch.clamp(
                    torch.nan_to_num(term, nan=0.0, posinf=self.finite_obs_clip, neginf=-self.finite_obs_clip),
                    -self.finite_obs_clip,
                    self.finite_obs_clip,
                )
                for term in term_values
            )
        self._push_term_history(term_values)
        return self._flatten_term_major_history()
