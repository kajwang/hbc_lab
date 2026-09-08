from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REWARD_PATH = (
    REPO_ROOT
    / "source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/mdp/rewards.py"
)


def _reward_source() -> str:
    return REWARD_PATH.read_text()


def test_dex1_couple_reward_uses_grasp_window_and_inner_pad_contact_gates():
    reward_source = _reward_source()

    assert "def active_inner_pad_contact" in reward_source
    assert "left_link1_3 = env._step_link_contact[\"left_Link1_3\"]" in reward_source
    assert "left_link2_3 = env._step_link_contact[\"left_Link2_3\"]" in reward_source
    assert "near = 1.0 - torch.tanh(env.d_active_hand / 0.35)" in reward_source
    assert "grasp_window = 1.0 - torch.tanh(env.d_active_hand / 0.1)" in reward_source
    assert "pad_gate = torch.clamp(inner_pad_contact / INNER_PAD_CONTACT_GATE_SCALE, min=0.0, max=1.0)" in reward_source
    assert "pad_gated_gripper_close = gripper_close * pad_gate" in reward_source
    assert "pad_early_close_penalty = gripper_close * (1.0 - pad_gate)" in reward_source
    assert "close_ready_gate = torch.maximum(grasp_window, pad_gate)" in reward_source
    assert "env.active_orientation_gate" not in reward_source
    assert "gated_gripper_close = gripper_close * close_ready_gate" in reward_source
    assert "early_close_penalty = gripper_close * (1.0 - close_ready_gate)" in reward_source


def test_dex1_couple_reward_restores_pre_approach_only_weights():
    reward_source = _reward_source()

    assert "0.4 * near" in reward_source
    assert "+ 0.1 * pad_gate" in reward_source
    assert "+ 0.1 * env.c_pinch" in reward_source
    assert "+ 0.2 * env.c_grasp" in reward_source
    assert "+ 0.2 * gated_gripper_close" in reward_source
    assert "- 0.2 * early_close_penalty" in reward_source
    assert "hand_center_approach_terms" not in reward_source
    assert "active_grip_penalty" not in reward_source


def test_dex1_pose_guidance_is_independent_and_released_only_by_stable_physical_grasp():
    reward_source = _reward_source()

    assert "def grasp_pose_guidance_reward" in reward_source
    assert "stable_grasp_released_pose_guidance" in reward_source
    assert "env.d_active_hand" in reward_source
    assert "env.active_orientation_error" in reward_source
    assert "torch.clamp(env.c_physical_grasp.detach()" in reward_source
    assert "grasp_pose_position_reward_weight * position_guidance" in reward_source
    assert "grasp_pose_orientation_reward_weight * orientation_guidance" in reward_source
    assert "r_pose_guidance = grasp_pose_guidance_reward(env)" in reward_source
    assert "+ env.W_couple * r_pose_guidance" in reward_source
    assert 'env.extras["log"]["GraspPose/couple_weighted_guidance_mean"]' in reward_source
    assert 'env.extras["log"]["GraspPose/position_score_mean"]' in reward_source
    assert 'env.extras["log"]["GraspPose/orientation_score_mean"]' in reward_source
    assert 'env.extras["log"]["GraspPose/stable_release_gate_mean"]' in reward_source
    assert 'env.extras["log"]["GraspPose/guidance_reward_mean"]' in reward_source


def test_dex1_root_object_facing_reward_uses_front_half_plane_alignment():
    reward_source = _reward_source()

    assert "def root_object_facing_reward" in reward_source
    assert "object_pos_w = env._object_frame_pos_w()" in reward_source
    assert "root_to_object_xy = object_pos_w[:, :2] - robot.data.root_pos_w[:, :2]" in reward_source
    assert "math_utils.yaw_quat(robot.data.root_quat_w)" in reward_source
    assert "facing_cos = torch.sum(forward_dir * root_to_object_dir, dim=-1)" in reward_source
    assert "reward = torch.square(torch.clamp(facing_cos, min=0.0, max=1.0))" in reward_source
    assert "env.extras[\"log\"][\"DRC/root_object_facing_mean\"] = reward.mean()" in reward_source
    assert "root_object_facing = RewTerm(func=root_object_facing_reward, weight=2.0)" in reward_source


def test_dex1_manip_reward_provides_linear_couple_incentive():
    reward_source = _reward_source()

    assert "def manip_reward" in reward_source
    assert "return 0.7 * progress + 0.3" in reward_source
    assert "return 0.7 * progress + 0.3 * env.c_couple" not in reward_source


def test_dex1_uses_shared_contact_conditioned_motion_quality_reward():
    reward_source = _reward_source()

    assert "def contact_conditioned_motion_reward" in reward_source
    assert "motion_quality = RewTerm(" in reward_source
    assert "env.contact_label.effector_mask" in reward_source
    assert "env.W_app" in reward_source
    assert "env.W_couple" in reward_source
    assert "env.W_manip" in reward_source
    assert "motion_keyframe_index" not in reward_source
    assert "contact_mode" not in reward_source
    assert "both_hand_center_tracking_error_penalty = RewTerm" not in reward_source
    assert "command_smoothness = RewTerm" not in reward_source
