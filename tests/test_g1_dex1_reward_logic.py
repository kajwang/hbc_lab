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
    assert "gated_gripper_close1 = gripper_close * pad_gate" in reward_source
    assert "early_close_penalty1 = gripper_close * (1.0 - pad_gate)" in reward_source
    assert "gated_gripper_close2 = gripper_close * grasp_window" in reward_source
    assert "early_close_penalty2 = gripper_close * (1.0 - grasp_window)" in reward_source


def test_dex1_couple_reward_restores_pre_approach_only_weights():
    reward_source = _reward_source()

    assert "1.5 * near" in reward_source
    assert "+ 0.50 * pad_gate" in reward_source
    assert "+ 0.20 * env.c_grasp" in reward_source
    assert "+ 0.50 * gated_gripper_close1" in reward_source
    assert "- 0.50 * early_close_penalty1" in reward_source
    assert "+ 0.50 * gated_gripper_close2" in reward_source
    assert "- 0.50 * early_close_penalty2" in reward_source
    assert "hand_center_approach_terms" not in reward_source
    assert "active_grip_penalty" not in reward_source
