from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REWARD_PATH = (
    REPO_ROOT
    / "source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/mdp/rewards.py"
)


def _reward_source() -> str:
    return REWARD_PATH.read_text()


def test_dex1_couple_reward_uses_single_inner_pad_contact_gate():
    reward_source = _reward_source()

    assert "def active_inner_pad_contact" in reward_source
    assert '"left_Link1_2"' in reward_source
    assert '"left_Link2_2"' in reward_source
    assert '"right_Link1_2"' in reward_source
    assert '"right_Link2_2"' in reward_source
    assert "torch.maximum(active_link1_2, active_link2_2)" in reward_source
    assert "pad_gate = torch.clamp(inner_pad_contact / INNER_PAD_CONTACT_GATE_SCALE" in reward_source
    assert "close_gate = torch.maximum(grasp_window, pad_gate)" in reward_source


def test_dex1_couple_reward_encourages_close_after_inner_pad_touch():
    reward_source = _reward_source()

    assert "0.35 * pad_gate" in reward_source
    assert "0.30 * gated_gripper_close" in reward_source
    assert "- 0.05 * early_close_penalty" in reward_source
    assert "gated_gripper_close = gripper_close * close_gate" in reward_source
    assert "early_close_penalty = gripper_close * (1.0 - close_gate)" in reward_source
