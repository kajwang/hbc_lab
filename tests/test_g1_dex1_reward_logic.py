from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REWARD_PATH = (
    REPO_ROOT
    / "source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/mdp/rewards.py"
)


def _reward_source() -> str:
    return REWARD_PATH.read_text()


def test_dex1_couple_reward_is_approach_only_without_contact_or_close_gates():
    reward_source = _reward_source()

    assert "def hand_center_approach_terms" in reward_source
    assert "near_broad = torch.exp(-d / 0.45)" in reward_source
    assert "near_mid = 1.0 - torch.tanh(d / 0.20)" in reward_source
    assert "near_fine = 1.0 - torch.tanh(d / 0.06)" in reward_source
    assert "active_grip_penalty = env.active_grip" in reward_source
    assert "pad_gate" not in reward_source
    assert "gated_gripper_close" not in reward_source
    assert "early_close_penalty" not in reward_source


def test_dex1_couple_reward_prioritizes_hand_center_distance_and_penalizes_active_grip():
    reward_source = _reward_source()

    assert "0.50 * near_broad" in reward_source
    assert "1.00 * near_mid" in reward_source
    assert "1.50 * near_fine" in reward_source
    assert "- 0.40 * active_grip_penalty" in reward_source
    assert "+ 0.20 * env.c_grasp" not in reward_source
