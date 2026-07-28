from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = (
    ROOT
    / "source"
    / "hbc_lab"
    / "hbc_lab"
    / "tasks"
    / "manager_based"
    / "skill"
)


def test_door_registration_and_agent_config():
    task_source = (SKILL_ROOT / "g1_dex1_door_open_hier_drc" / "__init__.py").read_text()
    agent_source = (
        SKILL_ROOT
        / "g1_dex1_door_open_hier_drc"
        / "config"
        / "agents"
        / "rsl_rl_ppo_cfg.py"
    ).read_text()
    assert "HBC-Isaac-G1-Dex1-DoorOpen-HierDrc-v0" in task_source
    assert "HBC-Isaac-G1-Dex1-DoorOpen-HierDrc-Play-v0" in task_source
    assert "play_env_cfg_entry_point" in task_source
    assert 'experiment_name = "g1_dex1_door_open_hier_drc"' in agent_source


def test_cart_registration_and_agent_config():
    task_source = (SKILL_ROOT / "g1_dex1_cart_push_hier_drc" / "__init__.py").read_text()
    agent_source = (
        SKILL_ROOT
        / "g1_dex1_cart_push_hier_drc"
        / "config"
        / "agents"
        / "rsl_rl_ppo_cfg.py"
    ).read_text()
    assert "HBC-Isaac-G1-Dex1-CartPush-HierDrc-v0" in task_source
    assert "HBC-Isaac-G1-Dex1-CartPush-HierDrc-Play-v0" in task_source
    assert "play_env_cfg_entry_point" in task_source
    assert 'experiment_name = "g1_dex1_cart_push_hier_drc"' in agent_source


def test_task_entry_point_imports_both_articulated_tasks():
    source = (
        ROOT / "source" / "hbc_lab" / "hbc_lab" / "tasks" / "__init__.py"
    ).read_text()
    assert "g1_dex1_door_open_hier_drc" in source
    assert "g1_dex1_cart_push_hier_drc" in source


def test_launch_json_contains_train_and_play_entries():
    source = (ROOT / ".vscode" / "launch.json").read_text()
    for name in (
        "g1_dex1_door_open_hier_drc_train",
        "g1_dex1_door_open_hier_drc_play",
        "g1_dex1_cart_push_hier_drc_train",
        "g1_dex1_cart_push_hier_drc_play",
    ):
        assert f'"name": "{name}"' in source
