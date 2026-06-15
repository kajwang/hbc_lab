from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_package_and_pyproject_exist():
    assert (REPO_ROOT / "pyproject.toml").is_file()
    assert (REPO_ROOT / "source/hbc_lab/hbc_lab/__init__.py").is_file()
    assert (REPO_ROOT / "source/hbc_lab/hbc_lab/tasks/__init__.py").is_file()


def test_rsl_rl_scripts_are_hbc_local_not_robot_lab_delegates():
    train_source = (REPO_ROOT / "scripts/rsl_rl/train.py").read_text()
    play_source = (REPO_ROOT / "scripts/rsl_rl/play.py").read_text()

    assert "import hbc_lab.tasks" in train_source
    assert "import hbc_lab.tasks" in play_source
    assert "runpy.run_path" not in train_source
    assert "runpy.run_path" not in play_source
    assert "robot_lab/scripts/reinforcement_learning/rsl_rl/train.py" not in train_source
    assert "robot_lab/scripts/reinforcement_learning/rsl_rl/play.py" not in play_source
    assert 'entry_point_key="play_env_cfg_entry_point"' in play_source


def test_readme_documents_first_training_command():
    readme = (REPO_ROOT / "README.md").read_text()

    assert "HBC-Isaac-Tracking-Flat-Unitree-G1-v0" in readme
    assert "scripts/rsl_rl/train.py" in readme
    assert "--max_iterations" in readme


def test_vscode_launch_includes_g1_whole_body_task():
    launch_source = (REPO_ROOT / ".vscode/launch.json").read_text()

    assert '"name": "g1_whole_body_train"' in launch_source
    assert '"name": "g1_whole_body_play"' in launch_source
    assert "--task=HBC-Isaac-WholeBody-Unitree-G1-v0" in launch_source
