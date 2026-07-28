import hashlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS_ROOT = REPO_ROOT / "source/hbc_lab/hbc_lab/assets"
ARTICULATED_ROOT = ASSETS_ROOT / "models/articulated"
CONFIG_PATH = ASSETS_ROOT / "articulated_objects.py"
DOOR_USD = ARTICULATED_ROOT / "door/door_0_bot.usd"
CART_USD = ARTICULATED_ROOT / "cart/instance_turn_R_10kg.usd"


def _compact(source: str) -> str:
    return "".join(source.split())


def _config_source() -> str:
    assert CONFIG_PATH.is_file(), f"Missing articulated asset configuration: {CONFIG_PATH}"
    return _compact(CONFIG_PATH.read_text())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_articulated_usd_assets_are_exact_local_copies():
    assert DOOR_USD.is_file(), f"Missing door USD: {DOOR_USD}"
    assert CART_USD.is_file(), f"Missing cart USD: {CART_USD}"
    assert sorted(path.relative_to(ARTICULATED_ROOT).as_posix() for path in ARTICULATED_ROOT.rglob("*.usd")) == [
        "cart/instance_turn_R_10kg.usd",
        "door/door_0_bot.usd",
    ]
    assert _sha256(DOOR_USD) == "cb8a34e14f5ce98adb005e1f414e5d1b8bf3b8dabc55e8d711810ff31e1f8c12"
    assert _sha256(CART_USD) == "9a803b0992815314f36d822b35dce3eceeb82f14eb772ec50593a1290e587b97"


def test_articulated_configs_use_repo_relative_paths_and_are_packaged():
    source = _config_source()
    pyproject_source = (REPO_ROOT / "pyproject.toml").read_text()

    assert "ARTICULATED_MODEL_DIR=Path(__file__).resolve().parent/\"models\"/\"articulated\"" in source
    assert "usd_path=str(ARTICULATED_MODEL_DIR/\"door\"/\"door_0_bot.usd\")" in source
    assert "usd_path=str(ARTICULATED_MODEL_DIR/\"cart\"/\"instance_turn_R_10kg.usd\")" in source
    assert "assets/models/articulated/**/*.usd" in pyproject_source
    assert "/home/kaijun/" not in source


def test_door_articulation_and_frames_preserve_reference_parameters():
    source = _config_source()

    assert "DOOR_CFG=ArticulationCfg(" in source
    assert "rot=(0.0,0.0,0.0,1.0)" in source
    assert '"joint_1":0.0' in source
    assert '"joint_2":0.0' in source
    assert 'joint_names_expr=["joint_1"],stiffness=0.0,damping=5.0,friction=0.2' in source
    assert 'joint_names_expr=["joint_2"],stiffness=1.0,damping=1.0' in source
    assert 'prim_path="{ENV_REGEX_NS}/door/link_2",name="door_handle"' in source
    assert "pos=(-0.04,0.0,0.04)" in source
    assert 'prim_path="{ENV_REGEX_NS}/door/link_0",name="door_handle_goal"' in source
    assert "pos=(-0.31822,-0.26341,-0.5)" in source


def test_cart_articulation_and_handle_frame_preserve_reference_parameters():
    source = _config_source()

    assert "CART_CFG=ArticulationCfg(" in source
    assert "scale=(0.8,0.8,0.8)" in source
    assert "pos=(0.0,0.0,0.15)" in source
    assert "rot=(0.5,0.5,-0.5,-0.5)" in source
    for joint_name in ("RL_joint", "RR_joint", "FL_joint", "FR_joint", "RL_turn_joint", "RR_turn_joint"):
        assert f'"{joint_name}":0.0' in source
    assert (
        'joint_names_expr=["RL_joint","RR_joint","FL_joint","FR_joint"],'
        "effort_limit=0.0,velocity_limit=100.0,stiffness=0.0,damping=0.1,friction=0.0"
    ) in source
    assert (
        'joint_names_expr=["RL_turn_joint","RR_turn_joint"],'
        "effort_limit=0.0,velocity_limit=50.0,stiffness=0.0,damping=1.0,friction=0.2"
    ) in source
    assert 'prim_path="{ENV_REGEX_NS}/cart/handle",name="cart_handle"' in source
    assert "pos=(0.0,0.0,0.0)" in source
    assert "rot=(0.707,0.0,0.707,0.0)" in source
