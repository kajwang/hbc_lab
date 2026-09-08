import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = REPO_ROOT / "source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc"
MODULE_PATH = TASK_ROOT / "mdp/object_shape_bps.py"
CONFIG_PATH = TASK_ROOT / "config/multishape_bps_env_cfg.py"
OBSERVATION_PATH = TASK_ROOT / "mdp/observations.py"
SCENE_PATH = TASK_ROOT / "mdp/scenes.py"
EVENT_PATH = TASK_ROOT / "mdp/events.py"
ENV_PATH = TASK_ROOT / "config/g1_dex1_env.py"
REWARD_PATH = TASK_ROOT / "mdp/rewards.py"
REGISTRATION_PATH = TASK_ROOT / "__init__.py"
AGENT_PATH = TASK_ROOT / "config/agents/rsl_rl_ppo_cfg.py"
LAUNCH_PATH = REPO_ROOT / ".vscode/launch.json"
PLAY_SCRIPT_PATH = REPO_ROOT / "scripts/rsl_rl/play.py"
GENERATOR_PATH = REPO_ROOT / "scripts/tools/generate_multishape_bps.py"
TRAY_ASSET_PATH = REPO_ROOT / "source/hbc_lab/hbc_lab/assets/models/platform/shallow_tray.usda"
VISUAL_EVAL_SPEC_PATH = (
    REPO_ROOT / "docs/superpowers/specs/2026-08-20-g1-dex1-multishape-visual-eval-design.md"
)


def _load_bps_module():
    spec = importlib.util.spec_from_file_location("g1_dex1_multishape_bps_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_directional_bps_basis_is_deterministic_and_fixed_size():
    module = _load_bps_module()

    first = module.fibonacci_ball_basis(64, radius=0.25, dtype=torch.float64)
    second = module.fibonacci_ball_basis(64, radius=0.25, dtype=torch.float64)

    assert first.shape == (64, 3)
    assert torch.equal(first, second)
    assert torch.linalg.vector_norm(first, dim=-1).max() <= 0.25 + 1.0e-9
    assert torch.unique(first, dim=0).shape[0] == 64


def test_directional_bps_selects_surface_offsets_around_geometry_center():
    module = _load_bps_module()
    surface_points = torch.tensor(
        [
            [-0.2, -0.1, -0.05],
            [-0.2, -0.1, 0.05],
            [-0.2, 0.1, -0.05],
            [-0.2, 0.1, 0.05],
            [0.2, -0.1, -0.05],
            [0.2, -0.1, 0.05],
            [0.2, 0.1, -0.05],
            [0.2, 0.1, 0.05],
        ],
        dtype=torch.float64,
    )
    basis = module.fibonacci_ball_basis(64, radius=0.25, dtype=torch.float64)

    descriptor, center, bounds_min, bounds_max = module.directional_bps_from_surface_points(
        surface_points,
        basis,
    )
    reconstructed_surface = basis + descriptor

    assert descriptor.shape == (64, 3)
    assert torch.allclose(center, torch.zeros(3, dtype=torch.float64))
    assert torch.allclose(bounds_min, torch.tensor([-0.2, -0.1, -0.05], dtype=torch.float64))
    assert torch.allclose(bounds_max, torch.tensor([0.2, 0.1, 0.05], dtype=torch.float64))
    assert all(
        torch.any(torch.all(torch.isclose(surface_points, point, atol=1.0e-9), dim=-1))
        for point in reconstructed_surface
    )


def test_directional_bps_transforms_object_local_vectors_into_robot_root_frame():
    module = _load_bps_module()
    offsets_o = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    object_quat_w = torch.tensor([[2.0**-0.5, 0.0, 0.0, 2.0**-0.5]])
    robot_root_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    offsets_b = module.directional_bps_in_robot_root_frame(
        offsets_o,
        object_quat_w,
        robot_root_quat_w,
    )

    expected = torch.tensor([[[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]])
    assert torch.allclose(offsets_b, expected, atol=1.0e-5)


def test_grasp_frame_bps_encodes_surface_relative_to_selected_contact_pose():
    module = _load_bps_module()
    basis = torch.zeros(2, 3)
    centered_surface = torch.tensor([[[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]]])
    geometry_center = torch.tensor([[1.0, 0.0, 0.0]])
    grasp_position = torch.tensor([[1.0, 0.0, 0.0]])
    quarter_turn_z = torch.tensor([[2.0**-0.5, 0.0, 0.0, 2.0**-0.5]])

    descriptor = module.directional_bps_in_grasp_frame(
        centered_surface,
        basis,
        geometry_center,
        grasp_position,
        quarter_turn_z,
        torch.ones(1),
    )

    expected = torch.tensor([[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
    assert torch.allclose(descriptor, expected, atol=1.0e-5)


def test_no_bps_ablation_preserves_descriptor_shape_and_returns_exact_zeros():
    module = _load_bps_module()
    descriptor = torch.randn(3, 64, 3)

    enabled = module.apply_directional_bps_ablation(descriptor, enabled=True)
    disabled = module.apply_directional_bps_ablation(descriptor, enabled=False)

    assert torch.equal(enabled, descriptor)
    assert disabled.shape == descriptor.shape
    assert torch.count_nonzero(disabled) == 0


def test_shape_index_is_decoded_from_spawn_mass_metadata():
    module = _load_bps_module()
    default_mass = torch.tensor([[10.0], [10.125], [10.250], [10.875]])

    indices = module.decode_shape_index_from_spawn_mass(default_mass, base_mass=10.0, stride=0.125)

    assert torch.equal(indices, torch.tensor([0, 1, 2, 7]))
    with pytest.raises(ValueError, match="shape index metadata"):
        module.decode_shape_index_from_spawn_mass(torch.tensor([[10.06]]), base_mass=10.0, stride=0.125)


def test_shape_scale_variant_metadata_decodes_balanced_cross_product():
    module = _load_bps_module()
    default_mass = torch.tensor([[10.0], [10.125], [10.5], [10.625], [11.125]])

    shape_indices, scale_indices = module.decode_shape_scale_variant_indices(
        default_mass,
        num_scale_factors=5,
    )

    assert torch.equal(shape_indices, torch.tensor([0, 0, 0, 1, 1]))
    assert torch.equal(scale_indices, torch.tensor([0, 1, 4, 0, 4]))


def test_uniform_size_factor_scales_reconstructed_bps_surface_not_raw_offset():
    module = _load_bps_module()
    basis = torch.tensor([[0.10, 0.00, 0.00], [0.00, 0.10, 0.00]])
    offsets = torch.tensor(
        [
            [[0.05, 0.02, 0.00], [0.01, 0.04, 0.00]],
            [[0.05, 0.02, 0.00], [0.01, 0.04, 0.00]],
        ]
    )
    scale = torch.tensor([0.8, 1.2])

    scaled_offsets = module.scale_directional_bps_offsets(offsets, basis, scale)
    reconstructed = basis.unsqueeze(0) + scaled_offsets
    expected = (basis.unsqueeze(0) + offsets) * scale[:, None, None]

    assert torch.allclose(reconstructed, expected)


def test_rotated_geometry_bounds_use_stable_asset_orientation_for_floor_placement():
    module = _load_bps_module()
    bounds_min = torch.tensor([[-0.02, 0.00, -0.04]])
    bounds_max = torch.tensor([[0.02, 0.08, 0.04]])
    quarter_turn_x = torch.tensor([[2.0**-0.5, 2.0**-0.5, 0.0, 0.0]])

    lower_z = module.rotated_box_min_z(bounds_min, bounds_max, quarter_turn_x)

    assert torch.allclose(lower_z, torch.tensor([0.0]), atol=1.0e-5)


def test_multishape_asset_split_and_scales_match_the_requested_experiment():
    module = _load_bps_module()

    assert module.TRAIN_SHAPE_NAMES == (
        "egg",
        "milk",
        "ketchup",
        "butter",
        "hotdog",
        "shaker",
        "cheese",
        "cucumber",
        "wine",
        "coke",
        "soup_can",
        "mango",
        "pear",
        "candle",
        "sneaker",
        "soap_dispenser",
        "sponge",
        "toy_car",
        "toy_ship",
        "orange_peel",
        "rotten_apple",
        "rotten_banana",
        "trash_can",
        "bowl",
    )
    assert module.HELD_OUT_SHAPE_NAMES == ("donut", "bread_bag", "toy_gun")
    assert module.MULTISHAPE_SIZE_FACTORS == pytest.approx((0.8, 0.9, 1.0, 1.1, 1.2))
    assert module.OBJECT_SHAPE_SPECS["egg"].scale == pytest.approx((0.07, 0.07, 0.07))
    assert module.OBJECT_SHAPE_SPECS["wine"].scale == pytest.approx((0.2, 0.2, 0.2))
    assert module.OBJECT_SHAPE_SPECS["sneaker"].scale == pytest.approx((0.005, 0.005, 0.005))
    assert module.OBJECT_SHAPE_SPECS["soap_dispenser"].scale == pytest.approx((0.004, 0.004, 0.005))
    assert module.OBJECT_SHAPE_SPECS["bowl"].scale == pytest.approx((0.0025, 0.0025, 0.0035))
    assert module.OBJECT_SHAPE_SPECS["orange_peel"].scale == pytest.approx((0.8, 0.8, 0.8))
    assert module.OBJECT_SHAPE_SPECS["rotten_apple"].scale == pytest.approx((0.8, 0.8, 0.8))
    assert module.OBJECT_SHAPE_SPECS["rotten_banana"].scale == pytest.approx((0.8, 0.8, 0.8))
    assert module.OBJECT_SHAPE_SPECS["coke"].stable_quat_wxyz == pytest.approx((1.0, 0.0, 0.0, 0.0))
    assert module.OBJECT_SHAPE_SPECS["toy_car"].stable_quat_wxyz == pytest.approx(
        (2.0**-0.5, 2.0**-0.5, 0.0, 0.0)
    )
    assert module.OBJECT_SHAPE_SPECS["bread_bag"].scale == pytest.approx((0.003, 0.003, 0.003))
    assert module.OBJECT_SHAPE_SPECS["toy_gun"].scale == pytest.approx((0.7, 0.7, 0.7))
    assert all(spec.usd_path.is_file() for spec in module.OBJECT_SHAPE_SPECS.values())


def test_visual_eval_formats_one_deterministic_environment_per_shape():
    module = _load_bps_module()

    lines = module.format_shape_assignment(module.ALL_SHAPE_NAMES).splitlines()

    assert lines[0] == "[MultiShape Visual Eval] deterministic asset assignment:"
    assert lines[1:9] == [
        "  env 00: egg (train)",
        "  env 01: milk (train)",
        "  env 02: ketchup (train)",
        "  env 03: butter (train)",
        "  env 04: hotdog (train)",
        "  env 05: shaker (train)",
        "  env 06: cheese (train)",
        "  env 07: cucumber (train)",
    ]
    assert len(lines) == 1 + len(module.ALL_SHAPE_NAMES) == 28
    assert lines[-3:] == [
        "  env 24: donut (OOD)",
        "  env 25: bread_bag (OOD)",
        "  env 26: toy_gun (OOD)",
    ]


def test_multishape_visual_play_uses_exactly_one_environment_per_shape():
    config_source = CONFIG_PATH.read_text()
    play_source = config_source.split("class G1Dex1HierDrcMultiShapePlayEnvCfg", maxsplit=1)[1]

    assert "num_envs=27" in play_source
    assert "env_spacing=6.0" in play_source
    assert "self.scene.num_envs = len(ALL_SHAPE_NAMES)" in play_source
    assert "format_shape_assignment(ALL_SHAPE_NAMES)" in play_source
    assert "@configclass\n@configclass" not in config_source


def test_multishape_task_spawns_one_asset_and_keeps_shape_out_of_history():
    config_source = CONFIG_PATH.read_text()
    observation_source = OBSERVATION_PATH.read_text()
    scene_source = SCENE_PATH.read_text()
    event_source = EVENT_PATH.read_text()

    assert "G1Dex1HierDrcMultiShapeBpsSceneCfg" in scene_source
    assert "MultiAssetSpawnerCfg" in scene_source
    assert "RigidObjectCollectionCfg" not in config_source
    assert "replicate_physics=False" in config_source
    assert "reset_multishape_object_and_support_platforms" in event_source
    assert "directional_object_bps_obs" in observation_source
    assert "grasp_frame_object_bps_obs" in observation_source
    graspref_source = observation_source.split(
        "class G1Dex1HierDrcMultiShapeGraspRefObservationsCfg", maxsplit=1
    )[1]
    assert "object_shape = ObsTerm(func=grasp_frame_object_bps_obs" in graspref_source
    assert "class G1Dex1HierDrcMultiShapeObservationsCfg" in observation_source
    policy_source = observation_source.split(
        "class G1Dex1HierDrcMultiShapeObservationsCfg", maxsplit=1
    )[1]
    assert "object_shape = ObsTerm" in policy_source
    assert "history_length=0" in policy_source
    assert "self.history_length = None" in policy_source
    assert "term.history_length = 10" in observation_source


def test_multishape_task_disables_nonshape_randomization_and_registers_one_parameterized_task():
    config_source = CONFIG_PATH.read_text()
    registration_source = REGISTRATION_PATH.read_text()
    agent_source = AGENT_PATH.read_text()
    launch_source = LAUNCH_PATH.read_text()

    assert "num_envs=4096" in config_source
    assert "domain_randomization_curriculum_enabled = False" in config_source
    assert "object_mass_curriculum_enabled = True" in config_source
    assert "object_shape_bps_enabled: bool = True" in config_source
    assert "def apply_object_shape_encoding" in config_source
    assert "HBC-Isaac-G1-Dex1-HierDrc-MultiShape-v0" in registration_source
    assert "HBC-Isaac-G1-Dex1-HierDrc-MultiShape-Play-v0" in registration_source
    assert "MultiShape-NoBps" not in registration_source
    assert "G1Dex1HierDrcMultiShapePPORunnerCfg" in agent_source
    assert "g1_dex1_hier_drc_multishape" in agent_source
    assert "g1_dex1_multishape_train" in launch_source
    assert "env.object_shape_bps_enabled=true" in launch_source
    assert "env.object_shape_bps_enabled=false" in launch_source
    assert '"--num_envs=4096"' in launch_source


def test_multishape_training_crosses_shapes_with_five_spawn_time_size_factors():
    scene_source = SCENE_PATH.read_text()
    config_source = CONFIG_PATH.read_text()
    env_source = ENV_PATH.read_text()

    assert "MULTISHAPE_SIZE_FACTORS" in scene_source
    assert "for size_factor in size_factors" in scene_source
    assert "object_size_scale_factors" in config_source
    assert "decode_shape_scale_variant_indices" in env_source
    assert "self.object_size_scale.copy_" in env_source


def test_multishape_support_uses_one_compound_shallow_tray_actor():
    scene_source = SCENE_PATH.read_text()
    generator_source = GENERATOR_PATH.read_text()

    assert TRAY_ASSET_PATH.is_file()
    tray_source = TRAY_ASSET_PATH.read_text()
    assert 'def Xform "ShallowTray"' in tray_source
    assert tray_source.count('prepend apiSchemas = ["PhysicsCollisionAPI"]') == 5
    assert 'def Cube "Pedestal"' in tray_source
    assert all(f'def Cube "Wall{name}"' in tray_source for name in ("Front", "Back", "Left", "Right"))
    assert "SHALLOW_TRAY_WALL_HEIGHT = 0.05" in scene_source
    assert "(0.34, 0.025, 0.05)" in tray_source
    assert "ComputeRelativeTransform(prim, default_prim)" in generator_source
    assert "SHALLOW_TRAY_USD_PATH" in scene_source
    assert "_multishape_tray_cfg" in scene_source


def test_multishape_logs_only_per_shape_stable_grasp_rate_and_compact_global_metrics():
    env_source = ENV_PATH.read_text()
    reward_source = REWARD_PATH.read_text()

    assert 'f"Shape/{shape_name}_grasp_rate"' in env_source
    assert "step_c_physical_grasp = self.c_physical_grasp.clone()" in env_source
    assert "self._log_shape_diagnostics(step_c_physical_grasp)" in env_source
    assert 'f"Shape/{shape_name}_count"' not in env_source
    assert 'f"Shape/{shape_name}_W_manip"' not in env_source
    assert 'f"Shape/{shape_name}_fall"' not in env_source
    assert '"ContactLink/' not in env_source
    assert '"Couple/' not in env_source
    assert '"DRC/object_mass_min"' not in env_source
    assert '"DRC/object_mass_max"' not in env_source
    assert '"Motion/quality_loss_mean"' in reward_source
    assert '"hand_position_tracking",' in reward_source
    assert '"hand_orientation_tracking",' in reward_source
    assert '"Motion/command_rate_norm"' not in reward_source


def test_multishape_shape_encoding_toggle_zero_fills_without_changing_observation_terms():
    config_source = CONFIG_PATH.read_text()
    observation_source = OBSERVATION_PATH.read_text()
    env_source = (TASK_ROOT / "config/g1_dex1_env.py").read_text()

    assert "self.observations.policy.object_shape = None" not in config_source
    assert "self.observations.critic.object_shape = None" not in config_source
    assert "apply_directional_bps_ablation" in observation_source
    assert "enabled=env.cfg.object_shape_bps_enabled" in observation_source
    assert "cfg.apply_object_shape_encoding()" in env_source
    assert env_source.index("cfg.apply_object_shape_encoding()") < env_source.index("super().__init__(cfg, render_mode, **kwargs)")


def test_multishape_visual_launches_compare_matched_bps_and_no_bps_policies():
    launch_source = LAUNCH_PATH.read_text()
    play_source = PLAY_SCRIPT_PATH.read_text()
    spec_source = VISUAL_EVAL_SPEC_PATH.read_text()

    bps_launch = launch_source.split('"name": "g1_dex1_multishape_bps_visual_play"', maxsplit=1)[1]
    bps_launch = bps_launch.split('"name":', maxsplit=1)[0]
    no_bps_launch = launch_source.split('"name": "g1_dex1_multishape_no_bps_visual_play"', maxsplit=1)[1]
    no_bps_launch = no_bps_launch.split('"name":', maxsplit=1)[0]

    shared_args = (
        '"--task=HBC-Isaac-G1-Dex1-HierDrc-MultiShape-Play-v0"',
        '"--num_envs=27"',
        '"--seed=42"',
        '"env.scene.env_spacing=6.0"',
        '"env.commands.high_level.left_hand_probability=0.0"',
        '"env.enable_debug_visualization=false"',
        '"env.platform_pose_debug_vis=true"',
        '"env.low_level_policy_path=logs/exported_policies/g1_dex1_handcenter_stickfigure_m49999.pt"',
    )
    assert all(argument in bps_launch for argument in shared_args)
    assert all(argument in no_bps_launch for argument in shared_args)
    assert '"--checkpoint=logs/5090/g1_dex1/multishape/bps/model_10500.pt"' in bps_launch
    assert '"env.object_shape_bps_enabled=true"' in bps_launch
    assert '"--checkpoint=logs/5090/g1_dex1/multishape/no_bps/model_10700.pt"' in no_bps_launch
    assert '"env.object_shape_bps_enabled=false"' in no_bps_launch
    assert 'parser.add_argument("--seed"' in play_source
    assert "env_cfg.seed = agent_cfg.seed" in play_source
    assert "env.unwrapped.seed(agent_cfg.seed)" in play_source
    assert "model_11300.pt" in spec_source
    assert "model_10500.pt" not in spec_source
