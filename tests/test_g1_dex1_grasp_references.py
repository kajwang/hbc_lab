import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "source/hbc_lab"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
TASK_ROOT = REPO_ROOT / "source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc"
MODULE_PATH = TASK_ROOT / "mdp/grasp_references.py"
CONTACT_PROGRESS_PATH = TASK_ROOT / "mdp/contact_progress.py"
OBJECT_SHAPE_MODULE_PATH = TASK_ROOT / "mdp/object_shape_bps.py"
SCENE_PATH = TASK_ROOT / "mdp/scenes.py"
CALIBRATOR_PATH = REPO_ROOT / "scripts/tools/calibrate_multishape_stable_poses.py"
EXPORTER_PATH = REPO_ROOT / "scripts/tools/export_multishape_grasp_meshes.py"
GENERATOR_PATH = REPO_ROOT / "scripts/tools/generate_graspgenx_references.py"
VISUALIZER_PATH = REPO_ROOT / "scripts/tools/visualize_graspgenx_references.py"
CANDIDATE_PLAY_PATH = REPO_ROOT / "scripts/rsl_rl/play_grasp_candidates.py"
PLAY_PATH = REPO_ROOT / "scripts/rsl_rl/play.py"
ENV_PATH = TASK_ROOT / "config/g1_dex1_env.py"
ENV_CFG_PATH = TASK_ROOT / "config/g1_dex1_env_cfg.py"
MULTISHAPE_CFG_PATH = TASK_ROOT / "config/multishape_bps_env_cfg.py"
OBSERVATION_PATH = TASK_ROOT / "mdp/observations.py"
EVENT_PATH = TASK_ROOT / "mdp/events.py"
REGISTRATION_PATH = TASK_ROOT / "__init__.py"
LAUNCH_PATH = REPO_ROOT / ".vscode/launch.json"


def _load_module():
    spec = importlib.util.spec_from_file_location("g1_dex1_grasp_references_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _transform(position, rotation=None):
    transform = torch.eye(4)
    transform[:3, 3] = torch.as_tensor(position)
    if rotation is not None:
        transform[:3, :3] = torch.as_tensor(rotation)
    return transform


def test_transform_validation_rejects_non_rigid_and_nonfinite_candidates():
    module = _load_module()
    transforms = torch.stack((_transform((0.1, 0.0, 0.0)), _transform((0.2, 0.0, 0.0))))
    transforms[1, 0, 0] = float("nan")

    valid = module.valid_rigid_transforms(transforms)

    assert torch.equal(valid, torch.tensor([True, False]))


def test_matrix_to_quaternion_uses_scalar_first_convention():
    module = _load_module()
    rotation_z_90 = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )

    quat = module.matrix_to_quaternion_wxyz(rotation_z_90.unsqueeze(0))[0]

    expected = torch.tensor([2.0**-0.5, 0.0, 0.0, 2.0**-0.5], dtype=torch.float64)
    assert torch.allclose(quat.abs(), expected, atol=1.0e-6)


def test_object_local_grasp_pose_is_updated_from_live_object_pose():
    module = _load_module()
    object_pos_w = torch.tensor([[1.0, 2.0, 0.5]])
    object_quat_w = torch.tensor([[2.0**-0.5, 0.0, 0.0, 2.0**-0.5]])
    grasp_pos_o = torch.tensor([[0.1, 0.0, 0.0]])
    grasp_quat_o = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    grasp_pos_w, grasp_quat_w = module.compose_object_grasp_pose(
        object_pos_w,
        object_quat_w,
        grasp_pos_o,
        grasp_quat_o,
    )

    assert torch.allclose(grasp_pos_w, torch.tensor([[1.0, 2.1, 0.5]]), atol=1.0e-6)
    assert torch.allclose(grasp_quat_w.abs(), object_quat_w.abs(), atol=1.0e-6)


def test_approach_mode_classification_uses_grasp_local_positive_z_axis():
    module = _load_module()
    top = torch.eye(3)
    side = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    angle = torch.deg2rad(torch.tensor(45.0))
    oblique = torch.tensor(
        [[torch.cos(angle), 0.0, torch.sin(angle)], [0.0, 1.0, 0.0], [-torch.sin(angle), 0.0, torch.cos(angle)]]
    )

    modes = module.classify_approach_modes(
        torch.stack((top, side, oblique)),
        support_up_o=torch.tensor([0.0, 0.0, 1.0]),
    )

    assert torch.equal(
        modes,
        torch.tensor([module.GRASP_MODE_TOP, module.GRASP_MODE_SIDE, module.GRASP_MODE_OBLIQUE]),
    )


def test_ergonomic_wrist_filter_rejects_inverted_side_grasps_but_keeps_top_grasps():
    module = _load_module()
    upright_side = torch.eye(3)
    inverted_side = torch.diag(torch.tensor([-1.0, 1.0, -1.0]))
    top_down = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    )

    valid = module.ergonomic_wrist_mask(
        torch.stack((upright_side, inverted_side, top_down)),
        support_up_o=torch.tensor([0.0, 0.0, 1.0]),
        max_roll_rad=torch.deg2rad(torch.tensor(75.0)),
    )

    assert torch.equal(valid, torch.tensor([True, False, True]))


def test_candidate_evaluation_yaws_cover_twelve_evenly_spaced_headings():
    module = _load_module()

    yaw = module.candidate_evaluation_yaws(torch.tensor([0, 1, 11, 12]), yaw_count=12)

    expected = torch.tensor([0.0, torch.pi / 6.0, 11.0 * torch.pi / 6.0, 0.0])
    assert torch.allclose(yaw, expected)


def test_candidate_selector_can_choose_highest_runtime_weight_deterministically():
    module = _load_module()
    weights = torch.tensor([[0.1, 0.7, 0.2], [2.0, 1.0, 0.0]])

    selected = module.select_candidate_indices(weights, deterministic=True)

    assert selected.tolist() == [1, 0]


def test_se3_nms_removes_only_nearby_candidates_with_similar_orientation():
    module = _load_module()
    rotation_z_90 = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    transforms = torch.stack(
        (
            _transform((0.0, 0.0, 0.1)),
            _transform((0.005, 0.0, 0.1)),
            _transform((0.005, 0.0, 0.1), rotation_z_90),
            _transform((0.08, 0.0, 0.1)),
        )
    )
    scores = torch.tensor([0.9, 0.8, 0.7, 0.6])

    indices = module.se3_nms_indices(
        transforms,
        scores,
        translation_threshold=0.02,
        rotation_threshold_rad=0.35,
    )

    assert indices.tolist() == [0, 2, 3]


def test_balanced_selection_retains_top_side_and_oblique_candidates():
    module = _load_module()
    transforms = torch.stack([_transform((0.04 * index, 0.0, 0.1)) for index in range(9)])
    scores = torch.tensor([0.99, 0.98, 0.97, 0.80, 0.79, 0.78, 0.60, 0.59, 0.58])
    modes = torch.tensor(
        [
            module.GRASP_MODE_TOP,
            module.GRASP_MODE_TOP,
            module.GRASP_MODE_TOP,
            module.GRASP_MODE_SIDE,
            module.GRASP_MODE_SIDE,
            module.GRASP_MODE_SIDE,
            module.GRASP_MODE_OBLIQUE,
            module.GRASP_MODE_OBLIQUE,
            module.GRASP_MODE_OBLIQUE,
        ]
    )

    selected = module.balanced_candidate_indices(transforms, scores, modes, max_candidates=6)

    selected_modes = modes[selected]
    assert selected.numel() == 6
    assert torch.bincount(selected_modes, minlength=3).tolist() == [2, 2, 2]


def test_random_candidate_sampling_never_selects_padded_entries():
    module = _load_module()
    valid = torch.tensor(
        [
            [True, False, True, False],
            [False, True, False, False],
        ]
    )
    shape_indices = torch.tensor([0, 0, 1, 1])
    generator = torch.Generator().manual_seed(3)

    selected = module.sample_valid_candidate_indices(valid, shape_indices, generator=generator)

    assert torch.all(valid[shape_indices, selected])
    assert selected[2:].tolist() == [1, 1]


def test_candidate_metadata_lookup_returns_only_valid_library_indices(tmp_path):
    module = _load_module()
    library_path = tmp_path / "references.npz"
    np.savez_compressed(
        library_path,
        shape_names=np.asarray(("egg", "wine")),
        grasp_valid=np.asarray(
            (
                (True, False, True, False),
                (False, True, True, False),
            ),
            dtype=np.bool_,
        ),
        grasp_scores=np.asarray(((0.8, 0.0, 0.6, 0.0), (0.0, 0.9, 0.7, 0.0))),
        grasp_modes=np.asarray(((0, -1, 2, -1), (-1, 1, 2, -1))),
    )

    metadata = module.load_candidate_evaluation_metadata(library_path, "wine")

    assert metadata.shape_name == "wine"
    assert metadata.candidate_ids == (1, 2)
    assert metadata.scores == pytest.approx((0.9, 0.7))
    assert metadata.modes == (1, 2)


def test_candidate_assignment_is_stable_by_global_environment_id():
    module = _load_module()

    assigned = module.assign_candidate_indices(
        torch.tensor([0, 2, 4, 7]),
        (1, 3, 7),
    )

    assert assigned.tolist() == [1, 7, 3, 3]


def test_accessible_candidate_weights_reject_below_support_and_prefer_robot_near_side():
    module = _load_module()
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0])
    candidate_positions_w = torch.tensor(
        [[[-0.12, 0.0, 0.82], [0.12, 0.0, 0.82], [-0.12, 0.0, 0.69]]]
    )
    candidate_quaternions_w = identity.expand(1, 3, 4).clone()

    weights = module.accessible_candidate_weights(
        candidate_positions_w=candidate_positions_w,
        candidate_quaternions_w=candidate_quaternions_w,
        candidate_scores=torch.tensor([[0.8, 0.8, 0.8]]),
        candidate_valid=torch.ones(1, 3, dtype=torch.bool),
        object_positions_w=torch.tensor([[0.0, 0.0, 0.80]]),
        robot_positions_w=torch.tensor([[-1.0, 0.0, 0.0]]),
        robot_quaternions_w=identity.unsqueeze(0),
        active_hand=torch.tensor([1]),
        support_top_z=torch.tensor([0.70]),
    )

    assert weights[0, 0] > 0.0
    assert weights[0, 1] == 0.0
    assert weights[0, 2] == 0.0


def test_accessible_candidate_weights_reject_far_side_pregrasp_approach():
    module = _load_module()
    root_half = 2.0**-0.5
    # Local hand +Y maps to world +X for the front approach and -X for the back approach.
    front_approach = torch.tensor([root_half, 0.0, 0.0, -root_half])
    back_approach = torch.tensor([root_half, 0.0, 0.0, root_half])

    weights = module.accessible_candidate_weights(
        candidate_positions_w=torch.tensor([[[0.0, 0.0, 0.82], [0.0, 0.0, 0.82]]]),
        candidate_quaternions_w=torch.stack((front_approach, back_approach)).unsqueeze(0),
        candidate_scores=torch.full((1, 2), 0.8),
        candidate_valid=torch.ones(1, 2, dtype=torch.bool),
        object_positions_w=torch.tensor([[0.0, 0.0, 0.80]]),
        robot_positions_w=torch.tensor([[-1.0, 0.0, 0.0]]),
        robot_quaternions_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        active_hand=torch.tensor([1]),
        support_top_z=torch.tensor([0.70]),
    )

    assert weights[0, 0] > 0.0
    assert weights[0, 1] == 0.0


def test_accessible_candidate_weights_allow_top_down_pregrasp_over_object_center():
    module = _load_module()
    top_down = module.matrix_to_quaternion_wxyz(
        torch.tensor([[[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]])
    )[0]

    weights = module.accessible_candidate_weights(
        candidate_positions_w=torch.tensor([[[0.0, 0.0, 0.82]]]),
        candidate_quaternions_w=top_down.reshape(1, 1, 4),
        candidate_scores=torch.tensor([[0.8]]),
        candidate_valid=torch.ones(1, 1, dtype=torch.bool),
        object_positions_w=torch.tensor([[0.0, 0.0, 0.80]]),
        robot_positions_w=torch.tensor([[-1.0, 0.0, 0.0]]),
        robot_quaternions_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        active_hand=torch.tensor([1]),
        active_shoulder_positions_w=torch.tensor([[-0.30, 0.0, 1.15]]),
        support_top_z=torch.tensor([0.70]),
    )

    assert weights[0, 0] > 0.0


def test_accessible_candidate_weights_reject_backhand_top_down_grasp():
    module = _load_module()
    comfortable_rotation = torch.tensor(
        [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
    )
    backhand_rotation = torch.tensor(
        [[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
    )
    candidate_quaternions_w = module.matrix_to_quaternion_wxyz(
        torch.stack((comfortable_rotation, backhand_rotation))
    ).unsqueeze(0)

    weights = module.accessible_candidate_weights(
        candidate_positions_w=torch.tensor([[[0.0, 0.0, 0.82], [0.0, 0.0, 0.82]]]),
        candidate_quaternions_w=candidate_quaternions_w,
        candidate_scores=torch.full((1, 2), 0.8),
        candidate_valid=torch.ones(1, 2, dtype=torch.bool),
        object_positions_w=torch.tensor([[0.0, 0.0, 0.80]]),
        robot_positions_w=torch.tensor([[-1.0, 0.0, 0.0]]),
        robot_quaternions_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        active_hand=torch.tensor([1]),
        active_shoulder_positions_w=torch.tensor([[-0.30, 0.0, 1.15]]),
        support_top_z=torch.tensor([0.70]),
    )

    assert weights[0, 0] > 0.0
    assert weights[0, 1] == 0.0


def test_accessible_candidate_weights_fall_back_only_to_top_grasp_when_support_gate_is_empty():
    module = _load_module()
    root_half = 2.0**-0.5
    top_down = module.matrix_to_quaternion_wxyz(
        torch.tensor([[[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]])
    )[0]
    side_from_front = torch.tensor([root_half, 0.0, 0.0, -root_half])

    weights = module.accessible_candidate_weights(
        candidate_positions_w=torch.tensor([[[-0.02, 0.0, 0.705], [-0.02, 0.0, 0.705]]]),
        candidate_quaternions_w=torch.stack((top_down, side_from_front)).unsqueeze(0),
        candidate_scores=torch.tensor([[0.5, 0.99]]),
        candidate_valid=torch.ones(1, 2, dtype=torch.bool),
        object_positions_w=torch.tensor([[0.0, 0.0, 0.70]]),
        robot_positions_w=torch.tensor([[-1.0, 0.0, 0.0]]),
        robot_quaternions_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        active_hand=torch.tensor([1]),
        active_shoulder_positions_w=torch.tensor([[-0.30, 0.0, 1.15]]),
        support_top_z=torch.tensor([0.70]),
    )

    assert weights[0, 0] > 0.0
    assert weights[0, 1] == 0.0


def test_accessible_candidate_weights_use_active_hand_as_soft_lateral_preference():
    module = _load_module()
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0])
    root_half = 2.0**-0.5
    top_down = torch.tensor([root_half, -root_half, 0.0, 0.0])
    candidate_positions_w = torch.tensor(
        [[[-0.1, 0.1, 0.82], [-0.1, -0.1, 0.82]], [[-0.1, 0.1, 0.82], [-0.1, -0.1, 0.82]]]
    )
    common = {
        "candidate_positions_w": candidate_positions_w,
        "candidate_quaternions_w": top_down.expand(2, 2, 4).clone(),
        "candidate_scores": torch.full((2, 2), 0.8),
        "candidate_valid": torch.ones(2, 2, dtype=torch.bool),
        "object_positions_w": torch.tensor([[0.0, 0.0, 0.80], [0.0, 0.0, 0.80]]),
        "robot_positions_w": torch.tensor([[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
        "robot_quaternions_w": identity.expand(2, 4).clone(),
        "active_hand": torch.tensor([0, 1]),
        "support_top_z": torch.tensor([0.70, 0.70]),
    }

    weights = module.accessible_candidate_weights(**common)

    assert weights[0, 0] > weights[0, 1]
    assert weights[1, 1] > weights[1, 0]


def test_weighted_candidate_sampling_never_selects_zero_weight_candidates():
    module = _load_module()
    weights = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])

    selected = module.sample_weighted_candidate_indices(
        weights,
        generator=torch.Generator().manual_seed(7),
    )

    assert selected.tolist() == [0, 1]


def test_parallel_gripper_orientation_gate_is_jaw_swap_symmetric_and_blocks_quarter_turn():
    contact_progress = _load_script(CONTACT_PROGRESS_PATH, "g1_dex1_contact_progress_under_test")
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    jaw_swap_about_approach = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    quarter_turn_about_world_z = torch.tensor([[2.0**-0.5, 0.0, 0.0, 2.0**-0.5]])

    symmetric_error = contact_progress.symmetric_parallel_gripper_orientation_error(
        jaw_swap_about_approach,
        identity,
    )
    wrong_error = contact_progress.symmetric_parallel_gripper_orientation_error(
        quarter_turn_about_world_z,
        identity,
    )
    gate = contact_progress.orientation_close_gate(
        torch.cat((symmetric_error, wrong_error)),
        zero_error=1.0471975512,
        transition_width=0.6981317008,
    )

    assert torch.allclose(symmetric_error, torch.zeros_like(symmetric_error), atol=1.0e-6)
    assert torch.allclose(wrong_error, torch.full_like(wrong_error, 0.5 * torch.pi), atol=1.0e-6)
    assert torch.allclose(gate, torch.tensor([1.0, 0.0]), atol=1.0e-6)


def test_grasp_reference_library_round_trip_preserves_fixed_size_contract(tmp_path):
    module = _load_module()
    path = tmp_path / "grasp_refs.npz"
    library = module.GraspReferenceLibrary(
        shape_names=("egg", "milk"),
        positions_o=torch.zeros(2, 4, 3),
        quaternions_o=torch.tensor([[[1.0, 0.0, 0.0, 0.0]] * 4] * 2),
        scores=torch.tensor([[0.9, 0.8, 0.0, 0.0], [0.7, 0.6, 0.5, 0.4]]),
        valid=torch.tensor([[True, True, False, False], [True, True, True, True]]),
        modes=torch.tensor([[0, 1, -1, -1], [0, 1, 2, 0]]),
    )

    module.save_grasp_reference_library(path, library)
    loaded = module.load_grasp_reference_library(path)

    assert loaded.shape_names == library.shape_names
    assert torch.equal(loaded.valid, library.valid)
    assert torch.equal(loaded.modes, library.modes)
    assert torch.allclose(loaded.scores, library.scores)
    with np.load(path, allow_pickle=False) as data:
        assert set(data.files) == {
            "shape_names",
            "grasp_positions_o",
            "grasp_quaternions_o",
            "grasp_scores",
            "grasp_valid",
            "grasp_modes",
        }


def test_grasp_mesh_manifest_contains_every_registered_shape_once():
    exporter_source = EXPORTER_PATH.read_text()

    assert "ALL_SHAPE_NAMES" in exporter_source
    assert "OBJECT_SHAPE_SPECS" in exporter_source
    assert '"stable_quat_wxyz"' in exporter_source
    assert '"mesh_file"' in exporter_source
    assert "mesh.export" in exporter_source


def test_offline_generator_is_resumable_and_uses_dex1_sweep_volume():
    generator_source = GENERATOR_PATH.read_text()

    assert "DEX1_SWEEP_VOLUME" in generator_source
    assert "make_sweep_volume_gripper_info" in generator_source
    assert "gripper_info=dex1_gripper_info" in generator_source
    assert "DEX1_GRIPPER_TO_HAND_CENTER" in generator_source
    assert 'default="graspmoe"' in generator_source
    assert "run_planner_on_batch" in generator_source
    assert "if raw_path.is_file() and not args.overwrite" in generator_source
    assert "se3_nms_indices" in generator_source
    assert "balanced_candidate_indices" in generator_source
    assert "_sample_support_scene_points" in generator_source
    assert "_swept_scene_collision_filter" in generator_source
    assert "sample_surface(gripper_mesh_g" in generator_source
    assert "num_gripper_collision_points" in generator_source
    assert "pregrasp_retreat" in generator_source
    assert "pregrasp_sweep_steps" in generator_source
    assert "multishape_grasp_references_k32.npz" in generator_source


def test_support_scene_point_cloud_covers_finite_platform_at_object_floor():
    generator = _load_script(GENERATOR_PATH, "g1_dex1_grasp_generator_scene_under_test")
    object_vertices_o = np.asarray(
        [[-0.02, -0.03, 0.01], [0.02, 0.03, 0.11]],
        dtype=np.float32,
    )

    scene_points_o = generator._sample_support_scene_points(
        object_vertices_o,
        stable_quat_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0]),
        support_size_xy=(0.34, 0.40),
        spacing=0.02,
    )

    assert scene_points_o.shape[1] == 3
    assert np.isclose(scene_points_o[:, 2], 0.01).all()
    assert np.isclose(scene_points_o[:, 0].min(), -0.17)
    assert np.isclose(scene_points_o[:, 0].max(), 0.17)
    assert np.isclose(scene_points_o[:, 1].min(), -0.20)
    assert np.isclose(scene_points_o[:, 1].max(), 0.20)


def test_swept_scene_collision_filter_rejects_table_contact_and_keeps_clear_grasp():
    generator = _load_script(GENERATOR_PATH, "g1_dex1_grasp_generator_collision_under_test")
    plane_x, plane_y = np.meshgrid(np.linspace(-0.2, 0.2, 21), np.linspace(-0.2, 0.2, 21))
    scene_points_o = np.stack((plane_x.reshape(-1), plane_y.reshape(-1), np.zeros(plane_x.size)), axis=-1)
    gripper_points_g = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)
    top_down_rotation = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    colliding = np.eye(4, dtype=np.float32)
    colliding[:3, :3] = top_down_rotation
    colliding[:3, 3] = [0.0, 0.0, 0.004]
    clear = colliding.copy()
    clear[:3, 3] = [0.0, 0.0, 0.10]

    accepted = generator._swept_scene_collision_filter(
        np.stack((colliding, clear)),
        gripper_points_g,
        scene_points_o,
        pregrasp_retreat=0.12,
        pregrasp_sweep_steps=5,
        collision_threshold=0.008,
    )

    assert accepted.tolist() == [False, True]


def test_filter_only_does_not_import_graspgenx_or_load_checkpoints():
    generator_source = GENERATOR_PATH.read_text()

    filter_branch = generator_source.index("if not args.filter_only:")
    graspgenx_import = generator_source.index("from graspgenx import get_checkpoints_version_dir")
    checkpoint_load = generator_source.index("checkpoint_root = Path(get_checkpoints_version_dir())")
    assert filter_branch < graspgenx_import
    assert filter_branch < checkpoint_load


def test_dex1_graspgenx_frame_maps_approach_axis_to_hand_forward_axis():
    generator = _load_script(GENERATOR_PATH, "g1_dex1_grasp_generator_under_test")

    transform_g_h = generator.DEX1_GRIPPER_TO_HAND_CENTER

    assert np.allclose(transform_g_h[:3, 3], [0.0, -0.0142, 0.09734], atol=1.0e-6)
    # Hand-center +Y (forward) is GraspGenX +Z (approach).
    assert np.allclose(transform_g_h[:3, 1], [0.0, 0.0, 1.0], atol=1.0e-6)


def test_reference_visualizer_renders_actual_gripper_mesh_and_all_shapes():
    visualizer_source = VISUALIZER_PATH.read_text()

    assert "gripper_mesh" in visualizer_source
    assert "ALL_SHAPE_NAMES" in visualizer_source
    assert "grasp_modes" in visualizer_source
    assert "contact_sheet" in visualizer_source
    assert "multishape_grasp_references_k32.npz" in visualizer_source
    assert "len(candidate_ids)" in visualizer_source
    assert "plot_surface" in visualizer_source


def test_grasp_reference_task_reuses_contact_target_pose_instead_of_parallel_observation():
    env_source = ENV_PATH.read_text()
    observation_source = OBSERVATION_PATH.read_text()
    event_source = EVENT_PATH.read_text()

    assert "load_grasp_reference_library" in env_source
    assert "accessible_candidate_weights" in env_source
    assert "select_candidate_indices" in env_source
    assert "compose_object_grasp_pose" in env_source
    assert "self.contact_label.set_target_region_pose" in env_source
    assert "env._sample_grasp_reference_indices(env_ids)" not in event_source
    assert "self._sample_grasp_reference_indices(env_ids)" in env_source
    assert "def object_goal_hand_pose_obs" in observation_source
    assert "target_orientation" in observation_source
    assert "matrix_from_quat" in observation_source
    assert "grasp_reference = ObsTerm" not in observation_source


def test_grasp_reference_sampling_uses_the_active_shoulder_for_top_down_wrist_filtering():
    env_source = ENV_PATH.read_text()

    assert "active_shoulder_positions_w=" in env_source
    assert "self.left_shoulder_body_id" in env_source
    assert "self.right_shoulder_body_id" in env_source


def test_grasp_reference_sampling_waits_until_object_reset_pose_is_available():
    env_source = ENV_PATH.read_text()
    constructor = env_source.split("    def __init__", maxsplit=1)[1].split(
        "    def _reset_contact_accumulators", maxsplit=1
    )[0]
    reset_idx = env_source.split("    def _reset_idx", maxsplit=1)[1].split(
        "    def _update_target_pose_visualization", maxsplit=1
    )[0]

    assert "self._sample_grasp_reference_indices" not in constructor
    assert "super()._reset_idx(env_ids)" in reset_idx
    assert reset_idx.index("super()._reset_idx(env_ids)") < reset_idx.index("self._sample_grasp_reference_indices(env_ids)")


def test_orientation_close_gate_is_disabled_without_a_grasp_pose_reference():
    env_source = ENV_PATH.read_text()

    assert "if self.grasp_reference_positions_o is None:" in env_source
    assert "left_orientation_error = torch.zeros_like(left_distance)" in env_source
    assert "right_orientation_error = torch.zeros_like(right_distance)" in env_source


def test_semantically_upright_packages_use_identity_stable_pose():
    shape_module = _load_script(OBJECT_SHAPE_MODULE_PATH, "g1_dex1_object_shapes_under_test")

    assert shape_module.SEMANTIC_UPRIGHT_SHAPE_NAMES == (
        "milk",
        "ketchup",
        "shaker",
        "wine",
        "coke",
        "soup_can",
        "candle",
        "soap_dispenser",
    )
    for shape_name in shape_module.SEMANTIC_UPRIGHT_SHAPE_NAMES:
        assert shape_module.OBJECT_SHAPE_SPECS[shape_name].stable_quat_wxyz == (1.0, 0.0, 0.0, 0.0)


def test_runtime_grasp_reference_diagnostics_compare_reference_and_actual_pose_only():
    env_source = ENV_PATH.read_text()
    env_cfg_source = ENV_CFG_PATH.read_text()
    marker_source = env_source.split("GRASP_REFERENCE_DIAGNOSTIC_MARKER_CFG", maxsplit=1)[1].split(
        "PLATFORM_MARKER_CFG", maxsplit=1
    )[0]
    visualizer_source = env_source.split(
        "    def _update_grasp_reference_pose_visualization", maxsplit=1
    )[1].split("    def _update_platform_pose_visualization", maxsplit=1)[0]

    assert "grasp_reference_pose_debug_vis: bool = False" in env_cfg_source
    assert "GRASP_REFERENCE_DIAGNOSTIC_MARKER_CFG" in env_source
    assert '"reference"' in marker_source
    assert '"actual"' in marker_source
    assert '"command"' not in marker_source
    assert "def _update_grasp_reference_pose_visualization" in env_source
    assert "self._selected_grasp_reference_pose_w()" in visualizer_source
    assert "self.low_level_obs_builder._target_pose_w" not in visualizer_source
    assert "hand_center = self.scene[HAND_CENTER_FRAME_NAME]" in visualizer_source
    assert "hand_center.data.target_pos_w" in visualizer_source
    assert "hand_center.data.target_quat_w" in visualizer_source
    assert "marker_indices" in visualizer_source


def test_grasp_reference_visual_launches_use_current_task_and_diagnostic_overlay():
    launch_source = LAUNCH_PATH.read_text()

    for launch_name, bps_enabled in (
        ("g1_dex1_graspref_bps_diagnostic_play", "true"),
        ("g1_dex1_graspref_no_bps_diagnostic_play", "false"),
    ):
        launch = launch_source.split(f'"name": "{launch_name}"', maxsplit=1)[1]
        launch = launch.split('"name":', maxsplit=1)[0]
        assert '"--task=HBC-Isaac-G1-Dex1-HierDrc-MultiShape-GraspRef-Play-v0"' in launch
        assert '"--num_envs=25"' in launch
        assert '"--seed=42"' in launch
        assert f'"env.object_shape_bps_enabled={bps_enabled}"' in launch
        assert '"env.grasp_reference_pose_debug_vis=true"' in launch
        assert '"env.enable_debug_visualization=false"' in launch
        assert '"env.target_pose_debug_vis=false"' in launch
        assert '"env.platform_pose_debug_vis=true"' in launch


def test_grasp_reference_isolated_task_does_not_break_existing_multishape_checkpoints():
    env_cfg_source = ENV_CFG_PATH.read_text()
    multishape_source = MULTISHAPE_CFG_PATH.read_text()
    registration_source = REGISTRATION_PATH.read_text()

    assert "grasp_reference_enabled: bool = False" in env_cfg_source
    assert "grasp_reference_data_path: str =" in env_cfg_source
    assert "G1Dex1HierDrcMultiShapeGraspRefEnvCfg" in multishape_source
    assert "G1Dex1HierDrcMultiShapeGraspRefPlayEnvCfg" in multishape_source
    assert "G1Dex1HierDrcMultiShapeGraspRefObservationsCfg" in multishape_source
    assert "HBC-Isaac-G1-Dex1-HierDrc-MultiShape-GraspRef-v0" in registration_source
    assert "HBC-Isaac-G1-Dex1-HierDrc-MultiShape-GraspRef-Play-v0" in registration_source


def test_grasp_reference_alone_enables_parameterized_pose_guidance():
    env_cfg_source = ENV_CFG_PATH.read_text()
    multishape_source = MULTISHAPE_CFG_PATH.read_text()

    assert "grasp_pose_guidance_enabled: bool = False" in env_cfg_source
    assert "grasp_pose_position_scale: float = 0.15" in env_cfg_source
    assert "grasp_pose_position_reward_weight: float = 2.0" in env_cfg_source
    assert "grasp_pose_orientation_reward_weight: float = 2.0" in env_cfg_source
    assert "grasp_pose_release_threshold: float = 0.50" in env_cfg_source
    assert "grasp_pose_release_width: float = 0.08" in env_cfg_source
    graspref_cfg = multishape_source.split(
        "class G1Dex1HierDrcMultiShapeGraspRefEnvCfg", maxsplit=1
    )[1].split("class G1Dex1HierDrcMultiShapeGraspRefPlayEnvCfg", maxsplit=1)[0]
    assert "self.grasp_pose_guidance_enabled = True" in graspref_cfg


def test_candidate_play_configures_one_asset_and_twelve_yaw_conditioned_best_poses():
    config_source = MULTISHAPE_CFG_PATH.read_text()
    play_source = PLAY_PATH.read_text()

    assert "grasp_candidate_asset_id: int = -1" in ENV_CFG_PATH.read_text()
    assert "grasp_candidate_yaw_count: int = 0" in ENV_CFG_PATH.read_text()
    assert "grasp_reference_select_highest_weight: bool = False" in ENV_CFG_PATH.read_text()
    assert "def apply_grasp_candidate_evaluation" in config_source
    assert "GRASP_REF_ALL_SHAPE_NAMES[self.grasp_candidate_asset_id]" in config_source
    assert "self.scene.num_envs = 12" in config_source
    assert "self.grasp_candidate_yaw_count = 12" in config_source
    assert "self.grasp_reference_select_highest_weight = True" in config_source
    assert "self.scene.object = _multishape_object_cfg((shape_name,), (1.0,))" in config_source
    assert "self.events.reset_object.params[\"pose_range\"]" in config_source
    assert "self.events.reset_base.params[\"pose_range\"]" in config_source
    assert "env_cfg.apply_grasp_candidate_evaluation()" in play_source
    assert play_source.index("_apply_cfg_overrides(env_cfg, agent_cfg, hydra_args)") < play_source.index(
        "env_cfg.apply_grasp_candidate_evaluation()"
    )
    assert play_source.index("env_cfg.apply_grasp_candidate_evaluation()") < play_source.index(
        "env = gym.make(args_cli.task"
    )


def test_generator_applies_ergonomic_filter_before_candidate_balancing():
    generator_source = GENERATOR_PATH.read_text()

    assert "ergonomic_wrist_mask" in generator_source
    assert generator_source.index("ergonomic_wrist_mask") < generator_source.index(
        "balanced_candidate_indices"
    )


def test_candidate_visualization_does_not_change_object_center_policy_target():
    env_cfg_source = ENV_CFG_PATH.read_text()
    multishape_source = MULTISHAPE_CFG_PATH.read_text()
    env_source = ENV_PATH.read_text()

    assert "grasp_reference_controls_target: bool = False" in env_cfg_source
    graspref_cfg = multishape_source.split(
        "class G1Dex1HierDrcMultiShapeGraspRefEnvCfg", maxsplit=1
    )[1].split("class G1Dex1HierDrcMultiShapeGraspRefPlayEnvCfg", maxsplit=1)[0]
    control_cfg = multishape_source.split(
        "class G1Dex1HierDrcMultiShapeObjectCenterControlEnvCfg", maxsplit=1
    )[1].split("class G1Dex1HierDrcMultiShapeObjectCenterControlPlayEnvCfg", maxsplit=1)[0]
    assert "self.grasp_reference_controls_target = True" in graspref_cfg
    assert "self.grasp_reference_controls_target = False" in control_cfg
    assert "def _selected_grasp_reference_pose_w" in env_source
    assert "if not self.cfg.grasp_reference_controls_target:" in env_source
    visualizer_source = env_source.split("    def _update_grasp_reference_pose_visualization", maxsplit=1)[1]
    assert "self._selected_grasp_reference_pose_w()" in visualizer_source


def test_candidate_play_wrapper_maps_both_policy_modes_without_copying_play_loop():
    assert CANDIDATE_PLAY_PATH.is_file()
    module = _load_script(CANDIDATE_PLAY_PATH, "g1_dex1_candidate_play_under_test")

    graspref = module.build_play_arguments(
        ["--mode", "graspref", "--asset_id", "8", "--checkpoint=model.pt"]
    )
    control = module.build_play_arguments(
        ["--mode", "object_center", "--asset_id", "8", "--checkpoint=model.pt"]
    )

    assert "--task=HBC-Isaac-G1-Dex1-HierDrc-MultiShape-GraspRef-Play-v0" in graspref
    assert "--task=HBC-Isaac-G1-Dex1-HierDrc-MultiShape-ObjectCenterControl-Play-v0" in control
    for arguments in (graspref, control):
        assert "env.grasp_candidate_asset_id=8" in arguments
        assert "env.object_shape_bps_enabled=true" in arguments
        assert "env.commands.high_level.left_hand_probability=0.0" in arguments
        assert "env.grasp_reference_pose_debug_vis=true" in arguments
        assert "env.platform_pose_debug_vis=true" in arguments
        assert "env.target_pose_debug_vis=false" in arguments
        assert "--checkpoint=model.pt" in arguments
    source = CANDIDATE_PLAY_PATH.read_text()
    assert "os.execv" in source
    assert "while simulation_app.is_running()" not in source


def test_object_center_control_matches_graspref_pose_observation_dimensions():
    multishape_source = MULTISHAPE_CFG_PATH.read_text()
    registration_source = REGISTRATION_PATH.read_text()

    control_cfg = multishape_source.split(
        "class G1Dex1HierDrcMultiShapeObjectCenterControlEnvCfg", maxsplit=1
    )[1].split("class G1Dex1HierDrcMultiShapeObjectCenterControlPlayEnvCfg", maxsplit=1)[0]
    assert "G1Dex1HierDrcMultiShapeGraspRefEnvCfg" in multishape_source
    assert "self.grasp_reference_enabled = False" in control_cfg
    assert "self.grasp_pose_guidance_enabled = False" in control_cfg
    assert "HBC-Isaac-G1-Dex1-HierDrc-MultiShape-ObjectCenterControl-v0" in registration_source
    assert "HBC-Isaac-G1-Dex1-HierDrc-MultiShape-ObjectCenterControl-Play-v0" in registration_source


def test_grasp_pose_diagnostics_separate_reference_generation_and_low_level_tracking():
    env_source = ENV_PATH.read_text()

    assert "def _active_grasp_pose_diagnostics" in env_source
    assert '"GraspPose/reference_command_orientation_error_mean"' in env_source
    assert '"GraspPose/command_actual_orientation_error_mean"' in env_source
    assert '"GraspPose/reference_actual_orientation_error_mean"' in env_source
    assert '"GraspPose/position_gate_mean"' in env_source
    assert '"GraspPose/orientation_gate_mean"' in env_source
    assert '"DRC/c_physical_grasp_mean"' in env_source


def test_grasp_reference_scene_uses_flat_platforms_without_tray_walls():
    scene_source = SCENE_PATH.read_text()
    config_source = MULTISHAPE_CFG_PATH.read_text()
    env_cfg_source = ENV_CFG_PATH.read_text()
    env_source = ENV_PATH.read_text()

    assert "class G1Dex1HierDrcMultiShapeGraspRefSceneCfg" in scene_source
    assert "_multishape_flat_platform_cfg" in scene_source
    assert 'initial_x = 2.0 if "object_init_platform" in platform_cfg.prim_path else -2.0' in scene_source
    assert "0.5 * SHALLOW_TRAY_SIZE[2]" in scene_source
    assert "object_init_platform: RigidObjectCfg = _multishape_flat_platform_cfg" in scene_source
    assert "object_target_platform: RigidObjectCfg = _multishape_flat_platform_cfg" in scene_source
    assert "_multishape_object_cfg(GRASP_REF_TRAIN_SHAPE_NAMES, (1.0,))" in scene_source
    assert "object_size_scale_factors: tuple[float, ...] = (1.0,)" in config_source
    assert "scene: G1Dex1HierDrcMultiShapeGraspRefSceneCfg" in config_source
    assert "multishape_platform_has_walls: bool = True" in env_cfg_source
    assert "self.multishape_platform_has_walls = False" in config_source
    assert "if self.cfg.multishape_platform_has_walls:" in env_source


def test_grasp_reference_training_quarantines_physically_unstable_assets():
    shape_source = OBJECT_SHAPE_MODULE_PATH.read_text()
    scene_source = SCENE_PATH.read_text()
    config_source = MULTISHAPE_CFG_PATH.read_text()

    assert 'GRASP_REF_EXCLUDED_SHAPE_NAMES = ("orange_peel", "trash_can")' in shape_source
    assert "GRASP_REF_TRAIN_SHAPE_NAMES" in shape_source
    assert "GRASP_REF_ALL_SHAPE_NAMES" in shape_source
    assert "_multishape_object_cfg(GRASP_REF_TRAIN_SHAPE_NAMES" in scene_source
    assert "object_shape_names: tuple[str, ...] = GRASP_REF_TRAIN_SHAPE_NAMES" in config_source


def test_stable_pose_calibration_uses_physics_settling_and_reports_residual_motion():
    calibrator_source = CALIBRATOR_PATH.read_text()

    assert "--settle-seconds" in calibrator_source
    assert "--drop-height" in calibrator_source
    assert "--linear-velocity-threshold" in calibrator_source
    assert "--angular-velocity-threshold" in calibrator_source
    assert "scene.update" in calibrator_source
    assert '"stable_quat_wxyz"' in calibrator_source
    assert '"linear_speed"' in calibrator_source
    assert '"angular_speed"' in calibrator_source
    assert '"settled"' in calibrator_source
    assert '"on_support"' in calibrator_source
    assert '"support_xy_error"' in calibrator_source
    assert "shape_names = tuple(env_cfg.object_shape_names)" in calibrator_source


def test_contact_sheet_visualizer_applies_manifest_stable_pose_before_rendering():
    visualizer_source = VISUALIZER_PATH.read_text()

    assert 'entry["stable_quat_wxyz"]' in visualizer_source
    assert "stable_rotation" in visualizer_source
    assert "object_triangles_stable" in visualizer_source
