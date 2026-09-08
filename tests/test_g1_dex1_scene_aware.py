from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = (
    REPO_ROOT
    / "source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_scene_aware_hier_drc"
)


def _load_function(path: Path, name: str):
    module = ast.parse(path.read_text())
    function = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def test_ray_hits_are_encoded_as_bounded_proximity() -> None:
    encode = _load_function(TASK_ROOT / "mdp/observations.py", "proximity_from_ray_hits")
    hits = torch.tensor([[[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [float("inf"), 0.0, 0.0]]])
    origin = torch.zeros(1, 3)

    proximity = encode(hits, origin, max_distance=3.0)

    torch.testing.assert_close(proximity, torch.tensor([[2.0 / 3.0, 0.0, 0.0]]))


def test_scene_aware_task_keeps_equal_dimensional_no_perception_ablation() -> None:
    config_source = (TASK_ROOT / "config/env_cfg.py").read_text()
    observation_source = (TASK_ROOT / "mdp/observations.py").read_text()
    assert "environment_perception_enabled: bool = False" in config_source
    assert "ENVIRONMENT_VOXEL_COUNT" in observation_source
    assert "torch.zeros(env.num_envs, ENVIRONMENT_VOXEL_COUNT" in observation_source


def test_environment_observation_is_a_decaying_ego_motion_compensated_voxel() -> None:
    observation_source = (TASK_ROOT / "mdp/observations.py").read_text()
    scene_source = (TASK_ROOT / "mdp/scenes.py").read_text()
    assert "_warp_previous_occupancy" in observation_source
    assert "environment_voxel_half_life_s" in observation_source
    assert "ENVIRONMENT_VOXEL_SHAPE_XYZ = (20, 20, 24)" in scene_source


def test_environment_scan_density_resolves_thin_interaction_geometry() -> None:
    scene_source = (TASK_ROOT / "mdp/scenes.py").read_text()
    observation_source = (TASK_ROOT / "mdp/observations.py").read_text()
    assert "ENVIRONMENT_LIDAR_EFFECTIVE_VERTICAL_CHANNELS = 25" in scene_source
    assert "ENVIRONMENT_LIDAR_EFFECTIVE_HORIZONTAL_SAMPLES = 49" in scene_source
    assert "_interlaced_angles" in observation_source


def test_scene_pairing_and_registration_are_explicit() -> None:
    event_source = (TASK_ROOT / "mdp/events.py").read_text()
    registration_source = (TASK_ROOT / "__init__.py").read_text()
    assert "torch.remainder(env_ids, 2) == 1" in event_source
    assert "SceneAware-NoPerception" in registration_source


def test_environment_scan_covers_overhead_and_side_geometry() -> None:
    scene_source = (TASK_ROOT / "mdp/scenes.py").read_text()
    assert "ENVIRONMENT_LIDAR_VERTICAL_FOV = (-60.0, 60.0)" in scene_source
    assert "ENVIRONMENT_LIDAR_HORIZONTAL_FOV = (-75.0, 75.0)" in scene_source


def test_oriented_box_signed_distance_is_negative_inside_and_positive_outside() -> None:
    geometry_path = TASK_ROOT / "mdp/geometry_shaping.py"
    module = ast.parse(geometry_path.read_text())
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "point_oriented_box_signed_distance"
    )

    class IdentityQuaternionMath:
        @staticmethod
        def quat_apply_inverse(_quaternion, vector):
            return vector

    namespace = {"torch": torch, "math_utils": IdentityQuaternionMath}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(geometry_path), "exec"), namespace)
    signed_distance = namespace["point_oriented_box_signed_distance"]
    points = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
    centers = torch.zeros(1, 1, 3)
    quaternions = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
    half_extents = torch.ones(1, 1, 3)

    distance = signed_distance(points, centers, quaternions, half_extents)

    torch.testing.assert_close(distance, torch.tensor([[[-1.0, 1.0]]]))


def test_checkpoint_expansion_preserves_old_columns_and_zeroes_new_inputs() -> None:
    script_path = REPO_ROOT / "scripts/rsl_rl/expand_checkpoint_for_scene_perception.py"
    spec = importlib.util.spec_from_file_location("scene_checkpoint_expansion", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    actor = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    critic = torch.arange(10, dtype=torch.float32).reshape(2, 5)
    checkpoint = {"model_state_dict": {"actor.0.weight": actor.clone(), "critic.0.weight": critic.clone()}}

    expanded = module.expand_scene_perception_checkpoint(checkpoint, actor_extra_dim=2, critic_extra_dim=3)

    torch.testing.assert_close(expanded["model_state_dict"]["actor.0.weight"][:, :4], actor)
    torch.testing.assert_close(expanded["model_state_dict"]["critic.0.weight"][:, :5], critic)
    assert torch.count_nonzero(expanded["model_state_dict"]["actor.0.weight"][:, 4:]) == 0
    assert torch.count_nonzero(expanded["model_state_dict"]["critic.0.weight"][:, 5:]) == 0


def test_rolling_voxel_checkpoint_replacement_preserves_critic_privileged_suffix() -> None:
    script_path = REPO_ROOT / "scripts/rsl_rl/expand_checkpoint_for_scene_perception.py"
    spec = importlib.util.spec_from_file_location("scene_checkpoint_replacement", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    actor = torch.arange(21, dtype=torch.float32).reshape(3, 7)
    critic = torch.arange(18, dtype=torch.float32).reshape(2, 9)
    checkpoint = {
        "model_state_dict": {
            "actor.latent_actor.0.weight": actor.clone(),
            "critic.0.weight": critic.clone(),
        }
    }

    replaced = module.replace_scan_with_rolling_voxel(
        checkpoint,
        old_environment_dim=3,
        new_environment_dim=5,
        actor_prefix_dim=4,
        critic_prefix_dim=4,
        critic_suffix_dim=2,
    )

    actor_new = replaced["model_state_dict"]["actor.latent_actor.0.weight"]
    critic_new = replaced["model_state_dict"]["critic.0.weight"]
    torch.testing.assert_close(actor_new[:, :4], actor[:, :4])
    torch.testing.assert_close(critic_new[:, :4], critic[:, :4])
    torch.testing.assert_close(critic_new[:, -2:], critic[:, -2:])
    assert torch.count_nonzero(actor_new[:, 4:]) == 0
    assert torch.count_nonzero(critic_new[:, 4:-2]) == 0
