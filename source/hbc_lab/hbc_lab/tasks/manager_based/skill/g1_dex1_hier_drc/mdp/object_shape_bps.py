from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


NUM_BPS_POINTS = 64
BPS_RADIUS = 0.25
SHAPE_INDEX_MASS_BASE = 10.0
SHAPE_INDEX_MASS_STRIDE = 0.125
MULTISHAPE_SIZE_FACTORS = (0.8, 0.9, 1.0, 1.1, 1.2)

OBJECT_MODEL_ROOT = Path(__file__).resolve().parents[5] / "assets" / "models" / "objects"
BPS_DATA_PATH = OBJECT_MODEL_ROOT / "bps" / "multishape_directional_bps64.npz"
SEMANTIC_UPRIGHT_SHAPE_NAMES = (
    "milk",
    "ketchup",
    "shaker",
    "wine",
    "coke",
    "soup_can",
    "candle",
    "soap_dispenser",
)


@dataclass(frozen=True)
class ObjectShapeSpec:
    relative_usd_path: str
    scale: tuple[float, float, float]
    stable_quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    @property
    def usd_path(self) -> Path:
        return OBJECT_MODEL_ROOT / self.relative_usd_path


OBJECT_SHAPE_SPECS = {
    "egg": ObjectShapeSpec(
        "food/egg/egg_rigid.usd", (0.07, 0.07, 0.07), (0.675791204, 0.710872591, 0.134250939, -0.141220063)
    ),
    "milk": ObjectShapeSpec(
        "food/milk/milk_rigid.usd", (0.12, 0.12, 0.12), (1.0, 0.0, 0.0, 0.0)
    ),
    "ketchup": ObjectShapeSpec(
        "food/ketchup/ketchup_rigid.usd", (0.12, 0.12, 0.12), (1.0, 0.0, 0.0, 0.0)
    ),
    "butter": ObjectShapeSpec(
        "food/butter/butter_rigid.usd", (1.0, 1.0, 1.0), (0.009974809, 0.999945402, 0.000031275, -0.003135032)
    ),
    "hotdog": ObjectShapeSpec(
        "food/hotdog/hotdog_rigid.usd", (0.10, 0.10, 0.10), (0.462164074, 0.886787474, -0.001628548, 0.003124782)
    ),
    "shaker": ObjectShapeSpec(
        "food/shaker/shaker_rigid.usd", (0.10, 0.10, 0.10), (1.0, 0.0, 0.0, 0.0)
    ),
    "cheese": ObjectShapeSpec(
        "food/cheese/cheese_rigid.usd", (1.5, 1.5, 1.5), (0.999992609, -0.002575672, -0.002863210, -0.000007431)
    ),
    "cucumber": ObjectShapeSpec(
        "vegetable/cucumber/cucumber_rigid.usd", (0.10, 0.10, 0.10), (0.228375658, -0.973546624, -0.001640793, -0.006994769)
    ),
    "wine": ObjectShapeSpec(
        "beverage/wine/wine_rigid.usd", (0.20, 0.20, 0.20), (1.0, 0.0, 0.0, 0.0)
    ),
    "coke": ObjectShapeSpec(
        "beverage/coke/coke_rigid.usd",
        (0.007, 0.007, 0.007),
        (1.0, 0.0, 0.0, 0.0),
    ),
    "soup_can": ObjectShapeSpec(
        "food/soup_can/soup_can_rigid.usd", (0.70, 0.70, 0.70), (1.0, 0.0, 0.0, 0.0)
    ),
    "mango": ObjectShapeSpec(
        "fruit/mango/mango_rigid.usd", (0.07, 0.07, 0.07), (0.612149537, 0.788315117, 0.037968319, -0.048894934)
    ),
    "pear": ObjectShapeSpec(
        "fruit/pear/pear_rigid.usd", (0.10, 0.10, 0.10), (0.542952061, 0.777261436, 0.182055920, -0.260621428)
    ),
    "candle": ObjectShapeSpec(
        "others/candle/candle_rigid.usd", (0.07, 0.07, 0.10), (1.0, 0.0, 0.0, 0.0)
    ),
    "sneaker": ObjectShapeSpec(
        "others/sneaker/sneaker_rigid.usd", (0.005, 0.005, 0.005), (0.700980067, 0.713180542, -0.000543760, 0.000553073)
    ),
    "soap_dispenser": ObjectShapeSpec(
        "others/soap_dispenser/soap_dispenser_rigid.usd",
        (0.004, 0.004, 0.005),
        (1.0, 0.0, 0.0, 0.0),
    ),
    "sponge": ObjectShapeSpec(
        "others/sponge/sponge_rigid.usd", (0.07, 0.07, 0.10), (0.148071036, 0.974494278, -0.025332004, 0.166715950)
    ),
    "toy_car": ObjectShapeSpec(
        "toy/toy_car/toy_car_rigid.usd",
        (0.005, 0.005, 0.005),
        (0.70710678, 0.70710678, 0.0, 0.0),
    ),
    "toy_ship": ObjectShapeSpec(
        "toy/toy_ship/toy_ship.usd", (0.0003, 0.0003, 0.0003), (0.020576978, 0.924345493, -0.008479409, 0.380906969)
    ),
    "orange_peel": ObjectShapeSpec(
        "trash/orange_peel/orange_peel.usd", (0.8, 0.8, 0.8), (0.707106054, 0.707107544, 0.000002267, -0.000002383)
    ),
    "rotten_apple": ObjectShapeSpec(
        "trash/rotten_apple/rotten_apple.usd", (0.8, 0.8, 0.8), (0.852943718, -0.053780735, -0.518195927, -0.032673683)
    ),
    "rotten_banana": ObjectShapeSpec(
        "trash/rotten_banana/rotten_banana.usd", (0.8, 0.8, 0.8), (0.131598100, -0.989935458, 0.006859728, 0.051601868)
    ),
    "trash_can": ObjectShapeSpec(
        "trash/trash_can/trash_can.usd", (0.007, 0.010, 0.007), (1.0, 0.0, 0.0, 0.0)
    ),
    "bowl": ObjectShapeSpec(
        "container/bowl_0/bowl_rigid.usd", (0.0025, 0.0025, 0.0035), (1.0, 0.0, 0.0, 0.0)
    ),
    "donut": ObjectShapeSpec(
        "food/donut/donut_rigid.usd", (0.07, 0.07, 0.07), (0.048109546, 0.998840213, -0.000092884, 0.001930463)
    ),
    "bread_bag": ObjectShapeSpec(
        "food/bread_bag/bread_bag_rigid.usd", (0.003, 0.003, 0.003), (0.655712306, 0.755007863, 0.001388243, -0.001598423)
    ),
    "toy_gun": ObjectShapeSpec(
        "toy/toy_gun/toy_gun_rigid.usd", (0.70, 0.70, 0.70), (1.0, -0.000064432, 0.000071284, -0.000000054)
    ),
}

TRAIN_SHAPE_NAMES = (
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
HELD_OUT_SHAPE_NAMES = ("donut", "bread_bag", "toy_gun")
ALL_SHAPE_NAMES = (*TRAIN_SHAPE_NAMES, *HELD_OUT_SHAPE_NAMES)
GRASP_REF_EXCLUDED_SHAPE_NAMES = ("orange_peel", "trash_can")
GRASP_REF_TRAIN_SHAPE_NAMES = tuple(
    shape_name for shape_name in TRAIN_SHAPE_NAMES if shape_name not in GRASP_REF_EXCLUDED_SHAPE_NAMES
)
GRASP_REF_ALL_SHAPE_NAMES = (*GRASP_REF_TRAIN_SHAPE_NAMES, *HELD_OUT_SHAPE_NAMES)


def format_shape_assignment(shape_names: tuple[str, ...]) -> str:
    """Format the deterministic environment-to-shape mapping for visual evaluation."""
    lines = ["[MultiShape Visual Eval] deterministic asset assignment:"]
    for env_id, shape_name in enumerate(shape_names):
        split = "OOD" if shape_name in HELD_OUT_SHAPE_NAMES else "train"
        lines.append(f"  env {env_id:02d}: {shape_name} ({split})")
    return "\n".join(lines)


def fibonacci_ball_basis(
    num_points: int,
    *,
    radius: float,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate a deterministic approximately uniform basis inside a sphere."""
    if num_points < 1:
        raise ValueError("num_points must be positive")
    indices = torch.arange(num_points, device=device, dtype=dtype)
    unit = (indices + 0.5) / float(num_points)
    z = 1.0 - 2.0 * unit
    azimuth = indices * (torch.pi * (3.0 - 5.0**0.5))
    radial_xy = torch.sqrt(torch.clamp(1.0 - z.square(), min=0.0))
    directions = torch.stack(
        (radial_xy * torch.cos(azimuth), radial_xy * torch.sin(azimuth), z),
        dim=-1,
    )
    radii = float(radius) * torch.pow(unit, 1.0 / 3.0)
    return directions * radii.unsqueeze(-1)


def directional_bps_from_surface_points(
    surface_points_o: torch.Tensor,
    basis_points_o: torch.Tensor,
    *,
    bounds_min_o: torch.Tensor | None = None,
    bounds_max_o: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode nearest-surface delta vectors from a shared object-local BPS basis."""
    if surface_points_o.ndim != 2 or surface_points_o.shape[-1] != 3:
        raise ValueError("surface_points_o must have shape (num_points, 3)")
    if basis_points_o.ndim != 2 or basis_points_o.shape[-1] != 3:
        raise ValueError("basis_points_o must have shape (num_basis, 3)")
    if surface_points_o.shape[0] == 0:
        raise ValueError("surface_points_o cannot be empty")

    if (bounds_min_o is None) != (bounds_max_o is None):
        raise ValueError("bounds_min_o and bounds_max_o must be provided together")
    if bounds_min_o is None:
        bounds_min = surface_points_o.amin(dim=0)
        bounds_max = surface_points_o.amax(dim=0)
    else:
        if bounds_min_o.shape != (3,) or bounds_max_o.shape != (3,):
            raise ValueError("geometry bounds must each have shape (3,)")
        bounds_min = bounds_min_o.to(device=surface_points_o.device, dtype=surface_points_o.dtype)
        bounds_max = bounds_max_o.to(device=surface_points_o.device, dtype=surface_points_o.dtype)
    geometry_center_o = 0.5 * (bounds_min + bounds_max)
    centered_surface_points = surface_points_o - geometry_center_o
    nearest_indices = torch.cdist(basis_points_o, centered_surface_points).argmin(dim=-1)
    surface_offsets_o = centered_surface_points[nearest_indices] - basis_points_o
    return surface_offsets_o, geometry_center_o, bounds_min, bounds_max


def _quat_conjugate(quat: torch.Tensor) -> torch.Tensor:
    return torch.cat((quat[..., :1], -quat[..., 1:]), dim=-1)


def _quat_apply(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    quat_xyz = quat[..., 1:]
    twice_cross = 2.0 * torch.linalg.cross(quat_xyz, vector, dim=-1)
    return vector + quat[..., :1] * twice_cross + torch.linalg.cross(quat_xyz, twice_cross, dim=-1)


def directional_bps_in_robot_root_frame(
    surface_offsets_o: torch.Tensor,
    object_quat_w: torch.Tensor,
    robot_root_quat_w: torch.Tensor,
) -> torch.Tensor:
    """Rotate object-local BPS surface offsets into the G1 articulation-root frame."""
    if surface_offsets_o.ndim != 3 or surface_offsets_o.shape[-1] != 3:
        raise ValueError("surface_offsets_o must have shape (num_envs, num_basis, 3)")
    if object_quat_w.shape != (surface_offsets_o.shape[0], 4):
        raise ValueError("object_quat_w must have shape (num_envs, 4)")
    if robot_root_quat_w.shape != (surface_offsets_o.shape[0], 4):
        raise ValueError("robot_root_quat_w must have shape (num_envs, 4)")

    num_basis = surface_offsets_o.shape[1]
    object_quat = object_quat_w.unsqueeze(1).expand(-1, num_basis, -1)
    root_quat_inv = _quat_conjugate(robot_root_quat_w).unsqueeze(1).expand(-1, num_basis, -1)
    offsets_w = _quat_apply(object_quat, surface_offsets_o)
    return _quat_apply(root_quat_inv, offsets_w)


def directional_bps_in_grasp_frame(
    surface_offsets_o: torch.Tensor,
    basis_points_o: torch.Tensor,
    geometry_center_o: torch.Tensor,
    grasp_position_o: torch.Tensor,
    grasp_quaternion_o: torch.Tensor,
    size_scale: torch.Tensor,
) -> torch.Tensor:
    """Express reconstructed object surface samples in the selected grasp frame."""
    if surface_offsets_o.ndim != 3 or surface_offsets_o.shape[-1] != 3:
        raise ValueError("surface_offsets_o must have shape (num_envs, num_basis, 3)")
    num_envs, num_basis, _ = surface_offsets_o.shape
    if basis_points_o.shape != (num_basis, 3):
        raise ValueError(f"basis_points_o must have shape ({num_basis}, 3)")
    if geometry_center_o.shape != (num_envs, 3):
        raise ValueError("geometry_center_o must have shape (num_envs, 3)")
    if grasp_position_o.shape != (num_envs, 3):
        raise ValueError("grasp_position_o must have shape (num_envs, 3)")
    if grasp_quaternion_o.shape != (num_envs, 4):
        raise ValueError("grasp_quaternion_o must have shape (num_envs, 4)")

    size_scale = size_scale.to(device=surface_offsets_o.device, dtype=surface_offsets_o.dtype).reshape(-1)
    if size_scale.shape[0] != num_envs:
        raise ValueError("size_scale must contain one value per environment")

    scaled_offsets_o = scale_directional_bps_offsets(surface_offsets_o, basis_points_o, size_scale)
    basis = basis_points_o.to(device=surface_offsets_o.device, dtype=surface_offsets_o.dtype).unsqueeze(0)
    surface_points_o = (
        basis
        + scaled_offsets_o
        + geometry_center_o.to(device=surface_offsets_o.device, dtype=surface_offsets_o.dtype).unsqueeze(1)
        * size_scale[:, None, None]
    )
    scaled_grasp_position_o = (
        grasp_position_o.to(device=surface_offsets_o.device, dtype=surface_offsets_o.dtype)
        * size_scale.unsqueeze(-1)
    )
    relative_points_o = surface_points_o - scaled_grasp_position_o.unsqueeze(1)
    grasp_quat_inv = _quat_conjugate(
        grasp_quaternion_o.to(device=surface_offsets_o.device, dtype=surface_offsets_o.dtype)
    ).unsqueeze(1).expand(-1, num_basis, -1)
    return _quat_apply(grasp_quat_inv, relative_points_o)


def apply_directional_bps_ablation(descriptor: torch.Tensor, *, enabled: bool) -> torch.Tensor:
    """Keep the BPS tensor contract fixed while removing shape information."""
    if enabled:
        return descriptor
    return torch.zeros_like(descriptor)


def scale_directional_bps_offsets(
    surface_offsets_o: torch.Tensor,
    basis_points_o: torch.Tensor,
    size_scale: torch.Tensor,
) -> torch.Tensor:
    """Uniformly scale reconstructed BPS surface points while keeping the basis fixed."""
    if surface_offsets_o.ndim != 3 or surface_offsets_o.shape[-1] != 3:
        raise ValueError("surface_offsets_o must have shape (num_envs, num_basis, 3)")
    if basis_points_o.shape != surface_offsets_o.shape[1:]:
        raise ValueError(
            f"basis_points_o must have shape {tuple(surface_offsets_o.shape[1:])}, "
            f"got {tuple(basis_points_o.shape)}"
        )
    size_scale = size_scale.to(device=surface_offsets_o.device, dtype=surface_offsets_o.dtype).reshape(-1)
    if size_scale.shape[0] != surface_offsets_o.shape[0]:
        raise ValueError("size_scale must contain one value per environment")
    basis = basis_points_o.to(device=surface_offsets_o.device, dtype=surface_offsets_o.dtype).unsqueeze(0)
    surface_points_o = basis + surface_offsets_o
    return surface_points_o * size_scale[:, None, None] - basis


def rotated_box_min_z(
    bounds_min_o: torch.Tensor,
    bounds_max_o: torch.Tensor,
    quat_w: torch.Tensor,
) -> torch.Tensor:
    """Return the lowest rotated AABB corner for each object-local geometry box."""
    if bounds_min_o.ndim != 2 or bounds_min_o.shape[-1] != 3:
        raise ValueError("bounds_min_o must have shape (num_envs, 3)")
    if bounds_max_o.shape != bounds_min_o.shape:
        raise ValueError("bounds_max_o must match bounds_min_o")
    if quat_w.shape != (bounds_min_o.shape[0], 4):
        raise ValueError("quat_w must have shape (num_envs, 4)")
    corner_bits = torch.tensor(
        (
            (0, 0, 0),
            (0, 0, 1),
            (0, 1, 0),
            (0, 1, 1),
            (1, 0, 0),
            (1, 0, 1),
            (1, 1, 0),
            (1, 1, 1),
        ),
        device=bounds_min_o.device,
        dtype=torch.bool,
    )
    corners_o = torch.where(
        corner_bits.unsqueeze(0),
        bounds_max_o.unsqueeze(1),
        bounds_min_o.unsqueeze(1),
    )
    quat = quat_w.unsqueeze(1).expand(-1, corners_o.shape[1], -1)
    corners_w = _quat_apply(quat, corners_o)
    return corners_w[..., 2].amin(dim=1)


def decode_shape_index_from_spawn_mass(
    default_mass: torch.Tensor,
    *,
    base_mass: float = SHAPE_INDEX_MASS_BASE,
    stride: float = SHAPE_INDEX_MASS_STRIDE,
) -> torch.Tensor:
    """Decode per-environment shape metadata authored in the spawn-time default mass."""
    if stride <= 0.0:
        raise ValueError("stride must be positive")
    flattened_mass = default_mass.reshape(default_mass.shape[0], -1)[:, 0]
    continuous_index = (flattened_mass - float(base_mass)) / float(stride)
    shape_index = torch.round(continuous_index).to(dtype=torch.long)
    residual = torch.abs(continuous_index - shape_index.to(dtype=continuous_index.dtype))
    if bool(torch.any(residual > 1.0e-3)):
        raise ValueError("Default object mass does not contain valid shape index metadata")
    return shape_index


def decode_shape_scale_variant_indices(
    default_mass: torch.Tensor,
    *,
    num_scale_factors: int,
    base_mass: float = SHAPE_INDEX_MASS_BASE,
    stride: float = SHAPE_INDEX_MASS_STRIDE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode shape-major `(shape, uniform scale)` spawn variants from authored mass metadata."""
    if num_scale_factors < 1:
        raise ValueError("num_scale_factors must be positive")
    variant_index = decode_shape_index_from_spawn_mass(
        default_mass,
        base_mass=base_mass,
        stride=stride,
    )
    return torch.div(variant_index, num_scale_factors, rounding_mode="floor"), variant_index % num_scale_factors


def load_directional_bps_data(
    path: str | Path = BPS_DATA_PATH,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor | tuple[str, ...]]:
    """Load generated directional BPS data without adding a runtime mesh dependency."""
    import numpy as np

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Directional BPS data not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        names = tuple(str(name) for name in data["shape_names"].tolist())
        tensors = {
            key: torch.as_tensor(data[key], device=device, dtype=dtype)
            for key in (
                "basis_points_o",
                "surface_offsets_o",
                "geometry_center_offsets_o",
                "bounds_min_o",
                "bounds_max_o",
                "principal_axes_o",
            )
        }
    return {"shape_names": names, **tensors}
