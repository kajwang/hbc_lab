from __future__ import annotations

from isaaclab.utils import configclass

from ..mdp.events import reset_multishape_object_and_support_platforms
from ..mdp.object_shape_bps import (
    ALL_SHAPE_NAMES,
    BPS_DATA_PATH,
    GRASP_REF_ALL_SHAPE_NAMES,
    GRASP_REF_TRAIN_SHAPE_NAMES,
    MULTISHAPE_SIZE_FACTORS,
    TRAIN_SHAPE_NAMES,
    format_shape_assignment,
)
from ..mdp.observations import G1Dex1HierDrcMultiShapeObservationsCfg
from ..mdp.observations import G1Dex1HierDrcMultiShapeGraspRefObservationsCfg
from ..mdp.scenes import (
    G1Dex1HierDrcMultiShapeBpsEvalSceneCfg,
    G1Dex1HierDrcMultiShapeBpsSceneCfg,
    G1Dex1HierDrcMultiShapeGraspRefEvalSceneCfg,
    G1Dex1HierDrcMultiShapeGraspRefSceneCfg,
    _multishape_object_cfg,
)
from .flat_env_cfg import G1Dex1HierDrcFlatEnvCfg


@configclass
class G1Dex1HierDrcMultiShapeEnvCfg(G1Dex1HierDrcFlatEnvCfg):
    scene: G1Dex1HierDrcMultiShapeBpsSceneCfg = G1Dex1HierDrcMultiShapeBpsSceneCfg(
        num_envs=4096,
        env_spacing=10.0,
        replicate_physics=False,
    )
    observations: G1Dex1HierDrcMultiShapeObservationsCfg = G1Dex1HierDrcMultiShapeObservationsCfg()
    object_shape_names: tuple[str, ...] = TRAIN_SHAPE_NAMES
    object_size_scale_factors: tuple[float, ...] = MULTISHAPE_SIZE_FACTORS
    object_shape_bps_data_path: str = str(BPS_DATA_PATH)
    object_shape_bps_enabled: bool = True

    def apply_object_shape_encoding(self) -> None:
        # Keep the observation terms in both ablation arms so actor and critic
        # architectures match. The observation function zeroes this tensor when disabled.
        return

    def apply_debug_visualization(self) -> None:
        super().apply_debug_visualization()
        # The USD root is not the geometry center for every asset. The target-region
        # marker already shows the live center supplied to the policy.
        self.scene.object_frame.debug_vis = False

    def __post_init__(self):
        super().__post_init__()
        self.domain_randomization_curriculum_enabled = False
        self.object_mass_curriculum_enabled = True
        self.events.reset_object.func = reset_multishape_object_and_support_platforms


@configclass
class G1Dex1HierDrcMultiShapePlayEnvCfg(G1Dex1HierDrcMultiShapeEnvCfg):
    scene: G1Dex1HierDrcMultiShapeBpsEvalSceneCfg = G1Dex1HierDrcMultiShapeBpsEvalSceneCfg(
        num_envs=27,
        env_spacing=6.0,
        replicate_physics=False,
    )
    object_shape_names: tuple[str, ...] = ALL_SHAPE_NAMES
    object_size_scale_factors: tuple[float, ...] = (1.0,)

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = len(ALL_SHAPE_NAMES)
        self.object_mass_start_w = self.object_mass_anchor_w
        self.enable_debug_visualization = True
        self.platform_pose_debug_vis = True
        print(format_shape_assignment(ALL_SHAPE_NAMES))


@configclass
class G1Dex1HierDrcMultiShapeGraspRefEnvCfg(G1Dex1HierDrcMultiShapeEnvCfg):
    scene: G1Dex1HierDrcMultiShapeGraspRefSceneCfg = G1Dex1HierDrcMultiShapeGraspRefSceneCfg(
        num_envs=4096,
        env_spacing=10.0,
        replicate_physics=False,
    )
    observations: G1Dex1HierDrcMultiShapeGraspRefObservationsCfg = (
        G1Dex1HierDrcMultiShapeGraspRefObservationsCfg()
    )
    object_shape_names: tuple[str, ...] = GRASP_REF_TRAIN_SHAPE_NAMES
    object_size_scale_factors: tuple[float, ...] = (1.0,)
    grasp_reference_enabled: bool = True

    def __post_init__(self):
        super().__post_init__()
        self.object_size_scale_factors = (1.0,)
        self.multishape_platform_has_walls = False
        self.grasp_pose_guidance_enabled = True
        self.object_mass_use_physical_grasp = True
        self.grasp_reference_controls_target = True

    def apply_grasp_candidate_evaluation(self) -> None:
        if self.grasp_candidate_asset_id < 0:
            return
        if self.grasp_candidate_asset_id >= len(GRASP_REF_ALL_SHAPE_NAMES):
            raise ValueError(
                f"grasp_candidate_asset_id must be in [0, {len(GRASP_REF_ALL_SHAPE_NAMES) - 1}], "
                f"got {self.grasp_candidate_asset_id}"
            )
        shape_name = GRASP_REF_ALL_SHAPE_NAMES[self.grasp_candidate_asset_id]
        self.object_shape_names = (shape_name,)
        self.object_size_scale_factors = (1.0,)
        self.scene.object = _multishape_object_cfg((shape_name,), (1.0,))
        self.scene.num_envs = 12
        self.scene.env_spacing = 6.0
        self.grasp_reference_enabled = True
        self.grasp_reference_candidate_indices = ()
        self.grasp_candidate_yaw_count = 12
        self.grasp_reference_select_highest_weight = True
        self.grasp_reference_pose_debug_vis = True
        self.enable_debug_visualization = False
        self.target_pose_debug_vis = False
        self.platform_pose_debug_vis = True
        self.commands.high_level.left_hand_probability = 0.0
        self.events.reset_object.params["pose_range"] = {
            "x": (1.8, 1.8),
            "y": (-0.2, -0.2),
            "yaw": (0.0, 0.0),
        }
        self.events.reset_base.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.events.push_robot = None
        print(f"[Grasp Candidate Play] asset {self.grasp_candidate_asset_id}: {shape_name}")
        for env_id in range(self.grasp_candidate_yaw_count):
            print(f"  env {env_id:02d}: object_yaw={30 * env_id:3d} deg")


@configclass
class G1Dex1HierDrcMultiShapeGraspRefPlayEnvCfg(G1Dex1HierDrcMultiShapeGraspRefEnvCfg):
    scene: G1Dex1HierDrcMultiShapeGraspRefEvalSceneCfg = G1Dex1HierDrcMultiShapeGraspRefEvalSceneCfg(
        num_envs=len(GRASP_REF_ALL_SHAPE_NAMES),
        env_spacing=6.0,
        replicate_physics=False,
    )
    object_shape_names: tuple[str, ...] = GRASP_REF_ALL_SHAPE_NAMES
    object_size_scale_factors: tuple[float, ...] = (1.0,)

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = len(GRASP_REF_ALL_SHAPE_NAMES)
        self.object_mass_start_w = self.object_mass_anchor_w
        self.enable_debug_visualization = True
        self.platform_pose_debug_vis = True
        print(format_shape_assignment(GRASP_REF_ALL_SHAPE_NAMES))


@configclass
class G1Dex1HierDrcMultiShapeObjectCenterControlEnvCfg(G1Dex1HierDrcMultiShapeGraspRefEnvCfg):
    """Equal-dimension control: object center position plus identity orientation."""

    def __post_init__(self):
        super().__post_init__()
        self.grasp_reference_enabled = False
        self.grasp_reference_controls_target = False
        self.grasp_pose_guidance_enabled = False


@configclass
class G1Dex1HierDrcMultiShapeObjectCenterControlPlayEnvCfg(
    G1Dex1HierDrcMultiShapeObjectCenterControlEnvCfg
):
    scene: G1Dex1HierDrcMultiShapeGraspRefEvalSceneCfg = G1Dex1HierDrcMultiShapeGraspRefEvalSceneCfg(
        num_envs=len(GRASP_REF_ALL_SHAPE_NAMES),
        env_spacing=6.0,
        replicate_physics=False,
    )
    object_shape_names: tuple[str, ...] = GRASP_REF_ALL_SHAPE_NAMES
    object_size_scale_factors: tuple[float, ...] = (1.0,)

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = len(GRASP_REF_ALL_SHAPE_NAMES)
        self.object_mass_start_w = self.object_mass_anchor_w
        self.enable_debug_visualization = True
        self.platform_pose_debug_vis = True
        print(format_shape_assignment(GRASP_REF_ALL_SHAPE_NAMES))
