import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    REPO_ROOT
    / "source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/mdp/object_mass_curriculum.py"
)
EVENTS_PATH = (
    REPO_ROOT
    / "source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/mdp/events.py"
)
ENV_CFG_PATH = (
    REPO_ROOT
    / "source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/config/g1_dex1_env_cfg.py"
)
FLAT_ENV_CFG_PATH = (
    REPO_ROOT
    / "source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/config/flat_env_cfg.py"
)
ENV_PATH = REPO_ROOT / "source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/config/g1_dex1_env.py"
OBJECTS_PATH = REPO_ROOT / "source/hbc_lab/hbc_lab/assets/objects.py"


def _read(path: Path) -> str:
    return path.read_text()


def _load_mass_curriculum_module():
    spec = importlib.util.spec_from_file_location("g1_dex1_object_mass_curriculum_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_balanced_active_hand_progress_uses_weaker_hand_for_mixed_training():
    module = _load_mass_curriculum_module()

    left_mean, right_mean, balanced = module.balanced_active_hand_progress(
        torch.tensor([0.8, 0.6, 0.1, 0.2]),
        torch.tensor([0, 0, 1, 1]),
    )

    assert torch.allclose(left_mean, torch.tensor(0.7))
    assert torch.allclose(right_mean, torch.tensor(0.15))
    assert torch.allclose(balanced, torch.tensor(0.15))


def test_balanced_active_hand_progress_preserves_single_hand_curriculum():
    module = _load_mass_curriculum_module()

    left_mean, right_mean, balanced = module.balanced_active_hand_progress(
        torch.tensor([0.8, 0.6]),
        torch.tensor([1, 1]),
    )

    assert torch.allclose(left_mean, torch.tensor(0.7))
    assert torch.allclose(right_mean, torch.tensor(0.7))
    assert torch.allclose(balanced, torch.tensor(0.7))


def test_object_mass_curriculum_defines_single_expression_log_schedule():
    source = _read(MODULE_PATH)

    assert "def object_mass_curriculum_parameters" in source
    assert "def sample_object_masses" in source
    assert "start_mass: float = 10.0" in source
    assert "ref_w: float = 0.10" in source
    assert "ref_mass: float = 5.0" in source
    assert "anchor_w: float = 0.20" in source
    assert "anchor_mass: float = 1.0" in source
    assert "final_mass: float = 0.5" in source
    assert "ref_log_std: float = 0.15" in source
    assert "final_log_std: float = 0.45" in source
    assert "torch.log(torch.as_tensor(start_mass" in source
    assert "torch.log(torch.as_tensor(ref_mass" in source
    assert "torch.log(torch.as_tensor(anchor_mass" in source
    assert "torch.log(torch.as_tensor(final_mass" in source
    assert "ref_decay_target = -torch.log(ref_log_ratio)" in source
    assert "anchor_decay_target = -torch.log(anchor_log_ratio)" in source
    assert "mass_power = torch.log(anchor_decay_target / ref_decay_target) / torch.log(anchor_w_t / ref_w_t)" in source
    assert "mass_decay = ref_decay_target / torch.pow(ref_w_t, mass_power)" in source
    assert "log_center = log_final + (log_start - log_final) * torch.exp(-mass_decay * torch.pow(w_manip, mass_power))" in source
    assert "std_decay = -torch.log(std_ratio) / ref_w_t" in source
    assert "log_std = final_log_std_t * (1.0 - torch.exp(-std_decay * w_manip))" in source
    assert "torch.exp(log_center + noise * log_std)" in source
    assert "torch.clamp(mass, min_mass, max_mass)" in source


def test_g1_dex1_reset_object_applies_mass_curriculum_to_physx_masses():
    events_source = _read(EVENTS_PATH)
    env_cfg_source = _read(ENV_CFG_PATH)
    env_source = _read(ENV_PATH)
    objects_source = _read(OBJECTS_PATH)

    assert "mass_props=sim_utils.MassPropertiesCfg(mass=10.0)" in objects_source
    assert "object_mass_curriculum_enabled: bool = True" in env_cfg_source
    assert "object_mass_w_manip_ema_alpha: float = 0.01" in env_cfg_source
    assert "object_mass_start_w: float = 0.0" in env_cfg_source
    assert "object_mass_ref_w: float = 0.10" in env_cfg_source
    assert "object_mass_start_mass: float = 10.0" in env_cfg_source
    assert "object_mass_ref_mass: float = 5.0" in env_cfg_source
    assert "object_mass_anchor_w: float = 0.20" in env_cfg_source
    assert "object_mass_anchor_mass: float = 1.0" in env_cfg_source
    assert "object_mass_max: float = 25.0" in env_cfg_source
    assert "self.object_mass_w_manip_ema" in env_source
    assert "self.object_mass_curriculum_level" in env_source
    assert "DRC/object_mass_curriculum_level" in env_source
    assert "DRC/object_mass_mean" in env_source
    assert "balanced_active_hand_progress" in env_source
    assert "DRC/object_mass_w_manip_left_mean" in env_source
    assert "DRC/object_mass_w_manip_right_mean" in env_source
    assert "DRC/object_mass_w_manip_balanced" in env_source
    update_source = env_source.split("    def _update_object_mass_curriculum", maxsplit=1)[1].split(
        "    def _check_success", maxsplit=1
    )[0]
    assert "self.W_manip.detach().mean()" not in update_source
    assert "sample_object_masses" in events_source
    assert "env.object_mass_curriculum_level" in events_source
    assert "anchor_w=env.cfg.object_mass_anchor_w" in events_source
    assert "anchor_mass=env.cfg.object_mass_anchor_mass" in events_source
    assert "root_physx_view.get_masses()" in events_source
    assert "root_physx_view.set_masses(masses, env_ids.cpu())" in events_source


def test_g1_dex1_play_starts_at_final_object_mass_curriculum_level():
    flat_env_cfg_source = _read(FLAT_ENV_CFG_PATH)
    play_cfg_source = flat_env_cfg_source.split("class G1Dex1HierDrcFlatPlayEnvCfg", maxsplit=1)[1]

    assert "self.object_mass_start_w = 1.0" in play_cfg_source
