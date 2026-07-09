from pathlib import Path


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
ENV_PATH = REPO_ROOT / "source/hbc_lab/hbc_lab/tasks/manager_based/skill/g1_dex1_hier_drc/config/g1_dex1_env.py"
OBJECTS_PATH = REPO_ROOT / "source/hbc_lab/hbc_lab/assets/objects.py"


def _read(path: Path) -> str:
    return path.read_text()


def test_object_mass_curriculum_defines_single_expression_log_schedule():
    source = _read(MODULE_PATH)

    assert "def object_mass_curriculum_parameters" in source
    assert "def sample_object_masses" in source
    assert "start_mass: float = 20.0" in source
    assert "ref_w: float = 0.10" in source
    assert "ref_mass: float = 5.0" in source
    assert "final_mass: float = 0.5" in source
    assert "ref_log_std: float = 0.15" in source
    assert "final_log_std: float = 0.45" in source
    assert "torch.log(torch.as_tensor(start_mass" in source
    assert "torch.log(torch.as_tensor(ref_mass" in source
    assert "torch.log(torch.as_tensor(final_mass" in source
    assert "mass_decay = -torch.log(mass_ratio) / ref_w_t" in source
    assert "log_center = log_final + (log_start - log_final) * torch.exp(-mass_decay * w_manip)" in source
    assert "std_decay = -torch.log(std_ratio) / ref_w_t" in source
    assert "log_std = final_log_std_t * (1.0 - torch.exp(-std_decay * w_manip))" in source
    assert "torch.exp(log_center + noise * log_std)" in source
    assert "torch.clamp(mass, min_mass, max_mass)" in source


def test_g1_dex1_reset_object_applies_mass_curriculum_to_physx_masses():
    events_source = _read(EVENTS_PATH)
    env_cfg_source = _read(ENV_CFG_PATH)
    env_source = _read(ENV_PATH)
    objects_source = _read(OBJECTS_PATH)

    assert "mass_props=sim_utils.MassPropertiesCfg(mass=20.0)" in objects_source
    assert "object_mass_curriculum_enabled: bool = True" in env_cfg_source
    assert "object_mass_w_manip_ema_alpha: float = 0.01" in env_cfg_source
    assert "object_mass_start_w: float = 0.0" in env_cfg_source
    assert "object_mass_ref_w: float = 0.10" in env_cfg_source
    assert "object_mass_ref_mass: float = 5.0" in env_cfg_source
    assert "object_mass_max: float = 25.0" in env_cfg_source
    assert "self.object_mass_w_manip_ema" in env_source
    assert "self.object_mass_curriculum_level" in env_source
    assert "DRC/object_mass_curriculum_level" in env_source
    assert "DRC/object_mass_mean" in env_source
    assert "sample_object_masses" in events_source
    assert "env.object_mass_curriculum_level" in events_source
    assert "root_physx_view.get_masses()" in events_source
    assert "root_physx_view.set_masses(masses, env_ids.cpu())" in events_source
