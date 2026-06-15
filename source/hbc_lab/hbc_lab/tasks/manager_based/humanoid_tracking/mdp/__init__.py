"""MDP terms for HBC humanoid locomotion and future tracking tasks."""

try:
    from isaaclab.envs.mdp import *  # noqa: F401, F403
    from isaaclab_tasks.manager_based.locomotion.velocity.mdp import *  # noqa: F401, F403

    from .commands import *  # noqa: F401, F403
    from .curriculums import *  # noqa: F401, F403
    from .rewards import *  # noqa: F401, F403
except (ModuleNotFoundError, ImportError) as exc:
    missing = getattr(exc, "name", "")
    if missing not in {"torch", "isaaclab", "isaaclab_tasks"} and "isaaclab" not in str(exc):
        raise
