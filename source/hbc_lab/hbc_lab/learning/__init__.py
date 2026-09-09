"""Project-local extensions to external reinforcement-learning libraries."""

from .squashed_gaussian import (
    SceneAwareSquashedGaussianActorCritic,
    SquashedGaussianActorCritic,
    SquashedGaussianPPO,
    register_rsl_rl_extensions,
)

__all__ = [
    "SceneAwareSquashedGaussianActorCritic",
    "SquashedGaussianActorCritic",
    "SquashedGaussianPPO",
    "register_rsl_rl_extensions",
]
