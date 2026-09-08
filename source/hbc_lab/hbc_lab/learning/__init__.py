"""Project-local extensions to external reinforcement-learning libraries."""

from .squashed_gaussian import (
    SquashedGaussianActorCritic,
    SquashedGaussianPPO,
    register_rsl_rl_extensions,
)

__all__ = [
    "SquashedGaussianActorCritic",
    "SquashedGaussianPPO",
    "register_rsl_rl_extensions",
]
