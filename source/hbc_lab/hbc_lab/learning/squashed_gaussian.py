"""Bounded Gaussian policy extensions for RSL-RL.

RSL-RL resolves policy and algorithm class names with ``eval`` inside its
``on_policy_runner`` module.  ``register_rsl_rl_extensions`` adds these local
classes to that namespace without modifying the installed RSL-RL package.
"""

from __future__ import annotations

import math

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic


class _SquashedActor(torch.nn.Module):
    """Keep the latent actor available while exporting bounded inference."""

    def __init__(self, latent_actor: torch.nn.Module):
        super().__init__()
        self.latent_actor = latent_actor

    def pre_tanh(self, observations: torch.Tensor) -> torch.Tensor:
        return self.latent_actor(observations)

    def __getitem__(self, index: int) -> torch.nn.Module:
        """Expose Sequential-style indexing expected by IsaacLab's ONNX exporter."""
        return self.latent_actor[index]

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.pre_tanh(observations))


class _FiniteAdam(torch.optim.Adam):
    """Fail before an optimizer step can write non-finite policy parameters."""

    def step(self, closure=None):
        for group in self.param_groups:
            for parameter in group["params"]:
                if not torch.isfinite(parameter).all():
                    raise FloatingPointError("Non-finite policy parameter detected before Adam.step().")
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError("Non-finite policy gradient detected before Adam.step().")
        return super().step(closure=closure)


class SquashedGaussianActorCritic(ActorCritic):
    """Gaussian latent policy with bounded deterministic environment actions."""

    def __init__(
        self,
        *args,
        min_noise_std: float = 0.08,
        max_noise_std: float = 0.40,
        actor_output_scale: float = 0.01,
        **kwargs,
    ):
        # IsaacLab 5.1 adds this policy option, while the installed RSL-RL
        # ActorCritic currently ignores it. Keep the extension compatible with
        # both the local and JD configurations without forwarding a noisy kwarg.
        kwargs.pop("state_dependent_std", None)
        if min_noise_std <= 0.0 or max_noise_std <= min_noise_std:
            raise ValueError(
                f"Expected 0 < min_noise_std < max_noise_std, got {min_noise_std}, {max_noise_std}"
            )
        if actor_output_scale <= 0.0:
            raise ValueError(f"actor_output_scale must be positive, got {actor_output_scale}")
        if kwargs.get("noise_std_type", "scalar") != "log":
            raise ValueError("SquashedGaussianActorCritic requires noise_std_type='log'")

        super().__init__(*args, **kwargs)
        self.min_log_std = math.log(min_noise_std)
        self.max_log_std = math.log(max_noise_std)

        latent_actor = self.actor
        output_layer = next(
            module for module in reversed(list(latent_actor.modules())) if isinstance(module, torch.nn.Linear)
        )
        torch.nn.init.orthogonal_(output_layer.weight, gain=actor_output_scale)
        torch.nn.init.zeros_(output_layer.bias)
        self.actor = _SquashedActor(latent_actor)

    def update_distribution(self, obs: torch.Tensor) -> None:
        with torch.no_grad():
            self.log_std.clamp_(self.min_log_std, self.max_log_std)
        latent_mean = self.actor.pre_tanh(obs)
        std = torch.exp(self.log_std).expand_as(latent_mean)
        if not torch.isfinite(latent_mean).all() or not torch.isfinite(std).all():
            raise FloatingPointError("Non-finite latent action distribution detected.")
        self.distribution = torch.distributions.Normal(latent_mean, std)


class SquashedGaussianPPO(PPO):
    """Optimize latent Gaussian actions and squash only the actions sent to the environment."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer = _FiniteAdam(self.policy.parameters(), lr=self.learning_rate)

    def act(self, obs):
        latent_actions = super().act(obs)
        actions = torch.tanh(latent_actions)
        latent_mean = self.policy.action_mean.detach()
        bounded_mean = torch.tanh(latent_mean)
        self._action_diagnostics = {
            "Policy/latent_mean_abs": latent_mean.abs().mean(),
            "Policy/latent_mean_abs_max": latent_mean.abs().max(),
            "Policy/latent_action_abs": latent_actions.detach().abs().mean(),
            "Policy/latent_action_abs_max": latent_actions.detach().abs().max(),
            "Policy/bounded_mean_abs": bounded_mean.abs().mean(),
            "Policy/executed_action_abs": actions.detach().abs().mean(),
            "Policy/action_boundary_095_ratio": (actions.detach().abs() > 0.95).float().mean(),
            "Policy/action_retention_ratio": (1.0 - bounded_mean.square()).mean(),
            "Policy/posture_boundary_095_ratio": (
                actions.detach()[:, 3:5].abs() > 0.95
            ).float().mean(),
        }
        self._record_multi_geometry_observation_diagnostics(obs)
        return actions

    def _record_multi_geometry_observation_diagnostics(self, obs) -> None:
        """Record the source of finite observation spikes in scene-aware policies."""
        actor_obs = self.policy.get_actor_obs(obs)
        actor_obs = self.policy.actor_obs_normalizer(actor_obs)
        if actor_obs.ndim != 2 or actor_obs.shape[-1] not in (1685, 4432):
            return

        detached_obs = actor_obs.detach()
        blocks = {
            "base_ang_vel": detached_obs[:, 0:30],
            "projected_gravity": detached_obs[:, 30:60],
            "task": detached_obs[:, 60:230],
            "grip": detached_obs[:, 230:250],
            "command_state": detached_obs[:, 250:480],
            "execution": detached_obs[:, 480:1170],
            "last_action": detached_obs[:, 1170:1360],
            "environment": detached_obs[:, 1360:],
        }
        self._action_diagnostics["Observation/abs_max"] = detached_obs.abs().max()
        self._action_diagnostics["Observation/over_10_ratio"] = (detached_obs.abs() > 10.0).float().mean()
        for name, values in blocks.items():
            self._action_diagnostics[f"Observation/{name}_abs_max"] = values.abs().max()

        execution = blocks["execution"].reshape(-1, 10, 69)
        execution_fields = {
            "base_lin_vel": execution[:, :, 0:3],
            "upper_joint_pos": execution[:, :, 3:18],
            "current_hand_pose": execution[:, :, 18:36],
            "hand_pose_error": execution[:, :, 36:48],
            "posture_error": execution[:, :, 48:50],
            "command_rate": execution[:, :, 50:69],
        }
        for name, values in execution_fields.items():
            self._action_diagnostics[f"Observation/execution_{name}_abs_max"] = values.abs().max()

    def process_env_step(self, obs, rewards, dones, extras):
        log = extras.setdefault("log", {})
        log.update(self._action_diagnostics)
        super().process_env_step(obs, rewards, dones, extras)


def register_rsl_rl_extensions() -> None:
    """Expose local classes to RSL-RL's class-name resolver."""
    import rsl_rl.runners.on_policy_runner as on_policy_runner

    on_policy_runner.SquashedGaussianActorCritic = SquashedGaussianActorCritic
    on_policy_runner.SquashedGaussianPPO = SquashedGaussianPPO
