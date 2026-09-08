from __future__ import annotations

import torch
import pytest

from hbc_lab.learning.squashed_gaussian import (
    SquashedGaussianActorCritic,
    SquashedGaussianPPO,
    _FiniteAdam,
)


def _make_policy() -> SquashedGaussianActorCritic:
    observations = {
        "policy": torch.zeros(8, 12),
        "critic": torch.zeros(8, 15),
    }
    return SquashedGaussianActorCritic(
        observations,
        {"policy": ["policy"], "critic": ["critic"]},
        19,
        actor_hidden_dims=[32, 16],
        critic_hidden_dims=[32, 16],
        activation="elu",
        init_noise_std=0.3,
        noise_std_type="log",
    )


def test_latent_log_prob_is_finite_and_inference_action_is_bounded():
    policy = _make_policy()
    observations = {
        "policy": torch.randn(8, 12),
        "critic": torch.randn(8, 15),
    }

    latent_action = policy.act(observations)
    inference_action = policy.act_inference(observations)
    log_prob = policy.get_actions_log_prob(latent_action)

    assert torch.all(inference_action.abs() < 1.0)
    assert torch.isfinite(log_prob).all()


def test_ppo_stores_latent_action_but_returns_bounded_environment_action():
    policy = _make_policy()
    algorithm = SquashedGaussianPPO(policy, learning_rate=5.0e-4, schedule="fixed")
    observations = {
        "policy": torch.randn(8, 12),
        "critic": torch.randn(8, 15),
    }

    environment_action = algorithm.act(observations)

    assert torch.all(environment_action.abs() < 1.0)
    assert torch.allclose(environment_action, torch.tanh(algorithm.transition.actions))


def test_noise_standard_deviation_is_projected_to_configured_range():
    policy = _make_policy()
    observations = {"policy": torch.zeros(8, 12), "critic": torch.zeros(8, 15)}

    with torch.no_grad():
        policy.log_std.fill_(10.0)
    policy.act(observations)
    assert torch.allclose(policy.action_std, torch.full_like(policy.action_std, 0.4))

    with torch.no_grad():
        policy.log_std.fill_(-10.0)
    policy.act(observations)
    assert torch.allclose(policy.action_std, torch.full_like(policy.action_std, 0.08))


def test_optimizer_rejects_non_finite_gradients():
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = _FiniteAdam([parameter], lr=1.0e-3)
    parameter.grad = torch.tensor([float("nan")])

    with pytest.raises(FloatingPointError, match="gradient"):
        optimizer.step()


def test_actor_wrapper_exposes_first_linear_layer_for_onnx_exporter():
    policy = _make_policy()

    assert isinstance(policy.actor[0], torch.nn.Linear)
    assert policy.actor[0].in_features == 12
