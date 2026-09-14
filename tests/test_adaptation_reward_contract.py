from __future__ import annotations

import copy
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from intact_tracking.adaptation_reward_contract import (
    REWARD_ARGUMENTS,
    assert_fixed_reward_checkpoint,
    assert_original_reward_arguments,
    assert_rewards_unchanged,
    capture_original_rewards,
)


def reward_a(env):
    return env.value


def reward_b(env):
    return env.value + 1


@dataclass
class Term:
    func: object
    weight: float
    params: dict


def configuration():
    return SimpleNamespace(
        rewards={"tracking": Term(reward_a, 2.0, {"scale": 0.1, "ids": slice(None)})},
        decimation=4, sim=SimpleNamespace(mujoco=SimpleNamespace(timestep=0.005)),
    )


@pytest.mark.parametrize("name", REWARD_ARGUMENTS)
def test_every_reward_override_rejected(name):
    values = dict(REWARD_ARGUMENTS)
    values[name] = "forbidden"
    with pytest.raises(ValueError, match="locked"):
        assert_original_reward_arguments(SimpleNamespace(**values))


def test_original_arguments_and_exact_definition():
    assert_original_reward_arguments(SimpleNamespace(**REWARD_ARGUMENTS))
    cfg = configuration()
    contract = capture_original_rewards(cfg)
    assert_rewards_unchanged(contract, copy.deepcopy(cfg))
    assert contract["definition"]["simulation_dt"] * contract["definition"]["decimation"] == 0.02


@pytest.mark.parametrize("change", ["weight", "function", "params", "added", "removed", "dt", "decimation"])
def test_configuration_mutations_are_detected(change):
    cfg = configuration()
    contract = capture_original_rewards(cfg)
    if change == "weight":
        cfg.rewards["tracking"].weight = 3
    elif change == "function":
        cfg.rewards["tracking"].func = reward_b
    elif change == "params":
        cfg.rewards["tracking"].params["scale"] = 0.2
    elif change == "added":
        cfg.rewards["failure"] = Term(reward_b, -1000, {})
    elif change == "removed":
        cfg.rewards.clear()
    elif change == "dt":
        cfg.sim.mujoco.timestep = 0.01
    else:
        cfg.decimation = 8
    with pytest.raises(ValueError, match="changed"):
        assert_rewards_unchanged(contract, cfg)


def test_checkpoint_lineage_fails_closed():
    contract = capture_original_rewards(configuration())
    good = {"residual_policy": {"reward_contract": contract, "reward_changes": {}}}
    assert_fixed_reward_checkpoint(good, contract)
    for bad in ({}, {"residual_policy": {"reward_changes": {}}}):
        with pytest.raises(ValueError, match="lineage"):
            assert_fixed_reward_checkpoint(bad, contract)
    shaped = copy.deepcopy(good)
    shaped["residual_policy"]["reward_changes"]["tracking"] = {"from": 1, "to": 4}
    with pytest.raises(ValueError, match="lineage"):
        assert_fixed_reward_checkpoint(shaped, contract)
    mismatch = copy.deepcopy(good)
    mismatch["residual_policy"]["reward_contract"]["sha256"] = "wrong"
    with pytest.raises(ValueError, match="lineage"):
        assert_fixed_reward_checkpoint(mismatch, contract)
