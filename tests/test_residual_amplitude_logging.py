import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tensordict import TensorDict

from intact_tracking.cli.memory350_policy_train import attach_json_logger, tracker_wandb_metrics
from intact_tracking.distributed import DistributedContext
from intact_tracking.limb_context_policy import LimbContextResidualActor
from intact_tracking.residual_runner import ResidualOnPolicyRunner


class DiagnosticActor(LimbContextResidualActor):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.residual_mlp = torch.nn.Identity()
        self.residual_output_mode = "unbounded"
        self.residual_scale = 1.
        self.use_dynamics_latent = False

    def _base_features_and_action(self, obs):
        return obs["correction"], obs["base"]


def _worker(rank, directory):
    torch.set_num_threads(1)
    directory = Path(directory)
    dist.init_process_group("gloo", init_method=(directory / "rendezvous").as_uri(), rank=rank, world_size=2)
    try:
        context = DistributedContext(rank, rank, 2, torch.device("cpu"), "gloo")
        corrections = torch.tensor([[[-1., 2.], [0., 1.]], [[-3., 4.], [2., -2.]]])
        scale = torch.tensor([.1, .3])
        actor = DiagnosticActor()
        actor.last_residual_mean = torch.full((2, 2), -999.)  # Stale PPO minibatch cache.
        obs = TensorDict({"correction": corrections[rank], "base": torch.full((2, 2), 2.)}, [2])
        action = SimpleNamespace(_scale=scale)
        env = SimpleNamespace(num_envs=2, unwrapped=SimpleNamespace(
            action_manager=SimpleNamespace(get_term=lambda name: action)))
        runner = SimpleNamespace(alg=SimpleNamespace(get_policy=lambda: actor), env=env,
                                 is_distributed=True, device="cpu", gpu_world_size=2)
        metrics = ResidualOnPolicyRunner._policy_diagnostics(runner, obs)
        expected_rms = float(corrections.square().mean().sqrt())
        assert metrics["residual_action_rms"] == pytest.approx(expected_rms)
        assert metrics["residual_action_abs_mean"] == pytest.approx(float(corrections.abs().mean()))
        assert metrics["residual_action_abs_max"] == 4.  # Not mean of rank maxima = 3.
        assert metrics["residual_to_base_rms_ratio"] == pytest.approx(expected_rms / 2)
        assert metrics["residual_target_rms_rad"] == pytest.approx(float((corrections * scale).square().mean().sqrt()))
        assert metrics["residual_target_abs_max_rad"] == pytest.approx(1.2)
        assert "residual_saturation_fraction" not in metrics
        torch.testing.assert_close(actor.last_residual_mean, corrections[rank])

        runner.cfg = {"num_steps_per_env": 24}
        runner.residual_metadata = {"policy_precision": "fp32"}
        forwarded = []
        runner.logger = SimpleNamespace(log=lambda **kw: forwarded.append(kw),
            process_env_step=lambda *a, **kw: None, writer=None,
            rewbuffer=[1.] if rank == 0 else [3., 5.],
            lenbuffer=[10.] if rank == 0 else [30., 50.],
            ep_extras=[{
                "Metrics/tracking_error": torch.tensor([1., 3.]) if rank == 0 else torch.tensor([9.]),
                "Episode_Reward/tracking": torch.tensor([2., 4.]) if rank == 0 else torch.tensor([10.]),
                "Episode_Termination/fall": float(rank),
                "custom_stat": 5. + 2. * rank,
            }])
        uploaded = []
        wandb = SimpleNamespace(log=lambda payload, step: uploaded.append({"payload": payload, "step": step}))
        attach_json_logger(runner, directory, context, wandb)
        runner.logger.log(loss_dict={**metrics, "value": 1. + 2. * rank, "AuxDR/loss": .25}, it=7,
                          collect_time=1. + rank, learn_time=.1 * (1 + rank),
                          learning_rate=.0001, action_std=torch.ones(2))
        assert forwarded[0]["loss_dict"]["residual_action_abs_max"] == 4.
        if rank == 0:
            assert len(uploaded) == 1
            (directory / "wandb_payload.json").write_text(json.dumps(uploaded[0]))
        else:
            assert not uploaded
    finally:
        dist.destroy_process_group()


def test_global_residual_amplitudes_reach_wandb_json_and_rsl_logger(tmp_path):
    mp.spawn(_worker, args=(str(tmp_path),), nprocs=2, join=True)
    record = json.loads((tmp_path / "metrics.jsonl").read_text())
    logged = json.loads((tmp_path / "wandb_payload.json").read_text())
    assert logged["step"] == logged["payload"]["completed_updates"] == record["completed_updates"] == 8
    payload = logged["payload"]
    assert payload["Residual/mean_rms"] == record["loss"]["residual_action_rms"]
    assert payload["Residual/mean_rms"] == payload["training/loss/residual_action_rms"]
    assert payload["Residual/mean_abs_max"] == 4.
    assert payload["Residual/mean_abs"] == pytest.approx(1.875)
    assert payload["Residual/target_abs_max_rad"] == pytest.approx(1.2)
    assert payload["Residual/output_bounded"] == 0.
    assert payload["Loss/value"] == 2.
    assert payload["AuxDR/loss"] == .25
    assert "Loss/AuxDR/loss" not in payload
    assert payload["Loss/learning_rate"] == .0001
    assert payload["Policy/mean_std"] == 1.
    assert payload["Train/mean_reward"] == 3.
    assert payload["Train/mean_episode_length"] == 30.
    assert payload["Perf/total_fps"] == 43  # Global 96 transitions / slowest rank's 2.2 seconds.
    assert payload["Perf/collection_time"] == 2.
    assert payload["Perf/learning_time"] == pytest.approx(.2)
    assert payload["Metrics/tracking_error"] == pytest.approx(13. / 3)
    assert payload["Episode_Reward/tracking"] == pytest.approx(16. / 3)
    assert payload["Episode_Termination/fall"] == .5
    assert payload["Episode/custom_stat"] == 6.
    assert "Episode/Metrics/tracking_error" not in payload
    assert "Train/mean_reward/time" not in payload


def test_tracker_wandb_names_skip_unavailable_episode_means():
    payload = tracker_wandb_metrics({"loss": {"entropy": -1.}, "collect_seconds": 1.,
        "learn_seconds": 1., "learning_rate": .0001, "action_std": .3,
        "global_num_envs": 4, "mean_reward": None, "mean_episode_length": None}, rollout_steps=24)
    assert "Train/mean_reward" not in payload and "Train/mean_episode_length" not in payload
    assert payload["Loss/entropy"] == -1.
