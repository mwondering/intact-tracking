import importlib.util
from pathlib import Path

from tensorboard.compat.proto import summary_pb2
from torch.utils.tensorboard import SummaryWriter

spec = importlib.util.spec_from_file_location(
    "sync_preview_wandb", Path(__file__).resolve().parents[1] / "scripts/sync_preview_wandb.py",
)
sync = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sync)


class FakeRun:
    def __init__(self):
        self.summary, self.rows, self.axes = {}, [], {}

    def define_metric(self, name, step_metric):
        self.axes[name] = step_metric

    def log(self, payload):
        self.rows.append(payload)


def test_tensorboard_tail_preserves_axes_and_does_not_duplicate_events(tmp_path):
    tail = object.__new__(sync.Tail)
    tail.directory = tmp_path
    tail.state = {"cursors": {}}
    tail.loaders, tail.pending, tail.defined = {}, {}, set()
    tail.run = FakeRun()
    tail.steps_per_update = 4096 * 24
    writer = SummaryWriter(str(tmp_path))
    writer.add_scalar("Metrics/motion/error_body_pos", 0.05, 3)
    writer.add_scalar("Train/mean_reward", 100.0, 3)
    writer.add_scalar("Train/mean_reward/time", 100.0, 42)
    writer.add_scalar("Loss/bad_value", float("nan"), 3)
    writer.flush()
    tail.read()
    assert tail.flush(force=True) == 2
    assert tail.run.axes["Train/mean_reward/time"] == "elapsed_seconds"
    assert tail.run.axes["Metrics/motion/error_body_pos"] == "iteration"
    assert any(row.get("elapsed_seconds") == 42 for row in tail.run.rows)
    assert any(row.get("iteration") == 3 and row["real_transitions"] == 4 * 98304
               for row in tail.run.rows)
    assert tail.run.summary["sync/last_iteration"] == 3
    tail.read()
    assert tail.flush(force=True) == 0
    # Re-open the file, as on a sidecar restart. Saved per-tag cursors skip old events.
    tail.loaders.clear()
    tail.read()
    assert tail.flush(force=True) == 0
    writer.add_scalar("Metrics/motion/error_body_pos", 0.04, 4)
    writer.flush()
    tail.read()
    assert tail.flush(force=True) == 1
    assert tail.run.rows[-1]["iteration"] == 4
    writer.close()


def test_scalar_filter_and_clock_selection():
    assert sync.scalar_value(summary_pb2.Summary.Value(simple_value=1.5)) == 1.5
    assert sync.scalar_value(summary_pb2.Summary.Value()) is None
    assert sync.metric_axis("Perf/collection_time") == "iteration"
    assert sync.metric_axis("Train/mean_episode_length/time") == "elapsed_seconds"


def test_multimotion_config_does_not_require_a_single_file():
    arguments = {key: 1 for key in ("num_envs", "seed", "rollout_steps", "iterations", "actor_lr",
                                    "critic_lr", "critic_warmup_updates", "epochs", "mini_batches")}
    metadata = {"arguments": arguments, "variant": "preview", "physics_mode": "dr",
                "motion_file": None, "motion_path": "/data/full", "motion_count": 129827,
                "dr_profile": "hands-shins-2-4kg", "extra_input_dim": 1065, "input_audit": {},
                "reward_contract": {"sha256": "unchanged"}, "reward_changes": {}}
    config = sync.run_config(metadata, Path("preview_full"))
    assert config["motion"] == "full" and config["motion_count"] == 129827
    assert config["dr_profile"] == "hands-shins-2-4kg"
