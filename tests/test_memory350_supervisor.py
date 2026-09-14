"""Guard against repeating training or accepting stale completion during adoption."""

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("memory350_supervisor", ROOT / "scripts/run_memory350_ppo_experiment.py")
SUPERVISOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUPERVISOR)


class AttachmentLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / ".runtime/memory350_ppo_v1")
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)
        (self.output / "checkpoint_update_005000.pt").touch()

    def observer(self):
        observer = object.__new__(SUPERVISOR.AttachedTrainingProcess)
        observer.pid = os.getpid()
        observer.start_ticks = "test-identity"
        observer.output = self.output
        observer.returncode = None
        observer.exit_observed_at = time.monotonic() - 31
        return observer

    def completion(self, updates=5000, finite=True):
        data = {"complete": updates == 5000, "target_updates": 5000, "completed_updates": updates,
                "stopped": False, "planned_sampling_transition": updates == 1000,
                "distributed_parameter_agreement": {"passed": True, "world_size": 2,
                    "ranks": [{"rank": rank, "completed_updates": updates, "finite": finite} for rank in (0, 1)]}}
        (self.output / "completion.json").write_text(json.dumps(data))

    def test_live_training_cannot_finish_from_completion_file_alone(self):
        self.completion()
        with patch.object(SUPERVISOR, "process_identity", return_value={"state": "S", "start_ticks": "test-identity"}):
            self.assertIsNone(self.observer().poll())

    def test_exited_process_requires_fresh_finite_5000_update_completion(self):
        with patch.object(SUPERVISOR, "process_identity", return_value=None):
            for updates, finite, expected in ((1000, True, 1), (5000, False, 1), (5000, True, 0)):
                with self.subTest(updates=updates, finite=finite):
                    self.completion(updates, finite)
                    self.assertEqual(self.observer().poll(), expected)

    def test_reused_pid_does_not_keep_old_training_alive(self):
        self.completion(1000)
        with patch.object(SUPERVISOR, "process_identity", return_value={"state": "S", "start_ticks": "different-identity"}):
            self.assertEqual(self.observer().poll(), 1)

    def test_attach_rejects_wrong_command_without_touching_process(self):
        job = {"phase": "adaptive", "pid": os.getpid(), "command": ["unrelated-command"], "gpus": [0, 1]}
        with self.assertRaisesRegex(RuntimeError, "command changed"):
            SUPERVISOR.AttachedTrainingProcess(self.output, "baseline", job)
        self.assertIsNotNone(SUPERVISOR.process_identity(os.getpid()))


if __name__ == "__main__":
    unittest.main()
