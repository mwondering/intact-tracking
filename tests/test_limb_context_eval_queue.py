import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("limb_eval_queue", SCRIPTS / "evaluate_limb_context_experiment.py")
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


def test_finished_controls_can_be_evaluated_before_latent_checkpoints_exist(tmp_path, monkeypatch):
    (tmp_path / "eval").mkdir()
    tracker = tmp_path / "tracker.pt"
    tracker.write_bytes(b"source")
    monkeypatch.setattr(queue, "TRACKER", str(tracker))
    control = queue.ppo_directory(tmp_path) / "baseline_121"
    control.mkdir(parents=True)
    (control / "checkpoint_final.pt").write_bytes(b"trained_control")
    case = {"name": "iid_full", "manifests": [str(tmp_path / "motions.txt")],
            "repeats": 1, "fixed_masses": None, "paired_starts": False}
    protocol = {"cases": [case, {**case, "name": "latent_intervention"}], "seed": 20001, "steps": 500}
    jobs = queue.evaluation_jobs(tmp_path, protocol, ("frozen", "baseline_121"))
    assert len(jobs) == 2
    assert {j["policy"] for j in jobs} == {"frozen", "baseline_121"}
    assert all(j["case"] == "iid_full" and j["seed"] == 20001 for j in jobs)
    output = Path(jobs[1]["output"])
    output.parent.mkdir()
    output.write_text(json.dumps({"checkpoint_sha256": "different-checkpoint",
                                 "seed": 20001, "latent_intervention": "correct"}))
    with pytest.raises(ValueError, match="different checkpoint/protocol"):
        queue.evaluation_jobs(tmp_path, protocol, ("frozen", "baseline_121"))
