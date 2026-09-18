"""Wait for one atomic formal checkpoint, then audit both PPO action modes."""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--update", type=int, required=True)
    parser.add_argument("--gpus", type=int, nargs=2, default=[6, 7])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run = args.run.resolve()
    checkpoint = run / "latent" / f"checkpoint_update_{args.update:06d}.pt"
    output = run / f"expert_mapping_u{args.update}"
    output.mkdir(exist_ok=True)
    status_file = output / "audit_status.json"
    if (output / "provenance.json").exists():
        raise FileExistsError("Refusing to overwrite an existing audit")
    status = {"checkpoint": str(checkpoint), "expected_updates": args.update,
              "status": "waiting", "pid": os.getpid()}
    save_json(status_file, status)
    next_report = 0
    deadline = time.monotonic() + 5400
    while not checkpoint.exists():
        now = time.monotonic()
        if now > deadline:
            raise TimeoutError("Formal checkpoint did not arrive within 90 minutes")
        if now >= next_report:
            progress = json.loads((run / "latent/progress.json").read_text())
            status.update(live_completed_updates=progress["completed_updates"],
                          last_checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            save_json(status_file, status)
            print(json.dumps(status), flush=True)
            if time.time() - progress["unix_time"] > 600:
                raise RuntimeError("Training progress has been stale for over ten minutes")
            next_report = now + 55
        time.sleep(5)

    import torch
    from intact_tracking.limb_context_protocol import local_process_environment
    from intact_tracking.payload_prototype_moe import VERSION

    state = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    meta = state["residual_policy"]
    if state["completed_updates"] != args.update or meta["version"] != VERSION:
        raise ValueError("Checkpoint update or architecture does not match the requested audit")
    if meta["arguments"]["route_mode"] != "latent":
        raise ValueError("This audit is specifically for the fixed latent router")
    prototype = run / "calibration/prototypes.pt"
    proto = torch.load(prototype, map_location="cpu", weights_only=False)
    if sha256(prototype) != meta["prototype_sha256"]:
        raise ValueError("Prototype file changed since this checkpoint")
    if sha256(meta["context_checkpoint"]) != proto["metadata"]["context_sha256"]:
        raise ValueError("Context encoder differs from the calibrated encoder")
    torch.testing.assert_close(state["actor_state_dict"]["gate.centers"], proto["centers"], rtol=0, atol=0)
    torch.testing.assert_close(state["actor_state_dict"]["gate.temperature"], torch.tensor(proto["temperature"]), rtol=0, atol=0)
    provenance = {"checkpoint": str(checkpoint), "completed_updates": state["completed_updates"],
                  "checkpoint_sha256": sha256(checkpoint), "prototype_sha256": sha256(prototype),
                  "context_checkpoint": meta["context_checkpoint"], "context_sha256": proto["metadata"]["context_sha256"],
                  "temperature": proto["temperature"], "formal_training_run": str(run / "latent"),
                  "motion_manifest_sha256": sha256(run / "temperature_probe_motions.txt"),
                  "script_sha256": {str(p.relative_to(root)): sha256(p) for p in
                      [Path(__file__), root / "scripts/audit_payload_router_temperature.py",
                       root / "scripts/report_payload_expert_mapping.py"]},
                  "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    save_json(output / "provenance.json", provenance)
    del state, proto
    jobs = []
    for mode, gpu in zip(("mean", "sample"), args.gpus, strict=True):
        command = [sys.executable, "-B", "-u", "scripts/audit_payload_router_temperature.py",
                   "--checkpoint", str(checkpoint), "--motion-manifest", str(run / "temperature_probe_motions.txt"),
                   "--output", str(output / f"runtime_{mode}.json"), "--mapping", "--action-mode", mode]
        env = dict(os.environ, **local_process_environment())
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                   SP_TRACKING_MULTIMOTION_MANIFEST_DIR=str(root / ".runtime/limb_context/manifests"),
                   MJLAB_BOOTSTRAP_DEBUG_DIR=str(root / ".runtime/limb_context/bootstrap"))
        with (output / f"runtime_{mode}.log").open("xb") as log:
            process = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        save_json(output / f"process_{mode}.json", {"pid": process.pid, "command": command, "gpu": gpu})
        jobs.append((mode, process))
    status.update(status="evaluating", provenance=str(output / "provenance.json"))
    save_json(status_file, status)
    print(json.dumps(status), flush=True)
    while any(process.poll() is None for _, process in jobs):
        time.sleep(5)
    status.update(status="succeeded" if all(process.returncode == 0 for _, process in jobs) else "failed",
                  returncodes={mode: process.returncode for mode, process in jobs},
                  finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save_json(status_file, status)
    print(json.dumps(status), flush=True)
    if status["status"] != "succeeded":
        raise RuntimeError("Inspect the diagnostic logs for the failed action mode")


if __name__ == "__main__":
    main()
