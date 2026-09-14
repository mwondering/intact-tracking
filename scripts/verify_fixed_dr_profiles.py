"""Materialize and verify each complete DR at two batch sizes and across resets."""

import argparse
import copy
import json
from pathlib import Path
import time

import torch
from mjlab.envs import ManagerBasedRlEnv

from intact_tracking.environment.runtime import prepare_rollout
from intact_tracking.fixed_dr_profiles import configure_fixed_dr, audit_fixed_dr, STATIC_EVENTS, file_sha256
from intact_tracking.limb_context_protocol import TRACKER


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--motion", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    bank = json.loads(args.bank.read_text())
    evidence = {"started_at": time.time(), "profiles": [], "batch_sizes": [8, 17]}
    for profile in bank["profiles"]:
        audits = []
        for count in evidence["batch_sizes"]:
            prepared = prepare_rollout(checkpoint_file=TRACKER, num_envs=count,
                                       motion_file=args.motion, motion_path=None)
            cfg = copy.deepcopy(prepared.env)
            cfg.seed = 900001 + count
            contract = configure_fixed_dr(cfg, cfg.seed, bank_path=args.bank, profile_id=profile["id"])
            env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
            try:
                initial = audit_fixed_dr(env, contract)
                # A startup DR call must neither depend on nor consume the motion/noise RNG.
                for name in STATIC_EVENTS:
                    term = env.event_manager.get_term_cfg(name)
                    cpu, cuda = torch.get_rng_state(), torch.cuda.get_rng_state()
                    term.func(env, None, **term.params)
                    assert torch.equal(cpu, torch.get_rng_state())
                    assert torch.equal(cuda, torch.cuda.get_rng_state())
                with torch.inference_mode():
                    env.reset()
                    for _ in range(5):
                        env.step(torch.zeros(count, 29, device=env.device))
                    env.reset()
                final = audit_fixed_dr(env, contract)
                assert initial["fixed_dr"]["physics_fingerprint"] == final["fixed_dr"]["physics_fingerprint"]
                audits.append(final)
            finally:
                env.close()
                del env
                torch.cuda.empty_cache()
        fingerprints = {row["fixed_dr"]["physics_fingerprint"] for row in audits}
        assert len(fingerprints) == 1, "The DR profile depends on world count / non-DR RNG seed"
        profile["physics_fingerprint"] = fingerprints.pop()
        profile["prototype_fields"] = audits[0]["fixed_dr"]["prototype_fields"]
        evidence["profiles"].append({"id": profile["id"], "name": profile["name"], "audits": audits,
                                      "batch_size_seed_and_reset_invariant": True, "preserves_rng": True})
        print(json.dumps({"verified_profile": profile["id"], "name": profile["name"],
                          "fingerprint": profile["physics_fingerprint"]}), flush=True)
    assert len({row["physics_fingerprint"] for row in bank["profiles"]}) == 8
    args.bank.write_text(json.dumps(bank, indent=2) + "\n")
    evidence.update(passed=True, finished_at=time.time(), bank_sha256=file_sha256(args.bank))
    args.output.write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    main()
