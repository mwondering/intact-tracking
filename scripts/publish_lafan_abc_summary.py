"""Publish aggregate endpoint results to the existing scalar-only W&B runs."""

import argparse
import json
from pathlib import Path


def summary_payload(result, arm, completed_updates):
    payload = {"experiment_status": "training_and_endpoint_evaluation_complete",
               "eligible_endpoint_result": True,
               "evaluation/training_updates": completed_updates,
               "evaluation/scope": "one training seed; three paired evaluation seeds; LaFAN fine-tuning"}
    for row in result["table"]:
        if row["policy"] != arm:
            continue
        for name in ("body_error", "joint_error", "failures", "episodes", "coverage_fraction"):
            if name in row:
                payload[f"evaluation/{row['endpoint']}/{name}"] = row[name]
    if arm == "C":
        for key in ("C_vs_A_nominal", "C_vs_B_hardest"):
            for name in ("C_over_reference", "ratio_ci95_low", "ratio_ci95_high",
                         "C_new_failure_episodes", "C_rescued_failure_episodes"):
                payload[f"evaluation/{key}/{name}"] = result[key][name]
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--upload", action="store_true", help="Otherwise print the proposed aggregate summary")
    args = parser.parse_args()
    result = json.loads((args.root / "comparison.json").read_text())
    state = json.loads((args.root / ".wandb_sync/state.json").read_text())
    payloads = {}
    for arm in "ABC":
        audit = json.loads((args.root / arm / "completion_audit.json").read_text())
        config = json.loads((args.root / arm / "run_config.json").read_text())
        if (audit["completed_updates"] != config["arguments"]["iterations"]
                or audit["global_envs"] != 4096
                or not config["initial_actor_and_critic_bitwise_restored"]
                or len(set(audit["rank_network_sha256"])) != 1):
            raise ValueError(f"Training audit failed for {arm}")
        payloads[arm] = summary_payload(result, arm, audit["completed_updates"])
    if not args.upload:
        print(json.dumps(payloads, indent=2))
        return
    import wandb

    api = wandb.Api()
    for arm, payload in payloads.items():
        run_id = state["runs"][arm]["id"]
        run = api.run(f"{state['entity']}/{state['project']}/{run_id}")
        for name, value in payload.items():
            run.summary[name] = value
        run.summary.update()
        print(json.dumps({"published": arm, "id": run_id, "summary_fields": len(payload)}), flush=True)


if __name__ == "__main__":
    main()
