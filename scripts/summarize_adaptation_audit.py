"""Report a complete frozen audit matrix, including paired failure transitions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from audit_adaptation_matrix import MODELS
from compare_adaptation_evals import compare


def paired_failure_audit(reference_paths, candidate_paths):
    failures = []
    totals = dict(reference=0, candidate=0, new_failures=0, recovered_failures=0, shared_failures=0)
    same_physics = True
    for ref_path, cand_path in zip(reference_paths, candidate_paths, strict=True):
        ref, cand = [json.loads(Path(p).read_text()) for p in (ref_path, cand_path)]
        for name in ("seed", "motion_ids", "start_frames"):
            if ref[name] != cand[name]:
                raise ValueError(f"Unmatched {name}")
        matched = ref["physics"]["physics"] == cand["physics"]["physics"]
        same_physics &= matched
        if matched and ref["physics_world_fingerprints"] != cand["physics_world_fingerprints"]:
            raise ValueError("Same-physics pair has different randomized worlds")
        for i, (a, b) in enumerate(zip(ref["failed"], cand["failed"], strict=True)):
            totals["reference"] += a
            totals["candidate"] += b
            totals["new_failures"] += b and not a
            totals["recovered_failures"] += a and not b
            totals["shared_failures"] += a and b
            if a or b:
                failures.append(dict(
                    seed=ref["seed"], world=i, motion=ref["motion_files"][ref["motion_ids"][i]],
                    start_frame=ref["start_frames"][i], reference_failed=a, candidate_failed=b,
                    reference_steps=ref["episode_lengths"][i], candidate_steps=cand["episode_lengths"][i],
                    reference_terms=[name for name, on in zip(ref["failure_term_names"], ref["failure_terms"][i], strict=True) if on],
                    candidate_terms=[name for name, on in zip(cand["failure_term_names"], cand["failure_terms"][i], strict=True) if on],
                ))
    return dict(**totals, same_physics_worlds_audited=same_physics, episodes=sum(
        len(json.loads(Path(p).read_text())["failed"]) for p in reference_paths), failed_segments=failures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[91001, 91002, 91003])
    args = parser.parse_args()
    root = args.directory
    def paths(model, physics, remove="normal"):
        return [root / f"{model}_{physics}_{remove}_seed{seed}.json" for seed in args.seeds]

    comparisons = {}
    def add(name, refs, candidates):
        report = compare(refs, candidates)
        comparisons[name] = dict(statistics=report, paired_failures=paired_failure_audit(refs, candidates))

    for physics in ("nominal", "dr"):
        for model in MODELS:
            if model != "frozen_tracker":
                add(f"{model}_vs_frozen_on_{physics}", paths("frozen_tracker", physics), paths(model, physics))
        add(f"matched_dr_vs_nominal_training_on_{physics}", paths("matched_nominal1750", physics), paths("matched_dr1750", physics))
        for remove in ("height", "contact", "height_contact"):
            add(f"teacher_remove_{remove}_on_{physics}", paths("clean_teacher500", physics), paths("clean_teacher500", physics, remove))
    add("stage1_target", paths("existing_nominal", "nominal"), paths("clean_teacher500", "dr"))
    add("stage2_target", paths("existing_nominal", "nominal"), paths("student_rotation350", "dr"))
    report = dict(cohort="audit/development, not final stage-two confirmation", seeds=args.seeds,
                  interpretation="Within-physics ratios and failure transitions answer audit questions. Legacy acceptance flags encode nominal-to-DR goals only and are not conclusions about same-environment comparisons.",
                  comparisons=comparisons)
    destination = root / ("summary.json" if len(args.seeds) == 3 else "summary_partial.json")
    destination.write_text(json.dumps(report, indent=2) + "\n")
    for name, result in comparisons.items():
        stats, fail = result["statistics"], result["paired_failures"]
        print(name, {k: round(v["ratio"], 5) for k, v in stats["metrics"].items()},
              f"fail {fail['candidate']}/{fail['episodes']} vs {fail['reference']}; new {fail['new_failures']}, recovered {fail['recovered_failures']}")


if __name__ == "__main__":
    main()
