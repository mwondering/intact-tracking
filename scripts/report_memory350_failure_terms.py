"""Count observed termination triggers in saved Memory350 evaluations."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/limb_context_20260910_memory350_ppo"
TERMS = ("anchor_pos", "anchor_ori", "ee_body_pos")


def summarize(path):
    row = json.loads(path.read_text())
    names = row["failure_term_names"]
    assert set(names) == set(TERMS)
    assert len(row["failed"]) == len(row["failure_terms"]) == row["episodes"]
    totals, combinations = Counter(), Counter()
    for failed, flags in zip(row["failed"], row["failure_terms"]):
        assert len(flags) == len(names)
        active = [name for name, flag in zip(names, flags) if flag]
        assert bool(failed) == bool(active)
        if active:
            totals.update(active)
            combinations[" + ".join(active)] += 1
    failed = sum(row["failed"])
    assert abs(failed / row["episodes"] - row["failure_rate"]) < 1e-8
    return {"episodes": row["episodes"], "failed_episodes": failed,
            "failure_rate": row["failure_rate"], "completed_training_updates": row["completed_training_updates"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "counts_by_term": {name: totals[name] for name in TERMS},
            "exact_combinations": dict(combinations),
            "source": str(path.relative_to(ROOT)),
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def run(update=None, final=False, supplemental=None):
    if final:
        cases = json.loads((RUN / "protocols/final.json").read_text())["cases"]
        directories = {arm: RUN / "final" / arm / "endpoint_eval" / f"update_{(0 if arm == 'frozen_tracker' else 5000):06d}"
                       for arm in ("baseline", "film", "frozen_tracker")}
        stem, title = "final_005000", "5000 轮最终测试"
    elif supplemental is not None:
        source = Path(supplemental).resolve()
        if not source.is_relative_to(RUN / "supplemental"):
            raise ValueError("Supplemental evaluations must stay inside this experiment")
        cases = json.loads(source.read_text())["cases"]
        directories = {arm: source.parent / arm for arm in ("baseline", "film")}
        stem, title = f"supplemental_{source.parent.name}", source.parent.name
    else:
        cases = json.loads((RUN / "protocols/periodic.json").read_text())["cases"]
        directories = {arm: RUN / "ppo" / f"{arm}_121/endpoint_eval/update_{update:06d}"
                       for arm in ("baseline", "film")}
        stem, title = f"update_{update:06d}", f"{update} 轮周期测试"
    rows = {case: {arm: summarize(directory / f"{case}.json") for arm, directory in directories.items()}
            for case in cases}
    for case, arms in rows.items():
        assert len({v["episodes"] for v in arms.values()}) == 1
    lines = [f"# Memory350 失败触发项：{title}", "",
             "统计已记录轨迹中的终止触发项。一次失败可能同时触发多项，JSON 保留组合计数；"
             "这些计数不能说明关闭某项后策略会恢复还是继续失稳。", "",
             "| 场景 | 策略 | Motions | 失败次数 | Anchor z 位置 | Anchor 姿态 | EE z 位置 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for case, arms in rows.items():
        for arm, row in arms.items():
            values = row["counts_by_term"]
            lines.append(f"| {case} | {arm} | {row['episodes']} | {row['failed_episodes']} | "
                         f"{values['anchor_pos']} | {values['anchor_ori']} | {values['ee_body_pos']} |")
    folder = RUN / "artifacts/failure_breakdown"
    folder.mkdir(exist_ok=True)
    output = folder / stem
    output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    output.with_suffix(".json").write_text(json.dumps({
        "created_at": time.time(), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Observed triggers only; no termination-off counterfactual",
        "rows": rows}, indent=2) + "\n")
    print(json.dumps({"markdown": str(output.with_suffix('.md')), "cases": len(rows)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--update", type=int)
    mode.add_argument("--final", action="store_true")
    mode.add_argument("--supplemental", type=Path)
    args = parser.parse_args()
    run(args.update, args.final, args.supplemental)
