"""Supplement the u1000 shared/specialist comparison with world-frame errors."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from intact_tracking.fixed_dr_profiles import file_sha256
from intact_tracking.global_tracking_metrics import METRICS, VERSION
from intact_tracking.memory350_policy_checkpoint_eval import evaluation_environment, write_json
from intact_tracking.memory350_policy_results import assert_paired, bootstrap, compare
from intact_tracking.residual_uniform_protocol import EVAL_VERSION, physics_contract

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv/bin/python"


def evaluate(root, output, entry):
    dr_id = entry["id"]
    for arm, checkpoint in (
        ("baseline", root / "baseline/checkpoint_update_001000.pt"),
        ("specialist", Path(entry["frozen_checkpoint"])),
    ):
        target = output / f"{arm}_dr_{dr_id:02d}.json"
        if target.exists():
            continue
        command = [str(PYTHON), "-B", "-u", "-m", "intact_tracking.cli.residual_uniform_eval",
                   "--checkpoint", str(checkpoint), "--output", str(target),
                   "--motion-manifest", str(root / "protocols/evaluation_motions.txt"),
                   "--steps", "1000", "--seed", "20001", "--policy-precision", "fp32",
                   "--dr-bank", str(root / "protocols/dr_bank.json"), "--dr-id", str(dr_id),
                   "--global-metrics"]
        env = evaluation_environment(dr_id)
        env.update(OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        with target.with_suffix(".log").open("w") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            write_json(target.with_suffix(".process.json"), {
                "pid": process.pid, "gpu": dr_id, "argv": command, "started_at": time.time()})
            try:
                code = process.wait(timeout=1800)
                if code:
                    raise RuntimeError(f"Evaluation failed ({code}): {target.with_suffix('.log')}")
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
        print(json.dumps({"completed": arm, "dr": dr_id, "output": str(target)}), flush=True)


def report(root, output, entries):
    baseline_sha = file_sha256(root / "baseline/checkpoint_update_001000.pt")
    rows, paired_a, paired_b, failures_a, failures_b = [], [], [], [], []
    metric_names = None
    for entry in entries:
        a_path, b_path = [output / f"{arm}_dr_{entry['id']:02d}.json"
                          for arm in ("baseline", "specialist")]
        a, b = [json.loads(path.read_text()) for path in (a_path, b_path)]
        assert_paired(a, b)
        assert a["checkpoint_sha256"] == baseline_sha
        assert b["checkpoint_sha256"] == entry["checkpoint_sha256"]
        previous_a = json.loads((root / "evaluation" / f"baseline_dr_{entry['id']:02d}.json").read_text())
        previous_b = json.loads(Path(entry["frozen_evaluation"]).read_text())
        replay = {}
        for arm, new, old in (("baseline", a, previous_a), ("specialist", b, previous_b)):
            assert new["completed_training_updates"] == 1000 and new["episodes"] == 512
            assert new["protocol"] == EVAL_VERSION and new["policy_precision"] == "fp32"
            assert new["residual_output_bounded"] is False
            assert new["physics"]["residual_physics_contract"] == physics_contract()
            assert new["global_metric_contract"]["version"] == VERSION
            new_dr = new["physics"]["runtime_audit"]["fixed_dr"]
            old_dr = old["physics"]["runtime_audit"]["fixed_dr"]
            # The comparison freezes a byte-identical copy of the original bank.
            assert {k: v for k, v in new_dr.items() if k != "bank"} == {
                k: v for k, v in old_dr.items() if k != "bank"}
            assert file_sha256(new_dr["bank"]) == file_sha256(old_dr["bank"]) == new_dr["bank_sha256"]
            assert new["checkpoint_sha256"] == old["checkpoint_sha256"]
            assert all(new[key] for key in ("reference_timeline_audited", "partial_reset_survivor_state_audited",
                                           "partial_reset_survivor_history_audited"))
            compatibility = {k: v for k, v in new.items() if k != "global_metric_contract"}
            compatibility["metric_names"] = old["metric_names"]
            assert_paired(compatibility, old)
            replay[arm] = {
                "paired_protocol_and_initial_state": True,
                "changed_episode_lengths": int(np.sum(np.asarray(new["episode_lengths"]) != old["episode_lengths"])),
                "old_failure_count": sum(old["failed"]), "new_failure_count": sum(new["failed"]),
                "legacy_mean_difference": {name: new["mean"][name] - old["mean"][name] for name in old["metric_names"]},
            }
        if metric_names is None:
            metric_names = a["metric_names"]
        assert a["metric_names"] == metric_names
        with np.load(a_path.with_suffix(".traces.npz")) as af, np.load(b_path.with_suffix(".traces.npz")) as bf:
            ta, tb = af["all_metrics"], bf["all_metrics"]
            for row, trace_file, trace in ((a, af, ta), (b, bf, tb)):
                assert trace_file["metric_names"].tolist() == metric_names
                np.testing.assert_array_equal(trace_file["lengths"], row["episode_lengths"])
                np.testing.assert_array_equal(trace_file["body_joint"], trace[:, :, :2])
                recovered = trace.sum(axis=1, dtype=np.float64) / np.asarray(row["episode_lengths"])[:, None]
                np.testing.assert_allclose(recovered, row["per_episode_metrics"], atol=1e-7, rtol=1e-6)
            result = compare(a, b, ta, tb)
            common = np.minimum(a["episode_lengths"], b["episode_lengths"])
            mask = np.arange(a["max_steps"])[None, :, None] < common[:, None, None]
            ca = np.where(mask, ta, 0).sum(axis=1, dtype=np.float64) / common[:, None]
            cb = np.where(mask, tb, 0).sum(axis=1, dtype=np.float64) / common[:, None]
        # Global anchor duplicates the existing world anchor definition exactly.
        for row in (a, b):
            for suffix in ("pos", "rot", "lin_vel", "ang_vel"):
                np.testing.assert_allclose(row["mean"][f"error_anchor_{suffix}"],
                                           row["mean"][f"error_anchor_{suffix}_global"], rtol=1e-5, atol=1e-7)
        paired_a.append(ca)
        paired_b.append(cb)
        failures_a.append(a["failed"])
        failures_b.append(b["failed"])
        rows.append({"id": entry["id"], "profile": entry["profile"], "comparison": result,
                     "replay_audit": replay, "baseline": str(a_path), "specialist": str(b_path)})
    mean_a, mean_b = np.mean(paired_a, axis=0), np.mean(paired_b, axis=0)
    overall = {name: bootstrap(mean_a[:, i], mean_b[:, i]) for i, name in enumerate(metric_names)}
    failure_summary = {
        "baseline_count": int(np.sum(failures_a)), "specialist_count": int(np.sum(failures_b)),
        "episodes_per_arm": int(np.size(failures_a)),
        "paired_motion_bootstrap": bootstrap(np.mean(failures_a, axis=0), np.mean(failures_b, axis=0)),
    }
    result = {"passed": True, "completed_training_updates": 1000, "metric_contract": a["global_metric_contract"],
              "aggregation": "Common survival per paired motion; equal-weight eight-DR mean per motion; 2000 paired bootstrap samples over the same 512 motion IDs",
              "training_seed_count": 1, "overall": overall, "failures": failure_summary, "rows": rows}
    write_json(output / "comparison.json", result)
    flat = []
    for row in rows:
        for name in METRICS:
            metric = row["comparison"]["common_" + name]
            flat.append({"dr": row["id"], "profile": row["profile"], "metric": name,
                         "baseline": metric["reference"], "specialist": metric["candidate"],
                         "reduction_percent": metric["reduction_percent"],
                         "ci95_low": metric["reduction_percent_ci95"][0],
                         "ci95_high": metric["reduction_percent_ci95"][1]})
    with (output / "comparison.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    lines = ["# 1000 轮共享 baseline / 专家：全局指标补测", "",
             "使用原来的双方 checkpoint、八组完整固定 DR、512 条 motion、起点和种子。新增指标直接使用世界坐标系参考，不做机器人水平平移或 yaw 对齐。所有下表指标按双方共同存活窗口统计。", "",
             f"Root/anchor 对应 `{a['global_metric_contract']['anchor_body_name']}`；body 为 {len(a['global_metric_contract']['body_names'])} 个指定 link 的等权平均。八组等权平均后，对同一 512 条 motion 做 2000 次配对 bootstrap；区间只反映 motion 抽样，不包括训练 seed 波动。", "",
             "| 全局指标 | baseline | 专家 | 专家改善 | 配对 95% 区间 |", "|---|---:|---:|---:|---:|"]
    for name in METRICS:
        metric = overall[name]
        unit = a["global_metric_contract"]["units"][name]
        scale = 100 if unit == "m" else 1
        unit = "cm" if unit == "m" else unit
        lo, hi = metric["reduction_percent_ci95"]
        lines.append(f"| {name} ({unit}) | {scale*metric['reference']:.4f} | {scale*metric['candidate']:.4f} | {metric['reduction_percent']:+.2f}% | [{lo:+.2f}%, {hi:+.2f}%] |")
    lines += ["", "| DR | 全局 body pos：baseline / 专家 (cm) | 改善 | 全局 root pos：baseline / 专家 (cm) | 改善 |", "|---|---:|---:|---:|---:|"]
    for row in rows:
        body = row["comparison"]["common_error_body_pos_global"]
        anchor = row["comparison"]["common_error_anchor_pos_global"]
        lines.append(f"| B{row['id']:02d} {row['profile']} | {100*body['reference']:.3f} / {100*body['candidate']:.3f} | {body['reduction_percent']:+.2f}% | {100*anchor['reference']:.3f} / {100*anchor['candidate']:.3f} | {anchor['reduction_percent']:+.2f}% |")
    lines += ["", f"失败数：baseline {failure_summary['baseline_count']}/{failure_summary['episodes_per_arm']}，专家 {failure_summary['specialist_count']}/{failure_summary['episodes_per_arm']}。完整单组、全局、原始对齐指标及重测一致性检查见 `comparison.json`。", "",
              "此次仅补充评测读数和逐步 trace，不改变 checkpoint、动作、reward 或 termination。GPU 接触仿真可能有非逐位确定性，旧指标的实际重测差异也记录在 replay_audit。", ""]
    (output / "README.md").write_text("\n".join(lines))
    print(json.dumps({"passed": True, "overall_global_body": overall["error_body_pos_global"],
                      "overall_global_root": overall["error_anchor_pos_global"], "failures": failure_summary}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    if not root.is_relative_to(ROOT / "runs"):
        raise ValueError("Use an existing project run")
    entries = json.loads((root / "protocols/selection.json").read_text())["specialists"]
    assert [entry["id"] for entry in entries] == list(range(8))
    output = root / "global_evaluation"
    output.mkdir(exist_ok=True)
    if args.evaluate:
        snapshot = output / "source_snapshot"
        for relative in ("src/intact_tracking/global_tracking_metrics.py", "src/intact_tracking/cli/memory350_policy_eval.py",
                         "src/intact_tracking/cli/residual_uniform_eval.py", "src/intact_tracking/memory350_policy_results.py",
                         "scripts/compare_residual_global_metrics.py", "tests/test_global_tracking_metrics.py",
                         "tests/test_memory350_policy_results.py"):
            target = snapshot / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)
        write_json(output / "state.json", {"status": "evaluating", "pid": os.getpid(), "started_at": time.time()})
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(evaluate, root, output, entry) for entry in entries]
            for future in as_completed(futures):
                future.result()
    report(root, output, entries)
    write_json(output / "state.json", {"status": "complete", "finished_at": time.time(), "report": str(output / "README.md")})


if __name__ == "__main__":
    main()
