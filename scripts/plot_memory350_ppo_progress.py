"""Export paired periodic tracking results without touching training processes."""

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/limb_context_20260910_memory350_ppo"
os.environ["MPLCONFIGDIR"] = str(ROOT / ".runtime/memory350_ppo_v1/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

files = sorted((RUN / "progress_comparisons").glob("update_*.json"))
assert files, "No completed paired evaluations"
rows = [json.loads(path.read_text()) for path in files]
updates = [row["completed_updates"] for row in rows]
assert updates == sorted(set(updates))
cases = (
    ("all_0_cold", "0 kg, cold", "#0072B2", "--"),
    ("all_0_warm", "0 kg, warm", "#0072B2", "-"),
    ("all_4_cold", "4 kg, cold", "#D55E00", "--"),
    ("all_4_warm", "4 kg, warm", "#D55E00", "-"),
)
panels = (
    ("common_error_body_pos", "Body position error reduction (%) — common survival", "relative"),
    ("common_error_joint_pos", "Joint position error reduction (%) — common survival", "relative"),
    ("truncated_error_anchor_pos", "Anchor position error reduction (%) — own survival", "relative"),
    ("truncated_error_body_lin_vel", "Body linear velocity error reduction (%) — own survival", "relative"),
    ("failure_rate", "Failure-rate reduction (percentage points)", "negative_delta"),
    ("coverage", "Coverage gain (percentage points)", "delta"),
)


def values(case, metric, mode):
    output = []
    for row in rows:
        record = row["cases"][case]
        assert record["paired_motions"] == 512 and record["training_seed_count"] == 1
        result = record[metric]
        if mode == "relative":
            value = result["reduction_percent"]
            low, high = result["reduction_percent_ci95"]
        else:
            value = 100 * result["candidate_minus_reference"]
            low, high = (100 * point for point in result["difference_ci95"])
            if mode == "negative_delta":
                value, low, high = -value, -high, -low
        output.append((value, low, high))
    return tuple(zip(*output))


fig, axes = plt.subplots(3, 2, figsize=(12, 10.5), sharex=True)
for axis, (metric, title, mode) in zip(axes.flat, panels):
    for case, label, color, style in cases:
        mean, low, high = values(case, metric, mode)
        axis.plot(updates, mean, label=label, color=color, linestyle=style, linewidth=1.8)
        axis.fill_between(updates, low, high, color=color, alpha=0.07 if style == "--" else 0.13)
    axis.axhline(0, color="#444444", linewidth=0.9)
    if updates[-1] >= 1000:
        axis.axvline(1000, color="#777777", linewidth=1, linestyle=":")
    axis.set_title(title, fontsize=11)
    axis.grid(alpha=0.18)
    axis.set_xlim(0, updates[-1] + 25)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
    axis.tick_params(labelsize=9)
    for edge in ("top", "right"):
        axis.spines[edge].set_visible(False)
for axis in axes[-1]:
    axis.set_xlabel("Completed PPO updates")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.948), ncol=4, frameon=False)
fig.suptitle(f"Memory350 latent vs baseline — through update {updates[-1]}", y=0.987, fontsize=15)
fig.text(0.5, 0.041, "Positive values favor latent in every panel. Dotted line: adaptive sampling begins after update 1000.", ha="center", fontsize=9)
fig.text(0.5, 0.024, "Own-survival errors may be biased by early failures; interpret them with failure rate and coverage.", ha="center", fontsize=9)
fig.text(0.5, 0.007, "512 paired motions; pointwise 95% motion-bootstrap intervals; one training seed. Full tracking metrics are in the CSV tables.", ha="center", fontsize=9)
fig.subplots_adjust(top=0.885, bottom=0.11, left=0.07, right=0.985, hspace=0.32, wspace=0.23)
output = RUN / "artifacts/plots"
output.mkdir(parents=True, exist_ok=True)
stem = output / f"tracking_progress_update_{updates[-1]:06d}"
fig.savefig(stem.with_suffix(".png"), dpi=180)
fig.savefig(stem.with_suffix(".pdf"))
plt.close(fig)
manifest = {
    "last_completed_update": updates[-1],
    "inputs": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "positive_values_favor": "latent",
    "motion_count": 512,
    "training_seed_count": 1,
    "final_protocol_motion_count": 4096,
    "panels": [{"metric": metric, "title": title, "mode": mode} for metric, title, mode in panels],
}
stem.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({"png": str(stem.with_suffix(".png")), "pdf": str(stem.with_suffix(".pdf")), "updates": updates}))
