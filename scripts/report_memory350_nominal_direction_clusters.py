"""CPU diagnostics of current held-out geometry and cached cross-motion clusters."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from analyze_forward_context_clusters import normalized, describe, select_pairs, pair_stats
from analyze_memory350_latent_clusters import disjoint_pairs
from analyze_memory350 import sha256
from evaluate_memory350_weak_pairs import write_json
from intact_tracking.memory350_inference import load_memory350_checkpoint


def pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


@torch.inference_mode()
def heldout(root, output, updates):
    run = root / "stage1_8192"
    checkpoints = {str(u): load_memory350_checkpoint(run / f"update_{u:06d}.pt", device="cpu") for u in updates}
    states = {str(u): torch.load(run / f"update_{u:06d}.pt", map_location="cpu", weights_only=False, mmap=True) for u in updates}
    anchor = np.array(states[str(updates[-1])]["nominal_direction_anchor"]["direction"])
    assert all(s["nominal_direction_anchor"] == states[str(updates[-1])]["nominal_direction_anchor"] for s in states.values())
    parts, hashes = [], {}
    for path in sorted(run.glob("validation_broad_rank_*.pt")):
        batch = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        hashes[path.name] = sha256(path)
        piece = {"world": batch["world_id"].numpy(), "nominal": batch["is_nominal"].numpy(),
                 "motion": batch["motion_id"].numpy(),
                 "full": (batch["history_valid"].all(1) & batch["memory_valid"].all(1)).numpy(),
                 "usable": (batch["history_valid"].any(1) | batch["memory_valid"].any(1)).numpy(),
                 "response_valid": batch["label_response_valid"].numpy(),
                 "response_rms": batch["label_response"].square().mean((1, 2)).sqrt().numpy()}
        for name, checkpoint in checkpoints.items():
            for key in ("state_mean", "state_std", "action_mean", "action_std"):
                torch.testing.assert_close(batch[key], getattr(checkpoint, key), atol=0, rtol=0)
            encoded = []
            for start in range(0, len(piece["world"]), 128):
                sl = slice(start, start+128)
                encoded.append(checkpoint.encoder(batch["history_state"][sl], batch["history_action"][sl],
                    batch["history_next_state"][sl], batch["history_valid"][sl],
                    batch["memory_interactions"][sl], batch["memory_valid"][sl]).numpy())
            piece["latent_"+name] = np.concatenate(encoded)
        parts.append(piece)
    data = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    scales = {name: state["loss_config"]["response_distance_scale"] for name, state in states.items()}
    result = {"validation_sha256": hashes, "models": {}, "sampling": "nominal10 plus independent-coordinate nominal50 DR",
              "caveat": "Fixed early held-out histories can overlap; this set does not establish disjoint cross-motion clustering."}
    for name in checkpoints:
        target = 2 * data["response_rms"] / (data["response_rms"] + scales[name])
        data["target_"+name] = target
        unit = normalized(data["latent_"+name].astype(np.float64))
        assert np.isfinite(unit).all()
        radius = np.linalg.norm(unit-anchor, axis=1)
        data["radius_"+name] = radius
        groups = {}
        for subset, mask in (("all_usable", data["usable"]), ("full350", data["full"])):
            nominal = mask & data["nominal"]
            dr = mask & ~data["nominal"] & data["response_valid"]
            error = np.abs(radius[dr] - target[dr])
            groups[subset] = {
                "nominal_samples": int(nominal.sum()), "dr_valid_samples": int(dr.sum()),
                "nominal_worlds": int(len(np.unique(data["world"][nominal]))),
                "dr_worlds": int(len(np.unique(data["world"][dr]))),
                "nominal_anchor_cosine": float((unit[nominal] @ anchor).mean()),
                "nominal_anchor_distance_rms": float(np.sqrt(np.square(radius[nominal]).mean())),
                "nominal_geometry": describe(unit[nominal]), "dr_geometry": describe(unit[dr]),
                "dr_response_pearson": float(np.corrcoef(radius[dr], data["response_rms"][dr])[0,1]),
                "dr_response_spearman": float(spearmanr(radius[dr], data["response_rms"][dr]).statistic),
                "dr_target_distance_mae": float(error.mean()),
                "dr_target_distance_smooth_l1": float(np.where(error < .25, error**2/.5, error-.125).mean()),
                "response_distance_scale": scales[name],
            }
        result["models"][name] = groups
    target = data["target_"+str(updates[-1])]
    result["target_comparison_note"] = "Each checkpoint uses its saved response scale; target-fit errors across different scales are different objectives."
    np.savez_compressed(output / "heldout_latents.npz", anchor=anchor, target=target, **data)
    write_json(output / "heldout_geometry.json", result)
    plt = pyplot()
    name = str(updates[-1]); full = data["full"]
    unit = normalized(data["latent_"+name][full].astype(np.float64))
    mean = unit.mean(0); _, singular, vh = np.linalg.svd(unit-mean, full_matrices=False)
    xy = (unit-mean) @ vh[:2].T; point = (anchor-mean) @ vh[:2].T
    nominal = data["nominal"][full]
    fig, axes = plt.subplots(1, 3, figsize=(14,4), layout="constrained")
    axes[0].scatter(*xy[~nominal].T, s=9, alpha=.35, label="DR", color="#df9327")
    axes[0].scatter(*xy[nominal].T, s=16, alpha=.7, label="Nominal", color="#267bb5")
    axes[0].scatter(*point, marker="*", s=120, color="black", label="Fixed anchor")
    axes[0].set(title=f"Current mixture, full350: PCA ({100*(singular[:2]**2).sum()/(singular**2).sum():.1f}% variance)", xlabel="PC1", ylabel="PC2")
    axes[0].legend(fontsize=8)
    for is_nominal, label, color in [(True,"Nominal","#267bb5"),(False,"DR","#df9327")]:
        axes[1].hist(data["radius_"+name][full & (data["nominal"]==is_nominal)], bins=np.linspace(0,2,31), density=True, histtype="step", lw=2, label=label, color=color)
    axes[1].set(title="Distance to fixed nominal anchor", xlabel="Unit latent distance", ylabel="Density"); axes[1].legend()
    dr = full & ~data["nominal"] & data["response_valid"]
    axes[2].scatter(target[dr], data["radius_"+name][dr], s=10, alpha=.35)
    axes[2].plot([0,2],[0,2],ls="--",color="gray")
    axes[2].set(title="DR radius vs 10-step response target", xlabel=f"Target: 2D/(D+{scales[name]:g})", ylabel="Actual unit latent distance", xlim=(0,2), ylim=(0,2))
    fig.suptitle(f"Memory350 u{name}: held-out current sampling; overlapping histories")
    fig.savefig(output / "current_mixture_geometry.png", dpi=170); plt.close(fig)
    print(json.dumps({"heldout_geometry": result["models"]}), flush=True)


def cached_geometry(output, cache, updates):
    summary = json.loads((output / "cached/summary.json").read_text())
    assert summary["complete"]
    result = {"profiles": {}, "caveat": "Legacy continuous-DR caches, not the current per-coordinate nominal mixture."}
    plt = pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(11,4), layout="constrained")
    for column, profile in enumerate(("common", "memory_training")):
        with np.load(output / "cached" / profile / "full_history_latents.npz") as saved:
            data = {k: saved[k] for k in saved.files}
        within = disjoint_pairs(data)
        between = select_pairs(data, "dr_different_world_same_motion_near_phase", np.random.default_rng(7319))
        profile_result = {"readout": summary["profiles"][profile]["readout"], "models": {}}
        for name, field, update in [("baseline","latent_baseline",updates[0]),("memory350","latent_candidate",updates[1])]:
            z = data[field]
            same, same_d = pair_stats(z,*within,data["world"],np.random.default_rng(821))
            different, different_d = pair_stats(z,*between,data["world"],np.random.default_rng(822))
            assert same["n"] > 0 and different["n"] > 0
            m = {"update":update,"same_world_cross_motion_disjoint":same,"different_world_matched_motion":different,
                 "within_over_between":same["unit_distance_rms"]/different["unit_distance_rms"],
                 "dr_geometry":describe(z[~data["nominal"]])}
            if data["nominal"].any():m["nominal_geometry"]=describe(z[data["nominal"]])
            profile_result["models"][name]=m
            if name=="memory350":
                for d,label,color in [(same_d,"Same world / different motion","#267bb5"),(different_d,"Different world / matched motion","#df9327")]:
                    axes[column].hist(d,bins=np.linspace(0,2,51),density=True,histtype="step",lw=2,label=label,color=color)
        result["profiles"][profile]=profile_result
        axes[column].set(title=f"u{updates[1]}: legacy {profile}",xlabel="Distance in unit 64-D latent",ylabel="Density")
        axes[column].legend(fontsize=8)
        if profile=="memory_training":
            from sklearn.manifold import TSNE
            worlds=np.unique(data["world"]); chosen=np.random.default_rng(1926).choice(worlds,24,replace=False)
            selected=np.isin(data["world"],chosen)
            unit=normalized(data["latent_candidate"][selected])
            xy=TSNE(n_components=2,perplexity=30,init="pca",learning_rate="auto",random_state=717,n_jobs=4).fit_transform(unit)
            labels=np.searchsorted(np.sort(chosen),data["world"][selected])
            from matplotlib.colors import hsv_to_rgb, ListedColormap, to_hex
            palette=hsv_to_rgb(np.column_stack((np.arange(24)/24, np.full(24,.78), np.full(24,.78))))
            tfig, tax=plt.subplots(figsize=(7,6),layout="constrained")
            tax.scatter(*xy.T,c=labels,cmap=ListedColormap(palette),s=12,alpha=.8)
            tax.set(title=f"u{updates[1]}: 24 preselected DR worlds across motions\nLegacy uniform payload cache; t-SNE is visualization only",xlabel="t-SNE 1",ylabel="t-SNE 2")
            tfig.savefig(output/"cross_motion_tsne.png",dpi=170);plt.close(tfig)
            np.savez_compressed(output/"tsne_points.npz",xy=xy,world=data["world"][selected],motion=data["motion"][selected],source_row=data["source_row"][selected])
            import plotly.graph_objects as go
            interactive=go.Figure()
            for label, world in enumerate(np.sort(chosen)):
                keep=labels==label
                custom=np.column_stack((data["world"][selected][keep],data["motion"][selected][keep],data["step"][selected][keep]))
                interactive.add_trace(go.Scattergl(x=xy[keep,0],y=xy[keep,1],mode="markers",name=f"world {world}",
                    marker={"color":to_hex(palette[label]),"size":7},customdata=custom,
                    hovertemplate="World %{customdata[0]}<br>Motion %{customdata[1]}<br>Step %{customdata[2]}<extra></extra>"))
            interactive.update_layout(template="plotly_white",height=760,
                title=f"u{updates[1]}: 24 fixed DR worlds, colored by world; legacy cache, visualization only",
                xaxis_title="t-SNE 1",yaxis_title="t-SNE 2")
            interactive.write_html(output/"cross_motion_tsne.html",include_plotlyjs=True)
    fig.savefig(output/"cross_motion_distances.png",dpi=170);plt.close(fig)
    write_json(output/"cross_motion_geometry.json",result)
    print(json.dumps({"cached_geometry":{p:{m:{"within_over_between":v["within_over_between"],"within":v["same_world_cross_motion_disjoint"]["unit_distance_rms"],"between":v["different_world_matched_motion"]["unit_distance_rms"]} for m,v in r["models"].items()} for p,r in result["profiles"].items()}}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--cache",type=Path,default=Path("runs/latent_cluster_probe_memory350_nominal50_u5000_20260911"))
    p.add_argument("--updates",type=int,nargs=2,default=[750,1500])
    p.add_argument("--mode",choices=("heldout","cached"),required=True)
    a=p.parse_args();torch.set_num_threads(4);a.output.mkdir(parents=True,exist_ok=True)
    if a.mode=="heldout":heldout(a.run_root,a.output,a.updates)
    else:cached_geometry(a.output,a.cache,a.updates)


if __name__=="__main__":main()
