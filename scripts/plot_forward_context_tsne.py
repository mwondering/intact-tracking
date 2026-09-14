"""t-SNE views of saved context latents, with exact 64-D distance inspection."""

from __future__ import annotations

import argparse
import colorsys
import hashlib
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from matplotlib.lines import Line2D
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import TSNE, trustworthiness


SEED = 20260911


def color(index, total):
    return matplotlib.colors.to_hex(colorsys.hsv_to_rgb(index / max(total, 1), .72, .82))


def save_figure(fig, output, name, pdf=False):
    fig.savefig(output / f"{name}.png", dpi=190, bbox_inches="tight")
    if pdf:
        fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def decorate(ax, title):
    ax.set_title(title, fontsize=12, pad=12)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(alpha=.15)
    ax.spines[["top", "right"]].set_visible(False)


def categorical(ax, xy, labels, palette, names, *, size=18, alpha=.8, legend=True):
    for key in np.unique(labels):
        mask = labels == key
        ax.scatter(*xy[mask].T, s=size, color=palette[int(key)], alpha=alpha,
                   edgecolors="none", rasterized=True, label=names[int(key)])
    if legend:
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.16), ncol=3,
                  fontsize=8, frameon=False, markerscale=1.3, columnspacing=.8)


def fit_embedding(x, output, key, perplexity, iterations, summary):
    target = output / f"embedding_{key}.npz"
    data_hash = hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
    if target.exists():
        with np.load(target) as cached:
            if str(cached["data_sha256"]) != data_hash:
                raise ValueError(f"Cached embedding has different input: {key}")
            xy = cached["xy"]
            stats = json.loads(str(cached["stats"]))
    else:
        print(json.dumps({"event": "tsne_start", "view": key, "samples": len(x),
                          "perplexity": perplexity}), flush=True)
        started = time.monotonic()
        model = TSNE(n_components=2, perplexity=perplexity, early_exaggeration=12,
                     learning_rate="auto", max_iter=iterations, metric="euclidean",
                     init="pca", random_state=SEED, method="barnes_hut", angle=.5,
                     n_jobs=4, verbose=1)
        xy = model.fit_transform(x)
        if not np.isfinite(xy).all() or not np.isfinite(model.kl_divergence_):
            raise ValueError(f"Nonfinite embedding: {key}")
        stats = {"n": len(x), "perplexity": perplexity, "seed": SEED,
                 "max_iter": iterations, "iterations_completed": int(model.n_iter_) + 1,
                 "kl_divergence": float(model.kl_divergence_),
                 "elapsed_seconds": time.monotonic() - started,
                 "data_sha256": data_hash}
        if len(x) < 2000:
            stats["trustworthiness_k10"] = float(trustworthiness(x, xy, n_neighbors=10))
        np.savez_compressed(target, xy=xy, data_sha256=data_hash, stats=json.dumps(stats))
    summary["embeddings"][key] = stats
    print(json.dumps({"event": "tsne_done", "view": key, **stats}), flush=True)
    return xy


def detail_plots(output, data, indices, xy, env_palette, env_names, motion_palette):
    selected = {key: value[indices] for key, value in data.items()}
    env = np.where(selected["nominal"], -1, selected["world"])
    fig, axes = plt.subplots(1, 3, figsize=(19, 8))
    categorical(axes[0], xy, env, env_palette, env_names, size=27)
    decorate(axes[0], "Color = physical environment")
    names = {int(key): f"M{key:02d}" for key in np.unique(selected["motion"])}
    categorical(axes[1], xy, selected["motion"], motion_palette, names, size=24, legend=False)
    handles = [Line2D([], [], color=motion_palette[key], marker="o", linestyle="",
                      markersize=5, label=names[key]) for key in names]
    axes[1].legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, -.16),
                   ncol=6, fontsize=8, frameon=False, columnspacing=.8)
    decorate(axes[1], "Color = exact motion file ID")
    p = axes[2].scatter(*xy.T, c=selected["phase"], cmap="viridis", vmin=0, vmax=1,
                        s=24, alpha=.85, edgecolors="none", rasterized=True)
    fig.colorbar(p, ax=axes[2], label="Normalized motion phase", fraction=.046, pad=.04)
    decorate(axes[2], "Color = phase")
    xmin, ymin = xy.min(0); xmax, ymax = xy.max(0)
    for ax in axes:
        ax.set_xlim(xmin - .05 * (xmax - xmin), xmax + .05 * (xmax - xmin))
        ax.set_ylim(ymin - .05 * (ymax - ymin), ymax + .05 * (ymax - ymin))
    fig.suptitle(f"Context latent t-SNE | update 8000 | {len(indices)} full histories\n"
                 "Same embedding in every panel: 16 randomly selected DR worlds + nominal", fontsize=16)
    fig.text(.5, .018, "Input: original 64-D encoder output. Perplexity=30, PCA initialization. "
             "Colors are labels only; labels were not used to fit t-SNE.\n"
             "Use local mixing to inspect clusters; t-SNE spacing and cluster sizes are not calibrated 64-D distances.",
             ha="center", fontsize=10, color="#444444")
    fig.subplots_adjust(left=.055, right=.96, top=.82, bottom=.32, wspace=.27)
    save_figure(fig, output, "tsne_detail", pdf=True)

    distances = squareform(pdist(selected["latent"].astype(float)))
    fig, ax = plt.subplots(figsize=(11, 10), constrained_layout=True)
    p = ax.imshow(distances, cmap="magma", interpolation="nearest", vmin=0)
    centers, labels = [], []
    for key in np.unique(env):
        positions = np.flatnonzero(env == key)
        centers.append(positions.mean()); labels.append(env_names[int(key)])
        boundary = positions[-1] + .5
        ax.axhline(boundary, color="white", alpha=.4, linewidth=.5)
        ax.axvline(boundary, color="white", alpha=.4, linewidth=.5)
    ax.set_xticks(centers, labels, rotation=90, fontsize=9)
    ax.set_yticks(centers, labels, fontsize=9)
    ax.set_title("Actual pairwise distances in the original 64-D latent space\n"
                 "Same samples as the detail t-SNE, ordered by environment then motion", pad=14)
    fig.colorbar(p, ax=ax, label="Euclidean distance (original encoder output)")
    save_figure(fig, output, "raw_latent_distances", pdf=True)


def overview_plots(output, data, xy, chosen, env_palette, env_names, motion_palette):
    fig, axes = plt.subplots(1, 3, figsize=(19, 7))
    nominal = data["nominal"]
    for mask, col, label in ((~nominal, "#cbd0d7", "Other DR worlds"), (nominal, "#111111", "Nominal")):
        axes[0].scatter(*xy[mask].T, s=2, color=col, alpha=.25, rasterized=True, edgecolors="none", label=label)
    for key in chosen:
        mask = (~nominal) & (data["world"] == key)
        axes[0].scatter(*xy[mask].T, s=9, color=env_palette[int(key)], alpha=.95,
                         rasterized=True, edgecolors="none", label=env_names[int(key)])
    axes[0].legend(loc="upper center", bbox_to_anchor=(.5, -.12), ncol=3, fontsize=8, frameon=False)
    decorate(axes[0], "All environments; selected 16 DR worlds highlighted")
    for key in np.unique(data["motion"]):
        mask = data["motion"] == key
        axes[1].scatter(*xy[mask].T, s=2, color=motion_palette[int(key)], alpha=.45,
                        rasterized=True, edgecolors="none")
    decorate(axes[1], "Exact motion colors (same IDs as detail view)")
    p = axes[2].scatter(*xy.T, c=data["phase"], cmap="viridis", vmin=0, vmax=1,
                        s=2, alpha=.45, rasterized=True, edgecolors="none")
    fig.colorbar(p, ax=axes[2], label="Normalized motion phase", fraction=.046, pad=.04)
    decorate(axes[2], "Phase")
    fig.suptitle(f"All {len(xy):,} nonoverlapping full-context samples | original 64-D latent | t-SNE perplexity=50", fontsize=15)
    fig.text(.5, .02, "All 512 DR worlds included. Nominal points share identical nominal physics. "
             "Detail-view coordinates and overview coordinates come from separate fits.\n"
             "t-SNE expands dense regions: apparent cluster area or global gap is not a physical/latent distance.",
             ha="center", fontsize=10, color="#444444")
    fig.subplots_adjust(left=.05, right=.96, top=.87, bottom=.3, wspace=.25)
    save_figure(fig, output, "tsne_overview", pdf=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    for key in np.unique(data["motion"][nominal]):
        mask = nominal & (data["motion"] == key)
        axes[0].scatter(*xy[mask].T, s=3, color=motion_palette[int(key)], alpha=.5,
                        edgecolors="none", rasterized=True)
    p = axes[1].scatter(*xy[nominal].T, c=data["phase"][nominal], cmap="viridis", vmin=0, vmax=1,
                        s=3, alpha=.5, edgecolors="none", rasterized=True)
    fig.colorbar(p, ax=axes[1], label="Normalized motion phase")
    decorate(axes[0], "Nominal only: exact motion colors")
    decorate(axes[1], "Nominal only: phase")
    fig.suptitle("Nominal points from the all-sample embedding (no separate t-SNE refit)", fontsize=14)
    save_figure(fig, output, "tsne_nominal")


def interactive(output, data, indices, xy, unit_xy, metadata, env_palette, motion_palette):
    import plotly.graph_objects as go
    import plotly.io as pio

    sub = {key: value[indices] for key, value in data.items()}
    env = np.where(sub["nominal"], -1, sub["world"])
    env_labels = {int(k): "nominal" if k == -1 else f"DR {k}" for k in np.unique(env)}
    custom = np.column_stack((np.arange(len(indices)), sub["world"], sub["motion"],
                              sub["phase"], [Path(metadata["motion_files"][i]).name for i in sub["motion"]]))
    fig = go.Figure()
    mode_ranges = {}
    for mode in ("environment", "motion", "phase"):
        begin = len(fig.data)
        if mode == "phase":
            masks = [("Phase", np.ones(len(indices), dtype=bool), None)]
        else:
            values = env if mode == "environment" else sub["motion"]
            masks = [(env_labels[int(key)] if mode == "environment" else f"M{key:02d} · {Path(metadata['motion_files'][key]).name.removesuffix('.motion.npz')}",
                      values == key, env_palette[int(key)] if mode == "environment" else motion_palette[int(key)])
                     for key in np.unique(values)]
        for name, mask, col in masks:
            marker = dict(size=8, opacity=.85)
            if col is None:
                marker.update(color=sub["phase"][mask], colorscale="Viridis", cmin=0, cmax=1,
                              colorbar=dict(title="Phase"))
            else:
                marker["color"] = col
            fig.add_trace(go.Scattergl(x=xy[mask, 0], y=xy[mask, 1], mode="markers", name=name,
                                      marker=marker, customdata=custom[mask], visible=mode == "environment",
                                      hovertemplate="World %{customdata[1]}<br>M%{customdata[2]} · %{customdata[4]}"
                                      "<br>Phase %{customdata[3]:.3f}<extra></extra>"))
        mode_ranges[mode] = (begin, len(fig.data))
    buttons = []
    for mode, title in (("environment", "按环境"), ("motion", "按具体 motion"), ("phase", "按 phase")):
        lo, hi = mode_ranges[mode]
        buttons.append(dict(label=title, method="update", args=[{"visible": [lo <= i < hi for i in range(len(fig.data))]}]))
    fig.update_layout(template="plotly_white", height=780, hovermode="closest",
                      title="update_008000 · 相同 t-SNE 坐标，切换着色方式",
                      xaxis_title="t-SNE 1", yaxis_title="t-SNE 2",
                      margin=dict(l=60, r=20, t=110, b=60), legend=dict(itemsizing="constant"),
                      updatemenus=[dict(type="buttons", direction="right", buttons=buttons, x=0, y=1.09)],
                      uirevision="keep-zoom")
    payload = {"z": sub["latent"].round(7).tolist(), "world": sub["world"].tolist(),
               "nominal": sub["nominal"].tolist(), "motion": sub["motion"].tolist(),
               "phase": sub["phase"].round(5).tolist()}
    script = """
const samples = __DATA__;
const graph = document.getElementById('{plot_id}');
const status = document.createElement('div');
status.style.cssText='padding:16px 22px;margin:8px 25px;background:#eff6ff;border-radius:8px;font:16px system-ui;line-height:1.7';
status.textContent='点击两个点，可比较它们在原始 64 维 latent 空间的真实距离。';
graph.parentNode.insertBefore(status,graph.nextSibling);
let first=null;
function describe(i){return (samples.nominal[i]?'Nominal':'DR '+samples.world[i])+' / M'+samples.motion[i]+' / phase '+samples.phase[i].toFixed(3);}
graph.on('plotly_click', function(event){
  const i=Number(event.points[0].customdata[0]);
  if(first===null){first=i;status.textContent='已选：'+describe(i)+'。请再点击一个点。';return;}
  const a=samples.z[first],b=samples.z[i];
  let aa=0,bb=0,ab=0,sq=0;
  for(let k=0;k<a.length;k++){aa+=a[k]*a[k];bb+=b[k]*b[k];ab+=a[k]*b[k];sq+=(a[k]-b[k])**2;}
  const cosine=ab/Math.sqrt(aa*bb);
  status.textContent=describe(first)+' ↔ '+describe(i)+' ｜原始 latent L2 距离：'+Math.sqrt(sq).toFixed(4)+' ｜单位化后 L2：'+Math.sqrt(Math.max(0,2-2*cosine)).toFixed(4)+' ｜余弦：'+cosine.toFixed(5)+'。点击新点可开始下一组比较。';
  first=null;
});
""".replace("__DATA__", json.dumps(payload, ensure_ascii=False).replace("</", "<\\/"))
    html = pio.to_html(fig, include_plotlyjs=True, full_html=True, post_script=script,
                       div_id="latent-tsne", config={"responsive": True, "displaylogo": False,
                                                       "toImageButtonOptions": {"format": "png", "scale": 2}})
    intro = """<div style="max-width:1200px;margin:24px auto 0;padding:0 24px;font:16px system-ui;line-height:1.7;color:#1f2937">
<h2>Context latent：不同环境、motion 与 phase</h2>
<p>固定随机抽取 16 个 DR world，保留其全部完整历史样本；nominal 按 motion 抽样。每个点是一个不重叠的 100 帧历史窗口。标签仅用于着色，没有参与 t-SNE 拟合。</p>
<p>切换按钮对比同一批点；悬停查看 motion 文件和 phase；点击图例可隐藏类别，双击可单独显示；拖动框选放大，双击绘图区复位。点击两个点可查看真实 64 维距离。</p>
<p><strong>图上的全局间距和簇面积不是原始 latent 的距离尺度。</strong>主图使用原始 encoder 输出，perplexity=30；全样本总览和参数对照图另附。</p></div>"""
    html = html.replace("<body>", "<body>" + intro)
    (output / "tsne_interactive.html").write_text(html)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((args.input / "metadata.json").read_text())
    with np.load(args.input / "latents.npz") as archive:
        keep = ~archive["neighbor"]
        data = {key: archive[key][keep] for key in ("latent", "world", "nominal", "motion", "phase", "step", "episode")}
    z = np.ascontiguousarray(data["latent"], dtype=np.float32)
    assert z.shape[1] == 64 and np.isfinite(z).all()
    rng = np.random.default_rng(SEED)
    chosen = np.sort(rng.choice(np.unique(data["world"][~data["nominal"]]), 16, replace=False))
    dr_ids = np.flatnonzero(~data["nominal"] & np.isin(data["world"], chosen))
    nominal_ids = []
    for motion in np.unique(data["motion"][data["nominal"]]):
        candidates = np.flatnonzero(data["nominal"] & (data["motion"] == motion))
        nominal_ids.extend(rng.choice(candidates, min(3, len(candidates)), replace=False))
    indices = np.concatenate((nominal_ids, dr_ids)).astype(int)
    env = np.where(data["nominal"][indices], -1, data["world"][indices])
    indices = indices[np.lexsort((data["phase"][indices], data["motion"][indices], env))]
    env_palette = {-1: "#111111", **{int(key): matplotlib.colors.to_hex(plt.get_cmap("tab20")(i)) for i, key in enumerate(chosen)}}
    env_names = {-1: "Nominal", **{int(key): f"DR {key}" for key in chosen}}
    motion_palette = {i: color(i, len(metadata["motion_files"])) for i in range(len(metadata["motion_files"]))}
    summary = {"checkpoint": metadata["checkpoint"], "checkpoint_sha256": metadata["checkpoint_sha256"],
               "source": str(args.input.resolve()), "seed": SEED, "sklearn_version": sklearn.__version__,
               "all_samples": len(z), "detail_samples": len(indices), "detail_nominal_samples": len(nominal_ids),
               "selected_dr_worlds": chosen.tolist(), "detail_main_row_indices": indices.tolist(),
               "motion_files": metadata["motion_files"], "embeddings": {},
               "input": "Original 64-D encoder output; Euclidean metric; no per-dimension standardization or pre-PCA reduction; labels not used in fitting.",
               "selection": "16 DR worlds chosen uniformly with fixed RNG before fitting; all their main windows retained; at most 3 nominal windows per motion file.",
               "sources": ["https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html",
                           "https://lvdmaaten.github.io/tsne/"]}
    (args.output / "tsne_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    detail = z[indices]
    xy = fit_embedding(detail, args.output, "detail_raw_p30", 30, 1500, summary)
    detail_plots(args.output, data, indices, xy, env_palette, env_names, motion_palette)
    unit = detail / np.linalg.norm(detail, axis=1, keepdims=True)
    unit_xy = fit_embedding(unit, args.output, "detail_unit_p30", 30, 1500, summary)
    variants = [("Raw latent · perplexity 10", fit_embedding(detail, args.output, "detail_raw_p10", 10, 1500, summary)),
                ("Raw latent · perplexity 30", xy),
                ("Raw latent · perplexity 50", fit_embedding(detail, args.output, "detail_raw_p50", 50, 1500, summary)),
                ("Unit latent · perplexity 30", unit_xy)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    env = np.where(data["nominal"][indices], -1, data["world"][indices])
    for ax, (title, embedding) in zip(axes.flat, variants, strict=True):
        categorical(ax, embedding, env, env_palette, env_names, size=19, legend=False)
        decorate(ax, title)
    handles = [Line2D([], [], color=env_palette[key], marker="o", linestyle="", label=env_names[key]) for key in sorted(env_names)]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=9, frameon=False)
    fig.suptitle("Same preselected samples; sensitivity to t-SNE settings\nSeparate embeddings: positions/gaps are not comparable across panels", fontsize=14)
    fig.subplots_adjust(top=.9, bottom=.12, hspace=.23, wspace=.2)
    save_figure(fig, args.output, "tsne_settings")
    interactive(args.output, data, indices, xy, unit_xy, metadata, env_palette, motion_palette)
    (args.output / "tsne_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    all_xy = fit_embedding(z, args.output, "all_raw_p50", 50, 1000, summary)
    overview_plots(args.output, data, all_xy, chosen, env_palette, env_names, motion_palette)
    summary["complete"] = True
    (args.output / "tsne_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"event": "complete", "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
