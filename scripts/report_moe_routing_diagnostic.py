"""Summarize and plot observed MoE routing without changing policy or centers."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from intact_tracking.moe_routing_diagnostics import (
    temporal_statistics, collect_latent_samples, geometry_statistics, weighted_physics_statistics,
    physics_shuffle_reference,
)


def percent(value):
    return None if value is None else value * 100


def formatted(value):
    return "NA" if value is None else f"{value:.4f}"


def combine(traces):
    sample_keys = {"centers", "sampled_steps", "motion_lengths"}
    merged = {}
    for key in traces[0]:
        if key in sample_keys:
            merged[key] = traces[0][key]
            for trace in traces[1:]:
                np.testing.assert_array_equal(merged[key], trace[key])
        else:
            merged[key] = np.concatenate([trace[key] for trace in traces], axis=0 if key == "raw_dr" else 1)
    return merged


def make_plots(root, results, plot_data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    plt.rcParams.update({"font.size": 10, "savefig.dpi": 170})
    names = ["limb_load", "torso_mass", "torso_com", "foot_friction", "armature", "encoder_bias"]
    labels = ["Limb loads", "Torso mass", "Torso COM", "Foot friction", "Armature (29)", "Encoder bias (29)"]
    colors = {"mean": "#2468a2", "sample": "#d58023"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    x = np.arange(6)
    for j, mode in enumerate(("mean", "sample")):
        res = results[mode]
        values = [res["physics_current_routes"]["groups"][n]["variance_explained"]*100 for n in names]
        axes[0, 0].bar(x+(j-.5)*.35, values, .35, label=mode, color=colors[mode])
        ratios = []
        for n in names:
            v = res["geometry"]["dr"][n]
            ratios.append(v["within_expert_different_worlds"]["mean"] / v["random_different_worlds"]["mean"])
        axes[0, 1].bar(x+(j-.5)*.35, ratios, .35, label=mode, color=colors[mode])
        keys = ["steady_full_memory_within_motion", "steady_short_refilling_within_motion", "steady_boundary"]
        rates = [percent(res["temporal"]["rates"][k]["switch_fraction"]) for k in keys]
        rates = [np.nan if value is None else value for value in rates]
        axes[1, 0].bar(np.arange(3)+(j-.5)*.35, rates, .35, label=mode, color=colors[mode])
        counts = plot_data[mode]["counts"]
        purity = counts.max(-1) / counts.sum(-1).clip(1)
        axes[1, 1].plot(np.sort(purity)*100, np.arange(1,len(purity)+1)/len(purity)*100, label=mode, color=colors[mode])
    axes[0,0].set(title="DR variance explained by current expert", ylabel="Explained variance (%)", ylim=(0,100))
    axes[0,1].set(title="Within-expert DR distance / random-pair distance", ylabel="Ratio (1 = no narrowing)")
    axes[0,1].axhline(1, ls="--", color="gray", lw=1)
    for ax in axes[0]:
        ax.set_xticks(x, labels, rotation=25, ha="right")
    axes[1,0].set(title="Actual adjacent-step expert switches", ylabel="Switch probability (%)")
    axes[1,0].set_xticks(np.arange(3), ["Full memory,\nsame motion", "Refilling short\nhistory, same motion", "Reset / motion\nboundary"])
    axes[1,1].set(title="How consistently one world uses its main expert", xlabel="Dominant expert share (%)", ylabel="Cumulative worlds (%)", xlim=(0,100), ylim=(0,100))
    for ax in axes.flat:
        ax.legend(); ax.grid(axis="y", alpha=.2)
    shape = plot_data['mean']['trace']['routes'].shape
    fig.suptitle(f"Shared MoE u1000 | {shape[1]:,} static DR worlds per action mode | policy steps 500–{shape[0]-1}")
    fig.savefig(root/"routing_summary.png"); fig.savefig(root/"routing_summary.svg"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
    coordinate_labels = ["Left hand", "Right hand", "Left shin", "Right shin", "Torso mass",
                         "COM x", "COM y", "COM z", "Foot friction", "Armature (29)", "Encoder bias (29)"]
    for j, mode in enumerate(("mean", "sample")):
        physics = results[mode]["physics_current_routes"]
        scores = physics["per_coordinate_variance_explained"][:9] + [
            physics["groups"][key]["variance_explained"] for key in ("armature", "encoder_bias")]
        ax.bar(np.arange(len(scores))+(j-.5)*.35, np.asarray(scores)*100, .35, color=colors[mode], label=mode)
    ax.set_xticks(np.arange(len(coordinate_labels)), coordinate_labels, rotation=25, ha="right")
    ax.set(ylabel="DR variance explained by expert (%)", ylim=(0,100),
           title="Which physical coordinates does the 16-expert partition actually separate?")
    ax.legend(); ax.grid(axis="y", alpha=.2)
    fig.savefig(root/"dr_parameter_separation.png"); plt.close(fig)
    for mode in ("mean", "sample"):
        rows = results[mode]["physics_current_routes"]["experts"]
        fig, axes = plt.subplots(1, 4, figsize=(15, 6), sharey=True, constrained_layout=True)
        for col, ax in enumerate(axes):
            for row in rows:
                if not row["samples"]:
                    continue
                lo, med, hi = row["raw_q10_q50_q90"][col]
                i = row["expert"]
                ax.plot([lo, hi], [i,i], color=colors[mode], lw=3, alpha=.8)
                ax.scatter([med], [i], color="black", s=15, zorder=3)
            ax.set(title=["Left hand", "Right hand", "Left shin", "Right shin"][col], xlabel="Added load (kg)", xlim=(0,2.5 if col<2 else 4), yticks=np.arange(16))
            ax.grid(axis="x", alpha=.2)
        axes[0].invert_yaxis(); axes[0].set_ylabel("Expert ID")
        fig.suptitle(f"{mode}: DR load within each expert | bars = 10–90% quantiles; dots = medians")
        fig.savefig(root/f"expert_payload_ranges_{mode}.png"); plt.close(fig)
        trace = plot_data[mode]["trace"]
        dt = plot_data[mode]["dt"]
        rng = np.random.default_rng(113)
        worlds = np.sort(rng.choice(trace["routes"].shape[1], 32, replace=False))
        fig, ax = plt.subplots(figsize=(15, 10), constrained_layout=True)
        palette = plt.get_cmap("tab20")(np.arange(16))
        cmap = ListedColormap(palette); norm = BoundaryNorm(np.arange(-.5,16.5), cmap.N)
        im = ax.imshow(trace["routes"][:,worlds].T, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm,
                       extent=(0,trace["routes"].shape[0]*dt,31.5,-.5))
        for row, w in enumerate(worlds):
            reset = np.flatnonzero(trace["episode_ids"][1:,w] != trace["episode_ids"][:-1,w]) + 1
            ax.scatter(reset*dt, np.full(len(reset),row), marker="|", s=25, color="black", linewidths=.7)
        ax.axvline(500*dt, color="white", ls="--", lw=2)
        ax.set_yticks(np.arange(32), [f"W{w}: " + "/".join(f"{v:.1f}" for v in trace["raw_dr"][w,:4]) for w in worlds], fontsize=8)
        ax.set(xlabel="Seconds of MoE interaction (after 500 tracker warm-up steps)",
               ylabel="Fixed DR world: left/right hand/shin load (kg)",
               title=f"{mode}: actual expert selected at every control step | 32 preselected random worlds\nBlack ticks = new trial/motion; dashed line = main analysis start")
        fig.colorbar(im, ax=ax, ticks=np.arange(16), label="Expert ID", fraction=.025)
        fig.savefig(root/f"expert_routes_{mode}.png"); plt.close(fig)


def write_readme(root, result):
    lines = ["# 共享 MoE u1000：簇内物理参数与实际路由诊断", "",
             "冻结 policy、context encoder 和 u1000 聚类中心，在连续随机 DR 上重新交互。每个动作模式使用两个 DR seed、各1024个world，500步冻结tracker预热后运行3000步MoE。使用完整129827条motion目录、uniform采样、原噪声/推力/termination，episode上限1000步，长期完整记忆跨trial保留。mean使用确定性动作；sample按checkpoint Gaussian采样，无PPO更新。", "",
             "主统计取MoE第500–2999步，latent几何和DR组成要求short50+long30×10均完整；稳定性另分充分记忆、短记忆补充期和trial边界。所有route来自actor实际前向，中心完全不更新，开始/结束核验物理参数逐项相同。", "",
             "同簇latent距离是单位归一化64维向量的L2距离（不是64维RMS）。配对排除同world，每world最多32个分散时间样本。DR按每坐标完整训练范围归一化，分六类单独报告；综合距离对六类等权，避免29维偏置淹没4维负载。DR组成按实际路由占用加权，另保存按world主expert归组的结果。", "",
             "## 几何与实际切换", "", "| 指标 | mean | sample |", "|---|---:|---:|"]
    rows = [
        ("同expert、不同world的latent距离", lambda r:r['geometry']['latent']['within_expert_different_worlds']['mean']),
        ("不同expert、不同world的latent距离", lambda r:r['geometry']['latent']['between_experts_different_worlds']['mean']),
        ("同一world、不同motion的latent距离", lambda r:r['geometry']['same_world_different_motion']['latent_distance']['mean']),
        ("簇内/簇间latent距离", lambda r:r['geometry']['within_over_between_latent']),
        ("充分记忆、同motion的每步切换率(%)", lambda r:percent(r['temporal']['rates']['steady_full_memory_within_motion']['switch_fraction'])),
        ("短记忆补充期、同motion的每步切换率(%)", lambda r:percent(r['temporal']['rates']['steady_short_refilling_within_motion']['switch_fraction'])),
        ("reset/motion边界切换率(%)", lambda r:percent(r['temporal']['rates']['steady_boundary']['switch_fraction'])),
        ("同world主expert占比均值(%)", lambda r:percent(r['temporal']['per_world']['dominant_expert_fraction']['mean'])),
        ("主expert占比≥90%的world(%)", lambda r:percent(r['temporal']['per_world']['dominant_at_least_90_percent'])),
        ("同world不同motion主expert一致率(%)", lambda r:percent(r['temporal']['cross_motion']['same_expert_across_motion_fraction_world_weighted']['mean'])),
    ]
    for name, fn in rows:
        lines.append(f"| {name} | {formatted(fn(result['mean']))} | {formatted(fn(result['sample']))} |")
    lines.extend(["", "## 簇内DR有多分散", "", "方差解释率越高，expert划分越能区分该参数；同簇/随机距离越接近1，说明同簇仍保留近乎完整的随机差异。", "",
                  "| 参数族 | mean方差解释率(%) | sample方差解释率(%) | mean同簇/随机距离 | sample同簇/随机距离 |",
                  "|---|---:|---:|---:|---:|"])
    for name in result['mean']['physics_current_routes']['groups']:
        values = []
        for mode in ('mean','sample'):
            values.append(result[mode]['physics_current_routes']['groups'][name]['variance_explained']*100)
        for mode in ('mean','sample'):
            d = result[mode]['geometry']['dr'][name]
            values.append(d['within_expert_different_worlds']['mean']/d['random_different_worlds']['mean'])
        lines.append('| '+name+' | '+' | '.join(f'{v:.4f}' for v in values)+' |')
    lines.extend(["", "负载与质心各坐标单独展开，避免族内平均掩盖左右肢体或轴向差异：", "",
                  "| DR坐标 | mean方差解释率(%) | sample方差解释率(%) |", "|---|---:|---:|"])
    for col, name in enumerate(result['mean']['dr_schema']['names'][:9]):
        values = [result[mode]['physics_current_routes']['per_coordinate_variance_explained'][col]*100 for mode in ('mean','sample')]
        lines.append('| '+name+' | '+' | '.join(f'{v:.4f}' for v in values)+' |')
    lines.extend(["", "## 每个expert的latent距离", "", "| expert | mean同簇距离 | sample同簇距离 | mean到保存中心距离 | sample到保存中心距离 |", "|---|---:|---:|---:|---:|"])
    for i in range(16):
        a,b = result['mean']['geometry']['per_expert'][i],result['sample']['geometry']['per_expert'][i]
        values=[a['within_pair_distance']['mean'],b['within_pair_distance']['mean'],a['distance_to_saved_center']['mean'],b['distance_to_saved_center']['mean']]
        lines.append(f'| {i} | '+' | '.join(formatted(v) for v in values)+' |')
    lines.extend(["", "## 解释边界", "", "这是固定u1000权重下的闭环诊断，不更新PPO或KMeans；sample复现探索动作分布，但不声称恢复训练时原来的全部轨迹。两个seed是交互/DR采样seed，不是两个policy训练seed。观察到DR参数混合不自动证明控制能力不足，也没有直接测量最优补偿动作是否相近。", "",
                  "图：`routing_summary.png`；`expert_payload_ranges_mean/sample.png`；`expert_routes_mean/sample.png`。完整指标、逐world统计与每参数分位数见`summary.json`及`expert_parameters.csv`，每步原始数据保存在各run的`traces.npz`。", ""])
    (root/'README.md').write_text('\n'.join(lines))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    args=p.parse_args(); root=Path(args.root)
    results, plots = {}, {}
    for mode in ('mean','sample'):
        traces, metas, seeds = [], [], []
        for seed in (30401,30402):
            directory=root/f'{mode}_seed{seed}'
            meta=json.loads((directory/'result.json').read_text())
            assert meta['static_DR_unchanged_verified'] and meta['router_state_unchanged_verified']
            assert meta['actual_actor_dispatch_verified_every_step'] and meta['motion_count']==129827
            with np.load(directory/'traces.npz') as f:
                trace={k:f[k] for k in f.files}
            temporal,_,_=temporal_statistics(trace,meta['control_dt_seconds'])
            seeds.append({'seed':seed,'temporal':temporal})
            traces.append(trace); metas.append(meta)
        assert metas[0]['dr_schema']==metas[1]['dr_schema']
        assert metas[0]['checkpoint_sha256']==metas[1]['checkpoint_sha256']
        trace=combine(traces); del traces
        temporal,counts,dominant=temporal_statistics(trace,metas[0]['control_dt_seconds'])
        samples=collect_latent_samples(trace)
        geometry=geometry_statistics(samples,trace['raw_dr'],metas[0]['dr_schema'],trace['centers'])
        physics=weighted_physics_statistics(trace['raw_dr'],counts,metas[0]['dr_schema'])
        onehot=np.eye(16)[dominant]*(counts.sum(-1)>0)[:,None]
        dominant_physics=weighted_physics_statistics(trace['raw_dr'],onehot,metas[0]['dr_schema'])
        results[mode]={'temporal':temporal,'geometry':geometry,'physics_current_routes':physics,
                       'physics_world_dominant_routes':dominant_physics,'by_seed':seeds,
                       'physics_world_shuffle_reference':physics_shuffle_reference(trace['raw_dr'],counts,metas[0]['dr_schema']),
                       'dr_schema':metas[0]['dr_schema'],'checkpoint_sha256':metas[0]['checkpoint_sha256']}
        np.savez_compressed(root/f'{mode}_world_summary.npz',memberships=counts,dominant_expert=dominant,raw_dr=trace['raw_dr'])
        plots[mode]={'trace':trace,'counts':counts,'dt':metas[0]['control_dt_seconds']}
        print(json.dumps({'mode':mode,'geometry':geometry['latent'],'temporal':temporal['rates']}),flush=True)
    np.testing.assert_array_equal(plots['mean']['trace']['raw_dr'],plots['sample']['trace']['raw_dr'])
    results['paired_static_DR_identical_between_action_modes']=True
    (root/'summary.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    with (root/'expert_parameters.csv').open('w') as f:
        writer=csv.writer(f); writer.writerow(['mode','expert','parameter','mean','std','p10','median','p90'])
        for mode in ('mean','sample'):
            for row in results[mode]['physics_current_routes']['experts']:
                for col,name in enumerate(results[mode]['dr_schema']['names']):
                    if not row['samples']:
                        continue
                    writer.writerow([mode,row['expert'],name,row['raw_mean'][col],row['raw_std'][col],*row['raw_q10_q50_q90'][col]])
    make_plots(root,results,plots)
    write_readme(root,results)


if __name__=='__main__':
    main()
