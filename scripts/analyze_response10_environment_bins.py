"""One category per environment using its complete saved mature-query trajectory."""

import json
from pathlib import Path

import numpy as np

from analyze_dr_distance_bins import distribution, csv_write, number
from analyze_response10_radial_bins import correlation
from encode_response10_full_interactions import OUT, PREVIOUS, ACTIVITY, digest


def stats(values):
    values = np.asarray(values)
    return {key: value for key, value in distribution(values[np.isfinite(values)]).items() if key != 'pair_mae'}


def main():
    manifest = json.loads((OUT/'encoding_manifest.json').read_text())
    parts = []
    for record in manifest['shards']:
        path = OUT/f'full_shard_{record["shard"]:02d}.npz'
        assert digest(path) == record['sha256']
        with np.load(path) as saved:
            parts.append({key: saved[key] for key in saved.files if key != 'activity_names'})
    data = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    with np.load(PREVIOUS/'radial_router_and_assignments.npz') as saved:
        previous = {key: saved[key] for key in saved.files}
    with np.load(PREVIOUS/'response10_selected_worlds.npz') as saved:
        np.testing.assert_array_equal(data['world'], saved['world'])
        np.testing.assert_array_equal(data['physics'], saved['physics'])
        old_steps, old_z = saved['step'], saved['z']
    row = np.broadcast_to(np.arange(len(old_steps))[:, None], old_steps.shape)
    np.testing.assert_array_equal(data['z'][row, old_steps//100-1], old_z)
    z = data['z'].astype(float)
    valid = data['valid']
    z[valid] /= np.linalg.norm(z[valid], axis=-1, keepdims=True)
    center = previous['nominal_center']
    radius = np.linalg.norm(z-center, axis=-1)
    assert np.array_equal(np.isfinite(radius), valid)
    count = valid.sum(1)
    assert count.min() >= 23 and count.max() <= 29
    mean = np.nanmean(radius, axis=1)
    means = np.nanmean(z, axis=1)
    alternatives = dict(mean_distance=mean, median_distance=np.nanmedian(radius, axis=1),
                        rms_distance=np.sqrt(np.nanmean(radius**2, axis=1)),
                        distance_of_mean_latent=np.linalg.norm(means-center, axis=1))
    dr = ~data['nominal']
    nt = np.isin(data['world'], previous['nominal_test_worlds'])
    calibration = np.isin(data['world'], previous['nominal_calibration_worlds'])
    all_nominal_center = means[calibration].mean(0)
    p = data['physics'].astype(float)
    total = p[:, 34:38].sum(1)
    response = previous['measured_response10_distance']
    probe = np.isfinite(response)
    features = dict(latent_distance_mean=mean, latent_distance_median=alternatives['median_distance'],
        left_hand_kg=p[:, 34], right_hand_kg=p[:, 35], left_shin_kg=p[:, 36], right_shin_kg=p[:, 37],
        total_load_kg=total, torso_mass_delta_kg=p[:, 3]/.12790995401552643,
        com_x_cm=p[:, 0]*100, com_y_cm=p[:, 1]*100, com_z_cm=p[:, 2]*100,
        com_norm_cm=np.linalg.norm(p[:, :3], axis=-1)*100, friction=p[:, 4],
        measured_response10_distance=response)
    summary = dict(protocol=dict(
        world_count=int(dr.sum()), nominal_worlds=int(data['nominal'].sum()),
        interaction='Complete available 3200-control-step collection (64 s at dt=.02). Use EVERY saved query with short50 and long30 valid: queries every100 steps, 23–29 queries per world. Incomplete startup/reset histories excluded. No new simulation or policy/encoder training.',
        aggregation='Arithmetic mean of each environment\'s per-query Euclidean distance to the fixed nominal reference. Each environment is assigned once and has equal statistical weight; no majority vote, maximum, or any-window assignment.',
        boundaries='Keep the previous nominal center and 16/64 equal-width boundaries unchanged, to isolate window-vs-environment aggregation. Near/far index retains the same meaning. Real DR does not enter classification.',
        activity='Measured RMS root XY velocity and joint velocity in the raw short50 and full long300+short50 encoder input. Low root speed alone does not establish weak limb excitation. Associations are descriptive, not causal tests.',
        limitation='Complete refers to the existing finite interaction record. Each world sees its own sampled motions, not a common exhaustive motion suite. Saved queries are 100 control steps apart and overlapping; they are not independent trials.'),
        checkpoint_sha256=manifest['checkpoint_sha256'],
        full_dr_windows=int(valid[dr].sum()), full_nominal_windows=int(valid[~dr].sum()),
        windows_per_world=stats(count),
        motions_per_world=stats([len(np.unique(data['motion'][i, valid[i]])) for i in np.flatnonzero(dr)]),
        nominal_reference_shift_if_refit_on_all_histories=float(np.linalg.norm(all_nominal_center-center)),
        mean_radius_dr=stats(mean[dr]), mean_radius_nominal_test=stats(mean[nt]),
        correlations=dict(total_load=correlation(mean[dr], total[dr]),
                          response10=correlation(mean[probe], response[probe])), partitions={}, sensitivity=[])
    bank = dict(world=data['world'], nominal=data['nominal'], physics=p, full_query_count=count,
                radius=radius, valid=valid, mean_radius=mean, median_radius=alternatives['median_distance'],
                centroid_radius=alternatives['distance_of_mean_latent'], nominal_center=center,
                measured_response10_distance=response)
    previous_c0_worlds = dr & (previous['labels_16']==0).any(1)
    summary['previous_window_c0_worlds'] = int(previous_c0_worlds.sum())
    summary['previous_window_c0_world_mean_radius'] = stats(mean[previous_c0_worlds])
    for k in (16, 64):
        edges = previous[f'edges_{k}']
        labels = np.searchsorted(edges[1:-1], mean, side='right')
        rows = []
        for label in range(k):
            chosen = dr & (labels == label)
            row = dict(bin=label, lower=float(edges[label]), upper=float(edges[label+1]),
                       dr_worlds=int(chosen.sum()), nominal_test_worlds=int((nt & (labels==label)).sum()),
                       response_probe_worlds=int((chosen & probe).sum()))
            for name, values in features.items():
                row.update({f'{name}_{stat}': value for stat, value in stats(values[chosen]).items()})
            rows.append(row)
        assert sum(r['dr_worlds'] for r in rows) == 16384
        c0 = dr & (labels==0)
        summary['partitions'][str(k)] = dict(bins=rows, occupied_dr_bins=sum(r['dr_worlds']>0 for r in rows),
            nominal_test_c0_fraction=float((labels[nt]==0).mean()),
            c0_dr_world_ids=data['world'][c0].tolist(),
            previously_c0_now_c0_count=int((previous_c0_worlds & (labels==0)).sum()),
            previously_c0_destination_counts=np.bincount(labels[previous_c0_worlds], minlength=k).tolist(),
            full_c0_parameter_means={name: stats(values[c0])['mean'] for name, values in features.items()},
            out_of_previous_distance_range=int((mean>edges[-1]).sum()))
        csv_write(OUT/f'bins_{k}.csv', rows)
        bank[f'labels_{k}'], bank[f'edges_{k}'] = labels, edges
        for name, values in alternatives.items():
            lab = np.searchsorted(edges[1:-1], values, side='right')
            mask = dr & (lab==0)
            summary['sensitivity'].append(dict(k=k, aggregation=name, c0_dr_worlds=int(mask.sum()),
                c0_mean_total_load=stats(total[mask])['mean'], c0_max_total_load=stats(total[mask])['max'],
                nominal_test_c0_fraction=float((lab[nt]==0).mean()),
                assignment_agreement_with_mean=float((lab[dr]==labels[dr]).mean())))
    np.testing.assert_array_equal(bank['labels_16'], bank['labels_64']//4)
    # Check the stationary-motion hypothesis using measured velocity, not motion names.
    window_c0 = valid & dr[:, None] & (radius<previous['edges_16'][1])
    window_other = valid & dr[:, None] & ~window_c0
    motion = dict(c0_windows=int(window_c0.sum()), c0_worlds=int(window_c0.any(1).sum()),
                  other_windows=int(window_other.sum()), by_activity={}, by_family=[])
    matched = np.flatnonzero(window_c0.any(1) & window_other.any(1))
    for j, name in enumerate(ACTIVITY):
        values = data['activity'][..., j]
        near_mean = np.array([values[i, window_c0[i]].mean() for i in matched])
        other_mean = np.array([values[i, window_other[i]].mean() for i in matched])
        motion['by_activity'][name] = dict(c0=stats(values[window_c0]), other=stats(values[window_other]),
            matched_worlds=len(matched), matched_c0_mean=float(near_mean.mean()),
            matched_other_mean=float(other_mean.mean()), matched_fraction_c0_lower=float((near_mean<other_mean).mean()))
    short_root = data['activity'][..., 0]
    full_root = data['activity'][..., 1]
    short_joint = data['activity'][..., 2]
    full_joint = data['activity'][..., 3]
    for name, mask in [('short_root_below_0p1', short_root<.1), ('full_root_below_0p1', full_root<.1),
                       ('short_root_below_0p1_and_joint_below_0p5', (short_root<.1)&(short_joint<.5)),
                       ('full_root_below_0p1_and_joint_below_0p5', (full_root<.1)&(full_joint<.5))]:
        motion[name] = dict(c0_fraction=float(mask[window_c0].mean()), other_fraction=float(mask[window_other].mean()))
    files = manifest['shards'][0]['motion_files']
    families = np.array([Path(f).name.split('_subject')[0] for f in files])
    for family in sorted(set(families)):
        mask = np.isin(data['motion'], np.flatnonzero(families==family)) & valid & dr[:, None]
        n = int(mask.sum());near = int((mask & window_c0).sum())
        motion['by_family'].append(dict(family=family, windows=n, c0_windows=near, c0_rate=near/n))
    summary['motion_diagnostic'] = motion
    # Follow the previously most heavily loaded C0 environment across all queries.
    heaviest = np.flatnonzero(previous_c0_worlds)[np.argmax(total[previous_c0_worlds])]
    cases = np.unique(np.r_[np.flatnonzero(dr & (bank['labels_16']==0)), heaviest])
    examples = []
    for i in cases:
        examples.append(dict(world=int(data['world'][i]), total_load_kg=float(total[i]),
            limb_loads_kg=p[i, 34:38].tolist(), mean_radius=float(mean[i]),
            mean_latent_radius=float(alternatives['distance_of_mean_latent'][i]),
            class16=int(bank['labels_16'][i]), class64=int(bank['labels_64'][i]),
            queries=[dict(step=int(data['step'][i, t]), radius=float(radius[i, t]),
                          motion=files[int(data['motion'][i, t])], episode=int(data['episode'][i, t]),
                          **{name: float(data['activity'][i, t, j]) for j, name in enumerate(ACTIVITY)})
                     for t in np.flatnonzero(valid[i])]))
    summary['examples'] = examples
    summary['script_sha256'] = digest(Path(__file__))
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    np.savez_compressed(OUT/'environment_assignments.npz', **bank)
    csv_write(OUT/'worlds.csv', [dict(world=int(data['world'][i]), nominal=bool(data['nominal'][i]),
        full_queries=int(count[i]), distinct_motions=len(np.unique(data['motion'][i, valid[i]])),
        class16=int(bank['labels_16'][i]), class64=int(bank['labels_64'][i]),
        radius_min=float(np.nanmin(radius[i])), radius_max=float(np.nanmax(radius[i])),
        **{name: float(values[i]) if np.isfinite(values[i]) else None for name, values in features.items()})
        for i in range(len(mean))])
    write_report(summary)
    print(json.dumps({key: value for key, value in summary.items() if key not in ('partitions','examples','motion_diagnostic')}, indent=2), flush=True)
    for k in ('16','64'):
        print(json.dumps(dict(k=k, c0=summary['partitions'][k]['bins'][0],
                             prior_c0_destinations=summary['partitions'][k]['previously_c0_destination_counts']), indent=2), flush=True)
    print(json.dumps(motion, indent=2), flush=True)


def write_report(s):
    lines = ['# Response10：每个环境汇总完整交互后只分配一个类别', '',
        '沿用旧 response10/u15000、固定 nominal 参考中心及此前的 16 / 64 档边界。只改变分类单位和时间汇总方式，真实 DR 参数仍不参与分类。',
        '每环境的分档值为 `mean_t(||unit(z_t) - nominal_center||₂)`。每个环境只计一次、只有一个类号。',
        f'采用现有 3,200 控制步（64 秒）记录中的全部充分记忆查询，每 100 步查询一次，每环境 {s["windows_per_world"]["min"]:.0f}–{s["windows_per_world"]["max"]:.0f} 个有效查询，共 {s["full_dr_windows"]:,} 个 DR 窗口；不再只抽取 16 个窗口。剔除启动和重置后记忆不完整的查询。',
        '每个 DR 环境实际覆盖的不同 motion 数 P10/P50/P90：'+ '/'.join(number(s['motions_per_world'][key],1) for key in ('p10','p50','p90'))+'。这里的完整是已有有限记录，不是对所有环境执行了相同的完整 motion 集。', '',
        '## C0：环境统计', '',
        '| 档数 | C0 DR 环境数 | 平均总负载 kg | 最大总负载 kg | 平均 latent 距离 | nominal 测试环境进入 C0 |',
        '|---|---:|---:|---:|---:|---:|']
    for k in ('16','64'):
        r=s['partitions'][k]['bins'][0]
        lines.append(f'| {k} | {r["dr_worlds"]} | {number(r["total_load_kg_mean"])} | {number(r["total_load_kg_max"])} | {number(r["latent_distance_mean_mean"],5)} | {s["partitions"][k]["nominal_test_c0_fraction"]:.2%} |')
    lines += ['', f'此前按 16 个窗口分析，有 {s["previous_window_c0_worlds"]} 个 DR 环境至少一次进入 C0；改为全程平均后，其中 {s["partitions"]["16"]["previously_c0_now_c0_count"]} 个仍在 16 档 C0。其余进入了更远档位。',
              '上表 DR 负载统计不混入零负载 nominal，C0 样本很少时其均值不能推广到整个参数空间。', '',
              '## 原地 / 低活动假设', '',
              '以下比较全程保存窗口中，瞬时落入旧 C0 的窗口和其他窗口。根部平面速度与关节速度均直接从原始物理历史计算；只看 motion 名称不能认定是否原地。', '',
              '| 历史速度指标 | C0 窗口中位数 | 其他窗口中位数 | 同环境配对：C0 均值 | 同环境配对：其他均值 |',
              '|---|---:|---:|---:|---:|']
    for key, row in s['motion_diagnostic']['by_activity'].items():
        lines.append(f'| {key} | {row["c0"]["p50"]:.4f} | {row["other"]["p50"]:.4f} | {row["matched_c0_mean"]:.4f} | {row["matched_other_mean"]:.4f} |')
    lines += ['', '这只是活动程度与低 latent 距离的关联，没有通过同一 motion、同一初态下扫描不同负载证明因果；原地也可能有很强的肢体动作。', '', '## 聚合方式敏感性', '',
              '| 档数 | 汇总方式 | C0 DR 数 | C0 平均总负载 kg | 与平均距离分类一致率 |',
              '|---|---|---:|---:|---:|']
    for row in s['sensitivity']:
        lines.append(f'| {row["k"]} | {row["aggregation"]} | {row["c0_dr_worlds"]} | {number(row["c0_mean_total_load"])} | {row["assignment_agreement_with_mean"]:.2%} |')
    for k in ('16','64'):
        lines += ['',f'## {k} 档：每环境一票','',
                  '| 类 | 环境数 | 平均 latent 距离 | 总负载均值 kg | 总负载 P10–P90 kg | COM 均值 cm | 摩擦均值 | 实测 H10 距离均值 |',
                  '|---|---:|---:|---:|---|---:|---:|---:|']
        for row in s['partitions'][k]['bins']:
            lines.append(f'| C{row["bin"]} | {row["dr_worlds"]} | {number(row["latent_distance_mean_mean"],4)} | {number(row["total_load_kg_mean"])} | {number(row["total_load_kg_p10"])}–{number(row["total_load_kg_p90"])} | {number(row["com_norm_cm_mean"])} | {number(row["friction_mean"])} | {number(row["measured_response10_distance_mean"],4)} |')
        lines += ['',f'[完整参数范围及分位数](bins_{k}.csv)','']
    lines += ['## 复现与范围', '',
              '动态距离复用上次 4,096 个不同 DR 环境的匹配 H10 GPU 仿真结果。本次没有新仿真或训练。全程平均 latent 距离与该动态距离的 Spearman 相关为 '+number(s['correlations']['response10'],3)+'。',
              '完整逐环境分类和参数见 `worlds.csv`；所有成熟窗口的距离、全程均值和类号见 `environment_assignments.npz`；C0 环境与原最大负载案例的逐查询 motion、速度、距离见 `summary.json` 的 `examples`。', '',
              '```sh','CUDA_VISIBLE_DEVICES=6 .venv/bin/python scripts/encode_response10_full_interactions.py',
              '.venv/bin/python scripts/analyze_response10_environment_bins.py','```','']
    (OUT/'README.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
