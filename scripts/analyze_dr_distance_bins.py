"""Describe raw DR parameters in equal-width bins of distance to nominal.

Uses one original parameter vector per world. No encoder or dynamics rollout.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


SOURCE = Path('runs/limb_context_20260916_dr16384/analysis/selected_worlds.npz')
OUT = Path('runs/limb_context_20260917_dr_distance_bins')


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return {key: None for key in ('mean', 'std', 'min', 'p10', 'p50', 'p90', 'max', 'range', 'pair_mae')}
    n = len(values)
    ordered = np.sort(values)
    pair_mae = None
    if n > 1:
        # Exact mean absolute difference over all distinct, ordered pairs.
        pair_mae = float(2 * np.dot(2 * np.arange(n) - n + 1, ordered) / (n * (n - 1)))
    return dict(mean=float(values.mean()), std=float(values.std()),
                min=float(ordered[0]), p10=float(np.quantile(values, .1)),
                p50=float(np.median(values)), p90=float(np.quantile(values, .9)),
                max=float(ordered[-1]), range=float(np.ptp(values)), pair_mae=pair_mae)


def csv_write(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def number(value, digits=2):
    return '—' if value is None else f'{value:.{digits}f}'


def interval(row, key, central=False):
    lo, hi = ('p10', 'p90') if central else ('min', 'max')
    return f'{number(row[f"{key}_{lo}"])}–{number(row[f"{key}_{hi}"])}'


def write_report(out, result):
    lines = [
        '# 原始 DR 参数：按到 nominal 的距离等宽分成 16 / 64 类',
        '',
        '本分析只使用 16,384 个不同环境的原始物理参数，每个环境计一次；没有调用网络，没有训练或仿真。',
        '2,048 个 nominal 对照不进入随机 DR 分档或统计。nominal 的距离为 0，低于随机样本最小值，因此不是以下 C0 的成员。',
        '',
        '## 距离与分类规则',
        '',
        '`d = sqrt(sum_j weight[j] * ((parameter[j] - nominal[j]) / span[j])**2)`。',
        '沿用原来的距离定义和权重：四肢负载各 0.1，COM 的三个轴各 0.1，躯干质量 0.1，摩擦 0.1，29 个 armature 坐标合计 0.1。encoder bias 不在距离中。',
        '跨度分别为：双手各 2.5 kg，双小腿各 4 kg，COM 各轴 15 cm，躯干质量变化 2 kg，摩擦 1.7，armature scale 0.4。nominal 摩擦为 0.6。',
        f'随机环境实际距离最小 {result["radius_min"]:.9f}，最大 {result["radius_max"]:.9f}。',
        '在这个最小值到最大值之间等宽分档。C0 距离最小，C15 / C63 距离最大。区间左闭右开，最后一档包含最大值。不是等数量分档，也不是 KMeans。',
        '16 类宽度为 0.033270637；64 类宽度为 0.008317659，后者每四档恰好对应前者的一档。',
        '',
        '## 主要结果',
        '',
        '按真实参数距离分档，可以明显分开总体负载的大小：16 类最近一档 C0 的总负载为 0.59–2.92 kg，最远一档 C15 为 11.54–12.17 kg。两端分别只有 18 和 4 个样本。',
        '样本更多的 C3（389 个）和 C12（475 个），总负载的中间 80% 范围分别为 2.27–4.37 kg 和 8.67–10.89 kg，也呈现清楚的大小差异。',
        '但是，同一个距离档内，各项参数仍然可以有很大变化。64 类中 C33 的距离仅为 0.4351–0.4434，却包含 747 个环境；左小腿负载的中间 80% 范围为 0.33–3.45 kg，摩擦为 0.49–1.82，COM 长度为 4.35–9.89 cm。',
        '从 16 类增至 64 类，四肢总负载的类内平均绝对差从 1.0941 降到 1.0689 kg。各单肢负载、躯干质量、COM 和摩擦的类内差异改善很小。',
        '因此，当前综合距离的分档能形成总体偏离程度的梯度；同档并不保证每项参数相似。这里观察的是原始参数本身，与网络是否学会表达参数无关，也没有直接测量动力学响应或专家策略表现。',
        '',
        '## 同类环境之间，各参数通常差多少',
        '',
        '每个环境等权，先选一个环境，再从其同类的其他环境中均匀选一个，计算参数绝对差的平均值。单样本类无法计算这项统计，单独排除。',
        '未分档基线使用相同的起点环境，另一个环境从所有其他随机 DR 环境中均匀选取。结果由全部可能配对精确计算，没有随机配对误差。',
        '',
        '| 参数 | 未分档 | 16 类内 | 64 类内 | 16 类减小 | 64 类减小 |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    display = [('left_hand_kg', '左手负载 / kg'), ('right_hand_kg', '右手负载 / kg'),
               ('left_shin_kg', '左小腿负载 / kg'), ('right_shin_kg', '右小腿负载 / kg'),
               ('total_load_kg', '四肢总负载 / kg'), ('torso_mass_delta_kg', '躯干质量变化 / kg'),
               ('torso_mass_abs_delta_kg', '躯干质量变化的绝对幅度 / kg'),
               ('com_x_cm', 'COM x / cm'), ('com_y_cm', 'COM y / cm'), ('com_z_cm', 'COM z / cm'),
               ('com_norm_cm', 'COM 偏移长度 / cm'), ('friction', '摩擦系数')]
    for key, title in display:
        a, b = result['partitions']['16']['pair_summary'][key], result['partitions']['64']['pair_summary'][key]
        lines.append(f'| {title} | {a["random_other_world"]:.4f} | {a["within_bin"]:.4f} | {b["within_bin"]:.4f} | {a["reduction_pct"]:.2f}% | {b["reduction_pct"]:.2f}% |')
    lines += ['', 'COM 长度接近不代表方向接近；对应的 x / y / z 绝对差也列在表中。',
              '64 类的未分档基线排除了一个单样本类的起点，和上表显示的 16 类基线只有极小差别；精确值见 summary.json。', '']
    for k in (16, 64):
        partition = result['partitions'][str(k)]
        rows = partition['bins']
        lines += [f'## {k} 类：完整分档统计', '',
                  f'非空类 {partition["occupied_bins"]}，空类 {partition["empty_bins"]}，单样本类 {partition["singleton_bins"]}；配对统计覆盖 {partition["pair_eligible_worlds"]} 个环境。',
                  '下表负载、COM 和摩擦均为类内实际最小值–最大值；括号是中间 80% 范围（P10–P90），不是置信区间。尾部类样本少，不能将其观测范围当作该距离下所有可能参数的边界。', '',
                  '| 类 | 距离区间 | 环境数 | 四肢总负载 kg：全范围（80%） | COM 长度 cm：全范围 | 摩擦：全范围 |',
                  '|---|---|---:|---|---|---|']
        for row in rows:
            lines.append(f'| C{row["bin"]} | {row["distance_lower"]:.4f}–{row["distance_upper"]:.4f} | {row["n"]} | {interval(row, "total_load_kg")}（{interval(row, "total_load_kg", True)}） | {interval(row, "com_norm_cm")} | {interval(row, "friction")} |')
        lines += ['', '各四肢的单独负载和躯干质量变化：', '',
                  '| 类 | 左手 kg | 右手 kg | 左小腿 kg | 右小腿 kg | 躯干质量有符号变化 kg |',
                  '|---|---|---|---|---|---|']
        for row in rows:
            ranges = [interval(row, key) for key in ('left_hand_kg', 'right_hand_kg', 'left_shin_kg', 'right_shin_kg', 'torso_mass_delta_kg')]
            lines.append(f'| C{row["bin"]} | ' + ' | '.join(ranges) + ' |')
        lines += ['', f'所有 38 个原始物理参数及派生量的均值、标准差、最小/最大值、P10/P50/P90、配对绝对差： [bins_{k}.csv](bins_{k}.csv)。', '']
    lines += ['## 可复现性', '',
              '`assignments.npz` 保存 world ID、原始参数、重新计算的距离、两套类号和边界；`world_assignments.csv` 提供便于查看的对应表。',
              '校验包括：原始参数向量及环境 ID 无重复；重新计算的距离与原统计一致；所有环境恰好分配一次；64 类与 16 类严格嵌套；抽样小类的绝对差公式与显式穷举配对一致。',
              f'数据 SHA256：`{result["source_sha256"]}`。',
              '', '运行命令：', '',
              '```sh', '.venv/bin/python scripts/analyze_dr_distance_bins.py', '```', '']
    (out / 'README.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with np.load(args.source) as archive:
        keep = ~archive['nominal']
        physics = archive['physics'][keep].astype(np.float64)
        world = archive['world'][keep]
        names, span, weights = archive['names'], archive['span'], archive['weights']
        saved_radius = archive['radius'][keep]
    assert len(world) == len(np.unique(world)) == len(np.unique(physics, axis=0)) == 16384
    nominal = np.zeros(len(names))
    nominal[4], nominal[5:34] = .6, 1.
    radius = np.sqrt(np.sum(((physics - nominal) / span)**2 * weights, axis=1))
    np.testing.assert_allclose(radius, saved_radius, rtol=1e-12, atol=1e-12)
    # The schema bounds are +/-1 kg, stored as relative torso-mass change.
    mass_delta_kg = physics[:, 3] / (span[3] / 2)
    features = dict(dr_distance=radius, left_hand_kg=physics[:, 34], right_hand_kg=physics[:, 35],
                    left_shin_kg=physics[:, 36], right_shin_kg=physics[:, 37],
                    hands_kg=physics[:, 34:36].sum(1), shins_kg=physics[:, 36:38].sum(1),
                    total_load_kg=physics[:, 34:38].sum(1), torso_mass_delta_kg=mass_delta_kg,
                    torso_mass_abs_delta_kg=np.abs(mass_delta_kg),
                    com_x_cm=physics[:, 0]*100, com_y_cm=physics[:, 1]*100, com_z_cm=physics[:, 2]*100,
                    com_norm_cm=np.linalg.norm(physics[:, :3], axis=1)*100,
                    friction=physics[:, 4], friction_abs_deviation=np.abs(physics[:, 4]-.6),
                    armature_scale_mean=physics[:, 5:34].mean(1),
                    armature_scale_rms_deviation=np.sqrt(np.mean((physics[:, 5:34]-1)**2, axis=1)))
    for index in range(5, 34):
        features[f'armature_scale_{names[index].rsplit("/", 1)[-1]}'] = physics[:, index]
    baseline = {name: distribution(values) for name, values in features.items()}
    result = dict(source=str(args.source), source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  worlds=len(world), nominal_controls_excluded=2048, latent_used=False,
                  rule='equal_width_observed_random_dr_min_to_max',
                  radius_min=float(radius.min()), radius_max=float(radius.max()),
                  parameter_names=names.tolist(), nominal=nominal.tolist(), span=span.tolist(), weights=weights.tolist(),
                  baseline=baseline, partitions={})
    assignments = dict(world=world, physics=physics, names=names, radius=radius)
    for k in (16, 64):
        edges = np.linspace(radius.min(), radius.max(), k+1)
        labels = np.searchsorted(edges[1:-1], radius, side='right')
        counts = np.bincount(labels, minlength=k)
        assert counts.sum() == len(world) and labels.min() == 0 and labels.max() == k-1
        assert np.all(radius >= edges[labels]) and np.all(radius <= edges[labels+1])
        rows = []
        for label in range(k):
            chosen = labels == label
            row = dict(bin=label, distance_lower=float(edges[label]), distance_upper=float(edges[label+1]), n=int(chosen.sum()))
            for name, values in features.items():
                row.update({f'{name}_{stat}': value for stat, value in distribution(values[chosen]).items()})
            rows.append(row)
        eligible = counts[labels] > 1
        pair_summary = {}
        for name, values in features.items():
            within = sum(row['n']*row[f'{name}_pair_mae'] for row in rows if row['n'] > 1) / int(eligible.sum())
            order = np.argsort(values)
            ordered = values[order]
            cumulative = np.concatenate(([0.], np.cumsum(ordered)))
            idx = np.arange(len(ordered))
            # Per-anchor sum to all other worlds, using the same eligible anchors.
            totals = ordered*idx-cumulative[:-1]+cumulative[-1]-cumulative[1:]-ordered*(len(ordered)-idx-1)
            random = float((totals / (len(ordered)-1))[eligible[order]].mean())
            if eligible.all():
                np.testing.assert_allclose(random, baseline[name]['pair_mae'], rtol=1e-9, atol=1e-10)
            pair_summary[name] = dict(within_bin=float(within), random_other_world=random,
                                      reduction_pct=float(100*(1-within/random)))
        # Independent check of the exact pairwise formula on a small populated bin.
        sample_label = next(row['bin'] for row in rows if 1 < row['n'] <= 40)
        sample = features['total_load_kg'][labels == sample_label]
        brute_force = np.abs(sample[:, None]-sample[None, :]).sum() / (len(sample)*(len(sample)-1))
        np.testing.assert_allclose(distribution(sample)['pair_mae'], brute_force)
        result['partitions'][str(k)] = dict(bin_width=float(edges[1]-edges[0]),
                                            occupied_bins=int((counts>0).sum()), empty_bins=int((counts==0).sum()),
                                            singleton_bins=int((counts==1).sum()),
                                            pair_eligible_worlds=int(eligible.sum()), pair_summary=pair_summary, bins=rows)
        assignments[f'labels_{k}'], assignments[f'edges_{k}'] = labels, edges
        csv_write(args.out / f'bins_{k}.csv', rows)
    np.testing.assert_array_equal(assignments['labels_16'], assignments['labels_64']//4)
    np.savez_compressed(args.out / 'assignments.npz', **assignments)
    csv_write(args.out / 'world_assignments.csv', [dict(world=int(world[i]), distance=float(radius[i]),
              bin16=int(assignments['labels_16'][i]), bin64=int(assignments['labels_64'][i]),
              **{name: float(values[i]) for name, values in features.items() if name != 'dr_distance'}) for i in range(len(world))])
    result['verification'] = 'passed: unique worlds and physics; recomputed metric; exhaustive assignment; nested bins; exact pair formula'
    (args.out / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    write_report(args.out, result)
    print(json.dumps({k: {name: value for name, value in part.items() if name != 'bins'} for k, part in result['partitions'].items()}, indent=2))


if __name__ == '__main__':
    main()
