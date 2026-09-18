"""Bin response10 latents solely by distance to an independent nominal reference."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from analyze_dr_distance_bins import distribution, csv_write
from intact_tracking.forward_predictor import physical_state_delta


OUT = Path('runs/limb_context_20260917_response10_radial_bins')
DYNAMICS = Path('runs/limb_context_20260916_latent_environment_partitions')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def correlation(x, y):
    def rank(values):
        _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
        starts = np.cumsum(counts)-counts
        return (starts+(counts-1)/2)[inverse]

    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def response_probe(data, manifest):
    """Recompute the original training-label norm on existing matched GPU rollouts."""
    checkpoint = torch.load(manifest['checkpoint_path'], map_location='cpu', weights_only=False, mmap=True)
    sigma = torch.as_tensor(checkpoint['normalization']['delta_std'], dtype=torch.float32)
    assert sigma.shape == (70,) and (sigma > 0).all()
    score = np.full(len(data['world']), np.nan)
    records = []
    for directory in ('dynamics', 'dynamics_confirmation'):
        path = DYNAMICS / directory
        audit = json.loads((path / 'audit.json').read_text())
        assert audit['complete']
        world = np.load(path / 'worlds.npz')
        ids = world['indices']
        assert np.isnan(score[ids]).all()
        np.testing.assert_array_equal(world['world'], data['world'][ids])
        np.testing.assert_allclose(world['physics'][2:2050], data['physics'][ids], atol=1e-6)
        state = np.load(path / 'state.npy', mmap_mode='r')
        assert state.shape[1:] == (2066, 11, 71)
        squared, per_anchor, repeat = [], [], []
        for start in range(0, len(state), 4):
            actual = torch.from_numpy(np.array(state[start:start+4, :, 1:]))
            reference = actual[:, :1].expand_as(actual)
            response = physical_state_delta(reference, actual) / sigma
            energy = response.square().mean((-1, -2)).numpy()
            squared.append(energy[:, 2:2050])
            per_anchor.append(np.sqrt(energy[:, 2:2050]))
            repeat.append(energy[:, 1])
        energy = np.concatenate(squared)
        score[ids] = np.sqrt(energy.mean(0))
        np.savez_compressed(OUT / f'{directory}_label_distances.npz', world_indices=ids,
                            world=data['world'][ids], distance_per_anchor=np.concatenate(per_anchor),
                            rms_distance_over_anchors=score[ids])
        records.append(dict(source=str(path), worlds=len(ids), anchors=len(state),
                            state_sha256=digest(path / 'state.npy'),
                            nominal_repeat_rms=float(np.sqrt(np.concatenate(repeat).mean()))))
    return score, records


def fmt(v, digits=2):
    return '—' if v is None else f'{v:.{digits}f}'


def main():
    torch.set_num_threads(3)
    manifest = json.loads((OUT / 'encoding_manifest.json').read_text())
    path = OUT / 'response10_selected_worlds.npz'
    assert digest(path) == manifest['archive_sha256']
    with np.load(path) as source:
        data = {key: source[key] for key in source.files}
    z = data.pop('z').astype(np.float64)
    norm = np.linalg.norm(z, axis=-1, keepdims=True)
    assert np.isfinite(z).all() and norm.min() > 1e-6
    z /= norm
    nominal = np.flatnonzero(data['nominal'])
    dr = np.flatnonzero(~data['nominal'])
    calibration = np.random.default_rng(20260917).permutation(nominal)[:1024]
    nominal_test = np.setdiff1d(nominal, calibration)
    center = z[calibration].mean((0, 1))
    radius = np.linalg.norm(z-center, axis=-1)
    assert len(dr) == 16384 and radius.shape == (18432, 16)
    response, probes = response_probe(data, manifest)
    p = data['physics'].astype(float)
    features = dict(dr_parameter_distance=data['radius'],
                    left_hand_kg=p[:, 34], right_hand_kg=p[:, 35], left_shin_kg=p[:, 36], right_shin_kg=p[:, 37],
                    total_load_kg=p[:, 34:38].sum(1), torso_mass_delta_kg=p[:, 3]/(data['span'][3]/2),
                    com_x_cm=p[:, 0]*100, com_y_cm=p[:, 1]*100, com_z_cm=p[:, 2]*100,
                    com_norm_cm=np.linalg.norm(p[:, :3], axis=-1)*100, friction=p[:, 4],
                    friction_abs_deviation=np.abs(p[:, 4]-.6),
                    armature_rms_deviation=np.sqrt(np.mean((p[:, 5:34]-1)**2, axis=-1)),
                    measured_response10_distance=response)
    response_ids = np.flatnonzero(np.isfinite(response))
    mean_radius = radius.mean(1)
    report = dict(checkpoint=manifest, protocol=dict(
        classification='Unit-normalize the original 64D encoder output. Nominal center is the arithmetic mean of 16 windows from each of 1024 separate nominal calibration worlds. Euclidean distance to this center only; no whitening, KMeans, physical labels or readout.',
        bins='Equal width from zero to the maximum observed latent radius over DR plus nominal windows. C0 nearest nominal, C15/C63 farthest. The same fixed center is used at K16 and K64.',
        sample_weighting='16 full-history observations per environment, each observation classified separately. Tables show routed-window parameter distributions and distinct-world counts; also save a separate table deduplicating worlds inside each bin. Same world may visit multiple bins.',
        nominal_evaluation='1024 held-out nominal worlds; calibration nominal worlds excluded from occupancy statistics.',
        dynamics='Original response10 label metric: RMS over 10 steps and 70 coordinates of physical_state_delta(nominal,DR)/checkpoint_delta_std. Reuse saved matched-state, identical-action GPU rollouts. First bank: 2048 worlds,96 motion anchors; second disjoint bank:2048 worlds,64 new anchors. These are independent probe motions, not the future of each encoder history. No new simulation or fitting.',
        scope='Descriptive partitions of the collected worlds. Bin maximum uses all observed radii, not a held-out prediction experiment. DR parameter similarity is not identical to response similarity.'),
        nominal_center_norm=float(np.linalg.norm(center)),
        nominal_center_split_difference=float(np.linalg.norm(center-z[nominal_test].mean((0, 1)))),
        radius_dr=distribution(radius[dr].ravel()), radius_nominal_test=distribution(radius[nominal_test].ravel()),
        probes=probes, response_distance_units='training-normalized RMS', partitions={},
        correlation_world_mean_radius_with=dict(
            total_load=correlation(mean_radius[dr], features['total_load_kg'][dr]),
            physical_dr_distance=correlation(mean_radius[dr], data['radius'][dr]),
            response10_pooled=correlation(mean_radius[response_ids], response[response_ids])) )
    for record in probes:
        ids = np.load(Path(record['source'])/'worlds.npz')['indices']
        record['radius_response_spearman'] = correlation(mean_radius[ids], response[ids])
    bank = dict(nominal_center=center, nominal_calibration_worlds=data['world'][calibration],
                nominal_test_worlds=data['world'][nominal_test], world=data['world'], radius=radius,
                measured_response10_distance=response)
    flatworld = np.repeat(dr, 16)
    maximum = float(radius.max())
    for k in (16, 64):
        edges = np.linspace(0, maximum, k+1)
        labels = np.searchsorted(edges[1:-1], radius, side='right')
        rows, world_rows = [], []
        for label in range(k):
            selected = flatworld[labels[dr].ravel() == label]
            unique = np.unique(selected)
            nt = int((labels[nominal_test] == label).sum())
            header = dict(bin=label, lower=float(edges[label]), upper=float(edges[label+1]),
                          dr_windows=len(selected), dr_worlds=len(unique), nominal_test_windows=nt)
            row, unique_row = dict(header), dict(header)
            for key, values in features.items():
                for target, ids in ((row, selected), (unique_row, unique)):
                    selected_values = values[ids]
                    selected_values = selected_values[np.isfinite(selected_values)]
                    target.update({f'{key}_{stat}': value for stat, value in distribution(selected_values).items() if stat != 'pair_mae'})
            row['response_probe_windows'] = int(np.isfinite(response[selected]).sum())
            row['response_probe_worlds'] = int(np.isfinite(response[unique]).sum())
            rows.append(row)
            world_rows.append(unique_row)
        counts = np.bincount(labels[dr].ravel(), minlength=k)
        assert counts.sum() == 16384*16
        votes = np.stack([(labels[dr] == c).sum(1) for c in range(k)], 1)
        modal_fraction = votes.max(1)/16
        quarter = []
        for title, chosen in [('near_quarter_bins', np.arange(k//4)), ('far_quarter_bins', np.arange(3*k//4, k))]:
            ids = flatworld[np.isin(labels[dr].ravel(), chosen)]
            entry = dict(group=title, bins=chosen.tolist(), windows=len(ids), worlds=len(np.unique(ids)))
            for name, values in features.items():
                v = values[ids]
                v = v[np.isfinite(v)]
                entry[name] = distribution(v)
            quarter.append(entry)
        report['partitions'][str(k)] = dict(width=float(edges[1]), occupied_dr_bins=int((counts>0).sum()),
            nominal_test_c0_fraction=float((labels[nominal_test]==0).mean()),
            nominal_test_first_quarter_fraction=float((labels[nominal_test]<k//4).mean()),
            modal_fraction_per_world=distribution(modal_fraction), quarter_groups=quarter, bins=rows)
        bank[f'edges_{k}'], bank[f'labels_{k}'] = edges, labels
        csv_write(OUT/f'bins_{k}.csv', rows)
        csv_write(OUT/f'bins_{k}_unique_worlds.csv', world_rows)
    np.testing.assert_array_equal(bank['labels_16'], bank['labels_64']//4)
    np.savez_compressed(OUT/'radial_router_and_assignments.npz', **bank)
    report['script_sha256'] = digest(Path(__file__))
    (OUT/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    lines = ['# response10/u15000：按 latent 到 nominal 的距离分档', '',
             '本次复用与原始 DR 分档完全相同的 16,384 个随机环境、每环境 16 个完整 350 步历史；在 GPU 上用旧 response10/u15000 重新编码，共 262,144 个 DR latent。另有 2,048 个 nominal 环境，每环境 16 个窗口。',
             '将 encoder 输出单位归一化，用 1,024 个 nominal 环境的 16,384 个 latent 的均值作为固定 nominal 中心。分类唯一依据为 `r = ||unit(z) - nominal_center||₂`。不白化、不用 DR 参数或线性读出参与分类。',
             f'在 0 到观测最大距离 {maximum:.6f} 之间等宽分成 16 / 64 档；编号越大越远。边界左闭右开，末档包含最大值。',
             '另 1,024 个 nominal 环境只用于检查 nominal 的实际分档，不参与参考中心估计。', '',
             '原始参数只用于描述各档。每个窗口独立分档，同一固定 DR 环境可能随历史进入多个档；下表参数分布按实际路由窗口统计，环境数为该档出现过的不同环境数量，不能跨档相加。完整 CSV 同时给出逐窗口和每档去重环境两种口径。', '',
             '## 距离与独立动态测试', '',
             f'Nominal 测试窗口距离 P10/P50/P90：{report["radius_nominal_test"]["p10"]:.4f} / {report["radius_nominal_test"]["p50"]:.4f} / {report["radius_nominal_test"]["p90"]:.4f}。',
             f'随机 DR 窗口距离 P10/P50/P90：{report["radius_dr"]["p10"]:.4f} / {report["radius_dr"]["p50"]:.4f} / {report["radius_dr"]["p90"]:.4f}。',
             '复用两组共 4,096 个不同 DR 环境的 GPU 仿真：每组内所有环境从同状态执行同一组动作，与 nominal 比较 10 步响应。用旧 checkpoint 的 delta_std 归一化 70 维物理状态差，复现原标签的距离定义。分别在 96 / 64 个 motion 起点上取平方平均后开方，得到每个环境的动态距离。',
             '这些 probe motion 与每个 latent 的交互历史并非逐窗口对齐，因此检验的是跨 motion 的平均影响大小，不能当作原训练窗口标签预测误差。',
             f'按环境平均 latent 距离与实测动态距离的排序相关：两组分别 {probes[0]["radius_response_spearman"]:.3f} / {probes[1]["radius_response_spearman"]:.3f}。排序相关越接近 1，越符合 latent 远的环境动态影响也大；不是分类准确率。', '']
    for k in (16, 64):
        result = report['partitions'][str(k)]
        lines += [f'## {k} 档', '',
                  f'随机 DR 占用 {result["occupied_dr_bins"]} 个档。独立 nominal 窗口落入 C0 的比例 {result["nominal_test_c0_fraction"]:.2%}，落入最近四分之一档位的比例 {result["nominal_test_first_quarter_fraction"]:.2%}。',
                  '下面总负载、COM 长度和摩擦列均为 P10–P90，动态距离为有实测数据的路由窗口的均值。完整最小/最大值及单独四肢参数在 CSV 中。', '',
                  '| 档 | latent 距离 | DR 窗口 / 不同环境 | nominal 测试窗口 | 总负载 kg | COM cm | 摩擦 | 实测动态距离均值 |',
                  '|---|---|---:|---:|---|---|---|---:|']
        for row in result['bins']:
            intervals = [f'{fmt(row[key+"_p10"])}–{fmt(row[key+"_p90"])}' for key in ('total_load_kg', 'com_norm_cm', 'friction')]
            lines.append(f'| C{row["bin"]} | {row["lower"]:.4f}–{row["upper"]:.4f} | {row["dr_windows"]} / {row["dr_worlds"]} | {row["nominal_test_windows"]} | '+' | '.join(intervals)+f' | {fmt(row["measured_response10_distance_mean"],4)} |')
        lines += ['', f'[完整逐窗口统计](bins_{k}.csv) · [每档去重环境统计](bins_{k}_unique_worlds.csv)', '']
    lines += ['## 产物', '',
              '`radial_router_and_assignments.npz` 保存固定 nominal 中心、16 / 64 档边界、所有窗口的距离和类号。`response10_selected_worlds.npz` 保存重编码结果及原始参数、world ID、step 和原始历史索引。',
              '每个原始历史文件的哈希在 GPU 重编码时逐一验证，编码器 checkpoint 哈希与历史记录一致；所有历史均满足 short50 / long30。没有修改训练任务或现有策略。',
              '', '```sh', 'CUDA_VISIBLE_DEVICES=6 .venv/bin/python scripts/encode_response10_dr16384.py',
              '.venv/bin/python scripts/analyze_response10_radial_bins.py', '```', '']
    (OUT/'README.md').write_text('\n'.join(lines))
    print(json.dumps(dict(correlation=report['correlation_world_mean_radius_with'], probes=probes,
                         nominal=report['radius_nominal_test'], dr=report['radius_dr']), indent=2), flush=True)
    for k in ('16', '64'):
        print(json.dumps({key: val for key, val in report['partitions'][k].items() if key != 'bins'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
