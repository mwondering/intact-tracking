"""Static and offline interactive plots for the paired Memory350 latent probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.spatial.distance import pdist, squareform

from plot_forward_context_tsne import categorical, color, decorate, fit_embedding, save_figure


def labels_and_selection(data, metadata):
    rng = np.random.default_rng(20260911)
    worlds = np.sort(rng.choice(np.unique(data['world'][~data['nominal']]), 16, replace=False))
    ids = np.flatnonzero(~data['nominal'] & np.isin(data['world'], worlds)).tolist()
    for motion in np.unique(data['motion'][data['nominal']]):
        candidates = np.flatnonzero(data['nominal'] & (data['motion'] == motion))
        ids.extend(rng.choice(candidates, min(3, len(candidates)), replace=False).tolist())
    ids = np.asarray(ids)
    env = np.where(data['nominal'][ids], -1, data['world'][ids])
    ids = ids[np.lexsort((data['phase'][ids], data['motion'][ids], env))]
    env_palette = {-1: '#111111', **{int(k): matplotlib.colors.to_hex(plt.get_cmap('tab20')(i))
                                   for i, k in enumerate(worlds)}}
    env_names = {-1: 'Nominal', **{int(k): f'DR {k}' for k in worlds}}
    motion_palette = {i: color(i, len(metadata['motion_files'])) for i in range(len(metadata['motion_files']))}
    return ids, worlds, env_palette, env_names, motion_palette


def three_panels(axes, xy, data, env_palette, env_names, motion_palette, name, legend=True):
    env = np.where(data['nominal'], -1, data['world'])
    categorical(axes[0], xy, env, env_palette, env_names, size=22, legend=legend)
    decorate(axes[0], name + ' | environment')
    motion_names = {int(k): f'M{k:02d}' for k in np.unique(data['motion'])}
    categorical(axes[1], xy, data['motion'], motion_palette, motion_names, size=22, legend=False)
    decorate(axes[1], name + ' | exact motion ID')
    scatter = axes[2].scatter(*xy.T, c=data['phase'], cmap='viridis', vmin=0, vmax=1,
                              s=22, alpha=.85, edgecolors='none', rasterized=True)
    axes[2].figure.colorbar(scatter, ax=axes[2], fraction=.046, pad=.04, label='Motion progress')
    decorate(axes[2], name + ' | phase')


def interactive(output, data, xy, metadata, env_palette, env_names, motion_palette):
    import plotly.graph_objects as go
    import plotly.io as pio
    env = np.where(data['nominal'], -1, data['world'])
    custom = np.column_stack((np.arange(len(xy)), data['world'], data['motion'], data['phase'],
                              [Path(metadata['motion_files'][i]).name for i in data['motion']]))
    figure = go.Figure()
    ranges = {}
    for mode in ['environment', 'motion', 'phase']:
        start = len(figure.data)
        values = env if mode == 'environment' else data['motion']
        categories = [None] if mode == 'phase' else np.unique(values)
        for key in categories:
            mask = np.ones(len(xy), bool) if key is None else values == key
            marker = dict(size=8, opacity=.85)
            if mode == 'phase':
                name = 'Phase'
                marker.update(color=data['phase'], colorscale='Viridis', cmin=0, cmax=1,
                              colorbar=dict(title='Phase'))
            elif mode == 'environment':
                name = env_names[int(key)]
                marker['color'] = env_palette[int(key)]
            else:
                name = f'M{key:02d} · {Path(metadata["motion_files"][key]).name}'
                marker['color'] = motion_palette[int(key)]
            figure.add_trace(go.Scattergl(x=xy[mask, 0], y=xy[mask, 1], name=name,
                mode='markers', marker=marker, customdata=custom[mask], visible=mode == 'environment',
                hovertemplate='World %{customdata[1]}<br>M%{customdata[2]} · %{customdata[4]}'
                              '<br>Phase %{customdata[3]:.3f}<extra></extra>'))
        ranges[mode] = (start, len(figure.data))
    buttons = []
    for mode, title in [('environment', '按环境'), ('motion', '按 motion'), ('phase', '按 phase')]:
        lo, hi = ranges[mode]
        buttons.append(dict(label=title, method='update', args=[{'visible': [lo <= i < hi for i in range(len(figure.data))]}]))
    figure.update_layout(template='plotly_white', height=800, title='Memory350 · update 22700',
        xaxis_title='t-SNE 1', yaxis_title='t-SNE 2', hovermode='closest',
        margin=dict(l=60, r=20, t=105, b=60), uirevision='keep-zoom',
        updatemenus=[dict(type='buttons', direction='right', buttons=buttons, x=0, y=1.08)])
    payload = {k: data[k].tolist() for k in ['latent', 'latent_v12', 'world', 'nominal', 'motion', 'phase']}
    script = """
const samples = __DATA__;
const graph = document.getElementById('{plot_id}');
const status = document.createElement('div');
status.style.cssText='padding:16px;margin:20px;background:#eff6ff;font:16px system-ui;line-height:1.8';
status.textContent='连续点击两个点，比较它们在 Memory350 和 v12 原始 64 维空间中的距离。';
graph.parentNode.insertBefore(status,graph.nextSibling);
let first=null;
function label(i){return (samples.nominal[i]?'Nominal':'DR '+samples.world[i])+' / M'+samples.motion[i]+' / phase '+samples.phase[i].toFixed(3);}
function distance(a,b){let sq=0,aa=0,bb=0,ab=0;for(let k=0;k<a.length;k++){sq+=(a[k]-b[k])**2;aa+=a[k]*a[k];bb+=b[k]*b[k];ab+=a[k]*b[k];}return '原始 L2 '+Math.sqrt(sq).toFixed(4)+'；单位化 L2 '+Math.sqrt(Math.max(0,2-2*ab/Math.sqrt(aa*bb))).toFixed(4);}
graph.on('plotly_click',function(event){const i=Number(event.points[0].customdata[0]);if(first===null){first=i;status.textContent='已选 '+label(i)+'，请再选一个点。';return;}status.textContent=label(first)+' ↔ '+label(i)+' ｜Memory350：'+distance(samples.latent[first],samples.latent[i])+' ｜v12：'+distance(samples.latent_v12[first],samples.latent_v12[i]);first=null;});
""".replace('__DATA__', json.dumps(payload).replace('</', '<\\/'))
    html = pio.to_html(figure, full_html=True, include_plotlyjs=True, post_script=script,
                      div_id='memory350-latent', config={'responsive': True, 'displaylogo': False})
    intro = '<div style="margin:25px;font:16px system-ui;line-height:1.7"><b>Memory350 latent，固定随机选出的 16 个 DR 环境</b><p>每个样本都有完整的短期 50 步和长期 30×10 步，v12 在同一条轨迹上也有完整 100 步。按钮切换同一批点的颜色，标签未参与拟合。悬停查看 motion；点图例可隐藏类别，双击图例可单独显示。</p><p>当前 profile：'+metadata['profile']+'。图中坐标只来自 Memory350 的 t-SNE；全局间距与簇面积不能直接表示原始 latent 距离。点击两个点可同时读取新旧模型的真实距离。</p></div>'
    (output / 'tsne_interactive.html').write_text(html.replace('<body>', '<body>'+intro))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    args = parser.parse_args()
    metadata = json.loads((args.input / 'metadata.json').read_text())
    analysis = args.input / 'analysis'
    with np.load(analysis / 'main_data.npz') as f:
        data = {k: f[k] for k in f.files}
    with np.load(analysis / 'common_full_pairs.npz') as f:
        pairs = {k: f[k] for k in f.files}
    output = args.input / 'plots'
    output.mkdir(exist_ok=True)
    ids, worlds, env_palette, env_names, motion_palette = labels_and_selection(data, metadata)
    sub = {k: v[ids] for k, v in data.items()}
    manifest = {'checkpoint': metadata['checkpoint'], 'profile': metadata['profile'],
                'seed': 20260911, 'selected_worlds': worlds.tolist(),
                'selected_main_rows': ids.tolist(), 'selected_source_rows': sub['source_row'].tolist(),
                'samples': len(ids), 'nominal_samples': int(sub['nominal'].sum()), 'embeddings': {}}
    coordinates = {}
    for key, field in [('memory350', 'latent'), ('v12', 'latent_v12')]:
        coordinates[key] = fit_embedding(sub[field], output, key+'_raw_p30', 30, 1500, manifest)
    fig, axes = plt.subplots(1, 3, figsize=(19, 8))
    three_panels(axes, coordinates['memory350'], sub, env_palette, env_names, motion_palette, 'Memory350 u22700')
    handles = [Line2D([], [], marker='o', linestyle='', color=motion_palette[k], markersize=5,
                      label=f'M{k:02d}') for k in np.unique(sub['motion'])]
    axes[1].legend(handles=handles, loc='upper center', bbox_to_anchor=(.5,-.16), ncol=6, fontsize=8, frameon=False)
    fig.suptitle(f'Memory350 latent | {metadata["profile"]} | {len(ids)} paired full-history samples\nSame coordinates in all three panels; colors are labels only', fontsize=16)
    fig.text(.5, .025, 'Original 64-D latent; Euclidean t-SNE, perplexity 30. Long histories can overlap across queries.\nGlobal gaps and cluster areas are not calibrated latent distances.', ha='center', fontsize=10)
    fig.subplots_adjust(left=.05, right=.96, top=.82, bottom=.32, wspace=.28)
    save_figure(fig, output, 'tsne_memory350', pdf=True)
    fig, axes = plt.subplots(2, 3, figsize=(19, 12))
    for row, key in enumerate(['v12', 'memory350']):
        three_panels(axes[row], coordinates[key], sub, env_palette, env_names, motion_palette, key, legend=False)
    handles = [Line2D([], [], color=env_palette[k], marker='o', linestyle='', label=env_names[k]) for k in np.unique(np.where(sub['nominal'],-1,sub['world']))]
    fig.legend(handles=handles, loc='lower center', ncol=9, fontsize=9, frameon=False)
    fig.suptitle('Same physical trajectories and sample labels | v12 u8000 vs Memory350 u22700\nEach model has its own t-SNE fit: compare label mixing, not coordinates or global gaps.', fontsize=15)
    fig.subplots_adjust(left=.05, right=.96, top=.89, bottom=.105, hspace=.30, wspace=.25)
    save_figure(fig, output, 'tsne_comparison', pdf=True)
    env = np.where(sub['nominal'], -1, sub['world'])
    values = []
    for field in ['latent_v12', 'latent']:
        z = sub[field].astype(float)
        values.append(squareform(pdist(z / np.linalg.norm(z, axis=1, keepdims=True))))
    fig, axes = plt.subplots(1, 2, figsize=(17, 8), constrained_layout=True)
    vmax = max(v.max() for v in values)
    centers = [np.flatnonzero(env == k).mean() for k in np.unique(env)]
    names = [env_names[k] for k in np.unique(env)]
    for ax, value, title in zip(axes, values, ['v12 update 8000', 'Memory350 update 22700']):
        im = ax.imshow(value, cmap='magma', vmin=0, vmax=vmax, interpolation='nearest')
        ax.set_xticks(centers, names, rotation=90, fontsize=8)
        ax.set_yticks(centers, names, fontsize=8)
        for k in np.unique(env):
            edge = np.flatnonzero(env == k)[-1] + .5
            ax.axhline(edge, color='white', alpha=.3, linewidth=.5)
            ax.axvline(edge, color='white', alpha=.3, linewidth=.5)
        ax.set_title(title)
    fig.colorbar(im, ax=axes, label='True 64-D distance after per-vector L2 normalization', shrink=.8)
    fig.suptitle('Actual latent distances | identical samples and ordering | shared color scale', fontsize=15)
    save_figure(fig, output, 'unit_distance_heatmaps', pdf=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    keys = [('dr_same_world_cross_motion_disjoint', 'Same DR, different motion, disjoint memory', '#2676b5'),
            ('dr_different_world_same_motion_near_phase', 'Different DR, matched motion/phase', '#d25d32'),
            ('nominal_dr_same_motion_near_phase', 'Nominal vs DR, matched motion/phase', '#29956d')]
    for ax, field, title in zip(axes, ['latent_v12', 'latent'], ['v12 update 8000', 'Memory350 update 22700']):
        z = data[field].astype(float)
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        for key, name, col in keys:
            if key+'_left' not in pairs or not len(pairs[key+'_left']):
                continue
            distance = np.linalg.norm(z[pairs[key+'_left']] - z[pairs[key+'_right']], axis=1)
            ax.hist(distance, bins=np.linspace(0, 2, 65), density=True, histtype='step', linewidth=2, color=col, label=name)
        ax.set(xlabel='True unit-latent distance', ylabel='Density', title=title, xlim=(0, 2))
        ax.legend(fontsize=8)
    save_figure(fig, output, 'distance_distributions', pdf=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for ax, perplexity in zip(axes, [10, 30, 50]):
        xy = coordinates['memory350'] if perplexity == 30 else fit_embedding(sub['latent'], output, f'memory350_raw_p{perplexity}', perplexity, 1500, manifest)
        categorical(ax, xy, env, env_palette, env_names, size=18, legend=False)
        decorate(ax, f'Memory350 | perplexity {perplexity}')
    fig.suptitle('Same preselected samples; independent fits at different perplexities', fontsize=14)
    save_figure(fig, output, 'tsne_settings')
    interactive(output, sub, coordinates['memory350'], metadata, env_palette, env_names, motion_palette)
    (output / 'tsne_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'event': 'plots_complete', 'output': str(output)}), flush=True)


if __name__ == '__main__':
    main()
