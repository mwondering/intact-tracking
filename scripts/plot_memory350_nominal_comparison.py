"""Plot paired nominal50, all-DR age control, late all-DR and v12 latent probes."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform

from plot_memory350_latent_clusters import labels_and_selection, three_panels
from plot_forward_context_tsne import fit_embedding, save_figure


MODELS = {
    'memory350': ('latent', 'Nominal50 / u1800'),
    'alldr_u1800': ('latent_alldr_u1800', 'All-DR / u1800'),
    'alldr_u22700': ('latent_alldr_u22700', 'All-DR / u22700'),
    'v12': ('latent_v12', 'v12 / u8000'),
}


def interactive(output, data, coordinates, env_palette, motion_palette, models):
    import plotly.graph_objects as go
    import plotly.io as pio
    env = np.where(data['nominal'], -1, data['world'])
    custom = np.column_stack((np.arange(len(env)), env, data['motion'], data['phase']))
    figure = go.Figure()
    options = []
    for key, (_, name) in models.items():
        for mode in ('environment', 'motion', 'phase'):
            xy = coordinates[key]
            marker = dict(size=8, opacity=.85)
            if mode == 'phase':
                marker.update(color=data['phase'], colorscale='Viridis', cmin=0, cmax=1,
                              colorbar=dict(title='Phase'))
            else:
                values = env if mode == 'environment' else data['motion']
                palette = env_palette if mode == 'environment' else motion_palette
                marker['color'] = [palette[int(k)] for k in values]
            figure.add_trace(go.Scattergl(x=xy[:, 0], y=xy[:, 1], mode='markers',
                marker=marker, customdata=custom, visible=len(figure.data) == 0,
                hovertemplate='World %{customdata[1]} (-1=nominal)<br>Motion %{customdata[2]}'
                              '<br>Phase %{customdata[3]:.3f}<extra></extra>'))
            options.append((name + ' | ' + mode, len(figure.data) - 1))
    figure.update_layout(template='plotly_white', height=800, showlegend=False,
        title=options[0][0], xaxis_title='t-SNE 1', yaxis_title='t-SNE 2',
        updatemenus=[dict(buttons=[dict(label=name, method='update', args=[
            {'visible': [j == i for j in range(len(options))]}, {'title': name}])
            for name, i in options], x=0, y=1.13)], margin=dict(t=130))
    payload = {'models': {key: {'name': name, 'latent': data[field].tolist()}
                          for key, (field, name) in models.items()},
               **{k: data[k].tolist() for k in ('world', 'nominal', 'motion', 'phase')}}
    script = """
const samples=__DATA__;
const graph=document.getElementById('{plot_id}');
const status=document.createElement('div');
status.style.cssText='padding:20px;background:#eff6ff;font:16px system-ui;line-height:1.8';
status.textContent='连续点击两个点，查看各模型在真实 64 维空间中的距离。';
graph.parentNode.insertBefore(status,graph.nextSibling);
let first=null;
function label(i){return (samples.nominal[i]?'Nominal':'DR '+samples.world[i])+' / M'+samples.motion[i]+' / phase '+samples.phase[i].toFixed(3);}
function distance(a,b){let sq=0,aa=0,bb=0,ab=0;for(let k=0;k<a.length;k++){sq+=(a[k]-b[k])**2;aa+=a[k]*a[k];bb+=b[k]*b[k];ab+=a[k]*b[k];}return 'raw L2 '+Math.sqrt(sq).toFixed(4)+'; unit L2 '+Math.sqrt(Math.max(0,2-2*ab/Math.sqrt(aa*bb))).toFixed(4);}
graph.on('plotly_click',function(event){const i=Number(event.points[0].customdata[0]);if(first===null){first=i;status.textContent='已选 '+label(i)+'，请再选一个点。';return;}status.textContent=label(first)+' ↔ '+label(i)+' ｜'+Object.values(samples.models).map(m=>m.name+': '+distance(m.latent[first],m.latent[i])).join(' ｜');first=null;});
""".replace('__DATA__', json.dumps(payload).replace('</', '<\\/'))
    html = pio.to_html(figure, full_html=True, include_plotlyjs=True, post_script=script,
                      div_id='nominal50-latent', config={'responsive': True, 'displaylogo': False})
    intro = '<p style="font:16px system-ui;padding:20px">相同轨迹、相同样本；下拉菜单切换模型及环境 / motion / phase 着色。每个模型独立拟合 t-SNE，图上全局间距不可直接比较。真实距离见点击结果。黑色为 nominal。</p>'
    (output / 'tsne_interactive.html').write_text(html.replace('<body>', '<body>' + intro))
    (output / 'interactive_payload.json').write_text(json.dumps(payload))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--subset', default='common_full', choices=('common_full', 'memory_full', 'long_full'))
    parser.add_argument('--model', action='append', default=[], metavar='NAME=LABEL',
                        help='Select and label a model from metadata.models; repeat in display order.')
    args = parser.parse_args()
    metadata = json.loads((args.input / 'metadata.json').read_text())
    models = dict(MODELS)
    if args.model:
        models = {}
        for spec in args.model:
            key, label = spec.split('=', 1)
            if key not in metadata['models'] or key in models or not label.strip():
                raise ValueError(f'Invalid or duplicate plot model: {spec}')
            models[key] = (metadata['models'][key], label)
    if 'memory350' not in models:
        raise ValueError('Include memory350 to render the primary checkpoint panel.')
    data_file = 'main_data.npz' if args.subset == 'common_full' else args.subset + '_data.npz'
    with np.load(args.input / 'analysis' / data_file) as f:
        data = {k: f[k] for k in f.files}
    with np.load(args.input / 'analysis' / (args.subset + '_pairs.npz')) as f:
        pairs = {k: f[k] for k in f.files}
    output = args.input / 'plots'
    output.mkdir(exist_ok=True)
    ids, worlds, env_palette, env_names, motion_palette = labels_and_selection(data, metadata)
    sub = {k: v[ids] for k, v in data.items()}
    manifest = {'seed': 20260911, 'selected_worlds': worlds.tolist(),
                'subset': args.subset,
                'models': models, 'checkpoint': metadata['checkpoint'],
                'selected_source_rows': sub['source_row'].tolist(), 'samples': len(ids),
                'selection': '16 random DR worlds plus at most three nominal samples per motion, before fitting',
                'input': 'raw 64-dimensional latent', 'embeddings': {}}
    coordinates = {key: fit_embedding(sub[field], output, key + '_raw_p30', 30, 1500, manifest)
                   for key, (field, _) in models.items()}
    fig, axes = plt.subplots(len(models), 3, figsize=(19, 5 * len(models)), squeeze=False)
    for row, (key, (_, name)) in enumerate(models.items()):
        three_panels(axes[row], coordinates[key], sub, env_palette, env_names, motion_palette,
                     name, legend=False)
    count = metadata['arguments']['num_envs']
    profile_label = (f'{count // 2} nominal + {count // 2} DR; no extra limb loads / pushes'
                     if metadata['profile'] == 'common'
                     else f'{count} DR worlds; four-limb loads + tracker pushes')
    fig.suptitle(profile_label + '\nIdentical trajectories; independent t-SNE fits. Compare label mixing, not global gaps.', fontsize=17)
    fig.tight_layout(rect=(0, 0, 1, .95))
    save_figure(fig, output, 'tsne_comparison', pdf=True)
    fig, axes = plt.subplots(1, 3, figsize=(19, 7))
    three_panels(axes, coordinates['memory350'], sub, env_palette, env_names, motion_palette,
                 models['memory350'][1])
    fig.suptitle(f'{models["memory350"][1]} | {profile_label}', fontsize=16)
    fig.tight_layout(rect=(0, .1, 1, .95))
    save_figure(fig, output, 'tsne_nominal50', pdf=True)
    env = np.where(sub['nominal'], -1, sub['world'])
    values = []
    for field, _ in models.values():
        z = sub[field].astype(float)
        values.append(squareform(pdist(z / np.linalg.norm(z, axis=1, keepdims=True))))
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 6),
                             constrained_layout=True, squeeze=False)
    axes = axes.ravel()
    centers = [np.flatnonzero(env == k).mean() for k in np.unique(env)]
    names = [env_names[k] for k in np.unique(env)]
    for ax, value, (_, title) in zip(axes, values, models.values()):
        im = ax.imshow(value, cmap='magma', vmin=0, vmax=max(v.max() for v in values))
        ax.set_xticks(centers, names, rotation=90, fontsize=7)
        ax.set_yticks(centers, names, fontsize=7)
        ax.set_title(title)
    fig.colorbar(im, ax=axes, label='True 64-D unit-latent distance', shrink=.75)
    save_figure(fig, output, 'unit_distance_heatmaps', pdf=True)
    nrows = (len(models) + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(15, 5 * nrows),
                             constrained_layout=True, squeeze=False)
    categories = [('nominal_cross_motion', 'Nominal, cross motion', '#111111'),
                  ('dr_same_world_cross_motion_disjoint', 'Same DR, cross motion, disjoint memory', '#2676b5'),
                  ('dr_different_world_same_motion_near_phase', 'Different DR, matched motion / phase', '#d25d32')]
    for ax, (field, title) in zip(axes.flat, models.values()):
        z = data[field].astype(float)
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        for key, name, color in categories:
            if key + '_left' not in pairs or not len(pairs[key + '_left']):
                continue
            distance = np.linalg.norm(z[pairs[key + '_left']] - z[pairs[key + '_right']], axis=1)
            ax.hist(distance, bins=np.linspace(0, 2, 81), density=True, histtype='step',
                    linewidth=2, color=color, label=name)
        ax.set(xlabel='True unit-latent distance', ylabel='Density', title=title, xlim=(0, 2))
        ax.legend(fontsize=9)
    for ax in list(axes.flat)[len(models):]:
        ax.set_visible(False)
    save_figure(fig, output, 'distance_distributions', pdf=True)
    interactive(output, sub, coordinates, env_palette, motion_palette, models)
    (output / 'tsne_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'samples': len(ids)}), flush=True)


if __name__ == '__main__':
    main()
