"""Render the saved joint t-SNE as a clear, two-policy scatter plot.

This only changes presentation: no fitting, jitter, point relocation, or
selection based on clustering. Display environments come from the protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

MARKERS = ('o', 's', '^', 'v', 'D', 'P', 'X', '*')
EMBEDDING = 'tsne_p30_s921'


def render(root: Path) -> None:
    root = root.resolve()
    protocol = json.loads((root / 'protocol.json').read_text())
    array_path = root / 'analysis' / 'analysis_arrays.npz'
    source_hash = hashlib.sha256(array_path.read_bytes()).hexdigest()
    with np.load(array_path, allow_pickle=False) as arrays:
        coords = arrays[EMBEDDING]
        worlds = arrays['worlds']
    if coords.shape != (2, len(worlds), len(MARKERS), 2):
        raise ValueError(f'Unexpected t-SNE array shape: {coords.shape}')
    if not np.isfinite(coords).all():
        raise ValueError('Nonfinite t-SNE coordinates')
    selected_worlds = protocol['display_world_ids_chosen_before_collection']
    rows = []
    for world in selected_worlds:
        matches = np.flatnonzero(worlds == world)
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one row for preselected world {world}')
        rows.append(int(matches[0]))
    selected = coords[:, rows]
    flat = selected.reshape(-1, 2)
    lower, upper = flat.min(0), flat.max(0)
    margin = (upper - lower) * .09
    colors = [plt.get_cmap('tab20')(i) for i in range(len(rows))]

    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 11,
        'pdf.fonttype': 42, 'axes.spines.top': False,
        'axes.spines.right': False,
    })
    fig, axes = plt.subplots(1, 2, figsize=(14, 9), sharex=True, sharey=True)
    for policy, (ax, title) in enumerate(zip(
            axes, ('Frozen tracker', 'Residual latent policy'))):
        for i, world in enumerate(selected_worlds):
            points = selected[policy, i]
            for motion, marker in enumerate(MARKERS):
                ax.scatter(points[motion, 0], points[motion, 1],
                           marker=marker, s=58 if marker != '*' else 95,
                           color=colors[i], alpha=.92, edgecolors='white',
                           linewidths=.45, zorder=3)
            ax.annotate(f'E{world:03d}', np.median(points, axis=0),
                        xytext=(6, 7), textcoords='offset points',
                        fontsize=8.5, color='#354052', zorder=4)
        ax.set_title(title, fontsize=15, pad=13)
        ax.set_xlabel('t-SNE 1', fontsize=12, labelpad=9)
        ax.set_ylabel('t-SNE 2', fontsize=12, labelpad=9)
        ax.set_xlim(lower[0] - margin[0], upper[0] + margin[0])
        ax.set_ylim(lower[1] - margin[1], upper[1] + margin[1])
        ax.set_aspect('equal', adjustable='box')
        ax.tick_params(labelsize=9, colors='#667085')
        for spine in ax.spines.values():
            spine.set_color('#b7c0cb')

    fig.suptitle('t-SNE of context latents across motions',
                 fontsize=20, fontweight='normal', y=.975)
    fig.text(.5, .932,
             'Color = environment    |    Shape = motion    |    8 motions per environment',
             ha='center', fontsize=12, color='#475467')
    fig.legend([Patch(facecolor=color) for color in colors],
               [f'E{world:03d}' for world in selected_worlds],
               title='16 environments selected before data collection',
               loc='lower center', bbox_to_anchor=(.5, .114), ncol=8,
               frameon=False, fontsize=9, title_fontsize=10,
               columnspacing=1.6, handlelength=1.4)
    fig.legend([Line2D([], [], color='#505968', marker=marker, linestyle='',
                       markersize=7) for marker in MARKERS],
               [f'M{i + 1}' for i in range(len(MARKERS))],
               title='Motion identity (same M1-M8 in every environment)',
               loc='lower center', bbox_to_anchor=(.5, .048), ncol=8,
               frameon=False, fontsize=9, title_fontsize=10,
               columnspacing=1.6, handlelength=1.4)
    fig.text(.5, .018,
             f'One joint t-SNE fit: {len(worlds)} environments, 8 motions, 2 policies'
             '   |   perplexity = 30, seed = 921',
             ha='center', fontsize=9, color='#667085')
    fig.subplots_adjust(left=.066, right=.985, top=.87, bottom=.25, wspace=.17)
    output = root / 'analysis' / 'environment_motion_tsne_clean'
    for suffix in ('.png', '.pdf'):
        fig.savefig(output.with_suffix(suffix), dpi=240, bbox_inches='tight',
                    facecolor='white')
    plt.close(fig)

    # Preserve an exact record of every displayed coordinate and its labels.
    np.savez_compressed(output.with_suffix('.npz'),
                        coordinates=selected,
                        worlds=np.asarray(selected_worlds),
                        policies=np.asarray(['tracker', 'residual']),
                        motion_ids=np.arange(8))
    if hashlib.sha256(array_path.read_bytes()).hexdigest() != source_hash:
        raise RuntimeError('Source arrays changed during rendering')
    with np.load(output.with_suffix('.npz'), allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['coordinates'], coords[:, rows])
    metadata = {
        'source': str(array_path), 'source_sha256': source_hash,
        'embedding_key': EMBEDDING,
        'selected_worlds_from_precollection_protocol': selected_worlds,
        'points_per_policy': int(selected[0, ..., 0].size),
        'joint_fit_points': int(coords[..., 0].size),
        'coordinates_equal_to_saved_joint_tsne': True,
        'refit': False, 'jitter': False,
        'omitted_preselected_worlds': [],
        'presentation': 'Two policy panels; shared axes; only preselected worlds shown.',
    }
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({'png': str(output.with_suffix('.png')),
                      'pdf': str(output.with_suffix('.pdf')),
                      'points_per_policy': metadata['points_per_policy'],
                      'coordinates_verified': True}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path, help='Completed eight-motion experiment root')
    render(parser.parse_args().root)
