"""Audit two nominal misroutes and attribute encoder sensitivity on saved histories.

Only GPU neural-network inference; branch swaps are input interventions, not
physically replayed trajectories or evidence that dynamics is unidentifiable.
"""
from contextlib import nullcontext
from pathlib import Path
import hashlib
import json

import numpy as np
import torch

from intact_tracking.memory350_inference import load_memory350_checkpoint


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = Path('runs/limb_context_20260916_dr_bad_history')
    out.mkdir(exist_ok=True)
    assert not any(out.iterdir()), 'Refuse to overwrite existing results'
    source = Path('runs/limb_context_20260915_dr_center_weight04_scale02_positive02/cluster_check_030750/capped_loads')
    meta = json.loads((source / 'metadata.json').read_text())
    cached = np.load(source / 'latents.npz')
    reencoded = np.load('runs/limb_context_20260916_dr_readout_u35857/predictions.npz')
    assert np.array_equal(cached['source_row'], reencoded['all_source_rows'])
    nominal = np.load('runs/limb_context_20260916_nominal_cluster_contamination/nominal_latents.npz')['dr_label'].mean(0)
    cp = load_memory350_checkpoint('runs/limb_context_20260915_dr_center_weight04_scale02_positive02/stage1_8192/update_035857.pt', device='cuda:0')
    base = {}
    natural = []
    skipped = []
    provenance = []
    for info in meta['queries']:
        step = info['step']
        if step < 1700:
            continue
        path = Path(info['path'])
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        assert digest == info['sha256']
        provenance.append(info)
        q = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        for world in (401, 293):
            index = torch.nonzero(q['world'] == world).item()
            row = np.flatnonzero((cached['world'] == world) & (cached['step'] == step)).item()
            fields = {key: q[key][index].clone() for key in ('short', 'short_valid', 'long', 'long_valid')}
            if not (fields['short_valid'].all() and fields['long_valid'].all()):
                skipped.append({'world': world, 'step': step,
                                'short_valid': int(fields['short_valid'].sum()),
                                'long_valid': int(fields['long_valid'].sum())})
                continue
            assert all(torch.isfinite(value).all() for value in fields.values())
            short = fields['short'].numpy()
            long = fields['long'].numpy()
            assert np.array_equal(short[:-1, 100:], short[1:, :71])
            assert np.array_equal(long[:, :-1, 100:], long[:, 1:, :71])
            base[world, step] = fields
            z = reencoded['latent_current'][row].astype(float)
            z /= np.linalg.norm(z)
            record = {
                'world': world, 'step': step, 'source_row': int(cached['source_row'][row]),
                'episode': int(cached['episode'][row]), 'memory_session': int(cached['memory_session'][row]),
                'motion': meta['motion_files'][int(cached['motion'][row])],
                'motion_step': int(cached['motion_step'][row]),
                'cached_dr_latent_nominal_distance': float(np.linalg.norm(z - nominal)),
                'short_root_height_mean_m': float(short[:, 2].mean()),
                'short_root_position_mean_m': short[:, :3].mean(0).tolist(),
                'short_root_speed_component_rms_mps': float(np.sqrt(np.mean(short[:, 7:10]**2))),
                'short_joint_speed_rms_radps': float(np.sqrt(np.mean(short[:, 42:71]**2))),
                'long_joint_speed_rms_radps': float(np.sqrt(np.mean(long[:, :, 42:71]**2))),
                'long_chunk_joint_speed_rms_radps': np.sqrt(np.mean(long[:, :, 42:71]**2, axis=(1, 2))).tolist(),
                'short_action_change_rms_rad': float(np.sqrt(np.mean(np.diff(short[:, 71:100], axis=0)**2))),
                'short_target_minus_q_rms_rad': float(np.sqrt(np.mean((short[:, 71:100]-short[:, 13:42])**2))),
                'long_root_height_mean_m': float(long[:, :, 2].mean()),
                'long_root_speed_component_rms_mps': float(np.sqrt(np.mean(long[:, :, 7:10]**2))),
                'long_nonzero_transition_count': int(np.any(long != 0, axis=-1).sum()),
                'short_valid': 50, 'long_valid': 30,
            }
            natural.append(record)
    cases = []
    inputs = []
    for record in natural:
        cases.append({'type': 'natural', 'world': record['world'], 'step': record['step']})
        inputs.append(base[record['world'], record['step']])
    for world, good, bad in ((401, 3000, 3200), (293, 2000, 2400), (293, 2600, 2400)):
        for short_step, long_step in ((good, bad), (bad, good)):
            fields = {key: base[world, short_step if key.startswith('short') else long_step][key].clone()
                      for key in ('short', 'short_valid', 'long', 'long_valid')}
            cases.append({'type': 'branch_swap', 'world': world, 'short_step': short_step, 'long_step': long_step})
            inputs.append(fields)
        fields = {key: value.clone() for key, value in base[world, bad].items()}
        offset = -fields['short'][-1, :2].clone()
        for key in ('short', 'long'):
            fields[key][..., :2] += offset
            fields[key][..., 100:102] += offset
        cases.append({'type': 'horizontal_translation', 'world': world, 'step': bad, 'offset_m': offset.tolist()})
        inputs.append(fields)
    for precision in ('fp32', 'bf16'):
        latents = []
        for start in range(0, len(inputs), 32):
            batch = {key: torch.stack([x[key] for x in inputs[start:start+32]]).to('cuda:0')
                     for key in ('short', 'short_valid', 'long', 'long_valid')}
            def normalized(value):
                return torch.cat(((value[..., :71]-cp.state_mean)/cp.state_std,
                                  (value[..., 71:100]-cp.action_mean)/cp.action_std,
                                  (value[..., 100:]-cp.state_mean)/cp.state_std), dim=-1)
            short, long = normalized(batch['short']), normalized(batch['long'])
            with torch.autocast('cuda', dtype=torch.bfloat16) if precision == 'bf16' else nullcontext():
                latent = cp.encoder(short[..., :71], short[..., 71:100], short[..., 100:],
                                    batch['short_valid'], long, batch['long_valid']).float()
            latents.append(torch.nn.functional.normalize(latent, dim=-1).cpu().numpy())
        z = np.concatenate(latents).astype(float)
        for i, case in enumerate(cases):
            case[precision + '_distance_to_nominal'] = float(np.linalg.norm(z[i]-nominal))
            if i < len(natural):
                row = np.flatnonzero(cached['source_row'] == natural[i]['source_row']).item()
                old = reencoded['latent_current'][row].astype(float)
                old /= np.linalg.norm(old)
                case[precision + '_unit_latent_l2_to_cached_cpu_fp32'] = float(np.linalg.norm(z[i]-old))
                if precision == 'fp32' and case[precision + '_unit_latent_l2_to_cached_cpu_fp32'] >= 1e-4:
                    print(json.dumps({'precision_difference': case}), flush=True)
        np.save(out / (precision + '_unit_latents.npy'), z)
    report = {
        'checkpoint': {'path': cp.path, 'sha256': cp.sha256},
        'protocol': 'Reencode audited raw histories on GPU0 in FP32 and BF16; compare cached CPU FP32. All natural windows have short50/long30 valid, finite data and exact state transition continuity within short and within each 10-step long chunk. Long chunks may legitimately span motion resets, not physics-session changes.',
        'intervention_limits': 'Branch swaps splice short and long from different times of the same DR world; not physically continuous replay. Translation shifts both state and next-state XY equally throughout the history. Neither experiment proves absent dynamics information or establishes the root training cause.',
        'raw_queries': provenance, 'natural': natural, 'encoder_cases': cases,
        'skipped_incomplete_windows': skipped,
    }
    (out / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    for record, case in zip(natural, cases):
        print(json.dumps({key: record[key] for key in ('world', 'step', 'cached_dr_latent_nominal_distance', 'short_joint_speed_rms_radps', 'long_joint_speed_rms_radps')} | {'fp32': case['fp32_distance_to_nominal'], 'bf16': case['bf16_distance_to_nominal']}), flush=True)
    for case in cases[len(natural):]:
        print(json.dumps(case), flush=True)


if __name__ == '__main__':
    main()
