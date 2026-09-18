"""Complete response10 encoding for every mature query in the saved 3200-step runs."""

import json
from pathlib import Path
import time

import numpy as np
import torch

from encode_response10_dr16384 import ROOT, CHECKPOINT, EXPECTED_SHA, digest
from intact_tracking.memory350_inference import load_memory350_checkpoint

PREVIOUS = Path('runs/limb_context_20260917_response10_radial_bins')
OUT = Path('runs/limb_context_20260917_response10_environment_bins')
ACTIVITY = ('short_root_xy_speed_rms_mps', 'full_root_xy_speed_rms_mps',
            'short_joint_speed_rms_radps', 'full_joint_speed_rms_radps')


def activity(short, long):
    short_xy = short[..., 7:9].square().sum(-1).mean(1)
    long_xy = long[..., 7:9].square().sum(-1).mean((1, 2))
    short_joint = short[..., 42:71].square().mean((1, 2))
    long_joint = long[..., 42:71].square().mean((1, 2, 3))
    return torch.stack((short_xy.sqrt(), ((50*short_xy+300*long_xy)/350).sqrt(),
                        short_joint.sqrt(), ((50*short_joint+300*long_joint)/350).sqrt()), -1).numpy()


@torch.inference_mode()
def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(3)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic()
    checkpoint = load_memory350_checkpoint(CHECKPOINT, device='cuda')
    assert checkpoint.sha256 == EXPECTED_SHA
    records = []
    for shard in range(8):
        target = OUT / f'full_shard_{shard:02d}.npz'
        record_path = OUT / f'full_shard_{shard:02d}.json'
        if target.exists() and record_path.exists():
            record = json.loads(record_path.read_text())
            assert record['sha256'] == digest(target) and record['checkpoint_sha256'] == EXPECTED_SHA
            records.append(record)
            continue
        folder = ROOT / f'shard_{shard:02d}'
        meta = json.loads((folder/'metadata.json').read_text())
        with np.load(folder/'physics.npz') as saved:
            world, physics, nominal = saved['world'], saved['values'], saved['nominal']
        n = len(world)
        with np.load(folder/'latents.npz') as saved:
            grids = {key: saved[key].reshape(32, n).T for key in
                     ('step', 'motion', 'motion_step', 'episode', 'short_steps', 'long_chunks')}
            np.testing.assert_array_equal(saved['world'].reshape(32, n), np.broadcast_to(world, (32, n)))
        valid = (grids['short_steps'] == 50) & (grids['long_chunks'] == 30)
        z = np.full((n, 32, 64), np.nan, np.float32)
        speeds = np.full((n, 32, len(ACTIVITY)), np.nan, np.float32)
        with np.load(PREVIOUS/f'reencoded_shard_{shard:02d}.npz') as old:
            np.testing.assert_array_equal(old['world'], world)
            previous_step = old['step']
            col = previous_step//100-1
            row = np.broadcast_to(np.arange(n)[:, None], col.shape)
            assert valid[row, col].all()
            z[row, col] = old['z']
        reused = int(np.isfinite(z[..., 0]).sum())
        record = dict(shard=shard, checkpoint_sha256=EXPECTED_SHA, reused_windows=reused,
                      queries=[], motion_files=meta['motion_files'])
        for col, info in enumerate(meta['queries']):
            ids = np.flatnonzero(valid[:, col])
            if not len(ids):
                continue
            assert info['step'] == (col+1)*100
            path = Path(info['path'])
            assert digest(path) == info['sha256']
            query = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
            np.testing.assert_array_equal(query['world'].numpy(), world)
            assert query['short_valid'][ids].all() and query['long_valid'][ids].all()
            for start in range(0, len(ids), 256):
                rows = ids[start:start+256]
                indices = torch.from_numpy(rows)
                short, long = query['short'][indices], query['long'][indices]
                speeds[rows, col] = activity(short, long)
                missing = np.isnan(z[rows, col, 0])
                if not missing.any():
                    continue

                def normalize(raw):
                    raw = raw[torch.from_numpy(missing)].to('cuda')
                    return torch.cat(((raw[..., :71]-checkpoint.state_mean)/checkpoint.state_std,
                                      (raw[..., 71:100]-checkpoint.action_mean)/checkpoint.action_std,
                                      (raw[..., 100:]-checkpoint.state_mean)/checkpoint.state_std), -1)

                short, long = normalize(short), normalize(long)
                chosen = indices[torch.from_numpy(missing)]
                out = checkpoint.encoder(short[..., :71], short[..., 71:100], short[..., 100:],
                                         query['short_valid'][chosen].to('cuda'), long,
                                         query['long_valid'][chosen].to('cuda'))
                z[rows[missing], col] = out.cpu().numpy()
            record['queries'].append(dict(step=info['step'], windows=len(ids), sha256=info['sha256']))
            print(json.dumps(dict(shard=shard, step=info['step'], seconds=round(time.monotonic()-started, 1))), flush=True)
            del query
        assert np.isfinite(z[valid]).all() and np.isfinite(speeds[valid]).all()
        np.testing.assert_array_equal(valid.sum(1), meta['full_query_counts'])
        np.savez_compressed(target, world=world, physics=physics, nominal=nominal, z=z, valid=valid,
                            activity=speeds, activity_names=np.asarray(ACTIVITY), **grids)
        record.update(full_windows=int(valid.sum()), encoded_new_windows=int(valid.sum())-reused,
                      full_windows_min=int(valid.sum(1).min()), full_windows_max=int(valid.sum(1).max()),
                      sha256=digest(target))
        record_path.write_text(json.dumps(record, indent=2)+'\n')
        records.append(record)
    (OUT/'encoding_manifest.json').write_text(json.dumps(dict(checkpoint_path=str(CHECKPOINT),
        checkpoint_sha256=EXPECTED_SHA, precision='FP32, no autocast or TF32',
        source=str(ROOT), previous=str(PREVIOUS), shards=records, script_sha256=digest(Path(__file__)),
        full_windows=sum(r['full_windows'] for r in records),
        encoded_new_windows=sum(r['encoded_new_windows'] for r in records),
        reused_windows=sum(r['reused_windows'] for r in records), seconds=time.monotonic()-started), indent=2)+'\n')
    print(json.dumps(dict(complete=True, seconds=time.monotonic()-started)), flush=True)


if __name__ == '__main__':
    main()
