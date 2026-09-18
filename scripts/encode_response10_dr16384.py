"""Reencode the same 16 selected histories/world with the frozen response10 encoder."""

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from intact_tracking.memory350_inference import load_memory350_checkpoint


ROOT = Path('runs/limb_context_20260916_dr16384')
OUT = Path('runs/limb_context_20260917_response10_radial_bins')
CHECKPOINT = Path('runs/limb_context_20260912_memory350_response_window_ablation/response10/stage1_8192/update_015000.pt')
EXPECTED_SHA = 'db5d7bd7df026db163b9eedf588d37dcc81b55551d28afe3744f67a8ab2642ac'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


@torch.inference_mode()
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint = load_memory350_checkpoint(CHECKPOINT, device='cuda')
    assert checkpoint.sha256 == EXPECTED_SHA
    with np.load(ROOT / 'analysis/selected_worlds.npz') as source:
        data = {key: source[key] for key in ('world', 'nominal', 'shard', 'step', 'source_rows', 'physics', 'names', 'span', 'weights', 'radius', 'motion', 'episode')}
    z = np.full((*data['step'].shape, 64), np.nan, np.float32)
    records = []
    for shard in range(8):
        cache = OUT / f'reencoded_shard_{shard:02d}.npz'
        record_path = OUT / f'reencoded_shard_{shard:02d}.json'
        selected = np.flatnonzero(data['shard'] == shard)
        if cache.exists() and record_path.exists():
            record = json.loads(record_path.read_text())
            assert record['checkpoint_sha256'] == EXPECTED_SHA and record['cache_sha256'] == digest(cache)
            with np.load(cache) as saved:
                np.testing.assert_array_equal(saved['world'], data['world'][selected])
                np.testing.assert_array_equal(saved['step'], data['step'][selected])
                z[selected] = saved['z']
            records.append(record)
            continue
        metadata_path = ROOT / f'shard_{shard:02d}/metadata.json'
        metadata = json.loads(metadata_path.read_text())
        assert metadata['complete'] and metadata['tracker_sha256'] == checkpoint.tracker_sha256
        record = dict(shard=shard, checkpoint_sha256=EXPECTED_SHA, metadata_sha256=digest(metadata_path), queries=[])
        for info in metadata['queries']:
            row, column = np.where(data['step'][selected] == info['step'])
            if not len(row):
                continue
            path = Path(info['path'])
            assert digest(path) == info['sha256'], str(path)
            query = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
            ids = data['source_rows'][selected[row], column] % len(query['world'])
            np.testing.assert_array_equal(query['world'][ids].numpy(), data['world'][selected[row]])
            assert query['short_valid'][ids].all() and query['long_valid'][ids].all()
            for start in range(0, len(ids), 256):
                local = torch.from_numpy(ids[start:start+256])

                def normalize(raw):
                    raw = raw[local].to('cuda')
                    return torch.cat(((raw[..., :71]-checkpoint.state_mean)/checkpoint.state_std,
                                      (raw[..., 71:100]-checkpoint.action_mean)/checkpoint.action_std,
                                      (raw[..., 100:]-checkpoint.state_mean)/checkpoint.state_std), dim=-1)

                short, long = normalize(query['short']), normalize(query['long'])
                output = checkpoint.encoder(short[..., :71], short[..., 71:100], short[..., 100:],
                                            query['short_valid'][local].to('cuda'), long,
                                            query['long_valid'][local].to('cuda'))
                z[selected[row[start:start+256]], column[start:start+256]] = output.cpu().numpy()
            record['queries'].append(dict(step=info['step'], sha256=info['sha256'], samples=len(ids)))
            print(json.dumps(dict(shard=shard, step=info['step'], samples=len(ids), seconds=round(time.monotonic()-started, 1))), flush=True)
            del query
        assert np.isfinite(z[selected]).all()
        np.savez_compressed(cache, world=data['world'][selected], step=data['step'][selected], z=z[selected])
        record['cache_sha256'] = digest(cache)
        record_path.write_text(json.dumps(record, indent=2)+'\n')
        records.append(record)
    assert np.isfinite(z).all()
    np.savez_compressed(OUT / 'response10_selected_worlds.npz', z=z, **data)
    manifest = dict(checkpoint_path=str(CHECKPOINT), checkpoint_sha256=EXPECTED_SHA,
                    source=str(ROOT / 'analysis/selected_worlds.npz'), source_sha256=digest(ROOT / 'analysis/selected_worlds.npz'),
                    script_sha256=digest(Path(__file__)), shards=records, device=torch.cuda.get_device_name(),
                    precision='FP32, no autocast, TF32 disabled', worlds=len(z), dr_worlds=int((~data['nominal']).sum()),
                    nominal_worlds=int(data['nominal'].sum()), windows_per_world=z.shape[1],
                    archive_sha256=digest(OUT / 'response10_selected_worlds.npz'), seconds=time.monotonic()-started)
    (OUT / 'encoding_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(complete=True, seconds=manifest['seconds'])), flush=True)


if __name__ == '__main__':
    main()
