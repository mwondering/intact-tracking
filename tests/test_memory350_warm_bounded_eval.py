import copy
import hashlib
import json

import pytest

from intact_tracking.memory350_policy_checkpoint_eval import (
    WARM_BOUNDED_CASES, WARM_BOUNDED_VERSION, load_protocol, resumed_evaluation_metadata)
from intact_tracking.memory350_policy_protocol import PERIODIC_VERSION


def protocol(tmp_path):
    manifest = tmp_path / 'motions.txt'
    manifest.write_text('motion\n')
    return {'version': WARM_BOUNDED_VERSION, 'interval_updates': 100,
        'motion_manifest': str(manifest), 'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'motions': 1, 'steps': 1000, 'warmup_steps': 500, 'seed': 20001,
        'cases': copy.deepcopy(WARM_BOUNDED_CASES), 'primary_cases': ['mixture_warm'],
        'global_metrics': True, 'primary_physics': {'limb_max_masses_kg': [2.5, 2.5, 4, 4]}}


def test_bounded_warm_protocol_and_masses(tmp_path):
    value = protocol(tmp_path)
    path = tmp_path / 'protocol.json'
    path.write_text(json.dumps(value))
    loaded, files = load_protocol(path)
    assert files == ['motion']
    assert loaded['cases']['upper_train_warm']['masses'] == [2.5, 2.5, 4, 4]
    value['cases']['upper_train_warm']['masses'] = [4, 4, 4, 4]
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='hand/shin'):
        load_protocol(path)


def test_protocol_revision_requires_explicit_change_and_records_boundary(tmp_path):
    new = protocol(tmp_path)
    old = {**copy.deepcopy(new), 'version': PERIODIC_VERSION}
    old['cases'] = {f'all_{mass}_{mode}': {'masses': [mass]*4, 'memory_start': mode}
                    for mass in (0, 4) for mode in ('cold', 'warm')}
    original = {'protocol': old, 'protocol_sha256': 'old', 'protocol_file': 'old.json', 'enabled_after_update': 0}
    current = {'protocol': new, 'protocol_sha256': 'new', 'protocol_file': 'new.json'}
    with pytest.raises(ValueError, match='without an allowed'):
        resumed_evaluation_metadata(original, current, 594)
    result = resumed_evaluation_metadata(original, current, 594, allow_change=True)
    assert result['enabled_after_update'] == 594
    assert result['revision']['from_sha256'] == 'old'
    assert result['revision']['to_sha256'] == 'new'
    assert original['enabled_after_update'] == 0
    unchanged = resumed_evaluation_metadata(result, current, 800)
    assert unchanged['enabled_after_update'] == 594
    assert unchanged['revision'] == result['revision']


@pytest.mark.parametrize('key,value', [('seed', 2), ('steps', 2000), ('warmup_steps', 100),
                                      ('manifest_sha256', 'changed'), ('global_metrics', False)])
def test_revision_rejects_changed_paired_conditions(tmp_path, key, value):
    new = protocol(tmp_path)
    old = {**copy.deepcopy(new), 'version': PERIODIC_VERSION}
    current = copy.deepcopy(new)
    current[key] = value
    with pytest.raises(ValueError, match='paired condition'):
        resumed_evaluation_metadata({'protocol': old, 'protocol_sha256': 'a', 'protocol_file': 'a.json'},
                                    {'protocol': current, 'protocol_sha256': 'b'}, 700, allow_change=True)


def test_history5_rejects_out_of_range_payload_before_creating_environment():
    from intact_tracking.memory350_history5_policy import configure_physics
    for masses in ([4, 4, 4, 4], [2.5, 2.5, 4.01, 4], [float('nan'), 0, 0, 0], [-1, 0, 0, 0]):
        with pytest.raises(ValueError, match='payload limits'):
            configure_physics(None, 0, profile='tracker_dr_plus_limb_payload', fixed_masses=masses)
