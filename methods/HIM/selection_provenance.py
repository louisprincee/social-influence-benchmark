"""Validate cache inputs and explicitly audited, unchanged historical selections."""
from pathlib import Path
import hashlib
import json

HOME = Path(__file__).resolve().parent
ROOT = HOME.parents[1]
SELECTOR = HOME / 'methods/HIM/structural_selection.py'
FROZEN_SELECTOR = HOME / 'reproduction_sources/structural_selection_2026-10-30.py.txt'
REVISION_AUDIT = ROOT / 'results/HIM/structural_revision_verification_2026-10-31.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_selector_fingerprint(recorded, artifact=None):
    current = sha(SELECTOR)
    if recorded == current:
        return
    if artifact is None or not FROZEN_SELECTOR.exists() or recorded != sha(FROZEN_SELECTOR):
        raise ValueError('Selector fingerprint changed; reselect before reusing outcomes')
    audit = json.loads(REVISION_AUDIT.read_text())
    path = Path(artifact).resolve()
    key = str(path.relative_to(ROOT))
    record = audit['records'].get(key, {})
    if (audit['current_selector_sha256'] != current or
            audit['historical_selector_sha256'] != recorded or
            record.get('artifact_sha256') != sha(path) or
            record.get('primary_selection_unchanged') is not True):
        raise ValueError('Historical selection has no matching current-code audit')


def validate_input_hashes(saved, expected, artifact):
    if set(saved) != set(expected) or any(saved[k] != expected[k] for k in expected if k != 'selector'):
        raise ValueError('Experiment inputs changed; cached outcomes cannot be reused')
    validate_selector_fingerprint(saved['selector'], artifact)


def validate_cached_structural(saved, expected, selections, artifact=None):
    for key, value in expected.items():
        if key != 'selector_sha256' and saved.get(key) != value:
            raise ValueError('Cached structural experiment input changed: ' + key)
    validate_selector_fingerprint(saved.get('selector_sha256'), artifact)
    if set(saved['methods']) != set(selections):
        raise ValueError('Cached method set changed')
    for method, seeds in selections.items():
        detail = saved['methods'][method]
        values = detail['activated_per_replicate']
        if (detail['seeds'] != seeds or len(values) != expected['evaluation_rounds'] or
                any(not len(seeds) <= x <= expected['nodes'] for x in values)):
            raise ValueError('Cached seed set or propagation outcomes differ: ' + method)
