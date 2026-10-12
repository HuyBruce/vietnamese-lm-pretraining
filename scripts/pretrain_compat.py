"""Narrow, audited logging-only upgrades; never relax data/config/runtime checks."""
import json
from pathlib import Path


def select_contract(current, previous, upgrades=None):
    if previous == current:
        return previous
    if upgrades is None:
        upgrades = json.loads(Path(__file__).with_name('pretrain_logging_upgrade.json').read_text(encoding='utf-8'))
    old_sources = previous.get('source_sha256', {})
    new_sources = current.get('source_sha256', {})
    if old_sources.keys() != new_sources.keys():
        raise RuntimeError('Run config/data/code/runtime differs; use original settings or a new run directory')
    upgraded = dict(previous, source_sha256=dict(old_sources))
    for name, old_hash in old_sources.items():
        if old_hash != new_sources[name]:
            if upgrades.get(name) != {'before': old_hash, 'after': new_sources[name]}:
                raise RuntimeError('Run config/data/code/runtime differs; source upgrade is not audited')
            upgraded['source_sha256'][name] = new_sources[name]
    if upgraded != current:
        raise RuntimeError('Run config/data/code/runtime differs; use original settings or a new run directory')
    # Retain the original contract in the run and its checkpoints. Each session
    # separately records the actually executed source hashes for provenance.
    return previous