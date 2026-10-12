"""Independently audit committed pretrain data without changing pipeline checkpoints."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import sentencepiece as spm
from prepare_news_inventory import atomic_json, file_sha


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sample_rows(path, columns):
    """Read first, middle and last row groups, not the whole shard."""
    with pq.ParquetFile(path) as parquet:
        for group in sorted({0, parquet.num_row_groups // 2, parquet.num_row_groups - 1}):
            rows = parquet.read_row_group(group, columns=columns).to_pylist()
            for index in sorted({0, len(rows) // 2, len(rows) - 1}):
                yield rows[index]


def audit(root, committed_only=False):
    clean = json.loads((root / 'clean/manifest.json').read_text())
    tokenizer = json.loads((root / 'checkpoints/tokenizer.json').read_text())
    require(file_sha(root / tokenizer['model']) == tokenizer['sha256'], 'Tokenizer checksum mismatch')
    model = spm.SentencePieceProcessor(model_file=str(root / tokenizer['model']))
    require(model.get_piece_size() == tokenizer['actual_vocab'], 'Vocabulary mismatch')
    require(model.eos_id() == tokenizer['eos_id'] and model.pad_id() == tokenizer['pad_id'], 'Special token mismatch')
    manifest = None
    if not committed_only:
        manifest = json.loads((root / 'binary/manifest.json').read_text())
        require(manifest.get('complete') is True, 'Pipeline verification is not complete')
        verification = json.loads((root / 'reports/verification.json').read_text())
        require(verification.get('passed') is True, 'Pipeline verification did not pass')
        require(manifest['tokenizer'] == tokenizer, 'Manifest tokenizer mismatch')
    records = []
    samples = 0
    totals = Counter()
    for index, source in enumerate(clean['files']):
        name = f"{source['split']}-{index:05d}"
        marker = root / 'checkpoints' / (name + '.json')
        if committed_only and not marker.exists():
            continue
        spec = json.loads(marker.read_text())
        require(spec['source_sha256'] == source['sha256'], f'{name}: source mismatch')
        require(spec['tokenizer_sha256'] == tokenizer['sha256'], f'{name}: tokenizer mismatch')
        require(spec['split'] == source['split'], f'{name}: split mismatch')
        require(spec['documents'] == source['rows'], f'{name}: document count mismatch')
        require(spec['context_length'] == 4096, f'{name}: context mismatch')
        for path, expected in [(source['path'], source['sha256']),
                               (spec['file'], spec['sha256']),
                               (spec['documents_file'], spec['documents_sha256'])]:
            require(file_sha(root / path) == expected, f'Checksum mismatch: {path}')
        selected = {row['id']: row['text'] for row in sample_rows(root / source['path'], ['id', 'text'])}
        tokens = np.memmap(root / spec['file'], mode='r', dtype=spec['dtype'])
        try:
            require(len(tokens) == spec['stored_tokens'], f'{name}: stored length mismatch')
            require(len(tokens) == spec['sequences'] * spec['context_length'] + 1, f'{name}: packing mismatch')
            found = set()
            with pq.ParquetFile(root / spec['documents_file']) as boundaries:
                require(boundaries.metadata.num_rows == spec['documents'], f'{name}: offset count mismatch')
                for batch in boundaries.iter_batches(batch_size=8192):
                    for row in batch.to_pylist():
                        if row['id'] not in selected:
                            continue
                        start, end = row['start'], row['end']
                        require(0 <= start < end <= spec['tokens'], f'{name}: invalid offset')
                        require(int(tokens[end - 1]) == model.eos_id(), f'{name}: missing EOS')
                        ids = tokens[start:end - 1].tolist()
                        require(model.unk_id() not in ids, f'{name}: UNK in sample')
                        require(model.decode(ids) == selected[row['id']], f'{name}: decoded text mismatch')
                        found.add(row['id'])
            require(found == set(selected), f'{name}: missing sample offsets')
            samples += len(found)
        finally:
            tokens._mmap.close()
        records.append(spec)
        totals[source['split'] + '_documents'] += spec['documents']
        totals[source['split'] + '_tokens'] += spec['tokens']
        print(f'Checked {name}: checksums and {len(found)} document round-trips', flush=True)
    require(records, 'No committed binary shards found')
    loader_checks = {}
    if manifest is not None:
        require(sorted(records, key=lambda s: s['file']) == manifest['shards'], 'Manifest/checkpoint mismatch')
        require(sum(s['documents'] for s in records) == manifest['total_documents'] == clean['documents'], 'Total document mismatch')
        require(sum(s['tokens'] for s in records) == manifest['total_tokens'], 'Total token mismatch')
        require(sum(s['sequences'] for s in records) == manifest['total_sequences'], 'Total sequence mismatch')
        loader_checks = check_loaders(root)
    report = dict(passed=True, scope='committed_only' if committed_only else 'complete_dataset',
                  checked_at=datetime.now(timezone.utc).isoformat(),
                  binary_shards_checked=len(records), expected_shards=len(clean['files']),
                  roundtrip_documents=samples, totals=dict(totals), dataloader=loader_checks)
    target = 'committed_audit.json' if committed_only else 'independent_verification.json'
    atomic_json(root / 'reports' / target, report)
    return report


def check_loaders(root):
    import torch
    from torch.utils.data import DataLoader
    from pretrain_dataset import CausalTokenDataset
    result = {}
    for split in ('train', 'validation'):
        dataset = CausalTokenDataset(root, split=split, max_open_shards=2)
        checked = 0
        try:
            base = 0
            for spec in dataset.shards:
                tokens = np.memmap(root / spec['file'], mode='r', dtype=spec['dtype'])
                try:
                    for local in sorted({0, spec['sequences'] // 2, spec['sequences'] - 1}):
                        item = dataset[base + local]
                        start = local * dataset.context
                        expected = np.array(tokens[start:start + dataset.context + 1], dtype=np.int64)
                        targets = expected[1:].copy()
                        targets[targets == dataset.pad] = -100
                        require(item['input_ids'].dtype == torch.int64, 'Input dtype mismatch')
                        require(item['targets'].dtype == torch.int64, 'Target dtype mismatch')
                        np.testing.assert_array_equal(item['input_ids'].numpy(), expected[:-1])
                        np.testing.assert_array_equal(item['targets'].numpy(), targets)
                        checked += 1
                finally:
                    tokens._mmap.close()
                base += spec['sequences']
            batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=True,
                                        generator=torch.Generator().manual_seed(20260929), num_workers=0)))
            require(tuple(batch['input_ids'].shape) == (2, 4096), 'Batch shape mismatch')
            require(tuple(batch['targets'].shape) == (2, 4096), 'Target batch shape mismatch')
            require(len(dataset._maps) <= 2, 'Open shard limit exceeded')
            result[split] = dict(sequences=len(dataset), checked_sequences=checked,
                                 batch_shape=list(batch['input_ids'].shape), dtype='torch.int64')
        finally:
            for mmap in dataset._maps.values():
                mmap._mmap.close()
            dataset._maps.clear()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'DataProcessed/pretrain_ready')
    parser.add_argument('--committed-only', action='store_true',
                        help='Audit available committed shards without claiming the dataset is ready.')
    args = parser.parse_args()
    print(json.dumps(audit(args.output.resolve(), args.committed_only), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
