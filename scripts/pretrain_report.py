"""Export local JSONL history to CSV/JSON/Markdown without loading model weights."""
import argparse
import csv
import io
import json
import os
from pathlib import Path


def atomic_text(path, text, encoding='utf-8'):
    temporary = path.with_name(path.name + '.tmp')
    try:
        with temporary.open('w', encoding=encoding, newline='') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_history(path):
    rows, warnings = [], []
    if not path.exists():
        return rows, ['No metrics history available.']
    legacy_session = 0
    with path.open(encoding='utf-8', errors='replace') as stream:
        for number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
                if not isinstance(row, dict) or not isinstance(row.get('event'), str):
                    raise ValueError('Missing event')
            except ValueError:
                warnings.append(f'Malformed history line {number}; preserved in source, omitted from report.')
                continue
            if row['event'] == 'session_start' and not row.get('session_id'):
                legacy_session += 1
            row.setdefault('session_id', f'legacy-{legacy_session}')
            row['history_line'] = number
            row['status'] = 'event'
            rows.append(row)
    return rows, warnings


def summarize(rows, warnings):
    active = []
    committed_step = 0
    committed_tokens = None
    committed_training_seconds = None
    total_steps = None
    sessions = {}
    for row in rows:
        event = row['event']
        session = sessions.setdefault(row['session_id'], dict(session_id=row['session_id'], status='incomplete'))
        if event == 'session_start':
            step = row['global_step']
            # A resumed checkpoint is authoritative. Never sum repeated steps.
            retained = []
            for old in active:
                if old['global_step'] > step:
                    old['status'] = 'rolled_back'
                else:
                    retained.append(old)
            active = retained
            committed_step = step
            committed_tokens = row.get('tokens_seen')
            committed_training_seconds = row.get('training_seconds')
            total_steps = row.get('total_steps', total_steps)
            session.update(start_step=step, started_at=row.get('timestamp', row.get('time')),
                           actual_source_sha256=row.get('actual_source_sha256'))
        elif event in ('train', 'validation_quick', 'validation_full'):
            row['status'] = 'uncommitted'
            active.append(row)
        elif event == 'checkpoint' or (event == 'session_end' and row.get('safely_saved')):
            committed_step = row['global_step']
            committed_tokens = row.get('tokens_seen', committed_tokens)
            committed_training_seconds = row.get('training_seconds', committed_training_seconds)
        if event in ('session_end', 'error'):
            session.update(status='error' if event == 'error' else ('complete' if row.get('complete') else 'stopped'),
                           end_step=row.get('global_step'), reason=row.get('reason', row.get('message')),
                           seconds=row.get('session_seconds'), ended_at=row.get('timestamp', row.get('time')))
    for row in active:
        row['status'] = 'committed' if row['global_step'] <= committed_step else 'uncommitted'
    train = [row for row in active if row['event'] == 'train' and row['status'] == 'committed']
    full = [row for row in active if row['event'] == 'validation_full' and row['status'] == 'committed']
    quick = [row for row in active if row['event'] == 'validation_quick' and row['status'] == 'committed']
    logged_tokens = sum(row.get('tokens', 0) for row in train)
    return dict(
        committed_step=committed_step, total_steps=total_steps, committed_tokens_seen=committed_tokens,
        committed_training_seconds=committed_training_seconds,
        recorded_session_seconds=sum(s.get('seconds') or 0 for s in sessions.values()),
        sessions=list(sessions.values()), committed_train_records=len(train),
        rolled_back_records=sum(row['status'] == 'rolled_back' for row in rows),
        uncommitted_records=sum(row['status'] == 'uncommitted' for row in rows),
        last_train_loss=train[-1]['loss'] if train else None,
        logged_token_weighted_train_loss=(sum(row['loss'] * row.get('tokens', 0) for row in train) / logged_tokens
                                          if logged_tokens else None),
        best_observed_full_validation_loss=min((row['loss'] for row in full), default=None),
        best_observed_quick_validation_loss=min((row['loss'] for row in quick), default=None),
        peak_allocated_vram_gib=max((row.get('peak_vram_gib', 0) for row in train), default=None),
        warnings=warnings,
        notes=['Committed means the optimizer step is covered by a checkpoint/resume/safe end event; '
               'a validation observation at that step is not proof of a saved best checkpoint.',
               'Loss average uses only recorded train updates, weighted by valid tokens; legacy logs may be sparse.',
               'Session time includes overhead and replay; committed training time excludes validation/checkpoint IO.',
               'Incomplete sessions may indicate a hard kill. Historical checkpoint files may have been pruned.',
               'Quick and full validation are separate. Missing legacy fields remain null.'])


def export_report(run, output=None):
    run = Path(run)
    output = Path(output) if output is not None else run
    output.mkdir(parents=True, exist_ok=True)
    rows, warnings = read_history(run / 'metrics.jsonl')
    summary = summarize(rows, warnings)
    marker = run / 'run.json'
    summary['run_contract'] = json.loads(marker.read_text(encoding='utf-8')) if marker.exists() else None
    summary['run_dir'] = str(run.resolve())
    # Checkpoint hints are provenance only, not used to guess missing metrics.
    summary['checkpoint_hints'] = {}
    for name in ('latest', 'best', 'final'):
        path = run / 'checkpoints' / f'{name}.json'
        if path.exists():
            try:
                summary['checkpoint_hints'][name] = json.loads(path.read_text(encoding='utf-8'))
            except ValueError:
                warnings.append(f'Invalid checkpoint hint: {path.name}')
    fields = ['history_line', 'session_id', 'timestamp', 'time', 'event', 'status', 'global_step']
    fields += sorted({key for row in rows for key in row} - set(fields))
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                         for key, value in row.items()})
    atomic_text(output / 'metrics.csv', stream.getvalue(), encoding='utf-8-sig')
    atomic_text(output / 'summary.json', json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    def shown(value):
        if isinstance(value, float):
            return f'{value:,.4f}'
        if isinstance(value, int):
            return f'{value:,}'
        return 'N/A' if value is None else str(value)
    lines = ['# Training report', '', f'Run: `{run.resolve()}`', '', '| Metric | Value |', '|---|---|']
    for key in ('committed_step', 'total_steps', 'committed_tokens_seen', 'committed_training_seconds',
                'recorded_session_seconds', 'last_train_loss', 'logged_token_weighted_train_loss',
                'best_observed_full_validation_loss', 'best_observed_quick_validation_loss',
                'peak_allocated_vram_gib', 'rolled_back_records', 'uncommitted_records'):
        lines.append(f'| {key} | {shown(summary[key])} |')
    lines += ['', '## Sessions', '', '| Session | Status | Start step | End step | Seconds |', '|---|---|---|---|---|']
    for session in summary['sessions']:
        lines.append('| ' + ' | '.join(shown(session.get(key)) for key in
                                      ('session_id', 'status', 'start_step', 'end_step', 'seconds')) + ' |')
    lines += ['', '## Interpretation', ''] + ['- ' + note for note in summary['notes'] + warnings]
    lines += ['', '## Run configuration and identity', '', '```json',
              json.dumps(summary['run_contract'], ensure_ascii=False, indent=2), '```',
              '', '## Checkpoint hints (not checksum-verified by this report)', '', '```json',
              json.dumps(summary['checkpoint_hints'], ensure_ascii=False, indent=2), '```', '']
    atomic_text(output / 'report.md', '\n'.join(lines))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if not (args.run_dir / 'metrics.jsonl').is_file():
        parser.error('Run directory has no metrics.jsonl')
    export_report(args.run_dir, args.output_dir)
    print(f'Report exported: {(args.output_dir or args.run_dir).resolve()}')


if __name__ == '__main__':
    main()