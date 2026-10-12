"""Readable local training logs. JSONL is the append-only source of truth."""
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time
import traceback
import uuid


def duration(seconds):
    if seconds is None:
        return 'N/A'
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def format_event(row):
    stamp = row.get('timestamp') or datetime.fromtimestamp(row['time'], timezone.utc).isoformat(timespec='seconds')
    event = row['event']
    prefix = f'[{stamp}] {event.upper():<18}'
    step = row.get('global_step', '?')
    if event == 'train':
        total = row.get('total_steps')
        progress = f'{step:,}/{total:,} ({100 * step / total:.2f}%)' if total else str(step)
        return (f'{prefix} Step {progress} | Loss {row["loss"]:.4f} | LR {row["learning_rate"]:.2e}'
                f' | Grad {row["grad_norm"]:.4f}\n'
                f'    Tokens {row["tokens_seen"]:,} | {row["tokens_per_second"]:,.0f} tok/s'
                f' | {row["seconds"]:.2f} s/step | Peak allocated {row["peak_vram_gib"]:.2f} GiB'
                f' | Elapsed {duration(row.get("session_seconds"))}'
                f' | ETA train-only ~{duration(row.get("eta_train_seconds"))}')
    if event.startswith('validation_'):
        perplexity = row.get('perplexity')
        ppl = f'{perplexity:.3f}' if perplexity is not None else 'N/A'
        return (f'{prefix} Step {step} | Loss {row["loss"]:.4f} | PPL {ppl}'
                f' | {row["sequences"]:,} sequences | {row["tokens"]:,} tokens')
    if event == 'session_start':
        return (f'{prefix} Step {step}/{row["total_steps"]} | {row["parameters"]:,} parameters'
                f' | Device {row["device"]} | Resume {row.get("resumed", False)}\n'
                f'    Micro-batch {row.get("micro_batch_size", "?")}'
                f' x accumulation {row.get("gradient_accumulation_steps", "?")}'
                f' | Epochs {row.get("epochs", "?")} | Precision {row.get("precision", "?")}')
    if event == 'session_end':
        return (f'{prefix} Step {step} | {row.get("reason", "stopped")}'
                f' | Safely saved {row["safely_saved"]} | Elapsed {duration(row["session_seconds"])}')
    if event in ('checkpoint', 'resume'):
        return (f'{prefix} Step {step} | {row["path"]}'
                f' | Best {row.get("is_best", False)} | Final {row.get("final", False)}')
    return f'{prefix} {row.get("message", row.get("run_dir", ""))}'


class RunLogger:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.session_id = uuid.uuid4().hex
        self.started = time.monotonic()

    def __enter__(self):
        # Preserve a partial record after a hard kill; the report flags it as malformed.
        path = self.directory / 'metrics.jsonl'
        if path.exists() and path.stat().st_size:
            with path.open('rb+') as stream:
                stream.seek(-1, os.SEEK_END)
                if stream.read(1) != b'\n':
                    stream.seek(0, os.SEEK_END)
                    stream.write(b'\n')
        self.emit('session_open', run_dir=str(self.directory.resolve()))
        return self

    def emit(self, event, console=True, **values):
        now = time.time()
        if event.startswith('validation_') and 'loss' in values:
            try:
                values['perplexity'] = math.exp(values['loss'])
            except OverflowError:
                values['perplexity'] = None
        row = dict(time=now, timestamp=datetime.fromtimestamp(now, timezone.utc).isoformat(timespec='seconds'),
                   session_id=self.session_id, event=event, **values)
        line = json.dumps(row, ensure_ascii=False, allow_nan=False)
        with (self.directory / 'metrics.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(line + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        text = format_event(row)
        with (self.directory / 'train.log').open('a', encoding='utf-8') as stream:
            stream.write(text + '\n')
        if console:
            print(text, flush=True)
        return row

    def __exit__(self, exc_type, exc, tb):
        if exc is not None:
            try:
                self.emit('error', message=f'{exc_type.__name__}: {exc}',
                          traceback=''.join(traceback.format_exception(exc_type, exc, tb)),
                          session_seconds=time.monotonic() - self.started, safely_saved=False)
            except Exception as log_error:
                print(f'WARNING: cannot persist error log: {log_error}', flush=True)
        try:
            from pretrain_report import export_report
            export_report(self.directory)
        except Exception as report_error:
            # Reporting must not turn a saved training session into a failed update,
            # nor hide the original training error (e.g. a full disk).
            print(f'WARNING: cannot export report: {report_error}', flush=True)