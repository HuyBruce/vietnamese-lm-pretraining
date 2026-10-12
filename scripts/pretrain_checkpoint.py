"""Committed checkpoints: fsync + rename + SHA-256, with bounded retention."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch


class IncompatibleCheckpoint(RuntimeError):
    pass


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    try:
        with temporary.open('w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def run_lock(directory):
    """OS releases lock after process death; an old lock file is harmless."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.training.lock').open('a+b') as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError('Another trainer holds this run directory') from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def capture_rng():
    numpy_state = np.random.get_state()
    return {
        'python': random.getstate(),
        'numpy': [numpy_state[0], torch.tensor(numpy_state[1].astype(np.int64)),
                  numpy_state[2], numpy_state[3], numpy_state[4]],
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state['python'])
    name, keys, position, gaussian, cached = state['numpy']
    np.random.set_state((name, keys.numpy().astype(np.uint32), position, gaussian, cached))
    torch.set_rng_state(state['torch'])
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_bytes(v) for v in value)
    return 0


class CheckpointManager:
    def __init__(self, directory, keep_last=3, min_free_gib=2, event_callback=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if keep_last < 1 or min_free_gib < 0:
            raise ValueError('Invalid checkpoint retention or disk reserve')
        self.keep_last = keep_last
        self.reserve = int(min_free_gib * 1024 ** 3)
        self.event_callback = event_callback

    def notify(self, event, message, **values):
        if self.event_callback is None:
            print(message, flush=True)
        else:
            self.event_callback(event, message=message, **values)

    def candidates(self):
        return sorted(self.directory.glob('step-*.json'), reverse=True)

    def load_latest(self, contract):
        failures = []
        for metadata_path in self.candidates():
            try:
                metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
                path = metadata_path.with_suffix('.pt')
                if metadata['file'] != path.name or file_sha(path) != metadata['sha256']:
                    raise RuntimeError('Checksum or filename mismatch')
                state = torch.load(path, map_location='cpu', weights_only=True)
                required = {'format_version', 'contract', 'model', 'optimizer', 'scheduler', 'progress', 'rng'}
                if not isinstance(state, dict) or not required.issubset(state) or state['format_version'] != 1:
                    raise RuntimeError('Invalid checkpoint schema')
                if state['progress']['global_step'] != metadata['step']:
                    raise RuntimeError('Checkpoint step mismatch')
            except Exception as exc:
                failures.append(f'{metadata_path.name}: {exc}')
                self.notify('warning', f'Skipping invalid checkpoint: {failures[-1]}')
                continue
            if state['contract'] != contract:
                raise IncompatibleCheckpoint('Checkpoint config/data/code/runtime differs; use the original settings or a new run directory')
            self.notify('resume', f'Resuming checkpoint: {path.resolve()}',
                        path=str(path.resolve()), global_step=metadata['step'])
            return state
        if failures or list(self.directory.glob('*.pt')):
            raise RuntimeError('No valid committed checkpoint; refusing to restart from scratch')
        return None

    def save(self, state, is_best=False, final=False):
        step = state['progress']['global_step']
        name = f'step-{step:09d}-{time.time_ns()}'
        path = self.directory / (name + '.pt')
        temporary = path.with_name(path.name + '.tmp')
        needed = int(tensor_bytes(state) * 1.05) + 64 * 1024 ** 2 + self.reserve
        if shutil.disk_usage(self.directory).free < needed:
            raise OSError(f'Insufficient disk space to safely save checkpoint (need {needed / 1024**3:.2f} GiB free); previous checkpoints retained')
        try:
            with temporary.open('wb') as stream:
                torch.save(state, stream)
                stream.flush()
                os.fsync(stream.fileno())
            checksum = file_sha(temporary)
            os.replace(temporary, path)
            metadata = dict(file=path.name, sha256=checksum, step=step,
                            is_best=is_best, final=final)
            atomic_json(path.with_suffix('.json'), metadata)
        finally:
            temporary.unlink(missing_ok=True)
        # Metadata is the commit record. latest.json is just a convenience hint.
        atomic_json(self.directory / 'latest.json', metadata)
        if is_best:
            atomic_json(self.directory / 'best.json', metadata)
        if final:
            atomic_json(self.directory / 'final.json', metadata)
        self.prune()
        self.notify('checkpoint', f'Checkpoint saved: {path.resolve()}',
                    path=str(path.resolve()), global_step=step, sha256=checksum,
                    is_best=is_best, final=final,
                    tokens_seen=state['progress'].get('tokens_seen'),
                    training_seconds=state['progress'].get('training_seconds'))
        return path

    def prune(self):
        committed = self.candidates()
        protected = {p.stem for p in committed[:self.keep_last]}
        for hint in ('best.json', 'final.json'):
            path = self.directory / hint
            if path.exists():
                try:
                    protected.add(Path(json.loads(path.read_text(encoding='utf-8'))['file']).stem)
                except (ValueError, KeyError):
                    return
        for metadata in committed:
            if metadata.stem not in protected:
                metadata.with_suffix('.pt').unlink(missing_ok=True)
                metadata.unlink(missing_ok=True)
