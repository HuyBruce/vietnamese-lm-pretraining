"""Single-GPU pretraining with automatic, exact-cursor checkpoint resume."""
import argparse
import json
import math
from pathlib import Path
import platform
import signal
import time

import torch

from pretrain_checkpoint import (CheckpointManager, atomic_json, capture_rng, file_sha,
                                 restore_rng, run_lock)
from pretrain_dataset import CausalTokenDataset
from pretrain_compat import select_contract
from pretrain_logging import RunLogger
from pretrain_model import CausalLM
from pretrain_training import (TokenSchedule, evaluate, make_loader, make_optimizer,
                               seed_all, train_update)

ROOT = Path(__file__).resolve().parents[1]


class StopRequest:
    def __init__(self, stop_file):
        self.stop_file = Path(stop_file)
        self.signaled = False
        self.handlers = {}

    @property
    def requested(self):
        return self.signaled or self.stop_file.exists()

    def handle(self, signum, frame):
        self.signaled = True
        print('\nStop requested: finish current optimizer step, save checkpoint, then exit. Please wait.', flush=True)

    def __enter__(self):
        for name in ('SIGINT', 'SIGTERM', 'SIGBREAK'):
            sig = getattr(signal, name, None)
            if sig is not None:
                self.handlers[sig] = signal.signal(sig, self.handle)
        return self

    def __exit__(self, *args):
        for sig, handler in self.handlers.items():
            signal.signal(sig, handler)


def validate_config(config):
    m, t = config['model'], config['training']
    for name in ('vocab_size', 'num_layers', 'hidden_size', 'num_heads', 'intermediate_size', 'context_length'):
        if type(m[name]) is not int or m[name] < 1:
            raise ValueError(f'Invalid model {name}')
    if m['rms_norm_eps'] <= 0 or m['rope_theta'] <= 0:
        raise ValueError('Invalid normalization/RoPE settings')
    for name in ('epochs', 'micro_batch_size', 'gradient_accumulation_steps', 'loss_chunk_size'):
        if type(t[name]) is not int or t[name] < 1:
            raise ValueError(f'Invalid training {name}')
    if t['precision'] not in ('bf16', 'fp32') or t['warmup_steps'] < 0:
        raise ValueError('Only bf16/fp32 and nonnegative warmup are supported')
    if not 0 < t['min_learning_rate'] <= t['learning_rate']:
        raise ValueError('Invalid learning rate range')
    if len(t['betas']) != 2 or any(not 0 <= b < 1 for b in t['betas']):
        raise ValueError('Invalid AdamW betas')
    if t['epsilon'] <= 0 or t['max_grad_norm'] <= 0 or t['weight_decay'] < 0:
        raise ValueError('Invalid optimizer settings')
    for group, names in [('checkpoint', ('every_steps', 'every_minutes', 'keep_last')),
                         ('evaluation', ('quick_every_steps', 'quick_sequences', 'full_every_steps')),
                         ('logging', ('every_steps',))]:
        if any(config[group][name] <= 0 for name in names):
            raise ValueError(f'Invalid {group} settings')


def data_identity(root, config):
    path = root / 'binary/manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if not manifest.get('complete'):
        raise RuntimeError('Dataset is not complete')
    tokenizer = manifest['tokenizer']
    if config['model']['context_length'] != manifest['context_length']:
        raise ValueError('Model context must match the existing dataset; do not edit its manifest')
    if config['model']['vocab_size'] != tokenizer['actual_vocab']:
        raise ValueError('Model vocabulary does not match tokenizer')
    actual = file_sha(root / tokenizer['model'])
    if actual != tokenizer['sha256']:
        raise RuntimeError('Tokenizer checksum mismatch')
    for shard in manifest['shards']:
        import numpy as np
        expected = (shard['sequences'] * manifest['context_length'] + 1) * np.dtype(shard['dtype']).itemsize
        if (root / shard['file']).stat().st_size != expected:
            raise RuntimeError(f'Binary length mismatch: {shard["file"]}')
    return dict(manifest_sha256=file_sha(path), tokenizer_sha256=actual)


def make_contract(config, identity, total_steps, device):
    folder = Path(__file__).resolve().parent
    sources = ['pretrain_model.py', 'pretrain_training.py', 'pretrain_checkpoint.py',
               'pretrain_dataset.py', 'train_pretrain.py']
    return dict(config=config, data=identity, total_steps=total_steps,
                source_sha256={name: file_sha(folder / name) for name in sources},
                runtime=dict(torch=str(torch.__version__), cuda=torch.version.cuda,
                             python=platform.python_version(), device=device.type))


def checkpoint_state(model, optimizer, scheduler, progress, contract):
    return dict(format_version=1, contract=contract, model=model.state_dict(),
                optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                progress=dict(progress), rng=capture_rng())


def train_session(args, config, device, logger):
    run, root = args.run_dir.resolve(), args.data.resolve()
    training = config['training']
    identity = data_identity(root, config)
    train = CausalTokenDataset(root, 'train')
    validation = CausalTokenDataset(root, 'validation')
    if not len(train) or not len(validation):
        raise RuntimeError('Both train and validation must be nonempty')
    updates_per_epoch = math.ceil(math.ceil(len(train) / training['micro_batch_size']) /
                                  training['gradient_accumulation_steps'])
    total_steps = updates_per_epoch * training['epochs']
    contract = make_contract(config, identity, total_steps, device)
    actual_sources = dict(contract['source_sha256'])
    for name in ('pretrain_logging.py', 'pretrain_report.py', 'pretrain_compat.py', 'pretrain_logging_upgrade.json'):
        actual_sources[name] = file_sha(Path(__file__).with_name(name))
    marker = run / 'run.json'
    if marker.exists():
        contract = select_contract(contract, json.loads(marker.read_text(encoding='utf-8')))
    manager = CheckpointManager(run / 'checkpoints', config['checkpoint']['keep_last'],
                                config['checkpoint']['min_free_gib'], event_callback=logger.emit)
    state = manager.load_latest(contract)
    resumed = state is not None
    if marker.exists():
        if json.loads(marker.read_text(encoding='utf-8')) != contract:
            raise RuntimeError('Run directory belongs to a different config/data/code/runtime')
        if state is None:
            raise RuntimeError('Existing run has no valid checkpoint; use a new run directory')
    seed_all(training['seed'])
    model = CausalLM(config['model']).to(device)
    optimizer = make_optimizer(model, training, device)
    scheduler = TokenSchedule(optimizer, training, total_steps)
    progress = dict(epoch=0, cursor=0, global_step=0, tokens_seen=0, best_validation_loss=None,
                    last_full_eval_step=-1, training_seconds=0.0, final_saved=False)
    if state is not None:
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        progress = state['progress']
        if scheduler.completed_steps != progress['global_step']:
            raise RuntimeError('Scheduler/progress step mismatch')
        if not (0 <= progress['epoch'] <= training['epochs'] and 0 <= progress['cursor'] < len(train)):
            raise RuntimeError('Invalid saved data cursor')
        effective_samples = training['micro_batch_size'] * training['gradient_accumulation_steps']
        expected_step = progress['epoch'] * updates_per_epoch + math.ceil(progress['cursor'] / effective_samples)
        if expected_step != progress['global_step']:
            raise RuntimeError('Data cursor/global step mismatch')
        restore_rng(state['rng'])
        del state
    else:
        # Initial checkpoint makes interruption before the first update recoverable.
        manager.save(checkpoint_state(model, optimizer, scheduler, progress, contract))
        atomic_json(marker, contract)
    logger.emit('session_start', global_step=progress['global_step'], epoch=progress['epoch'],
              cursor=progress['cursor'], parameters=sum(p.numel() for p in model.parameters()),
              total_steps=total_steps, device=str(device), resumed=resumed,
              tokens_seen=progress['tokens_seen'], training_seconds=progress['training_seconds'],
              actual_source_sha256=actual_sources, micro_batch_size=training['micro_batch_size'],
              gradient_accumulation_steps=training['gradient_accumulation_steps'],
              precision=training['precision'], epochs=training['epochs'],
              device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor(),
              session_steps_limit=args.session_steps, session_minutes_limit=args.session_minutes)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    started = last_save = time.monotonic()
    session_steps = 0
    session_train_seconds = 0.0
    last_saved_step = progress['global_step']

    def save(best=False, final=False):
        nonlocal last_save, last_saved_step
        manager.save(checkpoint_state(model, optimizer, scheduler, progress, contract),
                     is_best=best, final=final)
        last_save, last_saved_step = time.monotonic(), progress['global_step']

    def session_limit():
        return ((args.session_steps is not None and session_steps >= args.session_steps) or
                (args.session_minutes is not None and time.monotonic() - started >= args.session_minutes * 60))

    with StopRequest(run / 'STOP') as stop:
        while progress['epoch'] < training['epochs'] and not stop.requested and not session_limit():
            loader = make_loader(train, training, progress['epoch'], progress['cursor'], args.workers, device)
            iterator = iter(loader)
            while progress['cursor'] < len(train) and not stop.requested and not session_limit():
                before = time.monotonic()
                result = train_update(model, optimizer, scheduler, iterator, training, device)
                if result is None:
                    raise RuntimeError('Data loader ended before the committed cursor')
                elapsed = time.monotonic() - before
                progress['cursor'] += result['samples']
                progress['tokens_seen'] += result['tokens']
                progress['global_step'] += 1
                progress['training_seconds'] += elapsed
                session_steps += 1
                session_train_seconds += elapsed
                epoch_done = progress['cursor'] == len(train)
                if epoch_done:
                    progress['epoch'] += 1
                    progress['cursor'] = 0
                step = progress['global_step']
                logger.emit('train',
                              console=step % config['logging']['every_steps'] == 0 or session_steps == 1 or epoch_done,
                              global_step=step, epoch=progress['epoch'], cursor=progress['cursor'],
                              tokens_seen=progress['tokens_seen'], seconds=elapsed,
                              total_steps=total_steps, session_seconds=time.monotonic() - started,
                              training_seconds=progress['training_seconds'],
                              eta_train_seconds=(total_steps - step) * session_train_seconds / session_steps,
                              tokens_per_second=result['tokens'] / elapsed,
                              peak_vram_gib=torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == 'cuda' else 0,
                              **result)
                due = (step % config['checkpoint']['every_steps'] == 0 or
                       time.monotonic() - last_save >= config['checkpoint']['every_minutes'] * 60)
                if due or epoch_done or stop.requested or session_limit():
                    save()
                if not stop.requested and not session_limit():
                    full = epoch_done or step % config['evaluation']['full_every_steps'] == 0
                    if full or step % config['evaluation']['quick_every_steps'] == 0:
                        outcome = evaluate(model, validation, training, device,
                                           None if full else config['evaluation']['quick_sequences'], stop)
                        if outcome is not None:
                            logger.emit('validation_full' if full else 'validation_quick', global_step=step, **outcome)
                            if full:
                                progress['last_full_eval_step'] = step
                                best = progress['best_validation_loss'] is None or outcome['loss'] < progress['best_validation_loss']
                                if best:
                                    progress['best_validation_loss'] = outcome['loss']
                                save(best=best)
                if epoch_done:
                    break
            del iterator, loader
        completed = progress['epoch'] == training['epochs']
        if completed and not progress['final_saved'] and not stop.requested and not session_limit():
            best = False
            if progress['last_full_eval_step'] != progress['global_step']:
                outcome = evaluate(model, validation, training, device, stop=stop)
                if outcome is not None:
                    logger.emit('validation_full', global_step=progress['global_step'], **outcome)
                    progress['last_full_eval_step'] = progress['global_step']
                    best = progress['best_validation_loss'] is None or outcome['loss'] < progress['best_validation_loss']
                    if best:
                        progress['best_validation_loss'] = outcome['loss']
            if not stop.requested:
                progress['final_saved'] = True
                save(best=best, final=True)
        if last_saved_step != progress['global_step'] or stop.requested:
            save()
        stop_requested = stop.requested
        if stop_requested:
            logger.emit('stop', message='Stop requested; current optimizer update and checkpoint saved.')
            stop.stop_file.unlink(missing_ok=True)
        logger.emit('session_end', global_step=progress['global_step'], tokens_seen=progress['tokens_seen'],
                  complete=progress['final_saved'], safely_saved=True,
                  training_seconds=progress['training_seconds'],
                  reason='completed' if progress['final_saved'] else ('stop_requested' if stop_requested else 'session_limit'),
                  session_seconds=time.monotonic() - started)
    for dataset in (train, validation):
        for mapping in dataset._maps.values():
            mapping._mmap.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/pretrain_110m.json')
    parser.add_argument('--data', type=Path, default=ROOT / 'DataProcessed/pretrain_ready')
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'runs/pretrain_110m')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--session-steps', type=int, help='Stop safely after this many additional updates')
    parser.add_argument('--session-minutes', type=float, help='Stop safely after this session duration')
    args = parser.parse_args()
    if args.workers < 0 or any(value is not None and value <= 0 for value in (args.session_steps, args.session_minutes)):
        parser.error('Workers must be nonnegative; session limits must be positive')
    config = json.loads(args.config.read_text(encoding='utf-8'))
    validate_config(config)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable')
    if config['training']['precision'] == 'bf16' and (device.type != 'cuda' or not torch.cuda.is_bf16_supported()):
        parser.error('BF16 configuration requires a supported CUDA GPU; use an explicit fp32 test config for CPU')
    with run_lock(args.run_dir):
        with RunLogger(args.run_dir) as logger:
            train_session(args, config, device, logger)


if __name__ == '__main__':
    main()
