"""Short full-size GPU training benchmark. Does not create a training run/checkpoint."""
import argparse
import json
import math
from pathlib import Path
import time

import torch

from pretrain_checkpoint import atomic_json
from pretrain_dataset import CausalTokenDataset
from pretrain_model import CausalLM
from pretrain_training import TokenSchedule, make_loader, make_optimizer, seed_all, train_update
from train_pretrain import ROOT, data_identity, validate_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/pretrain_110m.json')
    parser.add_argument('--data', type=Path, default=ROOT / 'DataProcessed/pretrain_ready')
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--warmup-steps', type=int, default=1)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/benchmark_110m.json')
    args = parser.parse_args()
    if args.steps < 1 or args.warmup_steps < 0 or args.workers < 0:
        parser.error('Invalid benchmark steps/workers')
    config = json.loads(args.config.read_text(encoding='utf-8'))
    validate_config(config)
    if not torch.cuda.is_available():
        parser.error('CUDA is required')
    if config['training']['precision'] == 'bf16' and not torch.cuda.is_bf16_supported():
        parser.error('BF16 is unsupported')
    identity = data_identity(args.data, config)
    device = torch.device('cuda')
    training = config['training']
    seed_all(training['seed'])
    dataset = CausalTokenDataset(args.data)
    total_steps = math.ceil(math.ceil(len(dataset) / training['micro_batch_size']) /
                            training['gradient_accumulation_steps']) * training['epochs']
    model = CausalLM(config['model']).to(device)
    optimizer = make_optimizer(model, training, device)
    scheduler = TokenSchedule(optimizer, training, total_steps)
    iterator = iter(make_loader(dataset, training, 0, 0, args.workers, device))
    torch.cuda.reset_peak_memory_stats()
    records = []
    for index in range(args.warmup_steps + args.steps):
        torch.cuda.synchronize()
        start = time.monotonic()
        result = train_update(model, optimizer, scheduler, iterator, training, device)
        torch.cuda.synchronize()
        if result is None:
            raise RuntimeError('Benchmark exhausted dataset')
        record = dict(index=index + 1, warmup=index < args.warmup_steps,
                      seconds=time.monotonic() - start, **result)
        records.append(record)
        print(json.dumps(record), flush=True)
    measured = [r for r in records if not r['warmup']]
    rate = sum(r['tokens'] for r in measured) / sum(r['seconds'] for r in measured)
    result = dict(config=config, data=identity, gpu=torch.cuda.get_device_name(), torch=str(torch.__version__),
                  parameters=sum(p.numel() for p in model.parameters()), records=records,
                  measured_tokens_per_second=rate,
                  peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
                  peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3,
                  note='Short training-only benchmark; excludes evaluation/checkpoints and is not a thermal endurance test.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('config', 'data', 'records')}, indent=2), flush=True)
    print(f'Report: {args.output.resolve()}', flush=True)
    del iterator
    for mapping in dataset._maps.values():
        mapping._mmap.close()


if __name__ == '__main__':
    main()
