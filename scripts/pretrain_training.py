"""Training primitives independent of CLI, with committed data cursors."""
from contextlib import nullcontext
import math
import random

import numpy as np
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.utils.data import DataLoader, Sampler


class EpochBatches(Sampler):
    """Pure epoch permutation. Prefetch cannot mutate the trainer's cursor."""
    def __init__(self, size, batch_size, seed, epoch, cursor=0):
        if not 0 <= cursor <= size or batch_size < 1:
            raise ValueError('Invalid data cursor/batch size')
        self.size, self.batch_size, self.cursor = size, batch_size, cursor
        self.order = torch.randperm(size, generator=torch.Generator().manual_seed(seed + epoch)).tolist()

    def __iter__(self):
        for start in range(self.cursor, self.size, self.batch_size):
            yield self.order[start:start + self.batch_size]

    def __len__(self):
        return math.ceil((self.size - self.cursor) / self.batch_size)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(dataset, training, epoch, cursor, workers, device):
    sampler = EpochBatches(len(dataset), training['micro_batch_size'], training['seed'], epoch, cursor)
    return DataLoader(dataset, batch_sampler=sampler, num_workers=workers,
                      pin_memory=device.type == 'cuda',
                      generator=torch.Generator().manual_seed(training['seed'] + epoch))


class TokenSchedule:
    """Step schedule across the WHOLE run; session limits never alter the horizon."""
    def __init__(self, optimizer, training, total_steps):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup = min(training['warmup_steps'], max(0, total_steps - 1))
        self.peak = training['learning_rate']
        self.minimum = training['min_learning_rate']
        self.completed_steps = 0
        self.apply_next()

    def apply_next(self):
        step = min(self.completed_steps + 1, self.total_steps)
        if self.warmup and step <= self.warmup:
            lr = self.peak * step / self.warmup
        else:
            fraction = (step - self.warmup) / max(1, self.total_steps - self.warmup)
            lr = self.minimum + .5 * (self.peak - self.minimum) * (1 + math.cos(math.pi * fraction))
        for group in self.optimizer.param_groups:
            group['lr'] = lr

    def step(self):
        self.completed_steps += 1
        self.apply_next()

    def state_dict(self):
        return dict(completed_steps=self.completed_steps, total_steps=self.total_steps,
                    warmup=self.warmup, peak=self.peak, minimum=self.minimum)

    def load_state_dict(self, state):
        for key in ('total_steps', 'warmup', 'peak', 'minimum'):
            if state[key] != getattr(self, key):
                raise RuntimeError('Scheduler configuration mismatch')
        self.completed_steps = state['completed_steps']
        self.apply_next()


def make_optimizer(model, training, device):
    groups = [dict(params=[p for p in model.parameters() if p.ndim >= 2], weight_decay=training['weight_decay']),
              dict(params=[p for p in model.parameters() if p.ndim < 2], weight_decay=0.0)]
    return torch.optim.AdamW(groups, lr=training['learning_rate'], betas=tuple(training['betas']),
                             eps=training['epsilon'], foreach=False, fused=device.type == 'cuda')


def autocast_context(device, precision):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16) if precision == 'bf16' else nullcontext()


def attention_context(device):
    # Never silently fall back to quadratic math attention on the laptop GPU.
    return sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]) if device.type == 'cuda' else nullcontext()


def train_update(model, optimizer, scheduler, iterator, training, device, after_microbatch=None):
    """Stop requests are handled by caller ONLY after this entire update commits."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    tokens = samples = 0
    loss_total = torch.zeros((), device=device)
    for micro in range(training['gradient_accumulation_steps']):
        try:
            batch = next(iterator)
        except StopIteration:
            break
        x = batch['input_ids'].to(device, non_blocking=True)
        y = batch['targets'].to(device, non_blocking=True)
        valid = int((y != -100).sum().item())
        samples += x.shape[0]
        if valid:
            with attention_context(device), autocast_context(device, training['precision']):
                loss = model.loss_sum(x, y, training['loss_chunk_size'])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Non-finite loss; resume the previous committed checkpoint')
            # Recomputed attention must use the same allowed backends as forward.
            with attention_context(device):
                loss.backward()
            loss_total += loss.detach()
            tokens += valid
        if after_microbatch is not None:
            after_microbatch(micro)
    if not samples:
        return None
    if not tokens:
        raise RuntimeError('An entire optimizer update has no valid targets')
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(tokens)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training['max_grad_norm'], error_if_nonfinite=True)
    lr = optimizer.param_groups[0]['lr']
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return dict(loss=float(loss_total.item() / tokens), tokens=tokens, samples=samples,
                grad_norm=float(norm.item()), learning_rate=lr)


@torch.no_grad()
def evaluate(model, dataset, training, device, sequence_limit=None, stop=None):
    count = len(dataset) if sequence_limit is None else min(sequence_limit, len(dataset))
    # Same seeded subset at every quick evaluation, independent of global RNG.
    order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(training['seed'] + 1000000))[:count]
    subset = torch.utils.data.Subset(dataset, order.tolist())
    loader = DataLoader(subset, batch_size=training['micro_batch_size'], num_workers=0,
                        generator=torch.Generator().manual_seed(training['seed']))
    was_training = model.training
    model.eval()
    total_loss = total_tokens = 0
    try:
        for batch in loader:
            if stop is not None and stop.requested:
                return None
            x, y = (batch[key].to(device) for key in ('input_ids', 'targets'))
            valid = int((y != -100).sum().item())
            if valid:
                with attention_context(device), autocast_context(device, training['precision']):
                    loss = model.loss_sum(x, y, training['loss_chunk_size'])
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Non-finite validation loss')
                total_loss += loss.item()
                total_tokens += valid
        if not total_tokens:
            raise RuntimeError('Validation contains no valid targets')
        return dict(loss=total_loss / total_tokens, tokens=total_tokens, sequences=count)
    finally:
        model.train(was_training)
