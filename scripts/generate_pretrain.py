"""Generate a continuation from a committed pretraining checkpoint (read-only)."""
import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import re
import time
import unicodedata

import sentencepiece as spm
import torch
from torch.nn import functional as F

from pretrain_checkpoint import atomic_json, file_sha
from pretrain_model import CausalLM
from pretrain_training import attention_context

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_checkpoint(run_dir, selection='final'):
    """Read commit records, verify SHA-256, then load on CPU without restoring RNG/optimizer."""
    if selection not in ('final', 'best', 'latest'):
        raise ValueError('Checkpoint must be final, best or latest')
    directory = Path(run_dir).resolve() / 'checkpoints'
    if selection == 'latest':
        candidates = sorted(directory.glob('step-*.json'), reverse=True)
        if not candidates:
            raise RuntimeError('No committed checkpoint found in this run')
        metadata = read_json(candidates[0])
    else:
        hint = directory / f'{selection}.json'
        if not hint.is_file():
            raise RuntimeError(f'No {selection} checkpoint. For an unfinished run, explicitly use --checkpoint latest')
        metadata = read_json(hint)
    name = metadata.get('file', '')
    if not isinstance(name, str) or not re.fullmatch(r'step-\d{9,}-\d+\.pt', name):
        raise RuntimeError('Invalid checkpoint filename in commit record')
    path = directory / name
    committed = path.with_suffix('.json')
    if not committed.is_file() or read_json(committed) != metadata:
        raise RuntimeError('Checkpoint hint does not match its committed metadata')
    if selection == 'final' and metadata.get('final') is not True:
        raise RuntimeError('Selected checkpoint is not marked final')
    if selection == 'best' and metadata.get('is_best') is not True:
        raise RuntimeError('Selected checkpoint is not marked best')
    if not path.is_file() or file_sha(path) != metadata.get('sha256'):
        raise RuntimeError('Checkpoint SHA-256 mismatch or missing weights')
    # mmap keeps unused optimizer tensors out of an extra eager RAM allocation.
    state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    required = {'format_version', 'contract', 'model', 'progress'}
    if not isinstance(state, dict) or not required.issubset(state) or state['format_version'] != 1:
        raise RuntimeError('Invalid checkpoint schema')
    if state['progress'].get('global_step') != metadata.get('step'):
        raise RuntimeError('Checkpoint step mismatch')
    if selection == 'final' and state['progress'].get('final_saved') is not True:
        raise RuntimeError('Final checkpoint has no completed training marker')
    expected_source = state['contract']['source_sha256']['pretrain_model.py']
    if file_sha(Path(__file__).with_name('pretrain_model.py')) != expected_source:
        raise RuntimeError('Model source differs from checkpoint; use the original pretrain_model.py')
    return state, path, metadata


def load_model(run_dir, selection, tokenizer_path, device):
    state, path, metadata = load_checkpoint(run_dir, selection)
    expected = state['contract']['data']['tokenizer_sha256']
    tokenizer_path = Path(tokenizer_path).resolve()
    if not tokenizer_path.is_file() or file_sha(tokenizer_path) != expected:
        raise RuntimeError('Tokenizer SHA-256 mismatch or missing tokenizer.model')
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    config = state['contract']['config']['model']
    if tokenizer.get_piece_size() != config['vocab_size']:
        raise RuntimeError('Tokenizer vocabulary differs from model')
    ids = (tokenizer.eos_id(), tokenizer.pad_id(), tokenizer.unk_id(), tokenizer.bos_id())
    if any(not 0 <= value < config['vocab_size'] for value in ids) or len(set(ids)) != 4:
        raise RuntimeError('Invalid tokenizer special token IDs')
    model = CausalLM(config)
    model.load_state_dict(state['model'], strict=True)
    info = dict(path=str(path), sha256=metadata['sha256'], selection=selection,
                global_step=metadata['step'], tokens_seen=state['progress'].get('tokens_seen'),
                final=metadata.get('final', False), tokenizer_sha256=expected)
    del state  # Only model weights are needed; never restore saved optimizer or training RNG.
    model = model.to(device).eval()
    return model, tokenizer, info


def validate_sampling(max_new_tokens, temperature, top_k, top_p):
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError('max_new_tokens must be a positive integer')
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError('temperature must be finite and nonnegative (0 = greedy)')
    if type(top_k) is not int or top_k < 0:
        raise ValueError('top_k must be a nonnegative integer (0 = disabled)')
    if not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError('top_p must be in (0, 1]')


def sample_token(logits, temperature, top_k, top_p, generator):
    """Sample after forbidden tokens have been masked. Keep the first nucleus-crossing token."""
    scores = logits.float().clone()
    if bool(torch.isnan(scores).any()) or bool(torch.isposinf(scores).any()) or not bool(torch.isfinite(scores).any()):
        raise RuntimeError('No valid finite next-token scores')
    if temperature == 0:
        return int(scores.argmax().item())
    scores = (scores - scores.max()) / temperature
    if top_k:
        threshold = torch.topk(scores, min(top_k, scores.numel())).values[-1]
        scores[scores < threshold] = -torch.inf
    if top_p < 1:
        sorted_scores, indices = torch.sort(scores, descending=True)
        remove = torch.softmax(sorted_scores, dim=-1).cumsum(dim=-1) > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        scores[indices[remove]] = -torch.inf
    probabilities = torch.softmax(scores, dim=-1)
    return int(torch.multinomial(probabilities, 1, generator=generator).item())


@torch.inference_mode()
def generate(model, tokenizer, prompt, max_new_tokens=200, temperature=0.8,
             top_k=50, top_p=0.9, seed=20260929, precision='fp32'):
    validate_sampling(max_new_tokens, temperature, top_k, top_p)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError('Prompt must contain non-whitespace text')
    normalized = unicodedata.normalize('NFC', prompt)
    # Corpus tokenization does not prepend BOS; preserve this convention at inference.
    prompt_ids = tokenizer.encode(normalized, out_type=int)
    context = model.config['context_length']
    if not prompt_ids or len(prompt_ids) >= context:
        raise ValueError(f'Prompt must use 1..{context - 1} tokens to leave space for generation; got {len(prompt_ids)}')
    device = next(model.parameters()).device
    if precision not in ('fp32', 'bf16'):
        raise ValueError('Inference precision must be fp32 or bf16')
    if precision == 'bf16' and (device.type != 'cuda' or not torch.cuda.is_bf16_supported()):
        raise ValueError('BF16 inference requires a supported CUDA GPU')
    generator = torch.Generator(device=device).manual_seed(seed)
    current = list(prompt_ids)
    generated = []
    stop_reason = 'max_new_tokens'
    was_training = model.training
    model.eval()
    started = time.perf_counter()
    try:
        for _ in range(max_new_tokens):
            if len(current) >= context:
                stop_reason = 'context_limit'
                break
            inputs = torch.tensor([current], dtype=torch.long, device=device)
            autocast = torch.autocast('cuda', dtype=torch.bfloat16) if precision == 'bf16' else nullcontext()
            with attention_context(device), autocast:
                # Project only the last position, avoiding [context, vocabulary] logits.
                hidden = model.hidden(inputs)[:, -1, :]
                logits = F.linear(hidden, model.embedding.weight)[0].float()
            for token_id in (tokenizer.pad_id(), tokenizer.bos_id(), tokenizer.unk_id()):
                logits[token_id] = -torch.inf
            token = sample_token(logits, temperature, top_k, top_p, generator)
            current.append(token)
            generated.append(token)
            if token == tokenizer.eos_id():
                stop_reason = 'eos'
                break
        if stop_reason == 'max_new_tokens' and len(current) >= context:
            stop_reason = 'context_limit'
    finally:
        model.train(was_training)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    decoded_prompt = tokenizer.decode(prompt_ids)
    decoded_text = tokenizer.decode(current)
    # Joint decode preserves whitespace at the prompt/continuation boundary.
    prefix_stable = decoded_text.startswith(decoded_prompt)
    continuation = decoded_text[len(decoded_prompt):] if prefix_stable else tokenizer.decode(generated)
    return dict(prompt=prompt, normalized_prompt=normalized, continuation=continuation,
                text=normalized + continuation, decoded_text=decoded_text,
                decoded_prefix_stable=prefix_stable, prompt_tokens=len(prompt_ids),
                generated_tokens=len(generated), generated_token_ids=generated,
                stop_reason=stop_reason, seconds=time.perf_counter() - started,
                device=str(device), precision=precision,
                sampling=dict(max_new_tokens=max_new_tokens, temperature=temperature,
                              top_k=top_k, top_p=top_p, seed=seed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'runs/pretrain_110m')
    parser.add_argument('--checkpoint', choices=('final', 'best', 'latest'), default='final')
    parser.add_argument('--tokenizer', type=Path, default=ROOT / 'DataProcessed/pretrain_ready/tokenizer/tokenizer.model')
    prompts = parser.add_mutually_exclusive_group(required=True)
    prompts.add_argument('--prompt')
    prompts.add_argument('--prompt-file', type=Path, help='UTF-8 text file; multiline prompts are preserved')
    parser.add_argument('--max-new-tokens', type=int, default=200)
    parser.add_argument('--temperature', type=float, default=0.8, help='0 = greedy; positive = sampling')
    parser.add_argument('--top-k', type=int, default=50, help='0 = disabled')
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=20260929)
    parser.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='auto')
    parser.add_argument('--precision', choices=('auto', 'bf16', 'fp32'), default='auto')
    parser.add_argument('--output', type=Path, help='New JSON file, never overwrite an existing file')
    args = parser.parse_args()
    try:
        validate_sampling(args.max_new_tokens, args.temperature, args.top_k, args.top_p)
        prompt = args.prompt if args.prompt is not None else args.prompt_file.read_text(encoding='utf-8-sig')
        if not prompt.strip():
            raise ValueError('Prompt must contain non-whitespace text')
        if args.output:
            args.output = args.output.resolve()
            if args.output.exists() or args.output.suffix.lower() != '.json':
                raise ValueError('--output must be a new .json file; existing files are never overwritten')
            if args.output.is_relative_to(args.run_dir.resolve() / 'checkpoints'):
                raise ValueError('Output must not be inside the checkpoint directory')
        device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable; use --device cpu or install a CUDA-enabled PyTorch')
        precision = args.precision
        if precision == 'auto':
            precision = 'bf16' if device.type == 'cuda' and torch.cuda.is_bf16_supported() else 'fp32'
        model, tokenizer, checkpoint = load_model(args.run_dir, args.checkpoint, args.tokenizer, device)
        result = generate(model, tokenizer, prompt, args.max_new_tokens, args.temperature,
                          args.top_k, args.top_p, args.seed, precision)
        result['checkpoint'] = checkpoint
        result['runtime'] = dict(torch=str(torch.__version__), cuda=torch.version.cuda)
        print(f'Checkpoint: {checkpoint["path"]}\nStep: {checkpoint["global_step"]} | Selection: {args.checkpoint}')
        if not checkpoint['final']:
            print('NOTE: This is not a final checkpoint; output does not demonstrate completed pretraining quality.')
        print('\n=== PROMPT ===\n' + prompt)
        print('\n=== CONTINUATION ===\n' + result['continuation'])
        print('\n=== FULL TEXT ===\n' + result['text'])
        print(f'\nGenerated {result["generated_tokens"]} tokens | Stop: {result["stop_reason"]} | {result["seconds"]:.2f}s')
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(args.output, result)
            print(f'Result: {args.output}')
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        parser.exit(1, f'Generation error: {exc}\n')


if __name__ == '__main__':
    main()