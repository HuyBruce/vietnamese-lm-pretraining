import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import sentencepiece as spm
import torch
from torch import nn

from generate_pretrain import (generate, load_checkpoint, load_model, sample_token,
                               validate_sampling)
from pretrain_checkpoint import CheckpointManager, atomic_json, file_sha
from pretrain_model import CausalLM


class StubTokenizer:
    def encode(self, text, out_type=int):
        return [4] * len(text)

    def decode(self, ids):
        return ''.join('a' if token == 4 else 'b' if token == 5 else '' for token in ids)

    def eos_id(self): return 2
    def pad_id(self): return 3
    def unk_id(self): return 0
    def bos_id(self): return 1


class StubModel(nn.Module):
    def __init__(self, token=5, context=8):
        super().__init__()
        self.config = {'context_length': context}
        self.embedding = nn.Embedding(8, 8)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.eye(8))
        self.token = token
        self.lengths = []

    def hidden(self, inputs):
        self.lengths.append(inputs.shape[1])
        result = torch.zeros(1, inputs.shape[1], 8)
        result[..., self.token] = 10
        return result


class GenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_greedy_prompt_preserved_and_model_mode_restored(self):
        model = StubModel()
        result = generate(model, StubTokenizer(), 'aa', max_new_tokens=3, temperature=0)
        self.assertEqual(result['text'], 'aabbb')
        self.assertEqual(result['continuation'], 'bbb')
        self.assertEqual(result['generated_token_ids'], [5, 5, 5])
        self.assertEqual(result['stop_reason'], 'max_new_tokens')
        self.assertEqual(model.lengths, [2, 3, 4])
        self.assertTrue(model.training)

    def test_eos_stop(self):
        model = StubModel(token=2).eval()
        result = generate(model, StubTokenizer(), 'a', max_new_tokens=5, temperature=0)
        self.assertEqual(result['generated_tokens'], 1)
        self.assertEqual(result['stop_reason'], 'eos')
        self.assertEqual(result['continuation'], '')
        self.assertFalse(model.training)

    def test_context_limit_no_silent_truncation(self):
        result = generate(StubModel(context=4), StubTokenizer(), 'aa', max_new_tokens=5, temperature=0)
        self.assertEqual(result['generated_tokens'], 2)
        self.assertEqual(result['stop_reason'], 'context_limit')
        with self.assertRaisesRegex(ValueError, 'leave space'):
            generate(StubModel(context=4), StubTokenizer(), 'aaaa')

    def test_pad_bos_and_unk_not_generated(self):
        for token in (0, 1, 3):
            result = generate(StubModel(token=token), StubTokenizer(), 'a', max_new_tokens=2, temperature=0)
            self.assertFalse(set(result['generated_token_ids']) & {0, 1, 3})

    def test_sampling_seed_does_not_change_global_rng(self):
        logits = torch.zeros(8)
        original_rng = torch.get_rng_state().clone()
        sequences = []
        for _ in range(2):
            rng = torch.Generator().manual_seed(123)
            sequences.append([sample_token(logits, 0.8, 0, 1, rng) for _ in range(20)])
        self.assertEqual(sequences[0], sequences[1])
        torch.testing.assert_close(torch.get_rng_state(), original_rng)

    def test_top_k_and_nucleus_keep_highest_token(self):
        logits = torch.tensor([0., 1., 2., 3.])
        for top_k, top_p in ((1, 1), (0, 0.01), (100, 0.01)):
            rng = torch.Generator().manual_seed(1)
            self.assertEqual(sample_token(logits, 1., top_k, top_p, rng), 3)
        # P(first)=.6, P(second)=.3: p=.7 must retain the crossing second token.
        logits = torch.tensor([.6, .3, .1]).log()
        rng = torch.Generator().manual_seed(2)
        results = {sample_token(logits, 1., 0, .7, rng) for _ in range(100)}
        self.assertEqual(results, {0, 1})

    def test_invalid_inputs_and_nonfinite_logits(self):
        for values in ((0, .8, 50, .9), (1, -1., 0, 1), (1, float('nan'), 0, 1),
                       (1, .8, -1, 1), (1, .8, 0, 0), (1, .8, 0, 1.1)):
            with self.assertRaises(ValueError): validate_sampling(*values)
        for prompt in ('', ' \n\t'):
            with self.assertRaisesRegex(ValueError, 'Prompt'):
                generate(StubModel(), StubTokenizer(), prompt)
        for logits in (torch.full((8,), -torch.inf), torch.full((8,), torch.nan),
                       torch.full((8,), torch.inf)):
            with self.assertRaisesRegex(RuntimeError, 'finite'):
                sample_token(logits, .8, 0, 1, torch.Generator())
        with self.assertRaisesRegex(ValueError, 'BF16'):
            generate(StubModel(), StubTokenizer(), 'a', precision='bf16')

    def test_nfc_normalization_preserves_original_prompt_field(self):
        result = generate(StubModel(context=16), StubTokenizer(), 'a\u0301', max_new_tokens=1, temperature=0)
        self.assertEqual(result['prompt'], 'a\u0301')
        self.assertEqual(result['normalized_prompt'], 'á')


class CheckpointGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='generate-pretrain-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.tokenizer = self.root / 'tokenizer.model'
        spm.SentencePieceTrainer.train(
            sentence_iterator=iter(['Tiếng Việt có dấu. Sáng nay tại Hà Nội, công nghệ phát triển.'] * 30),
            model_prefix=str(self.root / 'tokenizer'), vocab_size=320, model_type='bpe',
            byte_fallback=True, normalization_rule_name='identity', remove_extra_whitespaces=False,
            unk_id=0, bos_id=1, eos_id=2, pad_id=3, hard_vocab_limit=False, minloglevel=2)
        tokenizer = spm.SentencePieceProcessor(model_file=str(self.tokenizer))
        config = dict(vocab_size=tokenizer.get_piece_size(), num_layers=1, hidden_size=16,
                      num_heads=2, intermediate_size=32, context_length=64,
                      rms_norm_eps=1e-5, rope_theta=10000., gradient_checkpointing=True)
        self.model = CausalLM(config)
        self.state = dict(format_version=1, model=self.model.state_dict(),
            contract=dict(config={'model': config}, data={'tokenizer_sha256': file_sha(self.tokenizer)},
                          source_sha256={'pretrain_model.py': file_sha(Path(__file__).with_name('pretrain_model.py'))}),
            optimizer={}, scheduler={}, rng={},
            progress=dict(global_step=3, tokens_seen=128, final_saved=True))
        self.manager = CheckpointManager(self.root / 'checkpoints', min_free_gib=0)
        self.checkpoint = self.manager.save(self.state, is_best=True, final=True)

    def test_all_checkpoint_selections_and_real_tiny_model(self):
        for selection in ('final', 'best', 'latest'):
            model, tokenizer, info = load_model(self.root, selection, self.tokenizer, torch.device('cpu'))
            self.assertEqual(info['global_step'], 3)
            self.assertFalse(model.training)
            torch.testing.assert_close(model.embedding.weight, self.model.embedding.weight)
            result = generate(model, tokenizer, 'Sáng nay', max_new_tokens=3, temperature=0)
            self.assertGreaterEqual(result['generated_tokens'], 1)
            self.assertLessEqual(result['generated_tokens'], 3)

    def test_missing_final_no_automatic_fallback(self):
        (self.root / 'checkpoints/final.json').unlink()
        with self.assertRaisesRegex(RuntimeError, 'No final'):
            load_checkpoint(self.root, 'final')
        self.assertEqual(load_checkpoint(self.root, 'latest')[2]['step'], 3)

    def test_corrupt_checkpoint_rejected(self):
        with self.checkpoint.open('r+b') as stream: stream.write(b'corrupt')
        with self.assertRaisesRegex(RuntimeError, 'SHA-256'):
            load_checkpoint(self.root, 'latest')

    def test_hint_path_traversal_and_uncommitted_weights_rejected(self):
        hint = self.root / 'checkpoints/final.json'
        metadata = json.loads(hint.read_text(encoding='utf-8'))
        unsafe = dict(metadata, file='../outside.pt')
        atomic_json(hint, unsafe)
        with self.assertRaisesRegex(RuntimeError, 'filename'):
            load_checkpoint(self.root, 'final')
        atomic_json(hint, metadata)
        self.checkpoint.with_suffix('.json').unlink()
        with self.assertRaisesRegex(RuntimeError, 'committed metadata'):
            load_checkpoint(self.root, 'final')

    def test_wrong_tokenizer_and_model_source_rejected(self):
        wrong = self.root / 'wrong.model'
        wrong.write_bytes(b'wrong')
        with self.assertRaisesRegex(RuntimeError, 'Tokenizer SHA-256'):
            load_model(self.root, 'final', wrong, torch.device('cpu'))
        state = copy.deepcopy(self.state)
        state['contract']['source_sha256']['pretrain_model.py'] = 'bad-hash'
        self.manager.save(state, final=True)
        with self.assertRaisesRegex(RuntimeError, 'Model source'):
            load_checkpoint(self.root, 'final')

    def test_step_and_final_markers_checked(self):
        state = copy.deepcopy(self.state)
        state['progress']['final_saved'] = False
        self.manager.save(state, final=True)
        with self.assertRaisesRegex(RuntimeError, 'completed training'):
            load_checkpoint(self.root, 'final')

    def test_cli_prompt_file_json_output_and_no_checkpoint_mutation(self):
        prompt_file = self.root / 'prompt.txt'
        prompt_file.write_text('Sáng nay, tại Hà Nội,\ncông nghệ', encoding='utf-8')
        output = self.root / 'result.json'
        before = {p.name: file_sha(p) for p in (self.root / 'checkpoints').iterdir()}
        command = [sys.executable, '-B', '-X', 'utf8', str(Path(__file__).with_name('generate_pretrain.py')),
                   '--run-dir', str(self.root), '--tokenizer', str(self.tokenizer), '--device', 'cpu',
                   '--prompt-file', str(prompt_file), '--max-new-tokens', '2', '--temperature', '0',
                   '--output', str(output)]
        completed = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', timeout=90)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn('=== CONTINUATION ===', completed.stdout)
        result = json.loads(output.read_text(encoding='utf-8'))
        self.assertEqual(result['prompt'], prompt_file.read_text(encoding='utf-8'))
        self.assertEqual(result['checkpoint']['selection'], 'final')
        self.assertLessEqual(result['generated_tokens'], 2)
        second = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', timeout=90)
        self.assertNotEqual(second.returncode, 0)
        self.assertIn('never overwritten', second.stderr)
        after = {p.name: file_sha(p) for p in (self.root / 'checkpoints').iterdir()}
        self.assertEqual(before, after)


if __name__ == '__main__':
    unittest.main()