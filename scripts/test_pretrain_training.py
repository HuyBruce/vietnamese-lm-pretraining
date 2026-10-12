import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from pretrain_checkpoint import (CheckpointManager, IncompatibleCheckpoint, capture_rng,
                                 restore_rng, run_lock)
from pretrain_model import CausalLM
from pretrain_training import (EpochBatches, TokenSchedule, evaluate, make_loader,
                               make_optimizer, seed_all, train_update)
from train_pretrain import StopRequest, checkpoint_state


def tiny_config():
    config = json.loads((Path(__file__).resolve().parents[1] / 'configs/pretrain_110m.json').read_text())
    config['model'].update(vocab_size=32, num_layers=2, hidden_size=16, num_heads=2,
                           intermediate_size=32, context_length=8)
    config['training'].update(precision='fp32', micro_batch_size=1,
                              gradient_accumulation_steps=2, loss_chunk_size=3, warmup_steps=2)
    return config


class TinyDataset:
    def __init__(self, size=11):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        stream = (torch.arange(9) + index) % 28 + 4
        targets = stream[1:].clone()
        if index % 2:
            targets[-3:] = -100
        return dict(input_ids=stream[:-1], targets=targets)


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        seed_all(123)
        self.config = tiny_config()
        self.device = torch.device('cpu')

    def components(self):
        model = CausalLM(self.config['model'])
        optimizer = make_optimizer(model, self.config['training'], self.device)
        return model, optimizer, TokenSchedule(optimizer, self.config['training'], 10)

    def test_parameter_count_and_causality(self):
        model, _, _ = self.components()
        m = self.config['model']
        d, f, layers = m['hidden_size'], m['intermediate_size'], m['num_layers']
        self.assertEqual(sum(p.numel() for p in model.parameters()), m['vocab_size'] * d + layers * (4*d*d + 3*d*f + 2*d) + d)
        model.eval()
        x = TinyDataset()[0]['input_ids'].unsqueeze(0)
        changed = x.clone()
        changed[:, 4:] = 1
        torch.testing.assert_close(model(x)[:, :4], model(changed)[:, :4], rtol=0, atol=0)
        self.assertEqual(model(x).shape, (1, 8, 32))
        with self.assertRaises(ValueError):
            model(torch.zeros(1, 9, dtype=torch.long))

    def test_full_model_count_and_tied_head(self):
        config = json.loads((Path(__file__).resolve().parents[1] / 'configs/pretrain_110m.json').read_text())
        with torch.device('meta'):
            full = CausalLM(config['model'])
        self.assertEqual(sum(p.numel() for p in full.parameters()), 109529856)
        model, _, _ = self.components()
        x = TinyDataset()[0]['input_ids'][None]
        torch.testing.assert_close(model(x), F.linear(model.hidden(x), model.embedding.weight))
        self.assertEqual(len([name for name, _ in model.named_parameters() if 'embedding' in name]), 1)

    def test_all_ignored_targets_have_zero_loss_and_gradient(self):
        model, _, _ = self.components()
        x = TinyDataset()[0]['input_ids'][None]
        loss = model.loss_sum(x, torch.full_like(x, -100), 3)
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertTrue(all(p.grad is None or not torch.count_nonzero(p.grad) for p in model.parameters()))

    def test_workers_prefetch_resumes_exact_next_batch(self):
        training = self.config['training']
        dataset = TinyDataset()
        iterator = iter(make_loader(dataset, training, 0, 0, 2, self.device))
        next(iterator)
        next(iterator)
        expected = next(iterator)
        del iterator
        resumed = iter(make_loader(dataset, training, 0, 2, 2, self.device))
        actual = next(resumed)
        for key in actual:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        del resumed

    def test_chunked_loss_and_gradients_match_full_head(self):
        model, _, _ = self.components()
        other = copy.deepcopy(model)
        item = TinyDataset()[1]
        x, y = item['input_ids'][None], item['targets'][None]
        loss = model.loss_sum(x, y, 3)
        reference = F.cross_entropy(other(x).reshape(-1, 32), y.reshape(-1), ignore_index=-100, reduction='sum')
        torch.testing.assert_close(loss, reference)
        loss.backward()
        reference.backward()
        for a, b in zip(model.parameters(), other.parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=1e-4, atol=1e-6)

    def test_accumulation_matches_batch_with_padding_and_partial_update(self):
        a, opt_a, sched_a = self.components()
        b = copy.deepcopy(a)
        cfg = dict(self.config['training'], micro_batch_size=2, gradient_accumulation_steps=1)
        opt_b = make_optimizer(b, cfg, self.device)
        sched_b = TokenSchedule(opt_b, cfg, 10)
        data = TinyDataset(3)
        ia = iter(make_loader(data, self.config['training'], 0, 0, 0, self.device))
        ib = iter(make_loader(data, cfg, 0, 0, 0, self.device))
        for expected_samples in (2, 1):
            ra = train_update(a, opt_a, sched_a, ia, self.config['training'], self.device)
            rb = train_update(b, opt_b, sched_b, ib, cfg, self.device)
            self.assertEqual(ra['samples'], expected_samples)
            self.assertAlmostEqual(ra['loss'], rb['loss'], places=5)
            for pa, pb in zip(a.parameters(), b.parameters()):
                torch.testing.assert_close(pa, pb, rtol=1e-4, atol=1e-6)

    def test_sampler_cursor_is_independent_of_prefetch(self):
        sampler = EpochBatches(19, 3, 123, 0)
        prefetched = list(sampler)
        self.assertEqual(sampler.cursor, 0)
        self.assertEqual(list(EpochBatches(19, 3, 123, 0, 6)), prefetched[2:])
        self.assertEqual(sorted(sum(prefetched, [])), list(range(19)))

    def test_stop_request_during_accumulation_finishes_update(self):
        model, optimizer, scheduler = self.components()
        with tempfile.TemporaryDirectory() as tmp:
            stop = StopRequest(Path(tmp) / 'STOP')
            def request(micro):
                if micro == 0:
                    stop.handle(None, None)
            iterator = iter(make_loader(TinyDataset(), self.config['training'], 0, 0, 0, self.device))
            result = train_update(model, optimizer, scheduler, iterator, self.config['training'], self.device, request)
            self.assertTrue(stop.requested)
            self.assertEqual(result['samples'], 2)
            self.assertEqual(scheduler.completed_steps, 1)

    def test_resume_matches_uninterrupted_model_optimizer_lr_and_rng(self):
        model, optimizer, scheduler = self.components()
        training = self.config['training']
        data = TinyDataset()
        iterator = iter(make_loader(data, training, 0, 0, 0, self.device))
        progress = dict(global_step=0, cursor=0)
        for _ in range(2):
            result = train_update(model, optimizer, scheduler, iterator, training, self.device)
            progress['global_step'] += 1
            progress['cursor'] += result['samples']
        with tempfile.TemporaryDirectory() as tmp:
            manager = CheckpointManager(tmp, min_free_gib=0)
            manager.save(checkpoint_state(model, optimizer, scheduler, progress, {'test': 1}))
            random_values = (random.random(), np.random.random(), torch.rand(3))
            expected = []
            for _ in range(2):
                expected.append(train_update(model, optimizer, scheduler, iterator, training, self.device))
            resumed, opt, sched = self.components()
            state = manager.load_latest({'test': 1})
            resumed.load_state_dict(state['model'])
            opt.load_state_dict(state['optimizer'])
            sched.load_state_dict(state['scheduler'])
            restore_rng(state['rng'])
            self.assertEqual(random.random(), random_values[0])
            self.assertEqual(np.random.random(), random_values[1])
            torch.testing.assert_close(torch.rand(3), random_values[2], rtol=0, atol=0)
            iterator = iter(make_loader(data, training, 0, state['progress']['cursor'], 0, self.device))
            for reference in expected:
                actual = train_update(resumed, opt, sched, iterator, training, self.device)
                self.assertEqual(actual, reference)
            for pa, pb in zip(model.parameters(), resumed.parameters()):
                torch.testing.assert_close(pa, pb, rtol=0, atol=0)
            self.assertEqual(scheduler.state_dict(), sched.state_dict())
            for a, b in zip(optimizer.state.values(), opt.state.values()):
                for key in a:
                    torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)

    def test_validation_does_not_change_weights_gradients_or_rng(self):
        model, _, _ = self.components()
        before = copy.deepcopy(model.state_dict())
        rng = torch.get_rng_state()
        outcome = evaluate(model, TinyDataset(), self.config['training'], self.device, 3)
        self.assertEqual(outcome['sequences'], 3)
        self.assertTrue(model.training)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        for key, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, before[key], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_tiny_batch_overfits(self):
        model, _, _ = self.components()
        training = dict(self.config['training'], learning_rate=.01, min_learning_rate=.005,
                        warmup_steps=0, gradient_accumulation_steps=1)
        optimizer = make_optimizer(model, training, self.device)
        scheduler = TokenSchedule(optimizer, training, 60)
        data = TinyDataset(1)
        first = None
        for _ in range(60):
            batch = iter(make_loader(data, training, 0, 0, 0, self.device))
            loss = train_update(model, optimizer, scheduler, batch, training, self.device)['loss']
            first = loss if first is None else first
        self.assertLess(loss, first * .3)


class SessionTests(unittest.TestCase):
    def test_cli_sessions_resume_and_finish_without_replaying_data(self):
        import subprocess
        import sys
        from pretrain_checkpoint import atomic_json, file_sha
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / 'data'
            (data / 'binary').mkdir(parents=True)
            (data / 'tokenizer').mkdir()
            tokenizer = data / 'tokenizer/test.model'
            tokenizer.write_bytes(b'test tokenizer identity')
            shards = []
            for split, count in [('train', 5), ('validation', 2)]:
                tokens = (np.arange(count * 8 + 1) % 28 + 4).astype('<u2')
                path = data / f'binary/{split}.bin'
                tokens.tofile(path)
                shards.append(dict(file=f'binary/{split}.bin', split=split, sequences=count, dtype='<u2'))
            atomic_json(data / 'binary/manifest.json', dict(complete=True, context_length=8,
                tokenizer=dict(actual_vocab=32, pad_id=3, model='tokenizer/test.model', sha256=file_sha(tokenizer)),
                shards=shards))
            config = tiny_config()
            config['training']['epochs'] = 2
            config['checkpoint'].update(min_free_gib=0, every_steps=1)
            config['evaluation'].update(quick_every_steps=1, full_every_steps=2, quick_sequences=1)
            config_path = root / 'config.json'
            atomic_json(config_path, config)
            run = root / 'run'
            command = [sys.executable, '-B', str(Path(__file__).with_name('train_pretrain.py')),
                       '--config', str(config_path), '--data', str(data), '--run-dir', str(run), '--device', 'cpu']
            for index, extra in enumerate((['--session-steps', '1'], ['--session-steps', '1'], [])):
                completed = subprocess.run(command + extra, capture_output=True, text=True, timeout=120)
                self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                if index == 0:
                    # Exercise the audited upgrade with a real serialized checkpoint,
                    # without changing model/optimizer/RNG/cursor or touching user runs.
                    contract = json.loads((run / 'run.json').read_text())
                    manager = CheckpointManager(run / 'checkpoints', min_free_gib=0)
                    state = manager.load_latest(contract)
                    upgrades = json.loads(Path(__file__).with_name('pretrain_logging_upgrade.json').read_text())
                    for name, hashes in upgrades.items():
                        contract['source_sha256'][name] = hashes['before']
                    state['contract'] = contract
                    manager.save(state)
                    atomic_json(run / 'run.json', contract)
            events = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
            starts = [row for row in events if row['event'] == 'session_start']
            self.assertEqual([row['global_step'] for row in starts], [0, 1, 2])
            self.assertEqual([row['cursor'] for row in starts], [0, 2, 4])
            self.assertEqual([row['global_step'] for row in events if row['event'] == 'train'], list(range(1, 7)))
            self.assertEqual(len({row['session_id'] for row in starts}), 3)
            self.assertTrue(any(row['event'] == 'resume' for row in events))
            self.assertTrue(any(row['event'] == 'checkpoint' for row in events))
            self.assertNotIn('"event": "train"', completed.stdout)
            self.assertIn('TRAIN', completed.stdout)
            summary = json.loads((run / 'summary.json').read_text())
            self.assertEqual(summary['committed_step'], 6)
            self.assertEqual(summary['committed_tokens_seen'], 80)
            self.assertEqual(summary['committed_train_records'], 6)
            self.assertEqual(summary['uncommitted_records'], 0)
            self.assertIsNotNone(summary['best_observed_full_validation_loss'])
            for name in ('train.log', 'metrics.csv', 'report.md'):
                self.assertTrue((run / name).is_file())
            manager = CheckpointManager(run / 'checkpoints', min_free_gib=0)
            contract = json.loads((run / 'run.json').read_text())
            state = manager.load_latest(contract)
            self.assertEqual(state['progress']['global_step'], 6)
            self.assertEqual(state['progress']['epoch'], 2)
            self.assertEqual(state['progress']['cursor'], 0)
            self.assertEqual(state['progress']['tokens_seen'], 80)
            self.assertTrue(state['progress']['final_saved'])
            completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            again = manager.load_latest(contract)
            self.assertEqual(again['progress'], state['progress'])
            # Changing config must fail, not start a new model in the same run.
            config['training']['learning_rate'] *= 2
            atomic_json(config_path, config)
            failed = subprocess.run(command, capture_output=True, text=True, timeout=120)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn('differs', failed.stderr)


class CheckpointTests(unittest.TestCase):
    def state(self, step):
        return dict(format_version=1, contract={'config': 1}, model={'w': torch.ones(2)},
                    optimizer={}, scheduler={}, progress={'global_step': step}, rng=capture_rng())

    def test_corruption_fallback_and_incompatible_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = CheckpointManager(tmp, min_free_gib=0)
            manager.save(self.state(1))
            last = manager.save(self.state(2))
            last.write_bytes(b'broken')
            (Path(tmp) / 'interrupted.pt.tmp').write_bytes(b'incomplete')
            self.assertEqual(manager.load_latest({'config': 1})['progress']['global_step'], 1)
            with self.assertRaises(IncompatibleCheckpoint):
                manager.load_latest({'config': 2})

    def test_all_corrupt_refuses_fresh_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = CheckpointManager(tmp, min_free_gib=0)
            manager.save(self.state(1)).write_bytes(b'broken')
            with self.assertRaisesRegex(RuntimeError, 'refusing to restart'):
                manager.load_latest({'config': 1})

    def test_failed_save_and_disk_full_preserve_previous(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = CheckpointManager(tmp, min_free_gib=0)
            first = manager.save(self.state(1))
            with patch('pretrain_checkpoint.torch.save', side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    manager.save(self.state(2))
            self.assertTrue(first.exists())
            self.assertFalse(list(Path(tmp).glob('*.tmp')))
            with patch('pretrain_checkpoint.shutil.disk_usage') as disk:
                disk.return_value.free = 0
                with self.assertRaisesRegex(OSError, 'Insufficient disk'):
                    manager.save(self.state(3))
            self.assertEqual(manager.load_latest({'config': 1})['progress']['global_step'], 1)

    def test_retention_protects_best_and_final(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = CheckpointManager(tmp, keep_last=2, min_free_gib=0)
            best = manager.save(self.state(1), is_best=True)
            manager.save(self.state(2))
            final = manager.save(self.state(3), final=True)
            manager.save(self.state(4))
            manager.save(self.state(5))
            self.assertTrue(best.exists())
            self.assertTrue(final.exists())
            self.assertEqual(len(list(Path(tmp).glob('step-*.pt'))), 4)

    def test_exclusive_run_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            with run_lock(tmp):
                with self.assertRaises(RuntimeError):
                    with run_lock(tmp):
                        self.fail('Second trainer must not acquire the lock')
            with run_lock(tmp):
                pass


if __name__ == '__main__':
    unittest.main()
