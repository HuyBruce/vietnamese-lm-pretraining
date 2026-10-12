import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout

from pretrain_compat import select_contract
from pretrain_logging import RunLogger, duration, format_event
from pretrain_report import export_report, read_history, summarize


class LoggingTests(unittest.TestCase):
    def test_format_has_readable_units_and_no_raw_json(self):
        text = format_event(dict(time=0, event='train', global_step=10, total_steps=100,
                                 loss=3.123456, learning_rate=.00001, grad_norm=1.23456,
                                 tokens_seen=123456, tokens_per_second=1234.5, seconds=2.456,
                                 peak_vram_gib=2.12345, eta_train_seconds=3601))
        for expected in ('10/100 (10.00%)', '3.1235', '1.00e-05', '123,456',
                         '1,234 tok/s', '2.46 s/step', '2.12 GiB', '01:00:01'):
            self.assertIn(expected, text)
        self.assertNotIn('{', text)
        self.assertEqual(duration(None), 'N/A')

    def test_logger_records_quiet_updates_appends_and_reports_errors(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()) as console:
            run = Path(tmp)
            for _ in range(2):
                with RunLogger(run) as logger:
                    logger.emit('validation_quick', console=False, global_step=1, loss=2.12345678,
                                tokens=8, sequences=1)
            with self.assertRaisesRegex(RuntimeError, 'test failure'):
                with RunLogger(run) as logger:
                    logger.emit('validation_full', global_step=2, loss=1000, tokens=8, sequences=1)
                    raise RuntimeError('test failure')
            rows, warnings = read_history(run / 'metrics.jsonl')
            self.assertFalse(warnings)
            self.assertEqual(len({row['session_id'] for row in rows}), 3)
            quick = [r for r in rows if r['event'] == 'validation_quick']
            self.assertEqual(len(quick), 2)
            self.assertEqual(quick[0]['loss'], 2.12345678)
            self.assertNotIn('VALIDATION_QUICK', console.getvalue())
            self.assertIn('VALIDATION_QUICK', (run / 'train.log').read_text())
            self.assertIsNone(next(r for r in rows if r['event'] == 'validation_full')['perplexity'])
            self.assertIn('RuntimeError: test failure', rows[-1]['traceback'])
            summary = json.loads((run / 'summary.json').read_text())
            self.assertEqual(summary['sessions'][-1]['status'], 'error')
            with (run / 'metrics.csv').open(encoding='utf-8-sig', newline='') as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), len(rows))
            self.assertIn('# Training report', (run / 'report.md').read_text())

    def test_partial_record_is_preserved_and_following_session_readable(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            run = Path(tmp)
            (run / 'metrics.jsonl').write_text('{"event":', encoding='utf-8')
            with RunLogger(run):
                pass
            rows, warnings = read_history(run / 'metrics.jsonl')
            self.assertEqual(len(warnings), 1)
            self.assertEqual(rows[0]['event'], 'session_open')
            self.assertTrue((run / 'metrics.jsonl').read_text().startswith('{"event":\n'))

    def test_rollback_excluded_and_uncommitted_not_reported_as_saved(self):
        def row(event, step, **values):
            return dict(event=event, global_step=step, session_id='test', status='event', **values)
        rows = [row('session_start', 0, total_steps=10),
                row('train', 1, loss=4, tokens=10),
                row('checkpoint', 1, tokens_seen=10),
                row('train', 2, loss=1, tokens=10),
                row('validation_full', 2, loss=.5),
                row('session_start', 1, tokens_seen=10),
                row('train', 2, loss=3, tokens=20),
                row('validation_quick', 2, loss=2),
                row('validation_full', 2, loss=3),
                row('session_end', 2, safely_saved=True, tokens_seen=30),
                row('train', 3, loss=0, tokens=20)]
        summary = summarize(rows, [])
        self.assertEqual(summary['rolled_back_records'], 2)
        self.assertEqual(summary['uncommitted_records'], 1)
        self.assertEqual(summary['committed_step'], 2)
        self.assertEqual(summary['committed_tokens_seen'], 30)
        self.assertAlmostEqual(summary['logged_token_weighted_train_loss'], 100 / 30)
        self.assertEqual(summary['best_observed_full_validation_loss'], 3)
        self.assertEqual(summary['best_observed_quick_validation_loss'], 2)

    def test_legacy_sparse_history_exports_without_modifying_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            rows = [dict(event='session_start', global_step=0, total_steps=20, time=1),
                    dict(event='train', global_step=10, loss=4, tokens=100, time=2),
                    dict(event='session_end', global_step=10, tokens_seen=1000,
                         safely_saved=True, session_seconds=10, time=3)]
            original = '\n'.join(json.dumps(row) for row in rows) + '\n'
            (run / 'metrics.jsonl').write_text(original)
            summary = export_report(run, run / 'export')
            self.assertEqual(summary['committed_train_records'], 1)
            self.assertEqual(summary['committed_tokens_seen'], 1000)
            self.assertIsNone(summary['committed_training_seconds'])
            self.assertIsNone(summary['best_observed_full_validation_loss'])
            self.assertEqual((run / 'metrics.jsonl').read_text(), original)
            self.assertEqual(summary['sessions'][0]['session_id'], 'legacy-1')


class CompatibilityTests(unittest.TestCase):
    def test_only_exact_audited_hash_pair_allowed(self):
        old = dict(config={'seed': 1}, data={'sha': 'data'}, runtime={'torch': 'version'},
                   source_sha256={'trainer': 'old', 'model': 'fixed'})
        new = copy.deepcopy(old)
        new['source_sha256']['trainer'] = 'new'
        upgrades = {'trainer': {'before': 'old', 'after': 'new'}}
        self.assertEqual(select_contract(new, old, upgrades), old)
        self.assertEqual(select_contract(new, new, {}), new)
        for section in ('config', 'data', 'runtime'):
            changed = copy.deepcopy(new)
            changed[section]['changed'] = True
            with self.assertRaisesRegex(RuntimeError, 'differs'):
                select_contract(changed, old, upgrades)
        for source in ('trainer', 'model'):
            changed = copy.deepcopy(new)
            changed['source_sha256'][source] = 'unaudited'
            with self.assertRaisesRegex(RuntimeError, 'not audited'):
                select_contract(changed, old, upgrades)

    def test_audit_matches_current_files(self):
        from pretrain_checkpoint import file_sha
        folder = Path(__file__).resolve().parent
        upgrades = json.loads((folder / 'pretrain_logging_upgrade.json').read_text())
        self.assertEqual(set(upgrades), {'train_pretrain.py', 'pretrain_checkpoint.py'})
        for name, hashes in upgrades.items():
            self.assertEqual(hashes['after'], file_sha(folder / name))


if __name__ == '__main__':
    unittest.main()