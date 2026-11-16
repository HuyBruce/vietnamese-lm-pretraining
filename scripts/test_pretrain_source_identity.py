"""Historical source identity checks without model weights or corpus artifacts."""
import copy
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pretrain_compat import select_contract
import train_pretrain


ROOT = Path(__file__).resolve().parents[1]
# Recorded from the historical run.json after checking the original source bytes.
HISTORICAL_SOURCES = {
    'pretrain_model.py': 'b174f36a9f1ddba2c8e34d24227caeca5917be1bc2c09bd37bbe7978c662bac5',
    'pretrain_training.py': 'b7575e632a69629f895b8fa775cc2837be15ae56e073891f2221ca95ef623c0b',
    'pretrain_checkpoint.py': '1ea74286111eaccfe117c3ca125d5e4640ee97e2f2724be60fb99ed780de0321',
    'pretrain_dataset.py': 'c240151ed3ddb7033694d9ee4fbfd94d817ae01119916297ebe153e1503d6acf',
    'train_pretrain.py': '7442792d72b0d5172db727f3530eccacfcad8a787f4c3592ca318f530297712e',
}


def fixture_contract():
    # Only source identities are historical. Config/data are synthetic; runtime
    # comes from this CPU test environment, not the original CUDA training run.
    return train_pretrain.make_contract(
        {'training': {'precision': 'fp32'}},
        {'manifest_sha256': hashlib.sha256(b'fixture manifest').hexdigest(),
         'tokenizer_sha256': hashlib.sha256(b'fixture tokenizer').hexdigest()},
        2, SimpleNamespace(type='cpu'))


def run_git(root, *arguments, data=None):
    return subprocess.run(['git', '-C', str(root), *arguments], input=data,
                          capture_output=True, check=True).stdout


class HistoricalSourceIdentityTests(unittest.TestCase):
    def historical_contract(self):
        contract = fixture_contract()
        contract['source_sha256'] = dict(HISTORICAL_SOURCES)
        return contract

    def test_source_bytes_match_historical_contract(self):
        for name, expected in HISTORICAL_SOURCES.items():
            with self.subTest(source=name):
                raw = (ROOT / 'scripts' / name).read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(), expected)

    def test_matching_historical_source_contract_is_accepted(self):
        previous = self.historical_contract()
        current = fixture_contract()
        self.assertIs(select_contract(current, previous), previous)

    def test_modified_source_file_is_rejected(self):
        previous = self.historical_contract()
        original = (ROOT / 'scripts/pretrain_dataset.py').read_bytes()
        changes = {
            'changed default': original.replace(b'max_open_shards=4', b'max_open_shards=5'),
            'CRLF conversion': original.replace(b'\n', b'\r\n'),
        }
        for change, raw in changes.items():
            with self.subTest(change=change), tempfile.TemporaryDirectory(
                    prefix='source-identity-') as tmp:
                self.assertNotEqual(raw, original)
                folder = Path(tmp)
                for name in HISTORICAL_SOURCES:
                    shutil.copyfile(ROOT / 'scripts' / name, folder / name)
                (folder / 'pretrain_dataset.py').write_bytes(raw)
                with patch.object(train_pretrain, '__file__', str(folder / 'train_pretrain.py')):
                    current = fixture_contract()
                self.assertNotEqual(current['source_sha256']['pretrain_dataset.py'],
                                    HISTORICAL_SOURCES['pretrain_dataset.py'])
                with self.assertRaisesRegex(RuntimeError, 'source upgrade is not audited'):
                    select_contract(current, previous)

    def test_data_config_runtime_and_horizon_mismatches_are_rejected(self):
        previous = self.historical_contract()
        changes = (
            ('config', 'training', {'precision': 'bf16'}),
            ('data', 'manifest_sha256', hashlib.sha256(b'changed manifest').hexdigest()),
            ('data', 'tokenizer_sha256', hashlib.sha256(b'changed tokenizer').hexdigest()),
            ('runtime', 'python', 'different runtime'),
            (None, 'total_steps', 3),
        )
        for section, key, value in changes:
            with self.subTest(section=section, key=key):
                current = copy.deepcopy(previous)
                (current if section is None else current[section])[key] = value
                with self.assertRaisesRegex(RuntimeError, 'Run config/data/code/runtime differs'):
                    select_contract(current, previous)

    @unittest.skipUnless(shutil.which('git'), 'Git is required for checkout regression')
    def test_git_checkout_preserves_all_contract_sources(self):
        # A disposable index exercises real checkout conversion without commits
        # or modifications to the project's index, source files or Git config.
        for autocrlf in ('false', 'true', 'input'):
            with self.subTest(autocrlf=autocrlf), tempfile.TemporaryDirectory(
                    prefix='source-checkout-') as tmp:
                folder = Path(tmp)
                run_git(folder, 'init', '--quiet')
                run_git(folder, 'config', 'core.autocrlf', autocrlf)
                paths = ['.gitattributes'] + ['scripts/' + name for name in HISTORICAL_SOURCES]
                for path in paths:
                    raw = (ROOT / path).read_bytes()
                    # Seed raw blobs as stored by the historical snapshot;
                    # git add here would normalize the unrelated CRLF files.
                    oid = run_git(folder, 'hash-object', '-w', '--stdin', data=raw).decode().strip()
                    run_git(folder, 'update-index', '--add', '--cacheinfo', f'100644,{oid},{path}')
                run_git(folder, 'checkout-index', '--all', '--force')
                for name, expected in HISTORICAL_SOURCES.items():
                    path = 'scripts/' + name
                    raw = (folder / path).read_bytes()
                    self.assertEqual(hashlib.sha256(raw).hexdigest(), expected, path)
                    attributes = run_git(folder, 'check-attr', 'text', 'eol', '--', path).decode()
                    if name == 'pretrain_dataset.py':
                        self.assertIn(': text: set', attributes)
                        self.assertIn(': eol: lf', attributes)
                    else:
                        self.assertIn(': text: unspecified', attributes)
                        self.assertIn(': eol: unspecified', attributes)


if __name__ == '__main__':
    unittest.main()
