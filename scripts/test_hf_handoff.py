import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from hf_handoff import (excluded, load_token, package, safe_path, selected,
                        sha256, verify_files, verify_remote, write_json)


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='hf-handoff-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_token_loading_and_environment_precedence(self):
        (self.root / '.env').write_text('# local only\nhugging_key="test-token"\n', encoding='utf-8')
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(load_token(self.root), 'test-token')
        with patch.dict(os.environ, {'HF_TOKEN': 'environment-token'}):
            self.assertEqual(load_token(self.root), 'environment-token')
        (self.root / '.env').unlink()
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(load_token(self.root))

    def test_profiles_and_exclusions(self):
        self.assertTrue(selected('DataProcessed/pretrain_ready/clean/split=train/a.parquet', 'train'))
        self.assertFalse(selected('DataProcessed/pretrain_ready/sketch/large.bin', 'train'))
        self.assertTrue(selected('source.zip', 'all'))
        for name in ('scripts/.env', 'scripts/__pycache__/a.pyc', 'DataProcessed/inventory.lock',
                     'DataProcessed/work.duckdb.wal.tmp', '.cache/huggingface/a'):
            self.assertTrue(excluded(Path(name)))
        self.assertFalse(excluded(Path('scripts/train_pretrain.py')))

    def test_unsafe_manifest_paths_rejected(self):
        for name in ('../secret', '/absolute', 'D:/secret', 'a\\..\\secret', ''):
            with self.assertRaises(ValueError):
                safe_path(name)

    def test_verification_detects_corruption_and_missing_files(self):
        path = self.root / 'data.bin'
        path.write_bytes(b'abc')
        manifest = {'files': [{'path': path.name, 'bytes': 3, 'sha256': sha256(path)}]}
        self.assertEqual(verify_files(self.root, manifest)['checked_files'], 1)
        path.write_bytes(b'abd')
        with self.assertRaisesRegex(RuntimeError, 'SHA-256'):
            verify_files(self.root, manifest)
        path.unlink()
        with self.assertRaisesRegex(RuntimeError, 'Missing file'):
            verify_files(self.root, manifest)
        with self.assertRaisesRegex(RuntimeError, 'No files'):
            verify_files(self.root, {'files': []})

    def fixture(self):
        write_json(self.root / 'configs/handoff.json', {'repo_id': 'test/dataset', 'revision': 'test-release'})
        write_json(self.root / 'DataProcessed/pretrain_ready/binary/manifest.json', {'complete': True})
        write_json(self.root / 'DataProcessed/pretrain_ready/reports/verification.json', {'passed': True})
        (self.root / 'README.md').write_text('Test handoff\n', encoding='utf-8')
        (self.root / 'scripts').mkdir()
        (self.root / 'scripts/train.py').write_text('print(1)\n', encoding='utf-8')
        (self.root / 'scripts/.env').write_text('DO_NOT_PUBLISH=secret', encoding='utf-8')
        (self.root / '.env').write_text('DO_NOT_PUBLISH=secret', encoding='utf-8')
        (self.root / 'DataProcessed/inventory.lock').write_bytes(b'1')
        return self.root / 'runs/hf_handoff/staging'

    def test_package_clean_source_and_all_current_data(self):
        stage = self.fixture()
        package(self.root, stage)
        manifest = json.loads((stage / 'handoff-manifest.json').read_text(encoding='utf-8'))
        self.assertFalse(manifest['raw_corpus_complete'])
        with zipfile.ZipFile(stage / 'source.zip') as archive:
            self.assertIn('scripts/train.py', archive.namelist())
            self.assertFalse(any('.env' in name or name.startswith('runs/') for name in archive.namelist()))
        self.assertFalse((stage / 'DataProcessed/inventory.lock').exists())
        self.assertEqual(verify_files(stage, manifest)['checked_files'], 3)
        (stage / 'stale.txt').write_text('stale')
        package(self.root, stage)
        self.assertFalse((stage / 'stale.txt').exists())

    def test_package_rejects_incomplete_and_credentials(self):
        stage = self.fixture()
        write_json(self.root / 'DataProcessed/pretrain_ready/binary/manifest.json', {'complete': False})
        with self.assertRaisesRegex(RuntimeError, 'incomplete'):
            package(self.root, stage)
        write_json(self.root / 'DataProcessed/pretrain_ready/binary/manifest.json', {'complete': True})
        (self.root / 'scripts/leaked.txt').write_text('hf_' + 'x' * 30)
        with self.assertRaisesRegex(RuntimeError, 'credential'):
            package(self.root, stage)
        with self.assertRaisesRegex(ValueError, 'Staging'):
            package(self.root, self.root / 'scripts')

    def test_remote_size_and_lfs_hash_checks(self):
        item = SimpleNamespace(path='data.bin', size=3, lfs=SimpleNamespace(sha256='abc'))
        api = SimpleNamespace(list_repo_tree=lambda *args, **kwargs: [item])
        manifest = {'files': [{'path': 'data.bin', 'bytes': 3, 'sha256': 'abc'}]}
        self.assertEqual(verify_remote(api, 'test/repo', 'commit', manifest)['remote_lfs_sha256_checked'], 1)
        item.lfs.sha256 = 'wrong'
        with self.assertRaisesRegex(RuntimeError, 'checksum'):
            verify_remote(api, 'test/repo', 'commit', manifest)
        item.size = 4
        with self.assertRaisesRegex(RuntimeError, 'size'):
            verify_remote(api, 'test/repo', 'commit', manifest)


if __name__ == '__main__':
    unittest.main()