from pathlib import Path
import tempfile
import unittest
import pyarrow as pa
import pyarrow.parquet as pq
import sentencepiece as spm
from prepare_news_inventory import atomic_json, file_sha
from pretrain_finish import tokenize_file
from verify_pretrain_ready import audit


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='pretrain-audit-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for folder in ('clean', 'tokenizer', 'binary', 'checkpoints', 'reports'):
            (self.root / folder).mkdir()
        text = 'Tiếng Việt có dấu.\nNguyễn Văn An có 1.234 đồng. '
        prefix = self.root / 'tokenizer/tiny'
        spm.SentencePieceTrainer.train(sentence_iterator=iter([text] * 100),
            model_prefix=str(prefix), vocab_size=320, model_type='bpe',
            byte_fallback=True, normalization_rule_name='identity',
            remove_extra_whitespaces=False, unk_id=0, bos_id=1, eos_id=2,
            pad_id=3, hard_vocab_limit=False, minloglevel=2)
        model = spm.SentencePieceProcessor(model_file=str(prefix.with_suffix('.model')))
        tokenizer = dict(model='tokenizer/tiny.model', sha256=file_sha(prefix.with_suffix('.model')),
                         actual_vocab=model.get_piece_size(), eos_id=2, pad_id=3)
        atomic_json(self.root / 'checkpoints/tokenizer.json', tokenizer)
        sources = []
        records = []
        for index, split in enumerate(('train', 'train', 'train', 'validation')):
            path = self.root / 'clean' / f'{index}.parquet'
            rows = [dict(id=f'{index}-{i}', text=text * 300) for i in range(2)]
            pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=1)
            source = dict(path=path.relative_to(self.root).as_posix(), split=split,
                          rows=len(rows), sha256=file_sha(path))
            sources.append(source)
            records.append(tokenize_file((self.root, source, index, 4096, tokenizer)))
        atomic_json(self.root / 'clean/manifest.json', dict(files=sources, documents=8))
        self.manifest = dict(complete=True, context_length=4096, tokenizer=tokenizer,
            shards=records, total_documents=8, total_tokens=sum(s['tokens'] for s in records),
            total_sequences=sum(s['sequences'] for s in records))
        atomic_json(self.root / 'binary/manifest.json', self.manifest)
        atomic_json(self.root / 'reports/verification.json', dict(passed=True))

    def test_complete_audit_and_loaders(self):
        result = audit(self.root)
        self.assertTrue(result['passed'])
        self.assertEqual(result['scope'], 'complete_dataset')
        self.assertEqual(result['binary_shards_checked'], 4)
        self.assertEqual(result['roundtrip_documents'], 8)
        self.assertEqual(set(result['dataloader']), {'train', 'validation'})
        self.assertEqual(result['dataloader']['train']['batch_shape'], [2, 4096])

    def test_incomplete_manifest_rejected_but_committed_audit_allowed(self):
        self.manifest['complete'] = False
        atomic_json(self.root / 'binary/manifest.json', self.manifest)
        with self.assertRaisesRegex(RuntimeError, 'not complete'):
            audit(self.root)
        result = audit(self.root, committed_only=True)
        self.assertEqual(result['scope'], 'committed_only')
        self.assertEqual(result['dataloader'], {})
        self.assertFalse((self.root / 'reports/independent_verification.json').exists())

    def test_corrupt_binary_rejected(self):
        path = self.root / self.manifest['shards'][0]['file']
        with path.open('r+b') as stream:
            stream.write(b'\xff\xff')
        with self.assertRaisesRegex(RuntimeError, 'Checksum mismatch'):
            audit(self.root)


if __name__ == '__main__':
    unittest.main()
