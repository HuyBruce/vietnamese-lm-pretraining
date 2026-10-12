import tempfile
from pathlib import Path
import unittest
import numpy as np
import json
import hashlib
import pyarrow as pa
import pyarrow.parquet as pq
from prepare_news_inventory import atomic_json, file_sha
from pretrain_core import mask_pii, metrics, reject_reasons, ngrams, word_bytes, minhash, union_bucket, compress_roots

class CoreTests(unittest.TestCase):
    def test_native_counts_match_python_unicode_semantics(self):
        from pretrain_core import character_counts
        text='Tiếng Việt 123² \t\n Ⅷ 👩‍💻\u2028' + ''.join(chr(i) for i in range(0,65536,17) if not 0xd800<=i<=0xdfff)
        self.assertEqual(character_counts(text),tuple(sum(getattr(c,n)() for c in text) for n in ['isalpha','isdigit','isspace']))
    def test_pii_preserves_dates_money_names(self):
        text='Nguyễn Văn An, ngày 29/09/2026, 1.234.567 đồng; email an@example.com; điện thoại 0912 345 678; CCCD: 012345678901.'
        value,counts=mask_pii(text)
        self.assertIn('Nguyễn Văn An, ngày 29/09/2026, 1.234.567 đồng',value)
        self.assertIn('<EMAIL>',value); self.assertIn('<PHONE>',value); self.assertIn('CCCD: <ID>',value)
        self.assertEqual(counts,dict(email=1,phone=1,identifier=1))

    def test_hashes_and_minhash(self):
        a=ngrams(word_bytes('Việt Nam có những thành phố rất đẹp.'),5)
        b=ngrams(word_bytes('VIỆT NAM có những thành phố rất đẹp!'),5)
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(minhash(a),minhash(b))

    def test_similarity_check_rejects_false_lsh_candidates(self):
        shingles=np.array([1,2,3,4,5,1,2,3,4,5,100,101,102,103,104],dtype=np.uint64)
        offsets=np.array([0,5,10,15],dtype=np.int64)
        parents=np.arange(3,dtype=np.int64)
        _,merged=union_bucket(np.arange(3),parents,offsets,shingles,.8)
        self.assertEqual(merged,1)
        np.testing.assert_array_equal(compress_roots(parents),[0,0,2])

    def test_quality_quarantines_encoding_and_not_natural_vietnamese(self):
        text='Hà Nội hôm nay có thời tiết đẹp. Người dân đi làm và học sinh đến trường như thường lệ. Các hoạt động diễn ra bình thường.'
        self.assertEqual(reject_reasons(text,metrics(text),'vi',.99,[]),[])
        self.assertIn('encoding_replacement',reject_reasons(text+'\ufffd',metrics(text),'vi',.99,[]))

class PipelineTests(unittest.TestCase):
    def test_exact_near_and_contamination_propagation(self):
        from prepare_pretrain import exact_dedup,sketch_all,near_dedup,database
        from pretrain_core import dedup_text,sha_text
        with tempfile.TemporaryDirectory(prefix='pretrain-test-') as tmp:
            root=Path(tmp)
            for folder in ['filtered','checkpoints/sketch','reports','sketch','scratch']: (root/folder).mkdir(parents=True,exist_ok=True)
            base=' '.join('từ'+str(i) for i in range(100))
            other=' '.join('chữ'+str(i) for i in range(100))
            third=' '.join('sách'+str(i) for i in range(100))
            texts=[base,base.upper(),base+' kết thúc',other,other+' đoạn cuối',third]
            rows=[dict(gid=i,id=hashlib.sha256(str(i).encode()).hexdigest(),source='test',text=t,
                dedup_sha256=sha_text(dedup_text(t)),benchmark_overlap=i==3,quality_score=1.,clean_char_count=len(t)) for i,t in enumerate(texts)]
            pq.write_table(pa.Table.from_pylist(rows),root/'filtered/part-000000.parquet')
            exact_dedup(root)
            sketch_all(root,[dict(name='part-000000.parquet',rows=6)],1)
            self.assertTrue((root/'checkpoints/sketch_all.json').exists())
            sketch_all(root,[dict(name='part-000000.parquet',rows=6)],1)
            near_dedup(root)
            with database(root) as db:
                self.assertEqual(db.execute('SELECT count(*) FROM representatives').fetchone()[0],5)
                self.assertEqual(db.execute('SELECT count(*) FROM selected').fetchone()[0],2)
                self.assertEqual(db.execute('SELECT count(*) FROM winners WHERE cluster_contaminated').fetchone()[0],1)
            decisions=pq.read_table(root/'decisions.parquet').to_pylist()
            self.assertEqual(sum(r['decision']=='benchmark_overlap_cluster' for r in decisions),2)
            self.assertTrue(all(r['retained_id'] is None for r in decisions if r['decision']=='benchmark_overlap_cluster'))
            from pretrain_finish import export_clean
            (root/'clean').mkdir()
            clean=export_clean(root)
            self.assertEqual(clean['documents'],2)
            ids=[]
            for spec in clean['files']:
                with pq.ParquetFile(root/spec['path']) as p: ids.extend(p.read(columns=['id']).column('id').to_pylist())
            self.assertEqual(len(set(ids)),2)
            from pretrain_finish import train_tokenizers,serialize,verify
            (root/'tokenizer').mkdir(); (root/'binary').mkdir()
            with (root/'tokenizer/train_sample.jsonl').open('w',encoding='utf-8') as f:
                for i,text in enumerate(texts): f.write(json.dumps(dict(id=str(i),text=text),ensure_ascii=False)+'\n')
            (root/'tokenizer/eval_sample.jsonl').write_text(json.dumps(dict(id='eval',text='Tiếng Việt có dấu.\nCâu cuối có 1.234 đồng.'),ensure_ascii=False)+'\n',encoding='utf-8')
            tokenizer=train_tokenizers(root,(320,384,448))
            manifest=serialize(root,clean,16,tokenizer)
            verify(root,manifest,clean)
            self.assertTrue(json.loads((root/'binary/manifest.json').read_text())['complete'])
            self.assertEqual(manifest['total_documents'],2)

    def test_tokenizer_sample_never_uses_validation(self):
        from pretrain_finish import make_tokenizer_samples
        with tempfile.TemporaryDirectory(prefix='sample-test-') as tmp:
            root=Path(tmp); (root/'tokenizer').mkdir(); (root/'checkpoints').mkdir()
            files=[]
            for split in ('train','validation'):
                path=root/(split+'.parquet')
                pq.write_table(pa.Table.from_pylist([dict(id='00000000'+split,text=split+' text')]),path)
                files.append(dict(path=path.name,split=split))
            with self.assertRaisesRegex(RuntimeError,'Insufficient tokenizer sample'):
                make_tokenizer_samples(root,dict(files=files))
            sample=(root/'tokenizer/train_sample.jsonl').read_text()
            self.assertIn('train text',sample); self.assertNotIn('validation text',sample)

    def test_restore_captions_and_news_agency_source(self):
        from prepare_pretrain import restore_meaningful_content
        import normalize_news as n
        with tempfile.TemporaryDirectory(prefix='restore-test-') as tmp:
            root=Path(tmp)
            for folder in ['inventory','manifests','normalized/audit']: (root/folder).mkdir(parents=True,exist_ok=True)
            text='Tiêu đề\nNội dung bài được giữ.\nẢnh: Người dân đi lại trên cầu mới.\nNguồn: Reuters\nTác giả: Nguyễn A'
            _,audit=n.normalize_text(text)
            path=root/'inventory/part-000000.parquet'
            pq.write_table(pa.Table.from_pylist([dict(id='a',text=text)]),path)
            atomic_json(root/'manifests/shards.json',[dict(name=path.name,sha256=file_sha(path))])
            (root/'normalized/audit'/f'{path.name}.jsonl').write_text(json.dumps(dict(id='a',**audit),ensure_ascii=False)+'\n',encoding='utf-8')
            result=restore_meaningful_content(root/'normalized',dict(name=path.name))['a'][0]
            self.assertIn('Ảnh: Người dân đi lại trên cầu mới.',result)
            self.assertIn('Nguồn: Reuters',result)
            self.assertNotIn('Tác giả:',result)

    def test_binary_roundtrip_eos_padding_and_shift(self):
        import sentencepiece as spm
        from pretrain_finish import tokenize_file
        from pretrain_dataset import CausalTokenDataset
        with tempfile.TemporaryDirectory(prefix='tokens-test-') as tmp:
            root=Path(tmp)
            for folder in ['binary','checkpoints','tokenizer']: (root/folder).mkdir()
            text='Tiếng Việt có dấu.\nNguyễn Văn An có 1.234 đồng.'
            prefix=root/'tokenizer/tiny'
            spm.SentencePieceTrainer.train(sentence_iterator=iter([text]*100),model_prefix=str(prefix),vocab_size=320,
                model_type='bpe',byte_fallback=True,normalization_rule_name='identity',remove_extra_whitespaces=False,
                unk_id=0,bos_id=1,eos_id=2,pad_id=3,hard_vocab_limit=False,minloglevel=2)
            model=spm.SentencePieceProcessor(model_file=str(prefix.with_suffix('.model')))
            pq.write_table(pa.Table.from_pylist([dict(id='a',text=text),dict(id='b',text=text+' Câu cuối.')]),root/'clean.parquet')
            tokenizer=dict(model='tokenizer/tiny.model',sha256=file_sha(prefix.with_suffix('.model')),pad_id=3)
            result=tokenize_file((root,dict(path='clean.parquet',split='train',rows=2,sha256=file_sha(root/'clean.parquet')),0,16,tokenizer))
            data=np.fromfile(root/result['file'],dtype=result['dtype'])
            boundaries=pq.read_table(root/result['documents_file']).to_pylist()
            for row,expected in zip(boundaries,[text,text+' Câu cuối.']):
                self.assertEqual(data[row['end']-1],2)
                self.assertEqual(model.decode(data[row['start']:row['end']-1].tolist()),expected)
            atomic_json(root/'binary/manifest.json',dict(complete=True,context_length=16,tokenizer=tokenizer,shards=[result]))
            dataset=CausalTokenDataset(root)
            self.assertEqual(len(dataset),result['sequences'])
            np.testing.assert_array_equal(dataset[0]['input_ids'].numpy(),data[:16])
            expected=data[1:17].astype(np.int64); expected[expected==3]=-100
            np.testing.assert_array_equal(dataset[0]['targets'].numpy(),expected)
            self.assertEqual(dataset[-1]['targets'].shape,(16,))
            # Close mmap handles before TemporaryDirectory removal on Windows.
            for mmap in dataset._maps.values(): mmap._mmap.close()

if __name__=='__main__': unittest.main()
