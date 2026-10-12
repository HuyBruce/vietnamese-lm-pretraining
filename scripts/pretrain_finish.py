"""Clean Parquet export, train-only tokenizer comparison, and token serialization."""
import json
import math
from pathlib import Path
import shutil
import uuid
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import sentencepiece as spm
from prepare_news_inventory import atomic_json, file_sha

def export_clean(output):
    from prepare_pretrain import database, log
    marker=output/'checkpoints/clean.json'
    if marker.exists(): return json.loads(marker.read_text())
    attempt=output/'checkpoints/clean_attempt.json'
    if attempt.exists():
        destination=Path(json.loads(attempt.read_text())['destination']).resolve()
        owned=(output/'clean').resolve()
        if destination.parent!=owned or not destination.name.startswith('corpus-') or destination.is_symlink():
            raise RuntimeError('Unsafe incomplete export path')
        # Only this script-owned, uncommitted output is regenerated on retry.
        if destination.exists(): shutil.rmtree(destination)
    else:
        destination=output/'clean'/('corpus-'+uuid.uuid4().hex[:8])
        atomic_json(attempt,dict(destination=str(destination.resolve())))
    with database(output) as db:
        db.execute('SET threads=1')
        db.execute('SET partitioned_write_max_open_files=4')
        db.execute('''COPY (SELECT f.*,s.cluster AS duplicate_cluster,s.split,
            (('0x'||substr(s.shuffle_key,1,2))::UBIGINT % 64) AS bucket
            FROM read_parquet($source) f JOIN selected s USING(gid)
            ORDER BY s.split,bucket,s.shuffle_key)
            TO $destination (FORMAT PARQUET, PARTITION_BY(split,bucket), COMPRESSION ZSTD, ROW_GROUP_SIZE 2048)''',
            {'source':str(output/'filtered/*.parquet'),'destination':str(destination)})
    files=[]
    for path in sorted(destination.rglob('*.parquet')):
        split=next(p[6:] for p in path.parts if p.startswith('split='))
        with pq.ParquetFile(path) as p: rows=p.metadata.num_rows
        files.append(dict(path=path.relative_to(output).as_posix(),split=split,rows=rows,sha256=file_sha(path)))
    if not files: raise RuntimeError('No clean documents survived')
    result=dict(files=files,documents=sum(f['rows'] for f in files))
    atomic_json(marker,result); atomic_json(output/'clean/manifest.json',result)
    log(f'Clean corpus: {result["documents"]:,} documents, {len(files)} files')
    return result

def make_tokenizer_samples(output,clean):
    from prepare_pretrain import log
    marker=output/'checkpoints/tokenizer_sample.json'
    if marker.exists(): return json.loads(marker.read_text())
    chars=0; train_count=0; eval_count=0
    with (output/'tokenizer/train_sample.jsonl').open('w',encoding='utf-8') as train, (output/'tokenizer/eval_sample.jsonl').open('w',encoding='utf-8') as evaluation:
        for spec in clean['files']:
            if spec['split']!='train': continue
            with pq.ParquetFile(output/spec['path']) as p:
                for batch in p.iter_batches(batch_size=128,columns=['id','text'],use_threads=False):
                    for row in batch.to_pylist():
                        bucket=int(row['id'][:8],16)%1000
                        record=json.dumps(row,ensure_ascii=False)+'\n'
                        if bucket<30 and chars<30000000:
                            train.write(record); chars+=len(row['text']); train_count+=1
                        elif 30<=bucket<35 and eval_count<2000:
                            evaluation.write(record); eval_count+=1
    if train_count<1000 or eval_count<100: raise RuntimeError('Insufficient tokenizer sample; need a larger clean corpus')
    result=dict(training_documents=train_count,training_characters=chars,evaluation_documents=eval_count,
        provenance='Both samples exclusively drawn from train split; disjoint deterministic ID buckets.')
    atomic_json(marker,result); log(f'Tokenizer samples: {result}')
    return result

def sample_sentences(path):
    with path.open(encoding='utf-8') as f:
        for line in f:
            text=json.loads(line)['text']
            # Bound sentencepiece training memory; documents remain intact in the corpus.
            for start in range(0,len(text),3000): yield text[start:start+3000]

def train_tokenizers(output,vocab_sizes=(32000,48000,64000)):
    from prepare_pretrain import log
    marker=output/'checkpoints/tokenizer.json'
    if marker.exists():
        saved=json.loads(marker.read_text())
        if [r['requested_vocab'] for r in saved['results']]!=list(vocab_sizes): raise RuntimeError('Tokenizer vocabulary candidates changed')
        return saved
    evaluation=[json.loads(line)['text'] for line in (output/'tokenizer/eval_sample.jsonl').read_text(encoding='utf-8').splitlines()]
    results=[]
    probes={'names':['Nguyễn Ái Quốc','Thành phố Hồ Chí Minh','Đắk Lắk','OpenAI'],
            'numbers':['1.234.567 đồng','29/09/2026','3,14159','2026'],
            'vietnamese':['học sinh','công nghệ thông tin','đất nước','chuẩn hóa dữ liệu'],
            'english':['machine learning','artificial intelligence','New York']}
    for size in vocab_sizes:
        prefix=output/'tokenizer'/f'bpe-{size}'
        trained=output/'tokenizer'/f'bpe-{size}.complete.json'
        if not trained.exists():
            log(f'Training SentencePiece BPE {size:,}')
            spm.SentencePieceTrainer.train(sentence_iterator=sample_sentences(output/'tokenizer/train_sample.jsonl'),
                model_prefix=str(prefix),model_type='bpe',vocab_size=size,
                character_coverage=1.,byte_fallback=True,normalization_rule_name='identity',
                remove_extra_whitespaces=False,unk_id=0,bos_id=1,eos_id=2,pad_id=3,
                user_defined_symbols=['<EMAIL>','<PHONE>','<ID>'],
                shuffle_input_sentence=False,num_threads=2,max_sentence_length=16384,
                hard_vocab_limit=False,minloglevel=1)
            atomic_json(trained,dict(sha256=file_sha(prefix.with_suffix('.model'))))
        model=spm.SentencePieceProcessor(model_file=str(prefix.with_suffix('.model')))
        encoded=model.encode(evaluation,num_threads=2)
        for text,ids in zip(evaluation,encoded):
            if model.decode(ids)!=text: raise RuntimeError('Tokenizer round-trip changed text')
            if model.unk_id() in ids: raise RuntimeError('Tokenizer emitted UNK despite byte fallback')
        total_tokens=sum(map(len,encoded)); chars=sum(map(len,evaluation)); whitespace=sum(len(t.split()) for t in evaluation)
        results.append(dict(requested_vocab=size,actual_vocab=model.get_piece_size(),tokens_per_character=total_tokens/chars,
            tokens_per_whitespace_token=total_tokens/whitespace,average_tokens_per_document=total_tokens/len(evaluation),
            probe_fragmentation={k:[dict(text=t,pieces=model.encode(t,out_type=str)) for t in texts] for k,texts in probes.items()}))
        log(f'BPE {size}: {total_tokens/chars:.5f} tokens/character')
    best=min(x['tokens_per_character'] for x in results)
    chosen=min(x['requested_vocab'] for x in results if x['tokens_per_character']<=best*1.03)
    shutil.copyfile(output/'tokenizer'/f'bpe-{chosen}.model',output/'tokenizer/tokenizer.model')
    shutil.copyfile(output/'tokenizer'/f'bpe-{chosen}.vocab',output/'tokenizer/tokenizer.vocab')
    model=spm.SentencePieceProcessor(model_file=str(output/'tokenizer/tokenizer.model'))
    result=dict(selected_vocab=chosen,actual_vocab=model.get_piece_size(),results=results,
        selection_rule='Smallest vocabulary within 3% of the lowest tokens/character on held-out train documents.',
        model='tokenizer/tokenizer.model',sha256=file_sha(output/'tokenizer/tokenizer.model'),
        eos_id=model.eos_id(),pad_id=model.pad_id(),unk_id=model.unk_id(),bos_id=model.bos_id(),
        normalization='identity; NFC performed upstream; whitespace and paragraph newlines preserved; byte fallback enabled')
    atomic_json(output/'reports/tokenizer_comparison.json',result); atomic_json(marker,result)
    return result

def tokenize_file(job):
    output,spec,number,context,tokenizer=job
    if file_sha(output/spec['path'])!=spec['sha256']: raise RuntimeError('Clean input checksum changed')
    model=spm.SentencePieceProcessor(model_file=str(output/tokenizer['model']))
    dtype=np.dtype('<u2' if model.get_piece_size()<=65536 else '<u4')
    name=f"{spec['split']}-{number:05d}"
    destination=output/'binary'/(name+'.bin'); temp=destination.with_suffix('.bin.tmp')
    boundary=output/'binary'/(name+'.documents.parquet'); btemp=boundary.with_suffix('.parquet.tmp')
    schema=pa.schema([('id',pa.string()),('start',pa.int64()),('end',pa.int64())])
    count=0; docs=0
    with temp.open('wb') as f, pq.ParquetFile(output/spec['path']) as p, pq.ParquetWriter(btemp,schema,compression='zstd') as writer:
        for batch in p.iter_batches(batch_size=128,columns=['id','text'],use_threads=False):
            rows=batch.to_pylist(); tokens=model.encode([r['text'] for r in rows],num_threads=2)
            boundaries=[]
            for row,ids in zip(rows,tokens):
                ids.append(model.eos_id())
                array=np.asarray(ids,dtype=dtype)
                if max(ids)>=model.get_piece_size() or min(ids)<0: raise RuntimeError('Invalid token ID')
                array.tofile(f)
                boundaries.append(dict(id=row['id'],start=count,end=count+len(ids)))
                count+=len(ids); docs+=1
            writer.write_table(pa.Table.from_pylist(boundaries,schema=schema))
        sequences=math.ceil(max(0,count-1)/context)
        padding=sequences*context+1-count
        if padding: np.full(padding,model.pad_id(),dtype=dtype).tofile(f)
    if not count or docs!=spec['rows']: raise RuntimeError('Tokenization count mismatch')
    temp.replace(destination); btemp.replace(boundary)
    result=dict(file=destination.relative_to(output).as_posix(),documents_file=boundary.relative_to(output).as_posix(),
        sha256=file_sha(destination),documents_sha256=file_sha(boundary),split=spec['split'],dtype=dtype.str,
        documents=docs,tokens=count,padding_tokens=padding,sequences=sequences,stored_tokens=count+padding,
        source_sha256=spec['sha256'],tokenizer_sha256=tokenizer['sha256'],context_length=context)
    atomic_json(output/'checkpoints'/(name+'.json'),result)
    return result

def serialize(output,clean,context,tokenizer):
    from prepare_pretrain import bounded_pool,log
    results=[]; jobs=[]
    for i,spec in enumerate(clean['files']):
        marker=output/'checkpoints'/f"{spec['split']}-{i:05d}.json"
        if marker.exists():
            result=json.loads(marker.read_text())
            if result['source_sha256']!=spec['sha256'] or result['tokenizer_sha256']!=tokenizer['sha256'] or result['context_length']!=context: raise RuntimeError('Tokenization config changed')
            if file_sha(output/result['file'])!=result['sha256'] or file_sha(output/result['documents_file'])!=result['documents_sha256']: raise RuntimeError('Binary checksum mismatch')
            results.append(result)
        else: jobs.append((output,spec,i,context,tokenizer))
    for result in bounded_pool(tokenize_file,jobs,2):
        results.append(result); log(f'Tokenization {len(results)}/{len(clean["files"])}: {result["tokens"]:,} tokens')
    result=dict(version='causal-mmap-v1',complete=False,context_length=context,seed=20260929,
        tokenizer=tokenizer,shards=sorted(results,key=lambda x:x['file']),
        packing='Contiguous document token streams with EOS after every document. Stride=context_length; target shifted once. Final shard padding masked from loss.',
        attention='Standard causal attention; cross-document attention allowed. Document offsets supplied for frameworks supporting document-aware masks.',
        total_documents=sum(r['documents'] for r in results),total_tokens=sum(r['tokens'] for r in results),
        total_padding_tokens=sum(r['padding_tokens'] for r in results),total_sequences=sum(r['sequences'] for r in results))
    atomic_json(output/'binary/manifest.json',result)
    return result

def verify(output,manifest,clean):
    from prepare_pretrain import database,log
    errors=[]
    with database(output) as db:
        paths=[str(output/f['path']) for f in clean['files']]
        overlaps=db.execute('SELECT count(*) FROM (SELECT duplicate_cluster FROM read_parquet(?) GROUP BY duplicate_cluster HAVING count(DISTINCT split)>1)',[paths]).fetchone()[0]
        if overlaps: errors.append('Detected duplicate clusters cross splits')
        if db.execute('SELECT count(*) FROM selected').fetchone()[0]!=manifest['total_documents']: errors.append('Selected/binary document counts differ')
        by_source=db.execute('SELECT source,split,count(*) AS documents,sum(clean_char_count) AS characters FROM read_parquet(?) GROUP BY source,split ORDER BY source,split',[paths]).fetchall()
        with pq.ParquetFile(output/clean['files'][0]['path']) as p: columns=set(p.schema_arrow.names)
        features=[name for name in ['clean_char_count','whitespace_token_count','paragraph_count','alphabetic_ratio','digit_ratio','special_char_ratio','url_ratio','duplicate_line_ratio','unique_token_ratio'] if name in columns]
        distributions={}
        for feature in features:
            values=db.execute(f'SELECT approx_quantile({feature},[0.01,0.05,0.5,0.95,0.99]) FROM read_parquet(?)',[paths]).fetchone()[0]
            distributions[feature]=dict(zip(['p1','p5','p50','p95','p99'],values))
        atomic_json(output/'reports/corpus_summary.json',dict(by_source=[dict(source=s,split=split,documents=count,characters=chars) for s,split,count,chars in by_source],
            approximate_percentiles=distributions))
    for spec in manifest['shards']:
        data=np.memmap(output/spec['file'],mode='r',dtype=spec['dtype'])
        if len(data)!=spec['stored_tokens'] or len(data)!=spec['sequences']*manifest['context_length']+1: errors.append('Packing length mismatch')
        for start in range(0,len(data),1000000):
            if int(data[start:start+1000000].max())>=manifest['tokenizer']['actual_vocab']: errors.append('Out-of-range token')
        last=0
        with pq.ParquetFile(output/spec['documents_file']) as p:
            for batch in p.iter_batches(batch_size=8192):
                rows=batch.to_pylist()
                for row in rows:
                    if row['start']!=last or int(data[row['end']-1])!=manifest['tokenizer']['eos_id']: errors.append('Document boundary/EOS mismatch')
                    last=row['end']
        if last!=spec['tokens']: errors.append('Boundary token count mismatch')
        if spec['padding_tokens'] and not np.all(data[-spec['padding_tokens']:]==manifest['tokenizer']['pad_id']): errors.append('Invalid padding')
        del data
    if errors: raise RuntimeError('; '.join(set(errors)))
    result=dict(passed=True,checks=['selected and serialized counts','detected cluster split isolation','all token ID ranges','all document boundaries and EOS','all shard lengths and padding'],
        documents=manifest['total_documents'],tokens=manifest['total_tokens'],sequences=manifest['total_sequences'])
    atomic_json(output/'reports/verification.json',result)
    manifest['complete']=True; atomic_json(output/'binary/manifest.json',manifest)
    log(f'PRETRAIN DATA READY: {result}')

def finish(output,context=4096):
    clean=export_clean(output)
    make_tokenizer_samples(output,clean)
    tokenizer=train_tokenizers(output)
    manifest=serialize(output,clean,context,tokenizer)
    verify(output,manifest,clean)
