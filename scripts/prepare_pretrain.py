"""Resume preprocessing from committed normalized Parquet to a causal-LM dataset."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import os
import inspect
import re
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from prepare_news_inventory import atomic_json, file_sha, run_lock
from pretrain_core import (metrics, mask_pii, reject_reasons, dedup_text, sha_text,
    word_bytes, ngrams, minhash, intersects, union_bucket, compress_roots)

PROJECT=Path(__file__).resolve().parents[1]

def log(message): print(time.strftime('%H:%M:%S'), message, flush=True)

def save_parquet(path, rows, schema=None):
    temp=path.with_suffix('.parquet.tmp')
    pq.write_table(pa.Table.from_pylist(rows,schema=schema),temp,compression='zstd')
    temp.replace(path)

def bounded_pool(function, jobs, workers, initializer=None, initargs=()):
    with ProcessPoolExecutor(max_workers=workers,initializer=initializer,initargs=initargs) as pool:
        jobs=iter(jobs); active=set()
        for _ in range(workers):
            job=next(jobs,None)
            if job is not None: active.add(pool.submit(function,job))
        last=time.monotonic()
        while active:
            ready,_=wait(active,timeout=1,return_when=FIRST_COMPLETED)
            for future in ready:
                active.remove(future)
                yield future.result()
                job=next(jobs,None)
                if job is not None: active.add(pool.submit(function,job))
            if time.monotonic()-last>30:
                log('Workers are processing shards; checkpoints are retained.'); last=time.monotonic()

def init_filter(output):
    global LID, BENCH, EXACT_BENCH
    import fasttext
    pa.set_cpu_count(1)
    LID=fasttext.load_model(str(output/'assets/lid.176.bin'))
    BENCH=np.load(output/'assets/benchmark_13grams.npy')
    EXACT_BENCH=set(json.loads((output/'assets/benchmark_exact.json').read_text()))

def restore_meaningful_content(input_root,spec):
    """Revisit audited deletions using immutable inventory, preserving captions/sources."""
    import normalize_news as normalization
    markers=['explicit_caption','explicit_byline','html_figure','html_figcaption','html_picture','html_marked_noise']
    ids=set()
    with (input_root/'audit'/(spec['name']+'.jsonl')).open(encoding='utf-8') as audit:
        for line in audit:
            if any('"'+marker+'"' in line for marker in markers): ids.add(json.loads(line)['id'])
    if not ids: return {}
    source=input_root.parent/'inventory'/spec['name']
    manifest=json.loads((input_root.parent/'manifests/shards.json').read_text())
    expected=next(s['sha256'] for s in manifest if s['name']==spec['name'])
    if file_sha(source)!=expected: raise RuntimeError('Inventory changed while restoring captions')
    original_reason=normalization.noise_reason
    original_drop=normalization.DROP; original_classes=normalization.NOISE_CLASSES; original_block=normalization.BLOCK
    def preserve(line,index,total):
        reason=original_reason(line,index,total)
        if reason=='explicit_caption': return None
        if reason=='explicit_byline' and line.casefold().startswith(('nguồn:', 'nguồn：')): return None
        return reason
    normalization.noise_reason=preserve
    normalization.DROP=original_drop-{'figure','figcaption','picture'}
    normalization.BLOCK=original_block|{'figure','figcaption','picture'}
    normalization.NOISE_CLASSES=original_classes-{'figcaption','image-caption','photo-caption','photocms_caption'}
    restored={}
    try:
        with pq.ParquetFile(source) as p:
            for batch in p.iter_batches(batch_size=128,columns=['id','text'],use_threads=False):
                for row in batch.to_pylist():
                    if row['id'] in ids: restored[row['id']]=normalization.normalize_text(row['text'])
    finally:
        normalization.noise_reason=original_reason; normalization.DROP=original_drop
        normalization.NOISE_CLASSES=original_classes; normalization.BLOCK=original_block
    if len(restored)!=len(ids): raise RuntimeError('Missing inventory records for audited deletions')
    return restored

def language_id(text):
    labels,scores=LID.predict(text.replace('\n',' ')[:20000])
    language,score=labels[0][9:],min(1.,float(scores[0])); method='document'
    if language!='vi' or score<.5:
        lead=' '.join(text.splitlines()[:3])[:1200]
        if sum(c.isalpha() for c in lead)>=80:
            labels2,scores2=LID.predict(lead)
            if labels2[0]=='__label__vi' and scores2[0]>=.9:
                language,score,method='vi',min(1.,float(scores2[0])),'vietnamese_lead'
    return language,score,method

def filter_shard(job):
    input_root,output,spec,start=job
    name=spec['name']; src=input_root/name
    if file_sha(src)!=spec['sha256']: raise RuntimeError(f'Changed input: {src}')
    restored=restore_meaningful_content(input_root,spec)
    destination=output/'filtered'/name; rejected=output/'rejected'/(name+'.jsonl')
    temp=destination.with_suffix('.parquet.tmp'); rtemp=rejected.with_suffix('.jsonl.tmp')
    counts=Counter(); writer=None; ordinal=0
    extra=pa.schema([('gid',pa.int64()),('language',pa.string()),('language_score',pa.float64()),
        ('language_method',pa.string()),('clean_char_count',pa.int64()),('whitespace_token_count',pa.int64()),
        ('paragraph_count',pa.int64()),*[(x,pa.float64()) for x in ['alphabetic_ratio','digit_ratio','special_char_ratio','url_ratio','duplicate_line_ratio','duplicate_paragraph_ratio','unique_token_ratio']],
        ('pii_email',pa.int64()),('pii_phone',pa.int64()),('pii_identifier',pa.int64()),
        ('clean_sha256',pa.string()),('clean_byte_count',pa.int64()),('clean_line_count',pa.int64()),('upstream_text_restored',pa.bool_()),
        ('dedup_sha256',pa.string()),('benchmark_overlap',pa.bool_()),('quality_score',pa.float64())])
    with pq.ParquetFile(src) as parquet, rtemp.open('w',encoding='utf-8') as reject_file:
        schema=pa.schema(list(parquet.schema_arrow)+list(extra))
        with pq.ParquetWriter(temp,schema,compression='zstd') as writer:
            for batch in parquet.iter_batches(batch_size=128,use_threads=False):
                kept=[]
                for row in batch.to_pylist():
                    gid=start+ordinal; ordinal+=1; counts['input']+=1
                    text=row['text']; flags=row.get('normalization_flags',[])
                    changed=False
                    if row['id'] in restored:
                        value,audit=restored[row['id']]; changed=value!=text; text=value; flags=audit['flags']
                    counts['restored_content']+=int(changed)
                    features=metrics(text)
                    language,score,method=language_id(text)
                    reasons=reject_reasons(text,features,language,score,flags)
                    if reasons:
                        counts['rejected']+=1; counts.update(reasons)
                        reject_file.write(json.dumps(dict(id=row['id'],gid=gid,source_path=row['source_path'],reasons=reasons,
                            language=language,language_score=score,**features),ensure_ascii=False)+'\n')
                        continue
                    # Match before PII replacement so masking cannot conceal benchmark overlap.
                    overlap=sha_text(dedup_text(text)) in EXACT_BENCH
                    if not overlap:
                        overlap=any(len(line.split())>=5 and sha_text(dedup_text(line)) in EXACT_BENCH for line in text.splitlines())
                    if not overlap: overlap=bool(intersects(ngrams(word_bytes(text),13),BENCH))
                    text,pii=mask_pii(text)
                    features['clean_char_count']=len(text)
                    row.update(text=text,gid=gid,language=language,language_score=score,language_method=method,
                        **features,**{'pii_'+k:v for k,v in pii.items()},clean_sha256=sha_text(text),
                        clean_byte_count=len(text.encode('utf-8')),clean_line_count=len(text.splitlines()),upstream_text_restored=changed,
                        dedup_sha256=sha_text(dedup_text(text)),benchmark_overlap=overlap,
                        quality_score=score-features['duplicate_line_ratio']-features['url_ratio']-features['special_char_ratio'])
                    kept.append(row); counts['kept']+=1; counts['benchmark_overlap']+=int(overlap)
                    counts.update({'pii_'+k:v for k,v in pii.items()})
                if kept: writer.write_table(pa.Table.from_pylist(kept,schema=schema))
    if ordinal!=spec['rows']: raise RuntimeError('Input row count mismatch')
    temp.replace(destination); rtemp.replace(rejected)
    result=dict(name=name,rows=counts['kept'],sha256=file_sha(destination),reject_sha256=file_sha(rejected),counts=dict(counts))
    atomic_json(output/'checkpoints/filter'/f'{name}.json',result)
    return result

def filter_all(input_root,output,specs,workers,max_shards):
    jobs=[]; results=[]; start=0
    for spec in specs:
        checkpoint=output/'checkpoints/filter'/f"{spec['name']}.json"
        if checkpoint.exists():
            result=json.loads(checkpoint.read_text())
            if file_sha(output/'filtered'/spec['name'])!=result['sha256'] or file_sha(output/'rejected'/(spec['name']+'.jsonl'))!=result['reject_sha256']:
                raise RuntimeError('Committed filter output changed')
            results.append(result)
        else: jobs.append((input_root,output,spec,start))
        start+=spec['rows']
    if max_shards: jobs=jobs[:max_shards]
    log(f'Filter: {len(results)}/{len(specs)} committed, {len(jobs)} pending this run')
    for result in bounded_pool(filter_shard,jobs,workers,init_filter,(output,)):
        results.append(result)
        if len(results)%10==0 or len(results)==len(specs): log(f'Filter {len(results)}/{len(specs)}; latest {result["counts"]}')
    counts=Counter()
    for result in results: counts.update(result['counts'])
    complete=len(results)==len(specs)
    atomic_json(output/'reports/filter.json',dict(complete=complete,shards=len(results),counts=dict(counts)))
    return complete

def database(output):
    import duckdb
    db=duckdb.connect(str(output/'work.duckdb'))
    db.execute("SET memory_limit='2GB'"); db.execute('SET threads=2')
    db.execute('SET preserve_insertion_order=false')
    db.execute('SET temp_directory=?',[str(output/'scratch')])
    return db

def exact_dedup(output):
    marker=output/'checkpoints/exact.json'
    if marker.exists(): return
    with database(output) as db:
        # Quality, completeness, and stable ID determine the retained exact copy.
        db.execute('CREATE OR REPLACE TABLE documents AS SELECT gid,id,source,dedup_sha256,benchmark_overlap,quality_score,clean_char_count FROM read_parquet(?)',[str(output/'filtered/*.parquet')])
        db.execute('''CREATE OR REPLACE TABLE exact_map AS SELECT gid,
            first_value(gid) OVER w AS representative,
            bool_or(benchmark_overlap) OVER (PARTITION BY dedup_sha256) AS contaminated
            FROM documents WINDOW w AS (PARTITION BY dedup_sha256 ORDER BY quality_score DESC,clean_char_count DESC,id)''')
        db.execute('''CREATE OR REPLACE TABLE representatives AS SELECT row_number() OVER (ORDER BY d.gid)-1 AS idx,
            d.*,e.contaminated FROM documents d JOIN exact_map e ON d.gid=e.gid WHERE e.gid=e.representative''')
        count=db.execute('SELECT count(*) FROM representatives').fetchone()[0]
        total=db.execute('SELECT count(*) FROM documents').fetchone()[0]
        db.execute('COPY (SELECT * FROM representatives ORDER BY idx) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)',[str(output/'representatives.parquet')])
        db.execute('CHECKPOINT')
    result=dict(input=total,retained=count,duplicates=total-count)
    atomic_json(marker,result); atomic_json(output/'reports/exact_dedup.json',result); log(f'Exact dedup: {result}')

def init_sketch(output):
    global GID_INDEX
    pa.set_cpu_count(1)
    GID_INDEX=np.load(output/'sketch/gid_index.npy',mmap_mode='r')

def sketch_shard(job):
    output,name=job; indices=[]; signatures=[]; offsets=[0]
    target=output/'sketch'/(name+'.bin'); temp=target.with_suffix('.bin.tmp')
    with temp.open('wb') as f, pq.ParquetFile(output/'filtered'/name) as p:
        for batch in p.iter_batches(batch_size=128,columns=['gid','text'],use_threads=False):
            for row in batch.to_pylist():
                idx=GID_INDEX[row['gid']]
                if idx<0: continue
                shingles=ngrams(word_bytes(row['text']),5)
                if not len(shingles): raise RuntimeError('Empty shingle set after quality filtering')
                shingles.tofile(f); offsets.append(offsets[-1]+len(shingles))
                indices.append(idx); signatures.append(minhash(shingles))
    temp.replace(target)
    np.savez(output/'sketch'/(name+'.npz'),indices=np.asarray(indices,dtype=np.int64),
        signatures=np.asarray(signatures,dtype=np.uint32).reshape(-1,64),offsets=np.asarray(offsets,dtype=np.int64))
    result=dict(name=name,rows=len(indices),sha256=file_sha(target),meta_sha256=file_sha(output/'sketch'/(name+'.npz')))
    atomic_json(output/'checkpoints/sketch'/(name+'.json'),result)
    return result

def sketch_all(output,specs,workers):
    marker=output/'checkpoints/sketch_all.json'
    if marker.exists():
        saved=json.loads(marker.read_text())
        for name,digest in saved['sha256'].items():
            if file_sha(output/'sketch'/name)!=digest: raise RuntimeError('Consolidated sketch checksum mismatch')
        return
    reps=pq.read_table(output/'representatives.parquet',columns=['gid','idx']).to_pydict()
    count=len(reps['gid']); total=sum(s['rows'] for s in specs)
    mapping=np.full(total,-1,dtype=np.int64); mapping[reps['gid']]=reps['idx']
    np.save(output/'sketch/gid_index.npy',mapping); del mapping,reps
    jobs=[]
    for spec in specs:
        name=spec['name']; shard_marker=output/'checkpoints/sketch'/(name+'.json')
        if shard_marker.exists():
            data=json.loads(shard_marker.read_text())
            if file_sha(output/'sketch'/(name+'.bin'))!=data['sha256'] or file_sha(output/'sketch'/(name+'.npz'))!=data['meta_sha256']: raise RuntimeError('Sketch checkpoint mismatch')
        else: jobs.append((output,name))
    done=len(specs)-len(jobs)
    for result in bounded_pool(sketch_shard,jobs,workers,init_sketch,(output,)):
        done+=1
        if done%10==0: log(f'MinHash {done}/{len(specs)}')
    sig=np.lib.format.open_memmap(output/'sketch/signatures.npy',mode='w+',dtype=np.uint32,shape=(count,64))
    offsets=np.zeros(count+1,dtype=np.int64); position=0; next_idx=0
    with (output/'sketch/shingles.bin').open('wb') as f:
        for spec in specs:
            name=spec['name']
            with np.load(output/'sketch'/(name+'.npz')) as data:
                ids=data['indices']; local=data['offsets']
                if len(ids) and not np.array_equal(ids,np.arange(next_idx,next_idx+len(ids))): raise RuntimeError('Non-contiguous sketch order')
                sig[ids]=data['signatures']; offsets[ids]=local[:-1]+position
                position+=int(local[-1]); next_idx+=len(ids)
            with (output/'sketch'/(name+'.bin')).open('rb') as src: shutil.copyfileobj(src,f,1024*1024)
    if next_idx!=count: raise RuntimeError('Missing sketches')
    offsets[-1]=position; np.save(output/'sketch/offsets.npy',offsets); sig.flush(); del sig
    atomic_json(marker,dict(documents=count,shingles=position,sha256={name:file_sha(output/'sketch'/name) for name in ['signatures.npy','offsets.npy','shingles.bin']}))
    # These are generated intermediates, now represented by the committed combined
    # files. Delete individual files only, retaining manifests and source data.
    for spec in specs:
        for suffix in ('.bin','.npz'):
            (output/'sketch'/(spec['name']+suffix)).unlink()

def near_dedup(output):
    marker=output/'checkpoints/near.json'
    if marker.exists(): return
    sig=np.load(output/'sketch/signatures.npy',mmap_mode='r'); n=len(sig)
    offsets=np.load(output/'sketch/offsets.npy',mmap_mode='r')
    shingles=np.memmap(output/'sketch/shingles.bin',mode='r',dtype=np.uint64)
    parents=np.arange(n,dtype=np.int64); comparisons=0; merged=0
    # Sort each of 16 bands in RAM (~100 MB per band), verify actual 5-gram
    # Jaccard >= .8 before union. No bucket truncation or candidate dropping.
    for band in range(16):
        columns=[sig[:,band*4+j] for j in range(4)]
        order=np.lexsort(columns)
        sorted_sig=np.asarray(sig[order,band*4:band*4+4])
        changes=np.flatnonzero(np.any(sorted_sig[1:]!=sorted_sig[:-1],axis=1))+1
        boundaries=np.concatenate(([0],changes,[n]))
        for left,right in zip(boundaries[:-1],boundaries[1:]):
            if right-left<2: continue
            checked,joins=union_bucket(order[left:right],parents,offsets,shingles,.8)
            comparisons+=checked; merged+=joins
        log(f'LSH band {band+1}/16: comparisons={comparisons:,}, unions={merged:,}')
    compress_roots(parents)
    clusters=pa.table({'idx':np.arange(n,dtype=np.int64),'cluster':parents})
    with database(output) as db:
        db.register('cluster_input',clusters)
        db.execute('CREATE OR REPLACE TABLE clusters AS SELECT * FROM cluster_input')
        db.execute('''CREATE OR REPLACE TABLE winners AS SELECT * EXCLUDE (rank) FROM (
            SELECT r.*,c.cluster,bool_or(r.contaminated) OVER (PARTITION BY c.cluster) AS cluster_contaminated,
            row_number() OVER (PARTITION BY c.cluster ORDER BY r.quality_score DESC,r.clean_char_count DESC,r.id) AS rank
            FROM representatives r JOIN clusters c USING(idx)) WHERE rank=1''')
        db.execute('''CREATE OR REPLACE TABLE selected AS SELECT *,
            CASE WHEN ('0x'||substr(sha256('20260929:'||id),1,8))::UBIGINT % 10000 < 50
                 THEN 'validation' ELSE 'train' END AS split,
            sha256('shuffle:20260929:'||id) AS shuffle_key
            FROM winners WHERE NOT cluster_contaminated''')
        db.execute('COPY (SELECT gid,idx,id,cluster,split,shuffle_key FROM selected) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)',[str(output/'selected.parquet')])
        db.execute('''COPY (SELECT d.id,d.gid,e.representative,c.cluster,
            CASE WHEN w.cluster_contaminated THEN 'benchmark_overlap_cluster'
                 WHEN d.gid!=e.representative THEN 'exact_duplicate'
                 WHEN d.gid!=w.gid THEN 'near_duplicate' ELSE 'keep' END AS decision,
            CASE WHEN w.cluster_contaminated THEN NULL ELSE w.id END AS retained_id FROM documents d JOIN exact_map e USING(gid)
            JOIN representatives r ON r.gid=e.representative JOIN clusters c USING(idx)
            JOIN winners w ON w.cluster=c.cluster) TO ? (FORMAT PARQUET,COMPRESSION ZSTD)''',[str(output/'decisions.parquet')])
        counts=dict(db.execute('SELECT split,count(*) FROM selected GROUP BY split').fetchall())
        contaminated=db.execute('SELECT count(*) FROM winners WHERE cluster_contaminated').fetchone()[0]
        db.execute('CHECKPOINT')
    result=dict(documents=n,comparisons=comparisons,near_duplicates=merged,contaminated_clusters=contaminated,splits=counts,
        minhash_permutations=64,lsh_bands=16,rows_per_band=4,jaccard_threshold=.8,
        limitations='LSH is probabilistic: only retrieved candidate pairs are verified. All detected duplicate components stay together.')
    atomic_json(marker,result); atomic_json(output/'reports/near_dedup.json',result)

def setup(input_root,output):
    for folder in ['assets','filtered','rejected','reports','checkpoints/filter','checkpoints/sketch','scratch','sketch','clean','tokenizer','binary']:
        (output/folder).mkdir(parents=True,exist_ok=True)
    report=json.loads((input_root/'reports/normalization_report.json').read_text())
    if not report['complete']: raise RuntimeError('Normalization not complete')
    specs=json.loads((input_root/'manifests/shards.json').read_text())
    if any(not re.fullmatch(r'part-\d+\.parquet',s['name']) for s in specs): raise RuntimeError('Unsafe shard name')
    if sum(s['rows'] for s in specs)!=report['stats']['rows']: raise RuntimeError('Normalization manifest count mismatch')
    config=dict(version='pretrain-v1',input=str(input_root),snapshot_sha256=file_sha(input_root/'manifests/shards.json'),
        core_sha256=file_sha(Path(__file__).with_name('pretrain_core.py')),
        filter_sha256=sha_text(inspect.getsource(filter_shard)+inspect.getsource(language_id)+inspect.getsource(restore_meaningful_content)),
        benchmark_sha256=file_sha(output/'assets/benchmark_13grams.npy'),
        benchmark_exact_sha256=file_sha(output/'assets/benchmark_exact.json'),
        lid_sha256=file_sha(output/'assets/lid.176.bin'),seed=20260929,context_length=4096,validation_fraction=.005)
    target=output/'config.json'
    if target.exists() and json.loads(target.read_text())!=config: raise RuntimeError('Configuration changed; choose a fresh output')
    atomic_json(target,config)
    return specs

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=PROJECT/'DataProcessed/normalized')
    p.add_argument('--output',type=Path,default=PROJECT/'DataProcessed/pretrain_ready')
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--stage',choices=['filter','exact','sketch','near','finish','all'],default='all')
    p.add_argument('--max-shards',type=int,default=0)
    a=p.parse_args(); a.input=a.input.resolve(); a.output=a.output.resolve()
    if a.output==a.input or a.input.is_relative_to(a.output) or a.output.is_relative_to(a.input): raise RuntimeError('Output overlaps input')
    if not 1<=a.workers<=8 or a.max_shards<0: raise ValueError('Invalid worker/shard limit')
    a.output.mkdir(parents=True,exist_ok=True)
    with run_lock(a.output):
        specs=setup(a.input,a.output)
        if a.stage in ('all','filter'):
            if not filter_all(a.input,a.output,specs,a.workers,a.max_shards): return
        if a.stage!='filter':
            if not json.loads((a.output/'reports/filter.json').read_text()).get('complete'): raise RuntimeError('Filtering must finish before downstream stages')
            downstream={name:file_sha(Path(__file__).with_name(name)) for name in ['prepare_pretrain.py','pretrain_core.py','pretrain_finish.py','pretrain_dataset.py']}
            guard=a.output/'checkpoints/downstream_code.json'
            if guard.exists() and json.loads(guard.read_text())!=downstream:
                raise RuntimeError('Downstream code changed since the checkpoint; use a fresh output')
            atomic_json(guard,downstream)
        if a.stage in ('all','exact'): exact_dedup(a.output)
        if a.stage in ('all','sketch'): sketch_all(a.output,specs,a.workers)
        if a.stage in ('all','near'): near_dedup(a.output)
        if a.stage in ('all','finish'):
            from pretrain_finish import finish
            finish(a.output,4096)

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8'); sys.stderr.reconfigure(encoding='utf-8')
    main()
