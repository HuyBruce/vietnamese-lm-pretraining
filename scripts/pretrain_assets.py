"""Fetch public evaluation assets; never add these texts to tokenizer training."""
import io
import json
from pathlib import Path
import zipfile
import requests
import numpy as np
from pretrain_core import ngrams, word_bytes, sha_text
from prepare_news_inventory import atomic_json, file_sha

def download(url, target):
    if target.exists(): return
    target.parent.mkdir(parents=True,exist_ok=True)
    temp=target.with_suffix(target.suffix+'.tmp')
    with requests.get(url,stream=True,timeout=(20,180)) as r:
        r.raise_for_status()
        with temp.open('wb') as f:
            for chunk in r.iter_content(1024*1024): f.write(chunk)
    temp.replace(target)

def prepare(output, urls):
    assets=output/'assets'; assets.mkdir(parents=True,exist_ok=True)
    download('https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin', assets/'lid.176.bin')
    texts={}; sources=[]
    def collect(obj):
        if isinstance(obj,list):
            for x in obj: collect(x)
        elif isinstance(obj,dict):
            for key,value in obj.items():
                if key in ('question','context','passage','text','prompt') and isinstance(value,str):
                    if len(value.split())>=5: texts[sha_text(value)]=value
                elif isinstance(value,(dict,list)): collect(value)
    for i,url in enumerate(urls):
        suffix=Path(url.split('?')[0]).suffix
        target=assets/f'benchmark-{i}{suffix}'; download(url,target)
        if suffix=='.zip':
            with zipfile.ZipFile(target) as z:
                for name in z.namelist():
                    if name.endswith('.json'):
                        collect(json.loads(z.read(name).decode('utf-8-sig')))
                    elif name.endswith('.jsonl'):
                        for line in z.read(name).decode('utf-8-sig').splitlines():
                            if line.strip(): collect(json.loads(line))
        elif suffix=='.json': collect(json.loads(target.read_text(encoding='utf-8-sig')))
        elif suffix=='.jsonl':
            for line in target.read_text(encoding='utf-8-sig').splitlines():
                if line.strip(): collect(json.loads(line))
        else: raise ValueError(f'Unsupported benchmark format: {suffix}')
        sources.append(dict(url=url,sha256=file_sha(target),file=target.name))
    if not texts: raise RuntimeError('Benchmark extraction returned no text')
    all_grams=[ngrams(word_bytes(t),13) for t in texts.values()]
    grams=np.unique(np.concatenate(all_grams))
    np.save(assets/'benchmark_13grams.npy',grams)
    atomic_json(assets/'benchmark_exact.json',sorted({sha_text(' '.join(t.lower().split())) for t in texts.values()}))
    atomic_json(assets/'benchmarks.json',dict(sources=sources,texts=len(texts),ngrams=len(grams),ngram_words=13,
        method='Exact normalized document/line matches (minimum 5 whitespace tokens), or any shared 13-word n-gram; punctuation-insensitive Unicode word hashing.',
        limitations='Heuristic lexical overlap; no claim of semantic or universal benchmark decontamination.'))
    print(f'Benchmark index: {len(texts):,} texts, {len(grams):,} n-grams',flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,default=Path('DataProcessed/pretrain_ready')); p.add_argument('urls',nargs='+')
    a=p.parse_args(); prepare(a.output,a.urls)
