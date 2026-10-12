"""Read-only sample used to calibrate the pretraining filters."""
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import fasttext
import argparse

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,default=Path('DataProcessed'))
p.add_argument('--output',type=Path,default=Path('DataProcessed/pretrain_ready'))
args=p.parse_args()
root = args.root
model = fasttext.load_model(str(args.output / 'assets/lid.176.bin'))
specs = json.loads((root / 'normalized/manifests/shards.json').read_text())
sample = []
for spec in specs[::20]:
    rows = pq.read_table(root / 'normalized' / spec['name'], columns=['id','source','text','normalization_flags']).to_pylist()[::100]
    for row in rows:
        text = row.pop('text')
        labels, scores = model.predict(text.replace('\n', ' ')[:20000])
        row.update(chars=len(text), tokens=len(text.split()), alpha=sum(c.isalpha() for c in text)/max(1,len(text)), language=labels[0][9:], score=min(1., scores[0]), preview=text[:300])
        sample.append(row)
report = {'sample_size': len(sample), 'percentiles': {k:dict(zip(['p1','p5','p50','p95','p99'], np.percentile([r[k] for r in sample],[1,5,50,95,99]).tolist())) for k in ['chars','tokens','alpha','score']}, 'non_vi':[r for r in sample if r['language']!='vi'], 'low_score':[r for r in sample if r['score']<.5][:30]}
(args.output / 'assets/profile.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(report, ensure_ascii=False, indent=2))
