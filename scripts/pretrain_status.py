"""Print committed progress without loading the corpus or locking the pipeline."""
import argparse
from collections import Counter
import json
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,default=Path('DataProcessed/pretrain_ready'))
args=p.parse_args(); output=args.output
config=json.loads((output/'config.json').read_text())
specs=json.loads((Path(config['input'])/'manifests/shards.json').read_text())
counts=Counter(); committed=0
for path in (output/'checkpoints/filter').glob('*.json'):
    record=json.loads(path.read_text()); counts.update(record['counts']); committed+=1
result={'filter_shards':committed,'total_shards':len(specs),'filter_counts':dict(counts)}
for stage in ['exact','sketch_all','near','clean','tokenizer_sample','tokenizer']:
    marker=output/'checkpoints'/f'{stage}.json'
    result[stage+'_complete']=marker.exists()
binary=output/'binary/manifest.json'
result['ready_for_pretrain']=binary.exists() and json.loads(binary.read_text()).get('complete',False)
print(json.dumps(result,ensure_ascii=False,indent=2))
