"""PyTorch map-style dataset. `targets` are ALREADY shifted: do not shift twice."""
from bisect import bisect_right
from collections import OrderedDict
import json
from pathlib import Path
import numpy as np

class CausalTokenDataset:
    def __init__(self,root,split='train',max_open_shards=4):
        self.root=Path(root)
        self.manifest=json.loads((self.root/'binary/manifest.json').read_text())
        if not self.manifest.get('complete'): raise RuntimeError('Dataset verification is not complete')
        if split not in ('train','validation'): raise ValueError('Unknown split')
        if max_open_shards<1: raise ValueError('max_open_shards must be positive')
        self.shards=[s for s in self.manifest['shards'] if s['split']==split]
        self.ends=[]; total=0
        for spec in self.shards: total+=spec['sequences']; self.ends.append(total)
        self.context=self.manifest['context_length']; self.pad=self.manifest['tokenizer']['pad_id']
        self.max_open=max_open_shards; self._maps=OrderedDict()

    def __len__(self): return self.ends[-1] if self.ends else 0

    def __getstate__(self):
        state=self.__dict__.copy(); state['_maps']=OrderedDict(); return state

    def __getitem__(self,index):
        import torch
        if index<0: index+=len(self)
        if not 0<=index<len(self): raise IndexError(index)
        shard=bisect_right(self.ends,index)
        local=index-(self.ends[shard-1] if shard else 0)
        if shard not in self._maps:
            spec=self.shards[shard]
            self._maps[shard]=np.memmap(self.root/spec['file'],mode='r',dtype=spec['dtype'])
            while len(self._maps)>self.max_open:
                _,old=self._maps.popitem(last=False); old._mmap.close()
        self._maps.move_to_end(shard)
        start=local*self.context
        tokens=np.array(self._maps[shard][start:start+self.context+1],dtype=np.int64)
        inputs=torch.from_numpy(tokens[:-1].copy()); targets=torch.from_numpy(tokens[1:].copy())
        targets[targets==self.pad]=-100
        return {'input_ids':inputs,'targets':targets}
