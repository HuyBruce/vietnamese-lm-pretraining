"""Export an unambiguous vocabulary view, including pieces containing newlines."""
import argparse
import json
from pathlib import Path
import sentencepiece as spm

def export(model_path):
    model=spm.SentencePieceProcessor(model_file=str(model_path))
    destination=model_path.with_suffix('.vocab.jsonl')
    temp=destination.with_suffix('.jsonl.tmp')
    with temp.open('w',encoding='utf-8',newline='\n') as f:
        for i in range(model.get_piece_size()):
            f.write(json.dumps(dict(id=i,piece=model.id_to_piece(i),score=model.get_score(i)),ensure_ascii=False)+'\n')
    temp.replace(destination)
    return destination

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('model',type=Path)
    a=p.parse_args(); print(export(a.model))
