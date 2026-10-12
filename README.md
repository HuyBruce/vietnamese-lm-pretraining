# Vietnamese Language Model Pretraining

A source-only portfolio snapshot of the **AgentContentVietnamese team project**, published by contributor **Pham Gia Huy**. Original project remote: https://github.com/anhnam26/AgentContentVietnamese . Original Git history records anhnam26 and HuyTramKam among contributors. This repository preserves team attribution and does not represent sole authorship.

## Recorded run

- 109,529,856 parameters; decoder-only Transformer with RoPE, RMSNorm, SwiGLU and tied embeddings.
- 12 layers, hidden size 768, 12 attention heads, context 4096, SentencePiece BPE vocabulary 32,000.
- 14,411 completed optimizer steps over 1,888,682,419 training tokens; epoch 1, final checkpoint saved according to the source run logs.
- Best observed full-validation loss: 2.3403474210523463. This is corpus validation loss, not chatbot accuracy or instruction-following quality.
- Team dataset snapshot reports 2,971,948 cleaned Vietnamese news articles.

A compact copy of the recorded run metadata is in [results/training_summary.json](results/training_summary.json). Training was not rerun for publication.

## Pipeline

Inventory and normalization -> language/quality filters -> PII masking -> exact and MinHash near deduplication -> benchmark overlap checks -> cluster-aware train/validation split -> binary token shards -> resumable PyTorch training. Regex masking and overlap checks are partial safeguards, not guarantees.

## Layout

- `scripts/prepare_news_inventory.py`, `normalize_news.py`: source inventory and normalization.
- `scripts/prepare_pretrain.py`: filtering, deduplication and shard preparation.
- `scripts/pretrain_model.py`: Transformer implementation.
- `scripts/train_pretrain.py`: training entry point, with checkpoint, validation and logging helpers.
- `configs/pretrain_110m.json`: recorded model/training configuration.
- `scripts/test_*.py`: unit tests.

## Setup and verification

Python 3.13 was used for the recorded pipeline. Install PyTorch appropriate to your CPU/GPU separately, then:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
python -m pip install -r scripts/requirements-pretrain.txt
python -B -m unittest discover -s scripts -p 'test_*.py' -v
```

**76 tests passed** in the original project environment during publication preparation with UTF-8 enabled. Review the detailed guides under `scripts/` before preparing data or training.

## Data and model artifacts

No news corpus, raw audit records, model weights, local credentials, or handoff archives are included. You must provide data you have rights to use. The original README references a dataset at https://huggingface.co/datasets/anhnam26/AgentContentVietnamese-data ; its availability and content rights must be checked separately. This source snapshot does not grant a license for third-party articles or claim that the private/local dataset is fully redistributable.

## Limitations

This is a base language-model training pipeline, not an instruction-tuned assistant. Source and run metadata demonstrate the experiment; generation quality and downstream performance require independent evaluation. Windows console encoding must be UTF-8 for Vietnamese text.
