# Embedding Inversion for Russian Dialogues

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

Download precomputed embeddings:

```bash
bash download_embds_data.sh
```

Or compute them yourself:
```bash
python create_embedding_dataset.py
```

## Run Baselines

### ALGEN

```bash
cd algen
./run_algen.sh
```

### zsinvert

```bash
cd zsinvert
./run_zsinvert.sh
```

## Run our Experiments

Soft Tokens Experiments with model/params grid search:
```bash
cd exps
./run_soft_tokens.sh
```

KV Prefix Experiments with model/params grid search:
```bash
cd exps
./run_kv_prefix.sh
```
