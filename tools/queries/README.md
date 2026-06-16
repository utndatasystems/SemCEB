# Query Embeddings

Reads the benchmark queries and computes embedding vectors for each row’s `filter` field.

## Usage

```bash
python tools/queries/compute_embeddings.py
```

Recompute existing embeddings:

```bash
python tools/queries/compute_embeddings.py --overwrite
```

Note: The JSONL file is updated in place.