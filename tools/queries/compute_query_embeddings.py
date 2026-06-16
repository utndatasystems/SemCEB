#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path


DEFAULT_JSONL_PATH = Path("benchmark_queries/queries.jsonl")
EMBEDDING_MODELS = [
    "google/siglip2-base-patch16-224",
    "Qwen/Qwen3-Embedding-0.6B",
]
FILTER_KEY = "filter"
BATCH_SIZE = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute embeddings for the `filter` field in benchmark_queries/queries.jsonl "
            "and replace the JSONL file in-place."
        )
    )
    parser.add_argument(
        "jsonl_path",
        nargs="?",
        default=str(DEFAULT_JSONL_PATH),
        help=f"Path to JSONL file. Default: {DEFAULT_JSONL_PATH}",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute embeddings even if they already exist.",
    )
    return parser.parse_args()


def resolve_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def last_token_pool(last_hidden_states, attention_mask, torch_module):
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]

    if left_padding:
        return last_hidden_states[:, -1]

    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]

    return last_hidden_states[
        torch_module.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


def load_text_model_bundle(model_name: str, device: str):
    import torch
    from transformers import AutoModel, AutoTokenizer

    if model_name == "Qwen/Qwen3-Embedding-0.6B":
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to(device)

    return tokenizer, model, torch



def get_safe_max_length(tokenizer, model) -> int | None:
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)

    config_limit = None
    if hasattr(model.config, "text_config"):
        config_limit = getattr(model.config.text_config, "max_position_embeddings", None)
    if config_limit is None:
        config_limit = getattr(model.config, "max_position_embeddings", None)

    limits = [
        limit
        for limit in [tokenizer_limit, config_limit]
        if isinstance(limit, int) and limit < 1_000_000_000
    ]

    if not limits:
        return None

    return min(limits)


def compute_batch_embeddings(
    tokenizer,
    model,
    torch_module,
    device: str,
    texts: list[str],
) -> list[list[float]]:
    if hasattr(model, "get_text_features"):
        batch = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
    else:
        batch = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=32768,
            return_tensors="pt",
        )

    batch = {key: value.to(device) for key, value in batch.items()}

    with torch_module.inference_mode():
        if hasattr(model, "get_text_features"):
            embeddings = model.get_text_features(**batch)
        else:
            outputs = model(**batch)
            embeddings = last_token_pool(
                outputs.last_hidden_state,
                batch["attention_mask"],
                torch_module,
            )

        embeddings = torch_module.nn.functional.normalize(embeddings, p=2, dim=1)

    return embeddings.detach().cpu().to(torch_module.float32).tolist()

def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc

            if not isinstance(obj, dict):
                raise ValueError(f"Line {line_number} is not a JSON object")

            rows.append(obj)

    return rows


def write_jsonl_in_place(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)

        for obj in rows:
            tmp.write(json.dumps(obj, ensure_ascii=False) + "\n")

    os.replace(tmp_path, path)


def add_embeddings_for_model(
    rows: list[dict],
    model_name: str,
    device: str,
    overwrite: bool,
) -> None:
    tokenizer, model, torch_module = load_text_model_bundle(model_name, device)

    jobs: list[tuple[int, str]] = []
    skipped_missing_filter = 0
    skipped_existing = 0

    for idx, obj in enumerate(rows):
        if not overwrite and model_name in obj:
            skipped_existing += 1
            continue

        raw_filter = obj.get(FILTER_KEY)

        if raw_filter is None:
            skipped_missing_filter += 1
            continue

        filter_text = normalize_whitespace(str(raw_filter))

        if not filter_text:
            skipped_missing_filter += 1
            continue

        jobs.append((idx, filter_text))

    if overwrite:
        print(
            f"[info] {model_name}: prepared {len(jobs)} rows "
            f"(overwrite=True, {skipped_missing_filter} missing/empty filter)"
        )
    else:
        print(
            f"[info] {model_name}: prepared {len(jobs)} rows "
            f"({skipped_existing} skipped because embeddings already existed, "
            f"{skipped_missing_filter} missing/empty filter)"
        )

    embedded = 0

    for batch_start in range(0, len(jobs), BATCH_SIZE):
        batch_jobs = jobs[batch_start : batch_start + BATCH_SIZE]

        batch_indices = [idx for idx, _ in batch_jobs]
        batch_texts = [text for _, text in batch_jobs]

        vectors = compute_batch_embeddings(
            tokenizer=tokenizer,
            model=model,
            torch_module=torch_module,
            device=device,
            texts=batch_texts,
        )

        for idx, vector in zip(batch_indices, vectors, strict=True):
            rows[idx][model_name] = vector

        embedded += len(batch_jobs)
        print(f"[info] {model_name}: embedded {embedded}/{len(jobs)} rows")


def main() -> int:
    args = parse_args()

    jsonl_path = Path(args.jsonl_path).expanduser().resolve()

    if not jsonl_path.exists():
        print(f"[error] JSONL file does not exist: {jsonl_path}")
        return 1

    device = resolve_device()

    print(f"[info] JSONL path: {jsonl_path}")
    print(f"[info] device: {device}")
    print(f"[info] overwrite: {args.overwrite}")

    try:
        rows = read_jsonl(jsonl_path)

        for model_name in EMBEDDING_MODELS:
            print(f"[info] loading model: {model_name}")
            add_embeddings_for_model(
                rows=rows,
                model_name=model_name,
                device=device,
                overwrite=args.overwrite,
            )

        write_jsonl_in_place(jsonl_path, rows)

    except Exception as exc:
        print(f"[error] {exc}")
        return 1

    print(f"[ok] replaced JSONL file in-place: {jsonl_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())