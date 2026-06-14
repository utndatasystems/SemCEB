#!/usr/bin/env python3
"""Compute image and text embeddings for Amazon Reviews products and reviews.

This script reads a processed dataset directory created by ``download_and_prepare_amazon_reviews_dataset.py``,
computes embeddings for selected columns in the filtered product and review
tables, and exports new parquet files without modifying the original artifacts.
"""

from __future__ import annotations

import argparse
import gc
import json
import pickle
import re
import sqlite3
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import duckdb
from PIL import Image, UnidentifiedImageError


SIGLIP_MODEL_NAME = "google/siglip2-base-patch16-224"
QWEN_TEXT_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
QWEN_TEXT_MAX_LENGTH = 32768
SIGLIP_TEXT_MAX_LENGTH = 64
DEFAULT_BATCH_SIZE = 32
DEFAULT_DATA_DIR = "Arts_Crafts_and_Sewing__raw_5core"
CACHE_DB_FILENAME = "embedding_cache.sqlite3"
SCRIPT_DIR = Path(__file__).resolve().parent
TextBatchEmbeddingFn = Callable[
    [object, object, object, str, list[str]],
    tuple[list[list[float]], list[bool]],
]
TEXT_EMBEDDING_COLUMNS = [
    "product_title",
    "description_json",
    "features_json",
    "details_json",
]
REVIEW_TEXT_EMBEDDING_COLUMNS = [
    "review_title",
    "review_text",
]


@dataclass(frozen=True)
class ImageEmbeddingJob:
    parent_asin: str
    main_image_url: str
    image_path: Path


@dataclass(frozen=True)
class TextEmbeddingModelBundle:
    model_name: str
    tokenizer: object
    model: object
    torch_module: object
    compute_batch: TextBatchEmbeddingFn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute embeddings for filtered product and review tables"
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help=(
            "Processed dataset directory created by download_and_prepare_amazon_reviews_dataset.py. "
            "Short names are resolved relative to the processed Amazon Reviews data directory. "
            f"Default: {DEFAULT_DATA_DIR}"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Embedding batch size. Default: {DEFAULT_BATCH_SIZE}",
    )
    return parser.parse_args()


def resolve_data_dir(data_dir_arg: str) -> Path:
    data_dir = Path(data_dir_arg)

    if data_dir.is_absolute():
        return data_dir

    package_root = SCRIPT_DIR.parents[1]
    processed_root = package_root / "data" / "amazon-reviews" / "processed"

    if len(data_dir.parts) == 1:
        return (processed_root / data_dir).resolve()

    return (package_root / data_dir).resolve()


def sql_string_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def sanitize_model_name(model_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")


def embedding_column_name(reference_column: str, model_name: str) -> str:
    return f"{reference_column}_embeddings_{sanitize_model_name(model_name)}"


def input_is_truncated_column_name(embedding_column: str) -> str:
    return f"{embedding_column}_input_is_truncated"


def output_products_parquet_path(data_dir: Path) -> Path:
    return data_dir / "products_filtered_with_embeddings.parquet"


def output_reviews_parquet_path(data_dir: Path) -> Path:
    return data_dir / "reviews_filtered_with_embeddings.parquet"


def image_path_from_url(images_dir: Path, url: str) -> Path | None:
    parsed = urlsplit(url)
    filename = Path(parsed.path).name
    if not filename:
        return None

    host = parsed.netloc or "unknown_host"
    rel_dir = Path(parsed.path.lstrip("/")).parent
    return images_dir / host / rel_dir / filename


def resolve_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_siglip_model_bundle(device: str) -> tuple[object, object, int, object]:
    import torch
    from transformers import AutoConfig, AutoImageProcessor, AutoModel

    config = AutoConfig.from_pretrained(SIGLIP_MODEL_NAME)
    dimension = getattr(getattr(config, "text_config", None), "projection_size", None)
    if dimension is None:
        dimension = getattr(getattr(config, "vision_config", None), "hidden_size", None)
    if dimension is None:
        raise RuntimeError(f"Could not determine embedding dimension for model {SIGLIP_MODEL_NAME}")

    processor = AutoImageProcessor.from_pretrained(SIGLIP_MODEL_NAME)
    model = AutoModel.from_pretrained(SIGLIP_MODEL_NAME)
    model.eval()
    model.to(device)
    return processor, model, int(dimension), torch


def load_qwen_text_model_bundle(device: str) -> tuple[object, object, object]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(QWEN_TEXT_MODEL_NAME, padding_side="left")
    model = AutoModel.from_pretrained(QWEN_TEXT_MODEL_NAME)
    model.eval()
    model.to(device)
    return tokenizer, model, torch


def load_siglip_text_tokenizer() -> object:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(SIGLIP_MODEL_NAME)


def validate_data_dir(data_dir: Path) -> tuple[Path, Path]:
    db_path = data_dir / "amazon_reviews.duckdb"
    images_dir = data_dir / "images"

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    if not db_path.exists():
        raise FileNotFoundError(f"DuckDB file not found: {db_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    return db_path, images_dir


def embedding_cache_path(data_dir: Path) -> Path:
    return data_dir / CACHE_DB_FILENAME


def open_embedding_cache(data_dir: Path) -> sqlite3.Connection:
    cache_conn = sqlite3.connect(str(embedding_cache_path(data_dir)))
    cache_conn.execute("PRAGMA journal_mode=WAL")
    cache_conn.execute("PRAGMA synchronous=NORMAL")
    cache_conn.execute("PRAGMA temp_store=MEMORY")
    cache_conn.execute(
        """
        CREATE TABLE IF NOT EXISTS embedding_cache (
            cache_key TEXT PRIMARY KEY,
            table_name TEXT NOT NULL,
            key_column_name TEXT NOT NULL,
            key_value TEXT NOT NULL,
            embedding_column_name TEXT NOT NULL,
            embedding BLOB NOT NULL,
            input_is_truncated INTEGER NOT NULL
        )
        """
    )
    cache_conn.execute(
        "CREATE INDEX IF NOT EXISTS embedding_cache_column_idx "
        "ON embedding_cache(embedding_column_name)"
    )
    return cache_conn


def cache_key_for_embedding(
    table_name: str,
    embedding_column_name_value: str,
    key_value: object,
) -> str:
    return f"{table_name}|{embedding_column_name_value}|{key_value}"


def fetch_cached_embeddings(
    cache_conn: sqlite3.Connection,
    cache_keys: list[str],
) -> dict[str, tuple[list[float], bool]]:
    if not cache_keys:
        return {}

    cached: dict[str, tuple[list[float], bool]] = {}
    for start in range(0, len(cache_keys), 900):
        chunk = cache_keys[start : start + 900]
        placeholders = ", ".join("?" for _ in chunk)
        rows = cache_conn.execute(
            f"""
            SELECT cache_key, embedding, input_is_truncated
            FROM embedding_cache
            WHERE cache_key IN ({placeholders})
            """,
            chunk,
        ).fetchall()
        for cache_key, embedding_blob, input_is_truncated in rows:
            cached[cache_key] = (pickle.loads(embedding_blob), bool(input_is_truncated))

    return cached


def store_cached_embeddings(
    cache_conn: sqlite3.Connection,
    rows: list[tuple[str, str, str, str, str, list[float], bool]],
) -> None:
    if not rows:
        return

    payload_rows = [
        (
            cache_key,
            table_name,
            key_column_name,
            key_value,
            embedding_column_name_value,
            sqlite3.Binary(pickle.dumps(embedding, protocol=pickle.HIGHEST_PROTOCOL)),
            int(input_is_truncated),
        )
        for (
            cache_key,
            table_name,
            key_column_name,
            key_value,
            embedding_column_name_value,
            embedding,
            input_is_truncated,
        ) in rows
    ]
    cache_conn.executemany(
        """
        INSERT INTO embedding_cache (
            cache_key,
            table_name,
            key_column_name,
            key_value,
            embedding_column_name,
            embedding,
            input_is_truncated
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(cache_key) DO UPDATE SET
            table_name = excluded.table_name,
            key_column_name = excluded.key_column_name,
            key_value = excluded.key_value,
            embedding_column_name = excluded.embedding_column_name,
            embedding = excluded.embedding,
            input_is_truncated = excluded.input_is_truncated
        """,
        payload_rows,
    )
    cache_conn.commit()


def add_embedding_columns(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    embedding_column_name_value: str,
    embedding_type: str,
) -> None:
    con.execute(
        f"ALTER TABLE {table_name} "
        f"ADD COLUMN {embedding_column_name_value} {embedding_type}"
    )
    con.execute(
        f"ALTER TABLE {table_name} "
        f"ADD COLUMN {input_is_truncated_column_name(embedding_column_name_value)} BOOLEAN"
    )


def ensure_products_filtered(con: duckdb.DuckDBPyConnection) -> None:
    exists = con.execute(
        """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = 'main'
              AND table_name = 'products_filtered'
        )
        """
    ).fetchone()[0]
    if not exists:
        raise RuntimeError("products_filtered table not found in DuckDB database")


def ensure_reviews_filtered(con: duckdb.DuckDBPyConnection) -> None:
    exists = con.execute(
        """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = 'main'
              AND table_name = 'reviews_filtered'
        )
        """
    ).fetchone()[0]
    if not exists:
        raise RuntimeError("reviews_filtered table not found in DuckDB database")


def create_working_table(
    con: duckdb.DuckDBPyConnection,
    image_column_name: str,
    image_dimension: int,
    text_model_names: list[str],
) -> None:
    con.execute("CREATE OR REPLACE TEMP TABLE products_with_embeddings AS SELECT * FROM products_filtered")
    add_embedding_columns(
        con,
        "products_with_embeddings",
        image_column_name,
        f"DOUBLE[{image_dimension}]",
    )
    for model_name in text_model_names:
        for source_column in TEXT_EMBEDDING_COLUMNS:
            add_embedding_columns(
                con,
                "products_with_embeddings",
                embedding_column_name(source_column, model_name),
                "DOUBLE[]",
            )


def create_reviews_working_table(
    con: duckdb.DuckDBPyConnection,
    text_model_names: list[str],
) -> None:
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE reviews_with_embeddings AS
        SELECT
            row_number() OVER () AS embedding_row_id,
            *
        FROM reviews_filtered
        """
    )
    for model_name in text_model_names:
        for source_column in REVIEW_TEXT_EMBEDDING_COLUMNS:
            add_embedding_columns(
                con,
                "reviews_with_embeddings",
                embedding_column_name(source_column, model_name),
                "DOUBLE[]",
            )


def create_image_embedding_job_table(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE image_embedding_jobs AS
        SELECT
            p.parent_asin,
            (
                SELECT NULLIF(trim(json_extract_string(e.value, '$.hi_res')), '')
                FROM json_each(p.images_json) AS e
                WHERE json_extract_string(e.value, '$.variant') = 'MAIN'
                  AND NULLIF(trim(json_extract_string(e.value, '$.hi_res')), '') IS NOT NULL
                LIMIT 1
            ) AS main_image_url
        FROM products_filtered p
        WHERE p.main_image_local IS NOT NULL
        ORDER BY p.parent_asin
        """
    )


def collect_image_jobs(
    con: duckdb.DuckDBPyConnection,
    images_dir: Path,
) -> tuple[list[ImageEmbeddingJob], int, int]:
    rows = con.execute(
        """
        SELECT parent_asin, main_image_url
        FROM image_embedding_jobs
        """
    ).fetchall()

    jobs: list[ImageEmbeddingJob] = []
    missing_url = 0
    missing_file = 0

    for parent_asin, main_image_url in rows:
        if main_image_url is None:
            missing_url += 1
            continue

        image_path = image_path_from_url(images_dir, main_image_url)
        if image_path is None or not image_path.exists():
            missing_file += 1
            continue

        jobs.append(
            ImageEmbeddingJob(
                parent_asin=parent_asin,
                main_image_url=main_image_url,
                image_path=image_path,
            )
        )

    return jobs, missing_url, missing_file


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def flatten_json_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return normalize_whitespace(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "\n".join(part for part in (flatten_json_value(item) for item in value) if part)
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            item_text = flatten_json_value(item)
            if item_text:
                parts.append(f"{key}: {item_text}")
        return "\n".join(parts)
    return normalize_whitespace(str(value))


def text_for_embedding(column_name: str, raw_value: str | None) -> str | None:
    if raw_value is None:
        return None

    if column_name == "product_title":
        text = normalize_whitespace(raw_value)
        return text or None

    try:
        payload = json.loads(raw_value)
    except json.JSONDecodeError:
        text = normalize_whitespace(raw_value)
        return text or None

    text = normalize_whitespace(flatten_json_value(payload))
    return text or None


def collect_text_jobs(
    con: duckdb.DuckDBPyConnection,
    column_name: str,
) -> tuple[list[tuple[str, str]], int]:
    rows = con.execute(
        f"""
        SELECT parent_asin, CAST({column_name} AS VARCHAR)
        FROM products_filtered
        ORDER BY parent_asin
        """
    ).fetchall()

    jobs: list[tuple[str, str]] = []
    skipped = 0

    for parent_asin, raw_value in rows:
        text = text_for_embedding(column_name, raw_value)
        if not text:
            skipped += 1
            continue
        jobs.append((parent_asin, text))

    return jobs, skipped


def collect_review_text_jobs(
    con: duckdb.DuckDBPyConnection,
    column_name: str,
) -> tuple[list[tuple[int, str]], int]:
    rows = con.execute(
        f"""
        SELECT embedding_row_id, CAST({column_name} AS VARCHAR)
        FROM reviews_with_embeddings
        ORDER BY embedding_row_id
        """
    ).fetchall()

    jobs: list[tuple[int, str]] = []
    skipped = 0

    for embedding_row_id, raw_value in rows:
        text = text_for_embedding(column_name, raw_value)
        if not text:
            skipped += 1
            continue
        jobs.append((embedding_row_id, text))

    return jobs, skipped


def load_images(batch_jobs: list[ImageEmbeddingJob]) -> tuple[list[Image.Image], list[ImageEmbeddingJob], int]:
    images: list[Image.Image] = []
    valid_jobs: list[ImageEmbeddingJob] = []
    failed = 0

    for job in batch_jobs:
        try:
            with Image.open(job.image_path) as image:
                images.append(image.convert("RGB"))
            valid_jobs.append(job)
        except (OSError, UnidentifiedImageError):
            failed += 1

    return images, valid_jobs, failed


def compute_image_batch_embeddings(
    processor: object,
    model: object,
    torch_module: object,
    device: str,
    images: list[Image.Image],
) -> tuple[list[list[float]], list[bool]]:
    inputs = processor(images=images, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch_module.inference_mode():
        image_features = model.get_image_features(**inputs)
        image_features = torch_module.nn.functional.normalize(image_features, p=2, dim=1)
        return image_features.detach().cpu().to(torch_module.float32).tolist(), [False] * len(images)


def last_token_pool(
    last_hidden_states: object,
    attention_mask: object,
    torch_module: object,
) -> object:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]

    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch_module.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


def compute_qwen_text_batch_embeddings(
    tokenizer: object,
    model: object,
    torch_module: object,
    device: str,
    texts: list[str],
) -> tuple[list[list[float]], list[bool]]:
    batch = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=QWEN_TEXT_MAX_LENGTH,
        return_tensors="pt",
        return_overflowing_tokens=True,
    )
    batch, input_is_truncated = select_primary_text_inputs(batch, len(texts))
    batch = {key: value.to(device) for key, value in batch.items()}

    with torch_module.inference_mode():
        outputs = model(**batch)
        embeddings = last_token_pool(outputs.last_hidden_state, batch["attention_mask"], torch_module)
        embeddings = torch_module.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.detach().cpu().to(torch_module.float32).tolist(), input_is_truncated


def compute_siglip_text_batch_embeddings(
    tokenizer: object,
    model: object,
    torch_module: object,
    device: str,
    texts: list[str],
) -> tuple[list[list[float]], list[bool]]:
    batch = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=SIGLIP_TEXT_MAX_LENGTH,
        return_tensors="pt",
        return_overflowing_tokens=True,
    )
    batch, input_is_truncated = select_primary_text_inputs(batch, len(texts))
    batch = {key: value.to(device) for key, value in batch.items()}

    with torch_module.inference_mode():
        embeddings = model.get_text_features(**batch)
        embeddings = torch_module.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.detach().cpu().to(torch_module.float32).tolist(), input_is_truncated


def compute_text_batch_embeddings(
    text_model_bundle: TextEmbeddingModelBundle,
    device: str,
    texts: list[str],
) -> tuple[list[list[float]], list[bool]]:
    return text_model_bundle.compute_batch(
        text_model_bundle.tokenizer,
        text_model_bundle.model,
        text_model_bundle.torch_module,
        device,
        texts,
    )


def select_primary_text_inputs(
    batch: object,
    expected_batch_size: int,
) -> tuple[dict[str, object], list[bool]]:
    overflow_to_sample_mapping = batch.pop("overflow_to_sample_mapping", None)
    if overflow_to_sample_mapping is None:
        return batch, [False] * expected_batch_size

    sample_to_first_index: dict[int, int] = {}
    input_is_truncated = [False] * expected_batch_size

    for flat_index, sample_index in enumerate(overflow_to_sample_mapping.tolist()):
        if sample_index not in sample_to_first_index:
            sample_to_first_index[sample_index] = flat_index
            continue
        input_is_truncated[sample_index] = True

    if len(sample_to_first_index) != expected_batch_size:
        raise RuntimeError(
            "Tokenizer overflow mapping did not include one primary encoding per input item"
        )

    primary_indices = [sample_to_first_index[index] for index in range(expected_batch_size)]
    primary_batch = {
        key: value[primary_indices]
        for key, value in batch.items()
        if key in {"input_ids", "attention_mask", "token_type_ids", "position_ids"}
    }
    return primary_batch, input_is_truncated


def clear_cuda_memory(torch_module: object) -> None:
    cuda_module = getattr(torch_module, "cuda", None)
    if cuda_module is not None and cuda_module.is_available():
        cuda_module.empty_cache()
    gc.collect()


def is_cuda_oom_error(exc: BaseException, torch_module: object) -> bool:
    cuda_oom_error = getattr(getattr(torch_module, "cuda", None), "OutOfMemoryError", None)
    if cuda_oom_error is not None and isinstance(exc, cuda_oom_error):
        return True

    message = str(exc).lower()
    return "cuda out of memory" in message or "cudnn_status_alloc_failed" in message


def process_batch_with_oom_retry(
    batch_items: list[object],
    process_batch: Callable[[list[object]], tuple[int, int, int]],
    torch_module: object,
    description: str,
) -> tuple[int, int, int]:
    if not batch_items:
        return 0, 0, 0

    try:
        return process_batch(batch_items)
    except Exception as exc:
        if not is_cuda_oom_error(exc, torch_module):
            raise

        clear_cuda_memory(torch_module)
        if len(batch_items) == 1:
            print(f"[warn] skipped 1 {description} row after CUDA OOM")
            return 0, 1, 0

        midpoint = len(batch_items) // 2
        left_items = batch_items[:midpoint]
        right_items = batch_items[midpoint:]
        print(
            f"[warn] {description} batch of {len(batch_items)} hit CUDA OOM; "
            f"retrying as {len(left_items)} + {len(right_items)}"
        )

        left_embedded, left_skipped, left_cached = process_batch_with_oom_retry(
            left_items,
            process_batch,
            torch_module,
            description,
        )
        right_embedded, right_skipped, right_cached = process_batch_with_oom_retry(
            right_items,
            process_batch,
            torch_module,
            description,
        )
        return (
            left_embedded + right_embedded,
            left_skipped + right_skipped,
            left_cached + right_cached,
        )


def write_embedding_batch(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    key_column_name: str,
    key_column_type: str,
    embedding_column_name_value: str,
    batch_rows: list[tuple[object, list[float], bool]],
    embedding_type: str,
) -> None:
    if not batch_rows:
        return

    temp_table_name = "batch_embedding_rows"
    truncation_column_name = input_is_truncated_column_name(embedding_column_name_value)
    con.execute(
        f"CREATE OR REPLACE TEMP TABLE {temp_table_name} "
        f"({key_column_name} {key_column_type}, embedding {embedding_type}, input_is_truncated BOOLEAN)"
    )
    con.executemany(f"INSERT INTO {temp_table_name} VALUES (?, ?, ?)", batch_rows)
    con.execute(
        f"""
        UPDATE {table_name} AS t
        SET {embedding_column_name_value} = b.embedding,
            {truncation_column_name} = b.input_is_truncated
        FROM {temp_table_name} AS b
        WHERE t.{key_column_name} = b.{key_column_name}
        """
    )
    con.execute(f"DROP TABLE {temp_table_name}")


def export_products_with_embeddings(
    con: duckdb.DuckDBPyConnection,
    output_path: Path,
) -> None:
    path_sql = sql_string_literal(str(output_path))
    con.execute(f"COPY products_with_embeddings TO {path_sql} (FORMAT PARQUET, COMPRESSION ZSTD)")


def export_reviews_with_embeddings(
    con: duckdb.DuckDBPyConnection,
    output_path: Path,
) -> None:
    path_sql = sql_string_literal(str(output_path))
    con.execute(
        f"""
        COPY (
            SELECT * EXCLUDE (embedding_row_id)
            FROM reviews_with_embeddings
        )
        TO {path_sql} (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )


def run_embeddings(
    data_dir: Path,
    batch_size: int,
) -> None:
    db_path, images_dir = validate_data_dir(data_dir)
    cache_conn = open_embedding_cache(data_dir)
    image_column_name = embedding_column_name("main_image_local", SIGLIP_MODEL_NAME)
    products_output_path = output_products_parquet_path(data_dir)
    reviews_output_path = output_reviews_parquet_path(data_dir)

    device = resolve_device()
    print(f"[info] loading image embedding model {SIGLIP_MODEL_NAME} on {device}")
    image_processor, image_model, image_dimension, image_torch = load_siglip_model_bundle(device)
    print(f"[info] loading text embedding model {QWEN_TEXT_MODEL_NAME} on {device}")
    text_tokenizer, text_model, text_torch = load_qwen_text_model_bundle(device)
    text_model_bundles = [
        TextEmbeddingModelBundle(
            model_name=QWEN_TEXT_MODEL_NAME,
            tokenizer=text_tokenizer,
            model=text_model,
            torch_module=text_torch,
            compute_batch=compute_qwen_text_batch_embeddings,
        )
    ]

    print(
        f"[info] loading text tokenizer for {SIGLIP_MODEL_NAME} "
        "and reusing image model for SigLIP text embeddings"
    )
    siglip_text_tokenizer = load_siglip_text_tokenizer()

    text_model_bundles.append(
        TextEmbeddingModelBundle(
            model_name=SIGLIP_MODEL_NAME,
            tokenizer=siglip_text_tokenizer,
            model=image_model,
            torch_module=image_torch,
            compute_batch=compute_siglip_text_batch_embeddings,
        )
    )
    text_model_names = [bundle.model_name for bundle in text_model_bundles]

    con: duckdb.DuckDBPyConnection | None = None
    try:
        print(f"[info] opening DuckDB database: {db_path}")
        con = duckdb.connect(str(db_path))
        ensure_products_filtered(con)
        ensure_reviews_filtered(con)
        create_working_table(con, image_column_name, image_dimension, text_model_names)
        create_reviews_working_table(con, text_model_names)
        create_image_embedding_job_table(con)

        image_jobs, missing_url, missing_file = collect_image_jobs(con, images_dir)
        print(
            f"[info] prepared {len(image_jobs)} image embedding jobs "
            f"({missing_url} missing MAIN hi_res URL, {missing_file} missing local file)"
        )

        image_embedded = 0
        failed_decode = 0
        image_truncated = 0
        for batch_start in range(0, len(image_jobs), batch_size):
            batch_jobs = image_jobs[batch_start : batch_start + batch_size]

            def process_image_batch(batch_items: list[object]) -> tuple[int, int, int]:
                nonlocal failed_decode, image_truncated

                batch_keys = [
                    cache_key_for_embedding("products_with_embeddings", image_column_name, job.parent_asin)
                    for job in batch_items
                ]
                cached_by_key = fetch_cached_embeddings(cache_conn, batch_keys)

                cached_rows: list[tuple[object, list[float], bool]] = []
                missing_jobs: list[ImageEmbeddingJob] = []
                decode_failed_batch = 0
                for job, cache_key in zip(batch_items, batch_keys, strict=True):  # type: ignore[arg-type]
                    cached = cached_by_key.get(cache_key)
                    if cached is None:
                        missing_jobs.append(job)
                    else:
                        embedding, input_is_truncated = cached
                        cached_rows.append((job.parent_asin, embedding, input_is_truncated))

                computed_rows: list[tuple[object, list[float], bool]] = []
                if missing_jobs:
                    images, valid_jobs, failed_in_batch = load_images(missing_jobs)
                    decode_failed_batch = failed_in_batch
                    if valid_jobs:
                        vectors, input_is_truncated = compute_image_batch_embeddings(
                            processor=image_processor,
                            model=image_model,
                            torch_module=image_torch,
                            device=device,
                            images=images,
                        )
                        computed_rows = [
                            (job.parent_asin, vector, truncated)
                            for job, vector, truncated in zip(
                                valid_jobs,
                                vectors,
                                input_is_truncated,
                                strict=True,
                            )
                        ]
                        store_cached_embeddings(
                            cache_conn,
                            [
                                (
                                    cache_key_for_embedding(
                                        "products_with_embeddings",
                                        image_column_name,
                                        job.parent_asin,
                                    ),
                                    "products_with_embeddings",
                                    "parent_asin",
                                    str(job.parent_asin),
                                    image_column_name,
                                    vector,
                                    truncated,
                                )
                                for job, vector, truncated in zip(
                                    valid_jobs,
                                    vectors,
                                    input_is_truncated,
                                    strict=True,
                                )
                            ],
                        )

                batch_rows = cached_rows + computed_rows
                if not batch_rows:
                    failed_decode += decode_failed_batch
                    return 0, 0, 0

                batch_truncated = sum(truncated for _, _, truncated in batch_rows)
                write_embedding_batch(
                    con,
                    table_name="products_with_embeddings",
                    key_column_name="parent_asin",
                    key_column_type="VARCHAR",
                    embedding_column_name_value=image_column_name,
                    batch_rows=batch_rows,
                    embedding_type=f"DOUBLE[{image_dimension}]",
                )
                failed_decode += decode_failed_batch
                image_truncated += batch_truncated
                return len(batch_rows), 0, len(cached_rows)

            embedded_in_batch, _, cached_in_batch = process_batch_with_oom_retry(
                batch_jobs,
                process_image_batch,
                image_torch,
                "image",
            )
            image_embedded += embedded_in_batch
            print(
                f"[info] embedded {image_embedded}/{len(image_jobs)} image rows "
                f"(batch size={embedded_in_batch}, cache hits={cached_in_batch})"
            )
        if failed_decode:
            print(f"[warn] skipped {failed_decode} image files that could not be decoded")
        if image_truncated:
            print(f"[info] image truncation flags set on {image_truncated} rows")

        for source_column in TEXT_EMBEDDING_COLUMNS:
            text_jobs, skipped = collect_text_jobs(con, source_column)
            for text_model_bundle in text_model_bundles:
                text_column_name = embedding_column_name(source_column, text_model_bundle.model_name)
                print(
                    f"[info] prepared {len(text_jobs)} text embedding jobs for {source_column} "
                    f"with {text_model_bundle.model_name} ({skipped} skipped)"
                )

                text_embedded = 0
                text_truncated = 0
                for batch_start in range(0, len(text_jobs), batch_size):
                    batch_jobs = text_jobs[batch_start : batch_start + batch_size]

                    def process_text_batch(batch_items: list[object]) -> tuple[int, int, int]:
                        nonlocal text_truncated

                        batch_keys = [
                            cache_key_for_embedding("products_with_embeddings", text_column_name, parent_asin)
                            for parent_asin, _ in batch_items  # type: ignore[misc]
                        ]
                        cached_by_key = fetch_cached_embeddings(cache_conn, batch_keys)

                        cached_rows: list[tuple[object, list[float], bool]] = []
                        missing_items: list[tuple[str, str]] = []
                        for (parent_asin, text), cache_key in zip(batch_items, batch_keys, strict=True):  # type: ignore[misc]
                            cached = cached_by_key.get(cache_key)
                            if cached is None:
                                missing_items.append((parent_asin, text))
                            else:
                                embedding, input_is_truncated = cached
                                cached_rows.append((parent_asin, embedding, input_is_truncated))

                        computed_rows: list[tuple[object, list[float], bool]] = []
                        if missing_items:
                            batch_parent_asins = [parent_asin for parent_asin, _ in missing_items]
                            batch_texts = [text for _, text in missing_items]
                            vectors, input_is_truncated = compute_text_batch_embeddings(
                                text_model_bundle=text_model_bundle,
                                device=device,
                                texts=batch_texts,
                            )
                            computed_rows = list(
                                zip(batch_parent_asins, vectors, input_is_truncated, strict=True)
                            )
                            store_cached_embeddings(
                                cache_conn,
                                [
                                    (
                                        cache_key_for_embedding(
                                            "products_with_embeddings",
                                            text_column_name,
                                            parent_asin,
                                        ),
                                        "products_with_embeddings",
                                        "parent_asin",
                                        str(parent_asin),
                                        text_column_name,
                                        vector,
                                        truncated,
                                    )
                                    for parent_asin, vector, truncated in zip(
                                        batch_parent_asins,
                                        vectors,
                                        input_is_truncated,
                                        strict=True,
                                    )
                                ],
                            )

                        batch_rows = cached_rows + computed_rows
                        if not batch_rows:
                            return 0, 0, 0

                        batch_truncated = sum(truncated for _, _, truncated in batch_rows)
                        write_embedding_batch(
                            con,
                            table_name="products_with_embeddings",
                            key_column_name="parent_asin",
                            key_column_type="VARCHAR",
                            embedding_column_name_value=text_column_name,
                            batch_rows=batch_rows,
                            embedding_type="DOUBLE[]",
                        )
                        text_truncated += batch_truncated
                        return len(batch_rows), 0, len(cached_rows)

                    embedded_in_batch, _, cached_in_batch = process_batch_with_oom_retry(
                        batch_jobs,
                        process_text_batch,
                        text_model_bundle.torch_module,
                        f"text for {source_column} with {text_model_bundle.model_name}",
                    )
                    text_embedded += embedded_in_batch
                    print(
                        f"[info] embedded {text_embedded}/{len(text_jobs)} text rows "
                        f"for {source_column} with {text_model_bundle.model_name} "
                        f"(batch size={embedded_in_batch}, cache hits={cached_in_batch})"
                    )
                if text_truncated:
                    print(
                        f"[info] truncation flags set on {text_truncated} "
                        f"rows for {source_column} with {text_model_bundle.model_name}"
                    )

        for source_column in REVIEW_TEXT_EMBEDDING_COLUMNS:
            text_jobs, skipped = collect_review_text_jobs(con, source_column)
            for text_model_bundle in text_model_bundles:
                text_column_name = embedding_column_name(source_column, text_model_bundle.model_name)
                print(
                    f"[info] prepared {len(text_jobs)} review text embedding jobs for {source_column} "
                    f"with {text_model_bundle.model_name} ({skipped} skipped)"
                )

                text_embedded = 0
                text_truncated = 0
                for batch_start in range(0, len(text_jobs), batch_size):
                    batch_jobs = text_jobs[batch_start : batch_start + batch_size]

                    def process_review_text_batch(batch_items: list[object]) -> tuple[int, int, int]:
                        nonlocal text_truncated

                        batch_keys = [
                            cache_key_for_embedding("reviews_with_embeddings", text_column_name, row_id)
                            for row_id, _ in batch_items  # type: ignore[misc]
                        ]
                        cached_by_key = fetch_cached_embeddings(cache_conn, batch_keys)

                        cached_rows: list[tuple[object, list[float], bool]] = []
                        missing_items: list[tuple[int, str]] = []
                        for (row_id, text), cache_key in zip(batch_items, batch_keys, strict=True):  # type: ignore[misc]
                            cached = cached_by_key.get(cache_key)
                            if cached is None:
                                missing_items.append((row_id, text))
                            else:
                                embedding, input_is_truncated = cached
                                cached_rows.append((row_id, embedding, input_is_truncated))

                        computed_rows: list[tuple[object, list[float], bool]] = []
                        if missing_items:
                            batch_row_ids = [row_id for row_id, _ in missing_items]
                            batch_texts = [text for _, text in missing_items]
                            vectors, input_is_truncated = compute_text_batch_embeddings(
                                text_model_bundle=text_model_bundle,
                                device=device,
                                texts=batch_texts,
                            )
                            computed_rows = list(
                                zip(batch_row_ids, vectors, input_is_truncated, strict=True)
                            )
                            store_cached_embeddings(
                                cache_conn,
                                [
                                    (
                                        cache_key_for_embedding(
                                            "reviews_with_embeddings",
                                            text_column_name,
                                            row_id,
                                        ),
                                        "reviews_with_embeddings",
                                        "embedding_row_id",
                                        str(row_id),
                                        text_column_name,
                                        vector,
                                        truncated,
                                    )
                                    for row_id, vector, truncated in zip(
                                        batch_row_ids,
                                        vectors,
                                        input_is_truncated,
                                        strict=True,
                                    )
                                ],
                            )

                        batch_rows = cached_rows + computed_rows
                        if not batch_rows:
                            return 0, 0, 0

                        batch_truncated = sum(truncated for _, _, truncated in batch_rows)
                        write_embedding_batch(
                            con,
                            table_name="reviews_with_embeddings",
                            key_column_name="embedding_row_id",
                            key_column_type="BIGINT",
                            embedding_column_name_value=text_column_name,
                            batch_rows=batch_rows,
                            embedding_type="DOUBLE[]",
                        )
                        text_truncated += batch_truncated
                        return len(batch_rows), 0, len(cached_rows)

                    embedded_in_batch, _, cached_in_batch = process_batch_with_oom_retry(
                        batch_jobs,
                        process_review_text_batch,
                        text_model_bundle.torch_module,
                        f"review text for {source_column} with {text_model_bundle.model_name}",
                    )
                    text_embedded += embedded_in_batch
                    print(
                        f"[info] embedded {text_embedded}/{len(text_jobs)} review text rows "
                        f"for {source_column} with {text_model_bundle.model_name} "
                        f"(batch size={embedded_in_batch}, cache hits={cached_in_batch})"
                    )
                if text_truncated:
                    print(
                        f"[info] truncation flags set on {text_truncated} "
                        f"review rows for {source_column} with {text_model_bundle.model_name}"
                    )

        if products_output_path.exists():
            products_output_path.unlink()
        if reviews_output_path.exists():
            reviews_output_path.unlink()

        export_products_with_embeddings(con, products_output_path)
        export_reviews_with_embeddings(con, reviews_output_path)
        print(
            f"[ok] wrote {products_output_path} and {reviews_output_path}; "
            f"product text columns: {', '.join(TEXT_EMBEDDING_COLUMNS)}; "
            f"review text columns: {', '.join(REVIEW_TEXT_EMBEDDING_COLUMNS)}; "
            f"text models: {', '.join(bundle.model_name for bundle in text_model_bundles)}"
        )
    finally:
        if con is not None:
            con.close()
        cache_conn.close()


def main() -> int:
    args = parse_args()
    data_dir = resolve_data_dir(args.data_dir)

    if args.batch_size <= 0:
        print("[error] --batch-size must be positive")
        return 2

    try:
        run_embeddings(
            data_dir=data_dir,
            batch_size=args.batch_size,
        )
    except Exception as exc:
        print(f"[error] {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
