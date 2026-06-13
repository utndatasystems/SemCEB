#!/usr/bin/env python3
"""Compute image and text embeddings for Amazon Reviews products and reviews.

This script reads a processed dataset directory created by ``download_and_prepare_amazon_reviews_dataset.py``,
computes embeddings for selected columns in the filtered product and review
tables, and exports new parquet files without modifying the original artifacts.

Example:
    python tools/amazon-reviews/compute_embeddings.py --run-dir Arts_Crafts_and_Sewing__raw_5core
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import duckdb
from PIL import Image, UnidentifiedImageError


DEFAULT_IMAGE_MODEL_NAME = "google/siglip2-base-patch16-224"
DEFAULT_TEXT_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_BATCH_SIZE = 32
SCRIPT_DIR = Path(__file__).resolve().parent
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute embeddings for filtered product and review tables"
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help=(
            "Processed dataset directory created by download_and_prepare_amazon_reviews_dataset.py. "
            "Example: processed/Arts_Crafts_and_Sewing__raw_5core"
        ),
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_IMAGE_MODEL_NAME,
        help=(
            "Hugging Face model name to use for image embeddings. "
            f"Default: {DEFAULT_IMAGE_MODEL_NAME}"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Embedding batch size. Default: {DEFAULT_BATCH_SIZE}",
    )
    return parser.parse_args()


def resolve_run_dir(run_dir_arg: str) -> Path:
    run_dir = Path(run_dir_arg)

    if run_dir.is_absolute():
        return run_dir

    repo_root = SCRIPT_DIR.parents[1]
    processed_root = SCRIPT_DIR / "data" / "processed"

    if len(run_dir.parts) == 1:
        return (processed_root / run_dir).resolve()

    return (repo_root / run_dir).resolve()


def sql_string_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def sanitize_model_name(model_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")


def embedding_column_name(reference_column: str, model_name: str) -> str:
    return f"{reference_column}_embeddings_{sanitize_model_name(model_name)}"


def output_products_parquet_path(run_dir: Path) -> Path:
    return run_dir / "products_filtered_with_embeddings.parquet"


def output_reviews_parquet_path(run_dir: Path) -> Path:
    return run_dir / "reviews_filtered_with_embeddings.parquet"


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


def load_image_model_bundle(model_name: str, device: str) -> tuple[object, object, int, object]:
    import torch
    from transformers import AutoConfig, AutoImageProcessor, AutoModel

    config = AutoConfig.from_pretrained(model_name)
    dimension = getattr(getattr(config, "text_config", None), "projection_size", None)
    if dimension is None:
        dimension = getattr(getattr(config, "vision_config", None), "hidden_size", None)
    if dimension is None:
        raise RuntimeError(f"Could not determine embedding dimension for model {model_name}")

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to(device)
    return processor, model, int(dimension), torch


def load_text_model_bundle(model_name: str, device: str) -> tuple[object, object, object]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to(device)
    return tokenizer, model, torch


def validate_run_dir(run_dir: Path) -> tuple[Path, Path]:
    db_path = run_dir / "amazon_reviews.duckdb"
    images_dir = run_dir / "images"

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    if not db_path.exists():
        raise FileNotFoundError(f"DuckDB file not found: {db_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    return db_path, images_dir


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
) -> None:
    con.execute("CREATE OR REPLACE TEMP TABLE products_with_embeddings AS SELECT * FROM products_filtered")
    con.execute(
        f"ALTER TABLE products_with_embeddings "
        f"ADD COLUMN {image_column_name} DOUBLE[{image_dimension}]"
    )
    for source_column in TEXT_EMBEDDING_COLUMNS:
        text_column_name = embedding_column_name(source_column, DEFAULT_TEXT_MODEL_NAME)
        con.execute(
            f"ALTER TABLE products_with_embeddings "
            f"ADD COLUMN {text_column_name} DOUBLE[]"
        )


def create_reviews_working_table(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE reviews_with_embeddings AS
        SELECT
            row_number() OVER () AS embedding_row_id,
            *
        FROM reviews_filtered
        """
    )
    for source_column in REVIEW_TEXT_EMBEDDING_COLUMNS:
        text_column_name = embedding_column_name(source_column, DEFAULT_TEXT_MODEL_NAME)
        con.execute(
            f"ALTER TABLE reviews_with_embeddings "
            f"ADD COLUMN {text_column_name} DOUBLE[]"
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
) -> list[list[float]]:
    inputs = processor(images=images, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch_module.inference_mode():
        if hasattr(model, "get_image_features"):
            image_features = model.get_image_features(**inputs)
        else:
            outputs = model(**inputs)
            if not hasattr(outputs, "image_embeds"):
                raise RuntimeError("Image model output does not expose image embeddings")
            image_features = outputs.image_embeds

        if not isinstance(image_features, torch_module.Tensor):
            if hasattr(image_features, "image_embeds") and isinstance(
                image_features.image_embeds,
                torch_module.Tensor,
            ):
                image_features = image_features.image_embeds
            elif hasattr(image_features, "pooler_output") and isinstance(
                image_features.pooler_output,
                torch_module.Tensor,
            ):
                image_features = image_features.pooler_output
            elif hasattr(image_features, "last_hidden_state") and isinstance(
                image_features.last_hidden_state,
                torch_module.Tensor,
            ):
                image_features = image_features.last_hidden_state[:, 0, :]
            else:
                raise RuntimeError(
                    "Image model did not return a tensor embedding; "
                    f"got {type(image_features).__name__}"
                )

        image_features = torch_module.nn.functional.normalize(image_features, p=2, dim=1)
        return image_features.detach().cpu().to(torch_module.float32).tolist()


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


def compute_text_batch_embeddings(
    tokenizer: object,
    model: object,
    torch_module: object,
    device: str,
    texts: list[str],
) -> list[list[float]]:
    batch = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    batch = {key: value.to(device) for key, value in batch.items()}

    with torch_module.inference_mode():
        outputs = model(**batch)
        embeddings = last_token_pool(outputs.last_hidden_state, batch["attention_mask"], torch_module)
        embeddings = torch_module.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.detach().cpu().to(torch_module.float32).tolist()


def write_batch_embeddings(
    con: duckdb.DuckDBPyConnection,
    column_name: str,
    dimension: int,
    batch_rows: list[tuple[str, list[float]]],
) -> None:
    if not batch_rows:
        return

    con.execute(
        f"CREATE OR REPLACE TEMP TABLE batch_image_embeddings "
        f"(parent_asin VARCHAR, embedding DOUBLE[{dimension}])"
    )
    con.executemany("INSERT INTO batch_image_embeddings VALUES (?, ?)", batch_rows)
    con.execute(
        f"""
        UPDATE products_with_embeddings AS p
        SET {column_name} = b.embedding
        FROM batch_image_embeddings AS b
        WHERE p.parent_asin = b.parent_asin
        """
    )
    con.execute("DROP TABLE batch_image_embeddings")


def write_batch_list_embeddings(
    con: duckdb.DuckDBPyConnection,
    column_name: str,
    batch_rows: list[tuple[str, list[float]]],
) -> None:
    if not batch_rows:
        return

    con.execute("CREATE OR REPLACE TEMP TABLE batch_text_embeddings (parent_asin VARCHAR, embedding DOUBLE[])")
    con.executemany("INSERT INTO batch_text_embeddings VALUES (?, ?)", batch_rows)
    con.execute(
        f"""
        UPDATE products_with_embeddings AS p
        SET {column_name} = b.embedding
        FROM batch_text_embeddings AS b
        WHERE p.parent_asin = b.parent_asin
        """
    )
    con.execute("DROP TABLE batch_text_embeddings")


def write_review_batch_list_embeddings(
    con: duckdb.DuckDBPyConnection,
    column_name: str,
    batch_rows: list[tuple[int, list[float]]],
) -> None:
    if not batch_rows:
        return

    con.execute(
        "CREATE OR REPLACE TEMP TABLE batch_review_text_embeddings "
        "(embedding_row_id BIGINT, embedding DOUBLE[])"
    )
    con.executemany("INSERT INTO batch_review_text_embeddings VALUES (?, ?)", batch_rows)
    con.execute(
        f"""
        UPDATE reviews_with_embeddings AS r
        SET {column_name} = b.embedding
        FROM batch_review_text_embeddings AS b
        WHERE r.embedding_row_id = b.embedding_row_id
        """
    )
    con.execute("DROP TABLE batch_review_text_embeddings")


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
    run_dir: Path,
    image_model_name: str,
    batch_size: int,
) -> None:
    db_path, images_dir = validate_run_dir(run_dir)
    image_column_name = embedding_column_name("main_image_local", image_model_name)
    products_output_path = output_products_parquet_path(run_dir)
    reviews_output_path = output_reviews_parquet_path(run_dir)

    device = resolve_device()
    print(f"[info] loading image embedding model {image_model_name} on {device}")
    image_processor, image_model, image_dimension, image_torch = load_image_model_bundle(
        image_model_name,
        device,
    )
    print(f"[info] loading text embedding model {DEFAULT_TEXT_MODEL_NAME} on {device}")
    text_tokenizer, text_model, text_torch = load_text_model_bundle(DEFAULT_TEXT_MODEL_NAME, device)

    print(f"[info] opening DuckDB database: {db_path}")
    con = duckdb.connect(str(db_path))
    try:
        ensure_products_filtered(con)
        ensure_reviews_filtered(con)
        create_working_table(con, image_column_name, image_dimension)
        create_reviews_working_table(con)
        create_image_embedding_job_table(con)

        image_jobs, missing_url, missing_file = collect_image_jobs(con, images_dir)
        print(
            f"[info] prepared {len(image_jobs)} image embedding jobs "
            f"({missing_url} missing MAIN hi_res URL, {missing_file} missing local file)"
        )

        image_embedded = 0
        failed_decode = 0
        for batch_start in range(0, len(image_jobs), batch_size):
            batch_jobs = image_jobs[batch_start : batch_start + batch_size]
            images, valid_jobs, failed_in_batch = load_images(batch_jobs)
            failed_decode += failed_in_batch

            if not valid_jobs:
                continue

            vectors = compute_image_batch_embeddings(
                processor=image_processor,
                model=image_model,
                torch_module=image_torch,
                device=device,
                images=images,
            )
            batch_rows = [
                (job.parent_asin, vector)
                for job, vector in zip(valid_jobs, vectors, strict=True)
            ]
            write_batch_embeddings(con, image_column_name, image_dimension, batch_rows)
            image_embedded += len(batch_rows)
            print(
                f"[info] embedded {image_embedded}/{len(image_jobs)} image rows "
                f"(batch size={len(batch_rows)})"
            )

        for source_column in TEXT_EMBEDDING_COLUMNS:
            text_column_name = embedding_column_name(source_column, DEFAULT_TEXT_MODEL_NAME)
            text_jobs, skipped = collect_text_jobs(con, source_column)
            print(
                f"[info] prepared {len(text_jobs)} text embedding jobs for {source_column} "
                f"({skipped} skipped)"
            )

            text_embedded = 0
            for batch_start in range(0, len(text_jobs), batch_size):
                batch_jobs = text_jobs[batch_start : batch_start + batch_size]
                batch_parent_asins = [parent_asin for parent_asin, _ in batch_jobs]
                batch_texts = [text for _, text in batch_jobs]
                vectors = compute_text_batch_embeddings(
                    tokenizer=text_tokenizer,
                    model=text_model,
                    torch_module=text_torch,
                    device=device,
                    texts=batch_texts,
                )
                batch_rows = list(zip(batch_parent_asins, vectors, strict=True))
                write_batch_list_embeddings(con, text_column_name, batch_rows)
                text_embedded += len(batch_rows)
                print(
                    f"[info] embedded {text_embedded}/{len(text_jobs)} text rows "
                    f"for {source_column} (batch size={len(batch_rows)})"
                )

        for source_column in REVIEW_TEXT_EMBEDDING_COLUMNS:
            text_column_name = embedding_column_name(source_column, DEFAULT_TEXT_MODEL_NAME)
            text_jobs, skipped = collect_review_text_jobs(con, source_column)
            print(
                f"[info] prepared {len(text_jobs)} review text embedding jobs for {source_column} "
                f"({skipped} skipped)"
            )

            text_embedded = 0
            for batch_start in range(0, len(text_jobs), batch_size):
                batch_jobs = text_jobs[batch_start : batch_start + batch_size]
                batch_row_ids = [row_id for row_id, _ in batch_jobs]
                batch_texts = [text for _, text in batch_jobs]
                vectors = compute_text_batch_embeddings(
                    tokenizer=text_tokenizer,
                    model=text_model,
                    torch_module=text_torch,
                    device=device,
                    texts=batch_texts,
                )
                batch_rows = list(zip(batch_row_ids, vectors, strict=True))
                write_review_batch_list_embeddings(con, text_column_name, batch_rows)
                text_embedded += len(batch_rows)
                print(
                    f"[info] embedded {text_embedded}/{len(text_jobs)} review text rows "
                    f"for {source_column} (batch size={len(batch_rows)})"
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
            f"review text columns: {', '.join(REVIEW_TEXT_EMBEDDING_COLUMNS)}"
        )
    finally:
        con.close()


def main() -> int:
    args = parse_args()
    run_dir = resolve_run_dir(args.run_dir)

    if args.batch_size <= 0:
        print("[error] --batch-size must be positive")
        return 2

    try:
        run_embeddings(
            run_dir=run_dir,
            image_model_name=args.model,
            batch_size=args.batch_size,
        )
    except Exception as exc:
        print(f"[error] {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
