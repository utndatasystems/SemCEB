#!/usr/bin/env python3
"""Import precomputed parquet embeddings into the Amazon Reviews cache.

This helper seeds the SQLite embedding cache from precomputed parquet files
without recomputing any embeddings.
"""

from __future__ import annotations

import argparse
import pickle
import sqlite3
import sys
from pathlib import Path

try:
    import duckdb
except ImportError as exc:  # pragma: no cover - environment-specific
    raise SystemExit(
        "duckdb is required to read the embedded parquet files; "
        "run this with the project dev environment or install duckdb"
    ) from exc


DEFAULT_DATA_DIR = "Arts_Crafts_and_Sewing__raw_5core"
CACHE_DB_FILENAME = "embedding_cache.sqlite3"
EMBEDDING_SUFFIX = "_embeddings_"
SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import existing parquet embeddings into the local cache"
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help=(
            "Processed dataset directory created by download_and_prepare_amazon_reviews_dataset.py. "
            f"Default: {DEFAULT_DATA_DIR}"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Number of rows to upsert per SQLite batch. Default: 1024",
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


def sql_string_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def embedding_columns(con: duckdb.DuckDBPyConnection, parquet_path: Path) -> list[str]:
    rows = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet({sql_string_literal(str(parquet_path))})"
    ).fetchall()
    return [
        name
        for name, *_ in rows
        if EMBEDDING_SUFFIX in name and not name.endswith("_input_is_truncated")
    ]


def truncation_column_name(embedding_column: str) -> str:
    return f"{embedding_column}_input_is_truncated"


def cache_key_for_embedding(table_name: str, embedding_column_name: str, key_value: object) -> str:
    return f"{table_name}|{embedding_column_name}|{key_value}"


def upsert_embedding_rows(
    cache_conn: sqlite3.Connection,
    table_name: str,
    key_column_name: str,
    key_values: list[object],
    embedded_batch: dict[str, list[object]],
    embedding_cols: list[str],
    default_truncated: bool,
) -> int:
    payload_rows: list[tuple[str, str, str, str, str, sqlite3.Binary, int]] = []
    key_values_list = [None if value is None else str(value) for value in key_values]

    for row_index, key_value in enumerate(key_values_list):
        if key_value is None:
            continue

        for embedding_column in embedding_cols:
            vector = embedded_batch[embedding_column][row_index]
            if vector is None:
                continue

            truncation_column = truncation_column_name(embedding_column)
            truncated_values = embedded_batch.get(truncation_column)
            truncated = (
                bool(truncated_values[row_index])
                if truncated_values is not None
                else default_truncated
            )

            payload_rows.append(
                (
                    cache_key_for_embedding(table_name, embedding_column, key_value),
                    table_name,
                    key_column_name,
                    key_value,
                    embedding_column,
                    sqlite3.Binary(pickle.dumps(vector, protocol=pickle.HIGHEST_PROTOCOL)),
                    int(truncated),
                )
            )

    if not payload_rows:
        return 0

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
    return len(payload_rows)


def import_products_embeddings(
    con: duckdb.DuckDBPyConnection,
    data_dir: Path,
    cache_conn: sqlite3.Connection,
    batch_size: int,
) -> tuple[int, list[str]]:
    embedded_path = data_dir / "products_filtered_with_embeddings.parquet"
    product_embedding_cols = embedding_columns(con, embedded_path)
    column_names = ["parent_asin", *product_embedding_cols]
    query = (
        f"SELECT {', '.join(column_names)} "
        f"FROM read_parquet({sql_string_literal(str(embedded_path))})"
    )
    result = con.execute(query)

    imported = 0
    while True:
        rows = result.fetchmany(batch_size)
        if not rows:
            break

        embedded_batch: dict[str, list[object]] = {column: [] for column in column_names}
        for row in rows:
            for column, value in zip(column_names, row, strict=True):
                embedded_batch[column].append(value)

        imported += upsert_embedding_rows(
            cache_conn=cache_conn,
            table_name="products_with_embeddings",
            key_column_name="parent_asin",
            key_values=embedded_batch["parent_asin"],
            embedded_batch=embedded_batch,
            embedding_cols=product_embedding_cols,
            default_truncated=False,
        )

    return imported, product_embedding_cols


def import_reviews_embeddings(
    con: duckdb.DuckDBPyConnection,
    data_dir: Path,
    cache_conn: sqlite3.Connection,
    batch_size: int,
) -> tuple[int, list[str]]:
    embedded_path = data_dir / "reviews_filtered_with_embeddings.parquet"
    review_embedding_cols = embedding_columns(con, embedded_path)
    query = (
        f"SELECT {', '.join(review_embedding_cols)} "
        f"FROM read_parquet({sql_string_literal(str(embedded_path))})"
    )
    result = con.execute(query)

    imported = 0
    row_offset = 0
    while True:
        rows = result.fetchmany(batch_size)
        if not rows:
            break

        embedded_batch: dict[str, list[object]] = {column: [] for column in review_embedding_cols}
        for row in rows:
            for column, value in zip(review_embedding_cols, row, strict=True):
                embedded_batch[column].append(value)

        row_ids = list(range(row_offset + 1, row_offset + 1 + len(rows)))
        imported += upsert_embedding_rows(
            cache_conn=cache_conn,
            table_name="reviews_with_embeddings",
            key_column_name="embedding_row_id",
            key_values=row_ids,
            embedded_batch=embedded_batch,
            embedding_cols=review_embedding_cols,
            default_truncated=False,
        )
        row_offset += len(rows)

    return imported, review_embedding_cols


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        print("[error] --batch-size must be positive")
        return 2

    data_dir = resolve_data_dir(args.data_dir)
    if not data_dir.exists():
        print(f"[error] data directory does not exist: {data_dir}")
        return 1

    products_path = data_dir / "products_filtered_with_embeddings.parquet"
    reviews_path = data_dir / "reviews_filtered_with_embeddings.parquet"
    if not products_path.exists():
        print(f"[error] missing file: {products_path}")
        return 1
    if not reviews_path.exists():
        print(f"[error] missing file: {reviews_path}")
        return 1

    con = duckdb.connect()
    cache_conn = open_embedding_cache(data_dir)
    try:
        products_imported, product_embedding_cols = import_products_embeddings(
            con, data_dir, cache_conn, args.batch_size
        )
        reviews_imported, review_embedding_cols = import_reviews_embeddings(
            con, data_dir, cache_conn, args.batch_size
        )
    finally:
        cache_conn.close()
        con.close()

    print(
        f"[ok] imported {products_imported} product embedding rows and "
        f"{reviews_imported} review embedding rows into {embedding_cache_path(data_dir)}"
    )
    print(f"[info] embedding columns detected in products: {', '.join(product_embedding_cols)}")
    print(f"[info] embedding columns detected in reviews: {', '.join(review_embedding_cols)}")
    print(
        "[info] cache rows overwrite on cache_key conflict; "
        "missing truncation columns default to false"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
