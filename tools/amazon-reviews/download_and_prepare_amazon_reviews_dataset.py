#!/usr/bin/env python3
"""Prepare Amazon Reviews 2023 category data with DuckDB.

This script supports an iterative workflow:
1) analyze category footprints (raw + metadata vs 5-core)
2) set up a chosen category locally with compact DuckDB/parquet tables

Examples:
  python tools/amazon-reviews/download_and_prepare_amazon_reviews_dataset.py --analyze-only
  python tools/amazon-reviews/download_and_prepare_amazon_reviews_dataset.py --category Arts_Crafts_and_Sewing --mode raw_5core
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit
from urllib.request import urlopen

import duckdb

DATASET_ID = "McAuley-Lab/Amazon-Reviews-2023"
HF_DATASET_BASE = f"https://huggingface.co/datasets/{DATASET_ID}/"
HF_RESOLVE_BASE = urljoin(HF_DATASET_BASE, "resolve/main/")
HF_TREE_API = f"https://huggingface.co/api/datasets/{DATASET_ID}/tree/main?recursive=1"
SCRIPT_DIR = Path(__file__).resolve().parent

MODE_RAW = "raw"
MODE_RAW_5CORE = "raw_5core"
MODES_REQUIRING_5CORE = {MODE_RAW_5CORE}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set up Amazon Reviews 2023 data")
    parser.add_argument(
        "--category",
        default=None,
        help="Category name (e.g., Arts_Crafts_and_Sewing). Required unless --analyze-only is used.",
    )
    parser.add_argument(
        "--mode",
        choices=[MODE_RAW, MODE_RAW_5CORE],
        default=MODE_RAW,
        help=(
            "raw: full review text + metadata. "
            "raw_5core: raw reviews filtered to 5-core interactions + metadata."
        ),
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze category footprints and exit.",
    )
    parser.add_argument(
        "--force-refresh-tree",
        action="store_true",
        help="Re-download Hugging Face tree metadata even if cached.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download data files even if they exist locally.",
    )
    parser.add_argument(
        "--base-dir",
        default=None,
        help=(
            "Base directory for raw downloads, cache, and processed output. "
            "Defaults to tools/amazon-reviews. "
            "Relative paths are resolved from the repo root."
        ),
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=12,
        help="How many smallest categories to print in analysis output.",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def sql_string_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def human_bytes(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "n/a"

    size = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024 or unit == "TB":
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{num_bytes} B"


def download_file(url: str, dest: Path, force: bool = False) -> None:
    if dest.exists() and not force:
        print(f"[skip] {dest} already exists")
        return

    ensure_parent(dest)
    tmp_dest = dest.with_suffix(dest.suffix + ".part")

    try:
        with urlopen(url) as response:
            with tmp_dest.open("wb") as f_out:
                shutil.copyfileobj(response, f_out)
    except (HTTPError, URLError) as exc:
        if tmp_dest.exists():
            tmp_dest.unlink()
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc

    tmp_dest.replace(dest)
    print(f"[ok] downloaded {dest} ({human_bytes(dest.stat().st_size)})")


def _image_path_from_url(url: str, images_dir: Path) -> Path | None:
    parsed = urlsplit(url)
    filename = Path(parsed.path).name
    if not filename:
        return None

    # Mirror URL host/path to avoid filename collisions across different URLs.
    host = parsed.netloc or "unknown_host"
    rel_dir = Path(parsed.path.lstrip("/")).parent
    return images_dir / host / rel_dir / filename


def collect_distinct_product_image_urls(
    con: duckdb.DuckDBPyConnection,
) -> tuple[bool, list[str]]:
    table_exists = con.execute("""
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = 'main'
              AND table_name = 'products_filtered'
        )
        """).fetchone()[0]

    if not table_exists:
        return False, []

    con.execute("""
        CREATE OR REPLACE TEMP TABLE product_image_urls AS
        SELECT DISTINCT image_url
        FROM (
            SELECT NULLIF(trim(json_extract_string(e.value, '$.hi_res')), '') AS image_url
            FROM products_filtered p
            CROSS JOIN json_each(p.images_json) AS e
            WHERE json_extract_string(e.value, '$.variant') = 'MAIN'
        )
        WHERE image_url IS NOT NULL
        """)
    return True, [
        row[0]
        for row in con.execute(
            "SELECT image_url FROM product_image_urls ORDER BY image_url"
        ).fetchall()
    ]


def download_images_from_products_filtered(
    con: duckdb.DuckDBPyConnection,
    images_dir: Path,
) -> None:
    table_exists, urls = collect_distinct_product_image_urls(con)
    if not table_exists:
        print("[info] skipping image download: products_filtered table not found.")
        return
    if not urls:
        print(
            "[info] skipping image download: no image URLs found in products_filtered."
        )
        return

    existing_urls: list[tuple[str, Path]] = []
    pending_urls: list[tuple[str, Path]] = []
    ignored = 0

    for url in urls:
        image_path = _image_path_from_url(url, images_dir)
        if image_path is None:
            ignored += 1
            continue

        if image_path.exists():
            existing_urls.append((url, image_path))
        else:
            pending_urls.append((url, image_path))

    print(
        f"[info] {len(pending_urls)} images will be downloaded, "
        f"{len(existing_urls)} already exist, {ignored} ignored "
        f"from {len(urls)} distinct MAIN hi_res URLs"
    )

    downloaded = 0
    skipped = len(existing_urls)
    failed = 0

    for url, image_path in pending_urls:
        tmp_path = image_path.with_suffix(image_path.suffix + ".part")
        ensure_parent(tmp_path)
        try:
            with urlopen(url, timeout=20) as response:
                with tmp_path.open("wb") as out_file:
                    shutil.copyfileobj(response, out_file)
            tmp_path.replace(image_path)
            downloaded += 1
        except Exception:
            failed += 1
            if tmp_path.exists():
                tmp_path.unlink()

    print(
        "[ok] images sync complete: "
        f"{downloaded} downloaded, {skipped} already present, {failed} failed, "
        f"{ignored} ignored, {len(urls)} total URLs"
    )


def load_tree_table(con: duckdb.DuckDBPyConnection, tree_json_path: Path) -> None:
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE hf_tree AS
        SELECT *
        FROM read_json(?, format='array')
        """,
        [str(tree_json_path)],
    )


def fetch_tree_json(cache_path: Path, force_refresh: bool = False) -> None:
    if cache_path.exists() and not force_refresh:
        print(f"[skip] using cached tree metadata: {cache_path}")
        return

    print(f"[info] fetching repository tree metadata from {HF_TREE_API}")
    download_file(HF_TREE_API, cache_path, force=True)


def write_category_report(con: duckdb.DuckDBPyConnection, out_csv: Path) -> None:
    ensure_parent(out_csv)
    out_csv_sql = sql_string_literal(str(out_csv))
    con.execute(
        f"""
        COPY (
            WITH review AS (
                SELECT
                    regexp_extract(path, 'raw/review_categories/(.*)\\.jsonl$', 1) AS category,
                    size AS raw_review_bytes
                FROM hf_tree
                WHERE type = 'file' AND path LIKE 'raw/review_categories/%.jsonl'
            ),
            meta AS (
                SELECT
                    regexp_extract(path, 'raw/meta_categories/meta_(.*)\\.jsonl$', 1) AS category,
                    size AS raw_meta_bytes
                FROM hf_tree
                WHERE type = 'file' AND path LIKE 'raw/meta_categories/meta_%.jsonl'
            ),
            k5 AS (
                SELECT
                    regexp_extract(path, 'benchmark/5core/rating_only/(.*)\\.csv$', 1) AS category,
                    size AS rating_only_5core_bytes
                FROM hf_tree
                WHERE type = 'file' AND path LIKE 'benchmark/5core/rating_only/%.csv'
            )
            SELECT
                coalesce(review.category, meta.category, k5.category) AS category,
                coalesce(review.raw_review_bytes, 0) AS raw_review_bytes,
                coalesce(meta.raw_meta_bytes, 0) AS raw_meta_bytes,
                coalesce(review.raw_review_bytes, 0) + coalesce(meta.raw_meta_bytes, 0) AS raw_total_bytes,
                k5.rating_only_5core_bytes,
                CASE
                    WHEN k5.rating_only_5core_bytes IS NULL OR k5.rating_only_5core_bytes = 0 THEN NULL
                    ELSE round(
                        CAST(coalesce(review.raw_review_bytes, 0) AS DOUBLE)
                        / CAST(k5.rating_only_5core_bytes AS DOUBLE),
                        2
                    )
                END AS raw_review_to_5core_ratio
            FROM review
            FULL OUTER JOIN meta USING (category)
            FULL OUTER JOIN k5 USING (category)
            ORDER BY raw_total_bytes ASC
        )
        TO {out_csv_sql} (HEADER, DELIMITER ',')
        """,
    )


def analyze_categories(
    con: duckdb.DuckDBPyConnection,
    top_n: int,
) -> None:
    print("\n[analysis] smallest categories by raw review + metadata footprint")
    top_rows = con.execute(
        """
        WITH review AS (
            SELECT regexp_extract(path, 'raw/review_categories/(.*)\\.jsonl$', 1) AS category,
                   size AS review_bytes
            FROM hf_tree
            WHERE type='file' AND path LIKE 'raw/review_categories/%.jsonl'
        ),
        meta AS (
            SELECT regexp_extract(path, 'raw/meta_categories/meta_(.*)\\.jsonl$', 1) AS category,
                   size AS meta_bytes
            FROM hf_tree
            WHERE type='file' AND path LIKE 'raw/meta_categories/meta_%.jsonl'
        )
        SELECT
            coalesce(review.category, meta.category) AS category,
            coalesce(review.review_bytes, 0) AS review_bytes,
            coalesce(meta.meta_bytes, 0) AS meta_bytes,
            coalesce(review.review_bytes, 0) + coalesce(meta.meta_bytes, 0) AS total_bytes
        FROM review
        FULL OUTER JOIN meta USING (category)
        ORDER BY total_bytes ASC
        LIMIT ?
        """,
        [top_n],
    ).fetchall()

    for i, (category, review_bytes, meta_bytes, total_bytes) in enumerate(
        top_rows, start=1
    ):
        print(
            f"  {i:>2}. {category:<30} "
            f"review={human_bytes(review_bytes):>10} "
            f"meta={human_bytes(meta_bytes):>10} "
            f"total={human_bytes(total_bytes):>10}"
        )

    print("\n[analysis] smallest categories within 5-core rating_only")
    smallest_5core_rows = con.execute(
        """
        SELECT
            regexp_extract(path, 'benchmark/5core/rating_only/(.*)\\.csv$', 1) AS category,
            size AS bytes
        FROM hf_tree
        WHERE type='file' AND path LIKE 'benchmark/5core/rating_only/%.csv'
        ORDER BY bytes ASC
        LIMIT ?
        """,
        [top_n],
    ).fetchall()

    if not smallest_5core_rows:
        print("  (none found)")
    else:
        for i, (category, size_bytes) in enumerate(smallest_5core_rows, start=1):
            print(f"  {i:>2}. {category:<30} rating_only={human_bytes(size_bytes):>10}")

    missing = [row[0] for row in con.execute("""
            WITH raw_cats AS (
                SELECT regexp_extract(path, 'raw/review_categories/(.*)\\.jsonl$', 1) AS category
                FROM hf_tree
                WHERE type='file' AND path LIKE 'raw/review_categories/%.jsonl'
            ),
            core5_cats AS (
                SELECT regexp_extract(path, 'benchmark/5core/rating_only/(.*)\\.csv$', 1) AS category
                FROM hf_tree
                WHERE type='file' AND path LIKE 'benchmark/5core/rating_only/%.csv'
            )
            SELECT raw_cats.category
            FROM raw_cats
            LEFT JOIN core5_cats USING (category)
            WHERE core5_cats.category IS NULL
            ORDER BY raw_cats.category ASC
            """).fetchall()]

    print("\n[analysis] categories missing in 5-core")
    if missing:
        print("  " + ", ".join(missing))
    else:
        print("  none")

    print("\n[analysis] 5-core suitability note")
    print(
        "  5-core files are rating-only interactions (user_id, parent_asin, rating, timestamp)."
    )
    print("  They do not include review text, and 6 categories are absent in 5-core.")
    print(
        "  Use mode=raw_5core to keep review text while restricting to the 5-core interaction set."
    )


def get_all_raw_categories(con: duckdb.DuckDBPyConnection) -> set[str]:
    return {row[0] for row in con.execute("""
            SELECT regexp_extract(path, 'raw/review_categories/(.*)\\.jsonl$', 1) AS category
            FROM hf_tree
            WHERE type='file' AND path LIKE 'raw/review_categories/%.jsonl'
            """).fetchall()}


def get_all_5core_categories(con: duckdb.DuckDBPyConnection) -> set[str]:
    return {row[0] for row in con.execute("""
            SELECT regexp_extract(path, 'benchmark/5core/rating_only/(.*)\\.csv$', 1) AS category
            FROM hf_tree
            WHERE type='file' AND path LIKE 'benchmark/5core/rating_only/%.csv'
            """).fetchall()}


def choose_category(
    con: duckdb.DuckDBPyConnection,
    requested: str,
    mode: str,
) -> str:
    raw_categories = get_all_raw_categories(con)
    core5_categories = get_all_5core_categories(con)

    if requested not in raw_categories:
        valid = ", ".join(sorted(raw_categories))
        raise ValueError(
            f"Unknown category '{requested}'. Available raw categories: {valid}"
        )

    if mode in MODES_REQUIRING_5CORE and requested not in core5_categories:
        missing = ", ".join(sorted(raw_categories - core5_categories))
        raise ValueError(
            f"Category '{requested}' is not available in 5-core. "
            f"Categories missing in 5-core: {missing}"
        )

    return requested


def required_remote_paths(category: str, mode: str) -> list[str]:
    paths = [f"raw/meta_categories/meta_{category}.jsonl"]
    if mode in {MODE_RAW, MODE_RAW_5CORE}:
        paths.append(f"raw/review_categories/{category}.jsonl")
    if mode in MODES_REQUIRING_5CORE:
        paths.append(f"benchmark/5core/rating_only/{category}.csv")
    return paths


def download_inputs(
    base_dir: Path,
    category: str,
    mode: str,
    force_download: bool,
) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for remote_relpath in required_remote_paths(category, mode):
        url = urljoin(HF_RESOLVE_BASE, remote_relpath)
        local_path = base_dir / "raw" / remote_relpath
        download_file(url, local_path, force=force_download)
        mapping[remote_relpath] = local_path
    return mapping


def create_products_table(
    con: duckdb.DuckDBPyConnection,
    meta_jsonl_path: Path,
) -> None:
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE products_stage AS
        SELECT
            parent_asin,
            main_category,
            product_title,
            average_rating,
            rating_number,
            price,
            store,
            categories_json,
            features_json,
            description_json,
            details_json,
            images_json,
            main_image_local,
            videos_json,
            bought_together,
            subtitle,
            author
        FROM (
            SELECT
                json_extract_string(j, '$.parent_asin') AS parent_asin,
                json_extract_string(j, '$.main_category') AS main_category,
                json_extract_string(j, '$.title') AS product_title,
                try_cast(json_extract(j, '$.average_rating') AS DOUBLE) AS average_rating,
                try_cast(json_extract(j, '$.rating_number') AS BIGINT) AS rating_number,
                json_extract_string(j, '$.price') AS price,
                json_extract_string(j, '$.store') AS store,
                json_extract(j, '$.categories') AS categories_json,
                json_extract(j, '$.features') AS features_json,
                json_extract(j, '$.description') AS description_json,
                json_extract(j, '$.details') AS details_json,
                json_extract(j, '$.images') AS images_json,
                (
                    SELECT NULLIF(
                        regexp_extract(
                            json_extract_string(e.value, '$.hi_res'),
                            '([^/?#]+)(?:[?#].*)?$',
                            1
                        ),
                        ''
                    )
                    FROM json_each(json_extract(j, '$.images')) AS e
                    WHERE json_extract_string(e.value, '$.variant') = 'MAIN'
                      AND NULLIF(trim(json_extract_string(e.value, '$.hi_res')), '') IS NOT NULL
                    LIMIT 1
                ) AS main_image_local,
                json_extract(j, '$.videos') AS videos_json,
                json_extract_string(j, '$.bought_together') AS bought_together,
                json_extract_string(j, '$.subtitle') AS subtitle,
                json_extract_string(j, '$.author') AS author
            FROM read_json_objects(?) AS t(j)
        ) p
        WHERE product_title IS NOT NULL
          AND features_json IS NOT NULL
          AND description_json IS NOT NULL
          AND details_json IS NOT NULL
          AND images_json IS NOT NULL
          AND length(trim(product_title)) >= 5
          AND length(trim(CAST(features_json AS VARCHAR))) >= 5
          AND length(trim(CAST(description_json AS VARCHAR))) >= 5
          AND length(trim(CAST(details_json AS VARCHAR))) >= 5
          AND length(trim(CAST(images_json AS VARCHAR))) >= 5
          AND lower(CAST(features_json AS VARCHAR)) <> 'null'
          AND lower(CAST(description_json AS VARCHAR)) <> 'null'
          AND lower(CAST(details_json AS VARCHAR)) <> 'null'
          AND lower(CAST(images_json AS VARCHAR)) <> 'null'
          AND NOT (json_type(features_json) = 'ARRAY' AND json_array_length(features_json) = 0)
          AND NOT (json_type(description_json) = 'ARRAY' AND json_array_length(description_json) = 0)
          AND NOT (json_type(images_json) = 'ARRAY' AND json_array_length(images_json) = 0)
          AND main_image_local IS NOT NULL
          AND length(trim(main_image_local)) >= 5
        """,
        [str(meta_jsonl_path)],
    )
    con.execute("""
        CREATE OR REPLACE TABLE products (
            parent_asin VARCHAR,
            main_category VARCHAR,
            product_title VARCHAR NOT NULL,
            average_rating DOUBLE,
            rating_number BIGINT,
            price VARCHAR,
            store VARCHAR,
            categories_json JSON,
            features_json JSON NOT NULL,
            description_json JSON NOT NULL,
            details_json JSON NOT NULL,
            images_json JSON NOT NULL,
            main_image_local VARCHAR,
            videos_json JSON,
            bought_together VARCHAR,
            subtitle VARCHAR,
            author VARCHAR
        )
        """)
    con.execute("INSERT INTO products BY NAME SELECT * FROM products_stage")
    con.execute("DROP TABLE IF EXISTS products_stage")


def create_raw_review_tables(
    con: duckdb.DuckDBPyConnection,
    review_jsonl_path: Path,
) -> None:
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE reviews_stage AS
        SELECT
            user_id,
            asin,
            parent_asin,
            rating,
            review_title,
            review_text,
            timestamp_ms,
            helpful_vote,
            verified_purchase,
            images_json
        FROM (
            SELECT
                json_extract_string(j, '$.user_id') AS user_id,
                json_extract_string(j, '$.asin') AS asin,
                json_extract_string(j, '$.parent_asin') AS parent_asin,
                try_cast(json_extract(j, '$.rating') AS DOUBLE) AS rating,
                json_extract_string(j, '$.title') AS review_title,
                json_extract_string(j, '$.text') AS review_text,
                try_cast(json_extract(j, '$.timestamp') AS BIGINT) AS timestamp_ms,
                try_cast(json_extract(j, '$.helpful_vote') AS BIGINT) AS helpful_vote,
                try_cast(json_extract(j, '$.verified_purchase') AS BOOLEAN) AS verified_purchase,
                json_extract(j, '$.images') AS images_json
            FROM read_json_objects(?) AS t(j)
        ) r
        WHERE NULLIF(trim(review_title), '') IS NOT NULL
          AND NULLIF(trim(review_text), '') IS NOT NULL
        """,
        [str(review_jsonl_path)],
    )
    con.execute("""
        CREATE OR REPLACE TABLE reviews (
            user_id VARCHAR,
            asin VARCHAR,
            parent_asin VARCHAR,
            rating DOUBLE,
            review_title VARCHAR NOT NULL,
            review_text VARCHAR NOT NULL,
            timestamp_ms BIGINT,
            helpful_vote BIGINT,
            verified_purchase BOOLEAN,
            images_json JSON
        )
        """)
    con.execute("INSERT INTO reviews BY NAME SELECT * FROM reviews_stage")
    con.execute("DROP TABLE IF EXISTS reviews_stage")


def create_raw_5core_filtered_tables(
    con: duckdb.DuckDBPyConnection,
    review_jsonl_path: Path,
    core5_csv_path: Path,
) -> None:
    create_raw_review_tables(con, review_jsonl_path)
    con.execute(
        """
        CREATE OR REPLACE TABLE interactions_5core AS
        SELECT
            user_id,
            parent_asin,
            try_cast(rating AS DOUBLE) AS rating,
            try_cast(timestamp AS BIGINT) AS timestamp_ms
        FROM read_csv_auto(?, header=true)
        """,
        [str(core5_csv_path)],
    )

    con.execute("""
        CREATE OR REPLACE TABLE products_filtered (
            parent_asin VARCHAR,
            main_category VARCHAR,
            product_title VARCHAR NOT NULL,
            average_rating DOUBLE,
            rating_number BIGINT,
            price VARCHAR,
            store VARCHAR,
            categories_json JSON,
            features_json JSON NOT NULL,
            description_json JSON NOT NULL,
            details_json JSON NOT NULL,
            images_json JSON NOT NULL,
            main_image_local VARCHAR,
            videos_json JSON,
            bought_together VARCHAR,
            subtitle VARCHAR,
            author VARCHAR
        )
        """)

    con.execute("""
        INSERT INTO products_filtered BY NAME
        SELECT p.*
        FROM products p
        INNER JOIN (
            SELECT DISTINCT parent_asin
            FROM interactions_5core
        ) keep USING (parent_asin)
        """)

    con.execute("""
        CREATE OR REPLACE TABLE reviews_filtered (
            user_id VARCHAR,
            asin VARCHAR,
            parent_asin VARCHAR,
            rating DOUBLE,
            review_title VARCHAR NOT NULL,
            review_text VARCHAR NOT NULL,
            timestamp_ms BIGINT,
            helpful_vote BIGINT,
            verified_purchase BOOLEAN,
            images_json JSON
        )
        """)

    con.execute("""
        INSERT INTO reviews_filtered BY NAME
        SELECT r.*
        FROM reviews r
        INNER JOIN interactions_5core i
            ON r.user_id = i.user_id
           AND r.parent_asin = i.parent_asin
           AND r.rating = i.rating
           AND r.timestamp_ms = i.timestamp_ms
        INNER JOIN products_filtered p
            ON p.parent_asin = r.parent_asin
        """)


def export_tables(
    con: duckdb.DuckDBPyConnection,
    output_dir: Path,
    mode: str,
) -> None:
    ensure_parent(output_dir / "dummy")
    stale_paths = [
        output_dir / "interactions_5core.parquet",
        output_dir / "products_filtered.parquet",
        output_dir / "products.parquet",
        output_dir / "reviews.parquet",
        output_dir / "reviews_filtered.parquet",
        output_dir / "reviews_5core_filtered.parquet",
        output_dir / "reviews_enriched.parquet",
        output_dir / "reviews_filtered_enriched.parquet",
        output_dir / "reviews_5core_filtered_enriched.parquet",
        output_dir / "reviews_filtered_unique.parquet",
        output_dir / "reviews_filtered_unique_enriched.parquet",
        output_dir / "reviews_5core_filtered_unique.parquet",
        output_dir / "reviews_5core_filtered_unique_enriched.parquet",
    ]
    for stale_path in stale_paths:
        if stale_path.exists():
            stale_path.unlink()

    if mode == MODE_RAW:
        products_path_sql = sql_string_literal(str(output_dir / "products.parquet"))
        reviews_path_sql = sql_string_literal(str(output_dir / "reviews.parquet"))
        con.execute(
            f"COPY products TO {products_path_sql} (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        con.execute(
            f"COPY reviews TO {reviews_path_sql} (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
    else:
        products_filtered_path_sql = sql_string_literal(
            str(output_dir / "products_filtered.parquet")
        )
        filtered_path_sql = sql_string_literal(
            str(output_dir / "reviews_filtered.parquet")
        )
        con.execute(
            f"COPY products_filtered TO {products_filtered_path_sql} (FORMAT PARQUET, COMPRESSION ZSTD)",
        )
        con.execute(
            f"COPY reviews_filtered TO {filtered_path_sql} (FORMAT PARQUET, COMPRESSION ZSTD)",
        )


def write_summary_json(
    con: duckdb.DuckDBPyConnection,
    output_path: Path,
    category: str,
    mode: str,
) -> None:
    if mode == MODE_RAW:
        counts = con.execute("""
            SELECT
                COUNT(*) AS review_rows,
                COUNT(DISTINCT r.user_id) AS users,
                COUNT(DISTINCT r.parent_asin) AS reviewed_products,
                SUM(CASE WHEN p.parent_asin IS NOT NULL THEN 1 ELSE 0 END) AS reviews_with_metadata
            FROM reviews r
            LEFT JOIN products p USING (parent_asin)
            """).fetchone()
        payload = {
            "category": category,
            "mode": mode,
            "review_rows": counts[0],
            "users": counts[1],
            "reviewed_products": counts[2],
            "reviews_with_metadata": counts[3],
        }
    else:
        counts = con.execute("""
            SELECT
                (SELECT COUNT(*) FROM interactions_5core) AS interaction_rows_5core,
                (SELECT COUNT(*) FROM reviews) AS raw_review_rows,
                (SELECT COUNT(*) FROM reviews_filtered) AS filtered_review_rows,
                (SELECT COUNT(DISTINCT user_id) FROM reviews_filtered) AS users,
                (SELECT COUNT(DISTINCT parent_asin) FROM reviews_filtered) AS reviewed_products,
                (
                    SELECT SUM(CASE WHEN p.parent_asin IS NOT NULL THEN 1 ELSE 0 END)
                    FROM reviews_filtered r
                    LEFT JOIN products_filtered p USING (parent_asin)
                ) AS reviews_with_metadata
            """).fetchone()
        reduction_pct = 0.0
        if counts[1]:
            reduction_pct = (1 - (counts[2] / counts[1])) * 100.0
        payload = {
            "category": category,
            "mode": mode,
            "interaction_rows_5core": counts[0],
            "raw_review_rows": counts[1],
            "filtered_review_rows": counts[2],
            "row_reduction_percent": round(reduction_pct, 4),
            "users": counts[3],
            "reviewed_products": counts[4],
            "reviews_with_metadata": counts[5],
        }

    ensure_parent(output_path)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_setup(
    base_dir: Path,
    category: str,
    mode: str,
    force_download: bool,
) -> None:
    print(f"\n[setup] preparing category={category}, mode={mode}")
    if mode == MODE_RAW_5CORE:
        print(
            "[info] raw_5core mode keeps review text but filters rows to the 5-core interaction subset."
        )

    paths = download_inputs(
        base_dir=base_dir,
        category=category,
        mode=mode,
        force_download=force_download,
    )

    run_dir = base_dir / "processed" / f"{category}__{mode}"
    run_dir.mkdir(parents=True, exist_ok=True)
    db_path = run_dir / "amazon_reviews.duckdb"

    print(f"[info] building DuckDB dataset: {db_path}")
    con = duckdb.connect(str(db_path))

    meta_path = paths[f"raw/meta_categories/meta_{category}.jsonl"]

    create_products_table(con, meta_path)

    if mode == MODE_RAW:
        review_path = paths[f"raw/review_categories/{category}.jsonl"]
        create_raw_review_tables(con, review_path)
    else:
        review_path = paths[f"raw/review_categories/{category}.jsonl"]
        core5_path = paths[f"benchmark/5core/rating_only/{category}.csv"]
        create_raw_5core_filtered_tables(con, review_path, core5_path)

    export_tables(con, run_dir, mode=mode)
    write_summary_json(
        con,
        run_dir / "summary.json",
        category=category,
        mode=mode,
    )

    if mode == MODE_RAW_5CORE:
        con.execute("DROP TABLE IF EXISTS interactions_5core")
        con.execute("DROP TABLE IF EXISTS reviews")
        con.execute("DROP TABLE IF EXISTS products")
        download_images_from_products_filtered(con, run_dir / "images")

    con.close()

    print(f"[ok] finished setup in {run_dir}")


def main() -> int:
    args = parse_args()
    repo_root = SCRIPT_DIR.parents[1]
    default_data_dir = SCRIPT_DIR

    if args.base_dir is None:
        base_dir = default_data_dir.resolve()
    else:
        base_dir_arg = Path(args.base_dir)
        if base_dir_arg.is_absolute():
            base_dir = base_dir_arg
        else:
            base_dir = (repo_root / base_dir_arg).resolve()

    tree_cache = base_dir / "cache" / "hf_tree_main.json"
    fetch_tree_json(tree_cache, force_refresh=args.force_refresh_tree)

    con = duckdb.connect()
    load_tree_table(con, tree_cache)

    report_csv = base_dir / "reports" / "category_size_report.csv"
    write_category_report(con, report_csv)
    print(f"[ok] wrote category size report: {report_csv}")

    analyze_categories(con, top_n=args.top_n)

    if args.analyze_only:
        con.close()
        print("\n[done] analysis only")
        return 0

    if not args.category:
        con.close()
        print("[error] --category is required unless --analyze-only is used.")
        return 2

    try:
        category = choose_category(con, args.category, args.mode)
    except ValueError as err:
        con.close()
        print(f"[error] {err}")
        return 2

    con.close()

    run_setup(
        base_dir=base_dir,
        category=category,
        mode=args.mode,
        force_download=args.force_download,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
