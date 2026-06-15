Amazon Reviews 2023 data preparation
====================================

This directory contains the local preparation pipeline for
`McAuley-Lab/Amazon-Reviews-2023` (MIT license). The default benchmark dataset
is the `Arts_Crafts_and_Sewing` category in `raw_5core` mode: raw review text and
metadata are retained, but rows are restricted to the 5-core interaction set.

The generated data is intentionally not tracked in git. By default, scripts use:

```text
tools/amazon-reviews/
  raw/        # downloaded Hugging Face source files
  cache/      # cached Hugging Face tree metadata and local helper caches
  reports/    # category size report
  processed/  # prepared dataset runs
```

## TL;DR Dataset Repro

Reproduces the dataset that is used in the SemCEB paper.

Note: This requires significant computing resources and time. It is better to just download the prepared dataset from S3 by running `semceb run`.

```bash
python tools/amazon-reviews/download_and_prepare_amazon_reviews_dataset.py --category Arts_Crafts_and_Sewing  --mode raw_5core

python tools/amazon-reviews/compute_embeddings.py --data-dir Arts_Crafts_and_Sewing__raw_5core

cd tools/amazon-reviews/
AWS_ACCESS_KEY_ID=xxxxx AWS_SECRET_ACCESS_KEY=xxxxx ./upload_data_to_s3.sh
```

## Workflow

Analyze category footprint:

```bash
python tools/amazon-reviews/download_and_prepare_amazon_reviews_dataset.py --analyze-only
```

Prepare the default benchmark dataset:

```bash
python tools/amazon-reviews/download_and_prepare_amazon_reviews_dataset.py \
  --category Arts_Crafts_and_Sewing \
  --mode raw_5core
```

Compute embeddings:

```bash
python tools/amazon-reviews/compute_embeddings.py \
  --data-dir Arts_Crafts_and_Sewing__raw_5core
```

If embedded parquet files already exist but the local embedding cache is missing,
seed the cache from those parquet files without recomputing existing embeddings:

```bash
python tools/amazon-reviews/import_embeddings_into_cache.py \
  --data-dir Arts_Crafts_and_Sewing__raw_5core
```

`--data-dir` accepts either an absolute path, a path relative to the repository
root, or a short processed-run name under `tools/amazon-reviews/processed/`.

Copy the prepared benchmark dataset into the runtime dataset directory:

```bash
python tools/amazon-reviews/copy_prepared_data_to_runtime.py
```

This overwrites the local runtime amazon data in:`data/datasets/` with the prepared dataset from `tools/amazon-reviews/processed/Arts_Crafts_and_Sewing__raw_5core/`

## Preparation

`download_and_prepare_amazon_reviews_dataset.py` downloads the required Amazon
Reviews files from Hugging Face, mass-downloads product images for retained
products, and builds a DuckDB-backed processed run.

Supported modes:

- `raw`: writes `products.parquet` and `reviews.parquet` after the metadata,
  image, and non-empty review text filters, without 5-core restriction.
- `raw_5core`: writes `products_filtered.parquet` and `reviews_filtered.parquet`
  after applying the same filters and restricting products/reviews to the 5-core
  interaction set.

For `raw_5core`, the output directory is:

```text
tools/amazon-reviews/processed/Arts_Crafts_and_Sewing__raw_5core/
```

Important preparation rules:

- Product metadata must have non-empty `product_title`, `features_json`,
  `description_json`, `details_json`, and `images_json`.
- Products must have a non-empty MAIN `hi_res` image URL. The script downloads
  these images into `images/`, mirroring URL host/path to avoid filename
  collisions.
- Reviews must have non-empty `review_title` and `review_text`.
- In `raw_5core`, reviews are matched to 5-core interactions by
  `user_id`, `parent_asin`, `rating`, and `timestamp_ms`, then restricted to
  products retained after metadata/image filtering.

For the current `Arts_Crafts_and_Sewing__raw_5core` run, the local parquet files
contain 45,693 products and 936,216 filtered reviews.

Core prepared artifacts:

```text
amazon_reviews.duckdb
summary.json
products_filtered.parquet
reviews_filtered.parquet
images/
```

## Embeddings

`compute_embeddings.py` reads a processed `raw_5core` run, requires
`amazon_reviews.duckdb` and `images/`, and writes embedded parquet files without
modifying the original filtered parquet files.

A machine with a GPU is recommended. The final parquet export is memory-heavy;
for the current dataset, use a machine with more than 100 GB RAM.

Output artifacts:

```text
products_filtered_with_embeddings.parquet
reviews_filtered_with_embeddings.parquet
embedding_cache.sqlite3
```

## Embedded table schemas

Current schemas from DuckDB `DESCRIBE` on the embedded parquet files.

### `products_filtered_with_embeddings.parquet`

Base columns:

| Column | Type |
| --- | --- |
| `parent_asin` | `VARCHAR` |
| `main_category` | `VARCHAR` |
| `product_title` | `VARCHAR` |
| `average_rating` | `DOUBLE` |
| `rating_number` | `BIGINT` |
| `price` | `VARCHAR` |
| `store` | `VARCHAR` |
| `categories_json` | `JSON` |
| `features_json` | `JSON` |
| `description_json` | `JSON` |
| `details_json` | `JSON` |
| `images_json` | `JSON` |
| `main_image_local` | `VARCHAR` |
| `videos_json` | `JSON` |
| `bought_together` | `VARCHAR` |
| `subtitle` | `VARCHAR` |
| `author` | `VARCHAR` |

Embedding columns:

| Source | Model | Embedding column | Type | Truncation flag column |
| --- | --- | --- | --- | --- |
| `main_image_local` | `google/siglip2-base-patch16-224` | `main_image_local_embeddings_google_siglip2_base_patch16_224` | `DOUBLE[]` | `main_image_local_embeddings_google_siglip2_base_patch16_224_input_is_truncated` |
| `product_title` | `Qwen/Qwen3-Embedding-0.6B` | `product_title_embeddings_qwen_qwen3_embedding_0_6b` | `DOUBLE[]` | `product_title_embeddings_qwen_qwen3_embedding_0_6b_input_is_truncated` |
| `description_json` | `Qwen/Qwen3-Embedding-0.6B` | `description_json_embeddings_qwen_qwen3_embedding_0_6b` | `DOUBLE[]` | `description_json_embeddings_qwen_qwen3_embedding_0_6b_input_is_truncated` |
| `features_json` | `Qwen/Qwen3-Embedding-0.6B` | `features_json_embeddings_qwen_qwen3_embedding_0_6b` | `DOUBLE[]` | `features_json_embeddings_qwen_qwen3_embedding_0_6b_input_is_truncated` |
| `details_json` | `Qwen/Qwen3-Embedding-0.6B` | `details_json_embeddings_qwen_qwen3_embedding_0_6b` | `DOUBLE[]` | `details_json_embeddings_qwen_qwen3_embedding_0_6b_input_is_truncated` |
| `product_title` | `google/siglip2-base-patch16-224` | `product_title_embeddings_google_siglip2_base_patch16_224` | `DOUBLE[]` | `product_title_embeddings_google_siglip2_base_patch16_224_input_is_truncated` |
| `description_json` | `google/siglip2-base-patch16-224` | `description_json_embeddings_google_siglip2_base_patch16_224` | `DOUBLE[]` | `description_json_embeddings_google_siglip2_base_patch16_224_input_is_truncated` |
| `features_json` | `google/siglip2-base-patch16-224` | `features_json_embeddings_google_siglip2_base_patch16_224` | `DOUBLE[]` | `features_json_embeddings_google_siglip2_base_patch16_224_input_is_truncated` |
| `details_json` | `google/siglip2-base-patch16-224` | `details_json_embeddings_google_siglip2_base_patch16_224` | `DOUBLE[]` | `details_json_embeddings_google_siglip2_base_patch16_224_input_is_truncated` |

### `reviews_filtered_with_embeddings.parquet`

Base columns:

| Column | Type |
| --- | --- |
| `user_id` | `VARCHAR` |
| `asin` | `VARCHAR` |
| `parent_asin` | `VARCHAR` |
| `rating` | `DOUBLE` |
| `review_title` | `VARCHAR` |
| `review_text` | `VARCHAR` |
| `timestamp_ms` | `BIGINT` |
| `helpful_vote` | `BIGINT` |
| `verified_purchase` | `BOOLEAN` |
| `images_json` | `JSON` |

Embedding columns:

| Source | Model | Embedding column | Type | Truncation flag column |
| --- | --- | --- | --- | --- |
| `review_title` | `Qwen/Qwen3-Embedding-0.6B` | `review_title_embeddings_qwen_qwen3_embedding_0_6b` | `DOUBLE[]` | `review_title_embeddings_qwen_qwen3_embedding_0_6b_input_is_truncated` |
| `review_text` | `Qwen/Qwen3-Embedding-0.6B` | `review_text_embeddings_qwen_qwen3_embedding_0_6b` | `DOUBLE[]` | `review_text_embeddings_qwen_qwen3_embedding_0_6b_input_is_truncated` |
| `review_title` | `google/siglip2-base-patch16-224` | `review_title_embeddings_google_siglip2_base_patch16_224` | `DOUBLE[]` | `review_title_embeddings_google_siglip2_base_patch16_224_input_is_truncated` |
| `review_text` | `google/siglip2-base-patch16-224` | `review_text_embeddings_google_siglip2_base_patch16_224` | `DOUBLE[]` | `review_text_embeddings_google_siglip2_base_patch16_224_input_is_truncated` |

Models:

- Images: `google/siglip2-base-patch16-224` on `main_image_local`.
- Product text: `Qwen/Qwen3-Embedding-0.6B` and
  `google/siglip2-base-patch16-224` on `product_title`, `description_json`,
  `features_json`, and `details_json`.
- Review text: `Qwen/Qwen3-Embedding-0.6B` and
  `google/siglip2-base-patch16-224` on `review_title` and `review_text`.

Embedding column names follow:

```text
<source_column>_embeddings_<sanitized_model_name>
```

Each embedding column has a sibling boolean column:

```text
<embedding_column>_input_is_truncated
```

Computing embeddings might be expensive. This script therefore employs a local
embedding cache using SQLite.
The embedding cache stores completed vectors keyed by table, source row key, and
embedding column. It allows interrupted embedding runs to resume. It also allows
backfilling the cache from already embedded parquet files via
`import_embeddings_into_cache.py`, which is useful when adding new embedding
columns without recomputing old ones. The cache is a performance artifact; the
canonical exported datasets are the parquet files.

## S3 upload

`upload_data_to_s3.sh` uploads the prepared benchmark artifacts for
`Arts_Crafts_and_Sewing__raw_5core` to:

```text
s3://azimmerer-semceb-datasets/amazon-reviews/
```

It requires configured AWS CLI credentials and `zip`. Before uploading, it
creates `images.zip` from the local image directory and
`embedding_cache.sqlite3.zip` from the SQLite embedding cache. It uploads both
archives plus the filtered and embedded parquet files.
