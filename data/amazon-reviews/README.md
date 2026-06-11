Amazon-Reviews-2023
=====

The benchmark uses `McAuley-Lab/Amazon-Reviews-2023` (MIT license).
The whole dataset is large, which would exceed the current capabilities of modern semantic operators.

Therefore, we apply the following transformations:

 - Only the category `Arts_Crafts_and_Sewing` is used.
 - The `products` table is filtered such that the following columns do not contain `NULL` values and all values must have at least 5 characters.
   - `product_title IS NOT NULL`
   - `features_json IS NOT NULL`
   - `description_json IS NOT NULL`
   - `details_json IS NOT NULL`
   - `images_json IS NOT NULL`
 - Further, `products` are filtered such that the following columns are neither `NULL` nor contain empty JSON array/structs:
   - `features_json`
   - `description_json`
   - `details_json`
   - `images_json`
 - Lastly, every product is associated with multiple images. For this dataset, we download only the `hi_res` version of the `"variant":"MAIN"` image such that we end up with exactly one image per product. The image file name is stored in the newly created `main_image_local` column. Further, products without such an image are removed such that every product is guaranteed to have one image.
 - The `reviews` table is filtered such that:
   - `review_title` is neither NULL nor empty.
   - `review_text` is neither NULL nor empty.
 - `5core` filtering is applied: both `products` and `reviews` are filtered such that only products/reviews are contained that appear in the `5core` interactions dataset.

In the end, two tables are created:
 - `products_filtered`: ~45k rows
 - `reviews_filtered`: ~940k rows


## Setup

To generate the dataset, run the following command:

```
python download_and_prepare_amazon_reviews_dataset.py --category Arts_Crafts_and_Sewing --mode raw_5core
```

To compute image and text embeddings for the filtered product and review tables, run:

```
python compute_embeddings.py --run-dir processed/Arts_Crafts_and_Sewing__raw_5core
```

This uses `google/siglip2-base-patch16-224` for images and `Qwen/Qwen3-Embedding-0.6B`
for the textual product and review columns, auto-selects the available device, and writes:

```
products_filtered_with_embeddings.parquet
reviews_filtered_with_embeddings.parquet
```

The original `products_filtered.parquet` and `reviews_filtered.parquet` remain untouched.



## Schema

`products_filtered`:

```
┌──────────────────┬─────────────┬─────────┬─────────┬─────────┬─────────┐
│   column_name    │ column_type │  null   │   key   │ default │  extra  │
│     varchar      │   varchar   │ varchar │ varchar │ varchar │ varchar │
├──────────────────┼─────────────┼─────────┼─────────┼─────────┼─────────┤
│ parent_asin      │ VARCHAR     │ YES     │ NULL    │ NULL    │ NULL    │
│ main_category    │ VARCHAR     │ YES     │ NULL    │ NULL    │ NULL    │
│ product_title    │ VARCHAR     │ NO      │ NULL    │ NULL    │ NULL    │
│ average_rating   │ DOUBLE      │ YES     │ NULL    │ NULL    │ NULL    │
│ rating_number    │ BIGINT      │ YES     │ NULL    │ NULL    │ NULL    │
│ price            │ VARCHAR     │ YES     │ NULL    │ NULL    │ NULL    │
│ store            │ VARCHAR     │ YES     │ NULL    │ NULL    │ NULL    │
│ categories_json  │ JSON        │ YES     │ NULL    │ NULL    │ NULL    │
│ features_json    │ JSON        │ NO      │ NULL    │ NULL    │ NULL    │
│ description_json │ JSON        │ NO      │ NULL    │ NULL    │ NULL    │
│ details_json     │ JSON        │ NO      │ NULL    │ NULL    │ NULL    │
│ images_json      │ JSON        │ NO      │ NULL    │ NULL    │ NULL    │
│ main_image_local │ VARCHAR     │ YES     │ NULL    │ NULL    │ NULL    │
│ videos_json      │ JSON        │ YES     │ NULL    │ NULL    │ NULL    │
│ bought_together  │ VARCHAR     │ YES     │ NULL    │ NULL    │ NULL    │
│ subtitle         │ VARCHAR     │ YES     │ NULL    │ NULL    │ NULL    │
│ author           │ VARCHAR     │ YES     │ NULL    │ NULL    │ NULL    │
├──────────────────┴─────────────┴─────────┴─────────┴─────────┴─────────┤
│ 17 rows                                                      6 columns │
└────────────────────────────────────────────────────────────────────────┘
```

`reviews_filtered`:

```
┌───────────────────┬─────────────┬─────────┬─────────┬─────────┬─────────┐
│    column_name    │ column_type │  null   │   key   │ default │  extra  │
│      varchar      │   varchar   │ varchar │ varchar │ varchar │ varchar │
├───────────────────┼─────────────┼─────────┼─────────┼─────────┼─────────┤
│ user_id           │ VARCHAR     │ YES     │ NULL    │ NULL    │ NULL    │
│ asin              │ VARCHAR     │ YES     │ NULL    │ NULL    │ NULL    │
│ parent_asin       │ VARCHAR     │ YES     │ NULL    │ NULL    │ NULL    │
│ rating            │ DOUBLE      │ YES     │ NULL    │ NULL    │ NULL    │
│ review_title      │ VARCHAR     │ NO      │ NULL    │ NULL    │ NULL    │
│ review_text       │ VARCHAR     │ NO      │ NULL    │ NULL    │ NULL    │
│ timestamp_ms      │ BIGINT      │ YES     │ NULL    │ NULL    │ NULL    │
│ helpful_vote      │ BIGINT      │ YES     │ NULL    │ NULL    │ NULL    │
│ verified_purchase │ BOOLEAN     │ YES     │ NULL    │ NULL    │ NULL    │
│ images_json       │ JSON        │ YES     │ NULL    │ NULL    │ NULL    │
├───────────────────┴─────────────┴─────────┴─────────┴─────────┴─────────┤
│ 10 rows                                                       6 columns │
└─────────────────────────────────────────────────────────────────────────┘
```
