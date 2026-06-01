Amazon-Reviews-2023
=====

The benchmark uses `McAuley-Lab/Amazon-Reviews-2023` (MIT license).
The whole dataset is large, which would exceed the current capabilities of modern semantic operators.

Therefore, we apply the following transformations:

 - Only the category `Arts_Crafts_and_Sewing` is used.
 - The `products` table is filtered such that the following columns do not contain `NULL` values.
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
 - The `reviews` table is filtered such that:
   - `review_title` is neither NULL nor empty.
   - `review_text` is neither NULL nor empty.
 - `5core` filtering is applied: both `products` and `reviews` are filtered such that only products/reviews are contained that appear in the `5core` interactions dataset.

In the end, two tables are created:
 - `products_filtered`: ~50k rows
 - `reviews_5core_filtered`: ~1.8m rows

