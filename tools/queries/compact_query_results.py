from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import duckdb


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from semceb.queries.query_specification import QuerySpecification  # noqa: E402


QUERY_RESULT_FILENAME_PATTERN = re.compile(r"_q(?P<query_id>\d+)\.parquet$")
PRODUCTS_TABLE_REF = "amazon-reviews/products_filtered_with_embeddings"
REVIEWS_TABLE_REF = "amazon-reviews/reviews_filtered_with_embeddings"
PRODUCT_ID_COLUMNS = ("parent_asin",)
REVIEW_ID_COLUMNS = ("asin", "user_id", "parent_asin", "timestamp_ms")


class QueryResultCompactionError(ValueError):
    pass


@dataclass(frozen=True)
class CompactionResult:
    path: Path
    status: str
    rows_before: int
    rows_after: int
    columns_before: int
    columns_after: int
    size_before: int
    size_after: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compact persisted query-result parquet files to ID columns."
    )
    parser.add_argument(
        "--query-results-dir",
        type=Path,
        default=Path("results") / "raw" / "query_results",
        help="Directory containing query-result parquet files.",
    )
    parser.add_argument(
        "--queries-file",
        type=Path,
        default=Path("benchmark_queries") / "queries.jsonl",
        help="JSONL file containing query specifications.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned changes without rewriting files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    query_specs = load_query_specs(args.queries_file)
    parquet_paths = sorted(args.query_results_dir.glob("*.parquet"))

    if not parquet_paths:
        raise SystemExit(f"No parquet files found in {args.query_results_dir}")

    con = duckdb.connect()
    results: list[CompactionResult] = []

    for path in parquet_paths:
        query_id = parse_query_id(path)
        query_spec = query_specs.get(query_id)

        if query_spec is None:
            raise QueryResultCompactionError(
                f"No query specification found for {path.name} with q{query_id}."
            )

        results.append(
            compact_one_file(
                con=con,
                path=path,
                query_spec=query_spec,
                dry_run=args.dry_run,
            )
        )

    print_summary(results=results, dry_run=args.dry_run)


def load_query_specs(path: Path) -> dict[int, QuerySpecification]:
    query_specs: dict[int, QuerySpecification] = {}

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            query_spec = QuerySpecification.from_dict(json.loads(line))
            query_specs[query_spec.id] = query_spec

    return query_specs


def parse_query_id(path: Path) -> int:
    match = QUERY_RESULT_FILENAME_PATTERN.search(path.name)

    if match is None:
        raise QueryResultCompactionError(
            f"Could not parse query ID from query-result filename: {path.name}"
        )

    return int(match.group("query_id"))


def compact_result_columns_for_query(
    query_spec: QuerySpecification,
    result_columns: list[str],
) -> list[str]:
    if len(query_spec.datasets) == 1:
        dataset_spec = query_spec.datasets[0]
        id_columns = id_columns_for_table_ref(dataset_spec.table_ref)
        return resolve_filter_result_columns(
            id_columns=id_columns,
            dataset_alias=dataset_spec.alias,
            result_columns=result_columns,
        )

    if len(query_spec.datasets) == 2:
        compact_columns: list[str] = []

        for dataset_spec in query_spec.datasets:
            for id_column in id_columns_for_table_ref(dataset_spec.table_ref):
                compact_columns.append(
                    require_column(
                        column_name=f"{dataset_spec.alias}.{id_column}",
                        result_columns=result_columns,
                        query_id=query_spec.id,
                    )
                )

        return compact_columns

    raise QueryResultCompactionError(
        f"Query {query_spec.id} uses {len(query_spec.datasets)} datasets; "
        "only filters and two-table joins are supported."
    )


def id_columns_for_table_ref(table_ref: str) -> tuple[str, ...]:
    if table_ref == PRODUCTS_TABLE_REF:
        return PRODUCT_ID_COLUMNS

    if table_ref == REVIEWS_TABLE_REF:
        return REVIEW_ID_COLUMNS

    raise QueryResultCompactionError(
        f"Unsupported dataset table reference for compact query results: {table_ref!r}"
    )


def resolve_filter_result_columns(
    id_columns: tuple[str, ...],
    dataset_alias: str,
    result_columns: list[str],
) -> list[str]:
    compact_columns: list[str] = []

    for id_column in id_columns:
        if id_column in result_columns:
            compact_columns.append(id_column)
            continue

        aliased_column = f"{dataset_alias}.{id_column}"
        if aliased_column in result_columns:
            compact_columns.append(aliased_column)
            continue

        raise QueryResultCompactionError(
            f"Missing ID column {id_column!r} for compact filter result. "
            f"Available columns are: {result_columns}."
        )

    return compact_columns


def require_column(
    column_name: str,
    result_columns: list[str],
    query_id: int,
) -> str:
    if column_name in result_columns:
        return column_name

    raise QueryResultCompactionError(
        f"Missing ID column {column_name!r} for compact query result q{query_id}. "
        f"Available columns are: {result_columns}."
    )


def compact_one_file(
    con: duckdb.DuckDBPyConnection,
    path: Path,
    query_spec: QuerySpecification,
    dry_run: bool,
) -> CompactionResult:
    columns_before = load_columns(con=con, path=path)
    compact_columns = compact_result_columns_for_query(
        query_spec=query_spec,
        result_columns=columns_before,
    )
    rows_before = row_count(con=con, path=path)
    size_before = path.stat().st_size

    if columns_before == compact_columns:
        return CompactionResult(
            path=path,
            status="skipped",
            rows_before=rows_before,
            rows_after=rows_before,
            columns_before=len(columns_before),
            columns_after=len(compact_columns),
            size_before=size_before,
            size_after=size_before,
        )

    if dry_run:
        return CompactionResult(
            path=path,
            status="planned",
            rows_before=rows_before,
            rows_after=rows_before,
            columns_before=len(columns_before),
            columns_after=len(compact_columns),
            size_before=size_before,
            size_after=size_before,
        )

    tmp_path = path.with_name(f"{path.name}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    copy_compacted_parquet(
        con=con,
        source_path=path,
        target_path=tmp_path,
        columns=compact_columns,
    )
    rows_after = row_count(con=con, path=tmp_path)

    if rows_after != rows_before:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Compaction changed row count for {path}: "
            f"{rows_before} before, {rows_after} after."
        )

    tmp_path.replace(path)
    size_after = path.stat().st_size

    return CompactionResult(
        path=path,
        status="compacted",
        rows_before=rows_before,
        rows_after=rows_after,
        columns_before=len(columns_before),
        columns_after=len(compact_columns),
        size_before=size_before,
        size_after=size_after,
    )


def load_columns(con: duckdb.DuckDBPyConnection, path: Path) -> list[str]:
    rows = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet({sql_string_literal(str(path))})"
    ).fetchall()
    return [row[0] for row in rows]


def row_count(con: duckdb.DuckDBPyConnection, path: Path) -> int:
    return int(
        con.execute(
            f"SELECT count(*) FROM read_parquet({sql_string_literal(str(path))})"
        ).fetchone()[0]
    )


def copy_compacted_parquet(
    con: duckdb.DuckDBPyConnection,
    source_path: Path,
    target_path: Path,
    columns: list[str],
) -> None:
    select_list = ", ".join(quote_identifier(column) for column in columns)
    con.execute(
        "COPY ("
        f"SELECT {select_list} "
        f"FROM read_parquet({sql_string_literal(str(source_path))})"
        f") TO {sql_string_literal(str(target_path))} "
        "(FORMAT PARQUET, COMPRESSION ZSTD)"
    )


def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def sql_string_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def print_summary(results: list[CompactionResult], dry_run: bool) -> None:
    status_counts: dict[str, int] = {}

    for result in results:
        status_counts[result.status] = status_counts.get(result.status, 0) + 1

    size_before = sum(result.size_before for result in results)
    size_after = sum(result.size_after for result in results)
    rows_before = sum(result.rows_before for result in results)
    rows_after = sum(result.rows_after for result in results)

    print("Query-result parquet compaction")
    print(f"  mode: {'dry-run' if dry_run else 'rewrite'}")
    print(f"  files: {len(results)}")
    print(f"  statuses: {status_counts}")
    print(f"  rows: {rows_before:,} -> {rows_after:,}")
    print(f"  size: {format_bytes(size_before)} -> {format_bytes(size_after)}")

    for result in results:
        if result.status == "skipped":
            continue

        print(
            f"  {result.status}: {result.path} "
            f"columns={result.columns_before}->{result.columns_after} "
            f"rows={result.rows_before:,}->{result.rows_after:,} "
            f"size={format_bytes(result.size_before)}->{format_bytes(result.size_after)}"
        )


def format_bytes(size: int) -> str:
    value = float(size)
    units = ["B", "KiB", "MiB", "GiB", "TiB"]

    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024

    raise AssertionError("unreachable")


if __name__ == "__main__":
    main()
