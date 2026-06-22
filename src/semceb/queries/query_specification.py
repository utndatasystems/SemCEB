from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import re

from semceb.queries.template_parser import QueryTemplate, QueryTemplateParser

QUERY_SPEC_RESERVED_KEYS = {
    "id",
    "type",
    "category",
    "datasets",
    "filter",
    "filter_parsed",
    "embeddings",
}


@dataclass(frozen=True)
class DatasetSpecification:
    """Represent one dataset reference and alias in a query specification."""

    alias: str
    """Alias of the table reference, e.g., "reviews AS r2" -> alias="r2"."""
    table_ref: str
    """Source table reference, e.g., "reviews AS r2" -> table_ref="reviews"."""

    @classmethod
    def from_str(cls, s: str) -> "DatasetSpecification":
        """Parse a dataset definition from a raw string, supporting optional aliases."""
        parts = re.split(r"\s+as\s+", s.strip(), maxsplit=1, flags=re.IGNORECASE)

        if len(parts) == 1:
            table_ref = parts[0].strip()
            return cls(table_ref, table_ref)

        table_ref, alias = parts[0].strip(), parts[1].strip()

        if not table_ref:
            raise ValueError(f"Missing table reference in dataset raw string: {s!r}")

        if not alias:
            raise ValueError(f"Missing alias in dataset raw string: {s!r}")

        return cls(alias, table_ref)

    def to_dict(self) -> dict:
        """Convert the dataset specification to a JSON-serializable dictionary."""
        return {
            "alias": self.alias,
            "table_ref": self.table_ref,
        }

    def to_query_string(self) -> str:
        """Serialize the dataset specification back to the JSONL query format."""
        if self.alias == self.table_ref:
            return self.table_ref

        return f"{self.table_ref} as {self.alias}"


@dataclass(frozen=True)
class QuerySpecification:
    """Represent a benchmark query along with parsed filter metadata."""

    id: int
    """Stable numeric query identifier."""
    type: str
    """Benchmark-specific query type label."""
    category: list[str]
    """Benchmark category label used for grouping queries."""
    datasets: list[DatasetSpecification]
    """Dataset references used by the query, including aliases."""
    filter: str
    """Raw filter template string as stored in the query source."""
    filter_parsed: QueryTemplate
    """Parsed representation of ``filter`` for downstream processing. Contains parsed column references.
    The predicate string is split into fixed string parts and column references, stored together in a list
    in the origianl order, which allows processing and re-assembling."""
    embeddings: dict[str, list[float]]
    """Embedding representations of the filter predicate with different models (keys of the dictionary)."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QuerySpecification":
        """Construct a QuerySpecification from a dictionary loaded from JSON."""
        raw_filter = data["filter"]
        raw_embeddings = data.get("embeddings")

        if raw_embeddings is None:
            raw_embeddings = {
                key: value
                for key, value in data.items()
                if key not in QUERY_SPEC_RESERVED_KEYS
            }
        elif not isinstance(raw_embeddings, dict):
            raise ValueError(
                f"Expected 'embeddings' to be an object for query id={data.get('id')!r}"
            )

        return cls(
            id=data["id"],
            type=data["type"],
            category=data["category"],
            datasets=[
                DatasetSpecification.from_str(dataset) for dataset in data["datasets"]
            ],
            filter=raw_filter,
            embeddings=raw_embeddings,
            filter_parsed=QueryTemplateParser.parse(raw_filter),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full query specification to a dictionary for JSON export."""
        return {
            "id": self.id,
            "type": self.type,
            "category": self.category,
            "datasets": [dataset.to_query_string() for dataset in self.datasets],
            "filter": self.filter,
            "filter_parsed": self.filter_parsed.to_dict(),
        }
