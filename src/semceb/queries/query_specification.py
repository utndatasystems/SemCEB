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
    table_ref: str

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
    type: str
    category: str
    datasets: list[DatasetSpecification]
    filter: str
    embeddings: dict[str, list[float]]
    filter_parsed: QueryTemplate

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
            datasets=[DatasetSpecification.from_str(dataset) for dataset in data["datasets"]],
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
