from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import re

from src.semceb.queries.template_parser import QueryTemplate, QueryTemplateParser

@dataclass(frozen=True)
class DatasetSpecification:
    alias: str
    table_ref: str

    @classmethod
    def from_str(cls, s: str) -> "DatasetSpecification":
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
        return {
            "alias": self.alias,
            "table_ref": self.table_ref,
        }


@dataclass(frozen=True)
class QuerySpecification:
    id: int
    category: str
    datasets: list[DatasetSpecification]
    filter: str
    filter_parsed: QueryTemplate

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QuerySpecification":
        raw_filter = data["filter"]

        return cls(
            id=data["id"],
            category=data["category"],
            datasets=[DatasetSpecification.from_str(dataset) for dataset in data["datasets"]],
            filter=raw_filter,
            filter_parsed=QueryTemplateParser.parse(raw_filter),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "datasets": self.datasets,
            "filter": self.filter,
            "filter_parsed": self.filter_parsed.to_dict(),
        }