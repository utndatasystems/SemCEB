from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from queries.template_parser import QueryTemplate, QueryTemplateParser


@dataclass(frozen=True)
class QuerySpecification:
    id: int
    category: str
    datasets: list[str]
    filter: str
    filter_parsed: QueryTemplate

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QuerySpecification":
        raw_filter = data["filter"]

        return cls(
            id=data["id"],
            category=data["category"],
            datasets=data["datasets"],
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