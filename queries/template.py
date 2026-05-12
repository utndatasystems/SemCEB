from itertools import product
from typing import Any


class QueryTemplate:
    """Template for query generation."""

    def __init__(
        self,
        name: str,
        version: str,
        dataset: str,
        column: str,
        query_structure: str,
        description: str | None = None,
        variables: dict[str, list[Any]] | None = None,
    ):
        self.name = name
        self.version = version
        self.dataset = dataset
        self.column = column
        self.query_structure = query_structure
        self.description = description
        self.variables = variables or {}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QueryTemplate":
        """Create a QueryTemplate from one TOML template entry."""

        fixed_fields = {
            "name",
            "version",
            "dataset",
            "column",
            "query_structure",
            "description",
        }

        variables = {}

        for key, value in data.items():
            if key not in fixed_fields:
                variables[key] = value

        return cls(
            name=data["name"],
            version=data["version"],
            dataset=data["dataset"],
            column=data["column"],
            query_structure=data["query_structure"],
            description=data.get("description"),
            variables=variables,
        )

    def generate_all_queries(self) -> list[str]:
        """Generate all possible queries from this template."""

        if not self.variables:
            return [self.query_structure]

        keys = sorted(self.variables.keys())
        values = [self.variables[key] for key in keys]

        queries = []

        for combination in product(*values):
            replacements = dict(zip(keys, combination))
            query = self.query_structure.format(**replacements)
            queries.append(query)

        return sorted(queries)
