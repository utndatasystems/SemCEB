import json
import tomllib
from pathlib import Path
from rich.console import Console

from queries.template import QueryTemplate


class QueryGenerator:
    """Generates queries for benchmark"""

    def __init__(self):
        self.next_id = 0
        self.templates = self._load_templates()

    def _load_templates(self):
        """Loads templates"""

        with open(r"queries\templates.toml", "rb") as file:
            data = tomllib.load(file)

        templates = {}

        for template_data in data["query_templates"]:
            template = QueryTemplate.from_dict(template_data)
            templates[template.name] = template

        return templates

    def generate(
        self,
        template: QueryTemplate,
        amount: int,
        file_path: str,
        console: Console,
    ) -> None:
        """Generates queries from template and stores them in file."""

        queries = template.generate_all_queries()

        if amount > len(queries):
            console.print(
                f"[yellow]Warning:[/yellow] Requested [bold]{amount}[/bold] queries, "
                f"but template [bold]'{template.name}'[/bold] can only generate "
                f"[bold]{len(queries)}[/bold] unique queries. "
                f"Generating [bold]{len(queries)}[/bold] queries instead."
            )

            amount = len(queries)

        generated_queries = queries[:amount]

        with open(file_path, "a", encoding="utf-8") as file:
            for query in generated_queries:
                query_data = {
                    "id": self.next_id,
                    "name": template.name,
                    "version": template.version,
                    "dataset": template.dataset,
                    "column": template.column,
                    "query": query,
                    "selectivity_ground_truth": 123,
                }

                file.write(json.dumps(query_data) + "\n")

                self.next_id += 1
