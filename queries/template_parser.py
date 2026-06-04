from dataclasses import dataclass
from enum import Enum


class QueryTemplatePartType(Enum):
    TEXT = "text"
    COLUMN_REF = "column_ref"


@dataclass(frozen=True)
class ColumnRef:
    column_name: str
    dataset_ref: str | None = None

    @staticmethod
    def parse(raw: str) -> "ColumnRef":
        raw = raw.strip()

        if not raw:
            raise ValueError("Empty column reference is not allowed.")

        parts = raw.split(".")

        if len(parts) == 1:
            return ColumnRef(
                dataset_ref=None,
                column_name=parts[0].strip(),
            )

        if len(parts) == 2:
            dataset_ref = parts[0].strip()
            column_name = parts[1].strip()

            if not dataset_ref:
                raise ValueError("Empty dataset reference is not allowed.")

            if not column_name:
                raise ValueError("Empty column name is not allowed.")

            return ColumnRef(
                dataset_ref=dataset_ref,
                column_name=column_name,
            )

        raise ValueError(
            f"Invalid column reference '{raw}'. "
            "Expected '{column}' or '{dataset.column}'."
        )

    def to_dict(self) -> dict:
        return {
            "dataset_ref": self.dataset_ref,
            "column_name": self.column_name,
        }


@dataclass(frozen=True)
class QueryTemplatePart:
    type: QueryTemplatePartType
    value: str | ColumnRef

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "value": (
                self.value.to_dict()
                if isinstance(self.value, ColumnRef)
                else self.value
            ),
        }


@dataclass(frozen=True)
class QueryTemplate:
    raw: str
    parts: list[QueryTemplatePart]

    def to_dict(self) -> dict:
        return {
            "raw": self.raw,
            "parts": [part.to_dict() for part in self.parts],
        }


class QueryTemplateParser:
    @staticmethod
    def parse(template: str) -> QueryTemplate:
        parts: list[QueryTemplatePart] = []
        current_text: list[str] = []
        current_column: list[str] | None = None

        for char in template:
            if char == "{":
                if current_column is not None:
                    raise ValueError("Nested '{' is not allowed.")

                if current_text:
                    parts.append(
                        QueryTemplatePart(
                            QueryTemplatePartType.TEXT,
                            "".join(current_text),
                        )
                    )
                    current_text = []

                current_column = []

            elif char == "}":
                if current_column is None:
                    raise ValueError("Found '}' without matching '{'.")

                raw_column_ref = "".join(current_column)
                column_ref = ColumnRef.parse(raw_column_ref)

                parts.append(
                    QueryTemplatePart(
                        QueryTemplatePartType.COLUMN_REF,
                        column_ref,
                    )
                )

                current_column = None

            else:
                if current_column is not None:
                    current_column.append(char)
                else:
                    current_text.append(char)

        if current_column is not None:
            raise ValueError("Found '{' without matching '}'.")

        if current_text:
            parts.append(
                QueryTemplatePart(
                    QueryTemplatePartType.TEXT,
                    "".join(current_text),
                )
            )

        return QueryTemplate(raw=template, parts=parts)