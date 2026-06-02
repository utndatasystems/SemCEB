from dataclasses import dataclass
from enum import Enum


class QueryTemplatePartType(Enum):
    TEXT = "text"
    COLUMN = "column"


@dataclass(frozen=True)
class QueryTemplatePart:
    type: QueryTemplatePartType
    value: str

    def to_dict(self) -> dict:
        return {
            "raw": self.type.value,
            "value": self.value,
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

                column_name = "".join(current_column).strip()

                if not column_name:
                    raise ValueError("Empty column reference is not allowed.")

                parts.append(
                    QueryTemplatePart(
                        QueryTemplatePartType.COLUMN,
                        column_name,
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