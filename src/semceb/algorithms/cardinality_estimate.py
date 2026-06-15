from dataclasses import dataclass
from enum import Enum, auto
from typing import Self


class CardinalityEstimateKind(Enum):
    INT = auto()
    UNSUPPORTED = auto()


@dataclass(frozen=True)
class CardinalityEstimate:
    kind: CardinalityEstimateKind
    value: int | None = None
    reason: str | None = None

    @classmethod
    def int(cls, value: int) -> Self:
        return cls(kind=CardinalityEstimateKind.INT, value=value)

    @classmethod
    def unsupported(cls, reason: str | None = None) -> Self:
        return cls(kind=CardinalityEstimateKind.UNSUPPORTED, reason=reason)