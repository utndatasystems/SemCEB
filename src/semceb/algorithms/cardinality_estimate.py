from dataclasses import dataclass
from enum import Enum, auto


class CardinalityEstimateKind(Enum):
    INT = auto()
    UNSUPPORTED = auto()


@dataclass(frozen=True)
class CardinalityEstimate:
    kind: CardinalityEstimateKind
    value: int | None = None
    reason: str | None = None

    @classmethod
    def int(cls, value: int) -> "CardinalityEstimate":
        return cls(kind=CardinalityEstimateKind.INT, value=value)

    @classmethod
    def unsupported(cls, reason: str | None = None) -> "CardinalityEstimate":
        return cls(kind=CardinalityEstimateKind.UNSUPPORTED, reason=reason)