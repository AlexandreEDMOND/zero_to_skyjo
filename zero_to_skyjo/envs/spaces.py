from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Discrete:
    n: int


@dataclass(frozen=True)
class Box:
    shape: Tuple[int, ...]
    low: float
    high: float
