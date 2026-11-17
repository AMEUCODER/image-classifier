"""Shared contract for every classifier in the package."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Result:
    """One label for one image (or one face / object inside it)."""
    label: str                      # becomes the folder name
    confidence: float = 0.0         # 0..1, higher is better
    details: dict = field(default_factory=dict)  # extra info for the log / dataframe


class BaseClassifier(ABC):
    """Every classifier takes a BGR image (as loaded by cv2) and returns
    a list of Result objects. An empty list is never returned: use a
    fallback label such as 'no_face' or 'unknown' instead."""

    name = "base"   # used as the column name in the dataframe

    @abstractmethod
    def classify(self, image) -> list[Result]:
        ...
