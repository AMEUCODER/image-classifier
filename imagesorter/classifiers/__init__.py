"""Classifier package.

Face names are imported lazily so that
`python -m imagesorter.classifiers.faces ...` runs without a RuntimeWarning.
"""
from .base import BaseClassifier, Result

__all__ = ["BaseClassifier", "Result", "FaceClassifier",
           "train_model", "add_person"]


def __getattr__(name):
    if name in ("FaceClassifier", "train_model", "add_person"):
        from . import faces
        return getattr(faces, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
