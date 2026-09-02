"""Trainable graph models."""

from .gat import GATGraphEncoder
from .normalization import EdgeAttrNormalizer

__all__ = ["EdgeAttrNormalizer", "GATGraphEncoder"]
