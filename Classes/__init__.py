"""
SQL Lineage Tool – Classes Package

This package contains the core classes for SQL lineage extraction and refinement.
"""

# Version of the package (optional, but useful)
__version__ = "0.1.0"

# Import main classes for easy access
from .model_classes import (
    LineageOutput,
    SQLLineageExtractor,
)


# Define what gets exported with "from Classes import *"
__all__ = [
    "LineageOutput",
    "SQLLineageExtractor",
]