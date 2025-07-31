"""
Building Blocks Package
======================

This package contains the core mathematical building blocks for semantic chunking algorithms.

Modules:
- similarity_matrices: Core similarity matrix creation and enhancement
- distance_calculations: Distance metrics and boundary detection
- dynamic_programming: Optimization framework (coming next)
- optimization_tools: Advanced optimization techniques (coming next)
"""

from .similarity_matrices import SimilarityMatrixBuilder, SimilarityMatrixConfig
from .distance_calculations import DistanceCalculator, DistanceConfig, BoundaryDetector

__version__ = "1.0.0"
__author__ = "Semantic Chunking Research"

__all__ = [
    'SimilarityMatrixBuilder',
    'SimilarityMatrixConfig',
    'DistanceCalculator',
    'DistanceConfig',
    'BoundaryDetector'
]