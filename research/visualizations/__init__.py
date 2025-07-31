"""
Visualizations Package
=====================

This package contains visualization tools for semantic chunking research.

Modules:
- matrix_visualizer: Comprehensive matrix visualization tools
"""

from .matrix_visualizer import MatrixVisualizer, quick_matrix_plot, quick_comparison_plot

__version__ = "1.0.0"
__author__ = "Semantic Chunking Research"

__all__ = [
    'MatrixVisualizer',
    'quick_matrix_plot',
    'quick_comparison_plot'
]