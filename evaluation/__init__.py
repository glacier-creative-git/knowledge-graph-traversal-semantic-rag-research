"""
DeepEval Integration Package for Semantic Traversal System
========================================================

Provides synthetic dataset generation and evaluation capabilities
for the semantic traversal retrieval system using deepeval framework.

This package implements:
- ModelManager: Centralized model configuration and validation
- DatasetBuilder: Synthetic dataset generation from semantic similarity graph
- EvaluationOrchestrator: Comprehensive algorithm evaluation and comparison

Key Features:
- Multi-provider LLM support (OpenAI, Ollama, Anthropic, OpenRouter)
- Evolution-based question complexity enhancement
- Comprehensive RAG metrics + custom G-Eval metrics
- Algorithm hyperparameter tracking for dashboard visibility
- Results compatible with existing visualization infrastructure
"""

from .dataset import DatasetBuilder
from .evaluation import EvaluationOrchestrator, EvaluationResult
from .models import ModelManager, ModelConfig

__all__ = [
    'DatasetBuilder',
    'EvaluationOrchestrator', 
    'EvaluationResult',
    'ModelManager',
    'ModelConfig'
]

__version__ = "1.0.0"
__author__ = "Semantic Traversal Research Team"
