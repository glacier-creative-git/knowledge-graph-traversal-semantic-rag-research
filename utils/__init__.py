"""
Semantic Graph RAG Research Utilities
====================================

A comprehensive toolkit for semantic graph RAG research including:
- Data loading from SQuAD datasets
- Semantic graph traversal algorithms
- 2D and 3D visualizations
- RAGAS evaluation pipeline
- Complete research pipelines

Quick Start:
-----------
from utils.pipeline import quick_demo, quick_visualization

# Run a quick demo
results = quick_demo('random')

# Run with visualizations
results = quick_visualization('focused', show_2d=True, show_3d=True)
"""

# Core components
from .config import ResearchConfig, get_config
from .data_loader import SQuADDataLoader, load_demo_data, load_random_squad_data, load_focused_squad_data
from .rag_system import SemanticGraphRAG, SentenceInfo, TraversalStep, create_rag_system
from .corpus_wide_rag import CorpusWideSemanticRAG, create_corpus_wide_rag
from .visualizations import SemanticGraphVisualizer, create_2d_visualization, create_3d_visualization, create_analysis_charts
from .evaluation import RAGASEvaluator, EvaluationResults, evaluate_rag_system, print_evaluation_results
from .pipeline import ResearchPipeline, quick_demo, quick_visualization, full_evaluation_pipeline, print_pipeline_summary

# Version info
__version__ = "1.0.0"
__author__ = "Semantic Chunking Research"

# Quick access imports for notebook usage
__all__ = [
    # Configuration
    'ResearchConfig',
    'get_config',

    # Data loading
    'SQuADDataLoader',
    'load_demo_data',
    'load_random_squad_data',
    'load_focused_squad_data',

    # RAG systems
    'SemanticGraphRAG',
    'CorpusWideSemanticRAG',
    'SentenceInfo',
    'TraversalStep',
    'create_rag_system',
    'create_corpus_wide_rag',

    # Visualizations
    'SemanticGraphVisualizer',
    'create_2d_visualization',
    'create_3d_visualization',
    'create_analysis_charts',

    # Evaluation
    'RAGASEvaluator',
    'EvaluationResults',
    'evaluate_rag_system',
    'print_evaluation_results',

    # High-level pipeline
    'ResearchPipeline',
    'quick_demo',
    'quick_visualization',
    'full_evaluation_pipeline',
    'print_pipeline_summary',
]

# Print welcome message when imported
def _print_welcome():
    """Print welcome message with usage examples"""
    print("🔬 Semantic Graph RAG Research Toolkit Loaded")
    print("=" * 50)
    print("Quick start examples:")
    print("  from utils import quick_demo, quick_visualization")
    print("  results = quick_demo('random')")
    print("  results = quick_visualization('focused')")
    print()
    print("For help: print_pipeline_summary()")
    print("=" * 50)

# Uncomment the line below to show welcome message on import
# _print_welcome()