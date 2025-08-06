"""
Enhanced Semantic Graph RAG Research Utilities
==============================================

A comprehensive toolkit for semantic graph RAG research supporting:
- Multiple datasets: WikiEval, Natural Questions, and extensible framework
- Semantic graph traversal algorithms
- 2D and 3D visualizations
- RAGAS evaluation pipeline
- Complete research pipelines with seamless dataset swapping

Quick Start:
-----------
from utils.pipeline import quick_demo, quick_visualization
from utils import get_available_datasets, print_dataset_info

# List available datasets
print_dataset_info()

# Run a quick demo with WikiEval
results = quick_demo('random', dataset_name='wikieval')

# Run with Natural Questions
results = quick_demo('focused', dataset_name='natural_questions')

# Run with visualizations
results = quick_visualization('focused', dataset_name='wikieval', show_2d=True, show_3d=True)
"""

# Core components
from .config import (
    ResearchConfig,
    get_config,
    get_available_datasets,
    get_dataset_info,
    print_dataset_info,
    validate_research_config
)

from .data_loader import (
    BaseDataLoader,
    WikiEvalDataLoader,
    NaturalQuestionsDataLoader,
    create_data_loader,
    load_demo_data,
    load_random_data,
    load_focused_data
)

from .rag_system import (
    SemanticGraphRAG,
    SentenceInfo,
    TraversalStep,
    create_rag_system
)

from .corpus_wide_rag import (
    CorpusWideSemanticRAG,
    create_corpus_wide_rag
)

from .visualizations import (
    SemanticGraphVisualizer,
    create_2d_visualization,
    create_3d_visualization,
    create_analysis_charts
)

from .evaluation import (
    RAGASEvaluator,
    EvaluationResults,
    evaluate_rag_system,
    print_evaluation_results
)

from .pipeline import (
    ResearchPipeline,
    quick_demo,
    quick_visualization,
    full_evaluation_pipeline,
    print_pipeline_summary
)

# Version info
__version__ = "1.1.0"  # Updated version for enhanced dataset support
__author__ = "Semantic Chunking Research"

# Quick access imports for notebook usage
__all__ = [
    # Configuration
    'ResearchConfig',
    'get_config',
    'get_available_datasets',
    'get_dataset_info',
    'print_dataset_info',
    'validate_research_config',

    # Enhanced data loading (replaces SQuAD-specific loaders)
    'BaseDataLoader',
    'WikiEvalDataLoader',
    'NaturalQuestionsDataLoader',
    'create_data_loader',
    'load_demo_data',
    'load_random_data',
    'load_focused_data',

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

# Dataset compatibility mapping for backward compatibility
DATASET_COMPATIBILITY = {
    'squad': 'natural_questions',  # Map old SQuAD usage to Natural Questions
    'squad_v2': 'natural_questions',
    'wikieval': 'wikieval',
    'natural_questions': 'natural_questions',
    'nq': 'natural_questions'
}

def get_compatible_dataset(old_dataset_name: str) -> str:
    """
    Map old dataset names to new supported datasets for backward compatibility.

    Args:
        old_dataset_name: Legacy dataset name

    Returns:
        Compatible dataset name
    """
    return DATASET_COMPATIBILITY.get(old_dataset_name.lower(), 'wikieval')

# Print welcome message when imported (enhanced)
def _print_welcome():
    """Print enhanced welcome message with dataset information"""
    print("🔬 Enhanced Semantic Graph RAG Research Toolkit Loaded")
    print("=" * 55)
    print("📊 Supported Datasets:")
    for dataset in get_available_datasets():
        info = get_dataset_info(dataset)
        print(f"   • {info['name']}: {info['description']}")
    print()
    print("Quick start examples:")
    print("  from utils import quick_demo, quick_visualization")
    print("  results = quick_demo('random', dataset_name='wikieval')")
    print("  results = quick_visualization('focused', dataset_name='natural_questions')")
    print()
    print("For dataset info: print_dataset_info()")
    print("For help: print_pipeline_summary()")
    print("=" * 55)

# Migration helper functions for users transitioning from SQuAD
def migrate_from_squad():
    """
    Helper function to assist users migrating from SQuAD-based code.
    """
    print("🔄 MIGRATING FROM SQuAD TO ENHANCED DATASET SYSTEM")
    print("=" * 55)
    print()
    print("📋 Changes Required:")
    print("   1. Replace 'SQuADDataLoader' with 'create_data_loader'")
    print("   2. Specify dataset_name in configuration:")
    print("      • 'wikieval' for WikiEval dataset")
    print("      • 'natural_questions' for Natural Questions")
    print()
    print("📝 Code Migration Examples:")
    print("   OLD: loader = SQuADDataLoader(config)")
    print("   NEW: loader = create_data_loader('wikieval', config)")
    print()
    print("   OLD: config = get_config('default')")
    print("   NEW: config = get_config('default', dataset_name='wikieval')")
    print()
    print("   OLD: results = quick_demo('random')")
    print("   NEW: results = quick_demo('random', dataset_name='wikieval')")
    print()
    print("💡 Benefits of Enhanced System:")
    print("   • Seamless dataset switching")
    print("   • Better evaluation datasets (no duplicates/unanswerable questions)")
    print("   • Real user queries from WikiEval and Natural Questions")
    print("   • Modular, extensible architecture")
    print("   • Future-ready for additional datasets")
    print("=" * 55)

# Backward compatibility warnings
def _check_for_squad_usage():
    """Check if user is trying to use old SQuAD-specific functions"""
    import warnings
    import inspect

    frame = inspect.currentframe()
    try:
        # Get the calling frame
        caller_frame = frame.f_back
        if caller_frame:
            caller_code = caller_frame.f_code
            if 'squad' in caller_code.co_filename.lower():
                warnings.warn(
                    "SQuAD-specific code detected. Consider migrating to the enhanced dataset system. "
                    "Use migrate_from_squad() for migration guidance.",
                    DeprecationWarning,
                    stacklevel=2
                )
    finally:
        del frame

# Check for legacy usage when imported
_check_for_squad_usage()

# Uncomment the line below to show welcome message on import
# _print_welcome()