#!/usr/bin/env python3
"""
DeepEval-Powered Semantic Traversal Benchmark Orchestrator
=========================================================

Orchestrates knowledge graph building, dataset generation, and algorithm evaluation
using deepeval's synthetic data generation and sophisticated evaluation metrics.

This script replaces the original benchmark.py with a clean three-phase architecture:
1. Knowledge Graph Construction (kg_pipeline.build)
2. Synthetic Dataset Generation (dataset.build) 
3. Algorithm Evaluation (evaluation.run)

Key Features:
- Configurable model providers (OpenAI, Ollama, Anthropic, OpenRouter)
- Evolution-based question complexity enhancement
- Comprehensive RAG + custom semantic traversal metrics
- Hyperparameter tracking for dashboard visibility
- Results compatible with existing visualization infrastructure

Usage:
    python benchmark.py                                    # Full pipeline with all algorithms
    python benchmark.py --algorithm triangulation_geometric_fulldim # Test specific algorithm
    python benchmark.py --dataset-only                    # Generate dataset only
    python benchmark.py --enable-visualizations           # Force enable traversal visualizations
    python benchmark.py --disable-visualizations          # Force disable visualizations
    python benchmark.py --models ollama openai            # Override model configuration
    python benchmark.py --force-rebuild-all               # Force rebuild everything
"""

import sys
import argparse
import logging
import yaml
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Environment variable management
from dotenv import load_dotenv

# Import our three main components
from utils.kg_pipeline import KnowledgeGraphPipeline
from evaluation.dataset import DatasetBuilder
from evaluation.evaluation import EvaluationOrchestrator

# Import visualization components
from utils.plotly_visualizer import create_algorithm_visualization
from utils.matplotlib_visualizer import (
    create_heatmap_visualization,
    create_global_visualization,
    create_sequential_visualization
)
from utils.algos.base_algorithm import RetrievalResult
from utils.traversal import TraversalPath, GranularityLevel, ConnectionType
from evaluation.context_grouping import ContextGroup


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup comprehensive logging for the benchmark orchestrator."""
    log_level = logging.DEBUG if verbose else logging.INFO
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('benchmark.log', mode='w', encoding='utf-8')
        ]
    )

    return logging.getLogger("DeepEvalBenchmark")


def setup_visualization_output(config: Dict[str, Any], output_prefix: Optional[str] = None) -> Optional[Path]:
    """Setup output directory for visualizations if enabled."""
    viz_config = config.get('visualization', {})

    if not viz_config.get('enabled', False):
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = output_prefix or "benchmark"
    output_dir = Path(config['directories']['visualizations']) / f"{prefix}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for benchmark configuration."""
    parser = argparse.ArgumentParser(
        description="DeepEval-Powered Semantic Traversal Benchmark System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python benchmark.py                                     # Full pipeline with all algorithms
    python benchmark.py --algorithm triangulation_geometric_fulldim # Test specific algorithm
    python benchmark.py --dataset-only                     # Generate dataset only  
    python benchmark.py --force-rebuild-all                # Force rebuild everything
    python benchmark.py --verbose                          # Enable debug logging
    python benchmark.py --config custom_config.yaml       # Use custom configuration
        """
    )
    
    # Algorithm selection
    parser.add_argument(
        '--algorithm',
        type=str,
        choices=[
            'basic_retrieval',
            'query_traversal',
            'kg_traversal',
            'triangulation_average',
            'triangulation_geometric_3d',
            'triangulation_geometric_fulldim',
            'llm_guided_traversal'
        ],
        help='Specific algorithm to benchmark (default: evaluate all algorithms)'
    )
    
    # Pipeline control
    parser.add_argument(
        '--dataset-only',
        action='store_true',
        help='Generate synthetic dataset only, skip evaluation'
    )
    
    parser.add_argument(
        '--evaluation-only',
        action='store_true',
        help='Run evaluation only, skip KG building and dataset generation'
    )
    
    # Force rebuild options
    parser.add_argument(
        '--force-rebuild-kg',
        action='store_true',
        help='Force knowledge graph rebuild even if cache exists'
    )
    
    parser.add_argument(
        '--force-rebuild-dataset',
        action='store_true',
        help='Force dataset regeneration even if cache exists'
    )
    
    parser.add_argument(
        '--force-rebuild-all',
        action='store_true',
        help='Force rebuild of all components (KG + dataset)'
    )
    
    # Configuration and output
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--output-prefix',
        type=str,
        help='Prefix for output files and reports'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose debug logging'
    )
    
    # Model Configuration
    parser.add_argument(
        '--embedding-model',
        type=str,
        help='Override embedding model (default: sentence-transformers/all-mpnet-base-v2)'
    )

    parser.add_argument(
        '--quality-scoring-provider',
        type=str,
        choices=['openai', 'anthropic', 'ollama', 'openrouter'],
        help='Override quality scoring provider (default: openrouter)'
    )

    parser.add_argument(
        '--quality-scoring-model-name',
        type=str,
        help='Override quality scoring model name'
    )

    parser.add_argument(
        '--reranking-model',
        type=str,
        help='Override reranking model (default: cross-encoder/ms-marco-MiniLM-L-6-v2)'
    )

    # DeepEval Model Configuration
    parser.add_argument(
        '--question-generation-provider',
        type=str,
        choices=['openai', 'anthropic', 'ollama', 'openrouter'],
        help='Override question generation provider (default: openai)'
    )

    parser.add_argument(
        '--question-generation-model',
        type=str,
        help='Override question generation model (default: gpt-4o)'
    )

    parser.add_argument(
        '--answer-generation-provider',
        type=str,
        choices=['openai', 'anthropic', 'ollama', 'openrouter'],
        help='Override answer generation provider (default: openai)'
    )

    parser.add_argument(
        '--answer-generation-model',
        type=str,
        help='Override answer generation model (default: gpt-4o)'
    )

    parser.add_argument(
        '--evaluation-judge-provider',
        type=str,
        choices=['openai', 'anthropic', 'ollama', 'openrouter'],
        help='Override evaluation judge provider (default: openrouter)'
    )

    parser.add_argument(
        '--evaluation-judge-model',
        type=str,
        help='Override evaluation judge model (default: meta-llama/llama-3.3-70b-instruct)'
    )

    # DeepEval Project Configuration
    parser.add_argument(
        '--deepeval-project-name',
        type=str,
        help='Override DeepEval project name'
    )

    # Dataset Configuration
    parser.add_argument(
        '--dataset-num-goldens',
        type=int,
        help='Number of golden examples to generate (default: 3)'
    )

    parser.add_argument(
        '--dataset-filtration-enabled',
        type=lambda x: x.lower() == 'true',
        help='Enable dataset filtration (true/false, default: true)'
    )

    parser.add_argument(
        '--dataset-filtration-model',
        type=str,
        help='Model for dataset filtration (default: gpt-4o)'
    )

    parser.add_argument(
        '--evolution-enabled',
        type=lambda x: x.lower() == 'true',
        help='Enable question evolution (true/false, default: true)'
    )

    parser.add_argument(
        '--num-evolutions',
        type=int,
        help='Number of evolutions to perform (default: 2)'
    )

    parser.add_argument(
        '--dataset-save-path',
        type=str,
        help='Path to save generated dataset (default: data/synthetic_dataset.json)'
    )

    parser.add_argument(
        '--push-dataset',
        action='store_true',
        help='Push dataset to DeepEval dashboard when complete'
    )

    parser.add_argument(
        '--pull-dataset',
        action='store_true',
        help='Pull dataset from DeepEval dashboard for evaluation'
    )

    parser.add_argument(
        '--dataset-name',
        type=str,
        help='Dataset alias/name for DeepEval dashboard'
    )

    parser.add_argument(
        '--generate-csv',
        action='store_true',
        help='Generate CSV file for DeepEval upload'
    )

    parser.add_argument(
        '--evaluation-run-async',
        action='store_true',
        help='Run evaluation asynchronously'
    )

    # Context Grouping Strategy Configuration
    parser.add_argument(
        '--enable-context-strategies',
        nargs='*',
        choices=['intra_document', 'theme_based', 'sequential_multi_hop', 'knowledge_graph_similarity', 'deepeval_native'],
        help='Enable specific context grouping strategies (space-separated list)'
    )

    parser.add_argument(
        '--disable-context-strategies',
        nargs='*',
        choices=['intra_document', 'theme_based', 'sequential_multi_hop', 'knowledge_graph_similarity', 'deepeval_native'],
        help='Disable specific context grouping strategies (space-separated list)'
    )

    # Retrieval Algorithm Configuration
    parser.add_argument(
        '--test-algorithms',
        nargs='*',
        choices=[
            'basic_retrieval',
            'query_traversal',
            'kg_traversal',
            'triangulation_average',
            'triangulation_geometric_3d',
            'triangulation_geometric_fulldim',
            'llm_guided_traversal'
        ],
        help='Specify which algorithms to test (space-separated list, default: all)'
    )

    # Evolution Type Configuration
    parser.add_argument(
        '--evolution-types',
        nargs='*',
        choices=['REASONING', 'COMPARATIVE', 'IN_BREADTH', 'MULTICONTEXT'],
        help='Specify which evolution types to use (space-separated list)'
    )

    # Legacy model overrides (for backward compatibility)
    parser.add_argument(
        '--question-model',
        type=str,
        help='Override question generation model (deprecated: use --question-generation-provider)'
    )

    parser.add_argument(
        '--evaluation-model',
        type=str,
        help='Override evaluation judge model (deprecated: use --evaluation-judge-model)'
    )

    # Visualization control
    parser.add_argument(
        '--enable-visualizations',
        action='store_true',
        help='Enable generation of traversal visualizations (overrides config)'
    )

    parser.add_argument(
        '--disable-visualizations',
        action='store_true',
        help='Disable generation of traversal visualizations (overrides config)'
    )

    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate configuration from YAML file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required configuration sections
    required_sections = ['directories', 'deepeval']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    return config


def apply_model_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Apply command-line overrides to configuration."""
    logger = logging.getLogger("DeepEvalBenchmark")

    # Model Configuration Overrides
    if args.embedding_model:
        config.setdefault('models', {}).setdefault('embedding_models', [])
        config['models']['embedding_models'] = [args.embedding_model]
        logger.info(f"🔧 Override: Embedding model -> {args.embedding_model}")

    if args.quality_scoring_provider:
        config.setdefault('knowledge_graph', {}).setdefault('quality_scoring', {})['provider'] = args.quality_scoring_provider
        logger.info(f"🔧 Override: Quality scoring provider -> {args.quality_scoring_provider}")

    if args.quality_scoring_model_name:
        config.setdefault('knowledge_graph', {}).setdefault('quality_scoring', {})['model_name'] = args.quality_scoring_model_name
        logger.info(f"🔧 Override: Quality scoring model -> {args.quality_scoring_model_name}")

    if args.reranking_model:
        config.setdefault('reranking', {})['model_name'] = args.reranking_model
        logger.info(f"🔧 Override: Reranking model -> {args.reranking_model}")

    # DeepEval Model Configuration Overrides
    if args.question_generation_provider:
        config.setdefault('deepeval', {}).setdefault('models', {}).setdefault('question_generation', {})['provider'] = args.question_generation_provider
        logger.info(f"🔧 Override: Question generation provider -> {args.question_generation_provider}")

    if args.question_generation_model:
        config.setdefault('deepeval', {}).setdefault('models', {}).setdefault('question_generation', {})['model_name'] = args.question_generation_model
        logger.info(f"🔧 Override: Question generation model -> {args.question_generation_model}")

    if args.answer_generation_provider:
        config.setdefault('deepeval', {}).setdefault('models', {}).setdefault('answer_generation', {})['provider'] = args.answer_generation_provider
        logger.info(f"🔧 Override: Answer generation provider -> {args.answer_generation_provider}")

    if args.answer_generation_model:
        config.setdefault('deepeval', {}).setdefault('models', {}).setdefault('answer_generation', {})['model_name'] = args.answer_generation_model
        logger.info(f"🔧 Override: Answer generation model -> {args.answer_generation_model}")

    if args.evaluation_judge_provider:
        config.setdefault('deepeval', {}).setdefault('models', {}).setdefault('evaluation_judge', {})['provider'] = args.evaluation_judge_provider
        logger.info(f"🔧 Override: Evaluation judge provider -> {args.evaluation_judge_provider}")

    if args.evaluation_judge_model:
        config.setdefault('deepeval', {}).setdefault('models', {}).setdefault('evaluation_judge', {})['model_name'] = args.evaluation_judge_model
        logger.info(f"🔧 Override: Evaluation judge model -> {args.evaluation_judge_model}")

    # DeepEval Project Configuration
    if args.deepeval_project_name:
        config.setdefault('deepeval', {}).setdefault('project', {})['name'] = args.deepeval_project_name
        logger.info(f"🔧 Override: DeepEval project name -> {args.deepeval_project_name}")

    # Dataset Configuration Overrides
    if args.dataset_num_goldens:
        config.setdefault('deepeval', {}).setdefault('dataset', {}).setdefault('generation', {})['num_goldens'] = args.dataset_num_goldens
        logger.info(f"🔧 Override: Dataset num goldens -> {args.dataset_num_goldens}")

    if args.dataset_filtration_enabled is not None:
        config.setdefault('deepeval', {}).setdefault('dataset', {}).setdefault('filtration', {})['enabled'] = args.dataset_filtration_enabled
        logger.info(f"🔧 Override: Dataset filtration enabled -> {args.dataset_filtration_enabled}")

    if args.dataset_filtration_model:
        config.setdefault('deepeval', {}).setdefault('dataset', {}).setdefault('filtration', {})['critic_model'] = args.dataset_filtration_model
        logger.info(f"🔧 Override: Dataset filtration model -> {args.dataset_filtration_model}")

    if args.evolution_enabled is not None:
        config.setdefault('deepeval', {}).setdefault('dataset', {}).setdefault('evolution', {})['enabled'] = args.evolution_enabled
        logger.info(f"🔧 Override: Evolution enabled -> {args.evolution_enabled}")

    if args.num_evolutions:
        config.setdefault('deepeval', {}).setdefault('dataset', {}).setdefault('evolution', {})['num_evolutions'] = args.num_evolutions
        logger.info(f"🔧 Override: Num evolutions -> {args.num_evolutions}")

    if args.dataset_save_path:
        config.setdefault('deepeval', {}).setdefault('dataset', {}).setdefault('output', {})['save_path'] = args.dataset_save_path
        logger.info(f"🔧 Override: Dataset save path -> {args.dataset_save_path}")

    if args.push_dataset:
        config.setdefault('deepeval', {}).setdefault('dataset', {}).setdefault('output', {})['push_to_dashboard'] = True
        logger.info("🔧 Override: Push dataset to dashboard enabled")

    if args.pull_dataset:
        config.setdefault('deepeval', {}).setdefault('dataset', {}).setdefault('output', {})['pull_from_dashboard'] = True
        logger.info("🔧 Override: Pull dataset from dashboard enabled")

    if args.dataset_name:
        config.setdefault('deepeval', {}).setdefault('dataset', {}).setdefault('output', {})['dataset_alias'] = args.dataset_name
        logger.info(f"🔧 Override: Dataset name -> {args.dataset_name}")

    if args.generate_csv:
        config.setdefault('deepeval', {}).setdefault('dataset', {}).setdefault('output', {})['generate_csv'] = True
        logger.info("🔧 Override: Generate CSV enabled")

    if args.evaluation_run_async:
        config.setdefault('deepeval', {}).setdefault('evaluation', {}).setdefault('async_config', {})['run_async'] = True
        logger.info("🔧 Override: Async evaluation enabled")

    # Context Grouping Strategy Overrides
    if args.enable_context_strategies or args.disable_context_strategies:
        context_strategies = config.setdefault('context_strategies', {})

        # If enabling specific strategies, disable all others first
        if args.enable_context_strategies is not None:
            for strategy in context_strategies:
                context_strategies[strategy]['enabled'] = False
            for strategy in args.enable_context_strategies:
                if strategy in context_strategies:
                    context_strategies[strategy]['enabled'] = True
                    logger.info(f"🔧 Override: Enabled context strategy -> {strategy}")

        # Disable specific strategies
        if args.disable_context_strategies:
            for strategy in args.disable_context_strategies:
                if strategy in context_strategies:
                    context_strategies[strategy]['enabled'] = False
                    logger.info(f"🔧 Override: Disabled context strategy -> {strategy}")

    # Test Algorithms Override
    if args.test_algorithms:
        config.setdefault('deepeval', {}).setdefault('evaluation', {}).setdefault('algorithms', {})['test_algorithms'] = args.test_algorithms
        logger.info(f"🔧 Override: Test algorithms -> {', '.join(args.test_algorithms)}")

    # Evolution Types Override
    if args.evolution_types:
        config.setdefault('deepeval', {}).setdefault('dataset', {}).setdefault('evolution', {})['evolution_types'] = args.evolution_types
        logger.info(f"🔧 Override: Evolution types -> {', '.join(args.evolution_types)}")

    # Legacy model overrides (backward compatibility)
    if args.question_model:
        config.setdefault('deepeval', {}).setdefault('models', {}).setdefault('question_generation', {})['provider'] = args.question_model
        logger.info(f"🔧 Override (legacy): Question generation provider -> {args.question_model}")

    if args.evaluation_model:
        config.setdefault('deepeval', {}).setdefault('models', {}).setdefault('evaluation_judge', {})['model_name'] = args.evaluation_model
        logger.info(f"🔧 Override (legacy): Evaluation judge model -> {args.evaluation_model}")

    # Apply visualization overrides
    if args.enable_visualizations:
        config.setdefault('visualization', {})['enabled'] = True
        logger.info("🔧 Override: Visualizations enabled")

    if args.disable_visualizations:
        config.setdefault('visualization', {})['enabled'] = False
        logger.info("🔧 Override: Visualizations disabled")

    return config


def create_context_grouping_visualizations(context_groups: List[ContextGroup], output_dir: Path,
                                          config: Dict[str, Any], logger: logging.Logger) -> None:
    """Create visualizations for context grouping process during dataset generation."""
    viz_config = config.get('visualization', {}).get('context_grouping', {})

    if not viz_config.get('enabled', True):
        return

    logger.info("🎨 Creating context grouping visualizations...")

    # Load knowledge graph for visualization with embeddings
    kg_path = Path(config['directories']['data']) / "knowledge_graph.json"
    if not kg_path.exists():
        logger.warning("   ⚠️ Knowledge graph not found, skipping context grouping visualizations")
        return

    from utils.knowledge_graph import KnowledgeGraph
    import json

    # Load embeddings for visualization - dynamically determine path from config
    embedding_model = config.get('models', {}).get('embedding_models', ['sentence-transformers/all-mpnet-base-v2'])[0]
    safe_model_name = embedding_model.replace("/", "_").replace("-", "_")
    embeddings_path = Path(f"embeddings/raw/{safe_model_name}_multi_granularity.json")
    embeddings_data = None
    if embeddings_path.exists():
        with open(embeddings_path, 'r') as f:
            raw_data = json.load(f)
        model_name = raw_data.get('metadata', {}).get('model_name', embedding_model)
        embeddings_data = {model_name: raw_data['embeddings']}

    kg = KnowledgeGraph.load(str(kg_path), embeddings_data)

    # Create visualizations for first few context groups (to avoid overwhelming output)
    max_groups_to_visualize = min(3, len(context_groups))

    for i, context_group in enumerate(context_groups[:max_groups_to_visualize]):
        try:
            logger.info(f"   📊 Creating visualizations for context group {i+1} ({context_group.strategy})")

            # Convert ContextGroup to RetrievalResult format for visualization compatibility
            pseudo_result = convert_context_group_to_retrieval_result(context_group)

            viz_types = viz_config.get('visualization_types', ['windowed', 'global', 'sequential'])

            for viz_type in viz_types:
                try:
                    fig = create_heatmap_visualization(
                        result=pseudo_result,
                        query=f"Context Group {i+1}",
                        knowledge_graph=kg,
                        visualization_type=viz_type
                    )

                    # Update title to reflect context grouping
                    fig.suptitle(f"Context Group {i+1} - {context_group.strategy.title()} Strategy ({viz_type.title()})",
                                fontsize=16, y=0.95)

                    # Save figure
                    filename = f"context_group_{i+1}_{context_group.strategy}_{viz_type}.png"
                    filepath = output_dir / filename
                    fig.savefig(str(filepath), dpi=config.get('visualization', {}).get('output', {}).get('dpi', 300),
                               bbox_inches='tight')
                    plt.close(fig)
                    logger.info(f"   ✅ {viz_type.title()} saved: {filename}")

                except Exception as e:
                    logger.warning(f"   ⚠️ {viz_type.title()} visualization failed: {str(e)}")

        except Exception as e:
            logger.warning(f"   ⚠️ Context group {i+1} visualization failed: {str(e)}")


def convert_context_group_to_retrieval_result(context_group: ContextGroup) -> RetrievalResult:
    """Convert ContextGroup to RetrievalResult format for visualization compatibility."""
    # Convert simple traversal path to TraversalPath object
    traversal_path = TraversalPath(
        nodes=context_group.traversal_path,
        connection_types=[ConnectionType.RAW_SIMILARITY] * max(0, len(context_group.traversal_path) - 1),
        granularity_levels=[GranularityLevel.CHUNK] * len(context_group.traversal_path),
        total_hops=max(0, len(context_group.traversal_path) - 1),
        is_valid=True,
        validation_errors=[]
    )

    # Create RetrievalResult representing context grouping
    return RetrievalResult(
        algorithm_name=f"context_grouping_{context_group.strategy}",
        traversal_path=traversal_path,
        retrieved_content=context_group.chunks,
        confidence_scores=[1.0] * len(context_group.chunks),  # Uniform confidence for context grouping
        query="Context Grouping Process",  # Pseudo-query
        total_hops=len(context_group.traversal_path) - 1,
        final_score=1.0,
        processing_time=0.0,
        metadata={
            "strategy": context_group.strategy,
            "context_metadata": context_group.metadata,
            "is_context_grouping": True,  # Flag to identify this as context grouping
            "sentence_count": len(context_group.sentences)
        }
    )


def create_retrieval_visualizations(results: Dict[str, Any], query: str, output_dir: Path,
                                   config: Dict[str, Any], logger: logging.Logger) -> None:
    """Create visualizations for algorithm retrieval paths during evaluation."""
    viz_config = config.get('visualization', {}).get('retrieval_paths', {})

    if not viz_config.get('enabled', True):
        return

    logger.info("🎨 Creating retrieval path visualizations...")

    # Load knowledge graph for visualization with embeddings
    kg_path = Path(config['directories']['data']) / "knowledge_graph.json"
    if not kg_path.exists():
        logger.warning("   ⚠️ Knowledge graph not found, skipping retrieval visualizations")
        return

    from utils.knowledge_graph import KnowledgeGraph
    import json

    # Load embeddings for visualization - dynamically determine path from config
    embedding_model = config.get('models', {}).get('embedding_models', ['sentence-transformers/all-mpnet-base-v2'])[0]
    safe_model_name = embedding_model.replace("/", "_").replace("-", "_")
    embeddings_path = Path(f"embeddings/raw/{safe_model_name}_multi_granularity.json")
    embeddings_data = None
    if embeddings_path.exists():
        with open(embeddings_path, 'r') as f:
            raw_data = json.load(f)
        model_name = raw_data.get('metadata', {}).get('model_name', embedding_model)
        embeddings_data = {model_name: raw_data['embeddings']}

    kg = KnowledgeGraph.load(str(kg_path), embeddings_data)

    # Create safe filename for query
    safe_query = query.replace(' ', '_').replace('?', '').replace('!', '').replace('.', '')[:50]

    for algorithm_name, result in results.items():
        if hasattr(result, 'metadata') and result.metadata.get('error'):
            logger.info(f"   ⏭️  Skipping {algorithm_name} - has errors")
            continue

        try:
            logger.info(f"   📊 Creating visualizations for {algorithm_name}...")

            # Create Plotly 3D visualization if enabled
            if viz_config.get('include_3d_plotly', True):
                try:
                    plotly_fig = create_algorithm_visualization(
                        result=result,
                        query=query,
                        knowledge_graph=kg,
                        method='pca',
                        max_nodes=viz_config.get('max_nodes', 40),
                        show_all_visited=viz_config.get('show_all_visited', True)
                    )

                    plotly_filename = f"{algorithm_name}_{safe_query}_3d.html"
                    plotly_path = output_dir / plotly_filename
                    plotly_fig.write_html(str(plotly_path))
                    logger.info(f"   ✅ Plotly 3D visualization saved: {plotly_filename}")

                except Exception as e:
                    logger.warning(f"   ⚠️ Plotly visualization failed for {algorithm_name}: {str(e)}")

            # Create matplotlib heatmap visualizations if enabled
            if viz_config.get('include_heatmaps', True):
                heatmap_types = ['global', 'sequential']  # Most useful for benchmark comparison

                for heatmap_type in heatmap_types:
                    try:
                        if heatmap_type == 'global':
                            fig = create_global_visualization(
                                result=result,
                                query=query,
                                knowledge_graph=kg,
                                figure_size=tuple(config.get('visualization', {}).get('output', {}).get('figure_size', [24, 10])),
                                max_documents=6
                            )
                        else:  # sequential
                            fig = create_sequential_visualization(
                                result=result,
                                query=query,
                                knowledge_graph=kg,
                                figure_size=tuple(config.get('visualization', {}).get('output', {}).get('figure_size', [20, 8]))
                            )

                        heatmap_filename = f"{algorithm_name}_{safe_query}_{heatmap_type}_heatmap.png"
                        heatmap_path = output_dir / heatmap_filename
                        fig.savefig(str(heatmap_path),
                                   dpi=config.get('visualization', {}).get('output', {}).get('dpi', 300),
                                   bbox_inches='tight')
                        plt.close(fig)
                        logger.info(f"   ✅ {heatmap_type.title()} heatmap saved: {heatmap_filename}")

                    except Exception as e:
                        logger.warning(f"   ⚠️ {heatmap_type.title()} heatmap failed for {algorithm_name}: {str(e)}")

        except Exception as e:
            logger.warning(f"   ⚠️ Visualization creation failed for {algorithm_name}: {str(e)}")


def validate_environment() -> None:
    """Validate environment setup and API keys."""
    import os
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Check for at least one valid model provider configuration
    providers_available = []
    
    if os.getenv('OPENAI_API_KEY'):
        providers_available.append('OpenAI')
    
    if os.getenv('ANTHROPIC_API_KEY'):
        providers_available.append('Anthropic')
    
    if os.getenv('OPENROUTER_API_KEY'):
        providers_available.append('OpenRouter')
    
    # Ollama doesn't require API key, just check if URL is configured
    ollama_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    providers_available.append('Ollama (local)')
    
    if not providers_available:
        raise ValueError(
            "No valid model provider configuration found. "
            "Please set appropriate API keys in your .env file."
        )
    
    logging.getLogger("DeepEvalBenchmark").info(f"✅ Available providers: {', '.join(providers_available)}")


def run_kg_pipeline_phase(config: Dict[str, Any], force_rebuild: bool, logger: logging.Logger) -> Dict[str, Any]:
    """Execute Phase 1: Knowledge Graph Construction."""
    logger.info("📊 Phase 1: Knowledge Graph Construction")
    logger.info("=" * 50)
    
    try:
        # Check if KG already exists and force_rebuild is False
        kg_path = Path(config['directories']['data']) / "knowledge_graph.json"
        
        if kg_path.exists() and not force_rebuild:
            logger.info(f"✅ Knowledge graph found at {kg_path} - skipping rebuild")
            logger.info("   Use --force-rebuild-kg to force regeneration")
            return {"status": "loaded_existing", "path": str(kg_path)}
        
        # Build knowledge graph using existing pipeline
        logger.info("🏗️ Building knowledge graph from Wikipedia data...")
        pipeline = KnowledgeGraphPipeline()
        result = pipeline.build()
        
        logger.info(f"✅ Knowledge graph construction completed")
        logger.info(f"   Result: {result}")
        
        return {"status": "built_new", "result": result}
        
    except Exception as e:
        logger.error(f"❌ Phase 1 failed: {e}")
        raise RuntimeError(f"Knowledge graph construction failed: {e}")


def run_dataset_generation_phase(config: Dict[str, Any], force_rebuild: bool, logger: logging.Logger,
                                visualization_output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Execute Phase 2: Synthetic Dataset Generation with optional visualization."""
    logger.info("🧠 Phase 2: Synthetic Dataset Generation")
    logger.info("=" * 50)

    try:
        # Initialize dataset builder
        dataset_builder = DatasetBuilder(config, logger)

        # Generate synthetic dataset with evolution techniques
        dataset = dataset_builder.build(force_regenerate=force_rebuild)

        logger.info(f"✅ Dataset generation completed")
        logger.info(f"   Generated: {len(dataset.goldens)} synthetic questions")

        # Log evolution statistics if available
        generation_stats = getattr(dataset_builder, 'generation_stats', {})
        evolution_dist = generation_stats.get('evolution_distribution', {})

        if evolution_dist:
            logger.info("   Evolution technique distribution:")
            for evolution_type, count in evolution_dist.items():
                logger.info(f"      {evolution_type}: {count}")

        # Create context grouping visualizations if enabled and output directory provided
        if visualization_output_dir and config.get('visualization', {}).get('enabled', False):
            context_groups = getattr(dataset_builder, 'context_groups', [])

            if context_groups:
                logger.info(f"🎨 Creating visualizations for {len(context_groups)} context groups...")
                try:
                    create_context_grouping_visualizations(context_groups, visualization_output_dir, config, logger)
                    logger.info("✅ Context grouping visualizations completed")
                except Exception as e:
                    logger.warning(f"⚠️ Context grouping visualization failed: {e}")
            else:
                logger.info("📊 No context groups available for visualization")

        return {
            "status": "generated",
            "dataset_size": len(dataset.goldens),
            "evolution_distribution": evolution_dist,
            "context_groups_count": len(getattr(dataset_builder, 'context_groups', []))
        }

    except Exception as e:
        logger.error(f"❌ Phase 2 failed: {e}")
        raise RuntimeError(f"Dataset generation failed: {e}")


def run_evaluation_phase(config: Dict[str, Any], algorithm: Optional[str],
                        output_prefix: Optional[str], logger: logging.Logger,
                        visualization_output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Execute Phase 3: Algorithm Evaluation with optional visualization."""
    logger.info("🔍 Phase 3: Algorithm Evaluation")
    logger.info("=" * 50)

    try:
        # Initialize evaluation orchestrator
        evaluation_orchestrator = EvaluationOrchestrator(config, logger)

        if algorithm:
            # Single algorithm evaluation
            logger.info(f"🎯 Evaluating single algorithm: {algorithm}")

            result = evaluation_orchestrator.run(
                algorithm_name=algorithm,
                output_prefix=output_prefix
            )

            logger.info(f"✅ Single algorithm evaluation completed")
            logger.info(f"   Algorithm: {result.algorithm_name}")
            logger.info(f"   Success rate: {result.summary_statistics['overall_success_rate']:.1%}")
            logger.info(f"   Average score: {result.summary_statistics['average_metric_score']:.3f}")

            # Create retrieval visualizations if enabled and we have retrieval results
            if visualization_output_dir and config.get('visualization', {}).get('enabled', False):
                retrieval_results = getattr(evaluation_orchestrator, 'retrieval_results', {})

                if retrieval_results:
                    # For single algorithm, create visualizations for a sample of test cases
                    sample_results = list(retrieval_results.items())[:3]  # Visualize first 3 test cases

                    for query, algo_results in sample_results:
                        if algorithm in algo_results:
                            single_algo_result = {algorithm: algo_results[algorithm]}
                            try:
                                create_retrieval_visualizations(single_algo_result, query,
                                                              visualization_output_dir, config, logger)
                            except Exception as e:
                                logger.warning(f"⚠️ Retrieval visualization failed for query '{query}': {e}")

            return {
                "status": "single_algorithm",
                "algorithm": result.algorithm_name,
                "success_rate": result.summary_statistics['overall_success_rate'],
                "average_score": result.summary_statistics['average_metric_score'],
                "hyperparameters": result.algorithm_hyperparameters
            }

        else:
            # Comparative evaluation across all algorithms
            logger.info("🏁 Evaluating all configured algorithms...")

            results = evaluation_orchestrator.run_comparison(output_prefix=output_prefix)

            logger.info(f"✅ Comparative evaluation completed")
            logger.info(f"   Algorithms tested: {len(results)}")

            # Log summary comparison with hyperparameters
            logger.info("📊 Algorithm Performance Summary:")
            for algorithm_name, result in results.items():
                avg_score = result.summary_statistics['average_metric_score']
                success_rate = result.summary_statistics['overall_success_rate']
                hyperparams = result.algorithm_hyperparameters

                logger.info(f"   {algorithm_name}:")
                logger.info(f"      Average score: {avg_score:.3f}")
                logger.info(f"      Success rate: {success_rate:.1%}")
                logger.info(f"      Hyperparameters: {hyperparams}")

            # Identify best performing algorithm
            best_algorithm = max(results.keys(),
                               key=lambda k: results[k].summary_statistics['average_metric_score'])
            logger.info(f"🏆 Best performing algorithm: {best_algorithm}")

            # Create retrieval visualizations if enabled
            if visualization_output_dir and config.get('visualization', {}).get('enabled', False):
                retrieval_results = getattr(evaluation_orchestrator, 'retrieval_results', {})

                if retrieval_results:
                    # For comparison, create visualizations for a sample of test cases
                    sample_results = list(retrieval_results.items())[:2]  # Visualize first 2 test cases

                    for query, algo_results in sample_results:
                        try:
                            create_retrieval_visualizations(algo_results, query,
                                                          visualization_output_dir, config, logger)
                        except Exception as e:
                            logger.warning(f"⚠️ Retrieval visualization failed for query '{query}': {e}")

            return {
                "status": "comparative_evaluation",
                "algorithms_tested": list(results.keys()),
                "best_algorithm": best_algorithm,
                "results_summary": {
                    alg: {
                        "average_score": res.summary_statistics['average_metric_score'],
                        "success_rate": res.summary_statistics['overall_success_rate'],
                        "hyperparameters": res.algorithm_hyperparameters
                    }
                    for alg, res in results.items()
                }
            }

    except Exception as e:
        logger.error(f"❌ Phase 3 failed: {e}")
        raise RuntimeError(f"Algorithm evaluation failed: {e}")


def main():
    """Main orchestrator function - coordinates all three phases."""
    # Parse arguments and setup logging
    args = parse_arguments()
    logger = setup_logging(args.verbose)
    
    logger.info("🚀 DeepEval Semantic Traversal Benchmark Starting")
    logger.info("=" * 70)
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    
    total_start_time = datetime.now()
    
    try:
        # Load and validate configuration
        logger.info("🔧 Loading configuration and validating environment...")
        config = load_config(args.config)
        config = apply_model_overrides(config, args)
        validate_environment()
        
        # Determine force rebuild flags
        force_rebuild_kg = args.force_rebuild_kg or args.force_rebuild_all
        force_rebuild_dataset = args.force_rebuild_dataset or args.force_rebuild_all

        # Setup visualization output directory if enabled
        visualization_output_dir = setup_visualization_output(config, args.output_prefix)
        if visualization_output_dir:
            logger.info(f"🎨 Visualizations will be saved to: {visualization_output_dir}")

            # Force dataset regeneration if context grouping visualizations are enabled
            context_viz_enabled = config.get('visualization', {}).get('context_grouping', {}).get('enabled', False)
            if context_viz_enabled and not force_rebuild_dataset:
                logger.info("🔄 Forcing dataset regeneration to capture context groups for visualization")
                force_rebuild_dataset = True

        phase_results = {}

        # Phase 1: Knowledge Graph Construction (unless evaluation-only)
        if not args.evaluation_only:
            kg_result = run_kg_pipeline_phase(config, force_rebuild_kg, logger)
            phase_results['knowledge_graph'] = kg_result

        # Phase 2: Synthetic Dataset Generation (unless evaluation-only or dataset-only completed)
        if not args.evaluation_only:
            dataset_result = run_dataset_generation_phase(config, force_rebuild_dataset, logger, visualization_output_dir)
            phase_results['dataset'] = dataset_result

            # Exit early if dataset-only mode
            if args.dataset_only:
                logger.info("🎯 Dataset-only mode completed successfully!")
                logger.info(f"   Generated: {dataset_result['dataset_size']} questions")
                if dataset_result.get('context_groups_count', 0) > 0:
                    logger.info(f"   Context groups: {dataset_result['context_groups_count']}")
                if visualization_output_dir:
                    logger.info(f"   Visualizations saved to: {visualization_output_dir}")
                return

        # Phase 3: Algorithm Evaluation (unless dataset-only)
        if not args.dataset_only:
            evaluation_result = run_evaluation_phase(config, args.algorithm, args.output_prefix, logger, visualization_output_dir)
            phase_results['evaluation'] = evaluation_result
        
        # Final summary
        total_duration = datetime.now() - total_start_time
        logger.info("🎉 BENCHMARK PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 70)
        logger.info(f"Total Duration: {total_duration}")
        
        # Log phase summaries
        for phase_name, result in phase_results.items():
            logger.info(f"{phase_name.title()}: {result['status']}")
        
        # Special summary for evaluation results
        if 'evaluation' in phase_results:
            eval_result = phase_results['evaluation']
            if eval_result['status'] == 'comparative_evaluation':
                logger.info(f"🏆 Champion Algorithm: {eval_result['best_algorithm']}")
            elif eval_result['status'] == 'single_algorithm':
                logger.info(f"🎯 Algorithm Tested: {eval_result['algorithm']} (Score: {eval_result['average_score']:.3f})")
        
        logger.info("📁 Results saved to benchmark_results/ directory")
        if visualization_output_dir:
            logger.info(f"🎨 Visualizations saved to: {visualization_output_dir}")
            logger.info("   📊 Generated traversal path visualizations for context grouping and retrieval")
        else:
            logger.info("📊 Use existing visualization tools to analyze results")
        
    except KeyboardInterrupt:
        logger.info("⚠️ Benchmark interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"💥 Benchmark failed: {e}")
        
        if args.verbose:
            logger.error("Full traceback:")
            logger.error(traceback.format_exc())
        else:
            logger.error("Use --verbose for full traceback")
        
        sys.exit(1)


if __name__ == "__main__":
    main()
