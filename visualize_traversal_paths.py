#!/usr/bin/env python3
"""
Traversal Path Visualization Script
==================================

Visualizes both context grouping and retrieval algorithm traversal paths using
all available matplotlib visualization methods. Allows you to see exactly how
semantic navigation works for dataset generation and algorithm retrieval.

Usage:
    python visualize_traversal_paths.py [--save-figs] [--show-figs]

Features:
- Generates single context group and visualizes its traversal path
- Runs all retrieval algorithms and visualizes their traversal paths
- Uses all 3 matplotlib visualization types: windowed, global, sequential
- Saves figures to visualizations/ directory
- No DeepEval evaluation - pure traversal path visualization
"""

import argparse
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt

# Local imports
from utils.knowledge_graph import KnowledgeGraph
from utils.retrieval import RetrievalOrchestrator
from utils.matplotlib_visualizer import (
    create_heatmap_visualization,
    create_global_visualization,
    create_sequential_visualization
)
from utils.traversal import TraversalPath, GranularityLevel, ConnectionType
from utils.algos.base_algorithm import RetrievalResult
from evaluation.context_grouping import ContextGroupingOrchestrator, ContextGroup


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging for the visualization script."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml."""
    config_path = Path("config.yaml")
    if not config_path.exists():
        raise FileNotFoundError("config.yaml not found. Please run from project root.")

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_knowledge_graph(config: Dict[str, Any], logger: logging.Logger) -> KnowledgeGraph:
    """Load knowledge graph from data directory with embeddings."""
    import json

    kg_path = Path(config['directories']['data']) / "knowledge_graph.json"

    if not kg_path.exists():
        raise FileNotFoundError(
            f"Knowledge graph not found at {kg_path}. "
            "Please run the knowledge graph pipeline first."
        )

    logger.info(f"📂 Loading knowledge graph from {kg_path}")

    # Load embeddings data
    embeddings_path = Path(config['directories']['embeddings']) / "raw" / "sentence_transformers_all_mpnet_base_v2_multi_granularity.json"
    embeddings_data = None

    if embeddings_path.exists():
        logger.info(f"📊 Loading embeddings from {embeddings_path}")
        with open(embeddings_path, 'r') as f:
            raw_embeddings_data = json.load(f)

        # Transform the JSON structure to match what load_phase3_embeddings expects
        # JSON structure: {"metadata": {...}, "embeddings": {"chunks": [...], "sentences": [...]}}
        # Expected structure: {"model_name": {"chunks": [...], "sentences": [...]}}
        model_name = raw_embeddings_data['metadata']['model_name']
        embeddings_data = {
            model_name: raw_embeddings_data['embeddings']
        }

        logger.info(f"✅ Embeddings loaded successfully for model: {model_name}")
    else:
        logger.warning(f"⚠️ Embeddings not found at {embeddings_path} - visualizations will be basic")

    # Load knowledge graph with embeddings
    kg = KnowledgeGraph.load(str(kg_path), embeddings_data)
    logger.info(f"✅ Knowledge graph loaded: {len(kg.chunks)} chunks, {len(kg.documents)} documents")

    return kg


def convert_context_group_to_retrieval_result(context_group: ContextGroup,
                                             kg: KnowledgeGraph) -> RetrievalResult:
    """
    Convert ContextGroup to RetrievalResult format for visualization compatibility.

    Creates a pseudo-RetrievalResult that represents the context grouping process
    as if it were a retrieval algorithm result.
    """
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


def generate_single_context_group(kg: KnowledgeGraph, config: Dict[str, Any],
                                 logger: logging.Logger) -> ContextGroup:
    """Generate a single context group using the configured strategies."""
    logger.info("🎯 Generating single context group for visualization")

    # Initialize context grouping orchestrator
    context_orchestrator = ContextGroupingOrchestrator(
        kg=kg,
        config=config,
        logger=logger
    )

    # Generate just 1 context group
    context_groups = context_orchestrator.generate_context_groups(total_groups=1)

    if not context_groups:
        raise RuntimeError("Failed to generate any context groups")

    context_group = context_groups[0]
    logger.info(f"✅ Generated context group using '{context_group.strategy}' strategy")
    logger.info(f"   Chunks: {len(context_group.chunks)}, Sentences: {len(context_group.sentences)}")
    logger.info(f"   Traversal path: {len(context_group.traversal_path)} nodes")

    return context_group


def create_context_group_visualizations(context_group: ContextGroup, kg: KnowledgeGraph,
                                       output_dir: Path, logger: logging.Logger) -> List[plt.Figure]:
    """Create all 3 matplotlib visualizations for the context grouping process."""
    logger.info("🎨 Creating context grouping visualizations...")

    # Convert to RetrievalResult format
    pseudo_result = convert_context_group_to_retrieval_result(context_group, kg)

    figures = []
    viz_types = ["windowed", "global", "sequential"]

    for viz_type in viz_types:
        try:
            logger.info(f"   Creating {viz_type} visualization...")

            fig = create_heatmap_visualization(
                result=pseudo_result,
                query="Context Grouping Process",
                knowledge_graph=kg,
                visualization_type=viz_type
            )

            # Update title to reflect context grouping
            fig.suptitle(f"Context Grouping Traversal - {context_group.strategy.title()} Strategy ({viz_type.title()})",
                        fontsize=16, y=0.95)

            figures.append(fig)

            # Save figure if output directory provided
            if output_dir:
                filename = f"context_grouping_{context_group.strategy}_{viz_type}.png"
                filepath = output_dir / filename
                fig.savefig(filepath, dpi=150, bbox_inches='tight')
                logger.info(f"   Saved: {filepath}")

        except Exception as e:
            logger.error(f"   Failed to create {viz_type} visualization: {e}")

    logger.info(f"✅ Created {len(figures)} context grouping visualizations")
    return figures


def create_retrieval_visualizations(kg: KnowledgeGraph, config: Dict[str, Any],
                                   output_dir: Path, logger: logging.Logger) -> List[plt.Figure]:
    """Create visualizations for all retrieval algorithms."""
    logger.info("🔍 Creating retrieval algorithm visualizations...")

    # Initialize retrieval orchestrator
    retrieval_orchestrator = RetrievalOrchestrator(kg, config, logger)

    # Get configured algorithms
    deepeval_config = config.get('deepeval', {})
    algorithm_config = deepeval_config.get('algorithms', {})
    test_algorithms = algorithm_config.get('test_algorithms', ['basic_retrieval'])

    # Use a sample query for testing
    test_query = "What are the key concepts in machine learning and neural networks?"

    figures = []
    viz_types = ["windowed", "global", "sequential"]

    for algorithm_name in test_algorithms:
        try:
            logger.info(f"   Running {algorithm_name} algorithm...")

            # Run algorithm
            result = retrieval_orchestrator.run_algorithm(algorithm_name, test_query)

            # Create all 3 visualizations for this algorithm
            for viz_type in viz_types:
                try:
                    logger.info(f"     Creating {viz_type} visualization...")

                    fig = create_heatmap_visualization(
                        result=result,
                        query=test_query,
                        knowledge_graph=kg,
                        visualization_type=viz_type
                    )

                    figures.append(fig)

                    # Save figure if output directory provided
                    if output_dir:
                        filename = f"retrieval_{algorithm_name}_{viz_type}.png"
                        filepath = output_dir / filename
                        fig.savefig(filepath, dpi=150, bbox_inches='tight')
                        logger.info(f"     Saved: {filepath}")

                except Exception as e:
                    logger.error(f"     Failed to create {viz_type} visualization: {e}")

        except Exception as e:
            logger.error(f"   Failed to run {algorithm_name}: {e}")

    logger.info(f"✅ Created retrieval visualizations for {len(test_algorithms)} algorithms")
    return figures


def main():
    """Main script execution."""
    parser = argparse.ArgumentParser(description="Visualize traversal paths for context grouping and retrieval")
    parser.add_argument("--save-figs", action="store_true", help="Save figures to visualizations/ directory")
    parser.add_argument("--show-figs", action="store_true", help="Display figures on screen")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--context-only", action="store_true", help="Only visualize context grouping (skip retrieval)")
    parser.add_argument("--retrieval-only", action="store_true", help="Only visualize retrieval (skip context grouping)")

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.verbose)
    logger.info("🎨 Starting traversal path visualization script")

    try:
        # Load configuration and knowledge graph
        config = load_config()
        kg = load_knowledge_graph(config, logger)

        # Setup output directory
        output_dir = None
        if args.save_figs:
            output_dir = Path("visualizations")
            output_dir.mkdir(exist_ok=True)
            logger.info(f"📁 Saving figures to {output_dir}")

        all_figures = []

        # Step 1 & 2: Generate and visualize context grouping (unless skipped)
        if not args.retrieval_only:
            logger.info("=" * 60)
            logger.info("STEP 1 & 2: Context Grouping Generation and Visualization")
            logger.info("=" * 60)

            context_group = generate_single_context_group(kg, config, logger)
            context_figures = create_context_group_visualizations(context_group, kg, output_dir, logger)
            all_figures.extend(context_figures)

        # Step 3: Retrieval algorithm visualization (unless skipped)
        if not args.context_only:
            logger.info("=" * 60)
            logger.info("STEP 3: Retrieval Algorithm Visualization")
            logger.info("=" * 60)

            retrieval_figures = create_retrieval_visualizations(kg, config, output_dir, logger)
            all_figures.extend(retrieval_figures)

        # Display results
        logger.info("=" * 60)
        logger.info("VISUALIZATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"✅ Created {len(all_figures)} total visualizations")

        if args.save_figs and output_dir:
            saved_files = list(output_dir.glob("*.png"))
            logger.info(f"📁 Saved {len(saved_files)} figures to {output_dir}")

        # Show figures if requested
        if args.show_figs:
            logger.info("🖼️  Displaying figures...")
            plt.show()
        else:
            logger.info("💡 Use --show-figs to display figures on screen")

    except Exception as e:
        logger.error(f"❌ Script failed: {e}")
        raise


if __name__ == "__main__":
    main()