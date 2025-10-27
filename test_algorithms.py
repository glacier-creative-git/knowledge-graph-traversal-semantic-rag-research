#!/usr/bin/env python3
"""
Algorithm Testing Script
=======================

Comprehensive testing script for all four retrieval algorithms.
Tests individual algorithms and runs comparative benchmarks.
"""

import sys
import logging
import yaml
from typing import Dict, Any, List
from pathlib import Path
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.retrieval import RetrievalOrchestrator
from utils.knowledge_graph import KnowledgeGraph
from utils.plotly_visualizer import create_algorithm_visualization
from utils.matplotlib_visualizer import (
    create_heatmap_visualization,
    create_global_visualization,
    create_sequential_visualization
)
from utils.algos.base_algorithm import RetrievalResult
from evaluation.context_grouping import ContextGroupingOrchestrator, ContextGroup
from evaluation.models import ModelManager
from utils.traversal import TraversalPath, GranularityLevel, ConnectionType


def setup_logging() -> logging.Logger:
    """Setup logging for the test script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("AlgorithmTester")


def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml."""
    config_path = project_root / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def load_knowledge_graph(config: Dict[str, Any], logger: logging.Logger) -> KnowledgeGraph:
    """Load an existing knowledge graph with cached embeddings for testing."""
    logger.info("Loading knowledge graph for testing...")

    # Load knowledge graph from data directory
    kg_file = project_root / "data" / "knowledge_graph.json"

    if not kg_file.exists():
        raise FileNotFoundError(f"Knowledge graph file not found: {kg_file}")

    logger.info(f"Loading knowledge graph from: {kg_file}")

    # Load cached embeddings
    embeddings_dir = project_root / "embeddings" / "raw"
    
    # Look for available embedding files (support both old and new models)
    possible_files = [
        embeddings_dir / "mixedbread_ai_mxbai_embed_large_v1_multi_granularity.json",
        embeddings_dir / "sentence_transformers_all_mpnet_base_v2_multi_granularity.json"
    ]
    
    embeddings_file = None
    for file_path in possible_files:
        if file_path.exists():
            embeddings_file = file_path
            break
    
    if not embeddings_file:
        logger.warning(f"Embeddings file not found: {embeddings_file}")
        logger.warning("Loading knowledge graph without embeddings")
        kg = KnowledgeGraph.load(str(kg_file))
    else:
        logger.info(f"Loading cached embeddings from: {embeddings_file}")

        # Load embeddings data
        import json
        with open(embeddings_file, 'r') as f:
            raw_embeddings = json.load(f)

        # Extract the model name from metadata (this is critical!)
        model_name = raw_embeddings.get('metadata', {}).get('model_name', 'unknown')
        logger.info(f"📊 Detected model name from embeddings: {model_name}")

        # FIXED: Extract the nested 'embeddings' structure
        # The cached file has structure: {"metadata": {...}, "embeddings": {"chunks": [...], "sentences": [...]}}
        # But the loading code expects: {"chunks": [...], "sentences": [...]}
        if 'embeddings' in raw_embeddings:
            nested_embeddings = raw_embeddings['embeddings']
            logger.info(f"Found nested embeddings structure with keys: {list(nested_embeddings.keys())}")
        else:
            # Fallback to direct structure
            nested_embeddings = raw_embeddings
            logger.info(f"Using direct embeddings structure with keys: {list(nested_embeddings.keys())}")

        # Convert to the expected format for KnowledgeGraph.load()
        # Use the DETECTED model name, not a hardcoded one!
        embeddings_data = {model_name: nested_embeddings}

        logger.info(f"Loaded embeddings data structure:")
        for key, value in nested_embeddings.items():
            if isinstance(value, list):
                logger.info(f"  {key}: {len(value)} items")
                if len(value) > 0 and isinstance(value[0], dict):
                    logger.info(f"    Sample item keys: {list(value[0].keys())}")
        # Load knowledge graph with embeddings
        kg = KnowledgeGraph.load(str(kg_file), embeddings_data)

    logger.info(f"Knowledge graph loaded:")
    logger.info(f"  Chunks: {len(kg.chunks)}")
    logger.info(f"  Sentences: {len(kg.sentences)}")
    logger.info(f"  Documents: {len(kg.documents) if hasattr(kg, 'documents') else 'N/A'}")

    return kg


def setup_visualization_output() -> Path:
    """Setup output directory for visualizations."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = project_root / "visualizations" / f"algorithm_test_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def create_visualizations(result, query: str, kg: KnowledgeGraph, algorithm_name: str,
                          output_dir: Path, logger: logging.Logger) -> None:
    """Create both plotly and matplotlib visualizations for an algorithm result."""
    try:
        logger.info(f"   📊 Creating visualizations for {algorithm_name}...")

        # Create safe filename
        safe_query = query.replace(' ', '_').replace('?', '').replace('!', '').replace('.', '')[:50]

        # Create Plotly 3D visualization
        try:
            plotly_fig = create_algorithm_visualization(
                result=result,
                query=query,
                knowledge_graph=kg,
                method='pca',
                max_nodes=40,
                show_all_visited=True
            )

            plotly_filename = f"{algorithm_name}_{safe_query}_3d.html"
            plotly_path = output_dir / plotly_filename
            plotly_fig.write_html(str(plotly_path))
            logger.info(f"   ✅ Plotly 3D visualization saved: {plotly_filename}")

        except Exception as e:
            logger.warning(f"   ⚠️ Plotly visualization failed: {str(e)}")

        # Create Matplotlib visualizations (global and sequential)
        try:
            # 2. Global heatmap (complete document view)
            global_fig = create_global_visualization(
                result=result,
                query=query,
                knowledge_graph=kg,
                figure_size=(24, 10),  # Larger for global view
                max_documents=6
            )

            global_filename = f"{algorithm_name}_{safe_query}_global_heatmap.png"
            global_path = output_dir / global_filename
            global_fig.savefig(str(global_path), dpi=300, bbox_inches='tight')
            plt.close(global_fig)
            logger.info(f"   ✅ Global heatmap saved: {global_filename}")

        except Exception as e:
            logger.warning(f"   ⚠️ Global heatmap visualization failed: {str(e)}")

        try:
            # 3. Sequential session heatmap (reading sessions)
            sequential_fig = create_sequential_visualization(
                result=result,
                query=query,
                knowledge_graph=kg,
                figure_size=(20, 8)
            )

            sequential_filename = f"{algorithm_name}_{safe_query}_sequential_heatmap.png"
            sequential_path = output_dir / sequential_filename
            sequential_fig.savefig(str(sequential_path), dpi=300, bbox_inches='tight')
            plt.close(sequential_fig)
            logger.info(f"   ✅ Sequential heatmap saved: {sequential_filename}")

        except Exception as e:
            logger.warning(f"   ⚠️ Sequential heatmap visualization failed: {str(e)}")

    except Exception as e:
        logger.error(f"   ❌ Visualization creation failed: {str(e)}")


def generate_test_golden(kg: KnowledgeGraph, config: Dict[str, Any],
                         logger: logging.Logger) -> tuple[str, ContextGroup]:
    """
    Generate a test golden by creating context group and question using Ollama.

    Returns:
        Tuple of (generated_question, context_group_with_ground_truth_path)
    """
    logger.info("🎯 Generating test golden with context grouping...")

    # Initialize context grouping orchestrator
    context_orchestrator = ContextGroupingOrchestrator(
        kg=kg,
        config=config,
        logger=logger
    )

    # Generate single context group
    logger.info("📋 Generating context group...")
    context_groups = context_orchestrator.generate_context_groups(total_groups=1)

    if not context_groups:
        raise RuntimeError("Failed to generate any context groups")

    context_group = context_groups[0]
    logger.info(f"✅ Generated context group using '{context_group.strategy}' strategy")
    logger.info(f"   Chunks: {len(context_group.chunks)}, Sentences: {len(context_group.sentences)}")
    logger.info(f"   Ground truth traversal path: {len(context_group.traversal_path)} nodes")

    # Initialize model manager and get question generation model
    logger.info("🤖 Initializing Ollama for question generation...")
    model_manager = ModelManager(config, logger)
    question_model = model_manager.get_question_generation_model()

    # Create prompt for question generation
    context_text = " ".join(context_group.chunks)
    # Truncate context if too long (Ollama has token limits)
    if len(context_text) > 3000:
        context_text = context_text[:3000] + "..."

    prompt = f"""Given the following context from a knowledge base, generate a single, clear, and specific question that would require information from this context to answer properly.

Context:
{context_text}

Requirements:
- Generate exactly ONE question
- The question should be specific and answerable from the context
- Make it a natural question someone might ask
- Do not include the answer, only the question
- End with a question mark

Question:"""

    logger.info("🎯 Generating question from context...")
    try:
        generated_response = question_model.generate(prompt)

        # Handle different return types from DeepEval models
        if isinstance(generated_response, tuple):
            # If it's a tuple, take the first element (usually the text response)
            generated_question = generated_response[0]
            logger.debug(f"Received tuple response, using first element: {type(generated_response[0])}")
        elif isinstance(generated_response, str):
            generated_question = generated_response
        else:
            # Convert to string if it's some other type
            generated_question = str(generated_response)
            logger.debug(f"Converting {type(generated_response)} to string")

        # Clean up the response (remove any extra text)
        if isinstance(generated_question, str):
            generated_question = generated_question.strip()
            if not generated_question.endswith('?'):
                generated_question += '?'
        else:
            raise ValueError(f"Expected string after processing, got {type(generated_question)}")

        logger.info(f"✅ Generated question: '{generated_question}'")
        return generated_question, context_group

    except Exception as e:
        logger.error(f"❌ Failed to generate question: {e}")
        logger.debug(
            f"Raw response type: {type(generated_response) if 'generated_response' in locals() else 'undefined'}")
        if 'generated_response' in locals():
            logger.debug(f"Raw response content: {generated_response}")

        # Fallback to a generic question based on the traversal path
        fallback_question = f"What information connects the topics discussed across {len(set([node.split('_')[0] for node in context_group.traversal_path]))} documents?"
        logger.warning(f"Using fallback question: '{fallback_question}'")
        return fallback_question, context_group


def cache_generated_golden(question: str, context_group: ContextGroup, output_dir: Path,
                           logger: logging.Logger) -> None:
    """
    Cache the generated golden in the same format as synthetic dataset.

    Saves the golden as JSON with question, expected_output, and retrieval_context
    in the visualizations directory for analysis.
    """
    import json
    from datetime import datetime

    try:
        logger.info("💾 Caching generated golden...")

        # Create expected output from the context (combine sentences)
        expected_output = " ".join(context_group.sentences)

        # Create golden in DeepEval/synthetic dataset format
        golden_data = {
            "input": question,
            "expected_output": expected_output,
            "retrieval_context": context_group.chunks,

            # Add metadata for analysis
            "metadata": {
                "strategy": context_group.strategy,
                "traversal_path": context_group.traversal_path,
                "chunk_count": len(context_group.chunks),
                "sentence_count": len(context_group.sentences),
                "generated_at": datetime.now().isoformat(),
                "context_metadata": context_group.metadata
            }
        }

        # Save as JSON file in visualizations directory
        golden_filename = f"generated_golden_{context_group.strategy}.json"
        golden_path = output_dir / golden_filename

        with open(golden_path, 'w', encoding='utf-8') as f:
            json.dump(golden_data, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ Golden cached: {golden_filename}")
        logger.info(f"   Question: {question[:80]}...")
        logger.info(f"   Expected output length: {len(expected_output)} chars")
        logger.info(f"   Context chunks: {len(context_group.chunks)}")
        logger.info(f"   Traversal path: {len(context_group.traversal_path)} nodes")

    except Exception as e:
        logger.error(f"❌ Failed to cache golden: {e}")


def create_context_grouping_visualizations(context_group: ContextGroup, kg: KnowledgeGraph,
                                           output_dir: Path, logger: logging.Logger) -> None:
    """Create all 3 matplotlib visualizations for the context grouping process."""
    logger.info("🎨 Creating context grouping visualizations...")

    # Convert ContextGroup to RetrievalResult format for visualization compatibility
    def convert_to_retrieval_result(context_group: ContextGroup, kg: KnowledgeGraph) -> RetrievalResult:
        """Convert ContextGroup to RetrievalResult format for visualization."""
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

    try:
        # Convert to RetrievalResult format
        pseudo_result = convert_to_retrieval_result(context_group, kg)
        query = "Context Grouping Process"

        viz_types = [
            ("windowed", "windowed heatmap"),
            ("global", "global heatmap"),
            ("sequential", "sequential heatmap")
        ]

        for viz_type, viz_name in viz_types:
            try:
                logger.info(f"   📊 Creating {viz_name} for context grouping...")

                fig = create_heatmap_visualization(
                    result=pseudo_result,
                    query=query,
                    knowledge_graph=kg,
                    visualization_type=viz_type
                )

                # Update title to reflect context grouping
                fig.suptitle(
                    f"Context Grouping Traversal - {context_group.strategy.title()} Strategy ({viz_type.title()})",
                    fontsize=16, y=0.95)

                # Save figure
                filename = f"context_grouping_{context_group.strategy}_{viz_type}.png"
                filepath = output_dir / filename
                fig.savefig(str(filepath), dpi=300, bbox_inches='tight')
                plt.close(fig)
                logger.info(f"   ✅ {viz_name.title()} saved: {filename}")

            except Exception as e:
                logger.warning(f"   ⚠️ {viz_name.title()} visualization failed: {str(e)}")

        logger.info(f"🎨 Context grouping visualizations completed")

    except Exception as e:
        logger.error(f"   ❌ Context grouping visualization creation failed: {str(e)}")


def compare_chunk_overlap(ground_truth_context: ContextGroup, algorithm_result,
                          algorithm_name: str, logger: logging.Logger) -> Dict[str, Any]:
    """
    Compare ground truth context with algorithm retrieved content at sentence level.

    Shows how many sentences from the original context group were successfully
    retrieved by the algorithm, accounting for sliding window chunks.

    Returns:
        Dictionary with sentence overlap metrics and analysis
    """
    import re

    def extract_sentences(text_chunks: List[str]) -> set:
        """Extract individual sentences from chunks of text."""
        sentences = set()
        for chunk in text_chunks:
            # Split on sentence boundaries (periods, exclamation marks, question marks)
            # followed by whitespace or end of string
            chunk_sentences = re.split(r'[.!?]+\s+|[.!?]+$', chunk.strip())
            for sentence in chunk_sentences:
                sentence = sentence.strip()
                if sentence:  # Only add non-empty sentences
                    sentences.add(sentence)
        return sentences

    if not algorithm_result or not hasattr(algorithm_result, 'retrieved_content'):
        return {
            'algorithm_name': algorithm_name,
            'sentences_retrieved': 0,
            'total_ground_truth_sentences': len(ground_truth_context.sentences) if hasattr(ground_truth_context, 'sentences') else 0,
            'overlap_percentage': 0.0,
            'analysis': 'Algorithm produced no retrieved content',
            'overlapping_sentences': []
        }

    # Extract sentences from ground truth context
    if hasattr(ground_truth_context, 'sentences') and ground_truth_context.sentences:
        # Use pre-existing sentences if available
        ground_truth_sentences = set(ground_truth_context.sentences)
    else:
        # Fall back to extracting from chunks
        ground_truth_sentences = extract_sentences(ground_truth_context.chunks)

    # Extract sentences from algorithm retrieved content
    algorithm_sentences = extract_sentences(algorithm_result.retrieved_content)

    # Find overlapping sentences (exact text matches)
    overlapping_sentences = ground_truth_sentences.intersection(algorithm_sentences)
    sentences_retrieved = len(overlapping_sentences)
    total_gt_sentences = len(ground_truth_sentences)
    overlap_percentage = (sentences_retrieved / total_gt_sentences) * 100 if total_gt_sentences > 0 else 0

    # Generate analysis text
    if sentences_retrieved > 0:
        analysis = f"Retrieved {sentences_retrieved}/{total_gt_sentences} original context sentences ({overlap_percentage:.1f}%)"
        if overlap_percentage >= 50:
            analysis += " - GOOD OVERLAP ✅"
        elif overlap_percentage >= 25:
            analysis += " - MODERATE OVERLAP ⚠️"
        else:
            analysis += " - LOW OVERLAP ❌"
    else:
        analysis = f"No original context sentences retrieved (0/{total_gt_sentences})"

    return {
        'algorithm_name': algorithm_name,
        'sentences_retrieved': sentences_retrieved,
        'total_ground_truth_sentences': total_gt_sentences,
        'overlap_percentage': overlap_percentage,
        'analysis': analysis,
        'overlapping_sentences': list(overlapping_sentences)
    }


def create_test_queries() -> List[str]:
    """Create fallback test queries for algorithm evaluation."""
    return [
        "How does machine learning relate to neural networks?",
        "How does artificial intelligence impact society?",
        "What are the benefits of renewable energy?",
        "Explain the process of photosynthesis",
        "What causes climate change?"
    ]


def test_individual_algorithm(orchestrator: RetrievalOrchestrator, algorithm_name: str,
                              query: str, kg: KnowledgeGraph, output_dir: Path,
                              logger: logging.Logger) -> None:
    """Test a single algorithm with detailed output."""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Testing {algorithm_name.upper()} Algorithm")
    logger.info(f"{'=' * 60}")
    logger.info(f"Query: '{query}'")
    logger.info(f"{'=' * 60}")

    try:
        result = orchestrator.retrieve(query, algorithm_name)

        logger.info(f"✅ SUCCESS - {algorithm_name}")
        logger.info(f"   Algorithm: {result.algorithm_name}")
        logger.info(f"   Total hops: {result.total_hops}")
        logger.info(f"   Processing time: {result.processing_time:.3f}s")
        logger.info(f"   Final score: {result.final_score:.3f}")
        logger.info(f"   Sentences retrieved: {len(result.retrieved_content)}")

        if result.retrieved_content:
            logger.info(f"   Sample sentences:")
            for i, sentence in enumerate(result.retrieved_content[:3]):
                logger.info(f"     {i + 1}. {sentence[:80]}...")

        if result.traversal_path and result.traversal_path.nodes:
            logger.info(f"   Traversal path: {len(result.traversal_path.nodes)} nodes")
            for i, node in enumerate(result.traversal_path.nodes[:3]):
                logger.info(f"     {i + 1}. {node[:30]}...")

        if hasattr(result, 'metadata') and result.metadata:
            logger.info(f"   Metadata: {result.metadata}")

        # Skip individual visualizations - they'll be created in benchmark phase
        logger.info(f"   📊 Visualizations will be created during benchmark comparison phase")

    except Exception as e:
        logger.error(f"❌ FAILED - {algorithm_name}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())


def run_benchmark_comparison(orchestrator: RetrievalOrchestrator, query: str,
                             kg: KnowledgeGraph, output_dir: Path,
                             logger: logging.Logger, ground_truth_context: ContextGroup = None) -> None:
    """Run all algorithms on the same query for comparison."""
    logger.info(f"\n{'=' * 80}")
    logger.info(f"BENCHMARK COMPARISON")
    logger.info(f"{'=' * 80}")
    logger.info(f"Query: '{query}'")
    logger.info(f"{'=' * 80}")

    try:
        results = orchestrator.benchmark_all_algorithms(query)

        logger.info(f"✅ Benchmark completed for {len(results)} algorithms")
        logger.info(f"\n{'Algorithm':<25} {'Sentences':<10} {'Hops':<6} {'Time':<8} {'Score':<8} {'Status'}")
        logger.info(f"{'-' * 70}")

        for algorithm_name, result in results.items():
            if result.metadata.get('error'):
                status = f"ERROR: {result.metadata['error']}"
                sentences = hops = time_taken = score = "N/A"
            else:
                status = "SUCCESS"
                sentences = len(result.retrieved_content)
                hops = result.total_hops
                time_taken = f"{result.processing_time:.3f}s"
                score = f"{result.final_score:.3f}"

            logger.info(f"{algorithm_name:<25} {sentences:<10} {hops:<6} {time_taken:<8} {score:<8} {status}")

        # Show detailed comparison
        successful_results = {name: result for name, result in results.items()
                              if not result.metadata.get('error')}

        if successful_results:
            logger.info(f"\n📊 DETAILED COMPARISON:")

            # Find algorithm with most sentences
            max_sentences = max(len(result.retrieved_content) for result in successful_results.values())
            best_coverage = [name for name, result in successful_results.items()
                             if len(result.retrieved_content) == max_sentences]
            logger.info(f"   Best coverage: {', '.join(best_coverage)} ({max_sentences} sentences)")

            # Find fastest algorithm
            min_time = min(result.processing_time for result in successful_results.values())
            fastest = [name for name, result in successful_results.items()
                       if result.processing_time == min_time]
            logger.info(f"   Fastest: {', '.join(fastest)} ({min_time:.3f}s)")

            # Find highest scoring algorithm
            max_score = max(result.final_score for result in successful_results.values())
            highest_score = [name for name, result in successful_results.items()
                             if result.final_score == max_score]
            logger.info(f"   Highest score: {', '.join(highest_score)} ({max_score:.3f})")

            # Show unique content analysis
            logger.info(f"\n🔍 CONTENT ANALYSIS:")
            all_sentences = set()
            for name, result in successful_results.items():
                sentences = set(result.retrieved_content)
                all_sentences.update(sentences)
                unique_count = len(sentences - set().union(*[set(r.retrieved_content)
                                                             for n, r in successful_results.items() if n != name]))
                logger.info(f"   {name}: {len(sentences)} total, {unique_count} unique")

            logger.info(f"   Total unique sentences across all algorithms: {len(all_sentences)}")

            # Ground truth sentence overlap comparison (if available)
            if ground_truth_context:
                logger.info(f"\n🎯 GROUND TRUTH SENTENCE OVERLAP ANALYSIS:")
                logger.info(f"   Ground truth strategy: {ground_truth_context.strategy}")
                gt_sentence_count = len(ground_truth_context.sentences) if hasattr(ground_truth_context, 'sentences') else len(ground_truth_context.chunks)
                logger.info(f"   Ground truth context: {gt_sentence_count} sentences")

                chunk_comparisons = []
                for algorithm_name, result in successful_results.items():
                    comparison = compare_chunk_overlap(
                        ground_truth_context,
                        result,
                        algorithm_name,
                        logger
                    )
                    chunk_comparisons.append(comparison)

                # Sort by sentences retrieved (best first)
                chunk_comparisons.sort(key=lambda x: x['sentences_retrieved'], reverse=True)

                for comparison in chunk_comparisons:
                    if comparison['overlap_percentage'] >= 50:
                        symbol = "✅"
                    elif comparison['overlap_percentage'] >= 25:
                        symbol = "⚠️"
                    else:
                        symbol = "❌"

                    logger.info(
                        f"   {symbol} {comparison['algorithm_name']}: {comparison['sentences_retrieved']}/{comparison['total_ground_truth_sentences']} sentences ({comparison['overlap_percentage']:.1f}%)")
                    logger.info(f"      {comparison['analysis']}")

                # Identify best sentence retriever
                if chunk_comparisons:
                    best_match = chunk_comparisons[0]
                    if best_match['sentences_retrieved'] > 0:
                        logger.info(
                            f"\n🏆 BEST SENTENCE OVERLAP: {best_match['algorithm_name']} ({best_match['sentences_retrieved']}/{best_match['total_ground_truth_sentences']} sentences)")
                        logger.info(f"    This algorithm successfully retrieved the most original context sentences!")
                    else:
                        logger.info(f"\n😞 NO SENTENCE OVERLAP: No algorithm retrieved any original context sentences")
                        logger.info(
                            f"    All algorithms found different content than what was used for question generation")

                # Summary statistics
                total_sentences_found = sum(c['sentences_retrieved'] for c in chunk_comparisons)
                total_possible = len(chunk_comparisons) * len(ground_truth_context.sentences) if hasattr(ground_truth_context, 'sentences') else len(chunk_comparisons) * len(ground_truth_context.chunks)
                overall_percentage = (total_sentences_found / total_possible) * 100 if total_possible > 0 else 0
                logger.info(
                    f"\n📊 OVERALL SENTENCE RETRIEVAL: {total_sentences_found}/{total_possible} possible retrievals ({overall_percentage:.1f}% across all algorithms)")
            else:
                logger.info(f"\n💡 No ground truth context available (using fallback query)")

            # Generate visualizations for benchmark results
            logger.info(f"\n🎨 Generating benchmark visualizations...")
            for algorithm_name, result in successful_results.items():
                if not result.metadata.get('error'):
                    create_visualizations(result, query, kg, f"benchmark_{algorithm_name}", output_dir, logger)

    except Exception as e:
        logger.error(f"❌ Benchmark failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())


def main():
    """Main testing function."""
    logger = setup_logging()

    logger.info("🚀 Starting Algorithm Testing Suite")
    logger.info("=" * 50)

    try:
        # Load configuration and knowledge graph
        config = load_config()
        kg = load_knowledge_graph(config, logger)

        # Initialize orchestrator
        orchestrator = RetrievalOrchestrator(kg, config, logger)

        # Setup visualization output directory
        output_dir = setup_visualization_output()
        logger.info(f"\ud83c� Visualizations will be saved to: {output_dir}")

        # Generate test golden with context grouping + Ollama question generation
        try:
            logger.info("\n🧬 GENERATING TEST GOLDEN WITH CONTEXT GROUPING")
            logger.info("=" * 60)
            primary_query, ground_truth_context = generate_test_golden(kg, config, logger)
            logger.info(f"✅ Using generated question: '{primary_query}'")
            logger.info(f"✅ Ground truth path: {len(ground_truth_context.traversal_path)} nodes")

            # Cache the generated golden
            logger.info("\n💾 CACHING GENERATED GOLDEN")
            logger.info("=" * 60)
            cache_generated_golden(primary_query, ground_truth_context, output_dir, logger)

            # Create context grouping visualizations
            logger.info("\n🎨 CREATING CONTEXT GROUPING VISUALIZATIONS")
            logger.info("=" * 60)
            create_context_grouping_visualizations(ground_truth_context, kg, output_dir, logger)
        except Exception as e:
            logger.warning(f"⚠️ Context grouping failed: {e}")
            logger.info("Falling back to hardcoded test queries...")
            test_queries = create_test_queries()
            primary_query = test_queries[0]
            ground_truth_context = None
            logger.info(f"Using fallback query: '{primary_query}'")

        # Test 1: Individual algorithm testing
        logger.info(f"\n🧪 PHASE 1: Individual Algorithm Testing")
        algorithms = [
            "basic_retrieval", 
            "query_traversal", 
            "kg_traversal", 
            "triangulation_average",
            "triangulation_geometric_3d",
            "triangulation_geometric_768d"
        ]

        for algorithm_name in algorithms:
            test_individual_algorithm(orchestrator, algorithm_name, primary_query, kg, output_dir, logger)

        # Test 2: Benchmark comparison (with ground truth path comparison if available)
        logger.info(f"\n🏁 PHASE 2: Benchmark Comparison")
        run_benchmark_comparison(orchestrator, primary_query, kg, output_dir, logger, ground_truth_context)

        # Test 3: Quick test on fallback queries (if context grouping was used, test some hardcoded queries too)
        if ground_truth_context:
            logger.info(f"\n⚡ PHASE 3: Quick Test on Fallback Queries")
            test_queries = create_test_queries()
            for i, query in enumerate(test_queries[:2], 1):  # Test first 2 fallback queries
                logger.info(f"\n--- Fallback Query {i}: '{query}' ---")
                try:
                    # Test just the full-dimension triangulation algorithm on other queries
                    result = orchestrator.retrieve(query, "triangulation_geometric_fulldim")
                    logger.info(f"✅ TriangulationGeometricFullDim: {len(result.retrieved_content)} sentences, "
                                f"{result.total_hops} hops, {result.processing_time:.3f}s")
                except Exception as e:
                    logger.error(f"❌ Failed on fallback query {i}: {str(e)}")
        else:
            logger.info(f"\n⚡ PHASE 3: Quick Test on Remaining Queries")
            test_queries = create_test_queries()
            for i, query in enumerate(test_queries[1:], 2):  # Skip first query (already used)
                logger.info(f"\n--- Query {i}: '{query}' ---")
                try:
                    # Test just the full-dimension triangulation algorithm on other queries
                    result = orchestrator.retrieve(query, "triangulation_geometric_fulldim")
                    logger.info(f"✅ TriangulationGeometricFullDim: {len(result.retrieved_content)} sentences, "
                                f"{result.total_hops} hops, {result.processing_time:.3f}s")
                except Exception as e:
                    logger.error(f"❌ Failed on query {i}: {str(e)}")

        logger.info(f"\n🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        logger.info(f"=" * 50)
        logger.info(f"Summary:")
        logger.info(f"  - Tested {len(algorithms)} algorithms")
        if ground_truth_context:
            logger.info(f"  - Used context-generated question (strategy: {ground_truth_context.strategy})")
            logger.info(f"  - Ground truth path comparison enabled ✅")
        else:
            logger.info(f"  - Used fallback hardcoded questions")
            logger.info(f"  - Ground truth path comparison disabled ❌")
        logger.info(f"  - Knowledge graph: {len(kg.chunks)} chunks, {len(kg.sentences)} sentences")
        logger.info(f"  - Visualizations saved to: {output_dir}")

    except Exception as e:
        logger.error(f"💥 CRITICAL ERROR: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
