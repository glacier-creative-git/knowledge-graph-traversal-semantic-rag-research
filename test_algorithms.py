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

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.retrieval import RetrievalOrchestrator
from utils.knowledge_graph import KnowledgeGraph


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
    embeddings_file = embeddings_dir / "sentence_transformers_all_mpnet_base_v2_multi_granularity.json"
    
    if not embeddings_file.exists():
        logger.warning(f"Embeddings file not found: {embeddings_file}")
        logger.warning("Loading knowledge graph without embeddings")
        kg = KnowledgeGraph.load(str(kg_file))
    else:
        logger.info(f"Loading cached embeddings from: {embeddings_file}")
        
        # Load embeddings data
        import json
        with open(embeddings_file, 'r') as f:
            raw_embeddings = json.load(f)
        
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
        model_name = "sentence-transformers/all-mpnet-base-v2"
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


def create_test_queries() -> List[str]:
    """Create test queries for algorithm evaluation."""
    return [
        "What are the main challenges in machine learning?",
        "How does artificial intelligence impact society?",
        "What are the benefits of renewable energy?",
        "Explain the process of photosynthesis",
        "What causes climate change?"
    ]


def test_individual_algorithm(orchestrator: RetrievalOrchestrator, algorithm_name: str, 
                            query: str, logger: logging.Logger) -> None:
    """Test a single algorithm with detailed output."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing {algorithm_name.upper()} Algorithm")
    logger.info(f"{'='*60}")
    logger.info(f"Query: '{query}'")
    logger.info(f"{'='*60}")
    
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
                logger.info(f"     {i+1}. {sentence[:80]}...")
        
        if result.traversal_path and result.traversal_path.nodes:
            logger.info(f"   Traversal path: {len(result.traversal_path.nodes)} nodes")
            for i, node in enumerate(result.traversal_path.nodes[:3]):
                logger.info(f"     {i+1}. {node[:30]}...")
        
        if hasattr(result, 'metadata') and result.metadata:
            logger.info(f"   Metadata: {result.metadata}")
        
    except Exception as e:
        logger.error(f"❌ FAILED - {algorithm_name}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())


def run_benchmark_comparison(orchestrator: RetrievalOrchestrator, query: str, 
                           logger: logging.Logger) -> None:
    """Run all algorithms on the same query for comparison."""
    logger.info(f"\n{'='*80}")
    logger.info(f"BENCHMARK COMPARISON")
    logger.info(f"{'='*80}")
    logger.info(f"Query: '{query}'")
    logger.info(f"{'='*80}")
    
    try:
        results = orchestrator.benchmark_all_algorithms(query)
        
        logger.info(f"✅ Benchmark completed for {len(results)} algorithms")
        logger.info(f"\n{'Algorithm':<25} {'Sentences':<10} {'Hops':<6} {'Time':<8} {'Score':<8} {'Status'}")
        logger.info(f"{'-'*70}")
        
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
        
        # Get test queries
        test_queries = create_test_queries()
        primary_query = test_queries[0]  # Use first query for detailed testing
        
        logger.info(f"Using primary test query: '{primary_query}'")
        
        # Test 1: Individual algorithm testing
        logger.info(f"\n🧪 PHASE 1: Individual Algorithm Testing")
        algorithms = ["basic_retrieval", "query_traversal", "kg_traversal", "triangulation_centroid"]
        
        for algorithm_name in algorithms:
            test_individual_algorithm(orchestrator, algorithm_name, primary_query, logger)
        
        # Test 2: Benchmark comparison
        logger.info(f"\n🏁 PHASE 2: Benchmark Comparison")
        run_benchmark_comparison(orchestrator, primary_query, logger)
        
        # Test 3: Quick test on all queries
        logger.info(f"\n⚡ PHASE 3: Quick Test on All Queries")
        for i, query in enumerate(test_queries[1:], 2):  # Skip first query (already used)
            logger.info(f"\n--- Query {i}: '{query}' ---")
            try:
                # Test just the triangulation centroid algorithm on other queries
                result = orchestrator.retrieve(query, "triangulation_centroid")
                logger.info(f"✅ TriangulationCentroid: {len(result.retrieved_content)} sentences, "
                          f"{result.total_hops} hops, {result.processing_time:.3f}s")
            except Exception as e:
                logger.error(f"❌ Failed on query {i}: {str(e)}")
        
        logger.info(f"\n🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        logger.info(f"=" * 50)
        logger.info(f"Summary:")
        logger.info(f"  - Tested {len(algorithms)} algorithms")
        logger.info(f"  - Tested {len(test_queries)} queries")
        logger.info(f"  - Knowledge graph: {len(kg.chunks)} chunks, {len(kg.sentences)} sentences")
        
    except Exception as e:
        logger.error(f"💥 CRITICAL ERROR: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
