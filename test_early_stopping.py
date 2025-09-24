#!/usr/bin/env python3
"""
Early Stopping Algorithm Test Script
===================================

Tests all four algorithms with enhanced early stopping on the synthetic dataset.
Compares extraction behavior and early stopping frequency across algorithms.
"""

import json
import sys
import logging
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.knowledge_graph import KnowledgeGraph
from utils.algos.basic_retrieval import BasicRetrievalAlgorithm
from utils.algos.query_traversal import QueryTraversalAlgorithm
from utils.algos.kg_traversal import KGTraversalAlgorithm
from utils.algos.triangulation_centroid import TriangulationCentroidAlgorithm

# Global model instance - load once and reuse
_sentence_transformer_model = None


def setup_logging():
    """Configure logging for test output."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("EarlyStoppingTest")


def load_test_data():
    """Load synthetic dataset and knowledge graph."""
    logger = logging.getLogger("EarlyStoppingTest")

    # Load synthetic dataset
    dataset_path = project_root / "data" / "synthetic_dataset.json"
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    # Load knowledge graph
    kg_path = project_root / "data" / "knowledge_graph.json"
    kg = KnowledgeGraph.load(str(kg_path))

    # Load cached embeddings for similarity calculations
    embeddings_path = project_root / "embeddings" / "raw" / "sentence_transformers_all_mpnet_base_v2_multi_granularity.json"
    with open(embeddings_path, 'r') as f:
        embeddings_data = json.load(f)

    # Load embeddings into knowledge graph cache
    kg.load_phase3_embeddings({"sentence-transformers/all-mpnet-base-v2": embeddings_data['embeddings']})

    logger.info(f"✅ Loaded {len(dataset)} test cases and KG with {len(kg.chunks)} chunks")
    return dataset, kg


def get_sentence_transformer_model():
    """Get or initialize the global sentence transformer model."""
    global _sentence_transformer_model
    if _sentence_transformer_model is None:
        from sentence_transformers import SentenceTransformer
        logging.getLogger("EarlyStoppingTest").info("🔄 Loading SentenceTransformer model (one-time initialization)...")
        _sentence_transformer_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        logging.getLogger("EarlyStoppingTest").info("✅ SentenceTransformer model loaded and cached")
    return _sentence_transformer_model


def create_similarity_cache(kg: KnowledgeGraph, query: str) -> Dict[str, float]:
    """Create query similarity cache for algorithms using cached model."""
    # Use the cached model instead of loading each time
    model = get_sentence_transformer_model()
    query_embedding = model.encode([query])[0]

    similarity_cache = {}

    # Calculate similarities for chunks
    for chunk_id in kg.chunks.keys():
        chunk_embedding = kg.get_chunk_embedding(chunk_id)
        if chunk_embedding is not None:
            similarity = float(np.dot(query_embedding, chunk_embedding) /
                             (np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)))
            similarity_cache[chunk_id] = similarity

    # Calculate similarities for sentences
    for sentence_id in kg.sentences.keys():
        sentence_embedding = kg.get_sentence_embedding(sentence_id)
        if sentence_embedding is not None:
            similarity = float(np.dot(query_embedding, sentence_embedding) /
                             (np.linalg.norm(query_embedding) * np.linalg.norm(sentence_embedding)))
            similarity_cache[sentence_id] = similarity

    return similarity_cache


def test_algorithm(algorithm_class, algorithm_name: str, kg: KnowledgeGraph,
                  test_case: Dict, config: Dict) -> Dict[str, Any]:
    """Test a single algorithm on a test case."""
    logger = logging.getLogger("EarlyStoppingTest")

    query = test_case['input']

    # Create similarity cache
    similarity_cache = create_similarity_cache(kg, query)

    # Find anchor chunk (highest similarity)
    chunk_similarities = [(cid, sim) for cid, sim in similarity_cache.items() if cid in kg.chunks]
    chunk_similarities.sort(key=lambda x: x[1], reverse=True)
    anchor_chunk = chunk_similarities[0][0]

    # Initialize algorithm
    algorithm = algorithm_class(kg, config, similarity_cache, logger)

    # Run retrieval
    logger.info(f"\n🔍 Testing {algorithm_name} on: '{query[:60]}...'")
    result = algorithm.retrieve(query, anchor_chunk)

    # Extract early stopping information
    early_stopping_info = {
        'algorithm': algorithm_name,
        'sentences_extracted': len(result.retrieved_content),
        'hops_taken': result.total_hops,
        'final_score': result.final_score,
        'processing_time': result.processing_time,
        'early_stopping_detected': False,
        'termination_reason': 'unknown'
    }

    # Check metadata for early stopping indicators
    if hasattr(result, 'metadata') and result.metadata:
        metadata = result.metadata

        # Check for explicit early stopping flags in metadata
        if 'early_stop_triggered' in metadata:
            early_stopping_info['early_stopping_detected'] = metadata['early_stop_triggered']
            early_stopping_info['termination_reason'] = 'early_stop_flag'

        # Check for algorithm-specific early stopping indicators
        elif any(key in metadata for key in ['early_stopping_triggered', 'precision_termination']):
            early_stopping_info['early_stopping_detected'] = True
            early_stopping_info['termination_reason'] = 'algorithm_specific'

        # Extract specific termination info
        if 'termination_reason' in metadata:
            early_stopping_info['termination_reason'] = metadata['termination_reason']

    # Only detect early stopping from explicit algorithm flags - no pattern inference
    # Real early stopping means the algorithm found high-precision content and terminated

    logger.info(f"✅ {algorithm_name}: {early_stopping_info['sentences_extracted']} sentences, "
              f"{early_stopping_info['hops_taken']} hops, "
              f"score={early_stopping_info['final_score']:.3f}")

    return early_stopping_info


def main():
    """Main test execution."""
    logger = setup_logging()
    logger.info("🚀 Starting Early Stopping Algorithm Test")

    # Preload the sentence transformer model to avoid reloading
    get_sentence_transformer_model()

    # Load test data
    dataset, kg = load_test_data()

    # Algorithm configuration - AGGRESSIVE SETTINGS to force early stopping detection
    config = {
        'retrieval': {
            'semantic_traversal': {
                'max_hops': 50,           # Much higher hop limit
                'similarity_threshold': 0.01,  # Lower threshold to allow more exploration
                'max_results': 100,        # Much higher sentence target
                'min_sentence_threshold': 50,  # Force algorithms to aim for 50 sentences
                'enable_early_stopping': True,
                'max_safety_hops': 30     # Safety limit for infinite loops
            }
        }
    }

    # Algorithm classes to test
    algorithms = [
        (BasicRetrievalAlgorithm, "Basic Retrieval"),
        (QueryTraversalAlgorithm, "Query Traversal"),
        (KGTraversalAlgorithm, "KG Traversal"),
        (TriangulationCentroidAlgorithm, "Triangulation Centroid")
    ]

    # Test results storage
    all_results = []

    # Test each algorithm on each test case
    for i, test_case in enumerate(dataset[:10]):  # Test on first 3 cases
        logger.info(f"\n{'='*80}")
        logger.info(f"📊 TEST CASE {i+1}/{min(3, len(dataset))}")
        logger.info(f"Query: {test_case['input']}")
        logger.info(f"{'='*80}")

        case_results = {'test_case_index': i, 'query': test_case['input'], 'algorithms': {}}

        for algorithm_class, algorithm_name in algorithms:
            try:
                result = test_algorithm(algorithm_class, algorithm_name, kg, test_case, config)
                case_results['algorithms'][algorithm_name] = result

            except Exception as e:
                logger.error(f"❌ {algorithm_name} failed: {e}")
                case_results['algorithms'][algorithm_name] = {
                    'algorithm': algorithm_name,
                    'error': str(e),
                    'sentences_extracted': 0,
                    'early_stopping_detected': False
                }

        all_results.append(case_results)

    # Summary analysis
    logger.info(f"\n{'='*80}")
    logger.info("📈 EARLY STOPPING ANALYSIS SUMMARY")
    logger.info(f"{'='*80}")

    early_stopping_counts = {}
    avg_sentences = {}
    avg_hops = {}

    for algorithm_class, algorithm_name in algorithms:
        early_stops = 0
        total_sentences = 0
        total_hops = 0
        valid_runs = 0

        for case in all_results:
            if algorithm_name in case['algorithms'] and 'error' not in case['algorithms'][algorithm_name]:
                result = case['algorithms'][algorithm_name]
                if result.get('early_stopping_detected', False):
                    early_stops += 1
                total_sentences += result.get('sentences_extracted', 0)
                total_hops += result.get('hops_taken', 0)
                valid_runs += 1

        if valid_runs > 0:
            early_stopping_counts[algorithm_name] = f"{early_stops}/{valid_runs}"
            avg_sentences[algorithm_name] = total_sentences / valid_runs
            avg_hops[algorithm_name] = total_hops / valid_runs
        else:
            early_stopping_counts[algorithm_name] = "0/0"
            avg_sentences[algorithm_name] = 0
            avg_hops[algorithm_name] = 0

    # Print summary table
    logger.info("\nAlgorithm Performance Comparison:")
    logger.info("-" * 80)
    logger.info(f"{'Algorithm':<25} {'Early Stops':<12} {'Avg Sentences':<15} {'Avg Hops':<10}")
    logger.info("-" * 80)

    for algorithm_class, algorithm_name in algorithms:
        logger.info(f"{algorithm_name:<25} {early_stopping_counts.get(algorithm_name, 'N/A'):<12} "
                   f"{avg_sentences.get(algorithm_name, 0):<15.1f} {avg_hops.get(algorithm_name, 0):<10.1f}")

    # Save detailed results
    results_path = project_root / "early_stopping_test_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\n💾 Detailed results saved to: {results_path}")
    logger.info("🎯 Early stopping test completed!")


if __name__ == "__main__":
    main()