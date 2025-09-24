#!/usr/bin/env python3
"""
Quick test for enhanced algorithms with sentence-level anchoring
"""

import sys
import logging
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.knowledge_graph import KnowledgeGraph
from utils.algos.kg_traversal import KGTraversalAlgorithm
from utils.algos.triangulation_centroid import TriangulationCentroidAlgorithm

def setup_logging():
    """Configure logging for test output."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger("EnhancedAlgorithmTest")

def create_test_similarity_cache(kg: KnowledgeGraph, query: str) -> dict:
    """Create a simple similarity cache for testing."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    query_embedding = model.encode([query])[0]

    similarity_cache = {}

    # Add similarities for first 10 chunks
    chunk_ids = list(kg.chunks.keys())[:10]
    for chunk_id in chunk_ids:
        chunk_embedding = kg.get_chunk_embedding(chunk_id)
        if chunk_embedding is not None:
            similarity = float(np.dot(query_embedding, chunk_embedding) /
                             (np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)))
            similarity_cache[chunk_id] = similarity

    # Add similarities for first 20 sentences
    sentence_ids = list(kg.sentences.keys())[:20]
    for sentence_id in sentence_ids:
        sentence_embedding = kg.get_sentence_embedding(sentence_id)
        if sentence_embedding is not None:
            similarity = float(np.dot(query_embedding, sentence_embedding) /
                             (np.linalg.norm(query_embedding) * np.linalg.norm(sentence_embedding)))
            similarity_cache[sentence_id] = similarity

    return similarity_cache

def test_enhanced_algorithms():
    """Test the enhanced algorithms."""
    logger = setup_logging()
    logger.info("🚀 Testing Enhanced Algorithms with Sentence-Level Anchoring")

    # Load knowledge graph
    kg_path = project_root / "data" / "knowledge_graph.json"
    kg = KnowledgeGraph.load(str(kg_path))

    # Load embeddings
    embeddings_path = project_root / "embeddings" / "raw" / "sentence_transformers_all_mpnet_base_v2_multi_granularity.json"
    with open(embeddings_path, 'r') as f:
        import json
        embeddings_data = json.load(f)

    kg.load_phase3_embeddings({"sentence-transformers/all-mpnet-base-v2": embeddings_data['embeddings']})

    logger.info(f"✅ Loaded KG with {len(kg.chunks)} chunks and {len(kg.sentences)} sentences")

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

    # Test query
    query = "How do neural networks learn patterns?"

    # Create similarity cache
    similarity_cache = create_test_similarity_cache(kg, query)

    # Find anchor chunk (highest similarity)
    chunk_similarities = [(cid, sim) for cid, sim in similarity_cache.items() if cid in kg.chunks]
    chunk_similarities.sort(key=lambda x: x[1], reverse=True)
    anchor_chunk = chunk_similarities[0][0]

    logger.info(f"🎯 Test query: '{query}'")
    logger.info(f"🔗 Anchor chunk: {anchor_chunk}")

    # Test KG Traversal with sentence-level anchoring
    logger.info("\n" + "="*60)
    logger.info("🧭 Testing Enhanced KG Traversal Algorithm")
    logger.info("="*60)

    kg_algorithm = KGTraversalAlgorithm(kg, config, similarity_cache, logger)
    kg_result = kg_algorithm.retrieve(query, anchor_chunk)

    logger.info(f"✅ KG Traversal: {len(kg_result.retrieved_content)} sentences, "
               f"{kg_result.total_hops} hops, score={kg_result.final_score:.3f}")

    # Test Triangulation Centroid with multi-vector anchoring
    logger.info("\n" + "="*60)
    logger.info("🔺 Testing Enhanced Triangulation Centroid Algorithm")
    logger.info("="*60)

    triangulation_algorithm = TriangulationCentroidAlgorithm(kg, config, similarity_cache, logger)
    triangulation_result = triangulation_algorithm.retrieve(query, anchor_chunk)

    logger.info(f"✅ Triangulation Centroid: {len(triangulation_result.retrieved_content)} sentences, "
               f"{triangulation_result.total_hops} hops, score={triangulation_result.final_score:.3f}")

    # Summary
    logger.info("\n" + "="*60)
    logger.info("📊 ENHANCED ALGORITHM COMPARISON")
    logger.info("="*60)
    logger.info(f"KG Traversal (sentence anchoring):     {len(kg_result.retrieved_content):2d} sentences, {kg_result.total_hops} hops")
    logger.info(f"Triangulation (multi-vector anchoring): {len(triangulation_result.retrieved_content):2d} sentences, {triangulation_result.total_hops} hops")

    logger.info("🎯 Enhanced algorithm test completed!")

if __name__ == "__main__":
    test_enhanced_algorithms()