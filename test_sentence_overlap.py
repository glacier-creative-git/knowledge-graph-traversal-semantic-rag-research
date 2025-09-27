#!/usr/bin/env python3
"""
Quick test script to verify sentence overlap analysis is working correctly.
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from evaluation.context_grouping import ContextGroup
from utils.algos.base_algorithm import RetrievalResult
from utils.traversal import TraversalPath, GranularityLevel, ConnectionType

def setup_logging() -> logging.Logger:
    """Setup simple logging."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    return logging.getLogger("SentenceOverlapTest")

def compare_chunk_overlap(ground_truth_context: ContextGroup, algorithm_result,
                          algorithm_name: str, logger: logging.Logger) -> Dict[str, Any]:
    """
    Compare ground truth context with algorithm retrieved content at sentence level.
    """
    import re

    def extract_sentences(text_chunks: List[str]) -> set:
        """Extract individual sentences from chunks of text."""
        sentences = set()
        for chunk in text_chunks:
            # Split on sentence boundaries (periods, exclamation marks, question marks)
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

def create_test_data():
    """Create test data to verify sentence overlap analysis."""

    # Create mock ground truth context
    ground_truth_chunks = [
        "Machine learning is a subset of AI. It involves training algorithms on data. These algorithms can then make predictions.",
        "Neural networks are inspired by the brain. They consist of interconnected nodes. Each node processes information like a neuron.",
        "Deep learning uses multi-layer networks. These networks can learn complex patterns. They are particularly good at image recognition."
    ]

    ground_truth_sentences = [
        "Machine learning is a subset of AI",
        "It involves training algorithms on data",
        "These algorithms can then make predictions",
        "Neural networks are inspired by the brain",
        "They consist of interconnected nodes",
        "Each node processes information like a neuron",
        "Deep learning uses multi-layer networks",
        "These networks can learn complex patterns",
        "They are particularly good at image recognition"
    ]

    # Create ContextGroup
    ground_truth = ContextGroup(
        chunks=ground_truth_chunks,
        chunk_ids=["chunk1", "chunk2", "chunk3"],
        sentences=ground_truth_sentences,
        strategy="test_strategy",
        traversal_path=["chunk1", "chunk2", "chunk3"],
        metadata={}
    )

    # Test Case 1: Perfect overlap - algorithm retrieves exact same sentences
    perfect_algorithm_result = type('obj', (object,), {
        'retrieved_content': ground_truth_sentences[:5],  # First 5 sentences
        'algorithm_name': 'perfect_overlap'
    })()

    # Test Case 2: Partial overlap - algorithm retrieves some overlapping sentences in different chunks
    partial_chunks = [
        "Machine learning is a subset of AI. It uses statistical methods for learning.",  # 1 overlap
        "Neural networks are inspired by the brain. They have multiple layers of computation.",  # 1 overlap
        "Computer vision is an important application. It can identify objects in images."  # 0 overlap
    ]
    partial_algorithm_result = type('obj', (object,), {
        'retrieved_content': partial_chunks,
        'algorithm_name': 'partial_overlap'
    })()

    # Test Case 3: No overlap - completely different content
    no_overlap_chunks = [
        "Natural language processing handles text data. It can understand human language.",
        "Reinforcement learning uses rewards and penalties. Agents learn through trial and error."
    ]
    no_overlap_algorithm_result = type('obj', (object,), {
        'retrieved_content': no_overlap_chunks,
        'algorithm_name': 'no_overlap'
    })()

    return ground_truth, [
        (perfect_algorithm_result, "Perfect Overlap"),
        (partial_algorithm_result, "Partial Overlap"),
        (no_overlap_algorithm_result, "No Overlap")
    ]

def main():
    """Run sentence overlap tests."""
    logger = setup_logging()

    logger.info("🧪 Testing Sentence Overlap Analysis")
    logger.info("=" * 50)

    # Create test data
    ground_truth, test_cases = create_test_data()

    logger.info(f"📋 Ground Truth Context:")
    logger.info(f"   Chunks: {len(ground_truth.chunks)}")
    logger.info(f"   Sentences: {len(ground_truth.sentences)}")
    logger.info(f"   Sample sentences:")
    for i, sent in enumerate(ground_truth.sentences[:3]):
        logger.info(f"     {i+1}. {sent}")
    logger.info("")

    # Test each case
    for algorithm_result, test_name in test_cases:
        logger.info(f"🔍 Testing: {test_name}")
        logger.info(f"   Algorithm retrieved content:")
        for i, content in enumerate(algorithm_result.retrieved_content[:2]):
            logger.info(f"     {i+1}. {content[:60]}...")

        # Run comparison
        comparison = compare_chunk_overlap(
            ground_truth,
            algorithm_result,
            algorithm_result.algorithm_name,
            logger
        )

        # Show results
        logger.info(f"   📊 Results:")
        logger.info(f"      Sentences retrieved: {comparison['sentences_retrieved']}/{comparison['total_ground_truth_sentences']}")
        logger.info(f"      Overlap percentage: {comparison['overlap_percentage']:.1f}%")
        logger.info(f"      Analysis: {comparison['analysis']}")

        if comparison['overlapping_sentences']:
            logger.info(f"      Overlapping sentences:")
            for sent in comparison['overlapping_sentences']:
                logger.info(f"        - {sent}")

        logger.info("")

    logger.info("✅ Sentence overlap testing completed!")

if __name__ == "__main__":
    main()