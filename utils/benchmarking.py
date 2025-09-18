#!/usr/bin/env python3
"""
Advanced RAG Benchmarking System
===============================

Comprehensive benchmarking utilities for evaluating RAG retrieval algorithms.
Implements both traditional Information Retrieval metrics and novel path-based
metrics that leverage rich ground truth traversal data.

Key Features:
- Traditional IR metrics: Precision@K, Recall@K, F1@K, MRR, NDCG@K, MAP@K  
- Path-based metrics: Node overlap, sequence accuracy, hop efficiency
- Question-type-specific evaluation with weighted scoring
- Difficulty-stratified analysis capabilities
- Rich performance analytics and comparative scoring
"""

import logging
import numpy as np
import yaml
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime

from utils.questions import GeneratedQuestion, EvaluationDataset
from utils.retrieval import RetrievalResult


@dataclass
class MetricResult:
    """Container for individual metric calculation results."""
    name: str
    value: float
    description: str
    higher_is_better: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper numpy scalar handling."""
        return {
            'name': self.name,
            'value': float(self.value),  # Convert numpy scalars to Python floats
            'description': self.description,
            'higher_is_better': self.higher_is_better
        }


@dataclass
class AlgorithmBenchmarkResult:
    """Complete benchmark results for a single algorithm on a single question."""
    algorithm_name: str
    question_id: str
    question_type: str
    difficulty_level: str
    question_text: str  # Add missing question text field
    expected_answer: str  # Add expected answer field
    
    # Traditional IR metrics
    traditional_metrics: Dict[str, MetricResult]
    
    # Path-based metrics (unique to this system)
    path_metrics: Dict[str, MetricResult]
    
    # Performance metrics
    performance_metrics: Dict[str, MetricResult]
    
    # Raw retrieval data for debugging
    retrieved_nodes: List[str]
    ground_truth_nodes: List[str]
    processing_time: float
    total_hops: int
    
    def _get_content_preview(self) -> List[str]:
        """Get a preview of the actual retrieved content for human analysis."""
        # This is a placeholder - you'll need to implement based on your knowledge graph structure
        # The goal is to show the first ~50 chars of each retrieved chunk's content
        return [f"Preview of {node_id}..." for node_id in self.retrieved_nodes[:3]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YAML serialization with clean numpy handling and human-readable format."""
        return {
            'algorithm_name': self.algorithm_name,
            'question_metadata': {
                'question_id': self.question_id,
                'question_type': self.question_type,
                'difficulty_level': self.difficulty_level,
                'question_text': self.question_text[:100] + "..." if len(self.question_text) > 100 else self.question_text,  # Truncate for readability
                'expected_answer': self.expected_answer[:100] + "..." if len(self.expected_answer) > 100 else self.expected_answer
            },
            'metrics_summary': {
                # Clean, concise metrics with format: "P@10: 0.400 (2/5)"
                'precision_at_10': f"{float(self.traditional_metrics['precision_at_k'].value):.3f}",
                'recall_at_10': f"{float(self.traditional_metrics['recall_at_k'].value):.3f}", 
                'f1_at_10': f"{float(self.traditional_metrics['f1_at_k'].value):.3f}",
                'mrr': f"{float(self.traditional_metrics['mrr'].value):.3f}",
                'ndcg_at_10': f"{float(self.traditional_metrics['ndcg_at_k'].value):.3f}",
                'node_overlap_ratio': f"{float(self.path_metrics['node_overlap_ratio'].value):.3f}",
                'path_sequence_accuracy': f"{float(self.path_metrics['path_sequence_accuracy'].value):.3f}",
                'hop_efficiency': f"{float(self.path_metrics['hop_efficiency'].value):.3f}",
                'processing_time_seconds': f"{float(self.performance_metrics['processing_time'].value):.4f}"
            },
            'traditional_metrics': {
                name: metric.to_dict() for name, metric in self.traditional_metrics.items()
            },
            'path_metrics': {
                name: metric.to_dict() for name, metric in self.path_metrics.items()
            },
            'performance_metrics': {
                name: metric.to_dict() for name, metric in self.performance_metrics.items()
            },
            'raw_data': {
                'retrieved_nodes': list(self.retrieved_nodes),  # Ensure it's a clean list
                'ground_truth_nodes': list(self.ground_truth_nodes),  # Ensure it's a clean list
                'processing_time': float(self.processing_time),
                'total_hops': self.total_hops,
                'retrieved_content_preview': self._get_content_preview()  # Add content preview
            }
        }


@dataclass 
class CompleteBenchmarkResult:
    """Complete benchmark results across all algorithms and questions."""
    benchmark_metadata: Dict[str, Any]
    question_results: Dict[str, Dict[str, AlgorithmBenchmarkResult]]  # question_id -> algorithm -> result
    aggregate_scores: Dict[str, Dict[str, float]]  # algorithm -> metric -> score
    
    def _calculate_generation_success_rates(self) -> Dict[str, str]:
        """Calculate question generation success rates by type."""
        question_counts = {}
        for q_results in self.question_results.values():
            # Get question type from first algorithm result
            first_result = next(iter(q_results.values()))
            q_type = first_result.question_type
            question_counts[q_type] = question_counts.get(q_type, 0) + 1
        
        # Format as readable percentages (assuming target distribution)
        target_distribution = self.benchmark_metadata.get('question_distribution', {}).get('by_type', {})
        success_rates = {}
        
        for q_type in ['single_hop', 'sequential_flow', 'multi_hop', 'theme_hop', 'hierarchical']:
            actual = question_counts.get(q_type, 0)
            target = target_distribution.get(q_type, 0)
            if target > 0:
                success_rate = (actual / target) * 100
                success_rates[q_type] = f"{actual}/{target} ({success_rate:.1f}%)"
            else:
                success_rates[q_type] = f"{actual}/0 (N/A)"
        
        return success_rates
    
    def _calculate_average_question_length(self) -> float:
        """Calculate average question text length."""
        total_length = 0
        count = 0
        
        for q_results in self.question_results.values():
            first_result = next(iter(q_results.values()))
            total_length += len(first_result.question_text)
            count += 1
        
        return total_length / count if count > 0 else 0
    
    def _get_sample_questions(self) -> List[Dict[str, str]]:
        """Get sample questions for quality review."""
        samples = []
        question_types_seen = set()
        
        for q_results in self.question_results.values():
            first_result = next(iter(q_results.values()))
            q_type = first_result.question_type
            
            # Include first question of each type
            if q_type not in question_types_seen and len(samples) < 5:
                samples.append({
                    'type': q_type,
                    'text': first_result.question_text,
                    'difficulty': first_result.difficulty_level,
                    'expected_nodes': len(first_result.ground_truth_nodes)
                })
                question_types_seen.add(q_type)
        
        return samples
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YAML serialization with scoring transparency."""
        # Calculate best performing algorithm for ranking
        best_algorithm = None
        best_score = -1
        algorithm_ranking = []
        
        for algorithm, scores in self.aggregate_scores.items():
            composite_score = scores.get('composite_score_overall', 0)
            algorithm_ranking.append((algorithm, composite_score))
            if composite_score > best_score:
                best_score = composite_score
                best_algorithm = algorithm
        
        # Sort algorithms by composite score (descending)
        algorithm_ranking.sort(key=lambda x: x[1], reverse=True)
        performance_ranking = [alg for alg, score in algorithm_ranking]
        
        return {
            'benchmark_metadata': self.benchmark_metadata,
            'scoring_methodology': {
                'composite_score_formula': "weighted_avg(precision*0.25 + recall*0.20 + mrr*0.20 + node_overlap*0.25 + speed_bonus*0.10)",
                'metric_weights': {
                    'precision_at_k': 0.25,
                    'recall_at_k': 0.20, 
                    'mrr': 0.20,
                    'node_overlap_ratio': 0.25,
                    'processing_time_bonus': 0.10
                },
                'scoring_notes': [
                    "Processing time is converted to bonus using inverse relationship: 1/(1+time)",
                    "Path-based metrics (node_overlap, sequence_accuracy) are weighted higher for complex question types",
                    "Composite scores are calculated per question type, then averaged for overall score"
                ],
                'best_performing_algorithm': best_algorithm,
                'performance_ranking': performance_ranking
            },
            'question_quality_summary': {
                'total_questions_generated': len(self.question_results),
                'question_distribution': self.benchmark_metadata.get('question_distribution', {}),
                'generation_success_rates': self._calculate_generation_success_rates(),
                'average_question_length': self._calculate_average_question_length(),
                'sample_questions': self._get_sample_questions()
            },
            'question_results': {
                q_id: {alg: result.to_dict() for alg, result in alg_results.items()}
                for q_id, alg_results in self.question_results.items()
            },
            'aggregate_scores': self.aggregate_scores
        }


class TraditionalIRMetrics:
    """Implementation of traditional Information Retrieval metrics adapted for RAG evaluation."""
    
    @staticmethod
    def precision_at_k(retrieved_nodes: List[str], ground_truth_nodes: List[str], k: int = 10) -> MetricResult:
        """
        Calculate Precision@K: fraction of retrieved items that are relevant.
        
        Args:
            retrieved_nodes: List of retrieved node IDs (up to k items)
            ground_truth_nodes: List of ground truth relevant node IDs  
            k: Number of top results to consider
            
        Returns:
            MetricResult with precision score (0.0 to 1.0)
        """
        retrieved_k = retrieved_nodes[:k]
        relevant_retrieved = len([node for node in retrieved_k if node in ground_truth_nodes])
        
        precision = relevant_retrieved / len(retrieved_k) if retrieved_k else 0.0
        
        return MetricResult(
            name="precision_at_k",
            value=precision,
            description=f"Precision@{k}: {relevant_retrieved}/{len(retrieved_k)} retrieved items were relevant",
            higher_is_better=True
        )
    
    @staticmethod 
    def recall_at_k(retrieved_nodes: List[str], ground_truth_nodes: List[str], k: int = 10) -> MetricResult:
        """
        Calculate Recall@K: fraction of relevant items that were retrieved.
        
        Args:
            retrieved_nodes: List of retrieved node IDs (up to k items)
            ground_truth_nodes: List of ground truth relevant node IDs
            k: Number of top results to consider
            
        Returns:
            MetricResult with recall score (0.0 to 1.0)
        """
        retrieved_k = retrieved_nodes[:k]
        relevant_retrieved = len([node for node in retrieved_k if node in ground_truth_nodes])
        
        recall = relevant_retrieved / len(ground_truth_nodes) if ground_truth_nodes else 0.0
        
        return MetricResult(
            name="recall_at_k",
            value=recall,
            description=f"Recall@{k}: {relevant_retrieved}/{len(ground_truth_nodes)} relevant items were retrieved",
            higher_is_better=True
        )
    
    @staticmethod
    def f1_at_k(retrieved_nodes: List[str], ground_truth_nodes: List[str], k: int = 10) -> MetricResult:
        """
        Calculate F1@K: harmonic mean of precision and recall.
        
        Args:
            retrieved_nodes: List of retrieved node IDs (up to k items)
            ground_truth_nodes: List of ground truth relevant node IDs
            k: Number of top results to consider
            
        Returns:
            MetricResult with F1 score (0.0 to 1.0)
        """
        precision_result = TraditionalIRMetrics.precision_at_k(retrieved_nodes, ground_truth_nodes, k)
        recall_result = TraditionalIRMetrics.recall_at_k(retrieved_nodes, ground_truth_nodes, k)
        
        precision = precision_result.value
        recall = recall_result.value
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return MetricResult(
            name="f1_at_k",
            value=f1,
            description=f"F1@{k}: harmonic mean of precision ({precision:.3f}) and recall ({recall:.3f})",
            higher_is_better=True
        )
    
    @staticmethod
    def mean_reciprocal_rank(retrieved_nodes: List[str], ground_truth_nodes: List[str]) -> MetricResult:
        """
        Calculate Mean Reciprocal Rank: 1/rank of first relevant item.
        
        Args:
            retrieved_nodes: List of retrieved node IDs (ordered by relevance)
            ground_truth_nodes: List of ground truth relevant node IDs
            
        Returns:
            MetricResult with MRR score (0.0 to 1.0)
        """
        for rank, node in enumerate(retrieved_nodes, 1):
            if node in ground_truth_nodes:
                mrr = 1.0 / rank
                return MetricResult(
                    name="mrr",
                    value=mrr,
                    description=f"MRR: first relevant item found at rank {rank}",
                    higher_is_better=True
                )
        
        return MetricResult(
            name="mrr", 
            value=0.0,
            description="MRR: no relevant items found in retrieved results",
            higher_is_better=True
        )
    
    @staticmethod
    def ndcg_at_k(retrieved_nodes: List[str], ground_truth_nodes: List[str], k: int = 10) -> MetricResult:
        """
        Calculate Normalized Discounted Cumulative Gain@K.
        
        Uses graded relevance where earlier ground truth nodes have higher relevance scores.
        
        Args:
            retrieved_nodes: List of retrieved node IDs (up to k items)
            ground_truth_nodes: List of ground truth relevant node IDs (ordered by importance)
            k: Number of top results to consider
            
        Returns:
            MetricResult with NDCG score (0.0 to 1.0)
        """
        def get_relevance_score(node: str, ground_truth: List[str]) -> float:
            """Get graded relevance score. Earlier nodes in path are more relevant."""
            if node not in ground_truth:
                return 0.0
            # Higher score for nodes that appear earlier in the ground truth path
            position = ground_truth.index(node)
            max_relevance = len(ground_truth)
            return (max_relevance - position) / max_relevance
        
        def calculate_dcg(nodes: List[str], ground_truth: List[str]) -> float:
            """Calculate Discounted Cumulative Gain."""
            dcg = 0.0
            for i, node in enumerate(nodes):
                relevance = get_relevance_score(node, ground_truth)
                if relevance > 0:
                    dcg += relevance / np.log2(i + 2)  # +2 because log2(1) = 0
            return dcg
        
        retrieved_k = retrieved_nodes[:k]
        
        # Calculate DCG for retrieved results
        dcg = calculate_dcg(retrieved_k, ground_truth_nodes)
        
        # Calculate IDCG (ideal DCG) - best possible ordering
        ideal_order = ground_truth_nodes[:k]  # Ground truth is already in optimal order
        idcg = calculate_dcg(ideal_order, ground_truth_nodes)
        
        ndcg = dcg / idcg if idcg > 0 else 0.0
        
        return MetricResult(
            name="ndcg_at_k",
            value=ndcg,
            description=f"NDCG@{k}: {dcg:.3f}/{idcg:.3f} considering graded relevance and ranking",
            higher_is_better=True
        )
    
    @staticmethod
    def map_at_k(retrieved_nodes: List[str], ground_truth_nodes: List[str], k: int = 10) -> MetricResult:
        """
        Calculate Mean Average Precision@K.
        
        Args:
            retrieved_nodes: List of retrieved node IDs (up to k items) 
            ground_truth_nodes: List of ground truth relevant node IDs
            k: Number of top results to consider
            
        Returns:
            MetricResult with MAP score (0.0 to 1.0)
        """
        retrieved_k = retrieved_nodes[:k]
        
        if not ground_truth_nodes:
            return MetricResult(
                name="map_at_k",
                value=0.0,
                description=f"MAP@{k}: no ground truth items to evaluate",
                higher_is_better=True
            )
        
        precision_at_i = []
        relevant_found = 0
        
        for i, node in enumerate(retrieved_k):
            if node in ground_truth_nodes:
                relevant_found += 1
                precision_at_i.append(relevant_found / (i + 1))
        
        # Average precision is mean of precision values at each relevant item
        avg_precision = np.mean(precision_at_i) if precision_at_i else 0.0
        
        return MetricResult(
            name="map_at_k",
            value=avg_precision,
            description=f"MAP@{k}: average precision across {len(precision_at_i)} relevant items found",
            higher_is_better=True
        )


class PathBasedMetrics:
    """Novel metrics that leverage the rich ground truth path information."""
    
    @staticmethod
    def node_overlap_ratio(retrieved_nodes: List[str], ground_truth_nodes: List[str]) -> MetricResult:
        """
        Calculate the ratio of ground truth nodes that were retrieved.
        
        This is different from recall in that it doesn't consider K - it looks at
        all retrieved nodes regardless of ranking.
        
        Args:
            retrieved_nodes: All retrieved node IDs
            ground_truth_nodes: Ground truth path nodes
            
        Returns:
            MetricResult with overlap ratio (0.0 to 1.0)
        """
        retrieved_set = set(retrieved_nodes)
        ground_truth_set = set(ground_truth_nodes)
        
        overlap = len(retrieved_set & ground_truth_set)
        total_ground_truth = len(ground_truth_set)
        
        ratio = overlap / total_ground_truth if total_ground_truth > 0 else 0.0
        
        return MetricResult(
            name="node_overlap_ratio",
            value=ratio,
            description=f"Node Overlap: {overlap}/{total_ground_truth} ground truth nodes were retrieved",
            higher_is_better=True
        )
    
    @staticmethod
    def path_sequence_accuracy(retrieved_nodes: List[str], ground_truth_nodes: List[str]) -> MetricResult:
        """
        Calculate how well the retrieval follows the expected sequence of the ground truth path.
        
        Uses Longest Common Subsequence (LCS) to measure sequence preservation.
        
        Args:
            retrieved_nodes: Retrieved node IDs in order
            ground_truth_nodes: Ground truth path nodes in order
            
        Returns:
            MetricResult with sequence accuracy (0.0 to 1.0)
        """
        def longest_common_subsequence(seq1: List[str], seq2: List[str]) -> int:
            """Calculate LCS length using dynamic programming."""
            if not seq1 or not seq2:
                return 0
                
            m, n = len(seq1), len(seq2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if seq1[i-1] == seq2[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            
            return dp[m][n]
        
        if not ground_truth_nodes:
            return MetricResult(
                name="path_sequence_accuracy",
                value=0.0,
                description="Path Sequence: no ground truth path to evaluate",
                higher_is_better=True
            )
        
        lcs_length = longest_common_subsequence(retrieved_nodes, ground_truth_nodes)
        sequence_accuracy = lcs_length / len(ground_truth_nodes)
        
        return MetricResult(
            name="path_sequence_accuracy", 
            value=sequence_accuracy,
            description=f"Path Sequence: LCS of {lcs_length}/{len(ground_truth_nodes)} maintains order",
            higher_is_better=True
        )
    
    @staticmethod
    def hop_efficiency(actual_hops: int, ground_truth_hops: int) -> MetricResult:
        """
        Calculate hop efficiency: how efficiently the algorithm reached the answer.
        
        Efficiency = ground_truth_hops / actual_hops
        Values > 1.0 indicate the algorithm was more efficient than expected.
        Values < 1.0 indicate the algorithm used more hops than necessary.
        
        Args:
            actual_hops: Number of hops the algorithm actually took
            ground_truth_hops: Number of hops in the ground truth path
            
        Returns:
            MetricResult with efficiency score (higher is better, ~1.0 is ideal)
        """
        if actual_hops == 0:
            return MetricResult(
                name="hop_efficiency",
                value=0.0,
                description="Hop Efficiency: algorithm took 0 hops",
                higher_is_better=True
            )
        
        efficiency = ground_truth_hops / actual_hops
        
        description = f"Hop Efficiency: {ground_truth_hops}/{actual_hops} hops"
        if efficiency > 1.0:
            description += f" (more efficient than expected)"
        elif efficiency < 1.0:
            description += f" (used more hops than optimal)"
        else:
            description += f" (optimal hop count)"
        
        return MetricResult(
            name="hop_efficiency",
            value=efficiency,
            description=description,
            higher_is_better=True
        )
    
    @staticmethod
    def connection_type_accuracy(retrieval_result: 'RetrievalResult', question: GeneratedQuestion) -> MetricResult:
        """
        Calculate accuracy of connection types used during retrieval.
        
        This requires the retrieval result to contain information about
        the types of connections traversed during retrieval.
        
        Args:
            retrieval_result: Result from retrieval algorithm
            question: Question with ground truth connection types
            
        Returns:
            MetricResult with connection type accuracy (0.0 to 1.0)
        """
        # Get connection types from retrieval result metadata if available
        actual_connections = []
        if hasattr(retrieval_result, 'traversal_path') and retrieval_result.traversal_path:
            actual_connections = retrieval_result.traversal_path.connection_types
        
        expected_connections = question.ground_truth_path.connection_types
        
        if not expected_connections:
            return MetricResult(
                name="connection_type_accuracy",
                value=0.0,
                description="Connection Type: no expected connections to evaluate",
                higher_is_better=True
            )
        
        if not actual_connections:
            return MetricResult(
                name="connection_type_accuracy", 
                value=0.0,
                description="Connection Type: no actual connections recorded",
                higher_is_better=True
            )
        
        # Calculate overlap between expected and actual connection types
        expected_set = set(conn_type.value if hasattr(conn_type, 'value') else str(conn_type) 
                          for conn_type in expected_connections)
        actual_set = set(conn_type.value if hasattr(conn_type, 'value') else str(conn_type) 
                        for conn_type in actual_connections)
        
        overlap = len(expected_set & actual_set)
        total_expected = len(expected_set)
        
        accuracy = overlap / total_expected
        
        return MetricResult(
            name="connection_type_accuracy",
            value=accuracy,
            description=f"Connection Type: {overlap}/{total_expected} connection types matched",
            higher_is_better=True
        )
    
    @staticmethod
    def granularity_coverage(retrieval_result: 'RetrievalResult', question: GeneratedQuestion) -> MetricResult:
        """
        Calculate coverage of expected granularity levels.
        
        Measures whether the retrieval covered the expected granularities
        (document, chunk, sentence levels).
        
        Args:
            retrieval_result: Result from retrieval algorithm
            question: Question with ground truth granularity levels
            
        Returns:
            MetricResult with granularity coverage (0.0 to 1.0)
        """
        # Get granularity levels from retrieval result if available
        actual_granularities = []
        if hasattr(retrieval_result, 'traversal_path') and retrieval_result.traversal_path:
            actual_granularities = retrieval_result.traversal_path.granularity_levels
        
        expected_granularities = question.ground_truth_path.granularity_levels
        
        if not expected_granularities:
            return MetricResult(
                name="granularity_coverage",
                value=0.0,
                description="Granularity Coverage: no expected granularities to evaluate",
                higher_is_better=True
            )
        
        if not actual_granularities:
            return MetricResult(
                name="granularity_coverage",
                value=0.0,
                description="Granularity Coverage: no actual granularities recorded", 
                higher_is_better=True
            )
        
        # Calculate coverage of granularity levels
        expected_set = set(gran_level.value if hasattr(gran_level, 'value') else str(gran_level)
                          for gran_level in expected_granularities)
        actual_set = set(gran_level.value if hasattr(gran_level, 'value') else str(gran_level)
                        for gran_level in actual_granularities)
        
        overlap = len(expected_set & actual_set)
        total_expected = len(expected_set)
        
        coverage = overlap / total_expected
        
        return MetricResult(
            name="granularity_coverage",
            value=coverage,
            description=f"Granularity Coverage: {overlap}/{total_expected} granularity levels covered",
            higher_is_better=True
        )


class PerformanceMetrics:
    """Performance and efficiency metrics for retrieval algorithms."""
    
    @staticmethod
    def processing_time_metric(processing_time: float) -> MetricResult:
        """
        Processing time as a metric (lower is better).
        
        Args:
            processing_time: Time taken in seconds
            
        Returns:
            MetricResult with processing time
        """
        return MetricResult(
            name="processing_time",
            value=processing_time,
            description=f"Processing Time: {processing_time:.3f} seconds",
            higher_is_better=False
        )
    
    @staticmethod
    def traversal_hops_metric(total_hops: int) -> MetricResult:
        """
        Number of traversal hops as a metric.
        
        Args:
            total_hops: Total number of hops taken
            
        Returns:
            MetricResult with hop count
        """
        return MetricResult(
            name="traversal_hops",
            value=total_hops,
            description=f"Traversal Hops: {total_hops} hops taken",
            higher_is_better=False  # Generally fewer hops is more efficient
        )
    
    @staticmethod
    def unique_content_ratio(retrieved_content: List[str]) -> MetricResult:
        """
        Ratio of unique content retrieved (diversity measure).
        
        Args:
            retrieved_content: List of retrieved content strings
            
        Returns:
            MetricResult with uniqueness ratio
        """
        if not retrieved_content:
            return MetricResult(
                name="unique_content_ratio",
                value=0.0,
                description="Unique Content: no content retrieved",
                higher_is_better=True
            )
        
        unique_count = len(set(retrieved_content))
        total_count = len(retrieved_content)
        
        ratio = unique_count / total_count
        
        return MetricResult(
            name="unique_content_ratio",
            value=ratio,
            description=f"Unique Content: {unique_count}/{total_count} items were unique",
            higher_is_better=True
        )


class BenchmarkEvaluator:
    """Main class for conducting comprehensive benchmarks of RAG retrieval algorithms."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize benchmark evaluator.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # Define question-type-specific metric weights for composite scoring
        self.type_weights = {
            "single_hop": {
                "precision_at_k": 0.25,
                "recall_at_k": 0.20,
                "mrr": 0.20,
                "node_overlap_ratio": 0.25,
                "processing_time": 0.10
            },
            "raw_similarity": {
                "precision_at_k": 0.25, 
                "recall_at_k": 0.20,
                "ndcg_at_k": 0.20,
                "node_overlap_ratio": 0.25,
                "processing_time": 0.10
            },
            "hierarchical": {
                "recall_at_k": 0.20,
                "path_sequence_accuracy": 0.30,
                "granularity_coverage": 0.25,
                "hop_efficiency": 0.15,
                "processing_time": 0.10
            },
            "theme_bridge": {
                "precision_at_k": 0.20,
                "recall_at_k": 0.25,
                "connection_type_accuracy": 0.25,
                "node_overlap_ratio": 0.20,
                "processing_time": 0.10
            },
            "sequential_flow": {
                "path_sequence_accuracy": 0.35,
                "hop_efficiency": 0.25,
                "recall_at_k": 0.20,
                "processing_time": 0.10,
                "ndcg_at_k": 0.10
            },
            "multi_hop": {
                "recall_at_k": 0.25,
                "path_sequence_accuracy": 0.25,
                "hop_efficiency": 0.20,
                "node_overlap_ratio": 0.20,
                "processing_time": 0.10
            }
        }
    
    def evaluate_single_result(self, 
                              retrieval_result: RetrievalResult,
                              question: GeneratedQuestion,
                              algorithm_name: str,
                              k: int = 10) -> AlgorithmBenchmarkResult:
        """
        Evaluate a single retrieval result against its ground truth question.
        
        Args:
            retrieval_result: Result from retrieval algorithm
            question: Question with ground truth information
            algorithm_name: Name of the algorithm being evaluated
            k: Number of top results to consider for @K metrics
            
        Returns:
            AlgorithmBenchmarkResult containing all calculated metrics
        """
        # Extract node IDs from retrieval result
        retrieved_nodes = self._extract_node_ids(retrieval_result)
        ground_truth_nodes = question.ground_truth_path.nodes
        
        # Calculate traditional IR metrics
        traditional_metrics = {
            "precision_at_k": TraditionalIRMetrics.precision_at_k(retrieved_nodes, ground_truth_nodes, k),
            "recall_at_k": TraditionalIRMetrics.recall_at_k(retrieved_nodes, ground_truth_nodes, k),
            "f1_at_k": TraditionalIRMetrics.f1_at_k(retrieved_nodes, ground_truth_nodes, k),
            "mrr": TraditionalIRMetrics.mean_reciprocal_rank(retrieved_nodes, ground_truth_nodes),
            "ndcg_at_k": TraditionalIRMetrics.ndcg_at_k(retrieved_nodes, ground_truth_nodes, k),
            "map_at_k": TraditionalIRMetrics.map_at_k(retrieved_nodes, ground_truth_nodes, k)
        }
        
        # Calculate path-based metrics  
        path_metrics = {
            "node_overlap_ratio": PathBasedMetrics.node_overlap_ratio(retrieved_nodes, ground_truth_nodes),
            "path_sequence_accuracy": PathBasedMetrics.path_sequence_accuracy(retrieved_nodes, ground_truth_nodes),
            "hop_efficiency": PathBasedMetrics.hop_efficiency(retrieval_result.total_hops, question.ground_truth_path.total_hops),
            "connection_type_accuracy": PathBasedMetrics.connection_type_accuracy(retrieval_result, question),
            "granularity_coverage": PathBasedMetrics.granularity_coverage(retrieval_result, question)
        }
        
        # Calculate performance metrics
        performance_metrics = {
            "processing_time": PerformanceMetrics.processing_time_metric(retrieval_result.processing_time),
            "traversal_hops": PerformanceMetrics.traversal_hops_metric(retrieval_result.total_hops),
            "unique_content_ratio": PerformanceMetrics.unique_content_ratio(retrieval_result.retrieved_content)
        }
        
        # Create comprehensive result
        result = AlgorithmBenchmarkResult(
            algorithm_name=algorithm_name,
            question_id=question.question_id,
            question_type=question.question_type,
            difficulty_level=question.difficulty_level,
            question_text=question.question_text,  # Add missing question text
            expected_answer=question.expected_answer,  # Add missing expected answer
            traditional_metrics=traditional_metrics,
            path_metrics=path_metrics, 
            performance_metrics=performance_metrics,
            retrieved_nodes=retrieved_nodes,
            ground_truth_nodes=ground_truth_nodes,
            processing_time=retrieval_result.processing_time,
            total_hops=retrieval_result.total_hops
        )
        
        self.logger.debug(f"Evaluated {algorithm_name} on {question.question_id}: "
                         f"P@{k}={traditional_metrics['precision_at_k'].value:.3f}, "
                         f"R@{k}={traditional_metrics['recall_at_k'].value:.3f}, "
                         f"Node_overlap={path_metrics['node_overlap_ratio'].value:.3f}")
        
        return result
    
    def calculate_aggregate_scores(self, 
                                 question_results: Dict[str, Dict[str, AlgorithmBenchmarkResult]]) -> Dict[str, Dict[str, float]]:
        """
        Calculate aggregate scores across all questions for each algorithm.
        
        Args:
            question_results: question_id -> algorithm -> AlgorithmBenchmarkResult
            
        Returns:
            Dictionary mapping algorithm -> metric -> average_score
        """
        aggregate_scores = {}
        
        # Get all algorithms
        algorithms = set()
        for q_results in question_results.values():
            algorithms.update(q_results.keys())
        
        for algorithm in algorithms:
            aggregate_scores[algorithm] = {}
            
            # Collect all metric values for this algorithm
            metric_values = {}
            
            for question_id, alg_results in question_results.items():
                if algorithm not in alg_results:
                    continue
                    
                result = alg_results[algorithm]
                
                # Traditional metrics
                for metric_name, metric_result in result.traditional_metrics.items():
                    if metric_name not in metric_values:
                        metric_values[metric_name] = []
                    metric_values[metric_name].append(metric_result.value)
                
                # Path metrics
                for metric_name, metric_result in result.path_metrics.items():
                    if metric_name not in metric_values:
                        metric_values[metric_name] = []
                    metric_values[metric_name].append(metric_result.value)
                
                # Performance metrics  
                for metric_name, metric_result in result.performance_metrics.items():
                    if metric_name not in metric_values:
                        metric_values[metric_name] = []
                    metric_values[metric_name].append(metric_result.value)
            
            # Calculate averages
            for metric_name, values in metric_values.items():
                if values:
                    aggregate_scores[algorithm][metric_name] = np.mean(values)
                    aggregate_scores[algorithm][f"{metric_name}_std"] = np.std(values)
                else:
                    aggregate_scores[algorithm][metric_name] = 0.0
                    aggregate_scores[algorithm][f"{metric_name}_std"] = 0.0
            
            # Calculate composite scores by question type
            self._calculate_composite_scores(aggregate_scores[algorithm], question_results, algorithm)
        
        return aggregate_scores
    
    def _calculate_composite_scores(self, 
                                   algorithm_scores: Dict[str, float],
                                   question_results: Dict[str, Dict[str, AlgorithmBenchmarkResult]],
                                   algorithm: str) -> None:
        """
        Calculate weighted composite scores by question type.
        
        Args:
            algorithm_scores: Dictionary to update with composite scores
            question_results: All question results
            algorithm: Algorithm name being processed
        """
        # Group results by question type
        type_results = {}
        
        for question_id, alg_results in question_results.items():
            if algorithm not in alg_results:
                continue
                
            result = alg_results[algorithm]
            q_type = result.question_type
            
            if q_type not in type_results:
                type_results[q_type] = []
            type_results[q_type].append(result)
        
        # Calculate composite score for each question type
        for q_type, results in type_results.items():
            if q_type not in self.type_weights:
                continue
                
            weights = self.type_weights[q_type]
            weighted_score = 0.0
            total_weight = 0.0
            
            for result in results:
                result_score = 0.0
                result_weight = 0.0
                
                # Weight traditional metrics
                for metric_name, weight in weights.items():
                    if metric_name in result.traditional_metrics:
                        metric_value = result.traditional_metrics[metric_name].value
                        if not result.traditional_metrics[metric_name].higher_is_better:
                            # Invert metrics where lower is better (like processing_time)
                            metric_value = 1.0 - min(metric_value, 1.0)
                        result_score += metric_value * weight
                        result_weight += weight
                    
                    elif metric_name in result.path_metrics:
                        metric_value = result.path_metrics[metric_name].value
                        if not result.path_metrics[metric_name].higher_is_better:
                            metric_value = 1.0 - min(metric_value, 1.0)
                        result_score += metric_value * weight
                        result_weight += weight
                    
                    elif metric_name in result.performance_metrics:
                        metric_value = result.performance_metrics[metric_name].value
                        if not result.performance_metrics[metric_name].higher_is_better:
                            # For processing_time, convert to 0-1 scale (lower is better)
                            if metric_name == "processing_time":
                                metric_value = 1.0 / (1.0 + metric_value)  # Inverse relationship
                            else:
                                metric_value = 1.0 - min(metric_value, 1.0)
                        result_score += metric_value * weight
                        result_weight += weight
                
                if result_weight > 0:
                    weighted_score += (result_score / result_weight)
                    total_weight += 1.0
            
            if total_weight > 0:
                algorithm_scores[f"composite_score_{q_type}"] = weighted_score / total_weight
        
        # Calculate overall composite score
        type_scores = [v for k, v in algorithm_scores.items() if k.startswith("composite_score_")]
        if type_scores:
            algorithm_scores["composite_score_overall"] = np.mean(type_scores)
    
    def _extract_node_ids(self, retrieval_result: RetrievalResult) -> List[str]:
        """
        Extract node IDs from retrieval result for metric calculation.
        
        Args:
            retrieval_result: Result from retrieval algorithm
            
        Returns:
            List of node IDs in retrieval order
        """
        node_ids = []
        
        # Try to get node IDs from traversal path first
        if hasattr(retrieval_result, 'traversal_path') and retrieval_result.traversal_path:
            node_ids = retrieval_result.traversal_path.nodes
        
        # Fallback: extract from retrieved_content if available
        # This is algorithm-specific and may need adjustment
        elif hasattr(retrieval_result, 'retrieved_content') and retrieval_result.retrieved_content:
            # For basic retrieval, we might need to map content back to node IDs
            # This is a placeholder - actual implementation depends on retrieval result structure
            node_ids = [f"content_{i}" for i, _ in enumerate(retrieval_result.retrieved_content)]
        
        return node_ids
    
    def save_benchmark_results(self, 
                             results: CompleteBenchmarkResult,
                             output_path: str) -> None:
        """
        Save benchmark results to YAML file.
        
        Args:
            results: Complete benchmark results
            output_path: Path to save results file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                yaml.dump(results.to_dict(), f, default_flow_style=False, indent=2, sort_keys=False)
            
            self.logger.info(f"💾 Benchmark results saved to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save benchmark results: {e}")
            raise


def create_benchmark_evaluator(logger: Optional[logging.Logger] = None) -> BenchmarkEvaluator:
    """
    Factory function to create a benchmark evaluator.
    
    Args:
        logger: Optional logger instance
        
    Returns:
        BenchmarkEvaluator instance
    """
    return BenchmarkEvaluator(logger)
