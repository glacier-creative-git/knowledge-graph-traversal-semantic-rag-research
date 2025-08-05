"""
RAGAS Evaluation Pipeline
========================

RAGAS evaluation framework for assessing RAG system performance.
Extracted and modularized from research_4.py.
"""

import os
import time
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass

# RAGAS imports
try:
    from ragas import evaluate, EvaluationDataset
    from ragas.metrics import (
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy
    )
    from datasets import Dataset
    from langchain_openai import ChatOpenAI

    RAGAS_AVAILABLE = True
except ImportError:
    print("⚠️ RAGAS not installed. Please install with: pip install ragas")
    RAGAS_AVAILABLE = False

from .rag_system import SemanticGraphRAG
from .config import ModelConfig


@dataclass
class EvaluationResults:
    """Results from RAGAS evaluation"""
    context_precision: float
    context_recall: float
    faithfulness: float
    answer_relevancy: float
    ragas_score: float
    ingest_time: float
    eval_time: float
    num_samples: int
    error: Optional[str] = None


class RAGASEvaluator:
    """
    RAGAS evaluation framework for RAG systems
    """

    def __init__(self, config: ModelConfig, max_samples: int = 20):
        if not RAGAS_AVAILABLE:
            raise ImportError("RAGAS is required for evaluation. Please install with: pip install ragas")

        self.config = config
        self.max_samples = max_samples

        # Set up OpenAI API key
        if config.openai_api_key:
            os.environ["OPENAI_API_KEY"] = config.openai_api_key

        # Initialize LLM for evaluation
        self.llm = ChatOpenAI(
            model=config.chat_model,
            api_key=config.openai_api_key,
            request_timeout=60,
            max_retries=3,
        )

        # RAGAS metrics
        self.metrics = [
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy
        ]

    def evaluate_rag_system(self, rag_system: SemanticGraphRAG,
                            documents: List[Dict],
                            queries: List[str],
                            ground_truths: Optional[List[str]] = None,
                            system_name: str = "RAG System") -> EvaluationResults:
        """
        Evaluate a RAG system using RAGAS metrics

        Args:
            rag_system: The RAG system to evaluate
            documents: List of document dictionaries
            queries: List of query strings
            ground_truths: Optional list of ground truth answers
            system_name: Name of the system for logging

        Returns:
            EvaluationResults object
        """
        print(f"🔍 Evaluating {system_name}...")

        try:
            # Ingest documents
            print("📚 Ingesting documents...")
            ingest_time = rag_system.ingest_contexts(documents)
            print(f"✅ Documents ingested in {ingest_time:.2f}s")

            # Create evaluation dataset
            print("🎯 Creating evaluation dataset...")
            dataset = self._create_rag_dataset(rag_system, documents, queries, ground_truths)

            if not dataset:
                return EvaluationResults(
                    context_precision=0.0,
                    context_recall=0.0,
                    faithfulness=0.0,
                    answer_relevancy=0.0,
                    ragas_score=0.0,
                    ingest_time=ingest_time,
                    eval_time=0.0,
                    num_samples=0,
                    error="No valid samples created"
                )

            print(f"📊 Created dataset with {len(dataset)} samples")

            # Convert to RAGAS format
            evaluation_dataset = EvaluationDataset.from_list(dataset)

            # Run evaluation
            print("⚡ Running RAGAS evaluation...")
            start_time = time.time()

            result = evaluate(
                dataset=evaluation_dataset,
                metrics=self.metrics,
                llm=self.llm,
                raise_exceptions=False
            )

            eval_time = time.time() - start_time
            print(f"✅ Evaluation completed in {eval_time:.2f}s")

            # Extract scores
            scores = self._extract_scores(result)

            return EvaluationResults(
                context_precision=scores.get('context_precision', 0.0),
                context_recall=scores.get('context_recall', 0.0),
                faithfulness=scores.get('faithfulness', 0.0),
                answer_relevancy=scores.get('answer_relevancy', 0.0),
                ragas_score=scores.get('ragas_score', 0.0),
                ingest_time=ingest_time,
                eval_time=eval_time,
                num_samples=len(dataset)
            )

        except Exception as e:
            print(f"❌ Evaluation failed: {str(e)}")
            return EvaluationResults(
                context_precision=0.0,
                context_recall=0.0,
                faithfulness=0.0,
                answer_relevancy=0.0,
                ragas_score=0.0,
                ingest_time=0.0,
                eval_time=0.0,
                num_samples=0,
                error=str(e)
            )

    def _create_rag_dataset(self, rag_system: SemanticGraphRAG,
                            documents: List[Dict],
                            queries: List[str],
                            ground_truths: Optional[List[str]] = None) -> List[Dict]:
        """Create evaluation dataset by running queries through RAG system"""
        max_queries = min(len(queries), self.max_samples)
        limited_queries = queries[:max_queries]
        limited_ground_truths = ground_truths[:max_queries] if ground_truths else None

        dataset = []

        for i, query in enumerate(limited_queries):
            try:
                # Run retrieval
                retrieved_texts, _, _ = rag_system.retrieve(query, top_k=3)

                # Create context from retrieved texts
                context_text = "\n".join(retrieved_texts)

                # Generate a simple answer based on context
                answer = f"Based on the provided context: {context_text[:200]}..."

                sample = {
                    "user_input": query,
                    "retrieved_contexts": retrieved_texts,
                    "response": answer
                }

                # Add ground truth if available
                if limited_ground_truths and i < len(limited_ground_truths):
                    sample["reference"] = limited_ground_truths[i]

                dataset.append(sample)

            except Exception as e:
                print(f"⚠️ Error processing query {i}: {str(e)}")
                continue

        return dataset

    def _extract_scores(self, result) -> Dict[str, float]:
        """Extract scores from RAGAS result object"""
        scores = {}
        metrics_to_extract = ['context_precision', 'context_recall', 'faithfulness', 'answer_relevancy']

        # Handle different RAGAS result formats
        result_scores = None
        if hasattr(result, 'scores') and result.scores is not None:
            result_scores = result.scores
        elif hasattr(result, '_scores_dict') and result._scores_dict is not None:
            result_scores = result._scores_dict

        if isinstance(result_scores, list) and len(result_scores) > 0:
            # Process list of sample scores
            metric_totals = {metric: [] for metric in metrics_to_extract}

            for sample_scores in result_scores:
                for metric_name in metrics_to_extract:
                    if metric_name in sample_scores:
                        score_val = sample_scores[metric_name]
                        if hasattr(score_val, 'item'):
                            score_val = float(score_val.item())
                        else:
                            score_val = float(score_val)

                        if not np.isnan(score_val):
                            metric_totals[metric_name].append(score_val)

            # Calculate averages
            for metric_name in metrics_to_extract:
                if metric_totals[metric_name]:
                    scores[metric_name] = np.mean(metric_totals[metric_name])
                else:
                    scores[metric_name] = 0.0
        else:
            # Process direct scores
            for metric_name in metrics_to_extract:
                try:
                    score_value = 0.0
                    if hasattr(result, metric_name):
                        score_value = float(getattr(result, metric_name))
                    elif isinstance(result, dict) and metric_name in result:
                        score_value = float(result[metric_name])
                    scores[metric_name] = score_value
                except (TypeError, ValueError, AttributeError):
                    scores[metric_name] = 0.0

        # Calculate overall RAGAS score
        valid_scores = [score for score in scores.values() if score > 0]
        if valid_scores:
            scores['ragas_score'] = np.mean(list(scores.values()))
        else:
            scores['ragas_score'] = 0.0

        return scores

    def benchmark_rag_system(self, rag_system: SemanticGraphRAG,
                             evaluation_dataset: Dict,
                             system_name: str = "RAG System") -> EvaluationResults:
        """
        Benchmark a RAG system using a prepared evaluation dataset

        Args:
            rag_system: The RAG system to benchmark
            evaluation_dataset: Dictionary with 'documents', 'queries', 'ground_truths'
            system_name: Name of the system for logging

        Returns:
            EvaluationResults object
        """
        return self.evaluate_rag_system(
            rag_system=rag_system,
            documents=evaluation_dataset['documents'],
            queries=evaluation_dataset['queries'],
            ground_truths=evaluation_dataset.get('ground_truths'),
            system_name=system_name
        )


def print_evaluation_results(results: EvaluationResults, system_name: str = "RAG System"):
    """
    Print evaluation results in a clean format

    Args:
        results: EvaluationResults object
        system_name: Name of the system
    """
    print(f"\n{'=' * 60}")
    print(f"RAGAS EVALUATION RESULTS: {system_name}")
    print(f"{'=' * 60}")

    if results.error:
        print(f"❌ Error: {results.error}")
        return

    print(f"📊 Evaluation Metrics:")
    print(f"   Context Precision: {results.context_precision:.3f}")
    print(f"   Context Recall: {results.context_recall:.3f}")
    print(f"   Faithfulness: {results.faithfulness:.3f}")
    print(f"   Answer Relevancy: {results.answer_relevancy:.3f}")
    print(f"   Overall RAGAS Score: {results.ragas_score:.3f}")

    print(f"\n⚡ Performance:")
    print(f"   Ingestion Time: {results.ingest_time:.2f}s")
    print(f"   Evaluation Time: {results.eval_time:.2f}s")
    print(f"   Samples Evaluated: {results.num_samples}")

    # Performance assessment
    if results.ragas_score >= 0.7:
        print(f"   ✅ Excellent performance!")
    elif results.ragas_score >= 0.5:
        print(f"   ✅ Good performance")
    elif results.ragas_score >= 0.3:
        print(f"   ⚠️ Fair performance")
    else:
        print(f"   ⚠️ Needs improvement")


# Convenience functions for easy notebook usage
def evaluate_rag_system(rag_system: SemanticGraphRAG,
                        evaluation_dataset: Dict,
                        openai_api_key: str,
                        max_samples: int = 20) -> EvaluationResults:
    """
    Quick function to evaluate a RAG system

    Args:
        rag_system: The RAG system to evaluate
        evaluation_dataset: Dataset dictionary
        openai_api_key: OpenAI API key for evaluation
        max_samples: Maximum samples to evaluate

    Returns:
        EvaluationResults object
    """
    if not RAGAS_AVAILABLE:
        print("⚠️ RAGAS not available. Returning dummy results.")
        return EvaluationResults(
            context_precision=0.0,
            context_recall=0.0,
            faithfulness=0.0,
            answer_relevancy=0.0,
            ragas_score=0.0,
            ingest_time=0.0,
            eval_time=0.0,
            num_samples=0,
            error="RAGAS not installed"
        )

    from .config import ModelConfig
    config = ModelConfig(openai_api_key=openai_api_key)
    evaluator = RAGASEvaluator(config, max_samples)

    results = evaluator.benchmark_rag_system(rag_system, evaluation_dataset)
    print_evaluation_results(results)

    return results