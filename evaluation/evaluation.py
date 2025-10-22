#!/usr/bin/env python3
"""
Evaluation Orchestration System
==============================

Comprehensive evaluation orchestrator for semantic traversal algorithms using deepeval.
Executes algorithms on synthetic datasets and measures performance using sophisticated
RAG metrics and custom semantic coherence evaluation.

Key Features:
- Multi-algorithm comparative evaluation
- Hyperparameter tracking for dashboard visibility
- Standard RAG metrics + custom G-Eval for semantic traversal assessment
- Results compatible with existing visualization infrastructure  
- Comprehensive error handling and performance monitoring
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

# DeepEval imports for evaluation
from deepeval import evaluate
from deepeval.dataset import EvaluationDataset
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import (
    AnswerRelevancyMetric, ContextualRecallMetric,
    ContextualPrecisionMetric, ContextualRelevancyMetric,
    FaithfulnessMetric, GEval
)
from deepeval.evaluate import AsyncConfig

# Local imports
from utils.retrieval import RetrievalOrchestrator
from utils.knowledge_graph import KnowledgeGraph
from utils.reranker import RerankerOrchestrator
from .models import ModelManager


@dataclass
class EvaluationResult:
    """
    Comprehensive evaluation results for a single algorithm.
    
    Contains all metrics, hyperparameters, and metadata needed for analysis
    and dashboard visualization in deepeval platform.
    """
    algorithm_name: str
    algorithm_hyperparameters: Dict[str, Any]  # For dashboard visibility
    total_test_cases: int
    successful_test_cases: int
    evaluation_time_seconds: float
    
    # Metric results
    metric_scores: Dict[str, float]            # metric_name -> average_score
    metric_success_rates: Dict[str, float]     # metric_name -> pass_rate
    
    # Individual test case results for detailed analysis
    individual_results: List[Dict[str, Any]]
    
    # Summary statistics
    summary_statistics: Dict[str, Any]
    
    # Generation metadata
    generation_metadata: Dict[str, Any]


class EvaluationOrchestrator:
    """
    Orchestrates comprehensive evaluation of retrieval algorithms using deepeval.
    
    Provides both individual algorithm assessment and comparative benchmarking
    across all semantic traversal methods with detailed hyperparameter tracking.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize EvaluationOrchestrator with configuration and logging.
        
        Args:
            config: Complete system configuration dictionary
            logger: Optional logger instance (creates default if None)
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.deepeval_config = config.get('deepeval', {})
        
        # Initialize model manager for evaluation judges
        self.model_manager = ModelManager(config, logger)

        # Initialize reranker if enabled
        self.reranker_orchestrator: Optional[RerankerOrchestrator] = None
        if self._is_reranking_enabled():
            self.reranker_orchestrator = RerankerOrchestrator(config, logger)
            self.logger.info("Reranking enabled - RerankerOrchestrator initialized")
        else:
            self.logger.info("Reranking disabled")

        # Core components (lazy-loaded)
        self.knowledge_graph: Optional[KnowledgeGraph] = None

        # Storage for visualization data
        self.retrieval_results: Dict[str, Dict[str, Any]] = {}  # query -> {algorithm_name -> result}
        self.retrieval_orchestrator: Optional[RetrievalOrchestrator] = None
        self.evaluation_dataset: Optional[EvaluationDataset] = None
        
        # Evaluation cache for efficiency
        self._metrics_cache: Dict[str, Any] = {}
        
        self.logger.info("EvaluationOrchestrator initialized")

    def _is_reranking_enabled(self) -> bool:
        """Check if reranking is enabled in configuration."""
        # Check in retrieval.semantic_traversal first (primary location)
        retrieval_reranking = self.config.get('retrieval', {}).get('semantic_traversal', {}).get('enable_reranking', None)
        if retrieval_reranking is not None:
            return retrieval_reranking

        # Fallback to old location for backwards compatibility
        return self.config.get('evaluation', {}).get('benchmarking', {}).get('enable_reranking', False)

    def _create_async_config(self) -> AsyncConfig:
        """
        Create AsyncConfig for deepeval evaluation execution.

        Configures sequential vs parallel execution and rate limiting based on
        the configuration file to prevent API rate limit issues.
        """
        # Get async configuration from deepeval config
        async_config = self.deepeval_config.get('evaluation', {}).get('async_config', {})

        # Extract configuration values with defaults
        run_async = async_config.get('run_async', False)  # Default to sequential (False)
        max_concurrent = async_config.get('max_concurrent', 1)  # Default to 1 for sequential
        throttle_value = async_config.get('throttle_value', 3.0)  # Default to 3 seconds delay

        # Log the configuration being used
        execution_mode = "parallel" if run_async else "sequential"
        self.logger.info(f"🔧 Evaluation execution mode: {execution_mode}")
        if run_async:
            self.logger.info(f"   Max concurrent: {max_concurrent}")
        self.logger.info(f"   Throttle delay: {throttle_value}s")

        return AsyncConfig(
            run_async=run_async,
            max_concurrent=max_concurrent,
            throttle_value=throttle_value
        )

    def _generate_answer_from_context(self, question: str, context: str) -> str:
        """
        Generate an answer to the question using the retrieved context.

        Uses the configured answer generation model to generate a proper response
        based on the retrieved context, simulating how a real RAG system works.

        Args:
            question: The input question
            context: The retrieved context text

        Returns:
            Generated answer string
        """
        try:
            # Get the answer generation model for answer generation
            answer_model = self.model_manager.get_answer_generation_model()

            # Create a prompt that asks the LLM to answer the question using the context
            prompt = f"""Based on the provided context, answer the following question. Use only the information from the context and be concise and accurate.

Context:
{context}

Question: {question}

Answer:"""

            # Generate the response using the model
            response = answer_model.generate(prompt)

            # Extract the actual response text
            if hasattr(response, 'response'):
                return response.response
            elif isinstance(response, str):
                return response
            else:
                return str(response)

        except Exception as e:
            self.logger.error(f"Failed to generate answer from context: {e}")
            # Fallback to truncated context if generation fails
            fallback_answer = context[:500] + "..." if len(context) > 500 else context
            self.logger.warning(f"Using fallback answer (truncated context)")
            return fallback_answer
    
    def run(self, algorithm_name: str, 
            algorithm_params: Optional[Dict[str, Any]] = None,
            dataset_path: Optional[str] = None,
            output_prefix: Optional[str] = None) -> EvaluationResult:
        """
        Execute comprehensive evaluation for a specific algorithm.
        
        Runs the algorithm on synthetic dataset questions, evaluates outputs using
        deepeval metrics, and returns comprehensive results with hyperparameter tracking.
        
        Args:
            algorithm_name: Name of retrieval algorithm to evaluate
            algorithm_params: Optional parameters to override defaults
            dataset_path: Optional path to specific dataset file
            output_prefix: Optional prefix for output filenames
            
        Returns:
            EvaluationResult: Comprehensive evaluation metrics and analysis
            
        Raises:
            FileNotFoundError: If required files (KG, dataset) not found
            ValueError: If algorithm_name not supported
            RuntimeError: If evaluation execution fails
        """
        self.logger.info(f"🚀 Starting evaluation for algorithm: {algorithm_name}")
        start_time = time.time()
        
        # Validate algorithm name
        supported_algorithms = self.deepeval_config.get('evaluation', {}).get('algorithms', {}).get('test_algorithms', [])
        if algorithm_name not in supported_algorithms:
            raise ValueError(f"Unsupported algorithm: {algorithm_name}. Supported: {supported_algorithms}")
        
        # Initialize evaluation components
        self._initialize_components(dataset_path)
        
        # Prepare algorithm parameters with hyperparameter tracking
        final_params = self._prepare_algorithm_hyperparameters(algorithm_name, algorithm_params)
        
        # Generate test cases by running algorithm on dataset questions
        test_cases = self._generate_test_cases(algorithm_name, final_params)
        
        # Create evaluation metrics (RAG + custom semantic traversal metrics)
        metrics = self._create_evaluation_metrics()
        
        # Execute deepeval evaluation
        self.logger.info(f"🔍 Evaluating {len(test_cases)} test cases with {len(metrics)} metrics")

        try:
            # Run single batch evaluation (no double evaluation!)
            self.logger.info(f"🔍 Running batch evaluation with deepeval.evaluate()...")
            project_config = self.deepeval_config.get('project', {})
            project_name = project_config.get('name', 'semantic-rag-research')
            project_id = project_config.get('id', 'cmfpz4kpj03i62ad3v3a098kv')
            identifier = f"{project_name}_{algorithm_name}"

            # Get experiment tracking configuration
            experiment_config = self.config.get('experiment', {})
            tracking_config = experiment_config.get('tracking', {})

            hyperparameters = {
                'algorithm': algorithm_name,
                'project_version': project_config.get('version', '1.0.0'),
                'project_description': project_config.get('description', ''),
                'project_tags': ', '.join(project_config.get('tags', [])),
                'project_id': project_id,

                # Experiment tracking (configurable)
                'reranking_enabled': self._is_reranking_enabled(),
                'reranking_strategy': self.config.get('reranking', {}).get('strategy', 'none') if self._is_reranking_enabled() else 'disabled',
                'experiment_notes': tracking_config.get('notes', 'No notes provided'),
                'run_type': tracking_config.get('run_type', 'standard'),
                'dataset_type': tracking_config.get('dataset_type', 'unknown'),
                'baseline_comparison': tracking_config.get('baseline_comparison', False),
                'expected_improvements': ', '.join(tracking_config.get('expected_improvements', [])),

                # Experiment metadata
                'experiment_name': experiment_config.get('name', 'semantic_rag_pipeline'),
                'experiment_version': experiment_config.get('version', '1.0.0'),

                **final_params
            }

            self.logger.info(f"📊 Uploading to DeepEval dashboard - Project: {project_name} (ID: {project_id}), Run: {identifier}")

            # Create async configuration for evaluation execution
            async_config = self._create_async_config()

            # Add generated test cases to the dataset for proper linkage
            if self.evaluation_dataset:
                for test_case in test_cases:
                    self.evaluation_dataset.add_test_case(test_case)

                self.logger.info(f"📊 Added {len(test_cases)} test cases to dataset for evaluation")

                # Use dataset.test_cases in evaluate() call to maintain dataset linkage
                evaluate(
                    test_cases=self.evaluation_dataset.test_cases,  # Use dataset's test_cases
                    metrics=metrics,
                    identifier=identifier,
                    hyperparameters=hyperparameters,
                    async_config=async_config
                )
                self.logger.info("✅ Evaluation completed using dataset.test_cases (maintains dataset linkage)")
            else:
                # Fallback to manually created test cases
                evaluate(
                    test_cases=test_cases,
                    metrics=metrics,
                    identifier=identifier,
                    hyperparameters=hyperparameters,
                    async_config=async_config
                )
                self.logger.info("✅ Evaluation completed via manually created test cases")

        except Exception as e:
            self.logger.error(f"❌ DeepEval execution failed: {e}")
            raise RuntimeError(f"Evaluation execution failed: {e}")

        # Extract metric results from test cases after evaluation
        self.logger.info("📊 Extracting metric results from evaluated test cases...")
        metric_results = {}

        for metric in metrics:
            metric_name = getattr(metric, 'name', None) or getattr(metric, '__name__', None) or metric.__class__.__name__
            metric_results[metric_name] = []

            for i, test_case in enumerate(test_cases):
                # Extract scores from test case after evaluation
                score = None
                success = False

                # Try to get metric results from test case
                if hasattr(test_case, 'metrics_metadata') and test_case.metrics_metadata:
                    metric_data = test_case.metrics_metadata.get(metric_name, {})
                    score = metric_data.get('score')
                    success = metric_data.get('success', False)

                # Fallback: get from metric object directly if available
                if score is None and hasattr(metric, 'score'):
                    score = getattr(metric, 'score', None)
                    success = getattr(metric, 'success', False)

                metric_results[metric_name].append({
                    'score': score,
                    'success': success,
                    'test_case_index': i
                })

                self.logger.debug(f"Extracted - Metric {metric_name}, test case {i}: score={score}, success={success}")

        # Process and aggregate results
        evaluation_time = time.time() - start_time

        # Debug: Log what's available after evaluation
        self.logger.debug(f"Post-evaluation debug:")
        for i, tc in enumerate(test_cases):
            self.logger.debug(f"  Test case {i}: {dir(tc)}")
            if hasattr(tc, 'metrics_metadata'):
                self.logger.debug(f"    metrics_metadata: {tc.metrics_metadata}")

        for i, metric in enumerate(metrics):
            self.logger.debug(f"  Metric {i} ({metric.__class__.__name__}): score={getattr(metric, 'score', 'None')}")

        result = self._process_evaluation_results(
            algorithm_name, final_params, test_cases, metrics, evaluation_time, metric_results
        )
        
        # Save results to configured output directory
        self._save_evaluation_results(result, output_prefix)
        
        self.logger.info(
            f"✅ Evaluation completed in {evaluation_time:.2f}s - "
            f"Success rate: {result.summary_statistics['overall_success_rate']:.1%}"
        )
        
        return result
    
    def run_comparison(self, algorithm_names: Optional[List[str]] = None,
                      output_prefix: Optional[str] = None) -> Dict[str, EvaluationResult]:
        """
        Run comparative evaluation across multiple algorithms.
        
        Evaluates all specified algorithms on the same dataset and generates
        comprehensive comparison reports highlighting semantic traversal superiority.
        
        Args:
            algorithm_names: List of algorithms to compare (defaults to all configured)
            output_prefix: Optional prefix for output filenames
            
        Returns:
            Dict mapping algorithm names to their evaluation results
        """
        if algorithm_names is None:
            algorithm_names = self.deepeval_config.get('evaluation', {}).get('algorithms', {}).get('test_algorithms', [])
        
        self.logger.info(f"🏁 Starting comparative evaluation for {len(algorithm_names)} algorithms")
        
        results = {}
        
        for algorithm_name in algorithm_names:
            try:
                self.logger.info(f"📊 Evaluating {algorithm_name}...")
                
                result = self.run(
                    algorithm_name=algorithm_name,
                    output_prefix=f"comparison_{output_prefix}" if output_prefix else "comparison"
                )
                
                results[algorithm_name] = result
                
                # Log intermediate results
                avg_score = sum(result.metric_scores.values()) / len(result.metric_scores) if result.metric_scores else 0
                self.logger.info(f"   {algorithm_name}: Average score {avg_score:.3f}")
                
            except Exception as e:
                self.logger.error(f"❌ Evaluation failed for {algorithm_name}: {e}")
                continue
        
        if not results:
            raise RuntimeError("All algorithm evaluations failed")
        
        # Generate comprehensive comparison report
        self._generate_comparison_report(results, output_prefix)
        
        self.logger.info(f"🎉 Comparative evaluation completed for {len(results)} algorithms")
        return results
    
    def _initialize_components(self, dataset_path: Optional[str] = None) -> None:
        """Initialize knowledge graph, retrieval orchestrator, and evaluation dataset."""
        self.logger.info("🔧 Initializing evaluation components...")
        
        # Load knowledge graph
        kg_path = Path(self.config['directories']['data']) / "knowledge_graph.json"
        if not kg_path.exists():
            raise FileNotFoundError(f"Knowledge graph not found at {kg_path}. Run kg_pipeline.build() first.")
        
        # Load embeddings for retrieval algorithms - dynamically determine path from config
        configured_model = self.config['models']['embedding_models'][0]  # Get first configured model
        model_path_safe = configured_model.replace('/', '_').replace('-', '_')
        embeddings_path = Path(f"embeddings/raw/{model_path_safe}_multi_granularity.json")
        embeddings_data = None
        self.logger.info(f"🔍 Looking for embeddings at: {embeddings_path}")
        if embeddings_path.exists():
            self.logger.info(f"📥 Loading embeddings from {embeddings_path}")
            with open(embeddings_path, 'r') as f:
                raw_data = json.load(f)

            # Transform the file structure to what knowledge graph expects
            # From: {"metadata": {...}, "embeddings": {"chunks": [...], "sentences": [...]}}
            # To: {"configured_model": {"chunks": [...], "sentences": [...]}}
            model_name = raw_data.get('metadata', {}).get('model_name', configured_model)
            embeddings_data = {
                model_name: raw_data['embeddings']
            }
            self.logger.info(f"📊 Embeddings transformed for model: {model_name}")
        else:
            self.logger.warning(f"⚠️ Embeddings not found at {embeddings_path} - retrieval may not work")

        self.knowledge_graph = KnowledgeGraph.load(str(kg_path), embeddings_data)

        # Debug: Check if embeddings were actually loaded
        if hasattr(self.knowledge_graph, '_embedding_cache'):
            cache_size = len(self.knowledge_graph._embedding_cache)
            cache_keys = list(self.knowledge_graph._embedding_cache.keys())
            self.logger.info(f"🧠 Knowledge graph embedding cache: {cache_size} models loaded")
            self.logger.info(f"🔑 Cache model keys: {cache_keys}")

            # Check specific cache contents for debugging
            for model_key in cache_keys:
                model_cache = self.knowledge_graph._embedding_cache[model_key]
                chunk_count = len(model_cache.get('chunks', {}))
                sentence_count = len(model_cache.get('sentences', {}))
                self.logger.info(f"   Model '{model_key}': {chunk_count} chunks, {sentence_count} sentences")
        else:
            self.logger.warning("⚠️ Knowledge graph has no embedding cache")
        self.logger.info(f"✅ Knowledge graph loaded: {len(self.knowledge_graph.chunks)} chunks")
        
        # Initialize retrieval orchestrator
        self.retrieval_orchestrator = RetrievalOrchestrator(
            self.knowledge_graph, self.config, self.logger
        )
        
        # Load evaluation dataset
        dataset_config = self.deepeval_config.get('dataset', {}).get('output', {})

        # Check if we should pull from dashboard
        if dataset_config.get('pull_from_dashboard', False) and not dataset_path:
            dataset_alias = dataset_config.get('dataset_alias', 'semantic-rag-benchmark')
            self._load_dataset_from_dashboard(dataset_alias)
        else:
            # Load from local file
            if dataset_path:
                dataset_file = Path(dataset_path)
            else:
                dataset_file = Path(dataset_config.get('save_path', 'data/synthetic_dataset.json'))

            if not dataset_file.exists():
                raise FileNotFoundError(f"Dataset not found at {dataset_file}. Run dataset.build() first.")

            # Load dataset using add_goldens_from_json_file method
            self.evaluation_dataset = EvaluationDataset()
            self.evaluation_dataset.add_goldens_from_json_file(str(dataset_file))
            self.logger.info(f"✅ Dataset loaded: {len(self.evaluation_dataset.goldens)} test questions")
    
    def _prepare_algorithm_hyperparameters(self, algorithm_name: str, 
                                         custom_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Prepare algorithm hyperparameters with proper defaults and overrides.
        
        Ensures hyperparameters are properly tracked for dashboard visibility
        in deepeval platform while maintaining algorithm performance.
        """
        # Get configured hyperparameters for the algorithm
        algorithm_config = self.deepeval_config.get('evaluation', {}).get('algorithms', {})
        default_hyperparams = algorithm_config.get('algorithm_hyperparameters', {}).get(algorithm_name, {})
        
        # Start with defaults and apply custom overrides
        if custom_params:
            final_params = {**default_hyperparams, **custom_params}
            self.logger.info(f"🔧 Using custom hyperparameters for {algorithm_name}")
        else:
            final_params = default_hyperparams.copy()
            self.logger.info(f"🔧 Using default hyperparameters for {algorithm_name}")
        
        # Log hyperparameters for transparency
        self.logger.info(f"   Hyperparameters: {final_params}")
        
        return final_params
    
    def _generate_test_cases(self, algorithm_name: str, 
                           algorithm_params: Dict[str, Any]) -> List[LLMTestCase]:
        """
        Generate LLMTestCases by executing algorithm on dataset questions.
        
        Runs the retrieval algorithm on each synthetic question to get actual outputs,
        then creates test cases comparing actual vs expected outputs.
        """
        test_cases = []
        failed_cases = 0
        
        self.logger.info(f"🎯 Generating test cases using {algorithm_name}...")
        
        for i, golden in enumerate(self.evaluation_dataset.goldens):
            try:
                # Execute retrieval algorithm on the synthetic question
                retrieval_result = self.retrieval_orchestrator.retrieve(
                    query=golden.input,
                    algorithm_name=algorithm_name,
                    algorithm_params=algorithm_params
                )

                # Store retrieval result for visualization
                if golden.input not in self.retrieval_results:
                    self.retrieval_results[golden.input] = {}
                self.retrieval_results[golden.input][algorithm_name] = retrieval_result

                # Apply reranking if enabled (standardizes output across all algorithms)
                if self.reranker_orchestrator:
                    reranked_sentences, rerank_metadata = self.reranker_orchestrator.rerank_retrieval_result(
                        retrieval_result, golden.input
                    )
                    context_text = "\n".join(reranked_sentences)
                    self.logger.debug(f"Reranking applied: {len(retrieval_result.retrieved_content)} -> {len(reranked_sentences)} sentences")
                else:
                    context_text = "\n".join(retrieval_result.retrieved_content)

                # Generate actual output using LLM with (potentially reranked) context
                actual_output = self._generate_answer_from_context(golden.input, context_text)
                
                # Create test case for deepeval evaluation
                test_case = LLMTestCase(
                    input=golden.input,
                    actual_output=actual_output,
                    expected_output=golden.expected_output,
                    retrieval_context=retrieval_result.retrieved_content,
                    context=golden.context if hasattr(golden, 'context') else None
                )
                
                test_cases.append(test_case)
                
                # Progress logging
                if (i + 1) % 20 == 0:
                    self.logger.info(f"   Generated {i + 1}/{len(self.evaluation_dataset.goldens)} test cases")
                    
            except Exception as e:
                failed_cases += 1
                self.logger.warning(f"⚠️ Failed to process question {i}: {e}")
                continue
        
        if failed_cases > 0:
            self.logger.warning(f"⚠️ {failed_cases} test cases failed during generation")
        
        self.logger.info(f"✅ Generated {len(test_cases)} valid test cases")
        return test_cases
    
    def _create_evaluation_metrics(self) -> List[Any]:
        """
        Create configured evaluation metrics for comprehensive assessment.
        
        Combines standard RAG metrics with custom G-Eval metrics designed
        specifically for semantic traversal algorithm evaluation.
        """
        metrics = []
        evaluation_judge = self.model_manager.get_evaluation_judge_model()
        
        # Create standard RAG metrics
        rag_metrics = self.deepeval_config.get('evaluation', {}).get('metrics', {}).get('rag_metrics', [])
        
        metric_mapping = {
            'answer_relevancy': AnswerRelevancyMetric,
            'contextual_recall': ContextualRecallMetric,
            'contextual_precision': ContextualPrecisionMetric, 
            'contextual_relevancy': ContextualRelevancyMetric,
            'faithfulness': FaithfulnessMetric
        }
        
        for metric_name in rag_metrics:
            if metric_name in metric_mapping:
                metric_class = metric_mapping[metric_name]
                
                # Create metric with evaluation judge (remove include_reason to avoid compatibility issues)
                metric = metric_class(
                    model=evaluation_judge,
                    threshold=0.5  # Standard threshold
                )
                
                metrics.append(metric)
                self.logger.debug(f"   Added RAG metric: {metric_name}")
        
        # Create custom G-Eval metrics for semantic traversal assessment
        custom_metrics = self.deepeval_config.get('evaluation', {}).get('metrics', {}).get('custom_metrics', [])
        
        for custom_metric_config in custom_metrics:
            # Convert evaluation_params strings to LLMTestCaseParams enum values
            evaluation_params = []
            for param_name in custom_metric_config.get('evaluation_params', []):
                if hasattr(LLMTestCaseParams, param_name):
                    evaluation_params.append(getattr(LLMTestCaseParams, param_name))
                else:
                    self.logger.warning(f"Unknown evaluation parameter: {param_name}")
            
            # Create G-Eval metric for semantic traversal assessment
            g_eval_metric = GEval(
                name=custom_metric_config['name'],
                criteria=custom_metric_config['criteria'],
                evaluation_params=evaluation_params,
                threshold=custom_metric_config.get('threshold', 0.5),
                model=evaluation_judge
                # Note: GEval doesn't support include_reason parameter
            )
            
            metrics.append(g_eval_metric)
            self.logger.debug(f"   Added custom G-Eval metric: {custom_metric_config['name']}")
        
        self.logger.info(f"📏 Created {len(metrics)} evaluation metrics ({len(rag_metrics)} RAG + {len(custom_metrics)} custom)")
        return metrics
    
    def _process_evaluation_results(self, algorithm_name: str, algorithm_params: Dict[str, Any],
                                  test_cases: List[LLMTestCase], metrics: List[Any],
                                  evaluation_time: float, metric_results: Dict = None) -> EvaluationResult:
        """
        Process deepeval results into structured format with comprehensive analysis.
        
        Aggregates metric scores, calculates success rates, and prepares data
        for dashboard visualization and comparison reporting.
        """
        # Aggregate metric scores and success rates
        metric_scores = {}
        metric_success_rates = {}

        # Use captured metric results if available
        if metric_results:
            for metric_name, results in metric_results.items():
                # Calculate average score across all test cases for this metric
                scores = [r['score'] for r in results if r['score'] is not None]
                if scores:
                    metric_scores[metric_name] = sum(scores) / len(scores)
                    self.logger.debug(f"Calculated average score for {metric_name}: {metric_scores[metric_name]}")

                # Calculate success rate
                successes = [r['success'] for r in results if r['success'] is not None]
                if successes:
                    metric_success_rates[metric_name] = sum(successes) / len(successes)
                    self.logger.debug(f"Calculated success rate for {metric_name}: {metric_success_rates[metric_name]}")

        # Fallback: Try to extract from metric objects if no captured results
        if not metric_results:
            for metric in metrics:
                try:
                    # Get metric name using proper attribute
                    metric_name = getattr(metric, 'name', None) or getattr(metric, '__name__', None) or metric.__class__.__name__

                    # Debug: Log metric attributes
                    self.logger.debug(f"Processing metric {metric_name}: score={getattr(metric, 'score', 'None')}, has_scores={hasattr(metric, 'scores')}")

                    # Handle metric score
                    if hasattr(metric, 'score') and metric.score is not None:
                        metric_scores[metric_name] = float(metric.score)
                        self.logger.debug(f"Added score for {metric_name}: {metric.score}")
                    elif hasattr(metric, 'scores') and metric.scores:
                        # Handle case where metric has multiple scores
                        avg_score = float(sum(metric.scores) / len(metric.scores))
                        metric_scores[metric_name] = avg_score
                        self.logger.debug(f"Added average score for {metric_name}: {avg_score}")
                    else:
                        self.logger.warning(f"No score found for metric {metric_name}")

                    # Handle metric success rate
                    if hasattr(metric, 'is_successful'):
                        try:
                            success_result = metric.is_successful()
                            if isinstance(success_result, bool):
                                metric_success_rates[metric_name] = float(success_result)
                            elif isinstance(success_result, (int, float)):
                                metric_success_rates[metric_name] = float(success_result)
                        except Exception as e:
                            self.logger.warning(f"Failed to get success rate for {metric_name}: {e}")
                            metric_success_rates[metric_name] = 0.0

                except Exception as e:
                    metric_name = getattr(metric, 'name', None) or getattr(metric, '__name__', None) or metric.__class__.__name__
                    self.logger.error(f"Error processing metric {metric_name}: {e}")
                    # Set default values to avoid breaking the evaluation
                    metric_scores[metric_name] = 0.0
                    metric_success_rates[metric_name] = 0.0
        
        # Count successful test cases (at least one metric passed)
        successful_test_cases = 0
        for tc in test_cases:
            test_case_success = False
            for metric in metrics:
                try:
                    if hasattr(metric, 'is_successful') and callable(metric.is_successful):
                        if metric.is_successful():
                            test_case_success = True
                            break
                except Exception as e:
                    self.logger.debug(f"Error checking success for metric {getattr(metric, 'name', 'unknown')}: {e}")
                    continue
            if test_case_success:
                successful_test_cases += 1
        
        # Create individual test case results for detailed analysis
        individual_results = []
        for i, test_case in enumerate(test_cases):
            case_result = {
                'test_case_index': i,
                'input': test_case.input,
                'actual_output': test_case.actual_output[:500] + "..." if len(test_case.actual_output) > 500 else test_case.actual_output,
                'expected_output': test_case.expected_output[:500] + "..." if len(test_case.expected_output) > 500 else test_case.expected_output,
                'retrieval_context_count': len(test_case.retrieval_context) if test_case.retrieval_context else 0,
                'metric_scores': {}
            }
            
            # Add individual metric scores if available
            if metric_results:
                # Use captured metric results
                for metric_name, results in metric_results.items():
                    if i < len(results):
                        case_result['metric_scores'][metric_name] = results[i]['score']
            else:
                # Fallback to metric objects
                for metric in metrics:
                    metric_name = getattr(metric, 'name', None) or getattr(metric, '__name__', None) or metric.__class__.__name__
                    if hasattr(metric, 'scores') and metric.scores and i < len(metric.scores):
                        case_result['metric_scores'][metric_name] = metric.scores[i]
                    elif hasattr(metric, 'score'):
                        case_result['metric_scores'][metric_name] = metric.score
            
            individual_results.append(case_result)
        
        # Calculate summary statistics
        overall_success_rate = successful_test_cases / len(test_cases) if test_cases else 0
        average_metric_score = sum(metric_scores.values()) / len(metric_scores) if metric_scores else 0
        
        summary_statistics = {
            'overall_success_rate': overall_success_rate,
            'average_metric_score': average_metric_score,
            'total_metrics_evaluated': len(metrics),
            'successful_metrics': sum(metric_success_rates.values()),
            'evaluation_efficiency': successful_test_cases / evaluation_time if evaluation_time > 0 else 0
        }
        
        # Prepare generation metadata for reproducibility
        generation_metadata = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'algorithm_name': algorithm_name,
            'dataset_size': len(self.evaluation_dataset.goldens),
            'model_info': {
                'evaluation_judge': self.model_manager.get_model_info('evaluation_judge')
            }
        }
        
        return EvaluationResult(
            algorithm_name=algorithm_name,
            algorithm_hyperparameters=algorithm_params,
            total_test_cases=len(test_cases),
            successful_test_cases=successful_test_cases,
            evaluation_time_seconds=evaluation_time,
            metric_scores=metric_scores,
            metric_success_rates=metric_success_rates,
            individual_results=individual_results,
            summary_statistics=summary_statistics,
            generation_metadata=generation_metadata
        )
    
    def _save_evaluation_results(self, result: EvaluationResult, 
                               output_prefix: Optional[str] = None) -> None:
        """
        Save evaluation results to configured output directory.
        
        Creates detailed JSON files compatible with existing visualization
        infrastructure and deepeval dashboard integration.
        """
        output_config = self.deepeval_config.get('evaluation', {}).get('output', {})
        output_dir = Path(output_config.get('results_directory', 'benchmark_results'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{output_prefix}_" if output_prefix else ""
        filename = f"{prefix}{result.algorithm_name}_{timestamp}.json"
        
        output_path = output_dir / filename
        
        # Save comprehensive results
        with open(output_path, 'w') as f:
            json.dump(asdict(result), f, indent=2, default=str)
        
        # Save summary for quick access
        summary_path = output_dir / f"{prefix}{result.algorithm_name}_summary.json"
        summary = {
            'algorithm_name': result.algorithm_name,
            'algorithm_hyperparameters': result.algorithm_hyperparameters,
            'evaluation_timestamp': result.generation_metadata['evaluation_timestamp'],
            'overall_success_rate': result.summary_statistics['overall_success_rate'],
            'average_metric_score': result.summary_statistics['average_metric_score'],
            'metric_scores': result.metric_scores,
            'total_test_cases': result.total_test_cases,
            'evaluation_time_seconds': result.evaluation_time_seconds
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"💾 Results saved: {output_path}")
        self.logger.info(f"📋 Summary saved: {summary_path}")
    
    def _generate_comparison_report(self, results: Dict[str, EvaluationResult], 
                                  output_prefix: Optional[str] = None) -> None:
        """
        Generate comprehensive comparison report across algorithms.
        
        Creates detailed analysis highlighting semantic traversal algorithm
        superiority with statistical significance and visualization data.
        """
        output_config = self.deepeval_config.get('evaluation', {}).get('output', {})
        output_dir = Path(output_config.get('results_directory', 'benchmark_results'))
        
        # Generate comparison report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{output_prefix}_" if output_prefix else ""
        report_path = output_dir / f"{prefix}algorithm_comparison_{timestamp}.json"
        
        # Prepare comparison data
        comparison_data = {
            'comparison_timestamp': datetime.now().isoformat(),
            'algorithms_compared': list(results.keys()),
            'dataset_size': results[list(results.keys())[0]].total_test_cases,
            'algorithm_results': {},
            'comparative_analysis': {}
        }
        
        # Add individual algorithm results
        for algorithm_name, result in results.items():
            comparison_data['algorithm_results'][algorithm_name] = {
                'hyperparameters': result.algorithm_hyperparameters,
                'overall_success_rate': result.summary_statistics['overall_success_rate'],
                'average_metric_score': result.summary_statistics['average_metric_score'],
                'metric_scores': result.metric_scores,
                'evaluation_time_seconds': result.evaluation_time_seconds
            }
        
        # Calculate comparative analysis
        best_algorithm = max(results.keys(), 
                           key=lambda k: results[k].summary_statistics['average_metric_score'])
        
        comparison_data['comparative_analysis'] = {
            'best_performing_algorithm': best_algorithm,
            'performance_rankings': sorted(
                results.keys(),
                key=lambda k: results[k].summary_statistics['average_metric_score'],
                reverse=True
            ),
            'metric_winners': {},
            'statistical_summary': {
                'max_score': max(r.summary_statistics['average_metric_score'] for r in results.values()),
                'min_score': min(r.summary_statistics['average_metric_score'] for r in results.values()),
                'score_std': self._calculate_score_std(results)
            }
        }
        
        # Determine metric winners
        all_metrics = set()
        for result in results.values():
            all_metrics.update(result.metric_scores.keys())
        
        for metric_name in all_metrics:
            metric_scores = {alg: res.metric_scores.get(metric_name, 0) for alg, res in results.items()}
            winner = max(metric_scores.keys(), key=lambda k: metric_scores[k])
            comparison_data['comparative_analysis']['metric_winners'][metric_name] = winner
        
        # Save comparison report
        with open(report_path, 'w') as f:
            json.dump(comparison_data, f, indent=2, default=str)
        
        self.logger.info(f"📊 Comparison report saved: {report_path}")
        self.logger.info(f"🏆 Best performing algorithm: {best_algorithm}")
    
    def _calculate_score_std(self, results: Dict[str, EvaluationResult]) -> float:
        """Calculate standard deviation of average scores across algorithms."""
        scores = [result.summary_statistics['average_metric_score'] for result in results.values()]
        if len(scores) < 2:
            return 0.0
        
        mean = sum(scores) / len(scores)
        variance = sum((score - mean) ** 2 for score in scores) / len(scores)
        return variance ** 0.5

    def _load_dataset_from_dashboard(self, alias: str) -> None:
        """
        Load evaluation dataset from DeepEval dashboard.

        Pulls the synthetic dataset from DeepEval's cloud platform for
        distributed benchmarking and reproducible evaluation workflows.

        Args:
            alias: The unique name of the dataset on the dashboard
        """
        try:
            self.logger.info(f"📥 Loading dataset from DeepEval dashboard: '{alias}'...")

            # Pull dataset from DeepEval cloud
            self.evaluation_dataset = EvaluationDataset()
            self.evaluation_dataset.pull(alias=alias)

            self.logger.info(f"✅ Dataset successfully loaded from dashboard")
            self.logger.info(f"   Questions loaded: {len(self.evaluation_dataset.goldens)}")

        except Exception as e:
            self.logger.error(f"❌ Failed to load dataset from dashboard: {e}")
            self.logger.warning(f"💾 Falling back to local dataset file...")
            # Fallback to local file loading
            dataset_config = self.deepeval_config.get('dataset', {}).get('output', {})
            dataset_file = Path(dataset_config.get('save_path', 'data/synthetic_dataset.json'))

            if dataset_file.exists():
                self.evaluation_dataset = EvaluationDataset()
                self.evaluation_dataset.add_goldens_from_json_file(str(dataset_file))
                self.logger.info(f"✅ Fallback successful: {len(self.evaluation_dataset.goldens)} questions")
            else:
                raise FileNotFoundError(f"No dataset available locally at {dataset_file} and dashboard pull failed")
