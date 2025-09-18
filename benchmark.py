#!/usr/bin/env python3
"""
Comprehensive RAG Algorithm Benchmarking System
==============================================

Advanced benchmarking orchestrator that evaluates all four retrieval algorithms
using sophisticated metrics and comprehensive visualization generation.

Features:
- Automatic question loading/generation with simplified caching
- Multi-algorithm execution with consistent evaluation
- Rich metrics calculation using traditional IR and path-based approaches
- Comprehensive visualization generation (2 plots per algorithm per question)
- Structured results output in human-readable YAML format

Usage:
    python benchmark.py                    # Run with default settings
    python benchmark.py --questions 20    # Specify number of questions
    python benchmark.py --force-regenerate # Force question regeneration
    python benchmark.py --algorithms basic_retrieval kg_traversal  # Test specific algorithms
"""

import os
import sys
import argparse
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import our components
from utils.pipeline import SemanticRAGPipeline
from utils.questions import EvaluationDataset, DatasetGenerator
from utils.benchmarking import BenchmarkEvaluator, CompleteBenchmarkResult, AlgorithmBenchmarkResult
from utils.retrieval import RetrievalOrchestrator
from utils.knowledge_graph import KnowledgeGraph
from utils.matplotlib_visualizer import create_global_visualization, create_sequential_visualization
from utils.reranking import create_reranker_orchestrator


def setup_logging() -> logging.Logger:
    """Setup comprehensive logging for the benchmarking process."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('benchmark.log', mode='w', encoding='utf-8')
        ]
    )
    
    return logging.getLogger("RAGBenchmark")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for benchmark configuration."""
    parser = argparse.ArgumentParser(
        description="Comprehensive RAG Algorithm Benchmarking System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark.py                                    # Default benchmark
  python benchmark.py --questions 25                    # 25 questions
  python benchmark.py --force-regenerate                # Force new questions  
  python benchmark.py --algorithms kg_traversal         # Test specific algorithm
  python benchmark.py --skip-visualizations             # Skip plot generation
  python benchmark.py --output-dir custom_results       # Custom output directory
        """
    )
    
    parser.add_argument(
        '--questions', 
        type=int, 
        default=25,
        help='Number of questions to generate/use (default: 15)'
    )
    
    parser.add_argument(
        '--algorithms',
        nargs='*',
        default=['basic_retrieval', 'query_traversal', 'kg_traversal', 'triangulation_centroid'],
        choices=['basic_retrieval', 'query_traversal', 'kg_traversal', 'triangulation_centroid'],
        help='Algorithms to benchmark (default: all algorithms)'
    )
    
    parser.add_argument(
        '--force-regenerate',
        action='store_true',
        help='Force regeneration of questions even if cache exists'
    )
    
    parser.add_argument(
        '--skip-visualizations',
        action='store_true', 
        help='Skip visualization generation to speed up benchmarking'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='benchmark_results',
        help='Output directory for results and visualizations (default: benchmark_results)'
    )
    
    parser.add_argument(
        '--k-value',
        type=int,
        default=10,
        help='K value for @K metrics (default: 10, matching retrieval steps)'
    )
    
    parser.add_argument(
        '--max-documents',
        type=int,
        default=6,
        help='Maximum documents to show in visualizations (default: 6)'
    )
    
    return parser.parse_args()


class BenchmarkOrchestrator:
    """Main orchestrator for comprehensive RAG algorithm benchmarking."""
    
    def __init__(self, args: argparse.Namespace, logger: logging.Logger):
        """Initialize the benchmark orchestrator with configuration."""
        self.args = args
        self.logger = logger
        self.start_time = datetime.now()
        
        # Setup output directories
        self.output_dir = Path(args.output_dir)
        self.visualization_dir = self.output_dir / "visualizations"
        self.results_file = self.output_dir / "benchmark_results.yaml"
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not args.skip_visualizations:
            self.visualization_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize core components
        self.pipeline = None
        self.knowledge_graph = None
        self.retrieval_orchestrator = None
        self.benchmark_evaluator = None
        
        self.logger.info(f"🚀 RAG Benchmark Orchestrator initialized")
        self.logger.info(f"   Questions: {args.questions}")
        self.logger.info(f"   Algorithms: {args.algorithms}")
        self.logger.info(f"   K-value: {args.k_value}")
        self.logger.info(f"   Output: {self.output_dir}")
        self.logger.info(f"   Visualizations: {'enabled' if not args.skip_visualizations else 'disabled'}")
    
    def initialize_pipeline(self) -> None:
        """Initialize the semantic RAG pipeline and load necessary components."""
        self.logger.info("🔧 Initializing SemanticRAGPipeline...")
        
        try:
            # Initialize pipeline
            self.pipeline = SemanticRAGPipeline()
            
            # Load configuration
            self.pipeline._load_config()
            self.pipeline._initialize_experiment_tracker()
            self.pipeline._initialize_logging()
            
            # Load existing knowledge graph
            kg_path = Path(self.pipeline.config['directories']['data']) / "knowledge_graph.json"
            if not kg_path.exists():
                raise FileNotFoundError(f"Knowledge graph not found at {kg_path}. Please run the full pipeline first.")
            
            self.logger.info(f"📂 Loading knowledge graph from {kg_path}")
            self.knowledge_graph = KnowledgeGraph.load(str(kg_path))
            
            # Load embeddings if available
            try:
                from utils.models import MultiGranularityEmbeddingEngine
                embedding_engine = MultiGranularityEmbeddingEngine(self.pipeline.config, self.logger)
                
                # Try to load cached embeddings for the configured models
                model_names = list(self.pipeline.config['models']['embedding_models'])
                if model_names:
                    primary_model = model_names[0]  # Use first model for compatibility
                    embeddings = embedding_engine.load_model_embeddings(primary_model)
                    
                    if embeddings:
                        # Convert to the format expected by knowledge graph
                        formatted_embeddings = {primary_model: embeddings}
                        self.knowledge_graph.load_phase3_embeddings(formatted_embeddings)
                        self.logger.info("✅ Loaded cached embeddings into knowledge graph")
                    else:
                        self.logger.warning("⚠️ No cached embeddings found - some algorithms may not work optimally")
                else:
                    self.logger.warning("⚠️ No embedding models configured")
                    
            except Exception as e:
                self.logger.warning(f"⚠️ Could not load embeddings: {e}")
            
            # Initialize retrieval orchestrator
            self.retrieval_orchestrator = RetrievalOrchestrator(
                self.knowledge_graph, 
                self.pipeline.config,
                self.logger
            )
            
            # Initialize benchmark evaluator
            self.benchmark_evaluator = BenchmarkEvaluator(self.logger)
            
            # Initialize reranker orchestrator
            self.reranker = create_reranker_orchestrator(self.pipeline.config, self.logger)
            
            self.logger.info(f"✅ Pipeline initialized successfully")
            self.logger.info(f"   Knowledge graph: {len(self.knowledge_graph.chunks)} chunks, "
                           f"{len(self.knowledge_graph.sentences)} sentences")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize pipeline: {e}")
            raise
    
    def load_or_generate_questions(self) -> EvaluationDataset:
        """Load questions from cache or generate new ones with simplified caching approach."""
        questions_file = Path("data/questions.json")
        
        # Check if cached questions exist and should be used
        if not self.args.force_regenerate and questions_file.exists():
            self.logger.info(f"📂 Loading cached questions from {questions_file}")
            try:
                dataset = EvaluationDataset.load(str(questions_file), self.logger)
                
                # Verify we have enough questions
                if len(dataset.questions) >= self.args.questions:
                    # Trim to requested number if we have more
                    if len(dataset.questions) > self.args.questions:
                        dataset.questions = dataset.questions[:self.args.questions]
                        self.logger.info(f"   Using first {self.args.questions} questions from cache")
                    
                    self.logger.info(f"✅ Loaded {len(dataset.questions)} questions from cache")
                    return dataset
                else:
                    self.logger.warning(f"⚠️ Cached dataset has only {len(dataset.questions)} questions, "
                                      f"but {self.args.questions} requested. Generating new questions.")
                    
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to load cached questions: {e}. Generating new questions.")
        
        # Generate new questions
        self.logger.info(f"🧠 Generating {self.args.questions} evaluation questions...")
        
        try:
            # Create dataset generator
            dataset_generator = DatasetGenerator(
                self.knowledge_graph,
                self.pipeline.config,
                self.logger
            )
            
            # Use the proper LLM-based TraversalQuestionGenerator instead of basic templates
            from utils.traversal_question_generator import TraversalQuestionGenerator
            
            question_generator = TraversalQuestionGenerator(
                self.knowledge_graph,
                self.pipeline.config,
                self.logger
            )
            
            # Override question generation config for benchmarking
            if 'question_generation' not in self.pipeline.config:
                self.pipeline.config['question_generation'] = {}
            
            self.pipeline.config['question_generation'].update({
                'generator_model_type': 'ollama',
                'critic_model_type': 'ollama',
                'question_distribution': {
                    'single_hop': 0.2,     # Most reliable for benchmarking
                    'sequential_flow': 0.2,
                    'multi_hop': 0.2,
                    'theme_hop': 0.2,
                    'hierarchical': 0.2
                },
                'max_hops': 3,
                'max_sentences': 10,
                'cache_questions': True,
                'num_questions': self.args.questions
            })
            
            self.logger.info(f"Using LLM-based question generation with:")
            self.logger.info(f"  Generator model: {self.pipeline.config['question_generation']['generator_model_type']}")
            self.logger.info(f"  Distribution: {self.pipeline.config['question_generation']['question_distribution']}")
            
            # Generate dataset using proper LLM-based approach
            dataset = question_generator.generate_dataset(
                num_questions=self.args.questions,
                cache_name="benchmark_questions"
            )
            
            # Save directly to our simplified location
            dataset.save(str(questions_file), self.logger)
            
            self.logger.info(f"✅ Generated and cached {len(dataset.questions)} questions")
            
            return dataset
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate questions: {e}")
            raise
    
    def run_algorithm_benchmark(self, question: Any, algorithm_name: str) -> AlgorithmBenchmarkResult:
        """Run a single algorithm on a single question with reranking and calculate all metrics."""
        self.logger.debug(f"🔍 Running {algorithm_name} on question {question.question_id}")
        
        try:
            # Execute retrieval (raw traversal)
            raw_retrieval_result = self.retrieval_orchestrator.retrieve(
                query=question.question_text,
                algorithm_name=algorithm_name
            )
            
            # Apply reranking to standardize output and optimize relevance
            reranked_sentences, reranking_metadata = self.reranker.rerank_retrieval_result(
                raw_retrieval_result, 
                question.question_text
            )
            
            # Create enhanced retrieval result with reranked content
            enhanced_result = self._create_enhanced_result(
                raw_retrieval_result, 
                reranked_sentences, 
                reranking_metadata
            )
            
            # Calculate comprehensive metrics on reranked result
            benchmark_result = self.benchmark_evaluator.evaluate_single_result(
                retrieval_result=enhanced_result,
                question=question,
                algorithm_name=algorithm_name,
                k=self.args.k_value
            )
            
            # Add reranking metadata to benchmark result
            if not hasattr(benchmark_result, 'reranking_metadata'):
                benchmark_result.reranking_metadata = reranking_metadata
            
            self.logger.debug(f"   ✅ {algorithm_name} (reranked): "
                            f"P@{self.args.k_value}={benchmark_result.traditional_metrics['precision_at_k'].value:.3f}, "
                            f"R@{self.args.k_value}={benchmark_result.traditional_metrics['recall_at_k'].value:.3f}, "
                            f"Node_overlap={benchmark_result.path_metrics['node_overlap_ratio'].value:.3f}, "
                            f"Sentences: {len(reranked_sentences)}")
            
            return benchmark_result, enhanced_result
            
        except Exception as e:
            self.logger.error(f"❌ {algorithm_name} failed on question {question.question_id}: {e}")
            raise
    
    def generate_visualizations(self, 
                              question: Any, 
                              algorithm_name: str, 
                              retrieval_result: Any) -> Dict[str, str]:
        """Generate matplotlib visualizations for a single algorithm result."""
        if self.args.skip_visualizations:
            return {}
        
        self.logger.debug(f"📊 Generating visualizations for {algorithm_name} on {question.question_id}")
        
        visualization_paths = {}
        
        try:
            # Create safe filename components
            safe_question_id = question.question_id.replace('/', '_').replace(' ', '_')[:50]
            safe_algorithm = algorithm_name.replace('/', '_')
            
            # Generate global visualization (strategic overview)
            global_fig = create_global_visualization(
                result=retrieval_result,
                query=question.question_text,
                knowledge_graph=self.knowledge_graph,
                figure_size=(24, 10),  # Larger for global view
                max_documents=self.args.max_documents
            )
            
            global_path = self.visualization_dir / f"{safe_algorithm}_{safe_question_id}_global.png"
            global_fig.savefig(str(global_path), dpi=300, bbox_inches='tight')
            plt.close(global_fig)
            visualization_paths['global'] = str(global_path)
            
            # Generate sequential visualization (tactical analysis)
            sequential_fig = create_sequential_visualization(
                result=retrieval_result,
                query=question.question_text,
                knowledge_graph=self.knowledge_graph,
                figure_size=(20, 8)
            )
            
            sequential_path = self.visualization_dir / f"{safe_algorithm}_{safe_question_id}_sequential.png"
            sequential_fig.savefig(str(sequential_path), dpi=300, bbox_inches='tight')
            plt.close(sequential_fig)
            visualization_paths['sequential'] = str(sequential_path)
            
            self.logger.debug(f"   ✅ Generated 2 visualizations for {algorithm_name}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Visualization generation failed for {algorithm_name}: {e}")
        
        return visualization_paths
    
    def run_comprehensive_benchmark(self) -> CompleteBenchmarkResult:
        """Run the complete benchmark across all questions and algorithms."""
        self.logger.info("🏁 Starting comprehensive benchmark execution...")
        
        # Load questions
        dataset = self.load_or_generate_questions()
        questions = dataset.questions
        
        # Log sample questions for quality assessment
        self._log_sample_questions(dataset)
        
        self.logger.info(f"📝 Benchmarking {len(self.args.algorithms)} algorithms on {len(questions)} questions")
        
        # Track results
        question_results = {}
        total_combinations = len(questions) * len(self.args.algorithms)
        completed_combinations = 0
        
        # Process each question
        for question_idx, question in enumerate(questions):
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"Question {question_idx + 1}/{len(questions)}: {question.question_id}")
            self.logger.info(f"Type: {question.question_type} | Difficulty: {question.difficulty_level}")
            self.logger.info(f"Text: {question.question_text[:100]}...")
            self.logger.info(f"{'='*80}")
            
            question_results[question.question_id] = {}
            
            # Run each algorithm on this question
            for algorithm_name in self.args.algorithms:
                self.logger.info(f"🔍 Running {algorithm_name}...")
                
                try:
                    # Execute algorithm and calculate metrics
                    benchmark_result, retrieval_result = self.run_algorithm_benchmark(question, algorithm_name)
                    
                    # Generate visualizations
                    visualization_paths = self.generate_visualizations(question, algorithm_name, retrieval_result)
                    
                    # Add visualization paths to benchmark result metadata
                    if visualization_paths:
                        if not hasattr(benchmark_result, 'visualization_paths'):
                            benchmark_result.visualization_paths = visualization_paths
                    
                    # Store results
                    question_results[question.question_id][algorithm_name] = benchmark_result
                    
                    # Progress tracking
                    completed_combinations += 1
                    progress = (completed_combinations / total_combinations) * 100
                    
                    self.logger.info(f"   ✅ {algorithm_name} completed successfully")
                    self.logger.info(f"      Precision@{self.args.k_value}: {benchmark_result.traditional_metrics['precision_at_k'].value:.3f}")
                    self.logger.info(f"      Recall@{self.args.k_value}: {benchmark_result.traditional_metrics['recall_at_k'].value:.3f}")
                    self.logger.info(f"      Node Overlap: {benchmark_result.path_metrics['node_overlap_ratio'].value:.3f}")
                    self.logger.info(f"      Processing Time: {benchmark_result.performance_metrics['processing_time'].value:.3f}s")
                    self.logger.info(f"   📊 Progress: {progress:.1f}% ({completed_combinations}/{total_combinations})")
                    
                except Exception as e:
                    self.logger.error(f"   ❌ {algorithm_name} failed: {e}")
                    completed_combinations += 1
                    continue
        
        # Calculate aggregate scores
        self.logger.info(f"\n📊 Calculating aggregate scores across all questions...")
        aggregate_scores = self.benchmark_evaluator.calculate_aggregate_scores(question_results)
        
        # Create complete benchmark result
        benchmark_metadata = {
            'benchmark_timestamp': self.start_time.isoformat(),
            'benchmark_duration_seconds': (datetime.now() - self.start_time).total_seconds(),
            'configuration': {
                'num_questions': len(questions),
                'algorithms_tested': self.args.algorithms,
                'k_value': self.args.k_value,
                'max_documents': self.args.max_documents,
                'visualizations_generated': not self.args.skip_visualizations
            },
            'dataset_metadata': dataset.dataset_metadata,
            'question_distribution': {
                'by_type': self._analyze_question_types(questions),
                'by_difficulty': self._analyze_question_difficulties(questions)
            },
            'knowledge_graph_stats': {
                'total_chunks': len(self.knowledge_graph.chunks),
                'total_sentences': len(self.knowledge_graph.sentences),
                'total_documents': len(self.knowledge_graph.documents) if hasattr(self.knowledge_graph, 'documents') else 'N/A'
            }
        }
        
        complete_result = CompleteBenchmarkResult(
            benchmark_metadata=benchmark_metadata,
            question_results=question_results,
            aggregate_scores=aggregate_scores
        )
        
        self.logger.info(f"✅ Comprehensive benchmark completed successfully!")
        return complete_result
    
    def save_results(self, results: CompleteBenchmarkResult) -> None:
        """Save benchmark results to YAML file."""
        self.logger.info(f"💾 Saving benchmark results to {self.results_file}")
        
        try:
            self.benchmark_evaluator.save_benchmark_results(results, str(self.results_file))
            self.logger.info(f"✅ Results saved successfully")
            
            # Log summary statistics
            self.logger.info(f"\n📊 BENCHMARK SUMMARY")
            self.logger.info(f"{'='*50}")
            
            for algorithm in self.args.algorithms:
                if algorithm in results.aggregate_scores:
                    scores = results.aggregate_scores[algorithm]
                    self.logger.info(f"{algorithm.upper()}:")
                    self.logger.info(f"  Precision@{self.args.k_value}: {scores.get('precision_at_k', 0.0):.3f}")
                    self.logger.info(f"  Recall@{self.args.k_value}: {scores.get('recall_at_k', 0.0):.3f}")
                    self.logger.info(f"  F1@{self.args.k_value}: {scores.get('f1_at_k', 0.0):.3f}")
                    self.logger.info(f"  MRR: {scores.get('mrr', 0.0):.3f}")
                    self.logger.info(f"  Node Overlap: {scores.get('node_overlap_ratio', 0.0):.3f}")
                    self.logger.info(f"  Avg Processing Time: {scores.get('processing_time', 0.0):.3f}s")
                    
                    if f"composite_score_overall" in scores:
                        self.logger.info(f"  Overall Composite Score: {scores['composite_score_overall']:.3f}")
                    self.logger.info("")
            
            # Calculate and log totals
            total_visualizations = 0
            if not self.args.skip_visualizations:
                total_visualizations = len(results.question_results) * len(self.args.algorithms) * 2
            
            self.logger.info(f"TOTALS:")
            self.logger.info(f"  Questions processed: {len(results.question_results)}")
            self.logger.info(f"  Algorithm runs: {len(results.question_results) * len(self.args.algorithms)}")
            self.logger.info(f"  Visualizations generated: {total_visualizations}")
            self.logger.info(f"  Total duration: {results.benchmark_metadata['benchmark_duration_seconds']:.1f}s")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save results: {e}")
            raise
    
    def _analyze_question_types(self, questions: List[Any]) -> Dict[str, int]:
        """Analyze distribution of question types."""
        type_counts = {}
        for question in questions:
            q_type = question.question_type
            type_counts[q_type] = type_counts.get(q_type, 0) + 1
        return type_counts
    
    def _create_enhanced_result(self, raw_result: Any, reranked_sentences: List[str], reranking_metadata: Dict) -> Any:
        """Create enhanced retrieval result with reranked content for fair evaluation."""
        # Create a copy of the original result with reranked content
        enhanced_result = raw_result
        
        # Update the retrieved content with reranked sentences
        enhanced_result.retrieved_content = reranked_sentences
        enhanced_result.reranking_applied = True
        enhanced_result.reranking_metadata = reranking_metadata
        
        # Preserve original metadata while ensuring standardized output
        enhanced_result.final_sentence_count = len(reranked_sentences)
        
        return enhanced_result
    
    def _analyze_question_difficulties(self, questions: List[Any]) -> Dict[str, int]:
        """Analyze distribution of question difficulties."""
        difficulty_counts = {}
        for question in questions:
            difficulty = question.difficulty_level
            difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
        return difficulty_counts
    
    def _log_sample_questions(self, dataset: Any) -> None:
        """Log sample questions for human review and quality assessment."""
        questions = dataset.questions
        
        self.logger.info(f"\n🔍 SAMPLE GENERATED QUESTIONS FOR QUALITY REVIEW")
        self.logger.info(f"{'='*80}")
        
        # Group questions by type
        questions_by_type = {}
        for question in questions:
            q_type = question.question_type
            if q_type not in questions_by_type:
                questions_by_type[q_type] = []
            questions_by_type[q_type].append(question)
        
        # Show up to 2 samples per type
        for q_type, type_questions in questions_by_type.items():
            self.logger.info(f"\n📝 {q_type.upper()} Questions ({len(type_questions)} total):")
            
            samples_to_show = min(2, len(type_questions))
            for i in range(samples_to_show):
                question = type_questions[i]
                
                self.logger.info(f"   Q{i+1} [{question.difficulty_level}]: {question.question_text}")
                self.logger.info(f"       Expected Answer: {question.expected_answer[:100]}{'...' if len(question.expected_answer) > 100 else ''}")
                self.logger.info(f"       Ground Truth Nodes: {len(question.ground_truth_path.nodes)} nodes")
                self.logger.info(f"       Connection Types: {[str(ct) for ct in question.ground_truth_path.connection_types]}")
                
                if i < samples_to_show - 1:  # Add separator between questions
                    self.logger.info("       ---")
        
        # Log generation success rates
        self.logger.info(f"\n📊 QUESTION GENERATION SUCCESS SUMMARY:")
        total_requested = sum(dataset.dataset_metadata.get('questions_by_type', {}).values())
        total_generated = len(questions)
        
        for q_type in ['single_hop', 'sequential_flow', 'multi_hop', 'theme_hop', 'hierarchical']:
            generated_count = len(questions_by_type.get(q_type, []))
            self.logger.info(f"   {q_type}: {generated_count} questions generated")
        
        success_rate = (total_generated / total_requested * 100) if total_requested > 0 else 0
        self.logger.info(f"\n   Overall Success Rate: {total_generated}/{total_requested} ({success_rate:.1f}%)")
        
        self.logger.info(f"{'='*80}\n")
    
    def run(self) -> None:
        """Execute the complete benchmarking process."""
        try:
            # Initialize all components
            self.initialize_pipeline()
            
            # Run comprehensive benchmark
            results = self.run_comprehensive_benchmark()
            
            # Save results
            self.save_results(results)
            
            # Final summary
            total_duration = (datetime.now() - self.start_time).total_seconds()
            self.logger.info(f"\n🎉 BENCHMARK COMPLETED SUCCESSFULLY!")
            self.logger.info(f"   Total Duration: {total_duration:.1f}s ({total_duration/60:.1f}m)")
            self.logger.info(f"   Results File: {self.results_file}")
            self.logger.info(f"   Visualizations: {self.visualization_dir}")
            
        except Exception as e:
            self.logger.error(f"💥 Benchmark failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            sys.exit(1)


def main():
    """Main entry point for the benchmarking system."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging()
    logger.info("🚀 RAG Algorithm Benchmarking System Starting...")
    
    # Create and run orchestrator
    orchestrator = BenchmarkOrchestrator(args, logger)
    orchestrator.run()


if __name__ == "__main__":
    main()
