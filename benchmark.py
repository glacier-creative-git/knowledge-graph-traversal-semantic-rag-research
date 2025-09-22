#!/usr/bin/env python3
"""
DeepEval-Powered Semantic Traversal Benchmark Orchestrator
=========================================================

Orchestrates knowledge graph building, dataset generation, and algorithm evaluation
using deepeval's synthetic data generation and sophisticated evaluation metrics.

This script replaces the original benchmark.py with a clean three-phase architecture:
1. Knowledge Graph Construction (kg_pipeline.build)
2. Synthetic Dataset Generation (dataset.build) 
3. Algorithm Evaluation (evaluation.run)

Key Features:
- Configurable model providers (OpenAI, Ollama, Anthropic, OpenRouter)
- Evolution-based question complexity enhancement
- Comprehensive RAG + custom semantic traversal metrics
- Hyperparameter tracking for dashboard visibility
- Results compatible with existing visualization infrastructure

Usage:
    python benchmark.py                                    # Full pipeline with all algorithms
    python benchmark.py --algorithm triangulation_centroid # Test specific algorithm
    python benchmark.py --dataset-only                    # Generate dataset only
    python benchmark.py --models ollama openai            # Override model configuration
    python benchmark.py --force-rebuild-all               # Force rebuild everything
"""

import sys
import argparse
import logging
import yaml
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Environment variable management
from dotenv import load_dotenv

# Import our three main components
from utils.kg_pipeline import KnowledgeGraphPipeline
from evaluation.dataset import DatasetBuilder
from evaluation.evaluation import EvaluationOrchestrator


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup comprehensive logging for the benchmark orchestrator."""
    log_level = logging.DEBUG if verbose else logging.INFO
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('benchmark.log', mode='w', encoding='utf-8')
        ]
    )
    
    return logging.getLogger("DeepEvalBenchmark")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for benchmark configuration."""
    parser = argparse.ArgumentParser(
        description="DeepEval-Powered Semantic Traversal Benchmark System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python benchmark.py                                     # Full pipeline with all algorithms
    python benchmark.py --algorithm triangulation_centroid # Test specific algorithm
    python benchmark.py --dataset-only                     # Generate dataset only  
    python benchmark.py --force-rebuild-all                # Force rebuild everything
    python benchmark.py --verbose                          # Enable debug logging
    python benchmark.py --config custom_config.yaml       # Use custom configuration
        """
    )
    
    # Algorithm selection
    parser.add_argument(
        '--algorithm',
        type=str,
        choices=['basic_retrieval', 'query_traversal', 'kg_traversal', 'triangulation_centroid'],
        help='Specific algorithm to benchmark (default: evaluate all algorithms)'
    )
    
    # Pipeline control
    parser.add_argument(
        '--dataset-only',
        action='store_true',
        help='Generate synthetic dataset only, skip evaluation'
    )
    
    parser.add_argument(
        '--evaluation-only',
        action='store_true',
        help='Run evaluation only, skip KG building and dataset generation'
    )
    
    # Force rebuild options
    parser.add_argument(
        '--force-rebuild-kg',
        action='store_true',
        help='Force knowledge graph rebuild even if cache exists'
    )
    
    parser.add_argument(
        '--force-rebuild-dataset',
        action='store_true',
        help='Force dataset regeneration even if cache exists'
    )
    
    parser.add_argument(
        '--force-rebuild-all',
        action='store_true',
        help='Force rebuild of all components (KG + dataset)'
    )
    
    # Configuration and output
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--output-prefix',
        type=str,
        help='Prefix for output files and reports'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose debug logging'
    )
    
    # Model overrides (for future use)
    parser.add_argument(
        '--question-model',
        type=str,
        help='Override question generation model (e.g., "ollama", "openai")'
    )
    
    parser.add_argument(
        '--evaluation-model',
        type=str,
        help='Override evaluation judge model (e.g., "gpt-4o", "claude-3-sonnet")'
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate configuration from YAML file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required configuration sections
    required_sections = ['directories', 'deepeval']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    return config


def apply_model_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Apply command-line model overrides to configuration."""
    if args.question_model:
        config['deepeval']['models']['question_generation']['provider'] = args.question_model
        logging.getLogger("DeepEvalBenchmark").info(f"🔧 Override: Question generation model -> {args.question_model}")
    
    if args.evaluation_model:
        config['deepeval']['models']['evaluation_judge']['model_name'] = args.evaluation_model
        logging.getLogger("DeepEvalBenchmark").info(f"🔧 Override: Evaluation judge model -> {args.evaluation_model}")
    
    return config


def validate_environment() -> None:
    """Validate environment setup and API keys."""
    import os
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Check for at least one valid model provider configuration
    providers_available = []
    
    if os.getenv('OPENAI_API_KEY'):
        providers_available.append('OpenAI')
    
    if os.getenv('ANTHROPIC_API_KEY'):
        providers_available.append('Anthropic')
    
    if os.getenv('OPENROUTER_API_KEY'):
        providers_available.append('OpenRouter')
    
    # Ollama doesn't require API key, just check if URL is configured
    ollama_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    providers_available.append('Ollama (local)')
    
    if not providers_available:
        raise ValueError(
            "No valid model provider configuration found. "
            "Please set appropriate API keys in your .env file."
        )
    
    logging.getLogger("DeepEvalBenchmark").info(f"✅ Available providers: {', '.join(providers_available)}")


def run_kg_pipeline_phase(config: Dict[str, Any], force_rebuild: bool, logger: logging.Logger) -> Dict[str, Any]:
    """Execute Phase 1: Knowledge Graph Construction."""
    logger.info("📊 Phase 1: Knowledge Graph Construction")
    logger.info("=" * 50)
    
    try:
        # Check if KG already exists and force_rebuild is False
        kg_path = Path(config['directories']['data']) / "knowledge_graph.json"
        
        if kg_path.exists() and not force_rebuild:
            logger.info(f"✅ Knowledge graph found at {kg_path} - skipping rebuild")
            logger.info("   Use --force-rebuild-kg to force regeneration")
            return {"status": "loaded_existing", "path": str(kg_path)}
        
        # Build knowledge graph using existing pipeline
        logger.info("🏗️ Building knowledge graph from Wikipedia data...")
        pipeline = KnowledgeGraphPipeline()
        result = pipeline.build()
        
        logger.info(f"✅ Knowledge graph construction completed")
        logger.info(f"   Result: {result}")
        
        return {"status": "built_new", "result": result}
        
    except Exception as e:
        logger.error(f"❌ Phase 1 failed: {e}")
        raise RuntimeError(f"Knowledge graph construction failed: {e}")


def run_dataset_generation_phase(config: Dict[str, Any], force_rebuild: bool, logger: logging.Logger) -> Dict[str, Any]:
    """Execute Phase 2: Synthetic Dataset Generation."""
    logger.info("🧠 Phase 2: Synthetic Dataset Generation")
    logger.info("=" * 50)
    
    try:
        # Initialize dataset builder
        dataset_builder = DatasetBuilder(config, logger)
        
        # Generate synthetic dataset with evolution techniques
        dataset = dataset_builder.build(force_regenerate=force_rebuild)
        
        logger.info(f"✅ Dataset generation completed")
        logger.info(f"   Generated: {len(dataset.goldens)} synthetic questions")
        
        # Log evolution statistics if available
        generation_stats = getattr(dataset_builder, 'generation_stats', {})
        evolution_dist = generation_stats.get('evolution_distribution', {})
        
        if evolution_dist:
            logger.info("   Evolution technique distribution:")
            for evolution_type, count in evolution_dist.items():
                logger.info(f"      {evolution_type}: {count}")
        
        return {
            "status": "generated",
            "dataset_size": len(dataset.goldens),
            "evolution_distribution": evolution_dist
        }
        
    except Exception as e:
        logger.error(f"❌ Phase 2 failed: {e}")
        raise RuntimeError(f"Dataset generation failed: {e}")


def run_evaluation_phase(config: Dict[str, Any], algorithm: Optional[str], 
                        output_prefix: Optional[str], logger: logging.Logger) -> Dict[str, Any]:
    """Execute Phase 3: Algorithm Evaluation."""
    logger.info("🔍 Phase 3: Algorithm Evaluation")
    logger.info("=" * 50)
    
    try:
        # Initialize evaluation orchestrator
        evaluation_orchestrator = EvaluationOrchestrator(config, logger)
        
        if algorithm:
            # Single algorithm evaluation
            logger.info(f"🎯 Evaluating single algorithm: {algorithm}")
            
            result = evaluation_orchestrator.run(
                algorithm_name=algorithm,
                output_prefix=output_prefix
            )
            
            logger.info(f"✅ Single algorithm evaluation completed")
            logger.info(f"   Algorithm: {result.algorithm_name}")
            logger.info(f"   Success rate: {result.summary_statistics['overall_success_rate']:.1%}")
            logger.info(f"   Average score: {result.summary_statistics['average_metric_score']:.3f}")
            
            return {
                "status": "single_algorithm",
                "algorithm": result.algorithm_name,
                "success_rate": result.summary_statistics['overall_success_rate'],
                "average_score": result.summary_statistics['average_metric_score'],
                "hyperparameters": result.algorithm_hyperparameters
            }
            
        else:
            # Comparative evaluation across all algorithms
            logger.info("🏁 Evaluating all configured algorithms...")
            
            results = evaluation_orchestrator.run_comparison(output_prefix=output_prefix)
            
            logger.info(f"✅ Comparative evaluation completed")
            logger.info(f"   Algorithms tested: {len(results)}")
            
            # Log summary comparison with hyperparameters
            logger.info("📊 Algorithm Performance Summary:")
            for algorithm_name, result in results.items():
                avg_score = result.summary_statistics['average_metric_score']
                success_rate = result.summary_statistics['overall_success_rate']
                hyperparams = result.algorithm_hyperparameters
                
                logger.info(f"   {algorithm_name}:")
                logger.info(f"      Average score: {avg_score:.3f}")
                logger.info(f"      Success rate: {success_rate:.1%}")
                logger.info(f"      Hyperparameters: {hyperparams}")
            
            # Identify best performing algorithm
            best_algorithm = max(results.keys(), 
                               key=lambda k: results[k].summary_statistics['average_metric_score'])
            logger.info(f"🏆 Best performing algorithm: {best_algorithm}")
            
            return {
                "status": "comparative_evaluation",
                "algorithms_tested": list(results.keys()),
                "best_algorithm": best_algorithm,
                "results_summary": {
                    alg: {
                        "average_score": res.summary_statistics['average_metric_score'],
                        "success_rate": res.summary_statistics['overall_success_rate'],
                        "hyperparameters": res.algorithm_hyperparameters
                    }
                    for alg, res in results.items()
                }
            }
            
    except Exception as e:
        logger.error(f"❌ Phase 3 failed: {e}")
        raise RuntimeError(f"Algorithm evaluation failed: {e}")


def main():
    """Main orchestrator function - coordinates all three phases."""
    # Parse arguments and setup logging
    args = parse_arguments()
    logger = setup_logging(args.verbose)
    
    logger.info("🚀 DeepEval Semantic Traversal Benchmark Starting")
    logger.info("=" * 70)
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    
    total_start_time = datetime.now()
    
    try:
        # Load and validate configuration
        logger.info("🔧 Loading configuration and validating environment...")
        config = load_config(args.config)
        config = apply_model_overrides(config, args)
        validate_environment()
        
        # Determine force rebuild flags
        force_rebuild_kg = args.force_rebuild_kg or args.force_rebuild_all
        force_rebuild_dataset = args.force_rebuild_dataset or args.force_rebuild_all
        
        phase_results = {}
        
        # Phase 1: Knowledge Graph Construction (unless evaluation-only)
        if not args.evaluation_only:
            kg_result = run_kg_pipeline_phase(config, force_rebuild_kg, logger)
            phase_results['knowledge_graph'] = kg_result
        
        # Phase 2: Synthetic Dataset Generation (unless evaluation-only or dataset-only completed)
        if not args.evaluation_only:
            dataset_result = run_dataset_generation_phase(config, force_rebuild_dataset, logger)
            phase_results['dataset'] = dataset_result
            
            # Exit early if dataset-only mode
            if args.dataset_only:
                logger.info("🎯 Dataset-only mode completed successfully!")
                logger.info(f"   Generated: {dataset_result['dataset_size']} questions")
                return
        
        # Phase 3: Algorithm Evaluation (unless dataset-only)
        if not args.dataset_only:
            evaluation_result = run_evaluation_phase(config, args.algorithm, args.output_prefix, logger)
            phase_results['evaluation'] = evaluation_result
        
        # Final summary
        total_duration = datetime.now() - total_start_time
        logger.info("🎉 BENCHMARK PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 70)
        logger.info(f"Total Duration: {total_duration}")
        
        # Log phase summaries
        for phase_name, result in phase_results.items():
            logger.info(f"{phase_name.title()}: {result['status']}")
        
        # Special summary for evaluation results
        if 'evaluation' in phase_results:
            eval_result = phase_results['evaluation']
            if eval_result['status'] == 'comparative_evaluation':
                logger.info(f"🏆 Champion Algorithm: {eval_result['best_algorithm']}")
            elif eval_result['status'] == 'single_algorithm':
                logger.info(f"🎯 Algorithm Tested: {eval_result['algorithm']} (Score: {eval_result['average_score']:.3f})")
        
        logger.info("📁 Results saved to benchmark_results/ directory")
        logger.info("📊 Use existing visualization tools to analyze results")
        
    except KeyboardInterrupt:
        logger.info("⚠️ Benchmark interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"💥 Benchmark failed: {e}")
        
        if args.verbose:
            logger.error("Full traceback:")
            logger.error(traceback.format_exc())
        else:
            logger.error("Use --verbose for full traceback")
        
        sys.exit(1)


if __name__ == "__main__":
    main()
