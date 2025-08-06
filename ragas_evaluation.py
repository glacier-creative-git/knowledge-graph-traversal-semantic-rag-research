#!/usr/bin/env python3
"""
Enhanced RAGAS Evaluation Script
===============================

Comprehensive RAGAS evaluation of the Semantic Graph RAG system
supporting WikiEval, Natural Questions, and future datasets with
multiple difficulty levels and configurations.

Updated to use the enhanced data loading system instead of SQuAD dependency.

Usage:
    python ragas_evaluation.py --dataset wikieval
    python ragas_evaluation.py --dataset natural_questions --max_samples 50
"""

import os
import sys
import time
import argparse
from typing import Dict, List

# Add utils to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import (
    ResearchPipeline,
    get_config,
    create_data_loader,
    print_evaluation_results,
    get_available_datasets,
    print_dataset_info
)

# ============================================================================
# CONFIGURATION - SET YOUR API KEY HERE
# ============================================================================

OPENAI_API_KEY = "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"  # 🔑 PUT YOUR API KEY HERE

# Enhanced evaluation configurations supporting multiple datasets
EVALUATION_CONFIGS = {
    "WikiEval_Conservative": {
        "dataset_name": "wikieval",
        "similarity_threshold": 0.4,
        "top_k_per_sentence": 15,
        "cross_doc_k": 8,
        "retrieval_top_k": 5,
        "max_eval_samples": 25  # WikiEval only has 50 samples
    },
    "WikiEval_Balanced": {
        "dataset_name": "wikieval",
        "similarity_threshold": 0.5,
        "top_k_per_sentence": 20,
        "cross_doc_k": 10,
        "retrieval_top_k": 8,
        "max_eval_samples": 30
    },
    "WikiEval_Aggressive": {
        "dataset_name": "wikieval",
        "similarity_threshold": 0.7,
        "top_k_per_sentence": 25,
        "cross_doc_k": 12,
        "retrieval_top_k": 10,
        "max_eval_samples": 20
    },
    "NaturalQuestions_Conservative": {
        "dataset_name": "natural_questions",
        "similarity_threshold": 0.3,
        "top_k_per_sentence": 20,
        "cross_doc_k": 12,
        "retrieval_top_k": 8,
        "max_eval_samples": 30
    },
    "NaturalQuestions_Balanced": {
        "dataset_name": "natural_questions",
        "similarity_threshold": 0.5,
        "top_k_per_sentence": 25,
        "cross_doc_k": 15,
        "retrieval_top_k": 10,
        "max_eval_samples": 40
    },
    "NaturalQuestions_Aggressive": {
        "dataset_name": "natural_questions",
        "similarity_threshold": 0.7,
        "top_k_per_sentence": 30,
        "cross_doc_k": 18,
        "retrieval_top_k": 12,
        "max_eval_samples": 25
    }
}


def print_header():
    """Print enhanced header with dataset support information"""
    print("🔥" * 70)
    print("🔥" + " " * 66 + "🔥")
    print("🔥" + "     ENHANCED RAGAS EVALUATION - SEMANTIC GRAPH RAG     ".center(66) + "🔥")
    print("🔥" + " " * 66 + "🔥")
    print("🔥" * 70)
    print()
    print("📊 This evaluation supports multiple datasets:")
    print("   • WikiEval: 50 human-annotated Wikipedia QA pairs")
    print("   • Natural Questions: Real Google search queries")
    print("⚡ Each configuration will be tested on 20-40 samples")
    print("🎯 Metrics: Context Precision, Context Recall, Faithfulness, Answer Relevancy")
    print("⏱️  Expected runtime: 10-20 minutes")
    print()


def validate_api_key(api_key: str) -> bool:
    """Validate the OpenAI API key format"""
    if not api_key or api_key == "your-openai-api-key-here":
        print("❌ ERROR: Please set your OpenAI API key in the script!")
        print("   Edit ragas_evaluation.py and replace 'your-openai-api-key-here'")
        print("   with your actual OpenAI API key.")
        return False

    if not api_key.startswith("sk-"):
        print("⚠️  WARNING: API key format looks incorrect (should start with 'sk-')")
        print("   Continuing anyway...")

    return True


def run_single_evaluation(config_name: str, config_params: Dict, target_dataset: str = None) -> Dict:
    """
    Run evaluation for a single configuration on the specified dataset.

    Args:
        config_name: Name of the configuration
        config_params: Configuration parameters
        target_dataset: Override dataset name (optional)

    Returns:
        Dictionary with evaluation results
    """
    # Override dataset if specified
    if target_dataset:
        config_params = config_params.copy()
        config_params["dataset_name"] = target_dataset

    dataset_name = config_params["dataset_name"]

    print(f"\n{'=' * 60}")
    print(f"🔬 EVALUATING: {config_name.upper()}")
    print(f"📊 Dataset: {dataset_name.replace('_', ' ').title()}")
    print(f"{'=' * 60}")

    # Print configuration details
    print(f"📋 Configuration Parameters:")
    for param, value in config_params.items():
        print(f"   {param}: {value}")
    print()

    try:
        # Create configuration using the enhanced system
        config = get_config(
            "default",
            openai_api_key=OPENAI_API_KEY,
            **config_params
        )

        # Create pipeline with the configured dataset
        pipeline = ResearchPipeline(config)

        # Verify data loader can load the dataset
        print(f"📚 Loading {dataset_name.replace('_', ' ').title()} dataset...")
        if not pipeline.data_loader.load_dataset():
            print(f"❌ Failed to load {dataset_name}, falling back to demo data")
            return {"error": f"Failed to load {dataset_name}"}

        # Get dataset info for logging
        eval_dataset = pipeline.data_loader.get_evaluation_dataset(config_params['max_eval_samples'])
        print(f"✅ Loaded {len(eval_dataset['documents'])} evaluation samples")

        # Run RAGAS evaluation
        print(f"⚡ Running RAGAS evaluation...")
        eval_results = pipeline.run_ragas_evaluation()

        # Add configuration info to results
        results = {
            'config_name': config_name,
            'dataset_name': dataset_name,
            'config_params': config_params,
            'ragas_results': eval_results,
            'success': eval_results.error is None
        }

        return results

    except Exception as e:
        print(f"❌ Evaluation failed for {config_name}: {str(e)}")
        return {
            'config_name': config_name,
            'dataset_name': dataset_name,
            'config_params': config_params,
            'error': str(e),
            'success': False
        }


def filter_configs_by_dataset(configs: Dict[str, Dict], target_dataset: str) -> Dict[str, Dict]:
    """
    Filter configurations to only include those for the specified dataset.

    Args:
        configs: All available configurations
        target_dataset: Target dataset name

    Returns:
        Filtered configurations
    """
    if target_dataset == "all":
        return configs

    filtered = {}
    for name, params in configs.items():
        if target_dataset in params.get("dataset_name", ""):
            filtered[name] = params
        # Also include configs that can be adapted
        elif target_dataset in ["wikieval", "natural_questions"] and not any(
                ds in name.lower() for ds in ["wikieval", "naturalquestions", "natural_questions"]
        ):
            # Generic config that can be adapted
            filtered[f"{target_dataset.replace('_', '').title()}_{name}"] = {**params, "dataset_name": target_dataset}

    return filtered


def print_comparison_table(all_results: List[Dict]):
    """Print enhanced comparison table with dataset information"""
    print(f"\n{'=' * 90}")
    print(f"📊 COMPREHENSIVE RESULTS COMPARISON")
    print(f"{'=' * 90}")

    # Filter successful results
    successful_results = [r for r in all_results if r.get('success', False)]

    if not successful_results:
        print("❌ No successful evaluations to compare!")
        return

    # Print table header
    print(
        f"{'Configuration':<20} {'Dataset':<15} {'RAGAS':<8} {'Precision':<10} {'Recall':<8} {'Faithful':<10} {'Relevancy':<10} {'Time':<8}")
    print(f"{'-' * 90}")

    # Print results for each configuration
    for result in successful_results:
        config_name = result['config_name'][:19]  # Truncate long names
        dataset_name = result['dataset_name'].replace('_', ' ')[:14]
        ragas_results = result['ragas_results']

        print(f"{config_name:<20} "
              f"{dataset_name:<15} "
              f"{ragas_results.ragas_score:<8.3f} "
              f"{ragas_results.context_precision:<10.3f} "
              f"{ragas_results.context_recall:<8.3f} "
              f"{ragas_results.faithfulness:<10.3f} "
              f"{ragas_results.answer_relevancy:<10.3f} "
              f"{ragas_results.eval_time:<8.1f}s")

    # Find best configuration overall
    best_config = max(successful_results, key=lambda x: x['ragas_results'].ragas_score)
    print(f"\n🏆 BEST OVERALL CONFIGURATION: {best_config['config_name']}")
    print(f"   📊 Dataset: {best_config['dataset_name'].replace('_', ' ').title()}")
    print(f"   🎯 RAGAS Score: {best_config['ragas_results'].ragas_score:.3f}")

    # Find best per dataset
    datasets = set(r['dataset_name'] for r in successful_results)
    for dataset in datasets:
        dataset_results = [r for r in successful_results if r['dataset_name'] == dataset]
        if dataset_results:
            best_for_dataset = max(dataset_results, key=lambda x: x['ragas_results'].ragas_score)
            print(f"   📊 Best for {dataset.replace('_', ' ').title()}: {best_for_dataset['config_name']} "
                  f"(RAGAS: {best_for_dataset['ragas_results'].ragas_score:.3f})")


def print_detailed_analysis(all_results: List[Dict]):
    """Print detailed analysis with dataset-specific insights"""
    successful_results = [r for r in all_results if r.get('success', False)]

    if len(successful_results) < 2:
        return

    print(f"\n{'=' * 90}")
    print(f"🔬 DETAILED ANALYSIS")
    print(f"{'=' * 90}")

    # Dataset performance comparison
    datasets = set(r['dataset_name'] for r in successful_results)
    print(f"📊 Dataset Performance Summary:")

    for dataset in datasets:
        dataset_results = [r for r in successful_results if r['dataset_name'] == dataset]
        if dataset_results:
            scores = [r['ragas_results'].ragas_score for r in dataset_results]
            avg_score = sum(scores) / len(scores)
            best_score = max(scores)
            dataset_display = dataset.replace('_', ' ').title()
            print(
                f"   {dataset_display}: Average RAGAS {avg_score:.3f}, Best {best_score:.3f} ({len(dataset_results)} configs)")

    # Threshold impact analysis
    threshold_analysis = {}
    for result in successful_results:
        threshold = result['config_params']['similarity_threshold']
        dataset = result['dataset_name']
        ragas_score = result['ragas_results'].ragas_score

        key = f"{dataset}_{threshold}"
        if key not in threshold_analysis:
            threshold_analysis[key] = []
        threshold_analysis[key].append(ragas_score)

    print(f"\n📈 Similarity Threshold Impact by Dataset:")
    for key, scores in threshold_analysis.items():
        dataset, threshold = key.split('_', 1)
        avg_score = sum(scores) / len(scores)
        dataset_display = dataset.replace('_', ' ').title()
        print(f"   {dataset_display} @ threshold {threshold}: RAGAS {avg_score:.3f}")

    # Performance analysis
    times = [r['ragas_results'].eval_time for r in successful_results]
    samples = [r['ragas_results'].num_samples for r in successful_results]

    print(f"\n⚡ Performance Analysis:")
    print(f"   Average evaluation time: {sum(times) / len(times):.1f}s")
    print(f"   Total samples evaluated: {sum(samples)}")
    print(f"   Average time per sample: {sum(times) / sum(samples):.2f}s")

    # Quality insights
    ragas_scores = [r['ragas_results'].ragas_score for r in successful_results]
    print(f"\n🎯 Quality Insights:")
    print(f"   RAGAS score range: {min(ragas_scores):.3f} - {max(ragas_scores):.3f}")
    print(f"   Performance improvement: {((max(ragas_scores) - min(ragas_scores)) / min(ragas_scores) * 100):.1f}%")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Enhanced RAGAS Evaluation with Multiple Dataset Support")

    parser.add_argument("--dataset", type=str, default="all",
                        choices=["all", "wikieval", "natural_questions"],
                        help="Dataset to evaluate on (default: all)")

    parser.add_argument("--config", type=str, default="all",
                        help="Specific configuration to run (default: all)")

    parser.add_argument("--max-samples", type=int, default=None,
                        help="Override max evaluation samples")

    parser.add_argument("--list-datasets", action="store_true",
                        help="List available datasets and exit")

    parser.add_argument("--api-key", type=str, default=OPENAI_API_KEY,
                        help="OpenAI API key (overrides script default)")

    return parser.parse_args()


def main():
    """Main evaluation function with enhanced dataset support"""
    args = parse_arguments()

    # Handle dataset listing
    if args.list_datasets:
        print_dataset_info()
        return

    print_header()

    # Update API key if provided
    global OPENAI_API_KEY
    if args.api_key:
        OPENAI_API_KEY = args.api_key

    # Validate API key
    if not validate_api_key(OPENAI_API_KEY):
        return

    # Set environment variable
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid tokenizer warnings

    # Filter configurations based on dataset selection
    if args.dataset != "all":
        configs_to_run = filter_configs_by_dataset(EVALUATION_CONFIGS, args.dataset)
        print(f"🎯 Running evaluations for dataset: {args.dataset.replace('_', ' ').title()}")
    else:
        configs_to_run = EVALUATION_CONFIGS
        print(f"🎯 Running evaluations for all datasets")

    # Filter by specific config if requested
    if args.config != "all":
        if args.config in configs_to_run:
            configs_to_run = {args.config: configs_to_run[args.config]}
        else:
            print(f"❌ Configuration '{args.config}' not found.")
            print(f"Available configurations: {list(configs_to_run.keys())}")
            return

    # Override max samples if specified
    if args.max_samples:
        for config in configs_to_run.values():
            config['max_eval_samples'] = args.max_samples

    print(f"📝 Testing {len(configs_to_run)} different configurations")

    # Run all evaluations
    all_results = []
    total_start_time = time.time()

    for i, (config_name, config_params) in enumerate(configs_to_run.items(), 1):
        print(f"\n⏳ Progress: {i}/{len(configs_to_run)} configurations")

        result = run_single_evaluation(config_name, config_params)
        all_results.append(result)

        # Brief summary after each evaluation
        if result.get('success'):
            ragas_score = result['ragas_results'].ragas_score
            dataset_name = result['dataset_name'].replace('_', ' ').title()
            print(f"✅ {config_name} ({dataset_name}): RAGAS Score {ragas_score:.3f}")
        else:
            print(f"❌ {config_name}: Failed")

    total_time = time.time() - total_start_time

    # Print comprehensive results
    print_comparison_table(all_results)
    print_detailed_analysis(all_results)

    # Final summary
    successful_count = sum(1 for r in all_results if r.get('success', False))
    print(f"\n{'=' * 90}")
    print(f"🎉 EVALUATION COMPLETE!")
    print(f"{'=' * 90}")
    print(f"✅ Successful evaluations: {successful_count}/{len(configs_to_run)}")
    print(f"⏱️  Total time: {total_time:.1f}s ({total_time / 60:.1f} minutes)")

    if successful_count > 0:
        best_result = max([r for r in all_results if r.get('success', False)],
                          key=lambda x: x['ragas_results'].ragas_score)
        print(f"🏆 Best overall RAGAS score: {best_result['ragas_results'].ragas_score:.3f}")
        print(f"   Configuration: {best_result['config_name']}")
        print(f"   Dataset: {best_result['dataset_name'].replace('_', ' ').title()}")

        # Recommendation
        print(f"\n💡 RECOMMENDATION:")
        print(f"   Use the '{best_result['config_name']}' configuration")
        print(f"   on {best_result['dataset_name'].replace('_', ' ').title()} for optimal performance")
    else:
        print(f"❌ No successful evaluations. Check your API key and internet connection.")


if __name__ == "__main__":
    main()