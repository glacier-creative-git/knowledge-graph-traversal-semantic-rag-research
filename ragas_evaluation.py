#!/usr/bin/env python3
"""
Challenging RAGAS Evaluation Script
==================================

This script runs a comprehensive RAGAS evaluation of the Semantic Graph RAG system
using a substantial portion of the SQuAD dataset with multiple difficulty levels.

Usage:
    python ragas_evaluation.py

Make sure to set your OpenAI API key in the script below.
"""

import os
import sys
import time
from typing import Dict, List

# Add utils to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import (
    ResearchPipeline,
    get_config,
    SQuADDataLoader,
    print_evaluation_results
)

# ============================================================================
# CONFIGURATION - SET YOUR API KEY HERE
# ============================================================================

OPENAI_API_KEY = "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"  # 🔑 PUT YOUR API KEY HERE

# Evaluation configurations to test
EVALUATION_CONFIGS = {
    "Conservative": {
        "similarity_threshold": 0.3,
        "top_k_per_sentence": 15,
        "cross_doc_k": 8,
        "num_contexts": 3,
        "retrieval_top_k": 5,
        "max_eval_samples": 25
    },
    "Balanced": {
        "similarity_threshold": 0.5,
        "top_k_per_sentence": 20,
        "cross_doc_k": 10,
        "num_contexts": 5,
        "retrieval_top_k": 8,
        "max_eval_samples": 30
    },
    "Aggressive": {
        "similarity_threshold": 0.7,
        "top_k_per_sentence": 25,
        "cross_doc_k": 12,
        "num_contexts": 5,
        "retrieval_top_k": 10,
        "max_eval_samples": 20
    }
}


def print_header():
    """Print a nice header for the evaluation"""
    print("🔥" * 70)
    print("🔥" + " " * 66 + "🔥")
    print("🔥" + "     CHALLENGING RAGAS EVALUATION - SEMANTIC GRAPH RAG     ".center(66) + "🔥")
    print("🔥" + " " * 66 + "🔥")
    print("🔥" * 70)
    print()
    print("📊 This evaluation will test multiple configurations on SQuAD 2.0")
    print("⚡ Each configuration will be tested on 20-30 samples")
    print("🎯 Metrics: Context Precision, Context Recall, Faithfulness, Answer Relevancy")
    print("⏱️  Expected runtime: 10-15 minutes")
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


def run_single_evaluation(config_name: str, config_params: Dict) -> Dict:
    """Run evaluation for a single configuration"""
    print(f"\n{'=' * 60}")
    print(f"🔬 EVALUATING: {config_name.upper()} CONFIGURATION")
    print(f"{'=' * 60}")

    # Print configuration details
    print(f"📋 Configuration Parameters:")
    for param, value in config_params.items():
        print(f"   {param}: {value}")
    print()

    try:
        # Create configuration
        config = get_config(
            "default",  # Use full dataset, not demo
            openai_api_key=OPENAI_API_KEY,
            **config_params
        )

        # Create pipeline
        pipeline = ResearchPipeline(config)

        # Load evaluation dataset
        print("📚 Loading SQuAD 2.0 evaluation dataset...")
        loader = SQuADDataLoader(config.data)
        if not loader.load_squad_data("2.0"):
            print("❌ Failed to load SQuAD 2.0, falling back to demo data")
            return {"error": "Failed to load SQuAD data"}

        eval_dataset = loader.get_evaluation_dataset(config_params['max_eval_samples'])
        print(f"✅ Loaded {len(eval_dataset['documents'])} evaluation samples")

        # Run RAGAS evaluation
        print(f"⚡ Running RAGAS evaluation...")
        eval_results = pipeline.run_ragas_evaluation()

        # Add configuration info to results
        results = {
            'config_name': config_name,
            'config_params': config_params,
            'ragas_results': eval_results,
            'success': eval_results.error is None
        }

        return results

    except Exception as e:
        print(f"❌ Evaluation failed for {config_name}: {str(e)}")
        return {
            'config_name': config_name,
            'config_params': config_params,
            'error': str(e),
            'success': False
        }


def print_comparison_table(all_results: List[Dict]):
    """Print a comparison table of all results"""
    print(f"\n{'=' * 80}")
    print(f"📊 COMPREHENSIVE RESULTS COMPARISON")
    print(f"{'=' * 80}")

    # Filter successful results
    successful_results = [r for r in all_results if r.get('success', False)]

    if not successful_results:
        print("❌ No successful evaluations to compare!")
        return

    # Print table header
    print(
        f"{'Configuration':<15} {'RAGAS':<8} {'Precision':<10} {'Recall':<8} {'Faithful':<10} {'Relevancy':<10} {'Time':<8}")
    print(f"{'-' * 80}")

    # Print results for each configuration
    for result in successful_results:
        config_name = result['config_name']
        ragas_results = result['ragas_results']

        print(f"{config_name:<15} "
              f"{ragas_results.ragas_score:<8.3f} "
              f"{ragas_results.context_precision:<10.3f} "
              f"{ragas_results.context_recall:<8.3f} "
              f"{ragas_results.faithfulness:<10.3f} "
              f"{ragas_results.answer_relevancy:<10.3f} "
              f"{ragas_results.eval_time:<8.1f}s")

    # Find best configuration
    best_config = max(successful_results, key=lambda x: x['ragas_results'].ragas_score)
    print(f"\n🏆 BEST CONFIGURATION: {best_config['config_name']}")
    print(f"   RAGAS Score: {best_config['ragas_results'].ragas_score:.3f}")
    print(f"   Key Settings:")
    for param, value in best_config['config_params'].items():
        if param != 'max_eval_samples':
            print(f"     {param}: {value}")


def print_detailed_analysis(all_results: List[Dict]):
    """Print detailed analysis of the results"""
    successful_results = [r for r in all_results if r.get('success', False)]

    if len(successful_results) < 2:
        return

    print(f"\n{'=' * 80}")
    print(f"🔬 DETAILED ANALYSIS")
    print(f"{'=' * 80}")

    # Analyze threshold impact
    threshold_analysis = {}
    for result in successful_results:
        threshold = result['config_params']['similarity_threshold']
        ragas_score = result['ragas_results'].ragas_score
        num_samples = result['ragas_results'].num_samples

        threshold_analysis[threshold] = {
            'ragas_score': ragas_score,
            'config_name': result['config_name'],
            'samples_evaluated': num_samples
        }

    print(f"📈 Similarity Threshold Impact:")
    for threshold in sorted(threshold_analysis.keys()):
        data = threshold_analysis[threshold]
        print(f"   Threshold {threshold}: RAGAS {data['ragas_score']:.3f} ({data['config_name']})")

    # Performance analysis
    times = [r['ragas_results'].eval_time for r in successful_results]
    samples = [r['ragas_results'].num_samples for r in successful_results]

    print(f"\n⚡ Performance Analysis:")
    print(f"   Average evaluation time: {sum(times) / len(times):.1f}s")
    print(f"   Total samples evaluated: {sum(samples)}")
    print(f"   Average time per sample: {sum(times) / sum(samples):.2f}s")

    # Quality insights
    ragas_scores = [r['ragas_results'].ragas_score for r in successful_results]
    precision_scores = [r['ragas_results'].context_precision for r in successful_results]

    print(f"\n🎯 Quality Insights:")
    print(f"   RAGAS score range: {min(ragas_scores):.3f} - {max(ragas_scores):.3f}")
    print(f"   Best precision: {max(precision_scores):.3f}")
    print(
        f"   Score improvement from best to worst: {((max(ragas_scores) - min(ragas_scores)) / min(ragas_scores) * 100):.1f}%")


def main():
    """Main evaluation function"""
    print_header()

    # Validate API key
    if not validate_api_key(OPENAI_API_KEY):
        return

    # Set environment variable
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid tokenizer warnings

    print(f"🚀 Starting comprehensive evaluation...")
    print(f"📝 Testing {len(EVALUATION_CONFIGS)} different configurations")

    # Run all evaluations
    all_results = []
    total_start_time = time.time()

    for i, (config_name, config_params) in enumerate(EVALUATION_CONFIGS.items(), 1):
        print(f"\n⏳ Progress: {i}/{len(EVALUATION_CONFIGS)} configurations")

        result = run_single_evaluation(config_name, config_params)
        all_results.append(result)

        # Brief summary after each evaluation
        if result.get('success'):
            ragas_score = result['ragas_results'].ragas_score
            print(f"✅ {config_name}: RAGAS Score {ragas_score:.3f}")
        else:
            print(f"❌ {config_name}: Failed")

    total_time = time.time() - total_start_time

    # Print comprehensive results
    print_comparison_table(all_results)
    print_detailed_analysis(all_results)

    # Final summary
    successful_count = sum(1 for r in all_results if r.get('success', False))
    print(f"\n{'=' * 80}")
    print(f"🎉 EVALUATION COMPLETE!")
    print(f"{'=' * 80}")
    print(f"✅ Successful evaluations: {successful_count}/{len(EVALUATION_CONFIGS)}")
    print(f"⏱️  Total time: {total_time:.1f}s ({total_time / 60:.1f} minutes)")

    if successful_count > 0:
        best_result = max([r for r in all_results if r.get('success', False)],
                          key=lambda x: x['ragas_results'].ragas_score)
        print(
            f"🏆 Best overall RAGAS score: {best_result['ragas_results'].ragas_score:.3f} ({best_result['config_name']})")

        # Recommendation
        print(f"\n💡 RECOMMENDATION:")
        print(f"   Use the '{best_result['config_name']}' configuration for optimal performance")
        print(f"   Key parameters: similarity_threshold={best_result['config_params']['similarity_threshold']}, "
              f"top_k={best_result['config_params']['top_k_per_sentence']}")
    else:
        print(f"❌ No successful evaluations. Check your API key and internet connection.")


if __name__ == "__main__":
    main()