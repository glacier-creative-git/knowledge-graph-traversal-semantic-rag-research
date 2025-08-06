"""
Example Usage of Enhanced Semantic Graph RAG Research Utils
==========================================================

This script demonstrates how to use the enhanced utils package for semantic graph RAG research
with support for WikiEval, Natural Questions, and future datasets.

Perfect for understanding the workflow before using in Jupyter notebooks.
"""

import os
import sys
import time
from typing import Optional

# Add utils to path if running as standalone script
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import (
    quick_demo,
    quick_visualization,
    full_evaluation_pipeline,
    print_pipeline_summary,
    get_config,
    ResearchPipeline,
    get_available_datasets,
    print_dataset_info
)

def example_1_dataset_overview():
    """Example 1: Overview of available datasets"""
    print("🚀 EXAMPLE 1: Available Datasets Overview")
    print("=" * 50)

    print_dataset_info()

    print("\n" + "=" * 50)

def example_2_quick_demos():
    """Example 2: Run quick demos with different datasets"""
    print("🚀 EXAMPLE 2: Quick Demos with Multiple Datasets")
    print("=" * 50)

    # Try different datasets and demo types
    test_configs = [
        ("wikieval", "demo"),
        ("wikieval", "random"),
        ("natural_questions", "demo"),
        ("natural_questions", "random"),
    ]

    for dataset_name, demo_type in test_configs:
        print(f"\n📊 Running {demo_type} demo on {dataset_name.replace('_', ' ').title()}...")
        try:
            results = quick_demo(demo_type=demo_type, dataset_name=dataset_name)

            print(f"✅ {dataset_name.title()} {demo_type} demo completed!")
            print(f"   Question: {results['question'][:60]}...")
            print(f"   Contexts: {len(results['contexts'])}")
            print(f"   Retrieved: {len(results['retrieved_texts'])} texts")
            print(f"   Traversal steps: {len(results['traversal_steps'])}")
            print(f"   Time: {results['total_time']:.2f}s")

            # Show analysis highlights
            analysis = results['analysis']
            if analysis.get('cross_document_rate', 0) > 0:
                print(f"   Cross-document traversal: {analysis['cross_document_rate']:.1f}%")

        except Exception as e:
            print(f"❌ {dataset_name} {demo_type} demo failed: {str(e)}")

    print("\n" + "=" * 50)

def example_3_with_visualizations():
    """Example 3: Run demos with visualizations on different datasets"""
    print("🎨 EXAMPLE 3: Demos with Visualizations")
    print("=" * 50)

    # Test with different datasets
    datasets_to_test = ["wikieval", "natural_questions"]

    for dataset_name in datasets_to_test:
        print(f"\n📊 Testing visualizations with {dataset_name.replace('_', ' ').title()}...")
        try:
            # Run focused demo with visualizations
            results = quick_visualization(
                demo_type="focused",
                dataset_name=dataset_name,
                show_2d=True,
                show_3d=False  # Skip 3D for faster execution
            )

            print(f"✅ {dataset_name.title()} visualization demo completed!")
            print(f"   Question: {results['demo']['question'][:60]}...")

            analysis = results['demo']['analysis']
            print(f"   Cross-document rate: {analysis['cross_document_rate']:.1f}%")
            print(f"   Contexts discovered: {analysis['num_contexts']}")

            # The visualizations will be displayed automatically
            # In a script, you might want to save them instead

        except Exception as e:
            print(f"❌ {dataset_name} visualization demo failed: {str(e)}")

    print("\n" + "=" * 50)

def example_4_advanced_pipeline():
    """Example 4: Advanced pipeline with dataset-specific configurations"""
    print("🔬 EXAMPLE 4: Advanced Research Pipeline with Dataset-Specific Configs")
    print("=" * 50)

    # Test different dataset-specific configurations
    configs_to_test = [
        ("wikieval", "wikieval"),  # Dataset name, config preset
        ("natural_questions", "natural_questions"),
        ("wikieval", "demo"),  # Use demo config with WikiEval dataset
    ]

    for dataset_name, config_type in configs_to_test:
        print(f"\n📋 Testing {config_type} configuration with {dataset_name.replace('_', ' ').title()}...")

        try:
            # Create dataset-specific configuration
            config = get_config(
                config_type,
                dataset_name=dataset_name,
                top_k_per_sentence=20,  # Override some parameters
                similarity_threshold=0.5,
                max_steps_shown=10
            )

            print(f"📋 Configuration Details:")
            print(f"   Dataset: {config.data.get_dataset_display_name()}")
            print(f"   RAG: top_k={config.rag.top_k_per_sentence}, cross_doc_k={config.rag.cross_doc_k}")
            print(f"   Similarity threshold: {config.rag.similarity_threshold}")
            print(f"   Sliding windows: {config.rag.use_sliding_window}")
            print(f"   Max eval samples: {config.data.max_eval_samples}")

            # Create pipeline
            pipeline = ResearchPipeline(config)

            # Run demo only (skip evaluation and heavy visualizations for example)
            print(f"   Running demo...")
            results = pipeline.run_single_demo(demo_type="focused")

            print(f"✅ Advanced pipeline completed!")
            print(f"   Demo results: {len(results['retrieved_texts'])} texts retrieved")

            # Access individual components
            analysis = results['analysis']
            print(f"   Connection breakdown:")
            for conn_type, percentage in analysis['connection_type_percentages'].items():
                if percentage > 0:
                    print(f"     {conn_type.replace('_', ' ').title()}: {percentage:.1f}%")

            # Show dataset-specific insights
            if analysis.get('reranking_improvement_percent', 0) != 0:
                improvement = analysis['reranking_improvement_percent']
                print(f"   Reranking improvement: {improvement:.1f}%")

        except Exception as e:
            print(f"❌ Advanced pipeline failed for {dataset_name}: {str(e)}")

    print("\n" + "=" * 50)

def example_5_configuration_comparison():
    """Example 5: Compare different configurations on the same dataset"""
    print("⚖️ EXAMPLE 5: Configuration Comparison")
    print("=" * 50)

    dataset_name = "wikieval"  # Use WikiEval for consistent comparison
    configurations = [
        ("Conservative", {"similarity_threshold": 0.3, "top_k_per_sentence": 15}),
        ("Balanced", {"similarity_threshold": 0.5, "top_k_per_sentence": 20}),
        ("Aggressive", {"similarity_threshold": 0.7, "top_k_per_sentence": 25}),
    ]

    print(f"📊 Comparing configurations on {dataset_name.replace('_', ' ').title()}:")

    results_comparison = []

    for config_name, overrides in configurations:
        print(f"\n🔧 Testing {config_name} configuration...")

        try:
            # Create configuration with overrides
            config = get_config("demo", dataset_name=dataset_name, **overrides)
            pipeline = ResearchPipeline(config)

            # Run demo
            start_time = time.time()
            demo_results = pipeline.run_single_demo("random")
            total_time = time.time() - start_time

            # Collect results for comparison
            analysis = demo_results['analysis']
            results_comparison.append({
                'config_name': config_name,
                'total_time': total_time,
                'texts_retrieved': len(demo_results['retrieved_texts']),
                'traversal_steps': len(demo_results['traversal_steps']),
                'cross_document_rate': analysis.get('cross_document_rate', 0),
                'similarity_threshold': overrides['similarity_threshold'],
                'top_k': overrides['top_k_per_sentence']
            })

            print(f"✅ {config_name}: {len(demo_results['retrieved_texts'])} texts, "
                  f"{analysis.get('cross_document_rate', 0):.1f}% cross-doc")

        except Exception as e:
            print(f"❌ {config_name} configuration failed: {str(e)}")

    # Print comparison table
    if results_comparison:
        print(f"\n📊 CONFIGURATION COMPARISON RESULTS:")
        print(f"{'Config':<12} {'Threshold':<10} {'Top-K':<8} {'Retrieved':<10} {'Steps':<8} {'Cross-Doc%':<10} {'Time(s)':<8}")
        print("-" * 75)

        for result in results_comparison:
            print(f"{result['config_name']:<12} "
                  f"{result['similarity_threshold']:<10.1f} "
                  f"{result['top_k']:<8} "
                  f"{result['texts_retrieved']:<10} "
                  f"{result['traversal_steps']:<8} "
                  f"{result['cross_document_rate']:<10.1f} "
                  f"{result['total_time']:<8.2f}")

    print("\n" + "=" * 50)

def example_6_with_evaluation(openai_api_key: Optional[str] = None):
    """Example 6: Full pipeline with RAGAS evaluation"""
    print("📊 EXAMPLE 6: Full Pipeline with RAGAS Evaluation")
    print("=" * 50)

    if not openai_api_key:
        openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        print("⚠️ Skipping evaluation example - no OpenAI API key provided")
        print("   Set OPENAI_API_KEY environment variable or pass as parameter")
        return

    # Test evaluation with different datasets
    datasets_to_evaluate = ["wikieval"]  # Start with smaller dataset

    for dataset_name in datasets_to_evaluate:
        print(f"\n🔍 Running RAGAS evaluation on {dataset_name.replace('_', ' ').title()}...")

        try:
            # Run full evaluation pipeline
            results = full_evaluation_pipeline(
                openai_api_key=openai_api_key,
                dataset_name=dataset_name,
                demo_type="focused",
                max_eval_samples=5  # Small number for quick demo
            )

            print(f"✅ Full evaluation pipeline completed for {dataset_name}!")

            # Access evaluation results
            if 'evaluation' in results:
                eval_results = results['evaluation']
                print(f"   RAGAS Score: {eval_results.ragas_score:.3f}")
                print(f"   Context Precision: {eval_results.context_precision:.3f}")
                print(f"   Context Recall: {eval_results.context_recall:.3f}")
                print(f"   Faithfulness: {eval_results.faithfulness:.3f}")
                print(f"   Answer Relevancy: {eval_results.answer_relevancy:.3f}")
                print(f"   Evaluation time: {eval_results.eval_time:.2f}s")

        except Exception as e:
            print(f"❌ Evaluation pipeline failed for {dataset_name}: {str(e)}")

    print("\n" + "=" * 50)

def main():
    """Run all examples"""
    print("🔬 ENHANCED SEMANTIC GRAPH RAG RESEARCH UTILS - EXAMPLES")
    print("=" * 70)
    print("🎯 Now supporting WikiEval, Natural Questions, and future datasets!")
    print()

    # Run examples
    example_1_dataset_overview()
    example_2_quick_demos()
    example_3_with_visualizations()
    example_4_advanced_pipeline()
    example_5_configuration_comparison()

    # Check for OpenAI API key for evaluation example
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        example_6_with_evaluation(openai_key)
    else:
        print("💡 EXAMPLE 6: Evaluation Example")
        print("=" * 50)
        print("⚠️ Set OPENAI_API_KEY environment variable to run evaluation example")
        print("   export OPENAI_API_KEY='your-openai-api-key'")
        print("   python example_usage.py")
        print("\n" + "=" * 50)

    print("\n✅ ALL EXAMPLES COMPLETED!")
    print("=" * 70)
    print()
    print("💡 Next steps:")
    print("   1. Try these examples in a Jupyter notebook")
    print("   2. Modify configurations in utils/config.py")
    print("   3. Experiment with different datasets:")
    for dataset in get_available_datasets():
        print(f"      • python example_usage.py (will test {dataset})")
    print("   4. Add your custom datasets to the data_loader system")
    print("   5. Customize visualizations in visualizations.py")
    print()
    print("🎯 Key improvements in this enhanced system:")
    print("   • Seamless dataset swapping (WikiEval ⟷ Natural Questions)")
    print("   • Modular data loading architecture")
    print("   • Dataset-specific configurations")
    print("   • Enhanced evaluation capabilities")
    print("   • Future-ready for additional datasets")

if __name__ == "__main__":
    main()