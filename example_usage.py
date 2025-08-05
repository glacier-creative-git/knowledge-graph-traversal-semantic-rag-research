"""
Example Usage of Semantic Graph RAG Research Utils
==================================================

This script demonstrates how to use the utils package for semantic graph RAG research.
Perfect for understanding the workflow before using in Jupyter notebooks.
"""

import os
import sys
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
    ResearchPipeline
)

def example_1_quick_demo():
    """Example 1: Run a quick demo with different data types"""
    print("🚀 EXAMPLE 1: Quick Demo")
    print("=" * 50)

    # Try different demo types
    for demo_type in ["demo", "random", "focused"]:
        print(f"\n📊 Running {demo_type} demo...")
        try:
            results = quick_demo(demo_type)

            print(f"✅ {demo_type.title()} demo completed!")
            print(f"   Question: {results['question'][:60]}...")
            print(f"   Contexts: {len(results['contexts'])}")
            print(f"   Retrieved: {len(results['retrieved_texts'])} texts")
            print(f"   Traversal steps: {len(results['traversal_steps'])}")
            print(f"   Time: {results['total_time']:.2f}s")

        except Exception as e:
            print(f"❌ {demo_type} demo failed: {str(e)}")

    print("\n" + "=" * 50)

def example_2_with_visualizations():
    """Example 2: Run demo with visualizations"""
    print("🎨 EXAMPLE 2: Demo with Visualizations")
    print("=" * 50)

    try:
        # Run focused demo with visualizations
        results = quick_visualization(
            demo_type="focused",
            show_2d=True,
            show_3d=True
        )

        print("✅ Visualization demo completed!")
        print(f"   Question: {results['demo']['question'][:60]}...")
        print(f"   Analysis: {results['demo']['analysis']['cross_document_rate']:.1f}% cross-document")

        # The visualizations will be displayed automatically
        # In a script, you might want to save them instead

    except Exception as e:
        print(f"❌ Visualization demo failed: {str(e)}")

    print("\n" + "=" * 50)

def example_3_advanced_pipeline():
    """Example 3: Advanced pipeline with custom configuration"""
    print("🔬 EXAMPLE 3: Advanced Research Pipeline")
    print("=" * 50)

    try:
        # Create custom configuration - fixed parameter passing
        config = get_config(
            "demo",  # Use demo config for faster execution
            top_k_per_sentence=15,   # These will be applied to rag config
            cross_doc_k=8,
            num_contexts=3,
            max_steps_shown=10       # This will be applied to viz config
        )

        print(f"📋 Configuration:")
        print(f"   RAG: top_k={config.rag.top_k_per_sentence}, cross_doc_k={config.rag.cross_doc_k}")
        print(f"   Similarity threshold: {config.rag.similarity_threshold}")
        print(f"   Sliding windows: {config.rag.use_sliding_window}")
        print(f"   Data: num_contexts={config.data.num_contexts if hasattr(config.data, 'num_contexts') else config.rag.num_contexts}")
        print(f"   Viz: max_steps={config.viz.max_steps_shown}")

        # Create pipeline
        pipeline = ResearchPipeline(config)

        # Run complete pipeline (without evaluation to avoid needing API key)
        results = pipeline.run_complete_pipeline(
            demo_type="focused",
            show_2d=True,
            show_3d=False,  # Skip 3D for faster execution
            show_analysis=True,
            run_evaluation=False  # Skip evaluation in this example
        )

        print("✅ Advanced pipeline completed!")
        print(f"   Demo results: {len(results['demo']['retrieved_texts'])} texts retrieved")

        # Access individual components
        demo_data = results['demo']
        analysis = demo_data['analysis']

        print(f"   Connection breakdown:")
        for conn_type, percentage in analysis['connection_type_percentages'].items():
            print(f"     {conn_type.replace('_', ' ').title()}: {percentage:.1f}%")

        # Show new metrics
        if analysis.get('reranking_improvement_percent', 0) != 0:
            improvement = analysis['reranking_improvement_percent']
            print(f"   Reranking improvement: {improvement:.1f}%")

        if analysis.get('sliding_window_enabled'):
            print(f"   ✅ Using 3-sentence sliding windows")

    except Exception as e:
        print(f"❌ Advanced pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 50)

def example_4_with_evaluation(openai_api_key: Optional[str] = None):
    """Example 4: Full pipeline with RAGAS evaluation"""
    print("📊 EXAMPLE 4: Full Pipeline with RAGAS Evaluation")
    print("=" * 50)

    if not openai_api_key:
        openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        print("⚠️ Skipping evaluation example - no OpenAI API key provided")
        print("   Set OPENAI_API_KEY environment variable or pass as parameter")
        return

    try:
        # Run full evaluation pipeline
        results = full_evaluation_pipeline(
            openai_api_key=openai_api_key,
            demo_type="focused",
            max_eval_samples=5  # Small number for quick demo
        )

        print("✅ Full evaluation pipeline completed!")

        # Access evaluation results
        if 'evaluation' in results:
            eval_results = results['evaluation']
            print(f"   RAGAS Score: {eval_results.ragas_score:.3f}")
            print(f"   Context Precision: {eval_results.context_precision:.3f}")
            print(f"   Context Recall: {eval_results.context_recall:.3f}")
            print(f"   Faithfulness: {eval_results.faithfulness:.3f}")
            print(f"   Answer Relevancy: {eval_results.answer_relevancy:.3f}")

    except Exception as e:
        print(f"❌ Evaluation pipeline failed: {str(e)}")

    print("\n" + "=" * 50)

def main():
    """Run all examples"""
    print("🔬 SEMANTIC GRAPH RAG RESEARCH UTILS - EXAMPLES")
    print("=" * 70)
    print()

    # Show available functions
    print_pipeline_summary()
    print()

    # Run examples
    example_1_quick_demo()
    example_2_with_visualizations()
    example_3_advanced_pipeline()

    # Check for OpenAI API key for evaluation example
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        example_4_with_evaluation(openai_key)
    else:
        print("💡 EXAMPLE 4: Evaluation Example")
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
    print("   3. Experiment with different demo types")
    print("   4. Add your own datasets to data_loader.py")
    print("   5. Customize visualizations in visualizations.py")

if __name__ == "__main__":
    main()