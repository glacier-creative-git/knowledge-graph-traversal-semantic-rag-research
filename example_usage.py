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
import logging
from typing import Optional
import traceback

# Add utils to path if running as standalone script
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# CRITICAL: Set up logging BEFORE importing utils
# This ensures we see all the detailed logs from data_loader.py
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Set the specific logger for data_loader to be more verbose
data_loader_logger = logging.getLogger('utils.data_loader')
data_loader_logger.setLevel(logging.INFO)

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
    """Example 2: Run quick demos with different datasets - WITH PROPER LOGGING"""
    print("🚀 EXAMPLE 2: Quick Demos with Multiple Datasets")
    print("=" * 50)

    # Enable verbose logging for debugging
    print("🔧 DEBUGGING: Setting up verbose logging for Natural Questions...")

    # Try different datasets and demo types
    test_configs = [
        ("wikieval", "demo"),
        ("wikieval", "random"),
        ("natural_questions", "demo"),
        ("natural_questions", "random"),  # This is the problematic one
    ]

    for dataset_name, demo_type in test_configs:
        print(f"\n📊 Running {demo_type} demo on {dataset_name.replace('_', ' ').title()}...")
        print(f"🔍 DEBUG: About to call quick_demo(demo_type='{demo_type}', dataset_name='{dataset_name}')")

        try:
            # Add some debug info about the config being used
            if dataset_name == "natural_questions" and demo_type == "random":
                print("🔧 DEBUG: This is the problematic combination - let's see detailed logs...")

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
            # Print full traceback for debugging
            import traceback
            print(f"🔧 DEBUG: Full traceback:")
            traceback.print_exc()

    print("\n" + "=" * 50)

def example_2b_debug_natural_questions():
    """Example 2b: Debug Natural Questions specifically"""
    print("🔬 EXAMPLE 2B: Debug Natural Questions Loading")
    print("=" * 50)

    try:
        print("🔧 Creating Natural Questions config...")
        config = get_config("natural_questions")
        print(f"   Dataset name: {config.data.dataset_name}")
        print(f"   NQ split: {config.data.nq_split}")
        print(f"   NQ max samples: {config.data.nq_max_samples}")
        print(f"   NQ streaming: {config.data.nq_streaming}")

        print("\n🔧 Creating pipeline...")
        pipeline = ResearchPipeline(config)

        print(f"🔧 Data loader type: {type(pipeline.data_loader).__name__}")

        print("\n🔧 Testing data loader methods directly...")

        # Test load_dataset
        print("🔧 Testing load_dataset()...")
        load_result = pipeline.data_loader.load_dataset()
        print(f"   Load result: {load_result}")

        if load_result:
            print("🔧 Testing select_random_question_with_contexts()...")
            try:
                question, contexts = pipeline.data_loader.select_random_question_with_contexts(2)
                print(f"   Success! Got question: {question[:100]}...")
                print(f"   Got {len(contexts)} contexts")
                for i, ctx in enumerate(contexts):
                    print(f"     Context {i+1}: {len(ctx['context'])} chars")
            except Exception as method_error:
                print(f"   Method failed: {method_error}")
                traceback.print_exc()
        else:
            print("   Load failed, so can't test streaming methods")

    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 50)

# Rest of the functions remain the same...
def example_3_visualizations():
    """Example 3: Create visualizations with different datasets"""
    print("🚀 EXAMPLE 3: Visualizations with Multiple Datasets")
    print("=" * 50)

    visualization_configs = [
        ("wikieval", "focused"),
        ("natural_questions", "demo"),  # Use demo for Natural Questions to avoid issues
    ]

    for dataset_name, demo_type in visualization_configs:
        print(f"\n📊 Running {demo_type} visualization on {dataset_name.replace('_', ' ').title()}...")
        try:
            results = quick_visualization(
                demo_type=demo_type,
                dataset_name=dataset_name,
                show_2d=True,
                show_3d=True
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

if __name__ == "__main__":
    print("🔬 ENHANCED SEMANTIC GRAPH RAG RESEARCH UTILS - EXAMPLES")
    print("=" * 70)
    print("🎯 Now supporting WikiEval, Natural Questions, and future datasets!")

    try:
        example_1_dataset_overview()
        example_2_quick_demos()
        example_2b_debug_natural_questions()  # NEW: Debug Natural Questions specifically
        # example_3_visualizations()  # Comment out for now to focus on the main issue
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()