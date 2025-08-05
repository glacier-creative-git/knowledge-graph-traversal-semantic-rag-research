"""
Research Pipeline Orchestration
===============================

High-level pipeline functions that orchestrate the entire research workflow.
Perfect for Jupyter notebook usage.
"""

import os
import time
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from .config import ResearchConfig, get_config
from .data_loader import SQuADDataLoader
from .rag_system import SemanticGraphRAG, TraversalStep
from .visualizations import SemanticGraphVisualizer
from .evaluation import RAGASEvaluator, EvaluationResults, print_evaluation_results

class ResearchPipeline:
    """
    Complete research pipeline orchestration
    """

    def __init__(self, config: ResearchConfig):
        self.config = config
        self.data_loader = SQuADDataLoader(config.data)
        self.rag_system = SemanticGraphRAG(
            top_k_per_sentence=config.rag.top_k_per_sentence,
            cross_doc_k=config.rag.cross_doc_k,
            embedding_model=config.models.embedding_model,
            traversal_depth=config.rag.traversal_depth,
            use_sliding_window=config.rag.use_sliding_window,
            num_contexts=config.rag.num_contexts,
            similarity_threshold=config.rag.similarity_threshold
        )
        self.visualizer = SemanticGraphVisualizer(
            figure_size_2d=config.viz.figure_size_2d,
            figure_size_3d=config.viz.figure_size_3d,
            dpi=config.viz.dpi
        )

        # Results storage
        self.last_question = None
        self.last_contexts = None
        self.last_retrieved_texts = None
        self.last_traversal_steps = None
        self.last_analysis = None

        # Set up output directory
        if config.viz.save_plots:
            os.makedirs(config.viz.output_dir, exist_ok=True)

    def run_single_demo(self, demo_type: str = "random") -> Dict:
        """
        Run a single demonstration of the RAG system

        Args:
            demo_type: "demo", "random", or "focused"

        Returns:
            Dictionary with all results
        """
        print("🚀 Starting Single RAG Demonstration")
        print("=" * 50)

        start_time = time.time()

        # Load data based on demo type
        if demo_type == "demo":
            question, contexts = self.data_loader.create_demo_dataset()
        elif demo_type == "focused":
            if not self.data_loader.load_squad_data():
                print("⚠️ Falling back to demo data")
                question, contexts = self.data_loader.create_demo_dataset()
            else:
                question, contexts = self.data_loader.create_focused_context_set(
                    ['technology', 'computer', 'science', 'research', 'data'],
                    self.config.rag.num_contexts
                )
        else:  # random
            if not self.data_loader.load_squad_data():
                print("⚠️ Falling back to demo data")
                question, contexts = self.data_loader.create_demo_dataset()
            else:
                question, contexts = self.data_loader.select_random_question_with_contexts(
                    self.config.rag.num_contexts
                )

        print(f"\n🎯 Question: {question}")
        print(f"📚 Contexts: {len(contexts)}")

        # Ingest contexts first
        print("\n📚 Ingesting contexts into RAG system...")
        ingest_time = self.rag_system.ingest_contexts(contexts)
        print(f"✅ Contexts ingested in {ingest_time:.2f}s")

        # Run RAG retrieval
        print("\n🔍 Running semantic graph traversal...")
        retrieved_texts, traversal_steps, analysis = self.rag_system.retrieve(
            question, top_k=self.config.rag.retrieval_top_k
        )

        # Store results
        self.last_question = question
        self.last_contexts = contexts
        self.last_retrieved_texts = retrieved_texts
        self.last_traversal_steps = traversal_steps
        self.last_analysis = analysis

        total_time = time.time() - start_time

        print(f"\n✅ Demo completed in {total_time:.2f}s")
        print(f"📊 Retrieved {len(retrieved_texts)} texts from {len(traversal_steps)} traversal steps")

        # Show key analysis metrics
        if analysis.get('reranking_improvement_percent', 0) != 0:
            improvement = analysis['reranking_improvement_percent']
            if improvement > 0:
                print(f"🎯 Reranking improved results by {improvement:.1f}%")
            else:
                print(f"📊 Original traversal order was already optimal")

        if analysis.get('sliding_window_enabled'):
            print(f"🪟 Used 3-sentence forward-looking sliding windows")

        print(f"🔗 Graph: {analysis['graph_parameters']['top_k_per_sentence']} connections/sentence, {analysis['graph_parameters']['cross_doc_k']} cross-doc, threshold: {analysis['graph_parameters']['similarity_threshold']}")

        return {
            'question': question,
            'contexts': contexts,
            'retrieved_texts': retrieved_texts,
            'traversal_steps': traversal_steps,
            'analysis': analysis,
            'total_time': total_time
        }

    def create_2d_visualization(self, save: bool = None) -> plt.Figure:
        """
        Create 2D matplotlib visualization of the last demo

        Args:
            save: Whether to save the plot (overrides config)

        Returns:
            Matplotlib figure
        """
        if not self.last_traversal_steps:
            print("⚠️ No traversal data available. Run a demo first.")
            return plt.Figure()

        print("🎨 Creating 2D visualization...")

        save_path = None
        if save or (save is None and self.config.viz.save_plots):
            save_path = os.path.join(self.config.viz.output_dir, "traversal_2d.png")

        fig = self.visualizer.create_2d_visualization(
            self.rag_system,
            self.last_question,
            self.last_traversal_steps,
            max_steps=self.config.viz.max_steps_shown,
            save_path=save_path
        )

        if save_path:
            print(f"💾 2D visualization saved to {save_path}")

        return fig

    def create_3d_visualization(self, method: str = "pca") -> go.Figure:
        """
        Create 3D plotly visualization of the last demo

        Args:
            method: Dimensionality reduction method ("pca" or "tsne")

        Returns:
            Plotly figure
        """
        if not self.last_traversal_steps:
            print("⚠️ No traversal data available. Run a demo first.")
            return go.Figure()

        print(f"🎨 Creating 3D visualization using {method.upper()}...")

        fig = self.visualizer.create_3d_visualization(
            self.last_question,
            self.last_traversal_steps,
            method=method,
            max_steps=self.config.viz.max_steps_shown
        )

        if self.config.viz.save_plots:
            save_path = os.path.join(self.config.viz.output_dir, f"traversal_3d_{method}.html")
            fig.write_html(save_path)
            print(f"💾 3D visualization saved to {save_path}")

        return fig

    def create_analysis_charts(self) -> go.Figure:
        """
        Create analysis charts for the last demo

        Returns:
            Plotly figure
        """
        if not self.last_analysis:
            print("⚠️ No analysis data available. Run a demo first.")
            return go.Figure()

        print("📊 Creating analysis charts...")

        fig = self.visualizer.create_analysis_charts(self.last_analysis)

        if self.config.viz.save_plots:
            save_path = os.path.join(self.config.viz.output_dir, "analysis_charts.html")
            fig.write_html(save_path)
            print(f"💾 Analysis charts saved to {save_path}")

        return fig

    def run_ragas_evaluation(self) -> EvaluationResults:
        """
        Run RAGAS evaluation on SQuAD dataset

        Returns:
            EvaluationResults object
        """
        print("🔍 Starting RAGAS Evaluation on SQuAD Dataset")
        print("=" * 50)

        # Load evaluation dataset
        if not self.data_loader.load_squad_data():
            print("⚠️ Could not load SQuAD data for evaluation")
            return EvaluationResults(
                context_precision=0.0,
                context_recall=0.0,
                faithfulness=0.0,
                answer_relevancy=0.0,
                ragas_score=0.0,
                ingest_time=0.0,
                eval_time=0.0,
                num_samples=0,
                error="Could not load SQuAD data"
            )

        eval_dataset = self.data_loader.get_evaluation_dataset(self.config.data.max_eval_samples)

        print(f"📊 Evaluation dataset: {eval_dataset['name']}")
        print(f"📄 Documents: {len(eval_dataset['documents'])}")
        print(f"❓ Queries: {len(eval_dataset['queries'])}")

        # Initialize evaluator
        evaluator = RAGASEvaluator(self.config.models, self.config.data.max_eval_samples)

        # Run evaluation
        results = evaluator.benchmark_rag_system(
            self.rag_system,
            eval_dataset,
            "Semantic Graph RAG"
        )

        # Print results
        print_evaluation_results(results, "Semantic Graph RAG")

        return results

    def run_complete_pipeline(self, demo_type: str = "random",
                             show_2d: bool = True,
                             show_3d: bool = True,
                             show_analysis: bool = True,
                             run_evaluation: bool = False) -> Dict:
        """
        Run the complete research pipeline

        Args:
            demo_type: Type of demo to run
            show_2d: Whether to create 2D visualization
            show_3d: Whether to create 3D visualization
            show_analysis: Whether to create analysis charts
            run_evaluation: Whether to run RAGAS evaluation

        Returns:
            Dictionary with all results
        """
        print("🚀 STARTING COMPLETE RESEARCH PIPELINE")
        print("=" * 70)

        results = {}

        # Step 1: Run demo
        print("STEP 1: Running RAG Demo")
        print("-" * 30)
        demo_results = self.run_single_demo(demo_type)
        results['demo'] = demo_results

        # Step 2: Create visualizations
        if show_2d:
            print("\nSTEP 2: Creating 2D Visualization")
            print("-" * 30)
            fig_2d = self.create_2d_visualization()
            results['fig_2d'] = fig_2d

            # Show plot if not saving
            if not self.config.viz.save_plots:
                plt.show()

        if show_3d:
            print("\nSTEP 3: Creating 3D Visualization")
            print("-" * 30)
            fig_3d = self.create_3d_visualization()
            results['fig_3d'] = fig_3d
            fig_3d.show()

        if show_analysis:
            print("\nSTEP 4: Creating Analysis Charts")
            print("-" * 30)
            fig_analysis = self.create_analysis_charts()
            results['fig_analysis'] = fig_analysis
            fig_analysis.show()

        # Step 5: Run evaluation
        if run_evaluation:
            print("\nSTEP 5: Running RAGAS Evaluation")
            print("-" * 30)
            eval_results = self.run_ragas_evaluation()
            results['evaluation'] = eval_results

        print("\n✅ COMPLETE PIPELINE FINISHED")
        print("=" * 70)

        return results

# Convenience functions for easy notebook usage
def quick_demo(demo_type: str = "random",
               openai_api_key: Optional[str] = None,
               config_type: str = "default") -> Dict:
    """
    Quick demo function for notebook usage

    Args:
        demo_type: "demo", "random", or "focused"
        openai_api_key: OpenAI API key
        config_type: "default" or "demo"

    Returns:
        Dictionary with results
    """
    config = get_config(config_type, openai_api_key=openai_api_key)
    pipeline = ResearchPipeline(config)
    return pipeline.run_single_demo(demo_type)

def quick_visualization(demo_type: str = "random",
                       openai_api_key: Optional[str] = None,
                       show_2d: bool = True,
                       show_3d: bool = True) -> Dict:
    """
    Quick visualization function for notebook usage

    Args:
        demo_type: Type of demo to run
        openai_api_key: OpenAI API key
        show_2d: Whether to show 2D visualization
        show_3d: Whether to show 3D visualization

    Returns:
        Dictionary with results and figures
    """
    config = get_config("demo", openai_api_key=openai_api_key)
    pipeline = ResearchPipeline(config)

    # Run demo
    demo_results = pipeline.run_single_demo(demo_type)

    results = {'demo': demo_results}

    # Create visualizations
    if show_2d:
        fig_2d = pipeline.create_2d_visualization()
        results['fig_2d'] = fig_2d
        plt.show()

    if show_3d:
        fig_3d = pipeline.create_3d_visualization()
        results['fig_3d'] = fig_3d
        fig_3d.show()

    return results

def full_evaluation_pipeline(openai_api_key: str,
                            demo_type: str = "focused",
                            max_eval_samples: int = 10) -> Dict:
    """
    Full evaluation pipeline including RAGAS assessment

    Args:
        openai_api_key: OpenAI API key (required for RAGAS)
        demo_type: Type of demo to run
        max_eval_samples: Maximum samples for evaluation

    Returns:
        Dictionary with all results
    """
    config = get_config("demo",
                       openai_api_key=openai_api_key,
                       max_eval_samples=max_eval_samples)

    pipeline = ResearchPipeline(config)

    return pipeline.run_complete_pipeline(
        demo_type=demo_type,
        show_2d=True,
        show_3d=True,
        show_analysis=True,
        run_evaluation=True
    )

def print_pipeline_summary():
    """Print a summary of available pipeline functions"""
    print("🔬 SEMANTIC GRAPH RAG RESEARCH PIPELINE")
    print("=" * 50)
    print()
    print("📚 Available Functions:")
    print()
    print("1. quick_demo(demo_type='random')")
    print("   - Run a single RAG demonstration")
    print("   - demo_type: 'demo', 'random', or 'focused'")
    print()
    print("2. quick_visualization(demo_type='random', show_2d=True, show_3d=True)")
    print("   - Run demo + create visualizations")
    print("   - Automatically shows plots")
    print()
    print("3. full_evaluation_pipeline(openai_api_key, demo_type='focused')")
    print("   - Complete pipeline with RAGAS evaluation")
    print("   - Requires OpenAI API key")
    print()
    print("4. ResearchPipeline(config) - for advanced usage")
    print("   - Full control over configuration")
    print("   - Run individual components")
    print()
    print("💡 Example notebook usage:")
    print("   from utils.pipeline import quick_visualization")
    print("   results = quick_visualization('focused')")