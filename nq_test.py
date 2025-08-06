#!/usr/bin/env python3
"""
Semantic Graph RAG Stress Test - Natural Questions (FIXED VERSION)
================================================================

This script stress tests the semantic graph traversal algorithm by:
1. Loading 100-500 DOCUMENTS from Natural Questions
2. Testing both random and focused approaches
3. Running RAGAS evaluation on results
4. Creating 2D and 3D visualizations
5. Analyzing cross-document traversal patterns

Goal: Traverse 10-20 documents on similar topics, stopping at topic boundaries

FIXES:
- Proper terminology: Documents vs Contexts
- Removed 20-document hard limit
- Fixed TraversalStep attribute access
- Better sentence extraction diagnostics
"""

import os
import sys
import time
import logging
from typing import Dict, List, Tuple, Optional

# Set up paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(name)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

from utils import (
    ResearchPipeline, get_config,
    SemanticGraphRAG, SemanticGraphVisualizer,
    create_data_loader, RAGASEvaluator
)

# ============================================================================
# CONFIGURATION - SET YOUR API KEY HERE
# ============================================================================

OPENAI_API_KEY = "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"


class SemanticGraphStressTester:
    """Stress test the semantic graph RAG algorithm with massive document sets"""

    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        self.results = {}

        # Set up environment
        os.environ["OPENAI_API_KEY"] = openai_api_key
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        print("🚀 SEMANTIC GRAPH RAG STRESS TEST - FIXED VERSION")
        print("=" * 70)
        print("🎯 Target: 250-500 DOCUMENTS, traverse 10-20 documents semantically")
        print("📊 Metrics: RAGAS evaluation + detailed visualizations")
        print("⚡ Algorithm: Advanced semantic graph traversal")
        print("🔧 FIXED: Proper terminology, no hard limits, correct attributes")
        print("=" * 70)

    def create_stress_test_config(self, test_name: str, num_documents: int,
                                  similarity_threshold: float = 0.3) -> 'ResearchConfig':
        """Create configuration optimized for stress testing with many DOCUMENTS"""
        return get_config(
            "natural_questions",
            openai_api_key=self.openai_api_key,

            # MASSIVE DOCUMENT LOADING (fixed terminology)
            num_contexts=num_documents,  # This actually means "documents" in the codebase
            nq_max_samples=num_documents * 3,  # Load 3x to ensure we get enough good ones

            # AGGRESSIVE GRAPH PARAMETERS
            top_k_per_sentence=50,  # Many connections per sentence
            cross_doc_k=30,  # High cross-document connections
            similarity_threshold=similarity_threshold,  # Lower threshold for exploration

            # RETRIEVAL TUNING
            retrieval_top_k=min(100, num_documents // 3),  # Return many results
            traversal_depth=5,  # Deep traversal

            # EVALUATION
            max_eval_samples=1,  # One question for focused testing

            # VISUALIZATION
            max_steps_shown=200,  # Show many traversal steps
            save_plots=True,
            output_dir=f"./stress_test_outputs/{test_name}"
        )

    def load_many_documents(self, data_loader, num_documents: int, method: str = "random",
                            focus_topics: List[str] = None) -> Tuple[str, List[Dict]]:
        """Load many documents using streaming without hard limits"""

        print(f"📚 Loading {num_documents} documents via {method} method...")

        if method == "random":
            # We need to bypass the hard-coded limit in the streaming method
            # For now, let's call it multiple times to accumulate documents
            all_documents = []
            target_per_batch = min(50, num_documents // 5)  # Reasonable batch size
            batches_needed = (num_documents + target_per_batch - 1) // target_per_batch

            print(f"🔧 Loading {num_documents} documents in {batches_needed} batches of ~{target_per_batch}")

            # Load in batches to work around the hard limit
            for batch in range(batches_needed):
                if len(all_documents) >= num_documents:
                    break

                print(f"   📦 Loading batch {batch + 1}/{batches_needed}...")

                try:
                    question, batch_documents = data_loader.select_random_question_with_contexts(target_per_batch)

                    # Add to collection, avoiding duplicates
                    new_documents = []
                    existing_ids = {doc.get('id', '') for doc in all_documents}

                    for doc in batch_documents:
                        if doc.get('id', '') not in existing_ids:
                            new_documents.append(doc)

                    all_documents.extend(new_documents)
                    print(f"      ✅ Added {len(new_documents)} unique documents (total: {len(all_documents)})")

                    # Use the last question (they're all random anyway)
                    final_question = question

                except Exception as batch_error:
                    print(f"      ⚠️ Batch {batch + 1} failed: {batch_error}")
                    continue

            # Trim to exact target
            if len(all_documents) > num_documents:
                all_documents = all_documents[:num_documents]

            print(f"✅ Collected {len(all_documents)} unique documents")
            return final_question, all_documents

        elif method == "focused":
            if not focus_topics:
                focus_topics = ['science', 'technology', 'computer', 'research', 'data']

            question, documents = data_loader.create_focused_context_set(focus_topics, num_documents)
            return question, documents

        else:
            raise ValueError(f"Unknown method: {method}")

    def run_random_stress_test(self, num_documents: int = 250) -> Dict:
        """Test with random Natural Questions sample and many DOCUMENTS"""
        print(f"\n🎲 STRESS TEST 1: Random Sampling Approach ({num_documents} documents)")
        print("-" * 60)

        # Create high-performance configuration
        config = self.create_stress_test_config(
            test_name="random_stress",
            num_documents=num_documents,
            similarity_threshold=0.2  # Very low threshold for exploration
        )

        # Create RAG system optimized for large-scale processing
        rag_system = SemanticGraphRAG(
            top_k_per_sentence=config.rag.top_k_per_sentence,
            cross_doc_k=config.rag.cross_doc_k,
            similarity_threshold=config.rag.similarity_threshold,
            traversal_depth=config.rag.traversal_depth,
            use_sliding_window=True,
            num_contexts=num_documents  # This is actually num_documents in the codebase
        )

        # Create data loader
        data_loader = create_data_loader("natural_questions", config.data)

        start_time = time.time()

        # Load dataset
        if not data_loader.load_dataset():
            raise RuntimeError("Failed to load Natural Questions dataset")

        # Load many documents using our fixed method
        question, documents = self.load_many_documents(
            data_loader, num_documents, method="random"
        )

        load_time = time.time() - start_time

        print(f"✅ Loaded {len(documents)} documents in {load_time:.2f}s")
        print(f"🎯 Question: {question}")
        print(f"📄 Document sizes: {[len(doc['context']) for doc in documents[:5]]}... chars")

        # Ingest documents (terminology fixed!)
        print(f"\n📊 Ingesting {len(documents)} documents into semantic graph...")
        ingest_start = time.time()
        ingest_time = rag_system.ingest_contexts(documents)  # Still uses 'contexts' in method name
        total_ingest_time = time.time() - ingest_start

        # Count sentences extracted
        total_sentences = len(rag_system.sentences_info) if hasattr(rag_system, 'sentences_info') else 0
        print(f"✅ Semantic graph built in {total_ingest_time:.2f}s")
        print(f"📊 Extracted {total_sentences} sentences from {len(documents)} documents")
        print(f"📊 Average: {total_sentences / len(documents):.1f} sentences per document")

        # Run retrieval with detailed analysis
        print(f"\n🔍 Running semantic graph traversal...")
        retrieval_start = time.time()

        retrieved_texts, traversal_steps, analysis = rag_system.retrieve(
            question,
            top_k=config.rag.retrieval_top_k
        )

        retrieval_time = time.time() - retrieval_start

        # FIXED: Use correct attribute access
        num_documents_traversed = len(set(step.sentence_info.doc_id for step in traversal_steps))
        cross_doc_steps = sum(1 for step in traversal_steps if step.connection_type == 'cross_document')
        cross_doc_rate = (cross_doc_steps / len(traversal_steps)) * 100 if traversal_steps else 0

        print(f"\n📈 RANDOM STRESS TEST RESULTS:")
        print(f"   Documents loaded: {len(documents)}")
        print(f"   Sentences extracted: {total_sentences}")
        print(f"   Documents traversed: {num_documents_traversed}")
        print(f"   Total traversal steps: {len(traversal_steps)}")
        print(f"   Cross-document traversals: {cross_doc_steps} ({cross_doc_rate:.1f}%)")
        print(f"   Final results retrieved: {len(retrieved_texts)}")
        print(f"   Retrieval time: {retrieval_time:.2f}s")
        print(f"   Average similarity: {analysis.get('average_similarity', 0):.3f}")

        # Store results
        results = {
            'test_type': 'random',
            'question': question,
            'documents': documents,  # Fixed terminology
            'retrieved_texts': retrieved_texts,
            'traversal_steps': traversal_steps,
            'analysis': analysis,
            'num_documents_loaded': len(documents),
            'num_sentences_extracted': total_sentences,
            'num_documents_traversed': num_documents_traversed,
            'cross_doc_rate': cross_doc_rate,
            'total_time': load_time + total_ingest_time + retrieval_time,
            'config': config
        }

        self.results['random'] = results
        return results

    def run_focused_stress_test(self, num_documents: int = 300) -> Dict:
        """Test with focused topic search and many DOCUMENTS"""
        print(f"\n🔍 STRESS TEST 2: Focused Topic Approach ({num_documents} documents)")
        print("-" * 60)

        # Create configuration optimized for focused search
        config = self.create_stress_test_config(
            test_name="focused_stress",
            num_documents=num_documents,
            similarity_threshold=0.25  # Slightly higher for quality
        )

        # Create RAG system
        rag_system = SemanticGraphRAG(
            top_k_per_sentence=config.rag.top_k_per_sentence,
            cross_doc_k=config.rag.cross_doc_k,
            similarity_threshold=config.rag.similarity_threshold,
            traversal_depth=config.rag.traversal_depth,
            use_sliding_window=True,
            num_contexts=num_documents
        )

        # Create data loader
        data_loader = create_data_loader("natural_questions", config.data)

        # Use focused topic keywords (science/technology focused)
        focus_topics = ['science', 'technology', 'computer', 'research', 'data', 'system',
                        'algorithm', 'software', 'internet', 'artificial', 'intelligence']

        start_time = time.time()

        # Load dataset
        if not data_loader.load_dataset():
            raise RuntimeError("Failed to load Natural Questions dataset")

        # Load many documents using focused method
        question, documents = self.load_many_documents(
            data_loader, num_documents, method="focused", focus_topics=focus_topics
        )

        load_time = time.time() - start_time

        print(f"✅ Loaded {len(documents)} focused documents in {load_time:.2f}s")
        print(f"🎯 Question: {question}")

        # Show topic distribution
        topic_matches = {}
        for doc in documents[:10]:  # Sample first 10
            matches = [topic for topic in focus_topics
                       if topic.lower() in doc['context'].lower()]
            for match in matches:
                topic_matches[match] = topic_matches.get(match, 0) + 1

        print(f"📊 Topic distribution (sample): {topic_matches}")

        # Ingest documents
        print(f"\n📊 Ingesting {len(documents)} documents into semantic graph...")
        ingest_start = time.time()
        ingest_time = rag_system.ingest_contexts(documents)
        total_ingest_time = time.time() - ingest_start

        # Count sentences extracted
        total_sentences = len(rag_system.sentences_info) if hasattr(rag_system, 'sentences_info') else 0
        print(f"✅ Semantic graph built in {total_ingest_time:.2f}s")
        print(f"📊 Extracted {total_sentences} sentences from {len(documents)} documents")
        print(f"📊 Average: {total_sentences / len(documents):.1f} sentences per document")

        # Run retrieval
        print(f"\n🔍 Running focused semantic graph traversal...")
        retrieval_start = time.time()

        retrieved_texts, traversal_steps, analysis = rag_system.retrieve(
            question,
            top_k=config.rag.retrieval_top_k
        )

        retrieval_time = time.time() - retrieval_start

        # FIXED: Use correct attribute access
        num_documents_traversed = len(set(step.sentence_info.doc_id for step in traversal_steps))
        cross_doc_steps = sum(1 for step in traversal_steps if step.connection_type == 'cross_document')
        cross_doc_rate = (cross_doc_steps / len(traversal_steps)) * 100 if traversal_steps else 0

        print(f"\n📈 FOCUSED STRESS TEST RESULTS:")
        print(f"   Documents loaded: {len(documents)}")
        print(f"   Sentences extracted: {total_sentences}")
        print(f"   Documents traversed: {num_documents_traversed}")
        print(f"   Total traversal steps: {len(traversal_steps)}")
        print(f"   Cross-document traversals: {cross_doc_steps} ({cross_doc_rate:.1f}%)")
        print(f"   Final results retrieved: {len(retrieved_texts)}")
        print(f"   Retrieval time: {retrieval_time:.2f}s")
        print(f"   Average similarity: {analysis.get('average_similarity', 0):.3f}")

        # Store results
        results = {
            'test_type': 'focused',
            'question': question,
            'documents': documents,  # Fixed terminology
            'retrieved_texts': retrieved_texts,
            'traversal_steps': traversal_steps,
            'analysis': analysis,
            'num_documents_loaded': len(documents),
            'num_sentences_extracted': total_sentences,
            'num_documents_traversed': num_documents_traversed,
            'cross_doc_rate': cross_doc_rate,
            'total_time': load_time + total_ingest_time + retrieval_time,
            'config': config,
            'focus_topics': focus_topics,
            'topic_matches': topic_matches
        }

        self.results['focused'] = results
        return results

    def run_ragas_evaluation(self, test_results: Dict) -> Dict:
        """Run RAGAS evaluation on stress test results"""
        print(f"\n📊 RUNNING RAGAS EVALUATION ({test_results['test_type'].upper()})")
        print("-" * 40)

        try:
            # Create evaluator
            evaluator = RAGASEvaluator(
                openai_api_key=self.openai_api_key,
                max_samples=1  # Just evaluate the one question
            )

            # Prepare evaluation data
            question = test_results['question']
            documents = test_results['documents']  # Fixed terminology
            retrieved_texts = test_results['retrieved_texts']

            print(f"🎯 Evaluating question: {question[:80]}...")
            print(f"📚 Using {len(documents)} documents")
            print(f"📄 Retrieved {len(retrieved_texts)} relevant texts")

            # Create simple RAG system for evaluation
            rag_system = SemanticGraphRAG(
                similarity_threshold=0.3,
                num_contexts=len(documents)
            )

            # Run evaluation
            eval_start = time.time()
            eval_results = evaluator.evaluate_rag_system(
                rag_system=rag_system,
                documents=documents,
                queries=[question],
                ground_truths=[f"Answer based on {len(retrieved_texts)} semantic connections"],
                system_name=f"Stress Test ({test_results['test_type'].title()})"
            )
            eval_time = time.time() - eval_start

            print(f"\n🎯 RAGAS EVALUATION RESULTS:")
            print(f"   Overall Score: {eval_results.ragas_score:.3f}")
            print(f"   Context Precision: {eval_results.context_precision:.3f}")
            print(f"   Context Recall: {eval_results.context_recall:.3f}")
            print(f"   Faithfulness: {eval_results.faithfulness:.3f}")
            print(f"   Answer Relevancy: {eval_results.answer_relevancy:.3f}")
            print(f"   Evaluation time: {eval_time:.2f}s")

            return {
                'ragas_results': eval_results,
                'eval_time': eval_time
            }

        except Exception as e:
            print(f"⚠️ RAGAS evaluation failed: {e}")
            return {'ragas_results': None, 'error': str(e)}

    def create_visualizations(self, test_results: Dict) -> Dict:
        """Create 2D and 3D visualizations of the semantic graph traversal"""
        print(f"\n🎨 CREATING VISUALIZATIONS ({test_results['test_type'].upper()})")
        print("-" * 40)

        try:
            # Create visualizer
            visualizer = SemanticGraphVisualizer(
                figure_size_2d=(25, 15),  # Large figures for many documents
                figure_size_3d=(18, 15),
                dpi=150
            )

            question = test_results['question']
            documents = test_results['documents']  # Fixed terminology
            traversal_steps = test_results['traversal_steps']
            test_type = test_results['test_type']

            print(f"📊 Visualizing {len(traversal_steps)} traversal steps")
            print(f"📄 Across {test_results['num_documents_traversed']} documents")
            print(f"📚 From {len(documents)} total documents loaded")

            # Create output directory
            output_dir = f"./stress_test_outputs/{test_type}"
            os.makedirs(output_dir, exist_ok=True)

            # Create 2D visualization
            print("🎯 Creating 2D semantic graph visualization...")
            try:
                fig_2d = visualizer.create_2d_traversal_plot(
                    question=question,
                    contexts=documents,  # Note: method still uses 'contexts' parameter name
                    traversal_steps=traversal_steps,
                    title=f"Semantic Graph Traversal - {test_type.title()} ({len(documents)} documents)"
                )

                # Save 2D plot
                fig_2d.savefig(f"{output_dir}/semantic_graph_2d.png",
                               dpi=150, bbox_inches='tight')
                print(f"   ✅ 2D plot saved to {output_dir}/semantic_graph_2d.png")
            except Exception as viz_2d_error:
                print(f"   ⚠️ 2D visualization failed: {viz_2d_error}")

            # Create 3D visualization
            print("🎯 Creating 3D semantic graph visualization...")
            try:
                fig_3d = visualizer.create_3d_traversal_plot(
                    question=question,
                    contexts=documents,  # Note: method still uses 'contexts' parameter name
                    traversal_steps=traversal_steps,
                    title=f"3D Semantic Traversal - {test_type.title()}"
                )

                # Save 3D plot
                fig_3d.write_html(f"{output_dir}/semantic_graph_3d.html")
                print(f"   ✅ 3D plot saved to {output_dir}/semantic_graph_3d.html")
            except Exception as viz_3d_error:
                print(f"   ⚠️ 3D visualization failed: {viz_3d_error}")

            # Create analysis charts
            print("📈 Creating traversal analysis charts...")
            try:
                analysis_figs = visualizer.create_analysis_charts(
                    traversal_steps=traversal_steps,
                    analysis=test_results['analysis']
                )

                # Save analysis charts
                for i, fig in enumerate(analysis_figs):
                    fig.savefig(f"{output_dir}/analysis_chart_{i + 1}.png",
                                dpi=150, bbox_inches='tight')
                print(f"   ✅ Analysis charts saved to {output_dir}/analysis_chart_*.png")
            except Exception as analysis_error:
                print(f"   ⚠️ Analysis charts failed: {analysis_error}")

            print(f"🎨 Visualizations saved to: {output_dir}/")

            return {
                'output_dir': output_dir,
                'figures_created': 3  # 2D + 3D + analysis
            }

        except Exception as e:
            print(f"⚠️ Visualization creation failed: {e}")
            return {'error': str(e)}

    def print_final_summary(self):
        """Print comprehensive summary of both stress tests"""
        print(f"\n" + "🏆" * 70)
        print("🏆" + " SEMANTIC GRAPH RAG STRESS TEST SUMMARY ".center(68) + "🏆")
        print("🏆" * 70)

        if 'random' in self.results and 'focused' in self.results:
            random_results = self.results['random']
            focused_results = self.results['focused']

            print(f"\n📊 PERFORMANCE COMPARISON:")
            print(f"{'Metric':<30} {'Random':<15} {'Focused':<15}")
            print("-" * 60)
            print(
                f"{'Documents loaded':<30} {random_results['num_documents_loaded']:<15} {focused_results['num_documents_loaded']:<15}")
            print(
                f"{'Sentences extracted':<30} {random_results['num_sentences_extracted']:<15} {focused_results['num_sentences_extracted']:<15}")
            print(
                f"{'Documents traversed':<30} {random_results['num_documents_traversed']:<15} {focused_results['num_documents_traversed']:<15}")
            print(
                f"{'Traversal steps':<30} {len(random_results['traversal_steps']):<15} {len(focused_results['traversal_steps']):<15}")
            print(
                f"{'Cross-doc rate %':<30} {random_results['cross_doc_rate']:<15.1f} {focused_results['cross_doc_rate']:<15.1f}")
            print(
                f"{'Total time (s)':<30} {random_results['total_time']:<15.2f} {focused_results['total_time']:<15.2f}")

            # Success metrics (≥10 documents traversed goal)
            random_success = random_results['num_documents_traversed'] >= 10
            focused_success = focused_results['num_documents_traversed'] >= 10

            print(f"\n🎯 SUCCESS CRITERIA (≥10 documents traversed):")
            print(f"   Random approach: {'✅ PASSED' if random_success else '❌ FAILED'}")
            print(f"   Focused approach: {'✅ PASSED' if focused_success else '❌ FAILED'}")

            # Sentence extraction efficiency
            random_sent_per_doc = random_results['num_sentences_extracted'] / random_results['num_documents_loaded']
            focused_sent_per_doc = focused_results['num_sentences_extracted'] / focused_results['num_documents_loaded']

            print(f"\n📊 SENTENCE EXTRACTION EFFICIENCY:")
            print(f"   Random: {random_sent_per_doc:.1f} sentences/document")
            print(f"   Focused: {focused_sent_per_doc:.1f} sentences/document")
            print(f"   Expected: 20-50 sentences/document for 10k+ char Wikipedia articles")

            if random_sent_per_doc < 5 or focused_sent_per_doc < 5:
                print(f"   ⚠️ LOW EXTRACTION RATE - May indicate processing issues")

        print(f"\n📁 Results and visualizations saved in:")
        print(f"   ./stress_test_outputs/random_stress/")
        print(f"   ./stress_test_outputs/focused_stress/")


def main():
    """Run the stress test with fixed document loading"""
    if not OPENAI_API_KEY or OPENAI_API_KEY == "your-api-key-here":
        print("❌ Please set your OpenAI API key in the script!")
        return

    # Create stress tester
    tester = SemanticGraphStressTester(OPENAI_API_KEY)

    try:
        # Run stress tests with reasonable document counts for testing
        print("🚀 Starting stress tests...")

        # Test 1: Random sampling (start with 100 documents)
        random_results = tester.run_random_stress_test(num_documents=100)

        # Test 2: Focused sampling (start with 150 documents)
        focused_results = tester.run_focused_stress_test(num_documents=150)

        # Run RAGAS evaluations
        print("\n📊 Running RAGAS evaluations...")
        random_eval = tester.run_ragas_evaluation(random_results)
        focused_eval = tester.run_ragas_evaluation(focused_results)

        # Add evaluation results to main results
        tester.results['random']['evaluation'] = random_eval
        tester.results['focused']['evaluation'] = focused_eval

        # Create visualizations
        print("\n🎨 Creating visualizations...")
        random_viz = tester.create_visualizations(random_results)
        focused_viz = tester.create_visualizations(focused_results)

        # Final summary
        tester.print_final_summary()

    except KeyboardInterrupt:
        print("\n👋 Stress test interrupted by user")
    except Exception as e:
        print(f"\n❌ Stress test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()