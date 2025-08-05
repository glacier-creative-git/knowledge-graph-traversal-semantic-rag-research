#!/usr/bin/env python3
"""
Corpus-Wide RAGAS Evaluation Script
==================================

This script evaluates the corpus-wide semantic graph RAG system that searches
across the entire SQuAD corpus rather than pre-selected documents.

This should achieve much higher RAGAS scores (target: 0.85+) because:
1. Searches across 1000+ unique contexts instead of 3-5
2. Proper deduplication of SQuAD contexts
3. Algorithm automatically discovers optimal number of documents
4. True corpus-wide semantic graph traversal

Usage:
    python corpus_ragas_evaluation.py
"""

import os
import sys
import time
from typing import Dict, List

# Add utils to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.corpus_wide_rag import CorpusWideSemanticRAG
from utils.config import get_config
from utils.data_loader import SQuADDataLoader
from utils.evaluation import RAGASEvaluator, print_evaluation_results
from datasets import load_dataset

# ============================================================================
# CONFIGURATION - SET YOUR API KEY HERE
# ============================================================================

OPENAI_API_KEY = "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"  # 🔑 PUT YOUR API KEY HERE

# Test configurations for corpus-wide search
CORPUS_CONFIGS = {
    "Corpus_Conservative": {
        "similarity_threshold": 0.4,
        "top_k_per_sentence": 25,
        "cross_doc_k": 15,
        "retrieval_top_k": 8,
        "max_corpus_size": 800
    },
    "Corpus_Balanced": {
        "similarity_threshold": 0.5,
        "top_k_per_sentence": 30,
        "cross_doc_k": 20,
        "retrieval_top_k": 10,
        "max_corpus_size": 1000
    },
    "Corpus_Aggressive": {
        "similarity_threshold": 0.6,
        "top_k_per_sentence": 35,
        "cross_doc_k": 25,
        "retrieval_top_k": 12,
        "max_corpus_size": 1000
    }
}


class CorpusWideEvaluator:
    """Evaluator for corpus-wide semantic graph RAG"""

    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        os.environ["OPENAI_API_KEY"] = openai_api_key
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def create_corpus_evaluation_dataset(self, squad_data, num_eval_questions: int = 30) -> Dict:
        """
        Create evaluation dataset from SQuAD data for corpus-wide evaluation

        Args:
            squad_data: SQuAD dataset
            num_eval_questions: Number of questions to evaluate

        Returns:
            Dictionary with evaluation data
        """
        print(f"📊 Creating corpus evaluation dataset...")

        # Select diverse questions for evaluation
        eval_questions = []
        eval_ground_truths = []
        seen_contexts = set()

        for item in squad_data:
            if len(eval_questions) >= num_eval_questions:
                break

            # Skip if we've seen this context before (for diversity)
            context_hash = hash(item['context'].strip())
            if context_hash in seen_contexts:
                continue
            seen_contexts.add(context_hash)

            eval_questions.append(item['question'])

            # Ground truth answer
            if item['answers']['text']:
                eval_ground_truths.append(item['answers']['text'][0])
            else:
                eval_ground_truths.append("No answer available")

        return {
            'queries': eval_questions,
            'ground_truths': eval_ground_truths,
            'name': f'Corpus-Wide SQuAD Evaluation ({len(eval_questions)} questions)'
        }

    def evaluate_corpus_rag(self, config_name: str, config_params: Dict, squad_data) -> Dict:
        """Evaluate corpus-wide RAG system"""
        print(f"\n{'=' * 70}")
        print(f"🌍 CORPUS-WIDE EVALUATION: {config_name.upper()}")
        print(f"{'=' * 70}")

        print(f"📋 Configuration:")
        for param, value in config_params.items():
            print(f"   {param}: {value}")
        print()

        try:
            # Create corpus-wide RAG system
            rag_system = CorpusWideSemanticRAG(
                top_k_per_sentence=config_params['top_k_per_sentence'],
                cross_doc_k=config_params['cross_doc_k'],
                similarity_threshold=config_params['similarity_threshold'],
                max_corpus_size=config_params['max_corpus_size']
            )

            # Ingest entire corpus
            print(f"🌍 Ingesting SQuAD corpus...")
            ingest_time = rag_system.ingest_corpus(squad_data, config_params['max_corpus_size'])

            # Create evaluation dataset (questions only - corpus is already ingested)
            eval_data = self.create_corpus_evaluation_dataset(squad_data, num_eval_questions=25)

            print(f"🎯 Running RAGAS evaluation on {len(eval_data['queries'])} questions...")

            # Create RAGAS evaluator
            from utils.config import ModelConfig
            model_config = ModelConfig(openai_api_key=self.openai_api_key)
            evaluator = RAGASEvaluator(model_config, max_samples=len(eval_data['queries']))

            # Create evaluation dataset for RAGAS
            ragas_dataset = []

            for i, query in enumerate(eval_data['queries']):
                print(f"   Processing query {i + 1}/{len(eval_data['queries'])}: {query[:60]}...")

                try:
                    # Retrieve from corpus
                    retrieved_texts, _, analysis = rag_system.retrieve_from_corpus(
                        query, top_k=config_params['retrieval_top_k']
                    )

                    # Create sample for RAGAS
                    sample = {
                        "user_input": query,
                        "retrieved_contexts": retrieved_texts,
                        "response": f"Based on the corpus search: {retrieved_texts[0][:200]}..." if retrieved_texts else "No relevant information found."
                    }

                    if i < len(eval_data['ground_truths']):
                        sample["reference"] = eval_data['ground_truths'][i]

                    ragas_dataset.append(sample)

                    # Show discovery stats
                    contexts_found = analysis.get('unique_contexts_discovered', 0)
                    coverage = analysis.get('context_coverage_rate', 0)
                    print(
                        f"     → Found {len(retrieved_texts)} results from {contexts_found} contexts ({coverage:.1f}% corpus coverage)")

                except Exception as e:
                    print(f"     ❌ Error processing query: {str(e)}")
                    continue

            if not ragas_dataset:
                return {"error": "No valid samples created"}

            # Run RAGAS evaluation
            print(f"\n⚡ Running RAGAS metrics on {len(ragas_dataset)} samples...")

            from ragas import evaluate, EvaluationDataset
            from ragas.metrics import context_precision, context_recall, faithfulness, answer_relevancy
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=self.openai_api_key)
            evaluation_dataset = EvaluationDataset.from_list(ragas_dataset)

            eval_start = time.time()
            result = evaluate(
                dataset=evaluation_dataset,
                metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
                llm=llm,
                raise_exceptions=False
            )
            eval_time = time.time() - eval_start

            # Extract scores
            scores = self._extract_ragas_scores(result)

            # Add metadata
            scores.update({
                'config_name': config_name,
                'config_params': config_params,
                'ingest_time': ingest_time,
                'eval_time': eval_time,
                'num_samples': len(ragas_dataset),
                'corpus_stats': {
                    'total_contexts': len(rag_system.corpus_contexts),
                    'total_sentences': len(rag_system.sentences_info),
                    'duplicates_removed': rag_system.duplicate_count,
                    'deduplication_rate': rag_system.duplicate_count / (
                                len(rag_system.corpus_contexts) + rag_system.duplicate_count) * 100
                }
            })

            return scores

        except Exception as e:
            print(f"❌ Evaluation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

    def _extract_ragas_scores(self, result) -> Dict:
        """Extract RAGAS scores from result"""
        import numpy as np

        scores = {}
        metrics = ['context_precision', 'context_recall', 'faithfulness', 'answer_relevancy']

        # Handle different result formats
        if hasattr(result, 'scores') and result.scores:
            # Average across samples
            metric_totals = {metric: [] for metric in metrics}

            for sample_scores in result.scores:
                for metric in metrics:
                    if metric in sample_scores:
                        val = sample_scores[metric]
                        if hasattr(val, 'item'):
                            val = float(val.item())
                        else:
                            val = float(val)
                        if not np.isnan(val):
                            metric_totals[metric].append(val)

            for metric in metrics:
                if metric_totals[metric]:
                    scores[metric] = np.mean(metric_totals[metric])
                else:
                    scores[metric] = 0.0
        else:
            # Direct attribute access
            for metric in metrics:
                try:
                    if hasattr(result, metric):
                        scores[metric] = float(getattr(result, metric))
                    else:
                        scores[metric] = 0.0
                except:
                    scores[metric] = 0.0

        # Calculate overall RAGAS score
        valid_scores = [s for s in scores.values() if s > 0]
        scores['ragas_score'] = np.mean(list(scores.values())) if valid_scores else 0.0

        return scores


def main():
    """Main evaluation function"""
    print("🔥" * 80)
    print("🔥" + " " * 76 + "🔥")
    print("🔥" + "          CORPUS-WIDE SEMANTIC GRAPH RAG EVALUATION          ".center(76) + "🔥")
    print("🔥" + " " * 76 + "🔥")
    print("🔥" * 80)
    print()
    print("🌍 This evaluation tests semantic graph traversal across ENTIRE SQuAD corpus")
    print("🎯 Target: RAGAS scores 0.85+ by leveraging full corpus discovery")
    print("✨ Key improvements:")
    print("   • Searches 1000+ unique contexts instead of 3-5")
    print("   • Automatic document quantity selection by traversal algorithm")
    print("   • Proper deduplication of SQuAD contexts")
    print("   • True corpus-wide semantic graph connections")
    print()

    # Validate API key
    if not OPENAI_API_KEY or OPENAI_API_KEY == "your-openai-api-key-here":
        print("❌ ERROR: Please set your OpenAI API key in the script!")
        return

    # Load SQuAD dataset
    print("📚 Loading SQuAD 2.0 dataset...")
    try:
        squad_data = load_dataset("squad_v2", split="validation[:1500]")  # More data for corpus
        print(f"✅ Loaded {len(squad_data)} SQuAD samples for corpus")
    except Exception as e:
        print(f"❌ Failed to load SQuAD: {e}")
        return

    # Run evaluations
    evaluator = CorpusWideEvaluator(OPENAI_API_KEY)
    all_results = []

    total_start = time.time()

    for i, (config_name, config_params) in enumerate(CORPUS_CONFIGS.items(), 1):
        print(f"\n⏳ Progress: {i}/{len(CORPUS_CONFIGS)} configurations")

        result = evaluator.evaluate_corpus_rag(config_name, config_params, squad_data)
        all_results.append(result)

        if 'error' not in result:
            print(f"✅ {config_name}: RAGAS {result['ragas_score']:.3f}")
            print(f"   Corpus: {result['corpus_stats']['total_contexts']} contexts, "
                  f"{result['corpus_stats']['duplicates_removed']} duplicates removed")
        else:
            print(f"❌ {config_name}: {result['error']}")

    total_time = time.time() - total_start

    # Results comparison
    successful_results = [r for r in all_results if 'error' not in r]

    if successful_results:
        print(f"\n{'=' * 80}")
        print(f"📊 CORPUS-WIDE RESULTS COMPARISON")
        print(f"{'=' * 80}")

        print(f"{'Config':<20} {'RAGAS':<8} {'Precision':<10} {'Recall':<8} {'Faithful':<10} {'Relevancy':<10}")
        print(f"{'-' * 80}")

        for result in successful_results:
            print(f"{result['config_name']:<20} "
                  f"{result['ragas_score']:<8.3f} "
                  f"{result['context_precision']:<10.3f} "
                  f"{result['context_recall']:<8.3f} "
                  f"{result['faithfulness']:<10.3f} "
                  f"{result['answer_relevancy']:<10.3f}")

        # Best result
        best = max(successful_results, key=lambda x: x['ragas_score'])
        print(f"\n🏆 BEST CORPUS-WIDE RESULT: {best['config_name']}")
        print(f"   🎯 RAGAS Score: {best['ragas_score']:.3f}")
        print(f"   📊 Corpus Stats:")
        print(f"      • {best['corpus_stats']['total_contexts']} unique contexts")
        print(f"      • {best['corpus_stats']['total_sentences']} total sentences")
        print(
            f"      • {best['corpus_stats']['duplicates_removed']} duplicates removed ({best['corpus_stats']['deduplication_rate']:.1f}%)")

        # Achievement check
        if best['ragas_score'] >= 0.85:
            print(f"\n🎉 SUCCESS! Achieved target RAGAS score of 0.85+")
            print(f"   📈 Score: {best['ragas_score']:.3f} (target: 0.85)")
        else:
            print(f"\n📈 Progress toward 0.85 target:")
            print(f"   Current: {best['ragas_score']:.3f}")
            print(f"   Target:  0.85")
            print(f"   Gap:     {0.85 - best['ragas_score']:.3f}")
    else:
        print(f"\n❌ No successful evaluations")

    print(f"\n✅ Corpus-wide evaluation complete! Total time: {total_time:.1f}s")


if __name__ == "__main__":
    main()