#!/usr/bin/env python3
"""
Enhanced Corpus-Wide RAGAS Evaluation Script
============================================

Evaluates corpus-wide semantic graph RAG system supporting WikiEval,
Natural Questions, and future datasets. Updated to use the enhanced
data loading system instead of SQuAD dependency.

This should achieve higher RAGAS scores by:
1. Searching across entire corpus instead of pre-selected documents
2. Proper deduplication of contexts
3. Algorithm automatically discovers optimal number of documents
4. True corpus-wide semantic graph traversal

Usage:
    python corpus_ragas_evaluation.py --dataset wikieval
    python corpus_ragas_evaluation.py --dataset natural_questions --corpus-size 1000
"""

import os
import sys
import time
import argparse
from typing import Dict, List

# Add utils to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.corpus_wide_rag import CorpusWideSemanticRAG
from utils.config import get_config, get_available_datasets, get_dataset_info, print_dataset_info
from utils.data_loader import create_data_loader
from utils.evaluation import RAGASEvaluator, print_evaluation_results

# ============================================================================
# CONFIGURATION - SET YOUR API KEY HERE
# ============================================================================

OPENAI_API_KEY = "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"  # 🔑 PUT YOUR API KEY HERE

# Test configurations for corpus-wide search with dataset support
CORPUS_CONFIGS = {
    "WikiEval_Corpus_Conservative": {
        "dataset_name": "wikieval",
        "similarity_threshold": 0.4,
        "top_k_per_sentence": 25,
        "cross_doc_k": 15,
        "retrieval_top_k": 8,
        "max_corpus_size": 50  # WikiEval only has 50 samples
    },
    "WikiEval_Corpus_Balanced": {
        "dataset_name": "wikieval",
        "similarity_threshold": 0.5,
        "top_k_per_sentence": 30,
        "cross_doc_k": 20,
        "retrieval_top_k": 10,
        "max_corpus_size": 50
    },
    "WikiEval_Corpus_Aggressive": {
        "dataset_name": "wikieval",
        "similarity_threshold": 0.6,
        "top_k_per_sentence": 35,
        "cross_doc_k": 25,
        "retrieval_top_k": 12,
        "max_corpus_size": 50
    },
    "NaturalQuestions_Corpus_Conservative": {
        "dataset_name": "natural_questions",
        "similarity_threshold": 0.4,
        "top_k_per_sentence": 25,
        "cross_doc_k": 15,
        "retrieval_top_k": 8,
        "max_corpus_size": 800
    },
    "NaturalQuestions_Corpus_Balanced": {
        "dataset_name": "natural_questions",
        "similarity_threshold": 0.5,
        "top_k_per_sentence": 30,
        "cross_doc_k": 20,
        "retrieval_top_k": 10,
        "max_corpus_size": 1000
    },
    "NaturalQuestions_Corpus_Aggressive": {
        "dataset_name": "natural_questions",
        "similarity_threshold": 0.6,
        "top_k_per_sentence": 35,
        "cross_doc_k": 25,
        "retrieval_top_k": 12,
        "max_corpus_size": 1000
    }
}


class EnhancedCorpusWideEvaluator:
    """Evaluator for corpus-wide semantic graph RAG with enhanced dataset support"""

    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        os.environ["OPENAI_API_KEY"] = openai_api_key
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def create_corpus_evaluation_dataset(self, data_loader, num_eval_questions: int = 30) -> Dict:
        """
        Create evaluation dataset from the configured data loader for corpus-wide evaluation.

        Args:
            data_loader: Enhanced data loader instance
            num_eval_questions: Number of questions to evaluate

        Returns:
            Dictionary with evaluation data
        """
        print(f"📊 Creating corpus evaluation dataset from {data_loader.get_dataset_name()}...")

        # Load the dataset first
        if not data_loader.load_dataset():
            raise ValueError(f"Failed to load {data_loader.get_dataset_name()} dataset")

        # Get evaluation dataset from the data loader
        eval_data = data_loader.get_evaluation_dataset(num_eval_questions)

        # Extract queries and ground truths
        eval_questions = eval_data['queries'][:num_eval_questions]
        eval_ground_truths = eval_data['ground_truths'][:num_eval_questions]

        return {
            'queries': eval_questions,
            'ground_truths': eval_ground_truths,
            'name': f'Corpus-Wide {data_loader.get_dataset_name()} Evaluation ({len(eval_questions)} questions)'
        }

    def create_corpus_data(self, data_loader, max_corpus_size: int) -> List[Dict]:
        """
        Create corpus data from the data loader in the format expected by CorpusWideSemanticRAG.

        Args:
            data_loader: Enhanced data loader instance
            max_corpus_size: Maximum corpus size

        Returns:
            List of dictionaries in format compatible with CorpusWideSemanticRAG
        """
        print(f"🌍 Creating corpus data from {data_loader.get_dataset_name()}...")

        if not data_loader.load_dataset():
            raise ValueError(f"Failed to load {data_loader.get_dataset_name()} dataset")

        # Get data in evaluation format and convert to corpus format
        eval_data = data_loader.get_evaluation_dataset(max_corpus_size)

        corpus_data = []
        for i, doc in enumerate(eval_data['documents']):
            # Convert to format expected by CorpusWideSemanticRAG.ingest_corpus()
            corpus_item = {
                'context': doc['context'],
                'question': doc['question'],
                'id': doc['id'],
                'title': f"{data_loader.get_dataset_name()} Document {i + 1}"
            }
            corpus_data.append(corpus_item)

        print(f"✅ Created corpus with {len(corpus_data)} documents from {data_loader.get_dataset_name()}")
        return corpus_data

    def evaluate_corpus_rag(self, config_name: str, config_params: Dict) -> Dict:
        """Evaluate corpus-wide RAG system with enhanced dataset support"""
        print(f"\n{'=' * 70}")
        print(f"🌍 CORPUS-WIDE EVALUATION: {config_name.upper()}")
        print(f"📊 Dataset: {config_params['dataset_name'].replace('_', ' ').title()}")
        print(f"{'=' * 70}")

        print(f"📋 Configuration:")
        for param, value in config_params.items():
            print(f"   {param}: {value}")
        print()

        try:
            # Create data loader for the specified dataset
            from utils.config import DataConfig
            data_config = DataConfig(
                dataset_name=config_params['dataset_name'],
                num_samples=config_params.get('max_corpus_size', 1000)
            )

            data_loader = create_data_loader(config_params['dataset_name'], data_config)

            # Create corpus-wide RAG system
            rag_system = CorpusWideSemanticRAG(
                top_k_per_sentence=config_params['top_k_per_sentence'],
                cross_doc_k=config_params['cross_doc_k'],
                similarity_threshold=config_params['similarity_threshold'],
                max_corpus_size=config_params['max_corpus_size']
            )

            # Create corpus data from the dataset
            corpus_data = self.create_corpus_data(data_loader, config_params['max_corpus_size'])

            # Ingest corpus
            print(f"🌍 Ingesting {data_loader.get_dataset_name()} corpus...")
            ingest_time = rag_system.ingest_corpus(corpus_data, config_params['max_corpus_size'])

            # Create evaluation dataset (questions only - corpus is already ingested)
            eval_data = self.create_corpus_evaluation_dataset(data_loader, num_eval_questions=25)

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
                        "response": f"Based on the {data_loader.get_dataset_name()} corpus search: {retrieved_texts[0][:200]}..." if retrieved_texts else "No relevant information found."
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
                'dataset_name': config_params['dataset_name'],
                'config_params': config_params,
                'ingest_time': ingest_time,
                'eval_time': eval_time,
                'num_samples': len(ragas_dataset),
                'corpus_stats': {
                    'total_contexts': len(rag_system.corpus_contexts),
                    'total_sentences': len(rag_system.sentences_info),
                    'duplicates_removed': rag_system.duplicate_count,
                    'deduplication_rate': rag_system.duplicate_count / (
                            len(rag_system.corpus_contexts) + rag_system.duplicate_count) * 100 if (
                                                                                                               len(rag_system.corpus_contexts) + rag_system.duplicate_count) > 0 else 0
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


def filter_configs_by_dataset(configs: Dict[str, Dict], target_dataset: str) -> Dict[str, Dict]:
    """Filter configurations to only include those for the specified dataset."""
    if target_dataset == "all":
        return configs

    filtered = {}
    dataset_key = target_dataset.replace('_', '').lower()

    for name, params in configs.items():
        name_lower = name.lower()
        if dataset_key in name_lower or params.get("dataset_name", "").lower() == target_dataset:
            filtered[name] = params

    return filtered


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Enhanced Corpus-Wide RAGAS Evaluation")

    parser.add_argument("--dataset", type=str, default="all",
                        choices=["all", "wikieval", "natural_questions"],
                        help="Dataset to evaluate on (default: all)")

    parser.add_argument("--config", type=str, default="all",
                        help="Specific configuration to run (default: all)")

    parser.add_argument("--corpus-size", type=int, default=None,
                        help="Override max corpus size")

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

    print("🔥" * 80)
    print("🔥" + " " * 76 + "🔥")
    print("🔥" + "          ENHANCED CORPUS-WIDE SEMANTIC GRAPH RAG EVALUATION          ".center(76) + "🔥")
    print("🔥" + " " * 76 + "🔥")
    print("🔥" * 80)
    print()
    print("🌍 This evaluation tests semantic graph traversal across ENTIRE corpus")
    print("🎯 Target: RAGAS scores 0.85+ by leveraging full corpus discovery")
    print("✨ Enhanced features:")
    print("   • Support for WikiEval and Natural Questions datasets")
    print("   • Searches entire corpus instead of pre-selected documents")
    print("   • Automatic document quantity selection by traversal algorithm")
    print("   • Proper deduplication of contexts")
    print("   • True corpus-wide semantic graph connections")
    print()

    # Update API key if provided
    global OPENAI_API_KEY
    if args.api_key:
        OPENAI_API_KEY = args.api_key

    # Validate API key
    if not OPENAI_API_KEY or OPENAI_API_KEY == "your-openai-api-key-here":
        print("❌ ERROR: Please set your OpenAI API key in the script!")
        return

    # Filter configurations based on dataset selection
    if args.dataset != "all":
        configs_to_run = filter_configs_by_dataset(CORPUS_CONFIGS, args.dataset)
        if not configs_to_run:
            print(f"❌ No configurations found for dataset: {args.dataset}")
            print(f"Available datasets: {get_available_datasets()}")
            return
        print(f"🎯 Running corpus evaluations for dataset: {args.dataset.replace('_', ' ').title()}")
    else:
        configs_to_run = CORPUS_CONFIGS
        print(f"🎯 Running corpus evaluations for all datasets")

    # Filter by specific config if requested
    if args.config != "all":
        if args.config in configs_to_run:
            configs_to_run = {args.config: configs_to_run[args.config]}
        else:
            print(f"❌ Configuration '{args.config}' not found.")
            print(f"Available configurations: {list(configs_to_run.keys())}")
            return

    # Override corpus size if specified
    if args.corpus_size:
        for config in configs_to_run.values():
            config['max_corpus_size'] = args.corpus_size

    # Run evaluations
    evaluator = EnhancedCorpusWideEvaluator(OPENAI_API_KEY)
    all_results = []

    total_start = time.time()

    for i, (config_name, config_params) in enumerate(configs_to_run.items(), 1):
        print(f"\n⏳ Progress: {i}/{len(configs_to_run)} configurations")

        result = evaluator.evaluate_corpus_rag(config_name, config_params)
        all_results.append(result)

        if 'error' not in result:
            dataset_display = result['dataset_name'].replace('_', ' ').title()
            print(f"✅ {config_name} ({dataset_display}): RAGAS {result['ragas_score']:.3f}")
            if 'corpus_stats' in result:
                print(f"   Corpus: {result['corpus_stats']['total_contexts']} contexts, "
                      f"{result['corpus_stats']['duplicates_removed']} duplicates removed")
        else:
            print(f"❌ {config_name}: {result['error']}")

    total_time = time.time() - total_start

    # Results comparison
    successful_results = [r for r in all_results if 'error' not in r]

    if successful_results:
        print(f"\n{'=' * 90}")
        print(f"📊 ENHANCED CORPUS-WIDE RESULTS COMPARISON")
        print(f"{'=' * 90}")

        print(
            f"{'Config':<25} {'Dataset':<15} {'RAGAS':<8} {'Precision':<10} {'Recall':<8} {'Faithful':<10} {'Relevancy':<10}")
        print(f"{'-' * 90}")

        for result in successful_results:
            config_name = result['config_name'][:24]
            dataset_name = result['dataset_name'].replace('_', ' ')[:14]
            print(f"{config_name:<25} "
                  f"{dataset_name:<15} "
                  f"{result['ragas_score']:<8.3f} "
                  f"{result['context_precision']:<10.3f} "
                  f"{result['context_recall']:<8.3f} "
                  f"{result['faithfulness']:<10.3f} "
                  f"{result['answer_relevancy']:<10.3f}")

        # Best result per dataset
        datasets = set(r['dataset_name'] for r in successful_results)

        print(f"\n🏆 BEST RESULTS BY DATASET:")
        for dataset in datasets:
            dataset_results = [r for r in successful_results if r['dataset_name'] == dataset]
            if dataset_results:
                best = max(dataset_results, key=lambda x: x['ragas_score'])
                dataset_display = dataset.replace('_', ' ').title()
                print(f"   📊 {dataset_display}: {best['ragas_score']:.3f} ({best['config_name']})")
                if 'corpus_stats' in best:
                    print(f"      • {best['corpus_stats']['total_contexts']} unique contexts")
                    print(f"      • {best['corpus_stats']['total_sentences']} total sentences")
                    print(
                        f"      • {best['corpus_stats']['duplicates_removed']} duplicates removed ({best['corpus_stats']['deduplication_rate']:.1f}%)")

        # Overall best result
        overall_best = max(successful_results, key=lambda x: x['ragas_score'])
        print(f"\n🏆 BEST OVERALL CORPUS-WIDE RESULT: {overall_best['config_name']}")
        print(f"   🎯 RAGAS Score: {overall_best['ragas_score']:.3f}")
        print(f"   📊 Dataset: {overall_best['dataset_name'].replace('_', ' ').title()}")

        # Achievement check
        if overall_best['ragas_score'] >= 0.85:
            print(f"\n🎉 SUCCESS! Achieved target RAGAS score of 0.85+")
            print(f"   📈 Score: {overall_best['ragas_score']:.3f} (target: 0.85)")
        else:
            print(f"\n📈 Progress toward 0.85 target:")
            print(f"   Current: {overall_best['ragas_score']:.3f}")
            print(f"   Target:  0.85")
            print(f"   Gap:     {0.85 - overall_best['ragas_score']:.3f}")
    else:
        print(f"\n❌ No successful evaluations")

    print(f"\n✅ Enhanced corpus-wide evaluation complete! Total time: {total_time:.1f}s")


if __name__ == "__main__":
    main()