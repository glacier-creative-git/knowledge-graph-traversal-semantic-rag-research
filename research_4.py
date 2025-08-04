"""
Focused Semantic Graph RAG Evaluation Framework
==============================================

This script provides a streamlined evaluation framework comparing:
1. Document-Aware Semantic Graph RAG (per-document sparse matrices)
2. Hierarchical Semantic Graph RAG (+ cross-document connections)

Configurable via argparse for systematic optimization and benchmarking.

Sections for Jupyter Notebook:
1. Setup and Imports
2. Matrix Building Strategies (Modular)
3. Document-Aware Semantic Graph RAG Implementation
4. Hierarchical Semantic Graph RAG Implementation
5. RAGAS Evaluation Framework
6. Dataset Loading and Preparation
7. Focused Benchmarking
8. Results Analysis
"""

# ============================================================================
# Section 1: Setup and Imports
# ============================================================================

import os
import argparse
import numpy as np
import pandas as pd
import nltk
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict, Optional
import time
import warnings
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json

# RAGAS imports
from ragas import evaluate, EvaluationDataset
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy
)
from datasets import Dataset, load_dataset
from langchain_openai import ChatOpenAI

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

warnings.filterwarnings('ignore')


# ============================================================================
# Section 2: Matrix Building Strategies (Modular)
# ============================================================================

class MatrixBuilder:
    """Modular matrix building strategies for semantic graphs"""

    def __init__(self, strategy: str = "standard"):
        self.strategy = strategy

    def build_similarity_matrix(self, embeddings: np.ndarray, sentences: List[str]) -> np.ndarray:
        """Build similarity matrix using specified strategy"""
        if self.strategy == "standard":
            return self._standard_matrix(embeddings)
        elif self.strategy == "sliding_window":
            return self._sliding_window_matrix(embeddings, sentences)
        else:
            raise ValueError(f"Unknown matrix strategy: {self.strategy}")

    def _standard_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Standard sentence-to-sentence dot product similarity"""
        return np.dot(embeddings, embeddings.T)

    def _sliding_window_matrix(self, embeddings: np.ndarray, sentences: List[str]) -> np.ndarray:
        """3-sentence sliding window strategy (future implementation)"""
        # Placeholder for sliding window implementation
        # TODO: Implement 3-sentence sliding windows anchored at middle sentence
        return self._standard_matrix(embeddings)


# ============================================================================
# Section 3: Document-Aware Semantic Graph RAG Implementation
# ============================================================================

class DocumentAwareSemanticRAG:
    """Document-Aware Semantic Graph RAG with configurable parameters"""

    def __init__(self, top_k_per_sentence: int = 10, embedding_model: str = "all-MiniLM-L6-v2",
                 matrix_strategy: str = "standard", traversal_depth: int = 2):
        self.model = SentenceTransformer(embedding_model)
        self.top_k_per_sentence = top_k_per_sentence
        self.traversal_depth = traversal_depth
        self.matrix_builder = MatrixBuilder(matrix_strategy)

        # Document-aware storage
        self.documents = []
        self.document_graphs = {}
        self.document_sentences = {}
        self.document_embeddings = {}
        self.global_sentence_index = {}

    def ingest_documents(self, documents: List[Dict]) -> float:
        """Build sparse similarity matrices per document"""
        start_time = time.time()
        self.documents = documents
        total_sentences = 0
        total_relationships = 0

        for doc_id, doc_data in enumerate(documents):
            text = doc_data['context']
            sentences = nltk.sent_tokenize(text)
            self.document_sentences[doc_id] = sentences
            total_sentences += len(sentences)

            # Get embeddings for this document's sentences
            embeddings = self.model.encode(sentences)
            self.document_embeddings[doc_id] = embeddings

            # Build similarity matrix using modular strategy
            similarity_matrix = self.matrix_builder.build_similarity_matrix(embeddings, sentences)

            # Adaptive top-K based on document length
            adaptive_k = min(self.top_k_per_sentence, len(sentences) - 1)
            if adaptive_k <= 0:
                adaptive_k = max(1, len(sentences) - 1)

            # Keep only top-K relationships per sentence (sparse)
            sparse_matrix = {}
            for i in range(len(sentences)):
                similarities = similarity_matrix[i]
                k_to_use = min(adaptive_k, len(sentences) - 1)
                if k_to_use > 0:
                    top_indices = np.argsort(similarities)[-k_to_use:][::-1]
                else:
                    top_indices = np.array([i])

                sparse_matrix[i] = {
                    'indices': top_indices.tolist(),
                    'scores': similarities[top_indices].tolist()
                }
                total_relationships += len(top_indices)

            # Add to global index for fast lookup
            for sent_idx, sentence in enumerate(sentences):
                sentence_key = f"{doc_id}_{sent_idx}_{hash(sentence.strip())}"
                self.global_sentence_index[sentence_key] = (doc_id, sent_idx)
                sentence_hash = hash(sentence.strip())
                if sentence_hash not in self.global_sentence_index:
                    self.global_sentence_index[sentence_hash] = (doc_id, sent_idx)

            self.document_graphs[doc_id] = sparse_matrix

        ingest_time = time.time() - start_time
        return ingest_time

    def find_global_anchors(self, query: str, top_candidates: int = 10) -> List[Tuple[int, int, float]]:
        """Find anchor sentences across all documents"""
        query_embedding = self.model.encode([query])
        candidates = []

        for doc_id, embeddings in self.document_embeddings.items():
            similarities = np.dot(query_embedding, embeddings.T)[0]
            for sent_idx, score in enumerate(similarities):
                candidates.append((doc_id, sent_idx, score))

        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates[:top_candidates]

    def traverse_document_graph(self, doc_id: int, anchor_sent_idx: int) -> List[Tuple[str, float]]:
        """Traverse within a single document's semantic graph"""
        if doc_id not in self.document_graphs:
            return []

        graph = self.document_graphs[doc_id]
        sentences = self.document_sentences[doc_id]
        visited = set()
        results = []

        # BFS traversal within document
        queue = [(anchor_sent_idx, 1.0, 0)]

        while queue:
            sent_idx, score, current_depth = queue.pop(0)

            if sent_idx in visited or current_depth > self.traversal_depth:
                continue

            visited.add(sent_idx)
            results.append((sentences[sent_idx], score))

            # Add connected sentences if we haven't reached max depth
            if current_depth < self.traversal_depth and sent_idx in graph:
                for related_idx, related_score in zip(
                        graph[sent_idx]['indices'],
                        graph[sent_idx]['scores']
                ):
                    if related_idx not in visited:
                        new_score = score * 0.8 * (related_score / graph[sent_idx]['scores'][0])
                        queue.append((related_idx, new_score, current_depth + 1))

        return results

    def retrieve(self, query: str, top_k: int = 5) -> Tuple[List[str], float]:
        """Document-aware retrieval using matrix traversal"""
        start_time = time.time()

        # Find anchor sentences globally
        global_anchors = self.find_global_anchors(query, top_candidates=15)

        # Traverse within each anchor's document
        all_results = []
        for doc_id, anchor_sent_idx, anchor_score in global_anchors[:5]:
            doc_results = self.traverse_document_graph(doc_id, anchor_sent_idx)
            weighted_results = [(sent, score * anchor_score) for sent, score in doc_results]
            all_results.extend(weighted_results)

        # Sort by final score and return top-K
        all_results.sort(key=lambda x: x[1], reverse=True)
        retrieved_sentences = [sent for sent, score in all_results[:top_k]]

        retrieval_time = time.time() - start_time
        return retrieved_sentences, retrieval_time


# ============================================================================
# Section 4: Hierarchical Semantic Graph RAG Implementation
# ============================================================================

class HierarchicalSemanticRAG(DocumentAwareSemanticRAG):
    """Hierarchical Semantic Graph RAG with cross-document connections"""

    def __init__(self, top_k_per_sentence: int = 10, cross_doc_k: int = 5,
                 embedding_model: str = "all-MiniLM-L6-v2", matrix_strategy: str = "standard",
                 traversal_depth: int = 2):
        super().__init__(top_k_per_sentence, embedding_model, matrix_strategy, traversal_depth)
        self.cross_doc_k = cross_doc_k
        self.cross_document_matrix = {}

    def build_cross_document_connections(self):
        """Build sparse cross-document connections"""
        # Collect all sentence embeddings with IDs
        all_embeddings = []
        sentence_ids = []

        for doc_id, embeddings in self.document_embeddings.items():
            for sent_idx, embedding in enumerate(embeddings):
                all_embeddings.append(embedding)
                sentence_ids.append((doc_id, sent_idx))

        all_embeddings = np.array(all_embeddings)
        cross_connections = 0

        for i, (doc_id, sent_idx) in enumerate(sentence_ids):
            # Dot product with all other sentences
            similarities = np.dot(all_embeddings[i:i + 1], all_embeddings.T)[0]

            # Filter out same-document connections
            cross_doc_candidates = []
            for j, (other_doc_id, other_sent_idx) in enumerate(sentence_ids):
                if other_doc_id != doc_id:
                    cross_doc_candidates.append((j, other_doc_id, other_sent_idx, similarities[j]))

            # Keep top cross-document connections
            cross_doc_candidates.sort(key=lambda x: x[3], reverse=True)
            top_cross_connections = cross_doc_candidates[:self.cross_doc_k]

            # Store sparse cross-document connections
            sentence_key = f"{doc_id}_{sent_idx}"
            self.cross_document_matrix[sentence_key] = [
                (other_doc_id, other_sent_idx, score)
                for _, other_doc_id, other_sent_idx, score in top_cross_connections
            ]
            cross_connections += len(top_cross_connections)

    def ingest_documents(self, documents: List[Dict]) -> float:
        """Build document graphs + cross-document connections"""
        # First build individual document graphs
        ingest_time = super().ingest_documents(documents)

        # Then build cross-document connections
        cross_start = time.time()
        self.build_cross_document_connections()
        cross_time = time.time() - cross_start

        return ingest_time + cross_time

    def traverse_cross_documents(self, anchor_results: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Traverse cross-document connections from anchor results"""
        cross_results = []

        for sentence, score in anchor_results[:3]:
            sentence_hash = hash(sentence)
            if sentence_hash in self.global_sentence_index:
                doc_id, sent_idx = self.global_sentence_index[sentence_hash]
                sentence_key = f"{doc_id}_{sent_idx}"

                if sentence_key in self.cross_document_matrix:
                    cross_connections = self.cross_document_matrix[sentence_key]

                    for other_doc_id, other_sent_idx, cross_score in cross_connections:
                        if other_doc_id in self.document_sentences:
                            other_sentence = self.document_sentences[other_doc_id][other_sent_idx]
                            final_score = score * 0.6 * cross_score
                            cross_results.append((other_sentence, final_score))

        return cross_results

    def retrieve(self, query: str, top_k: int = 5) -> Tuple[List[str], float]:
        """Hierarchical retrieval: document-aware + cross-document discovery"""
        start_time = time.time()

        # Document-aware retrieval
        doc_results, _ = super().retrieve(query, top_k // 2 + 1)
        doc_results_with_scores = [(sent, 1.0) for sent in doc_results]

        # Cross-document discovery
        cross_results = self.traverse_cross_documents(doc_results_with_scores)

        # Combine and re-rank
        all_results = doc_results_with_scores + cross_results
        all_results.sort(key=lambda x: x[1], reverse=True)

        # Remove duplicates and return top-K
        seen = set()
        final_results = []
        for sent, score in all_results:
            if sent not in seen:
                seen.add(sent)
                final_results.append(sent)
                if len(final_results) >= top_k:
                    break

        retrieval_time = time.time() - start_time
        return final_results, retrieval_time


# ============================================================================
# Section 5: RAGAS Evaluation Framework
# ============================================================================

class RAGEvaluator:
    """RAGAS evaluation framework with configurable models"""

    def __init__(self, openai_api_key: str, model: str = "gpt-3.5-turbo", max_samples: int = 20):
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.llm = ChatOpenAI(
            model=model,
            api_key=openai_api_key,
            request_timeout=60,
            max_retries=3,
        )
        self.max_samples = max_samples
        self.metrics = [
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy
        ]

    def create_rag_dataset(self, rag_system, documents: List[Dict], queries: List[str],
                           ground_truths: Optional[List[str]] = None) -> List[Dict]:
        """Create evaluation dataset by running queries through RAG system"""
        max_queries = min(len(queries), self.max_samples)
        limited_queries = queries[:max_queries]
        limited_ground_truths = ground_truths[:max_queries] if ground_truths else None

        dataset = []

        for i, query in enumerate(limited_queries):
            try:
                contexts, _ = rag_system.retrieve(query, top_k=3)
                context_text = "\n".join(contexts)
                answer = f"Based on the provided context, {query.lower()}"

                sample = {
                    "user_input": query,
                    "retrieved_contexts": contexts,
                    "response": answer
                }

                if limited_ground_truths and i < len(limited_ground_truths):
                    sample["reference"] = limited_ground_truths[i]

                dataset.append(sample)

            except Exception as e:
                continue

        return dataset

    def evaluate_rag_system(self, rag_system, documents: List[Dict],
                            queries: List[str], ground_truths: Optional[List[str]] = None,
                            system_name: str = "RAG System") -> Dict:
        """Evaluate a RAG system using RAGAS metrics"""
        # Ingest documents
        ingest_time = rag_system.ingest_documents(documents)

        # Create evaluation dataset
        dataset = self.create_rag_dataset(rag_system, documents, queries, ground_truths)

        if not dataset:
            return {"error": "No valid samples created"}

        # Convert to RAGAS format
        evaluation_dataset = EvaluationDataset.from_list(dataset)

        start_time = time.time()
        try:
            result = evaluate(
                dataset=evaluation_dataset,
                metrics=self.metrics,
                llm=self.llm,
                raise_exceptions=False
            )

            eval_time = time.time() - start_time
            scores = {}
            metrics_to_extract = ['context_precision', 'context_recall', 'faithfulness', 'answer_relevancy']

            # Handle RAGAS result extraction
            result_scores = None
            if hasattr(result, 'scores') and result.scores is not None:
                result_scores = result.scores
            elif hasattr(result, '_scores_dict') and result._scores_dict is not None:
                result_scores = result._scores_dict

            # Process results
            if isinstance(result_scores, list) and len(result_scores) > 0:
                metric_totals = {metric: [] for metric in metrics_to_extract}

                for sample_scores in result_scores:
                    for metric_name in metrics_to_extract:
                        if metric_name in sample_scores:
                            score_val = sample_scores[metric_name]
                            if hasattr(score_val, 'item'):
                                score_val = float(score_val.item())
                            else:
                                score_val = float(score_val)

                            if not np.isnan(score_val):
                                metric_totals[metric_name].append(score_val)

                for metric_name in metrics_to_extract:
                    if metric_totals[metric_name]:
                        avg_score = np.mean(metric_totals[metric_name])
                        scores[metric_name] = avg_score
                    else:
                        scores[metric_name] = 0.0
            else:
                for metric_name in metrics_to_extract:
                    try:
                        score_value = 0.0
                        if hasattr(result, metric_name):
                            score_value = float(getattr(result, metric_name))
                        elif isinstance(result, dict) and metric_name in result:
                            score_value = float(result[metric_name])
                        scores[metric_name] = score_value
                    except (TypeError, ValueError, AttributeError):
                        scores[metric_name] = 0.0

            # Calculate overall RAGAS score
            valid_scores = [score for score in scores.values() if score > 0]
            if valid_scores:
                scores['ragas_score'] = np.mean(list(scores.values()))
            else:
                scores['ragas_score'] = 0.0

            scores.update({
                'ingest_time': ingest_time,
                'eval_time': eval_time,
                'num_samples': len(dataset)
            })

            return scores

        except Exception as e:
            return {"error": str(e)}


# ============================================================================
# Section 6: Dataset Loading and Preparation
# ============================================================================

def load_squad_dataset(num_samples: int = 500, version: str = "2.0") -> Dict:
    """Load SQuAD dataset with version support"""
    dataset_name = "squad_v2" if version == "2.0" else "squad"

    try:
        squad = load_dataset(dataset_name, split=f"validation[:{num_samples}]")
        documents = []
        queries = []
        ground_truths = []

        for item in squad:
            documents.append({
                'context': item['context'],
                'question': item['question'],
                'id': item['id']
            })
            queries.append(item['question'])

            if item['answers']['text']:
                ground_truths.append(item['answers']['text'][0])
            else:
                ground_truths.append("No answer can be determined from the given context")

        return {
            'documents': documents,
            'queries': queries,
            'ground_truths': ground_truths,
            'name': f'SQuAD {version}'
        }

    except Exception as e:
        return create_long_document_dataset()


def create_long_document_dataset() -> Dict:
    """Create long document dataset for testing"""
    documents = [
        {
            'context': """Artificial intelligence has revolutionized many fields in recent years. Machine learning algorithms 
            can now process vast amounts of data to identify complex patterns that were previously invisible to 
            human analysts. Deep learning networks use multiple layers of artificial neurons to extract features 
            from raw input data in ways that mimic human cognition. Natural language processing enables computers 
            to understand and generate human text with remarkable accuracy. Computer vision allows machines to 
            interpret visual information from the world around them. These AI technologies are transforming 
            industries from healthcare to finance to transportation.""",
            'question': 'How is AI being used in healthcare diagnostics?',
            'id': 'long_doc_1'
        },
        {
            'context': """The history of space exploration spans over six decades of human ingenuity and determination. 
            It began in 1957 with the Soviet launch of Sputnik 1, the first artificial satellite to orbit Earth. 
            This achievement marked the beginning of the Space Age and sparked the Space Race between the Soviet 
            Union and the United States. The Apollo program represented the pinnacle of early space exploration efforts. 
            President John F. Kennedy's bold declaration in 1961 to land humans on the Moon before the decade's end 
            galvanized American space efforts.""",
            'question': 'What were the major achievements of the Apollo program?',
            'id': 'long_doc_2'
        }
    ]

    queries = [doc['question'] for doc in documents]
    ground_truths = [
        "AI diagnostic systems can detect diseases earlier than human doctors by analyzing medical images like X-rays, MRIs, and CT scans to identify subtle patterns indicating early-stage cancers or other conditions.",
        "The Apollo program achieved President Kennedy's goal of landing humans on the Moon before 1970, with Apollo 11 successfully landing on July 20, 1969, when Neil Armstrong and Buzz Aldrin became the first humans to walk on the Moon."
    ]

    return {
        'documents': documents,
        'queries': queries,
        'ground_truths': ground_truths,
        'name': 'Long Multi-Topic Documents'
    }


def load_evaluation_dataset(dataset_type: str = "long_docs", num_samples: int = 50) -> Dict:
    """Load evaluation dataset based on type"""
    if dataset_type == "squad":
        return load_squad_dataset(num_samples)
    elif dataset_type == "squad_2":
        return load_squad_dataset(num_samples, "2.0")
    else:
        return create_long_document_dataset()


# ============================================================================
# Section 7: Focused Benchmarking
# ============================================================================

def run_focused_evaluation(args) -> Dict:
    """Run focused evaluation with specified configuration"""

    # Initialize RAG system based on algorithm choice
    if args.algorithm == "document_aware":
        rag_system = DocumentAwareSemanticRAG(
            top_k_per_sentence=args.k_value,
            embedding_model=args.embedding_model,
            matrix_strategy=args.matrix_strategy,
            traversal_depth=args.traversal_depth
        )
        system_name = "Document-Aware Semantic Graph RAG"
    else:  # hierarchical
        rag_system = HierarchicalSemanticRAG(
            top_k_per_sentence=args.k_value,
            cross_doc_k=args.cross_k_value,
            embedding_model=args.embedding_model,
            matrix_strategy=args.matrix_strategy,
            traversal_depth=args.traversal_depth
        )
        system_name = "Hierarchical Semantic Graph RAG"

    # Initialize evaluator
    evaluator = RAGEvaluator(
        args.openai_api_key,
        model=args.eval_model,
        max_samples=args.max_eval_samples
    )

    # Load dataset
    dataset_info = load_evaluation_dataset(args.dataset_type, args.num_samples)
    documents = dataset_info['documents']
    queries = dataset_info['queries']
    ground_truths = dataset_info['ground_truths']

    # Run evaluation
    results = evaluator.evaluate_rag_system(
        rag_system, documents, queries, ground_truths, system_name
    )

    # Add configuration info to results
    results['config'] = {
        'algorithm': args.algorithm,
        'embedding_model': args.embedding_model,
        'eval_model': args.eval_model,
        'k_value': args.k_value,
        'cross_k_value': args.cross_k_value,
        'traversal_depth': args.traversal_depth,
        'matrix_strategy': args.matrix_strategy,
        'dataset_type': args.dataset_type,
        'num_samples': args.num_samples,
        'max_eval_samples': args.max_eval_samples
    }

    results['dataset_info'] = {
        'name': dataset_info['name'],
        'num_documents': len(documents),
        'num_queries': len(queries),
    }

    return results


# ============================================================================
# Section 8: Results Analysis
# ============================================================================

def print_results(results: Dict):
    """Print evaluation results in clean format"""

    if 'error' in results:
        print(f"Error: {results['error']}")
        return

    config = results['config']

    print(f"Configuration:")
    print(f"  Algorithm: {config['algorithm']}")
    print(f"  Embedding Model: {config['embedding_model']}")
    print(f"  K-Value: {config['k_value']}")
    if config['algorithm'] == 'hierarchical':
        print(f"  Cross-K Value: {config['cross_k_value']}")
    print(f"  Traversal Depth: {config['traversal_depth']}")
    print(f"  Matrix Strategy: {config['matrix_strategy']}")
    print(f"  Dataset: {results['dataset_info']['name']}")
    print()

    # Core metrics
    metrics = ['context_precision', 'context_recall', 'faithfulness', 'answer_relevancy', 'ragas_score']

    print("Results:")
    for metric in metrics:
        score = results.get(metric, 0)
        print(f"  {metric}: {score:.3f}")

    print()
    print(f"Performance:")
    print(f"  Ingestion Time: {results.get('ingest_time', 0):.3f}s")
    print(f"  Evaluation Time: {results.get('eval_time', 0):.3f}s")
    print(f"  Samples Evaluated: {results.get('num_samples', 0)}")


def save_results(results: Dict, filename: str):
    """Save results to JSON file"""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)


# ============================================================================
# Main Execution with Argparse
# ============================================================================

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Focused Semantic Graph RAG Evaluation')

    # Core configuration
    parser.add_argument('--algorithm', choices=['document_aware', 'hierarchical'],
                        default='hierarchical', help='RAG algorithm to evaluate')
    parser.add_argument('--embedding-model', default='all-MiniLM-L6-v2',
                        help='Sentence transformer model for embeddings')
    parser.add_argument('--eval-model', choices=['gpt-3.5-turbo', 'gpt-4'],
                        default='gpt-3.5-turbo', help='LLM model for RAGAS evaluation')

    # Matrix and graph parameters
    parser.add_argument('--k-value', type=int, default=15,
                        help='Top-K sentences per sentence in document graphs')
    parser.add_argument('--cross-k-value', type=int, default=8,
                        help='Top-K cross-document connections (hierarchical only)')
    parser.add_argument('--traversal-depth', type=int, default=2,
                        help='Maximum depth for graph traversal')
    parser.add_argument('--matrix-strategy', choices=['standard', 'sliding_window'],
                        default='standard', help='Matrix building strategy')

    # Dataset and evaluation parameters
    parser.add_argument('--dataset-type', choices=['squad', 'squad_2', 'long_docs'],
                        default='squad_2', help='Dataset type for evaluation')
    parser.add_argument('--num-samples', type=int, default=50,
                        help='Number of documents to load')
    parser.add_argument('--max-eval-samples', type=int, default=20,
                        help='Maximum samples for RAGAS evaluation')

    # API and output
    parser.add_argument('--openai-api-key', required=True,
                        help='OpenAI API key for RAGAS evaluation')
    parser.add_argument('--output-file', default=None,
                        help='JSON file to save results')

    # Preset configurations
    parser.add_argument('--aggressive-k', action='store_true',
                        help='Use aggressive K values (k=50, cross_k=20)')

    return parser.parse_args()


def main():
    """Main execution function"""
    args = parse_arguments()

    # Apply aggressive K preset if requested
    if args.aggressive_k:
        args.k_value = 50
        args.cross_k_value = 20

    # Set OpenAI API key environment variable
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OPENAI_API_KEY"] = args.openai_api_key

    # Run evaluation
    print("Starting focused semantic graph RAG evaluation...")
    results = run_focused_evaluation(args)

    # Print results
    print_results(results)

    # Save results if requested
    if args.output_file:
        save_results(results, args.output_file)
        print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()