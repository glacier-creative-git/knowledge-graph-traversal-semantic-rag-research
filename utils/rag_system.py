"""
Semantic Graph RAG System
========================

Core implementation of the semantic graph traversal algorithm for RAG.
Extracted and modularized from the visualization pipelines.
"""

import re
import time
import nltk
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        a, b: Input vectors

    Returns:
        Cosine similarity score between 0 and 1
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))

def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Calculate cosine similarity matrix for a set of embeddings.

    Args:
        embeddings: (n_samples, embedding_dim) array

    Returns:
        (n_samples, n_samples) cosine similarity matrix
    """
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    normalized = embeddings / norms

    # Compute cosine similarity matrix
    return np.dot(normalized, normalized.T)

@dataclass
class SentenceInfo:
    """Information about a sentence in the corpus"""
    doc_id: int
    paragraph_id: int
    sentence_id: int
    text: str
    embedding: np.ndarray
    position_in_doc: float
    position_in_paragraph: float
    source_context_id: str

@dataclass
class TraversalStep:
    """Information about a single step in semantic graph traversal"""
    sentence_info: SentenceInfo
    step_number: int
    relevance_score: float
    connection_type: str  # 'anchor', 'same_paragraph', 'neighboring_paragraph', 'distant_paragraph', 'cross_document'
    distance_from_anchor: int

class SemanticGraphRAG:
    """
    Semantic Graph RAG system with document-aware traversal

    This implements the core algorithm extracted from the visualization pipelines.
    """

    def __init__(self,
                 top_k_per_sentence: int = 20,
                 cross_doc_k: int = 10,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 traversal_depth: int = 3,
                 use_sliding_window: bool = True,
                 num_contexts: int = 5,
                 similarity_threshold: float = 0.5):

        self.model = SentenceTransformer(embedding_model)
        self.top_k_per_sentence = top_k_per_sentence
        self.cross_doc_k = cross_doc_k
        self.traversal_depth = traversal_depth
        self.use_sliding_window = use_sliding_window
        self.num_contexts = num_contexts
        self.similarity_threshold = similarity_threshold

        # Storage for semantic graph
        self.sentences_info: List[SentenceInfo] = []
        self.document_graphs = {}
        self.cross_document_matrix = {}
        self.global_sentence_index = {}
        self.similarity_matrices = []

        # Analysis data
        self.selected_question = None
        self.selected_contexts = []

    def ingest_contexts(self, contexts: List[Dict]) -> float:
        """
        Ingest contexts and build semantic graphs

        Args:
            contexts: List of context dictionaries

        Returns:
            Ingestion time in seconds
        """
        start_time = time.time()

        # Reset storage
        self.sentences_info = []
        self.document_graphs = {}
        self.cross_document_matrix = {}
        self.global_sentence_index = {}
        self.similarity_matrices = []
        self.selected_contexts = contexts

        print(f"🔄 Processing {len(contexts)} contexts...")
        print(f"📋 Using {'3-sentence forward-looking sliding windows' if self.use_sliding_window else 'single sentence embeddings'}")
        print(f"🧮 Using cosine similarity for all similarity calculations")

        # Process each context as a separate document
        for doc_id, context_data in enumerate(contexts):
            text = context_data['context']
            context_id = context_data['id']

            print(f"📄 Processing Document {doc_id} ({context_id})...")

            # Analyze document structure
            doc_sentence_infos = self._analyze_document_structure(doc_id, text, context_id)

            # Add to global sentence info
            for sentence_info in doc_sentence_infos:
                sentence_info.sentence_id = len(self.sentences_info)
                self.sentences_info.append(sentence_info)

                # Add to global index
                sentence_hash = hash(sentence_info.text.strip())
                self.global_sentence_index[sentence_hash] = len(self.sentences_info) - 1

        print(f"📊 Processed {len(contexts)} contexts into {len(self.sentences_info)} sentences")
        print(f"🔗 Building semantic graphs (top-{self.top_k_per_sentence} per sentence, {self.cross_doc_k} cross-doc connections)...")

        # Build graphs
        self._build_document_graphs()
        self._build_cross_document_connections()

        # Summary
        total_connections = sum(len(graph) for graph in self.document_graphs.values())
        cross_connections = len(self.cross_document_matrix)

        print(f"✅ Built {len(self.document_graphs)} document graphs with {total_connections} within-doc connections")
        print(f"✅ Built {cross_connections} cross-document connection mappings")

        return time.time() - start_time

    def retrieve(self, query: str, top_k: int = 10) -> Tuple[List[str], List[TraversalStep], Dict]:
        """
        Perform semantic graph traversal retrieval

        Args:
            query: Query string
            top_k: Number of results to return

        Returns:
            Tuple of (retrieved_texts, traversal_steps, analysis)
        """
        self.selected_question = query

        # Perform retrieval with traversal tracking
        retrieved_texts, traversal_steps = self._detailed_retrieve(query, top_k)

        # Analyze traversal patterns
        analysis = self._analyze_traversal_patterns(traversal_steps)

        print(f"🎯 Retrieval complete: {len(retrieved_texts)} texts from {len(traversal_steps)} steps")

        return retrieved_texts, traversal_steps, analysis

    def _extract_paragraphs(self, text: str) -> List[str]:
        """Extract paragraphs from text"""
        paragraphs = re.split(r'\n\s*\n|\r\n\s*\r\n|\.[\s]{3,}', text.strip())
        paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 10]

        # If very few paragraphs, treat whole text as one paragraph
        if len(paragraphs) <= 1:
            paragraphs = [text.strip()]

        return paragraphs

    def _create_sliding_window(self, sentences: List[str], anchor_idx: int) -> str:
        """Create 3-sentence forward-looking sliding window"""
        total_sentences = len(sentences)

        if total_sentences == 1:
            return sentences[0]
        elif total_sentences == 2:
            return ' '.join(sentences)
        elif anchor_idx >= total_sentences - 2:
            return ' '.join(sentences[-2:])
        else:
            end_idx = min(anchor_idx + 3, total_sentences)
            return ' '.join(sentences[anchor_idx:end_idx])

    def _analyze_document_structure(self, doc_id: int, text: str, source_context_id: str = "") -> List[SentenceInfo]:
        """Analyze document structure with paragraph awareness"""
        paragraphs = self._extract_paragraphs(text)
        sentence_infos = []
        global_sentence_id = 0

        for paragraph_id, paragraph_text in enumerate(paragraphs):
            sentences = nltk.sent_tokenize(paragraph_text)

            for sent_idx, sentence in enumerate(sentences):
                # Calculate relative positions
                total_sentences_in_doc = len(nltk.sent_tokenize(text))
                position_in_doc = global_sentence_id / max(1, total_sentences_in_doc - 1)
                position_in_paragraph = sent_idx / max(1, len(sentences) - 1)

                # Create embedding based on strategy
                if self.use_sliding_window:
                    window_text = self._create_sliding_window(sentences, sent_idx)
                    embedding = self.model.encode([window_text])[0]
                else:
                    embedding = self.model.encode([sentence])[0]

                sentence_info = SentenceInfo(
                    doc_id=doc_id,
                    paragraph_id=paragraph_id,
                    sentence_id=global_sentence_id,
                    text=sentence,
                    embedding=embedding,
                    position_in_doc=position_in_doc,
                    position_in_paragraph=position_in_paragraph,
                    source_context_id=source_context_id
                )

                sentence_infos.append(sentence_info)
                global_sentence_id += 1

        return sentence_infos

    def _build_document_graphs(self):
        """Build semantic graphs for each document using cosine similarity"""
        documents_sentences = defaultdict(list)

        # Group sentences by document
        for sentence_info in self.sentences_info:
            documents_sentences[sentence_info.doc_id].append(sentence_info)

        # Build graph for each document
        for doc_id, doc_sentences in documents_sentences.items():
            if len(doc_sentences) < 2:
                if len(doc_sentences) == 1:
                    self.similarity_matrices.append(np.array([[1.0]]))
                continue

            # Get embeddings for this document
            embeddings = np.array([s.embedding for s in doc_sentences])

            # Build cosine similarity matrix
            similarity_matrix = cosine_similarity_matrix(embeddings)
            self.similarity_matrices.append(similarity_matrix)

            # Build sparse graph
            sparse_matrix = {}
            for i, sentence_info in enumerate(doc_sentences):
                similarities = similarity_matrix[i]

                # Get top-K most similar sentences
                k_to_use = min(self.top_k_per_sentence, len(doc_sentences) - 1)
                if k_to_use > 0:
                    top_indices = np.argsort(similarities)[-k_to_use:][::-1]
                else:
                    top_indices = np.array([i])

                sparse_matrix[sentence_info.sentence_id] = {
                    'indices': [doc_sentences[idx].sentence_id for idx in top_indices],
                    'scores': similarities[top_indices].tolist()
                }

            self.document_graphs[doc_id] = sparse_matrix

    def _build_cross_document_connections(self):
        """Build cross-document semantic connections using cosine similarity"""
        if len(self.sentences_info) < 2:
            return

        # Get all embeddings
        all_embeddings = np.array([s.embedding for s in self.sentences_info])

        # Build cross-document connections using cosine similarity
        for i, sentence_info in enumerate(self.sentences_info):
            # Calculate cosine similarities to all other sentences
            current_embedding = all_embeddings[i]

            # Filter out same-document connections
            cross_doc_candidates = []
            for j, other_sentence_info in enumerate(self.sentences_info):
                if other_sentence_info.doc_id != sentence_info.doc_id:
                    similarity = cosine_similarity(current_embedding, all_embeddings[j])
                    cross_doc_candidates.append((j, similarity))

            # Keep top cross-document connections
            cross_doc_candidates.sort(key=lambda x: x[1], reverse=True)
            top_cross_connections = cross_doc_candidates[:self.cross_doc_k]

            self.cross_document_matrix[sentence_info.sentence_id] = [
                (self.sentences_info[idx].sentence_id, score)
                for idx, score in top_cross_connections
            ]

    def _classify_connection_type(self, from_sentence: SentenceInfo, to_sentence: SentenceInfo) -> str:
        """Classify the type of connection between two sentences"""
        if from_sentence.doc_id != to_sentence.doc_id:
            return 'cross_document'

        if from_sentence.paragraph_id == to_sentence.paragraph_id:
            return 'same_paragraph'

        paragraph_distance = abs(from_sentence.paragraph_id - to_sentence.paragraph_id)
        if paragraph_distance == 1:
            return 'neighboring_paragraph'
        else:
            return 'distant_paragraph'

    def _detailed_retrieve(self, query: str, top_k: int = 5) -> Tuple[List[str], List[TraversalStep]]:
        """Retrieve with detailed traversal tracking and two-phase reranking"""
        query_embedding = self.model.encode([query])

        # Find anchor sentences
        similarities = []
        for sentence_info in self.sentences_info:
            similarity = np.dot(query_embedding[0], sentence_info.embedding)
            similarities.append((sentence_info.sentence_id, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        top_anchors = similarities[:3]

        print(f"🎯 PHASE 1: Discovery via Semantic Graph Traversal")

        # Track detailed traversal
        all_traversal_steps = []
        visited = set()

        for anchor_sentence_id, anchor_score in top_anchors:
            if anchor_sentence_id in visited:
                continue

            anchor_sentence = self.sentences_info[anchor_sentence_id]

            # Start traversal from this anchor
            traversal_steps = self._detailed_traverse_from_anchor(
                anchor_sentence, anchor_score, visited
            )
            all_traversal_steps.extend(traversal_steps)

        print(f"Found {len(all_traversal_steps)} related sentences via graph traversal")

        # PHASE 2: Direct Query Re-ranking
        print(f"🎯 PHASE 2: Reranking by Direct Query Similarity")

        # Calculate direct query similarity for all discovered sentences
        reranked_steps = []
        for step in all_traversal_steps:
            # Get direct similarity to original query (not traversal-based score)
            direct_similarity = float(np.dot(query_embedding[0], step.sentence_info.embedding))

            # Create new step with direct query similarity as the relevance score
            reranked_step = TraversalStep(
                sentence_info=step.sentence_info,
                step_number=step.step_number,  # Keep original step number for visualization
                relevance_score=direct_similarity,  # Use direct query similarity
                connection_type=step.connection_type,
                distance_from_anchor=step.distance_from_anchor
            )
            reranked_steps.append(reranked_step)

        # Sort by direct query similarity (highest first)
        reranked_steps.sort(key=lambda x: x.relevance_score, reverse=True)

        print(f"Reranked {len(reranked_steps)} steps by direct query similarity")
        if reranked_steps:
            print(f"Top result similarity: {reranked_steps[0].relevance_score:.3f}")
            print(f"Lowest result similarity: {reranked_steps[-1].relevance_score:.3f}")

        # Deduplicate by text content while preserving new ranking
        seen_texts = set()
        unique_reranked_steps = []
        unique_texts = []

        for step in reranked_steps:
            text = step.sentence_info.text.strip()
            if text not in seen_texts:
                seen_texts.add(text)
                unique_reranked_steps.append(step)
                unique_texts.append(text)

                if len(unique_texts) >= top_k:
                    break

        # PHASE 3: Similarity Threshold Trimming
        print(f"🎯 PHASE 3: Similarity Threshold Trimming (threshold: {self.similarity_threshold})")

        # Filter by similarity threshold
        filtered_texts = []
        filtered_steps = []
        below_threshold_count = 0

        for i, step in enumerate(unique_reranked_steps):
            if step.relevance_score >= self.similarity_threshold:
                filtered_texts.append(unique_texts[i])
                filtered_steps.append(step)
            else:
                below_threshold_count += 1

        if below_threshold_count > 0:
            print(f"Trimmed {below_threshold_count} results below similarity threshold {self.similarity_threshold}")

        if filtered_texts:
            print(f"Final: {len(filtered_texts)} high-quality results (similarity range: {filtered_steps[0].relevance_score:.3f} - {filtered_steps[-1].relevance_score:.3f})")
        else:
            print(f"⚠️ No results above similarity threshold {self.similarity_threshold}, returning top result anyway")
            # Return at least one result if nothing passes threshold
            if unique_texts:
                filtered_texts = [unique_texts[0]]

        # Return both the filtered texts and ALL original traversal steps (for visualization)
        return filtered_texts, all_traversal_steps

    def _detailed_traverse_from_anchor(self, anchor_sentence: SentenceInfo,
                                     anchor_score: float, visited: set) -> List[TraversalStep]:
        """Perform detailed traversal from an anchor sentence"""
        traversal_steps = []
        queue = [(anchor_sentence.sentence_id, anchor_score, 0, anchor_sentence, None)]

        while queue:
            sentence_id, score, depth, current_sentence, previous_sentence = queue.pop(0)

            if sentence_id in visited or depth > self.traversal_depth:
                continue

            visited.add(sentence_id)

            # Determine connection type
            if depth == 0:
                connection_type = 'anchor'
            else:
                connection_type = self._classify_connection_type(previous_sentence, current_sentence)

            # Create traversal step
            step = TraversalStep(
                sentence_info=current_sentence,
                step_number=len(traversal_steps),
                relevance_score=score,
                connection_type=connection_type,
                distance_from_anchor=depth
            )
            traversal_steps.append(step)

            # Add connected sentences to queue
            if depth < self.traversal_depth:
                # Within-document connections
                doc_id = current_sentence.doc_id
                if doc_id in self.document_graphs and sentence_id in self.document_graphs[doc_id]:
                    connections = self.document_graphs[doc_id][sentence_id]

                    for connected_id, connection_score in zip(connections['indices'], connections['scores']):
                        if connected_id not in visited:
                            connected_sentence = self.sentences_info[connected_id]
                            new_score = score * 0.8 * connection_score
                            queue.append((connected_id, new_score, depth + 1, connected_sentence, current_sentence))

                # Cross-document connections
                if depth <= 1 and sentence_id in self.cross_document_matrix:
                    cross_connections = self.cross_document_matrix[sentence_id]

                    for connected_id, connection_score in cross_connections[:2]:
                        if connected_id not in visited:
                            connected_sentence = self.sentences_info[connected_id]
                            new_score = score * 0.6 * connection_score
                            queue.append((connected_id, new_score, depth + 1, connected_sentence, current_sentence))

        return traversal_steps

    def _analyze_traversal_patterns(self, traversal_steps: List[TraversalStep]) -> Dict:
        """Analyze patterns in traversal behavior"""
        if not traversal_steps:
            return {}

        connection_counts = defaultdict(int)
        document_distribution = defaultdict(int)

        for step in traversal_steps:
            connection_counts[step.connection_type] += 1
            document_distribution[step.sentence_info.doc_id] += 1

        total_steps = len(traversal_steps)

        # Calculate reranking improvement if we have query
        reranking_improvement = 0.0
        if hasattr(self, 'selected_question') and self.selected_question:
            # Compare original traversal order vs reranked order using cosine similarity
            query_embedding = self.model.encode([self.selected_question])[0]

            # Original order similarities (first 5 steps)
            original_similarities = []
            for step in traversal_steps[:5]:
                sim = cosine_similarity(query_embedding, step.sentence_info.embedding)
                original_similarities.append(sim)

            # Sort steps by relevance score (reranked order) and get top 5
            reranked_steps = sorted(traversal_steps, key=lambda x: x.relevance_score, reverse=True)[:5]
            reranked_similarities = [step.relevance_score for step in reranked_steps]

            # Calculate improvement
            if original_similarities and reranked_similarities:
                original_avg = np.mean(original_similarities)
                reranked_avg = np.mean(reranked_similarities)
                if original_avg > 0:
                    reranking_improvement = ((reranked_avg - original_avg) / original_avg) * 100

        analysis = {
            'total_steps': total_steps,
            'question': getattr(self, 'selected_question', 'Unknown'),
            'num_contexts': len(self.selected_contexts) if self.selected_contexts else 0,
            'connection_type_percentages': {
                conn_type: float((count / total_steps) * 100)
                for conn_type, count in connection_counts.items()
            },
            'document_distribution': dict(document_distribution),
            'cross_document_rate': float((connection_counts['cross_document'] / total_steps) * 100),
            'reranking_improvement_percent': float(reranking_improvement),
            'sliding_window_enabled': self.use_sliding_window,
            'graph_parameters': {
                'top_k_per_sentence': self.top_k_per_sentence,
                'cross_doc_k': self.cross_doc_k,
                'traversal_depth': self.traversal_depth,
                'similarity_threshold': self.similarity_threshold
            }
        }

        return analysis

# Convenience function for easy notebook usage
def create_rag_system(config=None) -> SemanticGraphRAG:
    """Create a RAG system with default or provided configuration"""
    if config is None:
        from .config import RAGConfig
        config = RAGConfig()

    return SemanticGraphRAG(
        top_k_per_sentence=config.top_k_per_sentence,
        cross_doc_k=config.cross_doc_k,
        traversal_depth=config.traversal_depth,
        use_sliding_window=config.use_sliding_window,
        num_contexts=config.num_contexts,
        similarity_threshold=config.similarity_threshold
    )