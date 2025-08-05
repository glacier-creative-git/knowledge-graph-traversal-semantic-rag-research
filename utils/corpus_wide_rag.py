"""
Corpus-Wide Semantic Graph RAG System
====================================

This implements the correct approach: semantic graph traversal across
an entire corpus rather than pre-selected documents.

The key insight: Let the traversal algorithm automatically discover
the most relevant content across thousands of contexts, not just 3-5.
"""

import re
import time
import nltk
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set
from sentence_transformers import SentenceTransformer
from .rag_system import cosine_similarity, SentenceInfo, TraversalStep


class CorpusWideSemanticRAG:
    """
    Semantic Graph RAG that operates on entire corpus rather than pre-selected documents
    """

    def __init__(self,
                 top_k_per_sentence: int = 20,
                 cross_doc_k: int = 15,  # Higher for corpus-wide
                 embedding_model: str = "all-MiniLM-L6-v2",
                 traversal_depth: int = 3,
                 use_sliding_window: bool = True,
                 similarity_threshold: float = 0.5,
                 max_corpus_size: int = 1000):

        self.model = SentenceTransformer(embedding_model)
        self.top_k_per_sentence = top_k_per_sentence
        self.cross_doc_k = cross_doc_k
        self.traversal_depth = traversal_depth
        self.use_sliding_window = use_sliding_window
        self.similarity_threshold = similarity_threshold
        self.max_corpus_size = max_corpus_size

        # Corpus-wide storage
        self.corpus_contexts: List[Dict] = []  # All unique contexts
        self.sentences_info: List[SentenceInfo] = []
        self.semantic_graph = {}  # Single graph across entire corpus
        self.similarity_matrices = {}  # Per-document matrices for visualization

        # Deduplication tracking
        self.context_hashes: Set[str] = set()
        self.duplicate_count = 0

    def ingest_corpus(self, squad_data, max_contexts: int = None) -> float:
        """
        Ingest entire SQuAD corpus with deduplication

        Args:
            squad_data: SQuAD dataset
            max_contexts: Maximum contexts to process (None for all)

        Returns:
            Ingestion time in seconds
        """
        start_time = time.time()

        if max_contexts is None:
            max_contexts = min(len(squad_data), self.max_corpus_size)

        print(f"🌍 CORPUS-WIDE INGESTION")
        print(f"📚 Processing up to {max_contexts} contexts from SQuAD dataset")
        print(f"🧮 Using cosine similarity for all calculations")
        print(
            f"🪟 Using {'3-sentence forward-looking sliding windows' if self.use_sliding_window else 'single sentence embeddings'}")

        # Reset storage
        self.corpus_contexts = []
        self.sentences_info = []
        self.semantic_graph = {}
        self.similarity_matrices = {}
        self.context_hashes = set()
        self.duplicate_count = 0

        # Process all contexts with deduplication
        processed_count = 0
        for item in squad_data:
            if processed_count >= max_contexts:
                break

            context_text = item['context'].strip()
            context_hash = hash(context_text)

            # Skip duplicates
            if context_hash in self.context_hashes:
                self.duplicate_count += 1
                continue

            self.context_hashes.add(context_hash)

            # Store unique context
            context_data = {
                'context': context_text,
                'question': item['question'],  # Store for reference
                'id': item['id'],
                'doc_id': processed_count,  # Sequential ID for unique contexts
                'title': f"Context {processed_count}: {item['id']}"
            }

            self.corpus_contexts.append(context_data)
            processed_count += 1

        print(f"✅ Deduplicated corpus: {processed_count} unique contexts (skipped {self.duplicate_count} duplicates)")

        # Analyze all contexts and build sentence-level representations
        print(f"🔍 Analyzing sentence structure across corpus...")
        self._analyze_corpus_structure()

        # Build corpus-wide semantic graph
        print(f"🕸️  Building corpus-wide semantic graph...")
        self._build_corpus_semantic_graph()

        # Build visualization matrices per document
        print(f"🎨 Building visualization matrices...")
        self._build_visualization_matrices()

        total_time = time.time() - start_time

        print(f"✅ Corpus ingestion complete!")
        print(f"   📊 {len(self.corpus_contexts)} unique contexts")
        print(f"   📝 {len(self.sentences_info)} total sentences")
        print(f"   🔗 {len(self.semantic_graph)} semantic connections")
        print(f"   ⏱️  Time: {total_time:.2f}s")

        return total_time

    def _analyze_corpus_structure(self):
        """Analyze sentence structure across entire corpus"""
        for doc_id, context_data in enumerate(self.corpus_contexts):
            text = context_data['context']
            context_id = context_data['id']

            # Extract paragraphs and sentences
            paragraphs = self._extract_paragraphs(text)

            for paragraph_id, paragraph_text in enumerate(paragraphs):
                sentences = nltk.sent_tokenize(paragraph_text)

                for sent_idx, sentence in enumerate(sentences):
                    # Calculate relative positions
                    total_sentences_in_doc = len(nltk.sent_tokenize(text))
                    position_in_doc = sent_idx / max(1, total_sentences_in_doc - 1)
                    position_in_paragraph = sent_idx / max(1, len(sentences) - 1)

                    # Create embedding
                    if self.use_sliding_window:
                        window_text = self._create_sliding_window(sentences, sent_idx)
                        embedding = self.model.encode([window_text])[0]
                    else:
                        embedding = self.model.encode([sentence])[0]

                    sentence_info = SentenceInfo(
                        doc_id=doc_id,
                        paragraph_id=paragraph_id,
                        sentence_id=len(self.sentences_info),  # Global sentence ID
                        text=sentence,
                        embedding=embedding,
                        position_in_doc=position_in_doc,
                        position_in_paragraph=position_in_paragraph,
                        source_context_id=context_id
                    )

                    self.sentences_info.append(sentence_info)

    def _build_corpus_semantic_graph(self):
        """Build semantic graph across entire corpus"""
        print(f"   Building connections for {len(self.sentences_info)} sentences...")

        # Get all embeddings
        all_embeddings = np.array([s.embedding for s in self.sentences_info])

        # Build semantic graph with corpus-wide connections
        for i, sentence_info in enumerate(self.sentences_info):
            current_embedding = all_embeddings[i]

            # Calculate similarities to ALL other sentences in corpus
            similarities = []
            for j, other_sentence_info in enumerate(self.sentences_info):
                if i != j:  # Skip self
                    similarity = cosine_similarity(current_embedding, all_embeddings[j])
                    similarities.append((j, other_sentence_info.doc_id, similarity))

            # Sort by similarity
            similarities.sort(key=lambda x: x[2], reverse=True)

            # Keep top-K connections overall
            top_connections = similarities[:self.top_k_per_sentence]

            # Separate within-document and cross-document connections
            within_doc_connections = []
            cross_doc_connections = []

            for sent_idx, other_doc_id, sim_score in top_connections:
                if other_doc_id == sentence_info.doc_id:
                    within_doc_connections.append((sent_idx, sim_score))
                else:
                    cross_doc_connections.append((sent_idx, sim_score))

            # Store in semantic graph
            self.semantic_graph[sentence_info.sentence_id] = {
                'within_doc': within_doc_connections,
                'cross_doc': cross_doc_connections[:self.cross_doc_k],  # Limit cross-doc
                'all_connections': [(idx, score) for idx, _, score in top_connections]
            }

    def _build_visualization_matrices(self):
        """Build per-document similarity matrices for visualization"""
        documents_sentences = defaultdict(list)

        # Group sentences by document
        for sentence_info in self.sentences_info:
            documents_sentences[sentence_info.doc_id].append(sentence_info)

        # Build similarity matrix for each document
        for doc_id, doc_sentences in documents_sentences.items():
            if len(doc_sentences) < 2:
                if len(doc_sentences) == 1:
                    self.similarity_matrices[doc_id] = np.array([[1.0]])
                continue

            # Get embeddings for this document
            embeddings = np.array([s.embedding for s in doc_sentences])

            # Build cosine similarity matrix
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            normalized = embeddings / norms
            similarity_matrix = np.dot(normalized, normalized.T)

            self.similarity_matrices[doc_id] = similarity_matrix

    def retrieve_from_corpus(self, query: str, top_k: int = 10) -> Tuple[List[str], List[TraversalStep], Dict]:
        """
        Retrieve from entire corpus using semantic graph traversal

        Args:
            query: Query string
            top_k: Maximum results to return (before threshold filtering)

        Returns:
            Tuple of (retrieved_texts, traversal_steps, analysis)
        """
        print(f"\n🌍 CORPUS-WIDE RETRIEVAL")
        print(f"🔍 Query: {query}")
        print(f"📊 Searching across {len(self.sentences_info)} sentences in {len(self.corpus_contexts)} contexts")

        query_embedding = self.model.encode([query])[0]

        # PHASE 1: Find anchor sentences across entire corpus
        print(f"🎯 PHASE 1: Finding anchor sentences across entire corpus")

        anchor_similarities = []
        for sentence_info in self.sentences_info:
            similarity = cosine_similarity(query_embedding, sentence_info.embedding)
            anchor_similarities.append((sentence_info.sentence_id, similarity))

        anchor_similarities.sort(key=lambda x: x[1], reverse=True)
        top_anchors = anchor_similarities[:5]  # More anchors for corpus-wide search

        print(f"Found {len(top_anchors)} anchor sentences")
        for i, (sent_id, score) in enumerate(top_anchors):
            doc_id = self.sentences_info[sent_id].doc_id
            print(f"   Anchor {i + 1}: similarity {score:.3f} (context {doc_id})")

        # PHASE 2: Semantic graph traversal from anchors
        print(f"🕸️  PHASE 2: Semantic graph traversal across corpus")

        all_traversal_steps = []
        visited = set()

        for anchor_sentence_id, anchor_score in top_anchors:
            if anchor_sentence_id in visited:
                continue

            anchor_sentence = self.sentences_info[anchor_sentence_id]
            traversal_steps = self._traverse_from_anchor(anchor_sentence, anchor_score, visited)
            all_traversal_steps.extend(traversal_steps)

        print(f"Discovered {len(all_traversal_steps)} sentences via corpus-wide traversal")

        # PHASE 3: Rerank by direct query similarity
        print(f"🎯 PHASE 3: Reranking by direct query cosine similarity")

        reranked_steps = []
        for step in all_traversal_steps:
            direct_similarity = cosine_similarity(query_embedding, step.sentence_info.embedding)

            reranked_step = TraversalStep(
                sentence_info=step.sentence_info,
                step_number=step.step_number,
                relevance_score=direct_similarity,
                connection_type=step.connection_type,
                distance_from_anchor=step.distance_from_anchor
            )
            reranked_steps.append(reranked_step)

        reranked_steps.sort(key=lambda x: x.relevance_score, reverse=True)

        if reranked_steps:
            print(
                f"Reranked results: similarity range {reranked_steps[0].relevance_score:.3f} - {reranked_steps[-1].relevance_score:.3f}")

        # PHASE 4: Similarity threshold filtering
        print(f"🎯 PHASE 4: Similarity threshold filtering (threshold: {self.similarity_threshold})")

        # Deduplicate and filter
        seen_texts = set()
        filtered_steps = []
        filtered_texts = []
        below_threshold = 0

        for step in reranked_steps:
            text = step.sentence_info.text.strip()
            if text in seen_texts:
                continue

            if step.relevance_score >= self.similarity_threshold:
                seen_texts.add(text)
                filtered_steps.append(step)
                filtered_texts.append(text)

                if len(filtered_texts) >= top_k:
                    break
            else:
                below_threshold += 1

        if below_threshold > 0:
            print(f"Filtered out {below_threshold} results below threshold {self.similarity_threshold}")

        if not filtered_texts and reranked_steps:
            # Fallback: return top result even if below threshold
            filtered_texts = [reranked_steps[0].sentence_info.text]
            filtered_steps = [reranked_steps[0]]
            print(
                f"⚠️ No results above threshold, returning top result (similarity: {reranked_steps[0].relevance_score:.3f})")

        print(f"✅ Final results: {len(filtered_texts)} high-quality sentences from corpus-wide search")

        # Analyze results
        analysis = self._analyze_corpus_traversal(all_traversal_steps, query)

        return filtered_texts, all_traversal_steps, analysis

    def _traverse_from_anchor(self, anchor_sentence: SentenceInfo, anchor_score: float, visited: set) -> List[
        TraversalStep]:
        """Traverse semantic graph from anchor sentence"""
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
            elif previous_sentence and current_sentence.doc_id != previous_sentence.doc_id:
                connection_type = 'cross_document'
            elif previous_sentence and current_sentence.doc_id == previous_sentence.doc_id:
                if current_sentence.paragraph_id == previous_sentence.paragraph_id:
                    connection_type = 'same_paragraph'
                elif abs(current_sentence.paragraph_id - previous_sentence.paragraph_id) == 1:
                    connection_type = 'neighboring_paragraph'
                else:
                    connection_type = 'distant_paragraph'
            else:
                connection_type = 'unknown'

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
            if depth < self.traversal_depth and sentence_id in self.semantic_graph:
                connections = self.semantic_graph[sentence_id]

                # Add all connections (both within-doc and cross-doc)
                for connected_id, connection_score in connections['all_connections']:
                    if connected_id not in visited:
                        connected_sentence = self.sentences_info[connected_id]
                        new_score = score * 0.8 * connection_score
                        queue.append((connected_id, new_score, depth + 1, connected_sentence, current_sentence))

        return traversal_steps

    def _analyze_corpus_traversal(self, traversal_steps: List[TraversalStep], query: str) -> Dict:
        """Analyze corpus-wide traversal patterns"""
        if not traversal_steps:
            return {}

        connection_counts = defaultdict(int)
        document_distribution = defaultdict(int)
        context_coverage = set()

        for step in traversal_steps:
            connection_counts[step.connection_type] += 1
            document_distribution[step.sentence_info.doc_id] += 1
            context_coverage.add(step.sentence_info.doc_id)

        total_steps = len(traversal_steps)
        unique_contexts_found = len(context_coverage)

        analysis = {
            'total_steps': total_steps,
            'question': query,
            'corpus_size': len(self.corpus_contexts),
            'unique_contexts_discovered': unique_contexts_found,
            'context_coverage_rate': float(unique_contexts_found / len(self.corpus_contexts) * 100),
            'connection_type_percentages': {
                conn_type: float((count / total_steps) * 100)
                for conn_type, count in connection_counts.items()
            },
            'document_distribution': dict(document_distribution),
            'cross_document_rate': float((connection_counts['cross_document'] / total_steps) * 100),
            'sliding_window_enabled': self.use_sliding_window,
            'similarity_threshold': self.similarity_threshold,
            'duplicate_contexts_removed': self.duplicate_count,
            'deduplication_rate': float(self.duplicate_count / (len(self.corpus_contexts) + self.duplicate_count) * 100)
        }

        return analysis

    def _extract_paragraphs(self, text: str) -> List[str]:
        """Extract paragraphs from text"""
        paragraphs = re.split(r'\n\s*\n|\r\n\s*\r\n|\.[\s]{3,}', text.strip())
        paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 10]

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


# Convenience function
def create_corpus_wide_rag(config=None) -> CorpusWideSemanticRAG:
    """Create corpus-wide RAG system"""
    if config is None:
        from .config import RAGConfig
        config = RAGConfig()

    return CorpusWideSemanticRAG(
        top_k_per_sentence=config.top_k_per_sentence,
        cross_doc_k=config.cross_doc_k,
        traversal_depth=config.traversal_depth,
        use_sliding_window=config.use_sliding_window,
        similarity_threshold=config.similarity_threshold,
        max_corpus_size=1000
    )