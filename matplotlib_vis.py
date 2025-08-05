"""
SQuAD Semantic Graph 2D Heatmap Visualizer - Research Publication Ready
========================================================================

This script creates beautiful 2D semantic heatmap visualizations using real SQuAD 2.0 data
and sophisticated semantic graph traversal, adapted from plotly_vis.py.

FIXES APPLIED:
- Changed num_contexts from 3 to 5 (matching plotly_vis.py)
- Fixed dummy matrix creation issue
- Added better debugging/logging
- Improved document selection randomness
- Enhanced sentence counting verification
"""

import os
import numpy as np
import pandas as pd
import nltk
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict, Optional
import re
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass
from collections import defaultdict
import random
import time
from datasets import load_dataset

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

@dataclass
class SentenceInfo:
    """Detailed information about a sentence in the corpus"""
    doc_id: int
    paragraph_id: int
    sentence_id: int
    text: str
    embedding: np.ndarray
    position_in_doc: float  # 0.0 to 1.0, relative position in document
    position_in_paragraph: float  # 0.0 to 1.0, relative position in paragraph
    source_context_id: str  # Original SQuAD context ID for traceability

@dataclass
class TraversalStep:
    """Information about a single step in semantic graph traversal"""
    sentence_info: SentenceInfo
    step_number: int
    relevance_score: float
    connection_type: str  # 'anchor', 'same_paragraph', 'neighboring_paragraph', 'distant_paragraph', 'cross_document'
    distance_from_anchor: int  # How many steps from the original anchor


class SQuADSemanticGraph2DVisualizer:
    """Enhanced semantic RAG system with SQuAD 2.0 integration and 2D visualizations"""

    def __init__(self, top_k_per_sentence: int = 20, cross_doc_k: int = 10,
                 embedding_model: str = "all-MiniLM-L6-v2", traversal_depth: int = 3,
                 use_sliding_window: bool = False, num_contexts: int = 5,
                 figure_size: Tuple[int, int] = (24, 8), dpi: int = 150):
        self.model = SentenceTransformer(embedding_model)
        self.top_k_per_sentence = top_k_per_sentence
        self.cross_doc_k = cross_doc_k
        self.traversal_depth = traversal_depth
        self.num_contexts = num_contexts
        self.figure_size = figure_size
        self.dpi = dpi

        # Enhanced storage with paragraph awareness
        self.sentences_info: List[SentenceInfo] = []
        self.document_graphs = {}
        self.cross_document_matrix = {}
        self.global_sentence_index = {}
        self.squad_data = None
        self.selected_question = None
        self.selected_contexts = []
        self.similarity_matrices = []

        self.use_sliding_window = use_sliding_window

    def load_squad_data(self, num_samples: int = 1000):
        """Load SQuAD 2.0 dataset"""
        print(f"📚 Loading SQuAD 2.0 dataset ({num_samples} samples)...")
        try:
            # Load SQuAD 2.0 validation set
            self.squad_data = load_dataset("squad_v2", split=f"validation[:{num_samples}]")
            print(f"✅ Loaded {len(self.squad_data)} SQuAD 2.0 samples")
            return True
        except Exception as e:
            print(f"❌ Failed to load SQuAD 2.0: {e}")
            print("🔄 Falling back to SQuAD 1.1...")
            try:
                self.squad_data = load_dataset("squad", split=f"validation[:{num_samples}]")
                print(f"✅ Loaded {len(self.squad_data)} SQuAD 1.1 samples")
                return True
            except Exception as e2:
                print(f"❌ Failed to load any SQuAD dataset: {e2}")
                return False

    def select_random_question_with_contexts(self, min_context_length: int = 400) -> Tuple[str, List[Dict]]:  # FIXED: Reduced min length
        """
        Select a random question and gather multiple related contexts
        to create a multi-document scenario for semantic graph analysis
        """
        if not self.squad_data:
            raise ValueError("SQuAD data not loaded. Call load_squad_data() first.")

        # Filter for substantial contexts
        substantial_samples = [
            sample for sample in self.squad_data
            if len(sample['context']) >= min_context_length
        ]

        if not substantial_samples:
            print(f"⚠️ No contexts found with minimum length {min_context_length}, trying with lower threshold...")
            substantial_samples = [
                sample for sample in self.squad_data
                if len(sample['context']) >= 200  # Fallback to shorter contexts
            ]

        if not substantial_samples:
            raise ValueError(f"No contexts found even with minimum length 200")

        # FIXED: Add randomness by shuffling and random seed
        random.shuffle(substantial_samples)

        # Select primary sample
        primary_sample = random.choice(substantial_samples)
        self.selected_question = primary_sample['question']

        print(f"🎯 Selected Question: {self.selected_question}")
        print(f"📄 Primary Context Length: {len(primary_sample['context'])} chars")

        # ADDED: Count sentences in primary context for debugging
        primary_sentences = nltk.sent_tokenize(primary_sample['context'])
        print(f"📝 Primary Context Sentences: {len(primary_sentences)}")

        # Get the primary context
        primary_context = {
            'context': primary_sample['context'],
            'question': primary_sample['question'],
            'id': primary_sample['id'],
            'title': f"Document 1: {primary_sample['id']}",
            'answers': primary_sample.get('answers', {'text': [], 'answer_start': []})
        }

        # Find additional related contexts using keyword overlap
        question_keywords = set(self.selected_question.lower().split())

        # Score other contexts by keyword overlap with the question
        context_scores = []
        seen_contexts = {primary_sample['context'].strip()}

        for sample in substantial_samples:
            if sample['id'] == primary_sample['id']:
                continue

            # Skip if we've already seen this exact context text
            context_text = sample['context'].strip()
            if context_text in seen_contexts:
                continue

            context_keywords = set(sample['context'].lower().split())
            overlap_score = len(question_keywords.intersection(context_keywords))

            # Also consider question similarity
            question_overlap = len(set(sample['question'].lower().split()).intersection(question_keywords))
            total_score = overlap_score + (question_overlap * 2)

            if total_score > 0:
                context_scores.append((total_score, sample))
                seen_contexts.add(context_text)

        # Sort by relevance and take top contexts
        context_scores.sort(key=lambda x: x[0], reverse=True)

        # Build final context list
        contexts = [primary_context]

        for i, (score, sample) in enumerate(context_scores[:self.num_contexts-1]):
            # ADDED: Count sentences for debugging
            sample_sentences = nltk.sent_tokenize(sample['context'])

            context = {
                'context': sample['context'],
                'question': sample['question'],
                'id': sample['id'],
                'title': f"Document {i+2}: {sample['id']}",
                'answers': sample.get('answers', {'text': [], 'answer_start': []})
            }
            contexts.append(context)
            print(f"📄 Added Related Context: {len(sample['context'])} chars, {len(sample_sentences)} sentences (relevance score: {score})")

        self.selected_contexts = contexts
        print(f"\n✅ Created multi-document dataset with {len(contexts)} contexts")
        print(f"📊 Total text length: {sum(len(c['context']) for c in contexts):,} characters")

        # ADDED: Print sentence count for each document
        total_sentences = 0
        for i, context in enumerate(contexts):
            sentences = nltk.sent_tokenize(context['context'])
            total_sentences += len(sentences)
            print(f"📝 Document {i+1}: {len(sentences)} sentences")
        print(f"📝 Total sentences across all documents: {total_sentences}")

        return self.selected_question, contexts

    def create_focused_context_set(self, topic_keywords: List[str] = None) -> Tuple[str, List[Dict]]:
        """Create a focused set of contexts around specific topics"""
        if not self.squad_data:
            raise ValueError("SQuAD data not loaded. Call load_squad_data() first.")

        if topic_keywords is None:
            topic_keywords = ['science', 'technology', 'computer', 'machine', 'learning',
                            'artificial', 'intelligence', 'algorithm', 'data', 'research']

        print(f"🔍 Searching for contexts containing: {topic_keywords}")

        # Find contexts that contain the topic keywords
        topic_contexts = []
        seen_contexts = set()

        for sample in self.squad_data:
            context_lower = sample['context'].lower()
            question_lower = sample['question'].lower()
            context_text = sample['context'].strip()

            # Skip duplicates
            if context_text in seen_contexts:
                continue

            # Count keyword matches
            keyword_matches = sum(1 for keyword in topic_keywords
                                if keyword in context_lower or keyword in question_lower)

            if keyword_matches >= 2 and len(sample['context']) >= 200:  # FIXED: Reduced minimum length
                topic_contexts.append((keyword_matches, sample))
                seen_contexts.add(context_text)

        if not topic_contexts:
            print("⚠️ No focused contexts found, falling back to random selection")
            return self.select_random_question_with_contexts()

        # Sort by keyword relevance and take the best ones
        topic_contexts.sort(key=lambda x: x[0], reverse=True)
        selected_samples = [sample for _, sample in topic_contexts[:self.num_contexts]]

        # Use the most relevant question
        primary_sample = selected_samples[0]
        self.selected_question = primary_sample['question']

        print(f"🎯 Selected Focused Question: {self.selected_question}")

        # Convert to our format
        contexts = []
        total_sentences = 0
        for i, sample in enumerate(selected_samples):
            sentences = nltk.sent_tokenize(sample['context'])
            total_sentences += len(sentences)

            context = {
                'context': sample['context'],
                'question': sample['question'],
                'id': sample['id'],
                'title': f"Document {i+1}: {sample['id']}",
                'answers': sample.get('answers', {'text': [], 'answer_start': []})
            }
            contexts.append(context)
            print(f"📄 Context {i+1}: {len(sample['context'])} chars, {len(sentences)} sentences")

        self.selected_contexts = contexts
        print(f"\n✅ Created focused dataset with {len(contexts)} contexts")
        print(f"📝 Total sentences across all documents: {total_sentences}")
        return self.selected_question, contexts

    def _extract_paragraphs(self, text: str) -> List[str]:
        """Extract paragraphs from text, handling various paragraph separators"""
        # ENHANCED: More lenient paragraph extraction
        paragraphs = re.split(r'\n\s*\n|\r\n\s*\r\n|\.[\s]{3,}', text.strip())
        paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 10]  # FIXED: Reduced minimum paragraph length

        # If we get very few paragraphs, treat the whole text as one paragraph
        if len(paragraphs) <= 1:
            paragraphs = [text.strip()]

        return paragraphs

    def _create_sliding_window(self, sentences: List[str], anchor_idx: int) -> str:
        """Create 3-sentence forward-looking sliding window starting at anchor_idx"""
        total_sentences = len(sentences)

        if total_sentences == 1:
            return sentences[0]
        elif total_sentences == 2:
            return ' '.join(sentences)
        elif anchor_idx >= total_sentences - 2:
            # Near the end - use last 2 sentences
            return ' '.join(sentences[-2:])
        else:
            # Forward-looking: anchor + next 2 sentences (or as many as available)
            end_idx = min(anchor_idx + 3, total_sentences)
            return ' '.join(sentences[anchor_idx:end_idx])

    def _analyze_document_structure(self, doc_id: int, text: str, source_context_id: str = "") -> List[SentenceInfo]:
        """Analyze document structure with paragraph awareness"""
        paragraphs = self._extract_paragraphs(text)
        sentence_infos = []
        global_sentence_id = 0

        print(f"🔍 Document {doc_id} ({source_context_id}): {len(paragraphs)} paragraphs")  # ADDED: Debug info

        for paragraph_id, paragraph_text in enumerate(paragraphs):
            sentences = nltk.sent_tokenize(paragraph_text)
            print(f"   Paragraph {paragraph_id}: {len(sentences)} sentences")  # ADDED: Debug info

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

        print(f"   Total sentences for document {doc_id}: {len(sentence_infos)}")  # ADDED: Debug info
        return sentence_infos

    def analyze_squad_question(self, question: str = None, contexts: List[Dict] = None) -> Tuple[List[str], List[TraversalStep], Dict]:
        """Main analysis function that processes SQuAD data and performs semantic graph traversal"""
        # Load SQuAD data if not already loaded
        if not self.squad_data:
            if not self.load_squad_data():
                raise ValueError("Failed to load SQuAD dataset")

        # Select question and contexts if not provided
        if question is None or contexts is None:
            question, contexts = self.select_random_question_with_contexts()

        print(f"\n🔍 Analyzing Question: '{question}'")
        print(f"📚 Processing {len(contexts)} contexts...")

        # Ingest the contexts
        start_time = time.time()
        ingest_time = self._ingest_squad_contexts(contexts)
        print(f"⚡ Ingestion completed in {ingest_time:.2f} seconds")

        # Perform detailed retrieval and analysis
        retrieved_texts, traversal_steps = self._detailed_retrieve(question, top_k=10)

        print(f"\n🎯 Semantic graph traversal complete!")
        print(f"Retrieved {len(retrieved_texts)} texts from {len(traversal_steps)} discovered sentences")

        # Analyze traversal patterns
        analysis = self._analyze_traversal_patterns(traversal_steps)

        return retrieved_texts, traversal_steps, analysis

    def _ingest_squad_contexts(self, contexts: List[Dict]) -> float:
        """Ingest SQuAD contexts with full paragraph structure analysis"""
        start_time = time.time()

        # Reset storage
        self.sentences_info = []
        self.document_graphs = {}
        self.cross_document_matrix = {}
        self.global_sentence_index = {}
        self.similarity_matrices = []

        print(f"🔄 Processing {len(contexts)} contexts...")  # ADDED: Debug info

        # Process each context as a separate document
        for doc_id, context_data in enumerate(contexts):
            text = context_data['context']
            context_id = context_data['id']

            print(f"\n📄 Processing Document {doc_id} ({context_id})...")  # ADDED: Debug info

            # Analyze document structure
            doc_sentence_infos = self._analyze_document_structure(doc_id, text, context_id)

            # Add to global sentence info
            for sentence_info in doc_sentence_infos:
                sentence_info.sentence_id = len(self.sentences_info)
                self.sentences_info.append(sentence_info)

                # Add to global index
                sentence_hash = hash(sentence_info.text.strip())
                self.global_sentence_index[sentence_hash] = len(self.sentences_info) - 1

        print(f"\n📊 Processed {len(contexts)} contexts into {len(self.sentences_info)} sentences")

        # Build similarity matrices for each document and visualization
        self._build_document_graphs()
        self._build_cross_document_connections()

        return time.time() - start_time

    def _build_document_graphs(self):
        """Build semantic graphs for each document and create similarity matrices for visualization"""
        documents_sentences = defaultdict(list)

        # Group sentences by document
        for sentence_info in self.sentences_info:
            documents_sentences[sentence_info.doc_id].append(sentence_info)

        print(f"🔨 Building graphs for {len(documents_sentences)} documents...")  # ADDED: Debug info

        # Build graph for each document
        for doc_id, doc_sentences in documents_sentences.items():
            if len(doc_sentences) < 2:
                print(f"⚠️ Document {doc_id} has only {len(doc_sentences)} sentence(s), skipping graph creation")
                # FIXED: Still create a matrix even for single sentence documents
                if len(doc_sentences) == 1:
                    self.similarity_matrices.append(np.array([[1.0]]))  # 1x1 matrix
                continue

            print(f"📊 Document {doc_id}: Building similarity matrix for {len(doc_sentences)} sentences")

            # Get embeddings for this document
            embeddings = np.array([s.embedding for s in doc_sentences])

            # Build similarity matrix
            similarity_matrix = np.dot(embeddings, embeddings.T)
            self.similarity_matrices.append(similarity_matrix)

            print(f"   Created {similarity_matrix.shape[0]}x{similarity_matrix.shape[1]} similarity matrix")

            # Build sparse graph
            sparse_matrix = {}
            for i, sentence_info in enumerate(doc_sentences):
                similarities = similarity_matrix[i]

                # Get top-K most similar sentences in this document
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

        print(f"✅ Created {len(self.similarity_matrices)} similarity matrices")

    def _build_cross_document_connections(self):
        """Build cross-document semantic connections"""
        if len(self.sentences_info) < 2:
            return

        # Get all embeddings
        all_embeddings = np.array([s.embedding for s in self.sentences_info])

        # Build cross-document connections
        for i, sentence_info in enumerate(self.sentences_info):
            # Calculate similarities to all other sentences
            similarities = np.dot(all_embeddings[i:i+1], all_embeddings.T)[0]

            # Filter out same-document connections
            cross_doc_candidates = []
            for j, other_sentence_info in enumerate(self.sentences_info):
                if other_sentence_info.doc_id != sentence_info.doc_id:
                    cross_doc_candidates.append((j, similarities[j]))

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
        """Retrieve with detailed traversal tracking"""
        query_embedding = self.model.encode([query])

        # Find anchor sentences
        similarities = []
        for sentence_info in self.sentences_info:
            similarity = np.dot(query_embedding[0], sentence_info.embedding)
            similarities.append((sentence_info.sentence_id, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        top_anchors = similarities[:3]

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

        # Sort by relevance score and return top-K
        all_traversal_steps.sort(key=lambda x: x.relevance_score, reverse=True)
        retrieved_texts = [step.sentence_info.text for step in all_traversal_steps[:top_k]]

        return retrieved_texts, all_traversal_steps

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

        analysis = {
            'total_steps': total_steps,
            'question': self.selected_question,
            'num_contexts': len(self.selected_contexts),
            'connection_type_percentages': {
                conn_type: float((count / total_steps) * 100)
                for conn_type, count in connection_counts.items()
            },
            'document_distribution': dict(document_distribution),
            'cross_document_rate': float((connection_counts['cross_document'] / total_steps) * 100),
        }

        return analysis

    def create_squad_traversal_visualization(self, question: str, traversal_steps: List[TraversalStep], max_steps: int = 15) -> plt.Figure:
        """Create 2D heatmap visualization of SQuAD semantic graph traversal"""

        if not self.similarity_matrices:
            print("⚠️ No similarity matrices available for visualization")
            return plt.Figure()

        # FIXED: Only show documents that were actually traversed
        traversed_doc_ids = set(step.sentence_info.doc_id for step in traversal_steps)
        traversed_doc_ids = sorted(traversed_doc_ids)  # Sort for consistent ordering

        # FIXED: Order documents by traversal order (left to right)
        doc_traversal_order = []
        seen_docs = set()
        for step in traversal_steps:
            doc_id = step.sentence_info.doc_id
            if doc_id not in seen_docs:
                doc_traversal_order.append(doc_id)
                seen_docs.add(doc_id)

        num_docs = len(doc_traversal_order)
        print(f"🎨 Creating visualization for {num_docs} TRAVERSED documents (out of {len(self.selected_contexts)} total)")
        print(f"📍 Traversal order (left→right): {doc_traversal_order}")

        # Create figure with white background for publication
        fig = plt.figure(figsize=(6 * num_docs, 8), facecolor='white', dpi=self.dpi)  # Dynamic width

        # Create grid layout - space at top for horizontal colorbar only
        gs = gridspec.GridSpec(2, num_docs, figure=fig,
                               height_ratios=[0.08, 1],
                               hspace=0.05, wspace=0.15)

        # Create a horizontal colorbar at the top
        cbar_ax = fig.add_subplot(gs[0, :])

        axes = []
        heatmap_extents = []

        # Create heatmaps ONLY for traversed documents in traversal order
        for plot_position, original_doc_id in enumerate(doc_traversal_order):
            ax = fig.add_subplot(gs[1, plot_position])
            axes.append(ax)

            # Get the similarity matrix for this document
            similarity_matrix = self.similarity_matrices[original_doc_id]
            print(f"📊 Plot position {plot_position}: Document {original_doc_id} using {similarity_matrix.shape[0]}x{similarity_matrix.shape[1]} matrix")

            # Create beautiful heatmap using the research proposal colors
            im = ax.imshow(similarity_matrix,
                          cmap='RdYlBu_r',  # Matching chunk_benchmark.py colors exactly
                          aspect='equal',
                          vmin=0, vmax=1,
                          interpolation='nearest')

            # FIXED: Clean up document titles
            context_title = f"Document {plot_position + 1}"
            if original_doc_id < len(self.selected_contexts):
                context_id = self.selected_contexts[original_doc_id]['id'][-8:]  # Last 8 chars of ID
                context_title += f"\n({context_id})"

            ax.set_title(context_title, fontsize=14, fontweight='bold', pad=15)
            ax.set_xlabel('Sentence Index', fontsize=12)
            ax.set_ylabel('Sentence Index', fontsize=12)

            # Set proper ticks
            n_sentences = similarity_matrix.shape[0]
            max_ticks = min(n_sentences, 10)  # Limit ticks for readability
            if n_sentences <= 10:
                ax.set_xticks(range(n_sentences))
                ax.set_yticks(range(n_sentences))
                ax.set_xticklabels(range(n_sentences))
                ax.set_yticklabels(range(n_sentences))
            else:
                tick_positions = np.linspace(0, n_sentences-1, max_ticks, dtype=int)
                ax.set_xticks(tick_positions)
                ax.set_yticks(tick_positions)
                ax.set_xticklabels(tick_positions)
                ax.set_yticklabels(tick_positions)

            # Store extent information for drawing lines
            bbox = ax.get_position()
            heatmap_extents.append({
                'ax': ax,
                'bbox': bbox,
                'original_doc_id': original_doc_id,  # Keep track of original doc ID
                'plot_position': plot_position,      # Position in the plot
                'matrix_size': n_sentences
            })

        # Add horizontal colorbar at the top
        cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Similarity Score', fontsize=12, fontweight='bold')
        cbar_ax.xaxis.set_label_position('top')

        # Draw traversal path
        self._draw_squad_traversal_path(fig, traversal_steps, heatmap_extents, doc_traversal_order, max_steps)

        # Add legend for traversal path
        self._add_squad_traversal_legend(fig)

        return fig

    def _draw_squad_traversal_path(self, fig: plt.Figure, traversal_steps: List[TraversalStep],
                                   heatmap_extents: List[Dict], doc_traversal_order: List[int], max_steps: int = 15):
        """Draw the traversal path as lines between and within heatmaps for SQuAD data"""

        if len(traversal_steps) < 2:
            return

        # Create mapping from original doc_id to plot position
        doc_id_to_plot_position = {doc_id: i for i, doc_id in enumerate(doc_traversal_order)}

        # Group steps by document for proper sentence indexing
        doc_sentence_counts = defaultdict(int)
        for sentence_info in self.sentences_info:
            doc_sentence_counts[sentence_info.doc_id] += 1

        # Map global sentence IDs to local document sentence IDs
        doc_sentence_mapping = {}
        for doc_id in doc_sentence_counts:
            doc_sentences = [s for s in self.sentences_info if s.doc_id == doc_id]
            doc_sentence_mapping[doc_id] = {s.sentence_id: i for i, s in enumerate(doc_sentences)}

        # FIXED: Show more steps and filter properly
        print(f"🎨 Drawing traversal path for {min(len(traversal_steps), max_steps)} steps (out of {len(traversal_steps)} total)")

        # Draw step markers and connections
        drawn_steps = []
        steps_shown = 0

        for i, step in enumerate(traversal_steps):
            if steps_shown >= max_steps:
                break

            original_doc_id = step.sentence_info.doc_id
            global_sent_id = step.sentence_info.sentence_id

            # Skip if document wasn't traversed (not in our plot)
            if original_doc_id not in doc_id_to_plot_position:
                continue

            plot_position = doc_id_to_plot_position[original_doc_id]

            # Get local sentence ID within document
            if original_doc_id in doc_sentence_mapping and global_sent_id in doc_sentence_mapping[original_doc_id]:
                local_sent_id = doc_sentence_mapping[original_doc_id][global_sent_id]
            else:
                continue

            # Find the corresponding heatmap info
            heatmap_info = None
            for extent in heatmap_extents:
                if extent['original_doc_id'] == original_doc_id:
                    heatmap_info = extent
                    break

            if not heatmap_info:
                continue

            # Ensure sentence ID is within matrix bounds
            matrix_size = heatmap_info['matrix_size']
            if local_sent_id >= matrix_size:
                continue

            ax = heatmap_info['ax']

            # Draw step marker on diagonal (sentence relates to itself)
            row, col = local_sent_id, local_sent_id

            if step.connection_type == 'anchor':
                marker_color = 'lightgreen'
                marker_size = 15
                edge_color = 'black'
                edge_width = 2
            else:
                # Use relevance score to fade from green to white
                # Normalize relevance score (assuming it's between 0 and 1)
                relevance = max(0, min(1, step.relevance_score))
                # Interpolate between white (1,1,1) and light green (0.5,1,0.5)
                white = np.array([1.0, 1.0, 1.0])
                light_green = np.array([0.5, 1.0, 0.5])
                marker_color = white + relevance * (light_green - white)  # Fade from white to green
                marker_size = 8 + (relevance * 7)  # Size 8-15 based on relevance
                edge_color = 'black' if relevance > 0.7 else 'gray'
                edge_width = 1 + (relevance * 1)  # Width 1-2 based on relevance

            # Use scatter plot for better alignment
            ax.scatter([col], [row], s=marker_size**2, c=marker_color,
                      edgecolors=edge_color, linewidths=edge_width, zorder=10)

            # Add step number text
            ax.text(col, row, str(steps_shown),
                   ha='center', va='center',
                   fontsize=10, fontweight='bold', color='black',
                   zorder=11)

            drawn_steps.append((step, original_doc_id, row, col, ax, plot_position))
            steps_shown += 1

        print(f"✅ Successfully drew {len(drawn_steps)} step markers")

        # Draw connections between steps
        connections_drawn = 0
        for i in range(len(drawn_steps) - 1):
            current_step, current_orig_doc, current_row, current_col, current_ax, current_plot_pos = drawn_steps[i]
            next_step, next_orig_doc, next_row, next_col, next_ax, next_plot_pos = drawn_steps[i + 1]

            # Explicitly check if crossing documents (more robust)
            is_cross_document = (current_orig_doc != next_orig_doc)

            if is_cross_document:
                line_color = 'black'
                line_style = '--'  # Dashed line for cross-document
                line_width = 1.5
                alpha = 0.7
                connection_label = "Cross-document"
                print(f"🔗 Cross-document line: Doc {current_orig_doc} → Doc {next_orig_doc}")
            else:
                line_color = 'black'
                line_style = '-'  # Solid line for within-document
                line_width = 1.5
                alpha = 0.7
                connection_label = "Within-document"

            # Draw connection line
            if current_orig_doc != next_orig_doc:
                # Cross-document connection using ConnectionPatch
                from matplotlib.patches import ConnectionPatch
                conn = ConnectionPatch(
                    xyA=(current_col, current_row), coordsA='data', axesA=current_ax,
                    xyB=(next_col, next_row), coordsB='data', axesB=next_ax,
                    arrowstyle='-',
                    linestyle=line_style,
                    linewidth=line_width,
                    color=line_color,
                    alpha=alpha,
                    zorder=5
                )
                fig.add_artist(conn)
                connections_drawn += 1
                print(f"🔗 Drew {connection_label} connection: Doc {current_orig_doc} → Doc {next_orig_doc}")
            else:
                # Same document connection
                current_ax.plot([current_col, next_col], [current_row, next_row],
                               color=line_color, linestyle=line_style,
                               linewidth=line_width, alpha=alpha, zorder=5)
                connections_drawn += 1

        print(f"✅ Drew {connections_drawn} connections between steps")

    def _add_squad_traversal_legend(self, fig: plt.Figure):
        """Add legend explaining the traversal symbols for SQuAD visualization"""

        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gold',
                       markersize=12, markeredgecolor='black', markeredgewidth=2,
                       linestyle='None', label='Anchor Sentence'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange',
                       markersize=10, markeredgecolor='white', markeredgewidth=1,
                       linestyle='None', label='Selected Sentence'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                       markersize=10, markeredgecolor='white', markeredgewidth=2,
                       linestyle='None', label='Cross-document Jump'),
            plt.Line2D([0], [0], color='orange', linestyle='--', linewidth=2,
                       label='Within Document'),
            plt.Line2D([0], [0], color='red', linestyle='-', linewidth=3,
                       label='Between Documents')
        ]

        fig.legend(handles=legend_elements, loc='lower center',
                   bbox_to_anchor=(0.5, -0.05), ncol=5, frameon=True,
                   fancybox=True, shadow=True, fontsize=11)

    def run_squad_analysis_pipeline(self, analysis_type: str = "random") -> plt.Figure:
        """Run the complete SQuAD analysis pipeline and create visualization"""

        print("🚀 Starting SQuAD Semantic Graph 2D Analysis Pipeline")
        print("=" * 70)

        # ADDED: Seed randomness for different results each time
        random.seed(int(time.time()))

        # Load SQuAD data
        if not self.load_squad_data(num_samples=1000):
            print("❌ Failed to load SQuAD data")
            return plt.Figure()

        # Select question and contexts based on analysis type
        if analysis_type == "focused":
            question, contexts = self.create_focused_context_set([
                'technology', 'computer', 'science', 'research', 'data', 'system'
            ])
        else:  # random
            question, contexts = self.select_random_question_with_contexts()

        # Run analysis
        retrieved_texts, traversal_steps, analysis = self.analyze_squad_question(question, contexts)

        # Create visualization
        print("🎨 Creating beautiful 2D SQuAD visualization...")
        fig = self.create_squad_traversal_visualization(question, traversal_steps)

        # Print results
        print("\n" + "="*60)
        print("SQUAD SEMANTIC GRAPH TRAVERSAL ANALYSIS")
        print("="*60)
        print(f"Question: {question}")
        print(f"Total traversal steps: {analysis['total_steps']}")
        print(f"Cross-document rate: {analysis['cross_document_rate']:.1f}%")

        print("\nConnection type breakdown:")
        for conn_type, percentage in analysis['connection_type_percentages'].items():
            print(f"  {conn_type.replace('_', ' ').title()}: {percentage:.1f}%")

        print(f"\nDocument distribution:")
        for doc_id, count in analysis['document_distribution'].items():
            context_id = contexts[doc_id]['id']
            matrix_shape = self.similarity_matrices[doc_id].shape if doc_id < len(self.similarity_matrices) else "N/A"
            print(f"  Document {doc_id+1} ({context_id}): {count} steps, matrix: {matrix_shape}")

        print("\n✅ SQuAD analysis pipeline complete!")
        print("\n💡 2D Visualization shows:")
        print("   • Real SQuAD 2.0 contexts as side-by-side heatmaps")
        print("   • Semantic similarity using sentence transformers")
        print("   • Gold circle: Query anchor sentence (on diagonal)")
        print("   • Orange/Red circles: Selected sentences (on diagonal)")
        print("   • Red blocks: High similarity between different sentences")
        print("   • Research publication quality with chunk_benchmark.py colors")

        return fig


# Example usage
if __name__ == "__main__":
    # Set matplotlib style for publication quality
    plt.style.use('default')

    # Create visualizer
    visualizer = SQuADSemanticGraph2DVisualizer(
        top_k_per_sentence=15,
        cross_doc_k=8,
        embedding_model="all-MiniLM-L6-v2",
        use_sliding_window=True,  # Add this line
        num_contexts=5,
        figure_size=(24, 8),
        dpi=150
    )

    # Run pipeline multiple times to see different results
    print("🎯 Running pipeline multiple times to demonstrate variety...")

    for i in range(3):
        print(f"\n{'='*50}")
        print(f"RUN {i+1}/3")
        print(f"{'='*50}")

        # Run pipeline with focused topics
        fig = visualizer.run_squad_analysis_pipeline(analysis_type="random")

        # Show the visualization
        plt.tight_layout()
        plt.show()

        # Optional: Save each run
        # fig.savefig(f"squad_semantic_traversal_2d_run_{i+1}.png", dpi=300, bbox_inches='tight')

        # Wait a moment before next run
        time.sleep(1)

    print("\n🎯 This demonstrates:")
    print("   • Different documents and questions each run")
    print("   • Varying sentence counts per document")
    print("   • Dynamic semantic graph traversal patterns")
    print("   • Real-world document structure variations")