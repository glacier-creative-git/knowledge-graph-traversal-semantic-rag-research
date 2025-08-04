"""
Enhanced Semantic Graph Traversal Visualizer with SQuAD 2.0 Integration
========================================================================

This enhanced version integrates SQuAD 2.0 dataset for real-world testing
of semantic graph traversal patterns in RAG systems.
"""

import os
import numpy as np
import pandas as pd
import nltk
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict, Optional
import re
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import argparse
from dataclasses import dataclass
from collections import defaultdict
import json
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

class SQuADSemanticGraphAnalyzer:
    """Enhanced semantic RAG system with SQuAD 2.0 integration"""

    def __init__(self, top_k_per_sentence: int = 20, cross_doc_k: int = 10,
                 embedding_model: str = "all-MiniLM-L6-v2", traversal_depth: int = 3,
                 use_sliding_window: bool = False, num_contexts: int = 5,
                 verbose_duplicates: bool = False):
        self.model = SentenceTransformer(embedding_model)
        self.top_k_per_sentence = top_k_per_sentence
        self.cross_doc_k = cross_doc_k
        self.traversal_depth = traversal_depth
        self.use_sliding_window = use_sliding_window
        self.num_contexts = num_contexts
        self.verbose_duplicates = verbose_duplicates  # Control duplicate warnings

        # Enhanced storage with paragraph awareness
        self.sentences_info: List[SentenceInfo] = []
        self.document_graphs = {}
        self.cross_document_matrix = {}
        self.global_sentence_index = {}
        self.squad_data = None
        self.selected_question = None
        self.selected_contexts = []

    def load_squad_data(self, num_samples: int = 1000):
        """Load SQuAD 2.0 dataset"""
        print(f"Loading SQuAD 2.0 dataset ({num_samples} samples)...")
        try:
            # Load SQuAD 2.0 validation set
            self.squad_data = load_dataset("squad_v2", split=f"validation[:{num_samples}]")
            print(f"✅ Loaded {len(self.squad_data)} SQuAD 2.0 samples")
            return True
        except Exception as e:
            print(f"❌ Failed to load SQuAD 2.0: {e}")
            print("Falling back to SQuAD 1.1...")
            try:
                self.squad_data = load_dataset("squad", split=f"validation[:{num_samples}]")
                print(f"✅ Loaded {len(self.squad_data)} SQuAD 1.1 samples")
                return True
            except Exception as e2:
                print(f"❌ Failed to load any SQuAD dataset: {e2}")
                return False

    def select_random_question_with_contexts(self, min_context_length: int = 500) -> Tuple[str, List[Dict]]:
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
            raise ValueError(f"No contexts found with minimum length {min_context_length}")

        # Select primary sample
        primary_sample = random.choice(substantial_samples)
        self.selected_question = primary_sample['question']

        print(f"🎯 Selected Question: {self.selected_question}")
        print(f"📄 Primary Context Length: {len(primary_sample['context'])} chars")

        # Get the primary context
        primary_context = {
            'context': primary_sample['context'],
            'question': primary_sample['question'],
            'id': primary_sample['id'],
            'title': f"Primary Context ({primary_sample['id']})",
            'answers': primary_sample.get('answers', {'text': [], 'answer_start': []})
        }

        # Find additional related contexts using keyword overlap
        question_keywords = set(self.selected_question.lower().split())

        # Score other contexts by keyword overlap with the question
        context_scores = []
        seen_contexts = {primary_sample['context'].strip()}  # Track seen context text to avoid duplicates
        duplicates_skipped = 0

        for i, sample in enumerate(substantial_samples):
            if sample['id'] == primary_sample['id']:
                continue

            # Skip if we've already seen this exact context text
            context_text = sample['context'].strip()
            if context_text in seen_contexts:
                duplicates_skipped += 1
                if self.verbose_duplicates:
                    print(f"⚠️  Skipping duplicate context: {sample['id']} (same text as previous context)")
                continue

            context_keywords = set(sample['context'].lower().split())
            overlap_score = len(question_keywords.intersection(context_keywords))

            # Also consider question similarity
            question_overlap = len(set(sample['question'].lower().split()).intersection(question_keywords))
            total_score = overlap_score + (question_overlap * 2)  # Weight question similarity higher

            if total_score > 0:  # Only include contexts with some relevance
                context_scores.append((total_score, sample))
                seen_contexts.add(context_text)  # Mark this context text as seen

        if duplicates_skipped > 0:
            print(f"🔍 Skipped {duplicates_skipped} duplicate contexts (use verbose_duplicates=True for details)")

        # Sort by relevance and take top contexts
        context_scores.sort(key=lambda x: x[0], reverse=True)

        # Build final context list
        contexts = [primary_context]

        for score, sample in context_scores[:self.num_contexts-1]:
            context = {
                'context': sample['context'],
                'question': sample['question'],
                'id': sample['id'],
                'title': f"Related Context ({sample['id']}) - Score: {score}",
                'answers': sample.get('answers', {'text': [], 'answer_start': []})
            }
            contexts.append(context)
            print(f"📄 Added Related Context: {len(sample['context'])} chars (relevance score: {score})")

        self.selected_contexts = contexts
        print(f"\n✅ Created multi-document dataset with {len(contexts)} contexts")
        print(f"📊 Total text length: {sum(len(c['context']) for c in contexts):,} characters")
        print(f"🔍 Unique contexts: {len(set(c['context'].strip() for c in contexts))}/{len(contexts)}")

        return self.selected_question, contexts

    def create_focused_context_set(self, topic_keywords: List[str] = None,
                                 max_contexts: int = 5) -> Tuple[str, List[Dict]]:
        """
        Alternative method: Create a focused set of contexts around specific topics
        """
        if not self.squad_data:
            raise ValueError("SQuAD data not loaded. Call load_squad_data() first.")

        if topic_keywords is None:
            # Default to some interesting topics that should have good cross-connections
            topic_keywords = ['science', 'technology', 'computer', 'machine', 'learning',
                            'artificial', 'intelligence', 'algorithm', 'data', 'research']

        print(f"🔍 Searching for contexts containing: {topic_keywords}")

        # Find contexts that contain the topic keywords
        topic_contexts = []
        seen_contexts = set()  # Track seen context text to avoid duplicates
        duplicates_skipped = 0

        for sample in self.squad_data:
            context_lower = sample['context'].lower()
            question_lower = sample['question'].lower()
            context_text = sample['context'].strip()

            # Skip duplicates
            if context_text in seen_contexts:
                duplicates_skipped += 1
                if self.verbose_duplicates:
                    print(f"⚠️  Skipping duplicate context: {sample['id']}")
                continue

            # Count keyword matches in both context and question
            keyword_matches = sum(1 for keyword in topic_keywords
                                if keyword in context_lower or keyword in question_lower)

            if keyword_matches >= 2 and len(sample['context']) >= 300:  # At least 2 keywords and substantial length
                topic_contexts.append((keyword_matches, sample))
                seen_contexts.add(context_text)

        if duplicates_skipped > 0:
            print(f"🔍 Skipped {duplicates_skipped} duplicate contexts during focused search")

        if not topic_contexts:
            print("⚠️ No focused contexts found, falling back to random selection")
            return self.select_random_question_with_contexts()

        # Sort by keyword relevance and take the best ones
        topic_contexts.sort(key=lambda x: x[0], reverse=True)
        selected_samples = [sample for _, sample in topic_contexts[:max_contexts]]

        # Use the most relevant question
        primary_sample = selected_samples[0]
        self.selected_question = primary_sample['question']

        print(f"🎯 Selected Focused Question: {self.selected_question}")

        # Convert to our format
        contexts = []
        for i, sample in enumerate(selected_samples):
            context = {
                'context': sample['context'],
                'question': sample['question'],
                'id': sample['id'],
                'title': f"Context {i+1}: {sample['id']}",
                'answers': sample.get('answers', {'text': [], 'answer_start': []})
            }
            contexts.append(context)
            print(f"📄 Context {i+1}: {len(sample['context'])} chars")

        self.selected_contexts = contexts
        print(f"\n✅ Created focused dataset with {len(contexts)} contexts")
        print(f"🔍 Unique contexts: {len(set(c['context'].strip() for c in contexts))}/{len(contexts)}")
        return self.selected_question, contexts

    def _extract_paragraphs(self, text: str) -> List[str]:
        """Extract paragraphs from text, handling various paragraph separators"""
        # Split on double newlines, multiple spaces, or other paragraph indicators
        paragraphs = re.split(r'\n\s*\n|\r\n\s*\r\n|\.[\s]{2,}', text.strip())
        # Clean up paragraphs and filter out very short ones
        paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 20]
        return paragraphs

    def _analyze_document_structure(self, doc_id: int, text: str,
                                  source_context_id: str = "") -> List[SentenceInfo]:
        """Analyze document structure with optional 3-sentence sliding window embeddings"""
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

    def analyze_squad_question(self, question: str = None, contexts: List[Dict] = None) -> Tuple[List[str], List[TraversalStep], Dict]:
        """
        Main analysis function that processes SQuAD data and performs semantic graph traversal
        """
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
        retrieved_texts, traversal_steps = self._detailed_retrieve(question, top_k=15)

        print(f"🎯 Retrieved {len(retrieved_texts)} texts through {len(traversal_steps)} traversal steps")

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

        # Process each context as a separate document
        for doc_id, context_data in enumerate(contexts):
            text = context_data['context']
            context_id = context_data['id']

            # Analyze document structure
            doc_sentence_infos = self._analyze_document_structure(doc_id, text, context_id)

            # Add to global sentence info
            for sentence_info in doc_sentence_infos:
                sentence_info.sentence_id = len(self.sentences_info)  # Global sentence ID
                self.sentences_info.append(sentence_info)

                # Add to global index
                sentence_hash = hash(sentence_info.text.strip())
                self.global_sentence_index[sentence_hash] = len(self.sentences_info) - 1

        print(f"📊 Processed {len(contexts)} contexts into {len(self.sentences_info)} sentences")

        # Build similarity matrices for each document
        self._build_document_graphs()

        # Build cross-document connections
        self._build_cross_document_connections()

        return time.time() - start_time

    def _build_document_graphs(self):
        """Build semantic graphs for each document"""
        documents_sentences = defaultdict(list)

        # Group sentences by document
        for sentence_info in self.sentences_info:
            documents_sentences[sentence_info.doc_id].append(sentence_info)

        # Build graph for each document
        for doc_id, doc_sentences in documents_sentences.items():
            if len(doc_sentences) < 2:
                continue

            # Get embeddings for this document
            embeddings = np.array([s.embedding for s in doc_sentences])

            # Build similarity matrix
            similarity_matrix = np.dot(embeddings, embeddings.T)

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
        top_anchors = similarities[:3]  # Top 3 anchor candidates

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

        # Sort by relevance and deduplicate based on text content
        all_traversal_steps.sort(key=lambda x: x.relevance_score, reverse=True)

        # Deduplicate by text content while preserving order
        seen_texts = set()
        unique_steps = []
        unique_texts = []

        for step in all_traversal_steps:
            text = step.sentence_info.text.strip()
            if text not in seen_texts:
                seen_texts.add(text)
                unique_steps.append(step)
                unique_texts.append(text)

                if len(unique_texts) >= top_k:
                    break

        return unique_texts, all_traversal_steps

    def _detailed_traverse_from_anchor(self, anchor_sentence: SentenceInfo,
                                     anchor_score: float, visited: set) -> List[TraversalStep]:
        """Perform detailed traversal from an anchor sentence"""
        traversal_steps = []
        queue = [(anchor_sentence.sentence_id, anchor_score, 0, anchor_sentence, None)]  # Added previous_sentence

        while queue:
            sentence_id, score, depth, current_sentence, previous_sentence = queue.pop(0)

            if sentence_id in visited or depth > self.traversal_depth:
                continue

            visited.add(sentence_id)

            # Determine connection type based on relationship to PREVIOUS step, not anchor
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
        depth_distribution = defaultdict(int)
        document_distribution = defaultdict(int)
        context_distribution = defaultdict(int)
        paragraph_jumps = []

        anchor_steps = [step for step in traversal_steps if step.connection_type == 'anchor']

        for step in traversal_steps:
            connection_counts[step.connection_type] += 1
            depth_distribution[step.distance_from_anchor] += 1
            document_distribution[step.sentence_info.doc_id] += 1
            context_distribution[step.sentence_info.source_context_id] += 1

        # Calculate paragraph jumping behavior
        for anchor in anchor_steps:
            anchor_paragraph = anchor.sentence_info.paragraph_id
            anchor_doc = anchor.sentence_info.doc_id

            related_steps = [s for s in traversal_steps
                           if s.sentence_info.doc_id == anchor_doc and s != anchor]

            for step in related_steps:
                paragraph_distance = abs(step.sentence_info.paragraph_id - anchor_paragraph)
                paragraph_jumps.append(paragraph_distance)

        # Calculate statistics
        total_steps = len(traversal_steps)

        # Check for duplicates in retrieval
        unique_texts = set(step.sentence_info.text for step in traversal_steps)
        duplicate_rate = (total_steps - len(unique_texts)) / total_steps * 100 if total_steps > 0 else 0

        # Check for duplicate contexts in the dataset
        unique_context_texts = set(c['context'].strip() for c in self.selected_contexts)
        context_duplicate_rate = (len(self.selected_contexts) - len(unique_context_texts)) / len(self.selected_contexts) * 100

        # Analyze cross-document connections for suspicious patterns
        cross_doc_steps = [s for s in traversal_steps if s.connection_type == 'cross_document']
        suspicious_cross_connections = 0

        for step in cross_doc_steps:
            # Check if this "cross-document" connection is actually to duplicate content
            current_context = self.selected_contexts[step.sentence_info.doc_id]['context']
            for other_context in self.selected_contexts:
                if (other_context != self.selected_contexts[step.sentence_info.doc_id] and
                    current_context.strip() == other_context['context'].strip()):
                    suspicious_cross_connections += 1
                    break

        analysis = {
            'total_steps': total_steps,
            'unique_texts': len(unique_texts),
            'duplicate_rate': duplicate_rate,
            'context_duplicate_rate': context_duplicate_rate,
            'suspicious_cross_connections': suspicious_cross_connections,
            'question': self.selected_question,
            'num_contexts': len(self.selected_contexts),
            'connection_type_percentages': {
                conn_type: (count / total_steps) * 100
                for conn_type, count in connection_counts.items()
            },
            'depth_distribution': dict(depth_distribution),
            'document_distribution': dict(document_distribution),
            'context_distribution': dict(context_distribution),
            'average_paragraph_jump': np.mean(paragraph_jumps) if paragraph_jumps else 0,
            'max_paragraph_jump': max(paragraph_jumps) if paragraph_jumps else 0,
            'paragraph_locality_score': (connection_counts['same_paragraph'] / total_steps) * 100,
            'cross_document_rate': (connection_counts['cross_document'] / total_steps) * 100,
            'semantic_diversity_score': len(set(step.sentence_info.doc_id for step in traversal_steps)) / len(self.selected_contexts) * 100
        }

        return analysis

class SemanticGraphVisualizer:
    """Create beautiful 3D visualizations of semantic graph traversal for SQuAD analysis"""

    def __init__(self, rag_system: SQuADSemanticGraphAnalyzer):
        self.rag_system = rag_system

    def create_3d_traversal_visualization(self, query: str, traversal_steps: List[TraversalStep],
                                        method: str = "pca") -> go.Figure:
        """Create 3D visualization of semantic graph traversal"""

        # Get all embeddings and reduce dimensionality
        all_embeddings = np.array([step.sentence_info.embedding for step in traversal_steps])

        if len(all_embeddings) < 2:
            print("⚠️ Not enough traversal steps for visualization")
            return go.Figure()

        if method == "pca":
            reducer = PCA(n_components=3)
            coords_3d = reducer.fit_transform(all_embeddings)
        else:  # t-SNE
            perplexity = min(30, len(all_embeddings)-1)
            if perplexity < 5:
                perplexity = max(1, len(all_embeddings)-1)
            reducer = TSNE(n_components=3, random_state=42, perplexity=perplexity)
            coords_3d = reducer.fit_transform(all_embeddings)

        # Sort steps by step_number to ensure correct visualization order
        sorted_steps = sorted(traversal_steps, key=lambda x: x.step_number)

        # Create mapping from step to coordinates
        step_to_coords = {}
        for i, step in enumerate(traversal_steps):
            step_to_coords[step.step_number] = coords_3d[i]

        # Prepare data for plotting using sorted steps
        plot_data = []
        for step in sorted_steps:
            coords = step_to_coords[step.step_number]
            plot_data.append({
                'x': coords[0],
                'y': coords[1],
                'z': coords[2],
                'doc_id': step.sentence_info.doc_id,
                'paragraph_id': step.sentence_info.paragraph_id,
                'connection_type': step.connection_type,
                'relevance_score': step.relevance_score,
                'distance_from_anchor': step.distance_from_anchor,
                'context_id': step.sentence_info.source_context_id,
                'text_preview': step.sentence_info.text[:100] + "..." if len(step.sentence_info.text) > 100 else step.sentence_info.text,
                'step_number': step.step_number
            })

        df = pd.DataFrame(plot_data)

        # Enhanced color mapping with better distinction for step-to-step connections
        connection_type_colors = {
            'anchor': 'red',
            'same_paragraph': 'blue',
            'neighboring_paragraph': 'green',
            'distant_paragraph': 'orange',
            'cross_document': 'purple'
        }

        # Create the 3D scatter plot
        fig = go.Figure()

        for connection_type in df['connection_type'].unique():
            mask = df['connection_type'] == connection_type
            subset = df[mask]

            # Enhanced hover information
            hover_text = []
            for _, row in subset.iterrows():
                if row['connection_type'] == 'anchor':
                    connection_desc = "🎯 Starting point"
                elif row['connection_type'] == 'cross_document':
                    connection_desc = f"🚀 Jumped from different document"
                elif row['connection_type'] == 'same_paragraph':
                    connection_desc = f"📝 Stayed in same paragraph"
                elif row['connection_type'] == 'neighboring_paragraph':
                    connection_desc = f"📄 Moved to neighboring paragraph"
                else:
                    connection_desc = f"📚 Moved to distant paragraph"

                hover_text.append(connection_desc)

            fig.add_trace(go.Scatter3d(
                x=subset['x'],
                y=subset['y'],
                z=subset['z'],
                mode='markers+text',
                marker=dict(
                    size=subset['relevance_score'] * 20 + 8,  # Slightly larger markers
                    color=connection_type_colors.get(connection_type, 'gray'),
                    opacity=0.8,
                    line=dict(width=2, color='black')
                ),
                text=subset['step_number'],
                textposition="middle center",
                name=connection_type.replace('_', ' ').title(),
                hovertemplate=(
                    "<b>Step %{text}</b><br>" +
                    "%{customdata[6]}<br>" +
                    "Document: %{customdata[0]} (Context: %{customdata[5]})<br>" +
                    "Paragraph: %{customdata[1]}<br>" +
                    "Relevance: %{customdata[2]:.3f}<br>" +
                    "Distance from Anchor: %{customdata[3]}<br>" +
                    "Text: %{customdata[4]}<br>" +
                    "<extra></extra>"
                ),
                customdata=list(zip(subset['doc_id'], subset['paragraph_id'], subset['relevance_score'],
                                   subset['distance_from_anchor'], subset['text_preview'], subset['context_id'], hover_text))
            ))

        # Add traversal path lines in CORRECT step order
        sorted_coords = [step_to_coords[step.step_number] for step in sorted_steps]

        for i in range(len(sorted_coords) - 1):
            current_step = sorted_steps[i]
            next_step = sorted_steps[i + 1]

            # Color the line based on the connection type of the destination step
            line_color = connection_type_colors.get(next_step.connection_type, 'gray')
            line_width = 4 if next_step.connection_type == 'cross_document' else 2
            line_dash = 'solid' if next_step.connection_type == 'cross_document' else 'dash'

            fig.add_trace(go.Scatter3d(
                x=[sorted_coords[i][0], sorted_coords[i+1][0]],
                y=[sorted_coords[i][1], sorted_coords[i+1][1]],
                z=[sorted_coords[i][2], sorted_coords[i+1][2]],
                mode='lines',
                line=dict(color=line_color, width=line_width, dash=line_dash),
                showlegend=False,
                hovertemplate=f"Step {current_step.step_number} → Step {next_step.step_number}<br>" +
                             f"Connection: {next_step.connection_type.replace('_', ' ').title()}<extra></extra>"
            ))

        # Update layout with enhanced information
        fig.update_layout(
            title=f"SQuAD Semantic Graph Traversal (Forward-Looking Windows)<br>" +
                  f"Query: '{query[:80]}...'<br>" +
                  f"Contexts: {len(self.rag_system.selected_contexts)} | " +
                  f"Steps: {len(sorted_steps)} | " +
                  f"Cross-doc jumps: {len([s for s in sorted_steps if s.connection_type == 'cross_document'])}",
            scene=dict(
                xaxis_title=f"Dimension 1 ({method.upper()})",
                yaxis_title=f"Dimension 2 ({method.upper()})",
                zaxis_title=f"Dimension 3 ({method.upper()})",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            height=900,
            font=dict(size=12),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        return fig

    def create_pattern_analysis_charts(self, analysis: Dict) -> go.Figure:
        """Create charts showing traversal pattern analysis"""

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Connection Type Distribution', 'Depth Distribution',
                          'Document Distribution', 'Data Quality Metrics'),
            specs=[[{'type': 'pie'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'indicator'}]]
        )

        # Connection type pie chart
        conn_types = list(analysis['connection_type_percentages'].keys())
        conn_percentages = list(analysis['connection_type_percentages'].values())

        fig.add_trace(go.Pie(
            labels=[t.replace('_', ' ').title() for t in conn_types],
            values=conn_percentages,
            name="Connection Types"
        ), row=1, col=1)

        # Depth distribution bar chart
        depths = list(analysis['depth_distribution'].keys())
        depth_counts = list(analysis['depth_distribution'].values())

        fig.add_trace(go.Bar(
            x=[f"Depth {d}" for d in depths],
            y=depth_counts,
            name="Depth Distribution"
        ), row=1, col=2)

        # Document distribution
        docs = list(analysis['document_distribution'].keys())
        doc_counts = list(analysis['document_distribution'].values())

        fig.add_trace(go.Bar(
            x=[f"Doc {d}" for d in docs],
            y=doc_counts,
            name="Document Distribution"
        ), row=2, col=1)

        # Data quality metrics gauge
        context_duplicate_rate = analysis.get('context_duplicate_rate', 0)

        # Determine color based on data quality
        if context_duplicate_rate > 20:
            gauge_color = "red"
            quality_label = "Poor"
        elif context_duplicate_rate > 0:
            gauge_color = "orange"
            quality_label = "Fair"
        else:
            gauge_color = "green"
            quality_label = "Good"

        fig.add_trace(go.Indicator(
            mode="number+gauge+delta",
            value=context_duplicate_rate,
            title={"text": f"Context Duplicate Rate %<br><span style='font-size:0.8em'>Quality: {quality_label}</span>"},
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': gauge_color},
                   'steps': [{'range': [0, 1], 'color': "lightgreen"},
                            {'range': [1, 20], 'color': "yellow"},
                            {'range': [20, 100], 'color': "lightcoral"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                               'thickness': 0.75, 'value': 20}}
        ), row=2, col=2)

        # Add annotation for suspicious connections if any
        suspicious = analysis.get('suspicious_cross_connections', 0)
        if suspicious > 0:
            fig.add_annotation(
                x=0.75, y=0.25,
                text=f"⚠️ {suspicious} suspicious<br>cross-connections",
                showarrow=False,
                font=dict(size=12, color="red"),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="red",
                borderwidth=1
            )

        fig.update_layout(height=800, showlegend=False,
                         title_text=f"SQuAD Semantic Graph Analysis<br>Question: {analysis.get('question', 'Unknown')[:60]}...")

        return fig

def run_squad_analysis(embedding_model: str = "all-MiniLM-L6-v2",
                      use_sliding_window: bool = False,
                      analysis_type: str = "random",
                      output_dir: str = "./squad_analysis",
                      verbose_duplicates: bool = False):
    """
    Run comprehensive SQuAD analysis with semantic graph traversal
    """

    os.makedirs(output_dir, exist_ok=True)

    # Initialize analyzer
    analyzer = SQuADSemanticGraphAnalyzer(
        top_k_per_sentence=20,
        cross_doc_k=10,
        embedding_model=embedding_model,
        traversal_depth=3,
        use_sliding_window=use_sliding_window,
        num_contexts=5,
        verbose_duplicates=verbose_duplicates
    )

    window_type = "forward-looking sliding window" if use_sliding_window else "single sentence"
    print("🚀 SQuAD Semantic Graph Analysis")
    print(f"📝 Window type: {window_type}")
    print(f"🔇 Duplicate warnings: {'verbose' if verbose_duplicates else 'summary only'}")
    print("=" * 50)

    # Load SQuAD data
    if not analyzer.load_squad_data(num_samples=2000):
        print("❌ Failed to load SQuAD data")
        return None

    # Select question and contexts based on analysis type
    if analysis_type == "focused":
        question, contexts = analyzer.create_focused_context_set([
            'technology', 'computer', 'science', 'research', 'data', 'system'
        ])
    else:  # random
        question, contexts = analyzer.select_random_question_with_contexts()

    print(f"\n🎯 Question: {question}")
    print(f"📚 Contexts: {len(contexts)}")

    # Run analysis
    retrieved_texts, traversal_steps, analysis = analyzer.analyze_squad_question(question, contexts)

    # Create visualizer and generate visualizations
    print("🎨 Creating visualizations...")
    visualizer = SemanticGraphVisualizer(analyzer)

    # Create 3D traversal visualization
    fig_3d = visualizer.create_3d_traversal_visualization(question, traversal_steps[:20])  # Limit for clarity

    # Create pattern analysis charts
    fig_patterns = visualizer.create_pattern_analysis_charts(analysis)

    # Print results
    print("\n" + "="*60)
    print("SEMANTIC GRAPH TRAVERSAL ANALYSIS")
    print("="*60)
    print(f"Question: {question}")
    print(f"Total traversal steps: {analysis['total_steps']}")
    print(f"Unique texts retrieved: {analysis.get('unique_texts', 'N/A')}")
    print(f"Duplicate rate: {analysis.get('duplicate_rate', 0):.1f}%")

    # Data quality diagnostics
    context_dup_rate = analysis.get('context_duplicate_rate', 0)
    suspicious_connections = analysis.get('suspicious_cross_connections', 0)

    print(f"\n🔍 Data Quality Diagnostics:")
    print(f"Context duplicate rate: {context_dup_rate:.1f}%")
    if context_dup_rate > 0:
        print(f"   ⚠️  Warning: Duplicate contexts detected!")
    if suspicious_connections > 0:
        print(f"   ⚠️  Warning: {suspicious_connections} suspicious cross-document connections detected!")

    print(f"Semantic diversity score: {analysis['semantic_diversity_score']:.1f}%")
    print(f"Cross-document rate: {analysis['cross_document_rate']:.1f}%")
    print(f"Paragraph locality score: {analysis['paragraph_locality_score']:.1f}%")

    print("\nConnection type breakdown:")
    for conn_type, percentage in analysis['connection_type_percentages'].items():
        print(f"  {conn_type.replace('_', ' ').title()}: {percentage:.1f}%")

    print(f"\nDocument distribution:")
    for doc_id, count in analysis['document_distribution'].items():
        context_id = contexts[doc_id]['id']
        print(f"  Context {doc_id} ({context_id}): {count} steps")

    # Save results
    window_suffix = "_sliding_window" if use_sliding_window else "_single_sentence"

    # Save visualizations
    viz_3d_filename = f"squad_3d_traversal_{analysis_type}{window_suffix}.html"
    fig_3d.write_html(os.path.join(output_dir, viz_3d_filename))
    print(f"🎨 3D visualization saved to {output_dir}/{viz_3d_filename}")

    viz_patterns_filename = f"squad_pattern_analysis_{analysis_type}{window_suffix}.html"
    fig_patterns.write_html(os.path.join(output_dir, viz_patterns_filename))
    print(f"📊 Pattern analysis saved to {output_dir}/{viz_patterns_filename}")

    # Save analysis
    filename = f"squad_analysis_{analysis_type}{window_suffix}.json"
    with open(os.path.join(output_dir, filename), 'w') as f:
        json.dump(analysis, f, indent=2)

    print(f"💾 Analysis saved to {output_dir}/{filename}")

    # Save retrieved texts
    retrieved_filename = f"retrieved_texts_{analysis_type}{window_suffix}.txt"
    with open(os.path.join(output_dir, retrieved_filename), 'w') as f:
        f.write(f"Question: {question}\n\n")
        f.write("="*60 + "\n")
        f.write("RETRIEVED TEXTS\n")
        f.write("="*60 + "\n\n")

        for i, text in enumerate(retrieved_texts):
            f.write(f"{i+1}. {text}\n\n")

    print(f"📄 Retrieved texts saved to {output_dir}/{retrieved_filename}")

    # Save context information for reference
    context_filename = f"contexts_{analysis_type}{window_suffix}.txt"
    with open(os.path.join(output_dir, context_filename), 'w') as f:
        f.write(f"Question: {question}\n\n")
        f.write("="*60 + "\n")
        f.write("SQUAD CONTEXTS USED\n")
        f.write("="*60 + "\n\n")

        for i, context in enumerate(contexts):
            f.write(f"Context {i+1}: {context['id']}\n")
            f.write(f"Title: {context.get('title', 'N/A')}\n")
            f.write(f"Length: {len(context['context'])} characters\n")
            f.write(f"Text: {context['context'][:200]}...\n\n")

    print(f"📋 Context info saved to {output_dir}/{context_filename}")

    return analyzer, retrieved_texts, traversal_steps, analysis

if __name__ == "__main__":
    print("🔬 SQuAD Semantic Graph Analysis")
    print("Choose analysis type:")
    print("1. Random question from SQuAD")
    print("2. Focused technology/science questions")
    print("3. Both")

    choice = input("Enter choice (1/2/3): ").strip()

    if choice in ['1', '3']:
        print("\n" + "="*50)
        print("RANDOM QUESTION ANALYSIS")
        print("="*50)
        analyzer1, texts1, steps1, analysis1 = run_squad_analysis(
            analysis_type="random",
            use_sliding_window=True,
            verbose_duplicates=False  # Suppress the flood of duplicate warnings
        )

    if choice in ['2', '3']:
        print("\n" + "="*50)
        print("FOCUSED TOPIC ANALYSIS")
        print("="*50)
        analyzer2, texts2, steps2, analysis2 = run_squad_analysis(
            analysis_type="focused",
            use_sliding_window=True,
            verbose_duplicates=False  # Suppress the flood of duplicate warnings
        )

    print("\n✅ Analysis complete! Check ./squad_analysis/ directory for results.")
    print("🎨 Open the HTML files in your browser to see the 3D visualizations!")
    print("\n💡 Tips:")
    print("   • Cross-document jumps now show as solid purple lines")
    print("   • Same-paragraph connections show as dashed blue lines")
    print("   • Forward-looking windows should fix the 'one off' anchor issue")
    print("   • Duplicate context warnings are now suppressed (use verbose_duplicates=True to see them)")