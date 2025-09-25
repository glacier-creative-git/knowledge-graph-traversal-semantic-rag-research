#!/usr/bin/env python3
"""
Simplified Hierarchical Knowledge Graph
======================================

Clean three-tier hierarchy: Document → Chunk → Sentence
Includes raw embeddings for lightning-fast traversal comparisons.
Only chunk-to-chunk connections for focused navigation.
"""

import json
import hashlib
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

import numpy as np

from utils.models import ChunkEmbedding, SentenceEmbedding, DocumentSummaryEmbedding


@dataclass
class Document:
    """Document node with summary and themes."""
    doc_id: str
    title: str
    doc_summary: str
    doc_themes: List[str]
    chunk_ids: List[str]  # List of chunk IDs in this document

    # Theme-based connections to other documents
    theme_similar_documents: List[str] = None  # Doc IDs of theme-similar documents
    theme_similarity_scores: Dict[str, float] = None  # doc_id -> theme_similarity_score
    theme_embedding_ref: Dict[str, str] = None  # Reference to theme embedding: {"model": "model_name", "id": "doc_id"}

    def __post_init__(self):
        """Initialize optional fields as empty containers."""
        if self.theme_similar_documents is None:
            self.theme_similar_documents = []
        if self.theme_similarity_scores is None:
            self.theme_similarity_scores = {}
        if self.theme_embedding_ref is None:
            self.theme_embedding_ref = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class Chunk:
    """Chunk node with text, themes, connections, and embedding reference."""
    chunk_id: str
    chunk_text: str
    source_document: str
    inherited_themes: List[str]
    sentence_ids: List[str]  # List of sentence IDs in this chunk
    embedding_ref: Dict[str, str]  # Reference to embedding: {"model": "model_name", "id": "chunk_id"}

    # Connections to other chunks
    intra_doc_connections: List[str]  # Chunk IDs within same document
    inter_doc_connections: List[str]  # Chunk IDs in different documents
    connection_scores: Dict[str, float]  # chunk_id -> similarity_score mapping

    # Theme-based connections inherited from parent document
    theme_similar_documents: List[str] = None  # Doc IDs of theme-similar documents (inherited from parent)
    theme_similarity_scores: Dict[str, float] = None  # doc_id -> theme_similarity_score (inherited from parent)

    def __post_init__(self):
        """Initialize optional fields as empty containers."""
        if self.theme_similar_documents is None:
            self.theme_similar_documents = []
        if self.theme_similarity_scores is None:
            self.theme_similarity_scores = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class Sentence:
    """Sentence node with text, themes, and embedding reference."""
    sentence_id: str
    sentence_text: str
    source_document: str
    source_chunk: str
    sentence_index: int  # Index within the source document
    inherited_themes: List[str]
    embedding_ref: Dict[str, str]  # Reference to embedding: {"model": "model_name", "id": "sentence_id"}

    # Theme-based connections inherited from parent document
    theme_similar_documents: List[str] = None  # Doc IDs of theme-similar documents (inherited from parent)
    theme_similarity_scores: Dict[str, float] = None  # doc_id -> theme_similarity_score (inherited from parent)

    def __post_init__(self):
        """Initialize optional fields as empty containers."""
        if self.theme_similar_documents is None:
            self.theme_similar_documents = []
        if self.theme_similarity_scores is None:
            self.theme_similarity_scores = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class KnowledgeGraph:
    """Simplified hierarchical knowledge graph with embedding references."""

    def __init__(self):
        """Initialize empty knowledge graph."""
        self.documents: Dict[str, Document] = {}
        self.chunks: Dict[str, Chunk] = {}
        self.sentences: Dict[str, Sentence] = {}
        self.metadata: Dict[str, Any] = {}

        # Runtime embedding cache loaded from Phase 3
        self._embedding_cache: Dict[str, Dict[str, np.ndarray]] = {}  # {model: {id: embedding}}

    def add_document(self, document: Document):
        """Add a document to the graph."""
        self.documents[document.doc_id] = document

    def add_chunk(self, chunk: Chunk):
        """Add a chunk to the graph."""
        self.chunks[chunk.chunk_id] = chunk

    def add_sentence(self, sentence: Sentence):
        """Add a sentence to the graph."""
        self.sentences[sentence.sentence_id] = sentence

    def load_phase3_embeddings(self, embeddings_data: Dict[str, Dict[str, List[Any]]]):
        """
        Load Phase 3 cached embeddings into memory for fast lookup.
        Handles both object and dictionary formats for compatibility.

        Args:
            embeddings_data: Multi-granularity embeddings from Phase 3
        """
        for model_name, granularity_embeddings in embeddings_data.items():
            if model_name not in self._embedding_cache:
                self._embedding_cache[model_name] = {'chunks': {}, 'sentences': {}, 'documents': {}}

            # Load chunk embeddings
            chunk_embeddings = granularity_embeddings.get('chunks', [])
            for chunk_emb in chunk_embeddings:
                # Handle both object attributes and dictionary keys
                if hasattr(chunk_emb, 'chunk_id'):
                    # Object format (from original pipeline)
                    chunk_id = chunk_emb.chunk_id
                    embedding = chunk_emb.embedding
                else:
                    # Dictionary format (from JSON cache)
                    chunk_id = chunk_emb['chunk_id']
                    embedding = chunk_emb['embedding']

                self._embedding_cache[model_name]['chunks'][chunk_id] = np.array(embedding)

            # Load sentence embeddings
            sentence_embeddings = granularity_embeddings.get('sentences', [])
            for sent_emb in sentence_embeddings:
                # Handle both object attributes and dictionary keys
                if hasattr(sent_emb, 'sentence_id'):
                    # Object format (from original pipeline)
                    sentence_id = sent_emb.sentence_id
                    embedding = sent_emb.embedding
                else:
                    # Dictionary format (from JSON cache)
                    sentence_id = sent_emb['sentence_id']
                    embedding = sent_emb['embedding']
                
                self._embedding_cache[model_name]['sentences'][sentence_id] = np.array(embedding)

            # Load document theme embeddings
            document_embeddings = granularity_embeddings.get('documents', [])
            for doc_emb in document_embeddings:
                # Handle both object attributes and dictionary keys
                if hasattr(doc_emb, 'doc_id'):
                    # Object format (from original pipeline)
                    doc_id = doc_emb.doc_id
                    theme_embedding = doc_emb.theme_embedding
                else:
                    # Dictionary format (from JSON cache)
                    doc_id = doc_emb['doc_id']
                    theme_embedding = doc_emb['theme_embedding']

                self._embedding_cache[model_name]['documents'][doc_id] = np.array(theme_embedding)

        # Count total embeddings across all models and granularities
        total_embeddings = 0
        for model_cache in self._embedding_cache.values():
            total_embeddings += len(model_cache.get('chunks', {}))
            total_embeddings += len(model_cache.get('documents', {}))
            total_embeddings += len(model_cache.get('sentences', {}))
        print(f"✅ Loaded {total_embeddings} embeddings into cache for {len(self._embedding_cache)} models")

    def get_chunk_connections(self, chunk_id: str) -> List[str]:
        """Get all connected chunk IDs for a given chunk."""
        chunk = self.chunks.get(chunk_id)
        if not chunk:
            return []
        return chunk.intra_doc_connections + chunk.inter_doc_connections

    def get_chunk_embedding(self, chunk_id: str) -> Optional[np.ndarray]:
        """Get embedding for a chunk using reference system."""
        chunk = self.chunks.get(chunk_id)
        if not chunk or not chunk.embedding_ref:
            return None

        model_name = chunk.embedding_ref["model"]
        chunk_ref_id = chunk.embedding_ref["id"]

        # Debug: Check the first lookup to see model name mismatch
        if not hasattr(self, '_debug_logged'):
            available_models = list(self._embedding_cache.keys())
            print(f"🔍 DEBUG: Looking for model '{model_name}', available: {available_models}")
            if model_name in self._embedding_cache:
                chunk_cache_size = len(self._embedding_cache[model_name].get('chunks', {}))
                print(f"🔍 DEBUG: Model '{model_name}' has {chunk_cache_size} chunks in cache")
            self._debug_logged = True

        return self._embedding_cache.get(model_name, {}).get('chunks', {}).get(chunk_ref_id)

    def get_sentence_embedding(self, sentence_id: str) -> Optional[np.ndarray]:
        """Get embedding for a sentence using reference system."""
        sentence = self.sentences.get(sentence_id)
        if not sentence or not sentence.embedding_ref:
            return None

        model_name = sentence.embedding_ref["model"]
        sentence_ref_id = sentence.embedding_ref["id"]

        return self._embedding_cache.get(model_name, {}).get('sentences', {}).get(sentence_ref_id)

    def get_document_theme_embedding(self, doc_id: str) -> Optional[np.ndarray]:
        """Get theme embedding for a document using reference system."""
        document = self.documents.get(doc_id)
        if not document or not document.theme_embedding_ref:
            return None

        model_name = document.theme_embedding_ref["model"]
        doc_ref_id = document.theme_embedding_ref["id"]

        return self._embedding_cache.get(model_name, {}).get('documents', {}).get(doc_ref_id)

    def get_chunk_sentences(self, chunk_id: str) -> List[Sentence]:
        """Get all sentences in a chunk."""
        chunk = self.chunks.get(chunk_id)
        if not chunk:
            return []
        return [self.sentences[sent_id] for sent_id in chunk.sentence_ids if sent_id in self.sentences]

    def calculate_theme_similarities(self, config: Dict[str, Any]) -> None:
        """
        Calculate theme similarities between documents and build theme-based connections.

        Uses theme embeddings to find semantically similar documents and creates
        sparse connections based on config parameters (top_r and similarity threshold).

        Args:
            config: Configuration dictionary with theme_bridging settings
        """
        theme_config = config.get('knowledge_graph_assembly', {}).get('theme_bridging', {})
        top_r = theme_config.get('top_k_bridges', 1)  # Number of theme-similar docs per document
        min_similarity = theme_config.get('min_bridge_similarity', 0.2)

        print(f"🌉 Calculating theme similarities with top_r={top_r}, min_similarity={min_similarity}")

        # Get all document IDs with theme embeddings
        doc_ids_with_embeddings = []
        for doc_id in self.documents.keys():
            if self.get_document_theme_embedding(doc_id) is not None:
                doc_ids_with_embeddings.append(doc_id)

        print(f"📊 Found {len(doc_ids_with_embeddings)} documents with theme embeddings")

        if len(doc_ids_with_embeddings) < 2:
            print("⚠️ Not enough documents with theme embeddings for similarity calculation")
            return

        # Calculate pairwise theme similarities
        for i, doc_id_1 in enumerate(doc_ids_with_embeddings):
            theme_emb_1 = self.get_document_theme_embedding(doc_id_1)
            if theme_emb_1 is None:
                continue

            similarities = []

            for doc_id_2 in doc_ids_with_embeddings:
                if doc_id_1 == doc_id_2:
                    continue

                theme_emb_2 = self.get_document_theme_embedding(doc_id_2)
                if theme_emb_2 is None:
                    continue

                # Calculate cosine similarity between theme embeddings
                try:
                    from sklearn.metrics.pairwise import cosine_similarity
                    similarity = cosine_similarity([theme_emb_1], [theme_emb_2])[0][0]
                except ImportError:
                    # Fallback to numpy-based cosine similarity
                    similarity = np.dot(theme_emb_1, theme_emb_2) / (np.linalg.norm(theme_emb_1) * np.linalg.norm(theme_emb_2))

                if similarity >= min_similarity:
                    similarities.append((doc_id_2, float(similarity)))

            # Sort by similarity (descending) and take top_r
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_similar_docs = similarities[:top_r]

            # Update document with theme-similar documents and scores
            document = self.documents[doc_id_1]
            document.theme_similar_documents = [doc_id for doc_id, _ in top_similar_docs]
            document.theme_similarity_scores = {doc_id: score for doc_id, score in top_similar_docs}

            if top_similar_docs:
                print(f"   📎 {doc_id_1} -> {len(top_similar_docs)} theme-similar docs "
                      f"(best: {top_similar_docs[0][1]:.3f})")

        # Count total theme connections created
        total_connections = sum(len(doc.theme_similar_documents) for doc in self.documents.values())
        print(f"✅ Created {total_connections} theme-based document connections")

        # Propagate theme similarities to all child nodes
        self._propagate_theme_similarities_to_children()

    def get_theme_similar_documents(self, doc_id: str) -> List[str]:
        """Get list of theme-similar document IDs for a given document."""
        document = self.documents.get(doc_id)
        if not document:
            return []
        return document.theme_similar_documents or []

    def get_theme_similar_documents_by_title(self, doc_title: str) -> List[str]:
        """Get list of theme-similar document IDs for a given document title."""
        # Find document by title
        for doc_id, document in self.documents.items():
            if document.title == doc_title:
                return document.theme_similar_documents or []
        return []

    def _propagate_theme_similarities_to_children(self) -> None:
        """
        Propagate document-level theme similarities to all child chunks and sentences.
        This ensures that all nodes have access to theme-based navigation capabilities.
        """
        propagated_chunks = 0
        propagated_sentences = 0

        print(f"🌊 Propagating theme similarities to child nodes...")

        # Iterate through all documents and propagate their theme similarities
        for doc_id, document in self.documents.items():
            if not document.theme_similar_documents:
                continue  # Skip documents without theme connections

            # Propagate to all chunks in this document
            for chunk_id in document.chunk_ids:
                if chunk_id in self.chunks:
                    chunk = self.chunks[chunk_id]
                    chunk.theme_similar_documents = document.theme_similar_documents.copy()
                    chunk.theme_similarity_scores = document.theme_similarity_scores.copy()
                    propagated_chunks += 1

                    # Propagate to all sentences in this chunk
                    for sentence_id in chunk.sentence_ids:
                        if sentence_id in self.sentences:
                            sentence = self.sentences[sentence_id]
                            sentence.theme_similar_documents = document.theme_similar_documents.copy()
                            sentence.theme_similarity_scores = document.theme_similarity_scores.copy()
                            propagated_sentences += 1

        print(f"✅ Propagated theme similarities to {propagated_chunks} chunks and {propagated_sentences} sentences")

    def get_document_title_by_id(self, doc_id: str) -> str:
        """Get document title from document ID."""
        document = self.documents.get(doc_id)
        if document:
            return document.title
        return ""

    def get_theme_similarity_score(self, doc_id_1: str, doc_id_2: str) -> float:
        """Get theme similarity score between two documents."""
        document = self.documents.get(doc_id_1)
        if not document or not document.theme_similarity_scores:
            return 0.0
        return document.theme_similarity_scores.get(doc_id_2, 0.0)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'metadata': self.metadata,
            'documents': {doc_id: doc.to_dict() for doc_id, doc in self.documents.items()},
            'chunks': {chunk_id: chunk.to_dict() for chunk_id, chunk in self.chunks.items()},
            'sentences': {sent_id: sent.to_dict() for sent_id, sent in self.sentences.items()}
        }

    def save(self, file_path: str):
        """Save knowledge graph to JSON file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, file_path: str,
             embeddings_data: Optional[Dict[str, Dict[str, List[Any]]]] = None) -> 'KnowledgeGraph':
        """
        Load knowledge graph from JSON file and optionally load Phase 3 embeddings.

        Args:
            file_path: Path to knowledge graph JSON file
            embeddings_data: Optional Phase 3 embeddings data to load into cache
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        kg = cls()
        kg.metadata = data.get('metadata', {})

        # Load documents
        for doc_id, doc_data in data.get('documents', {}).items():
            document = Document(**doc_data)
            kg.add_document(document)

        # Load chunks
        for chunk_id, chunk_data in data.get('chunks', {}).items():
            chunk = Chunk(**chunk_data)
            kg.add_chunk(chunk)

        # Load sentences
        for sent_id, sent_data in data.get('sentences', {}).items():
            sentence = Sentence(**sent_data)
            kg.add_sentence(sentence)

        # Load Phase 3 embeddings into cache if provided
        if embeddings_data:
            kg.load_phase3_embeddings(embeddings_data)

        return kg


class KnowledgeGraphBuilder:
    """Builder for constructing the simplified hierarchical knowledge graph."""

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """Initialize the knowledge graph builder."""
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

        self.logger.info("🏗️  Initialized simplified knowledge graph builder")

    def build_knowledge_graph(self, chunks: List[Dict[str, Any]],
                              multi_granularity_embeddings: Dict[str, Dict[str, List[Any]]],
                              similarity_data: Dict[str, Dict[str, Any]],
                              theme_data: Dict[str, Any]) -> KnowledgeGraph:
        """
        Build simplified hierarchical knowledge graph with embedding references.

        Args:
            chunks: List of chunk dictionaries from ChunkEngine
            multi_granularity_embeddings: Embeddings from Phase 3
            similarity_data: Similarity matrices from Phase 4
            theme_data: Theme extraction results from Phase 5

        Returns:
            KnowledgeGraph with hierarchical structure and embedding references
        """
        start_time = time.time()

        self.logger.info("🌟 Building simplified hierarchical knowledge graph")

        # Use first model for consistency
        model_name = list(multi_granularity_embeddings.keys())[0]
        granularity_embeddings = multi_granularity_embeddings[model_name]
        model_similarity_data = similarity_data[model_name]

        # Initialize knowledge graph
        kg = KnowledgeGraph()

        # Load Phase 3 embeddings into knowledge graph cache
        kg.load_phase3_embeddings(multi_granularity_embeddings)

        # Step 1: Extract themes by document for easy lookup
        doc_themes_lookup = self._extract_document_themes(theme_data)

        # Step 2: Create document nodes with theme embeddings
        documents = self._create_document_nodes(chunks, granularity_embeddings, doc_themes_lookup, model_name)

        # Generate theme embeddings for documents
        theme_embeddings = self._generate_theme_embeddings(documents, model_name)

        # Add theme embeddings to the knowledge graph cache
        if model_name not in multi_granularity_embeddings:
            multi_granularity_embeddings[model_name] = {}
        if 'documents' not in multi_granularity_embeddings[model_name]:
            multi_granularity_embeddings[model_name]['documents'] = []

        # Add theme embeddings to the embeddings data
        for doc_id, theme_embedding in theme_embeddings.items():
            multi_granularity_embeddings[model_name]['documents'].append({
                'doc_id': doc_id,
                'theme_embedding': theme_embedding.tolist()
            })

        # Reload embeddings into KG cache to include theme embeddings
        kg.load_phase3_embeddings(multi_granularity_embeddings)

        # Add documents to KG
        for document in documents:
            kg.add_document(document)

        # Step 3: Create chunk nodes with embedding references and theme inheritance
        chunk_nodes = self._create_chunk_nodes(chunks, granularity_embeddings, doc_themes_lookup, model_name)
        for chunk in chunk_nodes:
            kg.add_chunk(chunk)

        # Step 4: Add chunk-to-chunk connections from similarity matrices
        self._add_chunk_connections(kg, model_similarity_data)

        # Step 5: Create sentence nodes with embedding references and theme inheritance
        sentence_nodes = self._create_sentence_nodes(granularity_embeddings, doc_themes_lookup, model_name)
        for sentence in sentence_nodes:
            kg.add_sentence(sentence)

        # Step 6: Populate chunk-sentence relationships
        self._populate_chunk_sentence_relationships(kg)

        # Step 7: Calculate theme-based document similarities
        self.logger.info("🌉 Calculating theme-based document similarities...")
        kg.calculate_theme_similarities(self.config)

        # Count theme connections for reporting
        total_theme_connections = sum(len(doc.theme_similar_documents) for doc in kg.documents.values())

        # Debug: Check if theme connections are properly stored
        docs_with_connections = [(doc_id, len(doc.theme_similar_documents))
                               for doc_id, doc in kg.documents.items()
                               if doc.theme_similar_documents]
        self.logger.info(f"🔍 Debug: {len(docs_with_connections)} documents have theme connections")
        if docs_with_connections:
            self.logger.info(f"🔍 Sample connections: {docs_with_connections[:3]}")

        # Step 8: Add metadata
        build_time = time.time() - start_time
        kg.metadata = {
            'created_at': datetime.now().isoformat(),
            'architecture': 'simplified_hierarchical_with_embedding_references',
            'total_documents': len(kg.documents),
            'total_chunks': len(kg.chunks),
            'total_sentences': len(kg.sentences),
            'total_chunk_connections': sum(len(chunk.intra_doc_connections) + len(chunk.inter_doc_connections)
                                           for chunk in kg.chunks.values()),
            'total_theme_connections': total_theme_connections,
            'build_time': build_time,
            'model_used': model_name,
            'config': self.config.get('knowledge_graph', {}),
            'embedding_cache_loaded': True
        }

        self.logger.info(f"🎉 Knowledge graph built successfully in {build_time:.2f}s")
        self.logger.info(f"   Documents: {len(kg.documents)}")
        self.logger.info(f"   Chunks: {len(kg.chunks)}")
        self.logger.info(f"   Sentences: {len(kg.sentences)}")
        self.logger.info(f"   Chunk connections: {kg.metadata['total_chunk_connections']}")
        self.logger.info(f"   Theme connections: {total_theme_connections}")

        return kg

    def _extract_document_themes(self, theme_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Extract themes by document title for easy lookup."""
        doc_themes = {}

        document_themes = theme_data['extraction_results'].get('document_themes', [])
        for theme_result in document_themes:
            # Handle both object and dict formats
            if hasattr(theme_result, 'doc_title'):
                title = theme_result.doc_title
                themes = theme_result.themes
            else:
                title = theme_result['doc_title']
                themes = theme_result['themes']

            doc_themes[title] = themes

        return doc_themes

    def _create_document_nodes(self, chunks: List[Dict[str, Any]],
                               granularity_embeddings: Dict[str, List[Any]],
                               doc_themes_lookup: Dict[str, List[str]],
                               model_name: str) -> List[Document]:
        """Create document-level nodes."""
        documents = []
        doc_summaries = granularity_embeddings.get('doc_summaries', [])

        # Group chunks by document to get chunk IDs
        doc_chunk_mapping = {}
        for chunk in chunks:
            doc_name = chunk['source_article']
            if doc_name not in doc_chunk_mapping:
                doc_chunk_mapping[doc_name] = []
            doc_chunk_mapping[doc_name].append(chunk['chunk_id'])

        # Create document nodes with theme embeddings
        for doc_emb in doc_summaries:
            doc_title = doc_emb.source_article
            doc_themes = doc_themes_lookup.get(doc_title, [])
            chunk_ids = doc_chunk_mapping.get(doc_title, [])

            document = Document(
                doc_id=doc_emb.doc_id,
                title=doc_title,
                doc_summary=doc_emb.summary_text,
                doc_themes=doc_themes,
                chunk_ids=chunk_ids,
                theme_embedding_ref={
                    "model": model_name,
                    "id": doc_emb.doc_id  # Use doc_id as theme embedding ID
                }
            )
            documents.append(document)

        return documents

    def _generate_theme_embeddings(self, documents: List[Document], model_name: str) -> Dict[str, np.ndarray]:
        """
        Generate theme embeddings for documents by combining their themes.

        Args:
            documents: List of Document objects with themes
            model_name: Name of the embedding model to use

        Returns:
            Dictionary mapping doc_id to theme embedding
        """
        from utils.models import EmbeddingModel

        self.logger.info(f"🏷️  Generating theme embeddings for {len(documents)} documents")

        # Initialize embedding model
        embedding_model = EmbeddingModel(model_name, "cpu", self.logger)

        theme_embeddings = {}

        for document in documents:
            if not document.doc_themes:
                # Create zero embedding for documents without themes
                theme_embeddings[document.doc_id] = np.zeros(embedding_model.embedding_dimension)
                continue

            # Combine themes into a single text string as planned
            theme_text = " ".join(document.doc_themes)

            # Generate embedding for combined themes
            theme_embedding = embedding_model.encode_batch([theme_text], batch_size=1)[0]
            theme_embeddings[document.doc_id] = theme_embedding

            self.logger.debug(f"   📎 {document.title}: '{theme_text}' -> embedding shape {theme_embedding.shape}")

        self.logger.info(f"✅ Generated {len(theme_embeddings)} theme embeddings")
        return theme_embeddings

    def _create_chunk_nodes(self, chunks: List[Dict[str, Any]],
                            granularity_embeddings: Dict[str, List[Any]],
                            doc_themes_lookup: Dict[str, List[str]],
                            model_name: str) -> List[Chunk]:
        """Create chunk-level nodes with embedding references and inherited themes."""
        chunk_nodes = []
        chunk_embeddings = granularity_embeddings.get('chunks', [])

        # Create embedding lookup
        embedding_lookup = {emb.chunk_id: emb for emb in chunk_embeddings}

        for chunk_data in chunks:
            chunk_id = chunk_data['chunk_id']
            source_document = chunk_data['source_article']

            # Verify embedding exists for this chunk
            chunk_emb = embedding_lookup.get(chunk_id)
            if not chunk_emb:
                self.logger.warning(f"No embedding found for chunk {chunk_id}")
                continue

            # Inherit themes from document
            inherited_themes = doc_themes_lookup.get(source_document, [])

            # Create sentence IDs for this chunk (will be populated later)
            sentence_ids = []

            chunk_node = Chunk(
                chunk_id=chunk_id,
                chunk_text=chunk_data['text'],
                source_document=source_document,
                inherited_themes=inherited_themes,
                sentence_ids=sentence_ids,
                embedding_ref={"model": model_name, "id": chunk_id},  # Reference instead of full embedding

                # Initialize empty connections (will be populated later)
                intra_doc_connections=[],
                inter_doc_connections=[],
                connection_scores={}
            )
            chunk_nodes.append(chunk_node)

        return chunk_nodes

    def _add_chunk_connections(self, kg: KnowledgeGraph, similarity_data: Dict[str, Any]):
        """Add chunk-to-chunk connections from similarity matrices."""
        connections = similarity_data.get('connections', [])

        self.logger.info(f"Adding {len(connections)} chunk connections")

        for connection in connections:
            # Handle both object and dict formats
            if hasattr(connection, 'source_chunk_id'):
                source_id = connection.source_chunk_id
                target_id = connection.target_chunk_id
                similarity_score = connection.similarity_score
                connection_type = connection.connection_type
            else:
                source_id = connection['source_chunk_id']
                target_id = connection['target_chunk_id']
                similarity_score = connection['similarity_score']
                connection_type = connection['connection_type']

            # Add connection to source chunk
            source_chunk = kg.chunks.get(source_id)
            if source_chunk:
                if connection_type == 'intra_document':
                    source_chunk.intra_doc_connections.append(target_id)
                elif connection_type == 'inter_document':
                    source_chunk.inter_doc_connections.append(target_id)

                # Store similarity score for fast lookup during traversal
                source_chunk.connection_scores[target_id] = similarity_score

    def _create_sentence_nodes(self, granularity_embeddings: Dict[str, List[Any]],
                               doc_themes_lookup: Dict[str, List[str]],
                               model_name: str) -> List[Sentence]:
        """Create sentence-level nodes with embedding references and inherited themes."""
        sentence_nodes = []
        sentence_embeddings = granularity_embeddings.get('sentences', [])

        for sent_emb in sentence_embeddings:
            source_document = sent_emb.source_article

            # Inherit themes from document
            inherited_themes = doc_themes_lookup.get(source_document, [])

            # Store all containing chunks for proper sliding window overlap handling
            containing_chunks = sent_emb.containing_chunks if sent_emb.containing_chunks else []
            # Use first chunk as primary source for backwards compatibility
            source_chunk = containing_chunks[0] if containing_chunks else ""

            sentence_node = Sentence(
                sentence_id=sent_emb.sentence_id,
                sentence_text=sent_emb.sentence_text,
                source_document=source_document,
                source_chunk=source_chunk,
                sentence_index=sent_emb.sentence_index,
                inherited_themes=inherited_themes,
                embedding_ref={"model": model_name, "id": sent_emb.sentence_id}  # Reference instead of full embedding
            )
            # Store containing chunks for population logic
            sentence_node._containing_chunks = containing_chunks
            sentence_nodes.append(sentence_node)

        return sentence_nodes

    def _populate_chunk_sentence_relationships(self, kg: KnowledgeGraph):
        """Populate sentence_ids in chunks based on ALL containing chunks for proper sliding window overlap."""
        for sentence in kg.sentences.values():
            # Get all containing chunks for this sentence (handles sliding window overlap)
            containing_chunks = getattr(sentence, '_containing_chunks', [sentence.source_chunk])

            # Add this sentence to ALL chunks that contain it
            for chunk_id in containing_chunks:
                chunk = kg.chunks.get(chunk_id)
                if chunk and sentence.sentence_id not in chunk.sentence_ids:
                    chunk.sentence_ids.append(sentence.sentence_id)


# Factory function for backward compatibility
def create_knowledge_graph_builder(config: Dict[str, Any],
                                   logger: Optional[logging.Logger] = None) -> KnowledgeGraphBuilder:
    """Factory function to create a KnowledgeGraphBuilder."""
    return KnowledgeGraphBuilder(config, logger)