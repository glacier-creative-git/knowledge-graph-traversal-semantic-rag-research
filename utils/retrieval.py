#!/usr/bin/env python3
"""
Retrieval Engine
===============

Handles retrieval algorithms for the semantic RAG pipeline.
Implements semantic traversal retrieval using similarity matrices and baseline vector retrieval.
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
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from models import ChunkEmbedding, EmbeddingModel


@dataclass
class TraversalPath:
    """Represents a semantic traversal path from an anchor."""
    anchor_chunk: ChunkEmbedding
    path_chunks: List[ChunkEmbedding]
    similarities: List[float]  # Similarity scores for each step
    total_score: float
    path_length: int


@dataclass
class RetrievalResult:
    """Container for retrieval results."""
    chunks: List[ChunkEmbedding]
    scores: List[float]
    query: str
    retrieval_method: str
    metadata: Dict[str, Any]


class BaselineVectorRetriever:
    """Traditional vector similarity retrieval for finding anchor points."""
    
    def __init__(self, config: Dict[str, Any], embeddings: Dict[str, List[ChunkEmbedding]], 
                 logger: Optional[logging.Logger] = None):
        """Initialize baseline vector retriever."""
        self.config = config
        self.embeddings = embeddings
        self.logger = logger or logging.getLogger(__name__)
        self.baseline_config = config['retrieval']['baseline_vector']
        
        # Prepare embedding matrices for fast similarity computation
        self._prepare_embedding_matrices()
    
    def _prepare_embedding_matrices(self):
        """Prepare embedding matrices for efficient similarity computation."""
        self.embedding_matrices = {}
        self.chunk_indices = {}
        
        for model_name, chunk_embeddings in self.embeddings.items():
            # Create matrix of all embeddings
            embeddings_matrix = np.array([emb.embedding for emb in chunk_embeddings])
            self.embedding_matrices[model_name] = embeddings_matrix
            
            # Create mapping from index to chunk
            self.chunk_indices[model_name] = {i: chunk for i, chunk in enumerate(chunk_embeddings)}
            
            self.logger.debug(f"Prepared embedding matrix for {model_name}: {embeddings_matrix.shape}")
    
    def find_anchors(self, query: str, model_name: str, top_k: Optional[int] = None) -> List[ChunkEmbedding]:
        """
        Find anchor points using traditional vector similarity.
        
        Args:
            query: Query string
            model_name: Which embedding model to use
            top_k: Number of anchors to return (defaults to config)
            
        Returns:
            List of anchor chunks
        """
        if top_k is None:
            top_k = self.baseline_config['top_k']
        
        if model_name not in self.embedding_matrices:
            raise ValueError(f"Model {model_name} not available for retrieval")
        
        # Embed the query using the same model
        embedding_model = EmbeddingModel(model_name, self.config['system']['device'], self.logger)
        query_embedding = embedding_model.encode_single(query)
        
        # Compute similarities against all chunks
        chunk_embeddings_matrix = self.embedding_matrices[model_name]
        similarities = cosine_similarity([query_embedding], chunk_embeddings_matrix)[0]
        
        # Get top-k most similar chunks
        top_indices = np.argsort(similarities)[-top_k:][::-1]  # Descending order
        
        anchor_chunks = []
        for idx in top_indices:
            chunk = self.chunk_indices[model_name][idx]
            similarity_score = float(similarities[idx])
            
            # Check threshold
            if similarity_score >= self.baseline_config['similarity_threshold']:
                anchor_chunks.append(chunk)
        
        self.logger.debug(f"Found {len(anchor_chunks)} anchor chunks for query")
        return anchor_chunks
    
    def retrieve(self, query: str, model_name: str, top_k: Optional[int] = None) -> RetrievalResult:
        """Standard vector retrieval for comparison purposes."""
        anchor_chunks = self.find_anchors(query, model_name, top_k)
        
        # For baseline, we just return the anchor chunks
        scores = [1.0] * len(anchor_chunks)  # Placeholder scores
        
        return RetrievalResult(
            chunks=anchor_chunks,
            scores=scores,
            query=query,
            retrieval_method="baseline_vector",
            metadata={
                'model_name': model_name,
                'top_k': top_k or self.baseline_config['top_k'],
                'threshold': self.baseline_config['similarity_threshold']
            }
        )


class SemanticTraversalRetriever:
    """Novel semantic traversal retrieval using similarity matrices."""
    
    def __init__(self, config: Dict[str, Any], embeddings: Dict[str, List[ChunkEmbedding]], 
                 similarities: Dict[str, Dict[str, Any]], logger: Optional[logging.Logger] = None):
        """Initialize semantic traversal retriever."""
        self.config = config
        self.embeddings = embeddings
        self.similarities = similarities
        self.logger = logger or logging.getLogger(__name__)
        self.traversal_config = config['retrieval']['semantic_traversal']
        
        # Initialize baseline retriever for anchor finding
        self.baseline_retriever = BaselineVectorRetriever(config, embeddings, logger)
        
        # Prepare similarity matrices for fast lookup
        self._prepare_similarity_matrices()
    
    def _prepare_similarity_matrices(self):
        """Prepare similarity matrices for efficient traversal."""
        self.similarity_matrices = {}
        self.chunk_to_index = {}
        
        for model_name, similarity_data in self.similarities.items():
            # Get the combined similarity matrix
            combined_matrix = similarity_data['matrices']['combined']
            self.similarity_matrices[model_name] = combined_matrix
            
            # Create mapping from chunk_id to matrix index
            chunk_index_map = similarity_data['chunk_index_map']
            self.chunk_to_index[model_name] = chunk_index_map
            
            self.logger.debug(f"Prepared similarity matrix for {model_name}: {combined_matrix.shape}")
    
    def _get_similar_chunks(self, chunk: ChunkEmbedding, model_name: str) -> List[Tuple[ChunkEmbedding, float]]:
        """Get chunks similar to the given chunk."""
        chunk_id = chunk.chunk_id
        
        # Get chunk index in similarity matrix
        chunk_index_map = self.chunk_to_index[model_name]
        if chunk_id not in chunk_index_map:
            return []
        
        chunk_idx = chunk_index_map[chunk_id]
        
        # Get similarity scores for this chunk
        similarity_matrix = self.similarity_matrices[model_name]
        similarity_row = similarity_matrix[chunk_idx].toarray().flatten()
        
        # Find non-zero similarities (excluding self)
        similar_chunks = []
        chunk_embeddings = self.embeddings[model_name]
        
        for target_idx, similarity_score in enumerate(similarity_row):
            if target_idx != chunk_idx and similarity_score > 0:
                target_chunk = chunk_embeddings[target_idx]
                similar_chunks.append((target_chunk, float(similarity_score)))
        
        # Sort by similarity score (descending)
        similar_chunks.sort(key=lambda x: x[1], reverse=True)
        
        return similar_chunks
    
    def _traverse_from_anchor(self, anchor_chunk: ChunkEmbedding, model_name: str) -> TraversalPath:
        """
        Perform greedy semantic traversal from an anchor chunk.
        
        Args:
            anchor_chunk: Starting point for traversal
            model_name: Which embedding model/similarity matrix to use
            
        Returns:
            TraversalPath object containing the traversal results
        """
        max_hops = self.traversal_config['max_hops']
        similarity_threshold = self.traversal_config['similarity_threshold']
        
        path_chunks = [anchor_chunk]
        similarities = [1.0]  # Perfect similarity to self
        visited = {anchor_chunk.chunk_id}
        current_chunk = anchor_chunk
        
        self.logger.debug(f"Starting traversal from anchor: {anchor_chunk.chunk_id}")
        
        for hop in range(max_hops):
            # Get all chunks similar to current position
            similar_chunks = self._get_similar_chunks(current_chunk, model_name)
            
            # Find the MOST similar unvisited chunk
            best_chunk = None
            best_similarity = -1
            
            for chunk, similarity in similar_chunks:
                if chunk.chunk_id not in visited and similarity > best_similarity:
                    best_chunk = chunk
                    best_similarity = similarity
            
            # If we found a good next step, take it
            if best_chunk and best_similarity >= similarity_threshold:
                path_chunks.append(best_chunk)
                similarities.append(best_similarity)
                visited.add(best_chunk.chunk_id)
                current_chunk = best_chunk
                
                self.logger.debug(f"Hop {hop + 1}: {current_chunk.chunk_id} (similarity: {best_similarity:.3f})")
            else:
                self.logger.debug(f"Traversal ended at hop {hop + 1}: no valid next chunk")
                break
        
        # Calculate total path score (could be more sophisticated)
        total_score = sum(similarities) / len(similarities)
        
        return TraversalPath(
            anchor_chunk=anchor_chunk,
            path_chunks=path_chunks,
            similarities=similarities,
            total_score=total_score,
            path_length=len(path_chunks)
        )
    
    def _deduplicate_chunks(self, chunks: List[ChunkEmbedding]) -> List[ChunkEmbedding]:
        """Remove duplicate chunks by chunk_id."""
        seen_ids = set()
        unique_chunks = []
        
        for chunk in chunks:
            if chunk.chunk_id not in seen_ids:
                unique_chunks.append(chunk)
                seen_ids.add(chunk.chunk_id)
        
        return unique_chunks
    
    def _rerank_against_query(self, chunks: List[ChunkEmbedding], query: str, model_name: str) -> List[Tuple[ChunkEmbedding, float]]:
        """Rerank chunks against the original query."""
        if not chunks:
            return []
        
        # Embed the query
        embedding_model = EmbeddingModel(model_name, self.config['system']['device'], self.logger)
        query_embedding = embedding_model.encode_single(query)
        
        # Compute similarities against query
        chunk_embeddings = np.array([chunk.embedding for chunk in chunks])
        similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
        
        # Create scored chunks
        scored_chunks = [(chunk, float(sim)) for chunk, sim in zip(chunks, similarities)]
        
        # Sort by score (descending)
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        return scored_chunks
    
    def retrieve(self, query: str, model_name: str) -> RetrievalResult:
        """
        Perform semantic traversal retrieval.
        
        Args:
            query: Query string
            model_name: Which embedding model to use
            
        Returns:
            RetrievalResult containing retrieved chunks
        """
        start_time = time.time()
        
        # Step 1: Find anchor points using baseline retrieval
        num_anchors = self.traversal_config['num_anchors']
        anchors = self.baseline_retriever.find_anchors(query, model_name, top_k=num_anchors)
        
        if not anchors:
            self.logger.warning("No anchor points found for query")
            return RetrievalResult([], [], query, "semantic_traversal", {})
        
        self.logger.info(f"Found {len(anchors)} anchor points")
        
        # Step 2: Perform semantic traversal from each anchor
        all_paths = []
        all_chunks = []
        
        for i, anchor in enumerate(anchors):
            self.logger.debug(f"Traversing from anchor {i + 1}/{len(anchors)}")
            path = self._traverse_from_anchor(anchor, model_name)
            all_paths.append(path)
            all_chunks.extend(path.path_chunks)
        
        # Step 3: Deduplicate chunks
        unique_chunks = self._deduplicate_chunks(all_chunks)
        self.logger.info(f"Collected {len(all_chunks)} chunks, {len(unique_chunks)} unique")
        
        # Step 4: Rerank all chunks against original query
        scored_chunks = self._rerank_against_query(unique_chunks, query, model_name)
        
        # Step 5: Return top results
        max_results = self.traversal_config['max_results']
        final_chunks = [chunk for chunk, score in scored_chunks[:max_results]]
        final_scores = [score for chunk, score in scored_chunks[:max_results]]
        
        retrieval_time = time.time() - start_time
        
        self.logger.info(f"Semantic traversal completed: {len(final_chunks)} results in {retrieval_time:.3f}s")
        
        return RetrievalResult(
            chunks=final_chunks,
            scores=final_scores,
            query=query,
            retrieval_method="semantic_traversal",
            metadata={
                'model_name': model_name,
                'num_anchors': len(anchors),
                'total_paths': len(all_paths),
                'unique_chunks': len(unique_chunks),
                'retrieval_time': retrieval_time,
                'paths': [
                    {
                        'anchor_id': path.anchor_chunk.chunk_id,
                        'path_length': path.path_length,
                        'total_score': path.total_score
                    } for path in all_paths
                ]
            }
        )


class RetrievalEngine:
    """Main engine for handling different retrieval algorithms."""
    
    def __init__(self, config: Dict[str, Any], embeddings: Dict[str, List[ChunkEmbedding]], 
                 similarities: Dict[str, Dict[str, Any]], logger: Optional[logging.Logger] = None,
                 knowledge_graph: Optional[Any] = None):
        """Initialize the retrieval engine."""
        self.config = config
        self.embeddings = embeddings
        self.similarities = similarities
        self.knowledge_graph = knowledge_graph
        self.logger = logger or logging.getLogger(__name__)
        self.retrieval_config = config['retrieval']
        
        # Initialize retrievers based on config
        self._initialize_retrievers()
    
    def _initialize_retrievers(self):
        """Initialize retrieval algorithms."""
        algorithm = self.retrieval_config['algorithm']
        
        self.logger.info(f"Initializing retrieval engine with algorithm: {algorithm}")
        
        # Always initialize baseline (needed for anchors)
        self.baseline_retriever = BaselineVectorRetriever(self.config, self.embeddings, self.logger)
        
        # Initialize semantic traversal if needed
        if algorithm in ['semantic_traversal', 'hybrid']:
            self.semantic_retriever = SemanticTraversalRetriever(
                self.config, self.embeddings, self.similarities, self.logger
            )
        
        self.logger.info("Retrieval engine initialized successfully")
    
    def retrieve(self, query: str, model_name: str, algorithm: Optional[str] = None) -> RetrievalResult:
        """
        Retrieve chunks for a query using the specified algorithm.
        
        Args:
            query: Query string
            model_name: Which embedding model to use
            algorithm: Override default algorithm (optional)
            
        Returns:
            RetrievalResult containing retrieved chunks
        """
        if algorithm is None:
            algorithm = self.retrieval_config['algorithm']
        
        self.logger.info(f"Retrieving for query using {algorithm}: '{query[:50]}...'")
        
        if algorithm == "baseline_vector":
            return self.baseline_retriever.retrieve(query, model_name)
        elif algorithm == "semantic_traversal":
            return self.semantic_retriever.retrieve(query, model_name)
        else:
            raise ValueError(f"Unknown retrieval algorithm: {algorithm}")
    
    def get_retrieval_statistics(self) -> Dict[str, Any]:
        """Get statistics about the retrieval system."""
        stats = {
            'algorithm': self.retrieval_config['algorithm'],
            'models_available': list(self.embeddings.keys()),
            'total_chunks_per_model': {
                model: len(chunks) for model, chunks in self.embeddings.items()
            }
        }
        
        if hasattr(self, 'semantic_retriever'):
            stats['semantic_traversal_config'] = self.retrieval_config['semantic_traversal']
        
        return stats
