#!/usr/bin/env python3
"""
Base Algorithm Class for Semantic Retrieval
==========================================

Defines the abstract interface and common utilities for all retrieval algorithms.
"""

import time
import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity

from ..traversal import TraversalPath, GranularityLevel
from ..knowledge_graph import KnowledgeGraph


@dataclass
class RetrievalResult:
    """Standardized result format across all algorithms."""
    algorithm_name: str
    traversal_path: TraversalPath
    retrieved_content: List[str]  # Text content from retrieved nodes
    confidence_scores: List[float]  # Confidence score for each sentence
    query: str
    total_hops: int
    final_score: float  # Algorithm-specific final score
    processing_time: float
    metadata: Dict[str, Any]
    
    # Enhanced metadata for detailed analysis
    extraction_metadata: Optional[Dict[str, Any]] = None
    sentence_sources: Optional[Dict[str, str]] = None  # sentence_text -> source_chunk
    query_similarities: Optional[Dict[str, float]] = None  # node_id -> query_similarity


class BaseRetrievalAlgorithm(ABC):
    """Abstract base class defining the interface for all retrieval algorithms."""
    
    def __init__(self, knowledge_graph: KnowledgeGraph, config: Dict[str, Any], 
                 query_similarity_cache: Dict[str, float], 
                 logger: Optional[logging.Logger] = None):
        """
        Initialize base algorithm with shared resources.
        
        Args:
            knowledge_graph: The knowledge graph to traverse
            config: Configuration dictionary 
            query_similarity_cache: Pre-computed query similarities for all nodes
            logger: Optional logger instance
        """
        self.kg = knowledge_graph
        self.config = config
        self.query_similarity_cache = query_similarity_cache
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.traversal_config = config['retrieval']['semantic_traversal']
        
        # Algorithm parameters from config
        self.max_hops = self.traversal_config.get('max_safety_hops', 20)
        self.similarity_threshold = self.traversal_config.get('similarity_threshold', 0.3)
        self.min_sentence_threshold = self.traversal_config.get('min_sentence_threshold', 10)
        self.max_results = self.traversal_config.get('max_results', 10)
        self.enable_early_stopping = self.traversal_config.get('enable_early_stopping', True)
    
    @abstractmethod
    def retrieve(self, query: str, anchor_chunk: str) -> RetrievalResult:
        """
        Execute the retrieval algorithm and return standardized results.
        
        Args:
            query: The search query
            anchor_chunk: Starting chunk for traversal
            
        Returns:
            RetrievalResult with algorithm-specific results
        """
        pass
    
    def get_hybrid_connections(self, chunk_id: str) -> List[Tuple[str, str, float]]:
        """
        Get hybrid connections: both connected chunks and sentences within current chunk.
        Returns list of (node_id, node_type, query_similarity) tuples.
        """
        chunk = self.kg.chunks.get(chunk_id)
        if not chunk:
            self.logger.warning(f"Chunk {chunk_id} not found in knowledge graph")
            return []
        
        hybrid_nodes = []
        
        # Get connected chunks (intra and inter-document)
        all_connected_chunks = chunk.intra_doc_connections + chunk.inter_doc_connections
        for connected_chunk_id in all_connected_chunks:
            if connected_chunk_id in self.query_similarity_cache:
                query_sim = self.query_similarity_cache[connected_chunk_id]
                hybrid_nodes.append((connected_chunk_id, "chunk", query_sim))
        
        # Get sentences within current chunk
        chunk_sentences = self.kg.get_chunk_sentences(chunk_id)
        for sentence_obj in chunk_sentences:
            sentence_id = sentence_obj.sentence_id
            if sentence_id in self.query_similarity_cache:
                query_sim = self.query_similarity_cache[sentence_id]
                hybrid_nodes.append((sentence_id, "sentence", query_sim))
        
        # Sort by query similarity (descending)
        hybrid_nodes.sort(key=lambda x: x[2], reverse=True)
        return hybrid_nodes
    
    def get_chunk_sentences(self, chunk_id: str) -> List[str]:
        """Extract all sentences from a chunk."""
        chunk_sentences = self.kg.get_chunk_sentences(chunk_id)
        return [sent.sentence_text for sent in chunk_sentences]
    
    def get_chunk_text(self, chunk_id: str) -> str:
        """Get text content of a chunk."""
        chunk = self.kg.chunks.get(chunk_id)
        return chunk.chunk_text if chunk else ""
    
    def get_sentence_text(self, sentence_id: str) -> str:
        """Get text content of a sentence."""
        sentence = self.kg.sentences.get(sentence_id)
        return sentence.sentence_text if sentence else ""
    
    def deduplicate_sentences(self, new_sentences: List[str], existing_sentences: List[str]) -> List[str]:
        """Remove sentences that already exist in the extracted list."""
        existing_set = set(existing_sentences)
        return [sent for sent in new_sentences if sent not in existing_set]
    
    def find_next_chunk(self, hybrid_nodes: List[Tuple[str, str, float]], visited_chunks: Set[str]) -> Optional[str]:
        """Find the next best unvisited chunk from hybrid nodes (ignoring sentences)."""
        for node_id, node_type, similarity in hybrid_nodes:
            if node_type == "chunk" and node_id not in visited_chunks:
                return node_id
        return None
    
    def calculate_chunk_similarity(self, chunk1_id: str, chunk2_id: str) -> float:
        """Calculate similarity between two chunks using their embeddings."""
        chunk1_embedding = self.kg.get_chunk_embedding(chunk1_id)
        chunk2_embedding = self.kg.get_chunk_embedding(chunk2_id)
        
        if chunk1_embedding is not None and chunk2_embedding is not None:
            similarity = cosine_similarity([chunk1_embedding], [chunk2_embedding])[0][0]
            return float(similarity)
        return 0.0
    
    def calculate_confidence_scores(self, sentences: List[str]) -> List[float]:
        """Calculate confidence scores for extracted sentences based on query similarity."""
        scores = []
        for sentence in sentences:
            sentence_id = self._find_sentence_id(sentence)
            if sentence_id and sentence_id in self.query_similarity_cache:
                scores.append(self.query_similarity_cache[sentence_id])
            else:
                scores.append(0.5)  # Default score for missing similarities
        return scores
    
    def create_sentence_sources_mapping(self, sentences: List[str]) -> Dict[str, str]:
        """Create mapping from sentence text to source chunk ID."""
        sentence_sources = {}
        for sentence in sentences:
            # Find which chunk this sentence came from
            for chunk_id in self.kg.chunks.keys():
                chunk_sentences = self.get_chunk_sentences(chunk_id)
                if sentence in chunk_sentences:
                    sentence_sources[sentence] = chunk_id
                    break
        return sentence_sources
    
    def _find_sentence_id(self, sentence_text: str) -> Optional[str]:
        """Find sentence ID by text content."""
        for sentence_id, sentence_obj in self.kg.sentences.items():
            if sentence_obj.sentence_text == sentence_text:
                return sentence_id
        return None
