#!/usr/bin/env python3
"""
Hybrid Query-Aware Knowledge Graph Retrieval Engine
==================================================

Implements hybrid semantic traversal using query-aware crane algorithm.
Core principle: Maintain query relevance while leveraging knowledge graph structure.
"""

import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity

from traversal import (
    TraversalValidator, TraversalPath, ConnectionType, GranularityLevel,
    NavigationLogic, TraversalConstraints
)
from models import EmbeddingModel
from knowledge_graph import KnowledgeGraph


@dataclass
class RetrievalResult:
    """Enhanced container for hybrid retrieval results."""
    traversal_path: TraversalPath
    retrieved_content: List[str]  # Text content from retrieved nodes
    confidence_scores: List[float]  # Confidence score for each node
    query: str
    retrieval_method: str
    metadata: Dict[str, Any]
    # Enhanced metadata for hybrid algorithm
    extraction_metadata: Optional[Dict[str, Any]] = None
    sentence_sources: Optional[Dict[str, str]] = None  # sentence_text -> source_chunk
    query_similarities: Optional[Dict[str, float]] = None  # node_id -> query_similarity


class KnowledgeGraphNavigator:
    """Enhanced navigator with hybrid node collection capabilities."""
    
    def __init__(self, knowledge_graph: KnowledgeGraph, model_name: str, logger: Optional[logging.Logger] = None):
        """Initialize navigator with knowledge graph and embedding model."""
        self.kg = knowledge_graph
        self.model_name = model_name
        self.logger = logger or logging.getLogger(__name__)
    
    def get_hybrid_connections(self, chunk_id: str, query_similarity_cache: Dict[str, float]) -> List[Tuple[str, str, float]]:
        """
        Get hybrid connections: both connected chunks and sentences within current chunk.
        Returns list of (node_id, node_type, query_similarity) tuples.
        """
        chunk = self.kg.chunks.get(chunk_id)
        if not chunk:
            return []
        
        hybrid_nodes = []
        
        # Get connected chunks (intra and inter-document)
        all_connected_chunks = chunk.intra_doc_connections + chunk.inter_doc_connections
        for connected_chunk_id in all_connected_chunks:
            if connected_chunk_id in query_similarity_cache:
                query_sim = query_similarity_cache[connected_chunk_id]
                hybrid_nodes.append((connected_chunk_id, "chunk", query_sim))
        
        # Get sentences within current chunk
        chunk_sentences = self.kg.get_chunk_sentences(chunk_id)
        for sentence_obj in chunk_sentences:
            sentence_id = sentence_obj.sentence_id
            if sentence_id in query_similarity_cache:
                query_sim = query_similarity_cache[sentence_id]
                hybrid_nodes.append((sentence_id, "sentence", query_sim))
        
        # Sort by query similarity (descending)
        hybrid_nodes.sort(key=lambda x: x[2], reverse=True)
        
        self.logger.debug(f"Hybrid connections for {chunk_id}: {len(hybrid_nodes)} total nodes")
        return hybrid_nodes
    
    def get_chunk_connections(self, chunk_id: str, connection_type: ConnectionType) -> List[Tuple[str, float]]:
        """
        Get connections for a chunk based on connection type.
        Returns list of (target_chunk_id, similarity_score) tuples.
        """
        chunk = self.kg.chunks.get(chunk_id)
        if not chunk:
            return []
        
        connections = []
        
        if connection_type == ConnectionType.RAW_SIMILARITY:
            # Get both intra and inter-document connections
            all_connected_ids = chunk.intra_doc_connections + chunk.inter_doc_connections
            for target_id in all_connected_ids:
                if target_id in chunk.connection_scores:
                    score = chunk.connection_scores[target_id]
                    connections.append((target_id, score))
        
        elif connection_type == ConnectionType.THEME_BRIDGE:
            # Only inter-document connections for theme bridges
            for target_id in chunk.inter_doc_connections:
                if target_id in chunk.connection_scores:
                    score = chunk.connection_scores[target_id]
                    connections.append((target_id, score))
        
        elif connection_type == ConnectionType.SEQUENTIAL:
            # For now, use intra-document connections as proxy for sequential
            for target_id in chunk.intra_doc_connections:
                if target_id in chunk.connection_scores:
                    score = chunk.connection_scores[target_id]
                    connections.append((target_id, score))
        
        # Sort by similarity score (descending)
        connections.sort(key=lambda x: x[1], reverse=True)
        return connections
    
    def get_chunk_sentences(self, chunk_id: str) -> List[str]:
        """Get all sentences in a chunk."""
        sentences = self.kg.get_chunk_sentences(chunk_id)
        return [sent.sentence_text for sent in sentences]
    
    def get_chunk_text(self, chunk_id: str) -> str:
        """Get text content of a chunk."""
        chunk = self.kg.chunks.get(chunk_id)
        return chunk.chunk_text if chunk else ""
    
    def get_sentence_text(self, sentence_id: str) -> str:
        """Get text content of a sentence."""
        sentence = self.kg.sentences.get(sentence_id)
        return sentence.sentence_text if sentence else ""


class HybridTraversalEngine:
    """
    Executes hybrid query-aware semantic traversal.
    Implements the crane algorithm: anchor → hybrid traversal → dynamic extraction.
    """
    
    def __init__(self, knowledge_graph: KnowledgeGraph, config: Dict[str, Any], 
                 logger: Optional[logging.Logger] = None):
        """Initialize hybrid traversal engine."""
        self.kg = knowledge_graph
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.traversal_config = config['retrieval']['semantic_traversal']
        
        # Initialize components
        self.validator = TraversalValidator(logger)
        self.model_name = list(config['models']['embedding_models'])[0]  # Use first model
        self.navigator = KnowledgeGraphNavigator(knowledge_graph, self.model_name, logger)
    
    def find_anchor_and_cache_similarities(self, query_embedding: np.ndarray) -> Tuple[str, Dict[str, float]]:
        """
        Find single anchor point and cache query similarities for ALL nodes.
        Returns: (anchor_chunk_id, query_similarity_cache)
        """
        query_similarity_cache = {}
        chunk_similarities = []
        
        # Compute similarities to all chunks
        for chunk_id in self.kg.chunks.keys():
            chunk_embedding = self.kg.get_chunk_embedding(chunk_id)
            if chunk_embedding is not None:
                similarity = cosine_similarity([query_embedding], [chunk_embedding])[0][0]
                query_similarity_cache[chunk_id] = float(similarity)
                chunk_similarities.append((chunk_id, float(similarity)))
        
        # Compute similarities to all sentences
        for sentence_id in self.kg.sentences.keys():
            sentence_embedding = self.kg.get_sentence_embedding(sentence_id)
            if sentence_embedding is not None:
                similarity = cosine_similarity([query_embedding], [sentence_embedding])[0][0]
                query_similarity_cache[sentence_id] = float(similarity)
        
        # Find best anchor chunk (highest similarity chunk above threshold)
        chunk_similarities.sort(key=lambda x: x[1], reverse=True)
        threshold = self.traversal_config.get('similarity_threshold', 0.3)
        
        anchor_chunk = None
        for chunk_id, sim in chunk_similarities:
            if sim >= threshold:
                anchor_chunk = chunk_id
                break
        
        if not anchor_chunk:
            self.logger.warning("No anchor point found above threshold")
            # Fallback to highest similarity chunk
            anchor_chunk = chunk_similarities[0][0] if chunk_similarities else None
        
        self.logger.info(f"🎯 Anchor found: {anchor_chunk} (cached similarities for {len(query_similarity_cache)} nodes)")
        return anchor_chunk, query_similarity_cache
    
    def execute_hybrid_traversal(self, anchor_chunk: str, query_similarity_cache: Dict[str, float]) -> Tuple[TraversalPath, List[str]]:
        """
        Execute hybrid traversal with dynamic extraction.
        Returns: (traversal_path, extracted_sentences)
        """
        # Traversal state
        visited_chunks: Set[str] = set()
        extracted_sentences: List[str] = []
        sentence_sources: Dict[str, str] = {}  # sentence_text -> source_chunk
        path_nodes = []
        connection_types = []
        granularity_levels = []
        
        # Configuration
        min_sentence_threshold = self.traversal_config.get('min_sentence_threshold', 10)
        max_safety_hops = self.traversal_config.get('max_safety_hops', 20)
        
        current_chunk = anchor_chunk
        hop_count = 0
        
        self.logger.info(f"🚀 Starting hybrid traversal from {current_chunk}")
        
        while len(extracted_sentences) < min_sentence_threshold and hop_count < max_safety_hops:
            hop_count += 1
            
            # Add current chunk to visited and path
            visited_chunks.add(current_chunk)
            path_nodes.append(current_chunk)
            granularity_levels.append(GranularityLevel.CHUNK)
            
            self.logger.debug(f"🔍 Hop {hop_count}: Processing chunk {current_chunk}")
            
            # Get hybrid connections (chunks + sentences)
            hybrid_nodes = self.navigator.get_hybrid_connections(current_chunk, query_similarity_cache)
            
            if not hybrid_nodes:
                self.logger.debug(f"No hybrid connections found for {current_chunk}")
                break
            
            # Find highest similarity node
            best_node_id, best_node_type, best_similarity = hybrid_nodes[0]
            
            self.logger.debug(f"   Best node: {best_node_id} ({best_node_type}) sim={best_similarity:.3f}")
            
            if best_node_type == "sentence":
                # EXTRACT: Drop the crane and extract all sentences from current chunk
                self.logger.info(f"📦 EXTRACT triggered: Best node is sentence in current chunk")
                
                chunk_sentences = self._extract_chunk_sentences(current_chunk)
                newly_extracted = self._deduplicate_sentences(chunk_sentences, extracted_sentences)
                
                for sentence in newly_extracted:
                    extracted_sentences.append(sentence)
                    sentence_sources[sentence] = current_chunk
                
                self.logger.info(f"   Extracted {len(newly_extracted)} new sentences from {current_chunk}")
                
                # After extraction, find next best CHUNK (skip sentences)
                next_chunk = self._find_next_chunk(hybrid_nodes, visited_chunks)
                
                if next_chunk:
                    connection_types.append(ConnectionType.RAW_SIMILARITY)
                    current_chunk = next_chunk
                    self.logger.debug(f"   Moving to next chunk: {current_chunk}")
                else:
                    self.logger.debug("   No more unvisited chunks available")
                    break
            
            elif best_node_type == "chunk":
                # TRAVERSE: Move to the best connected chunk
                if best_node_id not in visited_chunks:
                    self.logger.info(f"🚶 TRAVERSE: Moving to chunk {best_node_id}")
                    connection_types.append(ConnectionType.RAW_SIMILARITY)
                    current_chunk = best_node_id
                else:
                    self.logger.debug(f"   Best chunk {best_node_id} already visited")
                    break
            else:
                self.logger.warning(f"Unknown node type: {best_node_type}")
                break
        
        # Create traversal path
        traversal_path = TraversalPath(
            nodes=path_nodes,
            connection_types=connection_types,
            granularity_levels=granularity_levels,
            total_hops=len(connection_types),
            is_valid=True,  # Assume valid for hybrid paths
            validation_errors=[]
        )
        
        self.logger.info(f"✅ Hybrid traversal completed: {len(extracted_sentences)} sentences, {hop_count} hops")
        
        return traversal_path, extracted_sentences
    
    def _extract_chunk_sentences(self, chunk_id: str) -> List[str]:
        """Extract all sentences from a chunk."""
        chunk_sentences = self.kg.get_chunk_sentences(chunk_id)
        return [sent.sentence_text for sent in chunk_sentences]
    
    def _deduplicate_sentences(self, new_sentences: List[str], existing_sentences: List[str]) -> List[str]:
        """Remove sentences that already exist in the extracted list."""
        existing_set = set(existing_sentences)
        return [sent for sent in new_sentences if sent not in existing_set]
    
    def _find_next_chunk(self, hybrid_nodes: List[Tuple[str, str, float]], visited_chunks: Set[str]) -> Optional[str]:
        """Find the next best unvisited chunk from hybrid nodes (ignoring sentences)."""
        for node_id, node_type, similarity in hybrid_nodes:
            if node_type == "chunk" and node_id not in visited_chunks:
                return node_id
        return None


class RetrievalOrchestrator:
    """
    Enhanced retrieval orchestrator with hybrid query-aware traversal.
    """
    
    def __init__(self, knowledge_graph: KnowledgeGraph, config: Dict[str, Any], 
                 logger: Optional[logging.Logger] = None):
        """Initialize enhanced retrieval orchestrator."""
        self.kg = knowledge_graph
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.traversal_config = config['retrieval']['semantic_traversal']
        
        # Query similarity cache for hybrid traversal
        self.query_similarity_cache: Dict[str, float] = {}
        
        # Initialize components
        self.hybrid_engine = HybridTraversalEngine(knowledge_graph, config, logger)
        self.model_name = list(config['models']['embedding_models'])[0]
        
        # Initialize embedding model for query encoding
        self.embedding_model = EmbeddingModel(self.model_name, config['system']['device'], logger)
    
    def retrieve(self, query: str, strategy: str = "semantic_traversal") -> RetrievalResult:
        """
        Enhanced retrieval method with hybrid traversal support.
        
        Args:
            query: Query string
            strategy: Retrieval strategy ("semantic_traversal", "baseline_vector")
            
        Returns:
            RetrievalResult with traversal path and extracted content
        """
        start_time = time.time()
        
        self.logger.info(f"🔍 Retrieving for query: '{query[:50]}...' using {strategy}")
        
        if strategy == "baseline_vector":
            return self._baseline_vector_retrieval(query)
        elif strategy == "semantic_traversal":
            return self._hybrid_semantic_traversal_retrieval(query)
        else:
            raise ValueError(f"Unknown retrieval strategy: {strategy}")
    
    def _hybrid_semantic_traversal_retrieval(self, query: str) -> RetrievalResult:
        """Execute hybrid query-aware semantic traversal retrieval."""
        start_time = time.time()
        
        # Step 1: ANCHOR & CACHE - Find anchor and cache query similarities
        self.logger.debug("🎯 Step 1: Finding anchor and caching query similarities")
        query_embedding = self.embedding_model.encode_single(query)
        anchor_chunk, self.query_similarity_cache = self.hybrid_engine.find_anchor_and_cache_similarities(query_embedding)
        
        if not anchor_chunk:
            self.logger.warning("No anchor point found")
            return RetrievalResult(
                traversal_path=TraversalPath([], [], [], 0, False, ["No anchors found"]),
                retrieved_content=[],
                confidence_scores=[],
                query=query,
                retrieval_method="hybrid_semantic_traversal",
                metadata={"error": "no_anchors"}
            )
        
        # Step 2: TRAVERSE - Execute hybrid traversal with dynamic extraction
        self.logger.debug("🚶 Step 2: Executing hybrid traversal")
        traversal_path, extracted_sentences = self.hybrid_engine.execute_hybrid_traversal(
            anchor_chunk, self.query_similarity_cache
        )
        
        # Step 3: FINALIZE - Apply reranking if enabled and finalize results
        final_content = extracted_sentences
        if self.traversal_config.get('enable_reranking', False):
            final_content = self._apply_reranking(extracted_sentences, query_embedding)
        
        # Limit to max results
        max_results = self.traversal_config.get('max_results', 10)
        final_content = final_content[:max_results]
        
        # Calculate confidence scores
        confidence_scores = self._calculate_hybrid_confidence_scores(final_content, query_embedding)
        
        retrieval_time = time.time() - start_time
        
        # Create sentence sources mapping
        sentence_sources = {}
        for sentence in final_content:
            # Find which chunk this sentence came from by checking all chunks
            for chunk_id in self.kg.chunks.keys():
                chunk_sentences = self.hybrid_engine.navigator.get_chunk_sentences(chunk_id)
                if sentence in chunk_sentences:
                    sentence_sources[sentence] = chunk_id
                    break
        
        self.logger.info(f"✅ Hybrid traversal completed: {len(final_content)} final sentences in {retrieval_time:.3f}s")
        
        return RetrievalResult(
            traversal_path=traversal_path,
            retrieved_content=final_content,
            confidence_scores=confidence_scores,
            query=query,
            retrieval_method="hybrid_semantic_traversal",
            metadata={
                'anchor_chunk': anchor_chunk,
                'total_hops': traversal_path.total_hops,
                'retrieval_time': retrieval_time,
                'extraction_count': len(extracted_sentences),
                'reranking_enabled': self.traversal_config.get('enable_reranking', False)
            },
            extraction_metadata={
                'extraction_points': len([node for node, granularity in zip(traversal_path.nodes, traversal_path.granularity_levels) if granularity == GranularityLevel.CHUNK]),
                'total_extracted': len(extracted_sentences),
                'final_count': len(final_content)
            },
            sentence_sources=sentence_sources,
            query_similarities={sent: self.query_similarity_cache.get(self._find_sentence_id(sent), 0.0) for sent in final_content}
        )
    
    def _apply_reranking(self, sentences: List[str], query_embedding: np.ndarray) -> List[str]:
        """Apply reranking to extracted sentences based on query similarity."""
        # For now, simple reranking based on cached query similarities
        sentence_scores = []
        
        for sentence in sentences:
            sentence_id = self._find_sentence_id(sentence)
            if sentence_id and sentence_id in self.query_similarity_cache:
                score = self.query_similarity_cache[sentence_id]
                sentence_scores.append((sentence, score))
            else:
                # Fallback: compute similarity directly
                sentence_embedding = self._get_sentence_embedding_by_text(sentence)
                if sentence_embedding is not None:
                    score = cosine_similarity([query_embedding], [sentence_embedding])[0][0]
                    sentence_scores.append((sentence, float(score)))
                else:
                    sentence_scores.append((sentence, 0.0))
        
        # Sort by score (descending) and return sentences
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        return [sent for sent, score in sentence_scores]
    
    def _find_sentence_id(self, sentence_text: str) -> Optional[str]:
        """Find sentence ID by text content."""
        for sentence_id, sentence_obj in self.kg.sentences.items():
            if sentence_obj.sentence_text == sentence_text:
                return sentence_id
        return None
    
    def _get_sentence_embedding_by_text(self, sentence_text: str) -> Optional[np.ndarray]:
        """Get sentence embedding by text content."""
        sentence_id = self._find_sentence_id(sentence_text)
        if sentence_id:
            return self.kg.get_sentence_embedding(sentence_id)
        return None
    
    def _calculate_hybrid_confidence_scores(self, sentences: List[str], query_embedding: np.ndarray) -> List[float]:
        """Calculate confidence scores for extracted sentences."""
        scores = []
        for sentence in sentences:
            sentence_id = self._find_sentence_id(sentence)
            if sentence_id and sentence_id in self.query_similarity_cache:
                scores.append(self.query_similarity_cache[sentence_id])
            else:
                scores.append(0.5)  # Default score
        return scores
    
    def _baseline_vector_retrieval(self, query: str) -> RetrievalResult:
        """Simple baseline vector similarity retrieval (unchanged)."""
        start_time = time.time()
        
        query_embedding = self.embedding_model.encode_single(query)
        chunk_similarities = []
        
        # Compute similarity to all chunks
        for chunk_id in self.kg.chunks.keys():
            chunk_embedding = self.kg.get_chunk_embedding(chunk_id)
            if chunk_embedding is not None:
                similarity = cosine_similarity([query_embedding], [chunk_embedding])[0][0]
                chunk_similarities.append((chunk_id, float(similarity)))
        
        # Sort and take top results
        chunk_similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = self.traversal_config.get('max_results', 10)
        top_chunks = chunk_similarities[:top_k]
        
        # Extract content
        content = []
        scores = []
        for chunk_id, score in top_chunks:
            chunk_text = self.hybrid_engine.navigator.get_chunk_text(chunk_id)
            if chunk_text:
                content.append(chunk_text)
                scores.append(score)
        
        # Create simple traversal path for consistency
        simple_path = TraversalPath(
            nodes=[chunk_id for chunk_id, _ in top_chunks],
            connection_types=[],
            granularity_levels=[GranularityLevel.CHUNK] * len(top_chunks),
            total_hops=0,
            is_valid=True,
            validation_errors=[]
        )
        
        retrieval_time = time.time() - start_time
        
        return RetrievalResult(
            traversal_path=simple_path,
            retrieved_content=content,
            confidence_scores=scores,
            query=query,
            retrieval_method="baseline_vector",
            metadata={'retrieval_time': retrieval_time}
        )


# Factory function for easy initialization
def create_retrieval_engine(knowledge_graph: KnowledgeGraph, config: Dict[str, Any], 
                          logger: Optional[logging.Logger] = None) -> RetrievalOrchestrator:
    """Factory function to create a retrieval engine."""
    return RetrievalOrchestrator(knowledge_graph, config, logger)
