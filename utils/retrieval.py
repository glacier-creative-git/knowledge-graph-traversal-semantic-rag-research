#!/usr/bin/env python3
"""
Knowledge Graph Retrieval Engine
===============================

Executes semantic traversal retrieval using the knowledge graph and traversal rules.
Implements the "cargo crane" approach: anchor → traverse → extract.
"""

import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
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
    """Container for retrieval results."""
    traversal_path: TraversalPath
    retrieved_content: List[str]  # Text content from retrieved nodes
    confidence_scores: List[float]  # Confidence score for each node
    query: str
    retrieval_method: str
    metadata: Dict[str, Any]


class ContextAssessment:
    """
    Context sufficiency assessment moved from traversal.py.
    This is computational work, not pure rules.
    """
    
    @staticmethod
    def assess_context_sufficiency(nodes: List[str], 
                                 question_complexity: str,
                                 node_properties: Dict[str, Dict[str, Any]]) -> List[float]:
        """
        Determine if retrieved nodes contain sufficient context for question complexity.
        Returns context sufficiency scores for each node.
        """
        context_scores = []
        
        complexity_requirements = {
            "simple": {"min_words": 50, "min_sentences": 2},
            "medium": {"min_words": 150, "min_sentences": 5},
            "hard": {"min_words": 300, "min_sentences": 10},
            "expert": {"min_words": 500, "min_sentences": 15}
        }
        
        requirements = complexity_requirements.get(question_complexity, complexity_requirements["medium"])
        
        for node_id in nodes:
            node_props = node_properties.get(node_id, {})
            
            # Get text content
            text = node_props.get('page_content', '') or node_props.get('text', '')
            word_count = len(text.split()) if text else 0
            
            # Estimate sentence count (rough approximation)
            sentence_count = max(1, text.count('.') + text.count('!') + text.count('?')) if text else 0
            
            # Calculate sufficiency score
            word_score = min(1.0, word_count / requirements["min_words"])
            sentence_score = min(1.0, sentence_count / requirements["min_sentences"])
            
            # Combined score with word count weighted more heavily
            context_score = (word_score * 0.7) + (sentence_score * 0.3)
            context_scores.append(context_score)
        
        return context_scores


class KnowledgeGraphNavigator:
    """Interfaces with knowledge graph for navigation and embedding lookup."""
    
    def __init__(self, knowledge_graph: KnowledgeGraph, model_name: str, logger: Optional[logging.Logger] = None):
        """Initialize navigator with knowledge graph and embedding model."""
        self.kg = knowledge_graph
        self.model_name = model_name
        self.logger = logger or logging.getLogger(__name__)
    
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
    
    def get_embedding_similarity(self, chunk_a: str, chunk_b: str) -> float:
        """Compute similarity between two chunks using embeddings."""
        emb_a = self.kg.get_chunk_embedding(chunk_a)
        emb_b = self.kg.get_chunk_embedding(chunk_b)
        
        if emb_a is None or emb_b is None:
            return 0.0
        
        # Compute cosine similarity
        similarity = cosine_similarity([emb_a], [emb_b])[0][0]
        return float(similarity)
    
    def get_chunk_sentences(self, chunk_id: str) -> List[str]:
        """Get all sentences in a chunk."""
        sentences = self.kg.get_chunk_sentences(chunk_id)
        return [sent.sentence_text for sent in sentences]
    
    def get_chunk_text(self, chunk_id: str) -> str:
        """Get text content of a chunk."""
        chunk = self.kg.chunks.get(chunk_id)
        return chunk.chunk_text if chunk else ""
    
    def get_document_chunks(self, doc_id: str) -> List[str]:
        """Get all chunk IDs in a document."""
        document = self.kg.documents.get(doc_id)
        return document.chunk_ids if document else []


class SemanticTraversalEngine:
    """
    Executes semantic traversal using traversal.py rules and knowledge graph.
    Implements the "cargo crane" algorithm: anchor → traverse → extract.
    """
    
    def __init__(self, knowledge_graph: KnowledgeGraph, config: Dict[str, Any], 
                 logger: Optional[logging.Logger] = None):
        """Initialize traversal engine."""
        self.kg = knowledge_graph
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.traversal_config = config['retrieval']['semantic_traversal']
        
        # Initialize components
        self.validator = TraversalValidator(logger)
        self.model_name = list(config['models']['embedding_models'])[0]  # Use first model
        self.navigator = KnowledgeGraphNavigator(knowledge_graph, self.model_name, logger)
    
    def find_anchor_points(self, query_embedding: np.ndarray, top_k: int) -> List[str]:
        """
        Find anchor points using direct similarity to query.
        Returns list of chunk IDs sorted by similarity to query.
        """
        chunk_similarities = []
        
        for chunk_id in self.kg.chunks.keys():
            chunk_embedding = self.kg.get_chunk_embedding(chunk_id)
            if chunk_embedding is not None:
                similarity = cosine_similarity([query_embedding], [chunk_embedding])[0][0]
                chunk_similarities.append((chunk_id, float(similarity)))
        
        # Sort by similarity (descending) and take top_k
        chunk_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Apply similarity threshold
        threshold = self.traversal_config.get('similarity_threshold', 0.3)
        anchors = [chunk_id for chunk_id, sim in chunk_similarities[:top_k] if sim >= threshold]
        
        self.logger.debug(f"Found {len(anchors)} anchor points from top {top_k} candidates")
        return anchors
    
    def execute_traversal(self, start_chunk_id: str, traversal_pattern: str, max_hops: Optional[int] = None) -> TraversalPath:
        """
        Execute semantic traversal from a starting chunk.
        Follows traversal.py rules for the specified pattern.
        """
        # Get constraints for the pattern from traversal rules
        constraints = TraversalConstraints.get_constraints_for_pattern(traversal_pattern)
        if not constraints:
            # Default to raw similarity
            traversal_pattern = "raw_similarity"
            constraints = TraversalConstraints.get_constraints_for_pattern(traversal_pattern)
        
        # Use max_hops from traversal constraints, not config
        if max_hops is None:
            max_hops = constraints.get('max_hops', 1)  # Get from traversal rules
        
        self.logger.debug(f"Executing {traversal_pattern} traversal with max_hops={max_hops} (from traversal constraints)")
        
        # Initialize path
        path_nodes = [start_chunk_id]
        connection_types = []
        granularity_levels = [GranularityLevel.CHUNK]  # Start at chunk level
        node_documents = [self._get_chunk_document(start_chunk_id)]
        
        current_chunk = start_chunk_id
        visited = {start_chunk_id}
        
        # Execute traversal based on pattern
        if traversal_pattern == "raw_similarity":
            path_nodes, connection_types, granularity_levels, node_documents = self._execute_raw_similarity_traversal(
                current_chunk, visited, max_hops, path_nodes, connection_types, granularity_levels, node_documents
            )
        
        elif traversal_pattern == "hierarchical":
            path_nodes, connection_types, granularity_levels, node_documents = self._execute_hierarchical_traversal(
                current_chunk, visited, path_nodes, connection_types, granularity_levels, node_documents
            )
        
        elif traversal_pattern == "theme_bridge":
            path_nodes, connection_types, granularity_levels, node_documents = self._execute_theme_bridge_traversal(
                current_chunk, visited, max_hops, path_nodes, connection_types, granularity_levels, node_documents
            )
        
        # Validate the final path
        traversal_path = self.validator.validate_path(
            path_nodes, connection_types, granularity_levels, node_documents, traversal_pattern
        )
        
        return traversal_path
    
    def _execute_raw_similarity_traversal(self, current_chunk: str, visited: set, max_hops: int,
                                        path_nodes: List[str], connection_types: List[ConnectionType],
                                        granularity_levels: List[GranularityLevel], 
                                        node_documents: List[str]) -> Tuple[List[str], List[ConnectionType], 
                                                                          List[GranularityLevel], List[str]]:
        """Execute raw similarity traversal (chunk-to-chunk only)."""
        similarity_threshold = self.traversal_config.get('similarity_threshold', 0.3)
        
        for hop in range(max_hops):
            # Get similar chunks
            connections = self.navigator.get_chunk_connections(current_chunk, ConnectionType.RAW_SIMILARITY)
            
            # Find best unvisited chunk
            best_chunk = None
            best_score = -1
            
            for target_chunk, score in connections:
                if target_chunk not in visited and score > best_score and score >= similarity_threshold:
                    best_chunk = target_chunk
                    best_score = score
            
            if best_chunk:
                path_nodes.append(best_chunk)
                connection_types.append(ConnectionType.RAW_SIMILARITY)
                granularity_levels.append(GranularityLevel.CHUNK)
                node_documents.append(self._get_chunk_document(best_chunk))
                
                visited.add(best_chunk)
                current_chunk = best_chunk
                
                self.logger.debug(f"Raw similarity hop {hop + 1}: {current_chunk} (score: {best_score:.3f})")
            else:
                self.logger.debug(f"Raw similarity traversal ended at hop {hop + 1}: no valid next chunk")
                break
        
        return path_nodes, connection_types, granularity_levels, node_documents
    
    def _execute_hierarchical_traversal(self, current_chunk: str, visited: set,
                                      path_nodes: List[str], connection_types: List[ConnectionType],
                                      granularity_levels: List[GranularityLevel], 
                                      node_documents: List[str]) -> Tuple[List[str], List[ConnectionType], 
                                                                        List[GranularityLevel], List[str]]:
        """Execute hierarchical traversal (Document → Chunk → Sentence)."""
        # For hierarchical traversal, we need to go: current_chunk → sentence
        # Get sentences from current chunk
        chunk_sentences = self.kg.get_chunk_sentences(current_chunk)
        
        if chunk_sentences:
            # Take the first sentence as hierarchical descent
            first_sentence = chunk_sentences[0]
            
            path_nodes.append(first_sentence.sentence_id)
            connection_types.append(ConnectionType.HIERARCHICAL)
            granularity_levels.append(GranularityLevel.SENTENCE)
            node_documents.append(self._get_chunk_document(current_chunk))  # Same document
            
            self.logger.debug(f"Hierarchical traversal: {current_chunk} → {first_sentence.sentence_id}")
        
        return path_nodes, connection_types, granularity_levels, node_documents
    
    def _execute_theme_bridge_traversal(self, current_chunk: str, visited: set, max_hops: int,
                                      path_nodes: List[str], connection_types: List[ConnectionType],
                                      granularity_levels: List[GranularityLevel], 
                                      node_documents: List[str]) -> Tuple[List[str], List[ConnectionType], 
                                                                        List[GranularityLevel], List[str]]:
        """Execute theme bridge traversal (cross-document navigation)."""
        current_doc = self._get_chunk_document(current_chunk)
        
        # Get inter-document connections (theme bridges)
        connections = self.navigator.get_chunk_connections(current_chunk, ConnectionType.THEME_BRIDGE)
        
        # Find best cross-document connection
        for target_chunk, score in connections:
            target_doc = self._get_chunk_document(target_chunk)
            
            if target_chunk not in visited and target_doc != current_doc:
                path_nodes.append(target_chunk)
                connection_types.append(ConnectionType.THEME_BRIDGE)
                granularity_levels.append(GranularityLevel.CHUNK)
                node_documents.append(target_doc)
                
                self.logger.debug(f"Theme bridge: {current_chunk} → {target_chunk} (cross-doc: {current_doc} → {target_doc})")
                break
        
        return path_nodes, connection_types, granularity_levels, node_documents
    
    def _get_chunk_document(self, chunk_id: str) -> str:
        """Get document name for a chunk."""
        chunk = self.kg.chunks.get(chunk_id)
        return chunk.source_document if chunk else "unknown"


class RetrievalOrchestrator:
    """
    Main retrieval interface that coordinates anchor finding, traversal, and content extraction.
    """
    
    def __init__(self, knowledge_graph: KnowledgeGraph, config: Dict[str, Any], 
                 logger: Optional[logging.Logger] = None):
        """Initialize retrieval orchestrator."""
        self.kg = knowledge_graph
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.traversal_config = config['retrieval']['semantic_traversal']
        
        # Initialize components
        self.traversal_engine = SemanticTraversalEngine(knowledge_graph, config, logger)
        self.model_name = list(config['models']['embedding_models'])[0]
        
        # Initialize embedding model for query encoding
        self.embedding_model = EmbeddingModel(self.model_name, config['system']['device'], logger)
    
    def retrieve(self, query: str, strategy: str = "semantic_traversal") -> RetrievalResult:
        """
        Main retrieval method implementing the cargo crane approach.
        
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
            return self._semantic_traversal_retrieval(query)
        else:
            raise ValueError(f"Unknown retrieval strategy: {strategy}")
    
    def _semantic_traversal_retrieval(self, query: str) -> RetrievalResult:
        """Execute semantic traversal retrieval using the cargo crane approach."""
        start_time = time.time()
        
        # Step 1: ANCHOR - Find anchor points using query similarity
        self.logger.debug("🎯 Step 1: Finding anchor points")
        query_embedding = self.embedding_model.encode_single(query)
        num_anchors = self.traversal_config.get('num_anchors', 3)
        anchor_chunks = self.traversal_engine.find_anchor_points(query_embedding, num_anchors)
        
        if not anchor_chunks:
            self.logger.warning("No anchor points found")
            return RetrievalResult(
                traversal_path=TraversalPath([], [], [], 0, False, ["No anchors found"]),
                retrieved_content=[],
                confidence_scores=[],
                query=query,
                retrieval_method="semantic_traversal",
                metadata={"error": "no_anchors"}
            )
        
        self.logger.info(f"🎯 Found {len(anchor_chunks)} anchor points")
        
        # Step 2: TRAVERSE - Execute semantic traversal from anchors
        self.logger.debug("🚶 Step 2: Executing semantic traversal")
        max_hops = self.traversal_config.get('max_hops', 3)  # No longer used - constraints come from traversal rules
        traversal_pattern = "raw_similarity"  # Start with simplest pattern
        
        all_paths = []
        all_content = []
        
        for i, anchor_chunk in enumerate(anchor_chunks):
            self.logger.debug(f"Traversing from anchor {i + 1}/{len(anchor_chunks)}: {anchor_chunk}")
            
            path = self.traversal_engine.execute_traversal(anchor_chunk, traversal_pattern)
            all_paths.append(path)
            
            # Step 3: EXTRACT - Extract content from nodes in path
            path_content = self._extract_content_from_path(path)
            all_content.extend(path_content)
        
        # Combine and deduplicate content
        unique_content = list(dict.fromkeys(all_content))  # Preserve order while deduplicating
        
        # Take best path as primary result
        best_path = max(all_paths, key=lambda p: len(p.nodes)) if all_paths else None
        
        # Calculate confidence scores based on similarity and path validity
        confidence_scores = self._calculate_confidence_scores(best_path, query_embedding)
        
        retrieval_time = time.time() - start_time
        
        self.logger.info(f"✅ Semantic traversal completed: {len(unique_content)} content pieces in {retrieval_time:.3f}s")
        
        return RetrievalResult(
            traversal_path=best_path,
            retrieved_content=unique_content[:self.traversal_config.get('max_results', 10)],
            confidence_scores=confidence_scores,
            query=query,
            retrieval_method="semantic_traversal",
            metadata={
                'num_anchors': len(anchor_chunks),
                'total_paths': len(all_paths),
                'retrieval_time': retrieval_time,
                'traversal_pattern': traversal_pattern
            }
        )
    
    def _baseline_vector_retrieval(self, query: str) -> RetrievalResult:
        """Simple baseline vector similarity retrieval."""
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
            chunk_text = self.traversal_engine.navigator.get_chunk_text(chunk_id)
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
    
    def _extract_content_from_path(self, path: TraversalPath) -> List[str]:
        """Extract text content from nodes in a traversal path."""
        content = []
        
        for node_id, granularity in zip(path.nodes, path.granularity_levels):
            if granularity == GranularityLevel.CHUNK:
                text = self.traversal_engine.navigator.get_chunk_text(node_id)
                if text:
                    content.append(text)
            
            elif granularity == GranularityLevel.SENTENCE:
                sentence = self.kg.sentences.get(node_id)
                if sentence:
                    content.append(sentence.sentence_text)
            
            elif granularity == GranularityLevel.DOCUMENT:
                document = self.kg.documents.get(node_id)
                if document:
                    content.append(document.doc_summary)
        
        return content
    
    def _calculate_confidence_scores(self, path: TraversalPath, query_embedding: np.ndarray) -> List[float]:
        """Calculate confidence scores for nodes in path."""
        if not path or not path.nodes:
            return []
        
        scores = []
        for node_id, granularity in zip(path.nodes, path.granularity_levels):
            if granularity == GranularityLevel.CHUNK:
                node_embedding = self.kg.get_chunk_embedding(node_id)
                if node_embedding is not None:
                    similarity = cosine_similarity([query_embedding], [node_embedding])[0][0]
                    scores.append(float(similarity))
                else:
                    scores.append(0.0)
            
            elif granularity == GranularityLevel.SENTENCE:
                node_embedding = self.kg.get_sentence_embedding(node_id)
                if node_embedding is not None:
                    similarity = cosine_similarity([query_embedding], [node_embedding])[0][0]
                    scores.append(float(similarity))
                else:
                    scores.append(0.0)
            
            else:
                # Default score for other granularities
                scores.append(0.5)
        
        return scores


# Factory function for easy initialization
def create_retrieval_engine(knowledge_graph: KnowledgeGraph, config: Dict[str, Any], 
                          logger: Optional[logging.Logger] = None) -> RetrievalOrchestrator:
    """Factory function to create a retrieval engine."""
    return RetrievalOrchestrator(knowledge_graph, config, logger)
