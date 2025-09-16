#!/usr/bin/env python3
"""
Knowledge Graph Traversal Algorithm
==================================

Knowledge graph-guided traversal that prioritizes similarity to the current chunk.
Follows the natural flow of the knowledge graph structure rather than query similarity.
"""

import time
from typing import List, Dict, Any, Set, Optional, Tuple
from .base_algorithm import BaseRetrievalAlgorithm, RetrievalResult
from ..traversal import TraversalPath, GranularityLevel, ConnectionType


class KGTraversalAlgorithm(BaseRetrievalAlgorithm):
    """Algorithm 3: Current chunk similarity-guided traversal."""
    
    def __init__(self, knowledge_graph, config: Dict[str, Any], 
                 query_similarity_cache: Dict[str, float], logger=None):
        super().__init__(knowledge_graph, config, query_similarity_cache, logger)
        
        self.logger.info(f"KGTraversalAlgorithm initialized: max_hops={self.max_hops}, "
                        f"min_sentences={self.min_sentence_threshold}")
    
    def retrieve(self, query: str, anchor_chunk: str) -> RetrievalResult:
        """
        Traverse graph following nodes most similar to current chunk location.
        
        Args:
            query: The search query (used for initial anchor but not traversal decisions)
            anchor_chunk: Starting chunk for traversal
            
        Returns:
            RetrievalResult with knowledge graph-guided traversal results
        """
        start_time = time.time()
        
        self.logger.info(f"🔍 KGTraversalAlgorithm: Starting from anchor {anchor_chunk}")
        
        # Traversal state
        current_chunk = anchor_chunk
        visited_chunks: Set[str] = {anchor_chunk}
        extracted_sentences: List[str] = []
        path_nodes = [anchor_chunk]
        connection_types = []
        granularity_levels = [GranularityLevel.CHUNK]
        hop_count = 0
        
        # Extract sentences from anchor chunk initially
        anchor_sentences = self.get_chunk_sentences(anchor_chunk)
        extracted_sentences.extend(anchor_sentences)
        
        self.logger.info(f"   Extracted {len(anchor_sentences)} sentences from anchor")
        
        # Main traversal loop
        while len(extracted_sentences) < self.min_sentence_threshold and hop_count < self.max_hops:
            hop_count += 1
            
            self.logger.debug(f"🚶 Hop {hop_count}: Processing chunk {current_chunk}")
            
            # ALWAYS extract sentences from current chunk first (fix for Issue #1)
            if hop_count > 1:  # Skip anchor as already extracted
                chunk_sentences = self.get_chunk_sentences(current_chunk)
                newly_extracted = self.deduplicate_sentences(chunk_sentences, extracted_sentences)
                extracted_sentences.extend(newly_extracted)
                self.logger.info(f"📦 EXTRACTED: {len(newly_extracted)} new sentences from {current_chunk}")
            
            # Get hybrid connections (chunks + sentences within current chunk)
            hybrid_nodes = self.get_hybrid_connections(current_chunk)
            
            if not hybrid_nodes:
                self.logger.debug(f"   No hybrid connections found for {current_chunk}")
                break
            
            # Calculate similarities to CURRENT CHUNK for all connected nodes
            chunk_similarity_scores = []
            
            for node_id, node_type, query_sim in hybrid_nodes:
                if node_type == "chunk":
                    # Calculate chunk-to-chunk similarity
                    chunk_sim = self.calculate_chunk_similarity(current_chunk, node_id)
                    chunk_similarity_scores.append((node_id, node_type, chunk_sim))
                elif node_type == "sentence":
                    # For sentences, use query similarity as proxy for current chunk similarity
                    # (since sentences are within current chunk, they're inherently related)
                    chunk_similarity_scores.append((node_id, node_type, query_sim))
            
            # Sort by chunk similarity (not query similarity)
            chunk_similarity_scores.sort(key=lambda x: x[2], reverse=True)
            
            if not chunk_similarity_scores:
                self.logger.debug(f"   No similarity scores calculated")
                break
            
            best_node_id, best_node_type, best_chunk_similarity = chunk_similarity_scores[0]
            
            self.logger.info(f"   Best node: {best_node_id[:30]}... ({best_node_type}) "
                           f"chunk_sim={best_chunk_similarity:.3f}")
            
            if best_node_type == "sentence":
                # TERMINATION: Best node is sentence - no better chunks to explore
                self.logger.info(f"🎯 TERMINATION: Best node is sentence - optimal extraction point reached")
                break
            
            elif best_node_type == "chunk":
                # TRAVERSE: Move to the best connected chunk (with stronger revisit prevention)
                if best_node_id not in visited_chunks:
                    self.logger.info(f"🚶 TRAVERSE: Moving to chunk {best_node_id}")
                    current_chunk = best_node_id
                    visited_chunks.add(best_node_id)
                    path_nodes.append(best_node_id)
                    connection_types.append(ConnectionType.RAW_SIMILARITY)
                    granularity_levels.append(GranularityLevel.CHUNK)
                else:
                    self.logger.debug(f"   Best chunk {best_node_id} already visited - finding alternative")
                    # Find next best unvisited chunk using chunk similarities
                    next_chunk = self._find_next_chunk_by_chunk_similarity(chunk_similarity_scores, visited_chunks)
                    if next_chunk:
                        self.logger.info(f"🚶 TRAVERSE: Moving to alternative chunk {next_chunk}")
                        current_chunk = next_chunk
                        visited_chunks.add(next_chunk)
                        path_nodes.append(next_chunk)
                        connection_types.append(ConnectionType.RAW_SIMILARITY)
                        granularity_levels.append(GranularityLevel.CHUNK)
                    else:
                        self.logger.debug("   No more unvisited chunks available")
                        break
            else:
                self.logger.warning(f"   Unknown node type: {best_node_type}")
                break
        
        # Finalize results
        final_sentences = extracted_sentences[:self.max_results]
        confidence_scores = self.calculate_confidence_scores(final_sentences)
        sentence_sources = self.create_sentence_sources_mapping(final_sentences)
        
        # Create traversal path
        traversal_path = TraversalPath(
            nodes=path_nodes,
            connection_types=connection_types,
            granularity_levels=granularity_levels,
            total_hops=len(connection_types),
            is_valid=True,
            validation_errors=[]
        )
        
        # Calculate final score as average query similarity of extracted sentences
        final_score = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        processing_time = time.time() - start_time
        
        self.logger.info(f"✅ KGTraversalAlgorithm completed: {len(final_sentences)} sentences, "
                        f"{hop_count} hops in {processing_time:.3f}s")
        
        return RetrievalResult(
            algorithm_name="KGTraversal",
            traversal_path=traversal_path,
            retrieved_content=final_sentences,
            confidence_scores=confidence_scores,
            query=query,
            total_hops=traversal_path.total_hops,
            final_score=final_score,
            processing_time=processing_time,
            metadata={
                'anchor_chunk': anchor_chunk,
                'hops_completed': hop_count,
                'chunks_visited': len(visited_chunks),
                'extraction_strategy': 'chunk_similarity_priority'
            },
            extraction_metadata={
                'total_extracted': len(extracted_sentences),
                'final_count': len(final_sentences),
                'extraction_points': len([node for node, granularity in zip(path_nodes, granularity_levels) 
                                        if granularity == GranularityLevel.CHUNK])
            },
            sentence_sources=sentence_sources,
            query_similarities={sent: self.query_similarity_cache.get(self._find_sentence_id(sent), 0.0) 
                              for sent in final_sentences}
        )
    
    def _find_next_chunk_by_chunk_similarity(self, similarity_scores: List[Tuple[str, str, float]], 
                                           visited_chunks: Set[str]) -> Optional[str]:
        """Find the next best unvisited chunk from chunk similarity scores."""
        for node_id, node_type, similarity in similarity_scores:
            if node_type == "chunk" and node_id not in visited_chunks:
                return node_id
        return None
