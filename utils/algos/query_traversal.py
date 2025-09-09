#!/usr/bin/env python3
"""
Query Traversal Algorithm
========================

Query-guided graph traversal that always prioritizes similarity to the original query.
Implements the "drop crane" extraction strategy when sentence nodes are best.
"""

import time
from typing import List, Dict, Any, Set, Optional, Tuple
from .base_algorithm import BaseRetrievalAlgorithm, RetrievalResult
from ..traversal import TraversalPath, GranularityLevel, ConnectionType


class QueryTraversalAlgorithm(BaseRetrievalAlgorithm):
    """Algorithm 2: Query similarity-guided graph traversal."""
    
    def __init__(self, knowledge_graph, config: Dict[str, Any], 
                 query_similarity_cache: Dict[str, float], logger=None):
        super().__init__(knowledge_graph, config, query_similarity_cache, logger)
        
        self.logger.info(f"QueryTraversalAlgorithm initialized: max_hops={self.max_hops}, "
                        f"min_sentences={self.min_sentence_threshold}")
    
    def retrieve(self, query: str, anchor_chunk: str) -> RetrievalResult:
        """
        Traverse graph following nodes most similar to original query.
        
        Args:
            query: The search query
            anchor_chunk: Starting chunk for traversal
            
        Returns:
            RetrievalResult with query-guided traversal results
        """
        start_time = time.time()
        
        self.logger.info(f"🔍 QueryTraversalAlgorithm: Starting from anchor {anchor_chunk}")
        
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
            
            # Get hybrid connections (chunks + sentences within current chunk)
            hybrid_nodes = self.get_hybrid_connections(current_chunk)
            
            if not hybrid_nodes:
                self.logger.debug(f"   No hybrid connections found for {current_chunk}")
                break
            
            # Find node with highest query similarity
            best_node_id, best_node_type, best_similarity = hybrid_nodes[0]
            
            self.logger.info(f"   Best node: {best_node_id[:30]}... ({best_node_type}) "
                           f"query_sim={best_similarity:.3f}")
            
            if best_node_type == "sentence":
                # EXTRACT: Drop the crane and extract all sentences from current chunk
                self.logger.info(f"📦 EXTRACT triggered: Best node is sentence in current chunk")
                
                chunk_sentences = self.get_chunk_sentences(current_chunk)
                newly_extracted = self.deduplicate_sentences(chunk_sentences, extracted_sentences)
                extracted_sentences.extend(newly_extracted)
                
                self.logger.info(f"   Extracted {len(newly_extracted)} new sentences from {current_chunk}")
                
                # Find next best CHUNK to continue traversal
                next_chunk = self.find_next_chunk(hybrid_nodes, visited_chunks)
                
                if next_chunk:
                    current_chunk = next_chunk
                    visited_chunks.add(next_chunk)
                    path_nodes.append(next_chunk)
                    connection_types.append(ConnectionType.RAW_SIMILARITY)
                    granularity_levels.append(GranularityLevel.CHUNK)
                    self.logger.debug(f"   Moving to next chunk: {current_chunk}")
                else:
                    self.logger.debug("   No more unvisited chunks available")
                    break
            
            elif best_node_type == "chunk":
                # TRAVERSE: Move to the best connected chunk
                if best_node_id not in visited_chunks:
                    self.logger.info(f"🚶 TRAVERSE: Moving to chunk {best_node_id}")
                    current_chunk = best_node_id
                    visited_chunks.add(best_node_id)
                    path_nodes.append(best_node_id)
                    connection_types.append(ConnectionType.RAW_SIMILARITY)
                    granularity_levels.append(GranularityLevel.CHUNK)
                else:
                    self.logger.debug(f"   Best chunk {best_node_id} already visited")
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
        
        self.logger.info(f"✅ QueryTraversalAlgorithm completed: {len(final_sentences)} sentences, "
                        f"{hop_count} hops in {processing_time:.3f}s")
        
        return RetrievalResult(
            algorithm_name="QueryTraversal",
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
                'extraction_strategy': 'query_similarity_priority'
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
