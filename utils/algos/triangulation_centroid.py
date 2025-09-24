#!/usr/bin/env python3
"""
Triangulation Centroid Algorithm
==============================

Advanced geometric navigation using triangle centroid calculations in similarity space.
Implements enhanced early stopping with sentence-level triangle analysis.
"""

import time
import numpy as np
from typing import List, Dict, Any, Set, Optional, Tuple
from dataclasses import dataclass
from .base_algorithm import BaseRetrievalAlgorithm, RetrievalResult
from ..traversal import TraversalPath, GranularityLevel, ConnectionType


@dataclass
class TriangleMetrics:
    """Triangle geometric properties for centroid algorithm."""
    query_to_current: float
    query_to_potential: float
    current_to_potential: float
    centroid_position: float
    centroid_to_query_distance: float
    node_id: str
    node_type: str


class TriangulationCentroidAlgorithm(BaseRetrievalAlgorithm):
    """Algorithm 4: Triangle centroid-based semantic navigation."""
    
    def __init__(self, knowledge_graph, config: Dict[str, Any], 
                 query_similarity_cache: Dict[str, float], logger=None):
        super().__init__(knowledge_graph, config, query_similarity_cache, logger)

        self.logger.info(f"TriangulationCentroidAlgorithm initialized: max_hops={self.max_hops}, "
                        f"min_sentences={self.min_sentence_threshold}, "
                        f"early_stopping={self.enable_early_stopping}")
    
    def retrieve(self, query: str, anchor_chunk: str) -> RetrievalResult:
        """
        Execute triangle centroid-based retrieval with enhanced early stopping.
        
        Args:
            query: The search query
            anchor_chunk: Starting chunk for traversal
            
        Returns:
            RetrievalResult with triangle centroid-guided traversal results
        """
        start_time = time.time()
        
        self.logger.info(f"🔍 TriangulationCentroidAlgorithm: Starting from anchor {anchor_chunk}")
        
        # Traversal state
        current_chunk = anchor_chunk
        visited_chunks: Set[str] = {anchor_chunk}
        extracted_sentences: List[str] = []
        path_nodes = [anchor_chunk]
        connection_types = []
        granularity_levels = [GranularityLevel.CHUNK]
        triangle_metrics_history = []
        hop_count = 0
        early_stop_triggered = False
        best_triangle_quality = 0.0  # Track best triangle centroid position found so far

        # Extract sentences from anchor chunk initially
        anchor_sentences = self.get_chunk_sentences(anchor_chunk)
        extracted_sentences.extend(anchor_sentences)

        # Track best triangle quality from anchor sentences
        for sentence in anchor_sentences:
            sentence_id = self._find_sentence_id(sentence)
            if sentence_id and sentence_id in self.query_similarity_cache:
                # Calculate triangle for this sentence within anchor
                query_to_current = self.query_similarity_cache.get(anchor_chunk, 0.0)
                query_to_sentence = self.query_similarity_cache[sentence_id]
                current_to_sentence = query_to_sentence  # Sentence is within current chunk
                triangle_centroid = (query_to_current + query_to_sentence + current_to_sentence) / 3.0
                best_triangle_quality = max(best_triangle_quality, triangle_centroid)

        self.logger.info(f"   Extracted {len(anchor_sentences)} sentences from anchor (best_triangle_quality: {best_triangle_quality:.3f})")
        
        # Main traversal loop
        while len(extracted_sentences) < self.min_sentence_threshold and hop_count < self.max_hops:
            hop_count += 1
            
            self.logger.debug(f"🔺 Hop {hop_count}: Processing chunk {current_chunk}")
            
            # ALWAYS extract sentences from current chunk first (fix for Issue #1)
            if hop_count > 1:  # Skip anchor as already extracted
                chunk_sentences = self.get_chunk_sentences(current_chunk)
                newly_extracted = self.deduplicate_sentences(chunk_sentences, extracted_sentences)
                extracted_sentences.extend(newly_extracted)

                # Update best triangle quality from newly extracted sentences
                for sentence in newly_extracted:
                    sentence_id = self._find_sentence_id(sentence)
                    if sentence_id and sentence_id in self.query_similarity_cache:
                        # Calculate triangle for this sentence within current chunk
                        query_to_current = self.query_similarity_cache.get(current_chunk, 0.0)
                        query_to_sentence = self.query_similarity_cache[sentence_id]
                        current_to_sentence = query_to_sentence  # Sentence is within current chunk
                        triangle_centroid = (query_to_current + query_to_sentence + current_to_sentence) / 3.0
                        best_triangle_quality = max(best_triangle_quality, triangle_centroid)

                self.logger.info(f"📦 EXTRACTED: {len(newly_extracted)} new sentences from {current_chunk} "
                               f"(best_triangle_quality: {best_triangle_quality:.3f})")
            
            # Get hybrid connections (chunks + sentences within current chunk)
            hybrid_nodes = self.get_hybrid_connections(current_chunk)
            
            if not hybrid_nodes:
                self.logger.debug(f"   No hybrid connections found for {current_chunk}")
                break
            
            # Calculate triangle metrics for all potential moves
            triangle_metrics = self._calculate_triangle_metrics(query, current_chunk, hybrid_nodes)
            
            if not triangle_metrics:
                self.logger.debug(f"   No triangle metrics calculated")
                break
            
            # Sort by centroid-to-query distance (ascending - closer is better)
            triangle_metrics.sort(key=lambda x: x.centroid_to_query_distance)
            
            best_triangle = triangle_metrics[0]
            triangle_metrics_history.append(best_triangle)

            self.logger.info(f"   Best triangle: {best_triangle.node_id[:30]}... ({best_triangle.node_type}) "
                           f"centroid={best_triangle.centroid_position:.3f}, "
                           f"distance_to_query={best_triangle.centroid_to_query_distance:.3f}")

            # Multi-vector triangulation anchoring early stopping check
            if self.enable_early_stopping and len(extracted_sentences) >= 8:  # Need some sentences to compare
                # Calculate triangles between query, current_chunk, and each extracted sentence (multi-vector anchoring)
                extracted_sentence_triangles = []
                for sentence in extracted_sentences:
                    sentence_id = self._find_sentence_id(sentence)
                    if sentence_id and sentence_id in self.query_similarity_cache:
                        query_to_current = self.query_similarity_cache.get(current_chunk, 0.0)
                        query_to_sentence = self.query_similarity_cache[sentence_id]
                        # Estimate current_chunk to sentence similarity (cross-chunk semantic similarity)
                        current_to_sentence = self.calculate_chunk_similarity(current_chunk, self._find_sentence_chunk(sentence)) if self._find_sentence_chunk(sentence) else query_to_sentence * 0.8
                        triangle_centroid = (query_to_current + query_to_sentence + current_to_sentence) / 3.0
                        extracted_sentence_triangles.append(triangle_centroid)

                # Get best triangle quality from extracted sentences (multi-vector anchor constellation)
                best_extracted_triangle_quality = max(extracted_sentence_triangles) if extracted_sentence_triangles else 0.0

                # Get best chunk triangle option from potential destinations
                chunk_triangles = [t for t in triangle_metrics if t.node_type == "chunk"]
                if chunk_triangles and extracted_sentence_triangles:
                    best_chunk_triangle_quality = chunk_triangles[0].centroid_position  # Already sorted

                    # Multi-vector anchoring early stopping: extracted sentence constellation vs potential chunks
                    if best_extracted_triangle_quality > best_chunk_triangle_quality:
                        early_stop_triggered = True
                        self.logger.info(f"🎯 MULTI-VECTOR ANCHORING EARLY STOPPING: Best extracted triangle ({best_extracted_triangle_quality:.3f}) > "
                                       f"best potential chunk triangle ({best_chunk_triangle_quality:.3f}). "
                                       f"Stopping with {len(extracted_sentences)} sentences ({len(extracted_sentence_triangles)} anchor triangles).")
                        break
            
            if best_triangle.node_type == "sentence":
                # TERMINATION: Best triangle is sentence - no better chunks to explore
                self.logger.info(f"🎯 TERMINATION: Best triangle is sentence - optimal extraction point reached")
                break
            
            elif best_triangle.node_type == "chunk":
                # Update best triangle quality as we explore
                best_triangle_quality = max(best_triangle_quality, best_triangle.centroid_position)
                # TRAVERSE: Move to the chunk with the best triangle centroid (with stronger revisit prevention)
                if best_triangle.node_id not in visited_chunks:
                    self.logger.info(f"🚶 TRAVERSE: Moving to chunk {best_triangle.node_id}")
                    current_chunk = best_triangle.node_id
                    visited_chunks.add(best_triangle.node_id)
                    path_nodes.append(best_triangle.node_id)
                    connection_types.append(ConnectionType.RAW_SIMILARITY)
                    granularity_levels.append(GranularityLevel.CHUNK)
                else:
                    self.logger.debug(f"   Best chunk {best_triangle.node_id} already visited - finding alternative")
                    # Find next best unvisited chunk using triangle metrics
                    next_chunk = self._find_next_chunk_by_triangle(triangle_metrics, visited_chunks)
                    if next_chunk:
                        # Update triangle quality for alternative chunk too
                        for t in triangle_metrics:
                            if t.node_id == next_chunk:
                                best_triangle_quality = max(best_triangle_quality, t.centroid_position)
                                break
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
                self.logger.warning(f"   Unknown node type: {best_triangle.node_type}")
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
        
        # Calculate final score as average centroid position of all triangles
        final_score = (sum(t.centroid_position for t in triangle_metrics_history) / 
                      len(triangle_metrics_history) if triangle_metrics_history else 0.0)
        
        processing_time = time.time() - start_time
        
        self.logger.info(f"✅ TriangulationCentroidAlgorithm completed: {len(final_sentences)} sentences, "
                        f"{hop_count} hops in {processing_time:.3f}s "
                        f"{'(early stop)' if early_stop_triggered else ''}")
        
        return RetrievalResult(
            algorithm_name="TriangulationCentroid",
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
                'extraction_strategy': 'triangle_centroid_optimization',
                'early_stop_triggered': early_stop_triggered,
                'triangles_calculated': len(triangle_metrics_history)
            },
            extraction_metadata={
                'total_extracted': len(extracted_sentences),
                'final_count': len(final_sentences),
                'extraction_points': len([node for node, granularity in zip(path_nodes, granularity_levels) 
                                        if granularity == GranularityLevel.CHUNK]),
                'triangle_metrics': [t.__dict__ for t in triangle_metrics_history]
            },
            sentence_sources=sentence_sources,
            query_similarities={sent: self.query_similarity_cache.get(self._find_sentence_id(sent), 0.0) 
                              for sent in final_sentences}
        )
    
    def _calculate_triangle_metrics(self, query: str, current_chunk: str, 
                                  hybrid_nodes: List[Tuple[str, str, float]]) -> List[TriangleMetrics]:
        """
        Calculate triangle metrics for all potential nodes.
        Creates triangles with (query, current_chunk, potential_node).
        """
        triangle_metrics = []
        
        for node_id, node_type, query_similarity in hybrid_nodes:
            # Get the three triangle "sides" (similarities)
            query_to_current = self.query_similarity_cache.get(current_chunk, 0.0)
            query_to_potential = query_similarity  # Already provided from hybrid_nodes
            
            if node_type == "chunk":
                current_to_potential = self.calculate_chunk_similarity(current_chunk, node_id)
            else:  # sentence
                # For sentences within current chunk, use query similarity as proxy
                current_to_potential = query_similarity
            
            # Calculate triangle centroid (average of three similarities)
            centroid_position = (query_to_current + query_to_potential + current_to_potential) / 3.0
            
            # Calculate distance from centroid to query (distance from perfect similarity)
            centroid_to_query_distance = abs(centroid_position - 1.0)
            
            triangle_metric = TriangleMetrics(
                query_to_current=query_to_current,
                query_to_potential=query_to_potential,
                current_to_potential=current_to_potential,
                centroid_position=centroid_position,
                centroid_to_query_distance=centroid_to_query_distance,
                node_id=node_id,
                node_type=node_type
            )
            
            triangle_metrics.append(triangle_metric)
            
            self.logger.debug(f"   Triangle for {node_id[:20]}... ({node_type}): "
                            f"centroid={centroid_position:.3f}, distance={centroid_to_query_distance:.3f}")
        
        return triangle_metrics
    
    def _should_early_stop_sentence(self, sentence_triangle: TriangleMetrics, current_chunk: str) -> bool:
        """
        Enhanced early stopping check: sentence must have best triangle AND high similarity to current chunk.
        This represents finding the exact "needle in the haystack".
        """
        # Check if this sentence has higher similarity to current chunk than average
        sentence_to_chunk_similarity = sentence_triangle.current_to_potential
        chunk_similarity_threshold = self.similarity_threshold
        
        # Early stop if sentence has high similarity to current chunk
        # This indicates we've found a highly relevant sentence within a relevant chunk
        should_stop = sentence_to_chunk_similarity >= chunk_similarity_threshold
        
        self.logger.debug(f"   Early stop check: sentence_to_chunk_sim={sentence_to_chunk_similarity:.3f}, "
                         f"threshold={chunk_similarity_threshold:.3f}, should_stop={should_stop}")
        
        return should_stop
    
    def _find_next_chunk_by_triangle(self, triangle_metrics: List[TriangleMetrics], 
                                   visited_chunks: Set[str]) -> Optional[str]:
        """Find the next best unvisited chunk based on triangle metrics."""
        for triangle in triangle_metrics:
            if triangle.node_type == "chunk" and triangle.node_id not in visited_chunks:
                return triangle.node_id
        return None

    def _find_sentence_chunk(self, sentence: str) -> Optional[str]:
        """Find which chunk a sentence belongs to."""
        for chunk_id, chunk in self.kg.chunks.items():
            chunk_sentences = self.get_chunk_sentences(chunk_id)
            if sentence in chunk_sentences:
                return chunk_id
        return None
