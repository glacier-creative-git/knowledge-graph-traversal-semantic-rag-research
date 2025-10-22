#!/usr/bin/env python3
"""
Triangulation Average Algorithm
================================

Semantic navigation using averaged similarity scores (NOT true geometric triangulation).
Averages three cosine similarity values to create a "triangle quality" score.
Implements enhanced early stopping with sentence-level triangle analysis.

NOTE: This algorithm averages SCALAR similarity values, not geometric centroids.
For true geometric triangulation, see triangulation_geometric_3d.py or triangulation_geometric_fulldim.py
"""

import time
import numpy as np
from typing import List, Dict, Any, Set, Optional, Tuple
from dataclasses import dataclass
from .base_algorithm import BaseRetrievalAlgorithm, RetrievalResult
from ..traversal import TraversalPath, GranularityLevel, ConnectionType


@dataclass
class TriangleMetrics:
    """Triangle properties based on averaging similarity scores."""
    query_to_current: float
    query_to_potential: float
    current_to_potential: float
    average_similarity: float  # Average of three similarities
    distance_from_perfect: float  # Distance from 1.0 (perfect similarity)
    node_id: str
    node_type: str


class TriangulationAverageAlgorithm(BaseRetrievalAlgorithm):
    """Algorithm: Similarity-averaging based semantic navigation (similarity space triangulation)."""
    
    def __init__(self, knowledge_graph, config: Dict[str, Any],
                 query_similarity_cache: Dict[str, float], logger=None,
                 shared_embedding_model=None):
        super().__init__(knowledge_graph, config, query_similarity_cache, logger, shared_embedding_model)

        self.logger.info(f"TriangulationAverageAlgorithm initialized: max_hops={self.max_hops}, "
                        f"min_sentences={self.min_sentence_threshold}, "
                        f"early_stopping={self.enable_early_stopping}")
    
    def retrieve(self, query: str, anchor_chunk: str) -> RetrievalResult:
        """
        Execute similarity-averaging retrieval with enhanced early stopping.
        
        Averages three cosine similarity scores (query→current, query→prospective, current→prospective)
        to create a triangle quality score, then selects moves that maximize average similarity.
        
        Args:
            query: The search query
            anchor_chunk: Starting chunk for traversal
            
        Returns:
            RetrievalResult with similarity-averaging guided traversal results
        """
        start_time = time.time()
        
        self.logger.info(f"🔍 TriangulationAverageAlgorithm: Starting from anchor {anchor_chunk}")
        
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
        best_average_quality = 0.0  # Track best average similarity found so far

        # Extract sentences from anchor chunk initially
        anchor_sentences = self.get_chunk_sentences(anchor_chunk)
        extracted_sentences.extend(anchor_sentences)

        # Track best average quality from anchor sentences
        for sentence in anchor_sentences:
            sentence_id = self._find_sentence_id(sentence)
            if sentence_id and sentence_id in self.query_similarity_cache:
                # Calculate average similarity for this sentence within anchor
                query_to_current = self.query_similarity_cache.get(anchor_chunk, 0.0)
                query_to_sentence = self.query_similarity_cache[sentence_id]
                current_to_sentence = query_to_sentence  # Sentence is within current chunk
                average_similarity = (query_to_current + query_to_sentence + current_to_sentence) / 3.0
                best_average_quality = max(best_average_quality, average_similarity)

        self.logger.info(f"   Extracted {len(anchor_sentences)} sentences from anchor (best_avg_quality: {best_average_quality:.3f})")
        
        # Main traversal loop
        while len(extracted_sentences) < self.min_sentence_threshold and hop_count < self.max_hops:
            hop_count += 1
            
            self.logger.debug(f"🔺 Hop {hop_count}: Processing chunk {current_chunk}")
            
            # ALWAYS extract sentences from current chunk first
            if hop_count > 1:  # Skip anchor as already extracted
                chunk_sentences = self.get_chunk_sentences(current_chunk)
                newly_extracted = self.deduplicate_sentences(chunk_sentences, extracted_sentences)
                extracted_sentences.extend(newly_extracted)

                # Update best average quality from newly extracted sentences
                for sentence in newly_extracted:
                    sentence_id = self._find_sentence_id(sentence)
                    if sentence_id and sentence_id in self.query_similarity_cache:
                        # Calculate average similarity for this sentence
                        query_to_current = self.query_similarity_cache.get(current_chunk, 0.0)
                        query_to_sentence = self.query_similarity_cache[sentence_id]
                        current_to_sentence = query_to_sentence  # Sentence is within current chunk
                        average_similarity = (query_to_current + query_to_sentence + current_to_sentence) / 3.0
                        best_average_quality = max(best_average_quality, average_similarity)

                self.logger.info(f"📦 EXTRACTED: {len(newly_extracted)} new sentences from {current_chunk} "
                               f"(best_avg_quality: {best_average_quality:.3f})")
            
            # Get hybrid connections (chunks + sentences within current chunk)
            hybrid_nodes = self.get_hybrid_connections(current_chunk)
            
            if not hybrid_nodes:
                self.logger.debug(f"   No hybrid connections found for {current_chunk}")
                break
            
            # Calculate triangle metrics (similarity averages) for all potential moves
            triangle_metrics = self._calculate_triangle_metrics(query, current_chunk, hybrid_nodes)
            
            if not triangle_metrics:
                self.logger.debug(f"   No triangle metrics calculated")
                break
            
            # Sort by distance from perfect (ascending - closer to 1.0 is better)
            triangle_metrics.sort(key=lambda x: x.distance_from_perfect)
            
            best_triangle = triangle_metrics[0]
            triangle_metrics_history.append(best_triangle)

            self.logger.info(f"   Best triangle: {best_triangle.node_id[:30]}... ({best_triangle.node_type}) "
                           f"avg_sim={best_triangle.average_similarity:.3f}, "
                           f"distance_from_perfect={best_triangle.distance_from_perfect:.3f}")

            # Multi-vector anchoring early stopping check
            if self.enable_early_stopping and len(extracted_sentences) >= 8:  # Need some sentences to compare
                # Calculate averages for extracted sentences (multi-vector anchoring)
                extracted_sentence_averages = []
                for sentence in extracted_sentences:
                    sentence_id = self._find_sentence_id(sentence)
                    if sentence_id and sentence_id in self.query_similarity_cache:
                        query_to_current = self.query_similarity_cache.get(current_chunk, 0.0)
                        query_to_sentence = self.query_similarity_cache[sentence_id]
                        # Estimate current_chunk to sentence similarity
                        current_to_sentence = self.calculate_chunk_similarity(current_chunk, self._find_sentence_chunk(sentence)) if self._find_sentence_chunk(sentence) else query_to_sentence * 0.8
                        average_similarity = (query_to_current + query_to_sentence + current_to_sentence) / 3.0
                        extracted_sentence_averages.append(average_similarity)

                # Get best average quality from extracted sentences
                best_extracted_avg_quality = max(extracted_sentence_averages) if extracted_sentence_averages else 0.0

                # Get best chunk average from potential destinations
                chunk_triangles = [t for t in triangle_metrics if t.node_type == "chunk"]
                if chunk_triangles and extracted_sentence_averages:
                    best_chunk_avg_quality = chunk_triangles[0].average_similarity  # Already sorted

                    # Multi-vector anchoring early stopping: extracted sentences vs potential chunks
                    if best_extracted_avg_quality > best_chunk_avg_quality:
                        early_stop_triggered = True
                        self.logger.info(f"🎯 MULTI-VECTOR ANCHORING EARLY STOPPING: Best extracted avg ({best_extracted_avg_quality:.3f}) > "
                                       f"best potential chunk avg ({best_chunk_avg_quality:.3f}). "
                                       f"Stopping with {len(extracted_sentences)} sentences ({len(extracted_sentence_averages)} anchors).")
                        break
            
            if best_triangle.node_type == "sentence":
                # TERMINATION: Best option is sentence - optimal extraction point reached
                self.logger.info(f"🎯 TERMINATION: Best triangle is sentence - optimal extraction point reached")
                break
            
            elif best_triangle.node_type == "chunk":
                # Update best average quality as we explore
                best_average_quality = max(best_average_quality, best_triangle.average_similarity)
                # TRAVERSE: Move to the chunk with the best average similarity
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
                        # Update average quality for alternative chunk too
                        for t in triangle_metrics:
                            if t.node_id == next_chunk:
                                best_average_quality = max(best_average_quality, t.average_similarity)
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
        
        # Calculate final score as average similarity of all triangles
        final_score = (sum(t.average_similarity for t in triangle_metrics_history) / 
                      len(triangle_metrics_history) if triangle_metrics_history else 0.0)
        
        processing_time = time.time() - start_time
        
        self.logger.info(f"✅ TriangulationAverageAlgorithm completed: {len(final_sentences)} sentences, "
                        f"{hop_count} hops in {processing_time:.3f}s "
                        f"{'(early stop)' if early_stop_triggered else ''}")
        
        return RetrievalResult(
            algorithm_name="TriangulationAverage",
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
                'extraction_strategy': 'similarity_averaging_triangulation',
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
        Calculate triangle metrics (similarity averages) for all potential nodes.
        Creates "triangles" by averaging three similarity scores.
        """
        triangle_metrics = []
        
        for node_id, node_type, query_similarity in hybrid_nodes:
            # Get the three triangle "sides" (similarity scores)
            query_to_current = self.query_similarity_cache.get(current_chunk, 0.0)
            query_to_potential = query_similarity  # Already provided from hybrid_nodes
            
            if node_type == "chunk":
                current_to_potential = self.calculate_chunk_similarity(current_chunk, node_id)
            else:  # sentence
                # For sentences within current chunk, use query similarity as proxy
                current_to_potential = query_similarity
            
            # Calculate average similarity (what this algorithm actually does!)
            average_similarity = (query_to_current + query_to_potential + current_to_potential) / 3.0
            
            # Calculate distance from perfect similarity (1.0)
            distance_from_perfect = abs(average_similarity - 1.0)
            
            triangle_metric = TriangleMetrics(
                query_to_current=query_to_current,
                query_to_potential=query_to_potential,
                current_to_potential=current_to_potential,
                average_similarity=average_similarity,
                distance_from_perfect=distance_from_perfect,
                node_id=node_id,
                node_type=node_type
            )
            
            triangle_metrics.append(triangle_metric)
            
            self.logger.debug(f"   Triangle for {node_id[:20]}... ({node_type}): "
                            f"avg_sim={average_similarity:.3f}, distance={distance_from_perfect:.3f}")
        
        return triangle_metrics
    
    def _should_early_stop_sentence(self, sentence_triangle: TriangleMetrics, current_chunk: str) -> bool:
        """
        Enhanced early stopping check: sentence must have best average AND high similarity to current chunk.
        This represents finding the exact "needle in the haystack".
        """
        # Check if this sentence has higher similarity to current chunk than threshold
        sentence_to_chunk_similarity = sentence_triangle.current_to_potential
        chunk_similarity_threshold = self.similarity_threshold
        
        # Early stop if sentence has high similarity to current chunk
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
