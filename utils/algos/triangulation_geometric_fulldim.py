#!/usr/bin/env python3
"""
Triangulation Geometric Full-Dimensional Algorithm
==================================================

TRUE geometric triangulation in FULL embedding space (no dimensionality reduction).
Works directly with raw embeddings, creating triangles with Query, Current, and
Prospective nodes as vertices in the full embedding space (e.g., 768D, 1024D, etc.).

This is the most mathematically rigorous approach, preserving all information
from the embedding space, but operates in high dimensions that cannot be visualized.
"""

import time
import numpy as np
from typing import List, Dict, Any, Set, Optional, Tuple
from dataclasses import dataclass

from .base_algorithm import BaseRetrievalAlgorithm, RetrievalResult
from ..traversal import TraversalPath, GranularityLevel, ConnectionType


@dataclass
class GeometricTriangleMetricsFullDim:
    """Triangle geometric properties in full embedding space."""
    centroid_to_query_distance: float
    edge_lengths: Dict[str, float]
    node_id: str
    node_type: str


class TriangulationGeometricFullDimAlgorithm(BaseRetrievalAlgorithm):
    """Algorithm: TRUE geometric triangulation in full embedding space (no dimensionality reduction)."""
    
    def __init__(self, knowledge_graph, config: Dict[str, Any],
                 query_similarity_cache: Dict[str, float], logger=None,
                 shared_embedding_model=None):
        super().__init__(knowledge_graph, config, query_similarity_cache, logger, shared_embedding_model)
        
        # Auto-detect embedding dimension from knowledge graph
        self.embedding_dimension = self._detect_embedding_dimension()
        
        self.logger.info(f"TriangulationGeometricFullDimAlgorithm initialized: max_hops={self.max_hops}, "
                        f"embedding_dim={self.embedding_dimension}")
    
    def _detect_embedding_dimension(self) -> int:
        """
        Auto-detect embedding dimension from knowledge graph.
        Samples a chunk embedding to determine actual dimension.
        """
        # Try to get a sample embedding from chunks
        for chunk_id in list(self.kg.chunks.keys())[:5]:  # Check first 5 chunks
            sample_embedding = self.kg.get_chunk_embedding(chunk_id)
            if sample_embedding is not None:
                detected_dim = len(sample_embedding)
                self.logger.info(f"📐 Detected embedding dimension: {detected_dim}D")
                return detected_dim
        
        # Fallback to common dimensions if no embeddings found
        self.logger.warning("Could not detect embedding dimension, defaulting to 1024")
        return 1024
    
    def _get_embedding(self, node_id: str, node_type: str) -> Optional[np.ndarray]:
        """
        Get full embedding for a node based on its type.
        """
        if node_type == "chunk":
            return self.kg.get_chunk_embedding(node_id)
        elif node_type == "sentence":
            return self.kg.get_sentence_embedding(node_id)
        else:
            return None
    
    def _calculate_geometric_triangle_fulldim(self, query_emb: np.ndarray,
                                              current_emb: np.ndarray,
                                              prospective_emb: np.ndarray,
                                              node_id: str,
                                              node_type: str) -> GeometricTriangleMetricsFullDim:
        """
        Calculate TRUE geometric triangle properties in full-dimensional Euclidean space.

        Args:
            query_emb: Query embedding (full dimensionality)
            current_emb: Current chunk embedding (full dimensionality)
            prospective_emb: Prospective node embedding (full dimensionality)
            node_id: ID of prospective node
            node_type: Type of prospective node

        Returns:
            GeometricTriangleMetricsFullDim with centroid and distance measurements
        """
        # Calculate geometric centroid in full embedding space
        centroid_fulldim = (query_emb + current_emb + prospective_emb) / 3.0

        # Euclidean distance from centroid to query in full embedding space
        centroid_to_query_distance = np.linalg.norm(centroid_fulldim - query_emb)
        
        # Calculate edge lengths for additional geometric analysis
        edge_lengths = {
            'query_to_current': float(np.linalg.norm(query_emb - current_emb)),
            'query_to_prospective': float(np.linalg.norm(query_emb - prospective_emb)),
            'current_to_prospective': float(np.linalg.norm(current_emb - prospective_emb))
        }
        
        return GeometricTriangleMetricsFullDim(
            centroid_to_query_distance=float(centroid_to_query_distance),
            edge_lengths=edge_lengths,
            node_id=node_id,
            node_type=node_type
        )
    
    def retrieve(self, query: str, anchor_chunk: str) -> RetrievalResult:
        """
        Execute TRUE geometric triangulation retrieval in full embedding space.

        Args:
            query: The search query
            anchor_chunk: Starting chunk for traversal

        Returns:
            RetrievalResult with full-dimensional geometric triangulation results
        """
        start_time = time.time()

        self.logger.info(f"🔺 TriangulationGeometricFullDim: Starting from anchor {anchor_chunk}")
        
        # Get query embedding
        query_embedding = self._get_query_embedding(query)
        
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
        
        # Extract sentences from anchor chunk
        anchor_sentences = self.get_chunk_sentences(anchor_chunk)
        extracted_sentences.extend(anchor_sentences)
        
        self.logger.info(f"   Extracted {len(anchor_sentences)} sentences from anchor")
        
        # Main traversal loop
        while len(extracted_sentences) < self.min_sentence_threshold and hop_count < self.max_hops:
            hop_count += 1
            
            self.logger.debug(f"🔺 Hop {hop_count}: Processing chunk {current_chunk}")
            
            # Extract sentences from current chunk (skip anchor)
            if hop_count > 1:
                chunk_sentences = self.get_chunk_sentences(current_chunk)
                newly_extracted = self.deduplicate_sentences(chunk_sentences, extracted_sentences)
                extracted_sentences.extend(newly_extracted)
                
                self.logger.info(f"📦 EXTRACTED: {len(newly_extracted)} new sentences from {current_chunk}")
            
            # Get embedding for current chunk
            current_embedding = self._get_embedding(current_chunk, "chunk")
            if current_embedding is None:
                self.logger.warning(f"   Could not get embedding for current chunk {current_chunk}")
                break
            
            # Get hybrid connections
            hybrid_nodes = self.get_hybrid_connections(current_chunk)
            
            if not hybrid_nodes:
                self.logger.debug(f"   No hybrid connections found for {current_chunk}")
                break
            
            # Calculate geometric triangle metrics for all candidates in full embedding space
            candidate_metrics = []
            
            for node_id, node_type, query_sim in hybrid_nodes:
                # Get embedding for prospective node
                prospective_embedding = self._get_embedding(node_id, node_type)
                
                if prospective_embedding is None:
                    continue
                
                # Calculate geometric triangle in full embedding space
                metrics = self._calculate_geometric_triangle_fulldim(
                    query_embedding, current_embedding, prospective_embedding,
                    node_id, node_type
                )
                
                candidate_metrics.append(metrics)
                
                self.logger.debug(f"   Triangle {node_id[:20]}... ({node_type}): "
                                f"centroid_dist={metrics.centroid_to_query_distance:.4f}")
            
            if not candidate_metrics:
                self.logger.debug(f"   No valid triangle metrics calculated")
                break
            
            # Sort by centroid-to-query distance (ascending - closer is better)
            candidate_metrics.sort(key=lambda x: x.centroid_to_query_distance)
            
            best_triangle = candidate_metrics[0]
            triangle_metrics_history.append(best_triangle)
            
            self.logger.info(f"   Best triangle: {best_triangle.node_id[:30]}... ({best_triangle.node_type}) "
                           f"centroid_dist={best_triangle.centroid_to_query_distance:.4f}")
            
            # Decision logic: sentence vs chunk
            if best_triangle.node_type == "sentence":
                # TERMINATION: Best option is sentence - optimal extraction point reached
                self.logger.info(f"🎯 TERMINATION: Best triangle is sentence - stopping extraction")
                break
            
            elif best_triangle.node_type == "chunk":
                # TRAVERSE: Move to chunk with best geometric triangle
                if best_triangle.node_id not in visited_chunks:
                    self.logger.info(f"🚶 TRAVERSE: Moving to chunk {best_triangle.node_id}")
                    current_chunk = best_triangle.node_id
                    visited_chunks.add(best_triangle.node_id)
                    path_nodes.append(best_triangle.node_id)
                    connection_types.append(ConnectionType.RAW_SIMILARITY)
                    granularity_levels.append(GranularityLevel.CHUNK)
                else:
                    # Find next best unvisited chunk
                    next_chunk = None
                    for metrics in candidate_metrics[1:]:
                        if metrics.node_type == "chunk" and metrics.node_id not in visited_chunks:
                            next_chunk = metrics.node_id
                            break
                    
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
        
        # Calculate final score as average centroid distance
        final_score = (sum(m.centroid_to_query_distance for m in triangle_metrics_history) / 
                      len(triangle_metrics_history) if triangle_metrics_history else 0.0)
        
        processing_time = time.time() - start_time
        
        self.logger.info(f"✅ TriangulationGeometricFullDim completed: {len(final_sentences)} sentences, "
                        f"{hop_count} hops in {processing_time:.3f}s")

        return RetrievalResult(
            algorithm_name="TriangulationGeometricFullDim",
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
                'extraction_strategy': 'geometric_fulldim_triangulation',
                'early_stop_triggered': early_stop_triggered,
                'triangles_calculated': len(triangle_metrics_history),
                'embedding_dimension': self.embedding_dimension
            },
            extraction_metadata={
                'total_extracted': len(extracted_sentences),
                'final_count': len(final_sentences),
                'extraction_points': len([node for node, granularity in zip(path_nodes, granularity_levels) 
                                        if granularity == GranularityLevel.CHUNK]),
                'geometric_triangles': [{
                    'node_id': m.node_id,
                    'node_type': m.node_type,
                    'centroid_distance': m.centroid_to_query_distance,
                    'edge_lengths': m.edge_lengths
                } for m in triangle_metrics_history]
            },
            sentence_sources=sentence_sources,
            query_similarities={sent: self.query_similarity_cache.get(self._find_sentence_id(sent), 0.0) 
                              for sent in final_sentences}
        )
