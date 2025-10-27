#!/usr/bin/env python3
"""
Triangulation Geometric 3D Algorithm
====================================

TRUE geometric triangulation using PCA projection to 3D space.
Creates actual triangles with Query, Current, and Prospective chunks as vertices
in 3D Euclidean space, calculates geometric centroid, and measures Euclidean
distance to query point.

This improves upon the legacy centroid variant that averaged scalar similarities.
Here we work with actual 3D coordinates and geometric centroids.
"""

import time
import numpy as np
from typing import List, Dict, Any, Set, Optional, Tuple
from dataclasses import dataclass
from sklearn.decomposition import PCA

from .base_algorithm import BaseRetrievalAlgorithm, RetrievalResult
from ..traversal import TraversalPath, GranularityLevel, ConnectionType


@dataclass
class GeometricTriangleMetrics3D:
    """Triangle geometric properties in 3D space."""
    query_3d: np.ndarray
    current_3d: np.ndarray
    prospective_3d: np.ndarray
    centroid_3d: np.ndarray
    centroid_to_query_distance: float
    edge_lengths: Dict[str, float]
    node_id: str
    node_type: str


class TriangulationGeometric3DAlgorithm(BaseRetrievalAlgorithm):
    """Algorithm: TRUE geometric triangulation in 3D space via PCA."""

    def __init__(self, knowledge_graph, config: Dict[str, Any],
                 query_similarity_cache: Dict[str, float], logger=None,
                 shared_embedding_model=None,
                 shared_pca_reducer=None, shared_pca_coords_cache=None,
                 shared_pca_explained_variance=None):
        super().__init__(knowledge_graph, config, query_similarity_cache, logger, shared_embedding_model)

        # PCA components - use shared resources if provided (memory optimization)
        # This allows reusing the same PCA projection across multiple retrievals
        self.pca_reducer = shared_pca_reducer
        self.coords_3d_cache = shared_pca_coords_cache if shared_pca_coords_cache is not None else {}
        self.embedding_dimension = 3
        self.pca_explained_variance = shared_pca_explained_variance

        if self.pca_reducer is not None:
            self.logger.info(f"TriangulationGeometric3DAlgorithm initialized with SHARED PCA: "
                           f"max_hops={self.max_hops}, projection_dim={self.embedding_dimension}, "
                           f"cached_coords={len(self.coords_3d_cache)}")
        else:
            self.logger.info(f"TriangulationGeometric3DAlgorithm initialized: max_hops={self.max_hops}, "
                           f"projection_dim={self.embedding_dimension} (PCA will be computed on first use)")
    
    def _initialize_pca(self, query_embedding: np.ndarray, anchor_chunk: str):
        """
        Initialize PCA by fitting on all available embeddings from the knowledge graph.
        This is done once at the start of retrieval for consistency.
        """
        if self.pca_reducer is not None:
            return  # Already initialized
        
        self.logger.info("🔄 Initializing PCA projection to 3D space...")
        start_time = time.time()
        
        # Collect all embeddings from knowledge graph
        all_embeddings = [query_embedding]
        node_ids = ["__query__"]
        
        # Add chunk embeddings
        for chunk_id in self.kg.chunks.keys():
            emb = self.kg.get_chunk_embedding(chunk_id)
            if emb is not None:
                all_embeddings.append(emb)
                node_ids.append(chunk_id)
        
        # Add sentence embeddings
        for sentence_id in self.kg.sentences.keys():
            emb = self.kg.get_sentence_embedding(sentence_id)
            if emb is not None:
                all_embeddings.append(emb)
                node_ids.append(sentence_id)
        
        # Fit PCA
        embeddings_array = np.array(all_embeddings)
        self.pca_reducer = PCA(n_components=self.embedding_dimension, random_state=42)
        coords_3d = self.pca_reducer.fit_transform(embeddings_array)
        
        # Cache all 3D coordinates
        for i, node_id in enumerate(node_ids):
            self.coords_3d_cache[node_id] = coords_3d[i]
        
        self.pca_explained_variance = self.pca_reducer.explained_variance_ratio_.sum()
        
        init_time = time.time() - start_time
        self.logger.info(f"✅ PCA initialized: {len(all_embeddings)} embeddings projected to 3D "
                        f"({self.pca_explained_variance:.1%} variance explained) in {init_time:.2f}s")
    
    def _get_3d_coordinates(self, node_id: str, node_type: str) -> Optional[np.ndarray]:
        """
        Get 3D coordinates for a node. Uses cache if available, otherwise projects on-the-fly.
        """
        # Check cache first
        if node_id in self.coords_3d_cache:
            return self.coords_3d_cache[node_id]
        
        # Get embedding based on node type
        if node_type == "chunk":
            embedding = self.kg.get_chunk_embedding(node_id)
        elif node_type == "sentence":
            embedding = self.kg.get_sentence_embedding(node_id)
        else:
            return None
        
        if embedding is None:
            return None
        
        # Project to 3D using fitted PCA
        coords_3d = self.pca_reducer.transform([embedding])[0]
        
        # Cache for future use
        self.coords_3d_cache[node_id] = coords_3d
        
        return coords_3d
    
    def _calculate_geometric_triangle_3d(self, query_3d: np.ndarray, 
                                        current_3d: np.ndarray,
                                        prospective_3d: np.ndarray,
                                        node_id: str,
                                        node_type: str) -> GeometricTriangleMetrics3D:
        """
        Calculate TRUE geometric triangle properties in 3D Euclidean space.
        
        Args:
            query_3d: Query position in 3D (3,)
            current_3d: Current chunk position in 3D (3,)
            prospective_3d: Prospective node position in 3D (3,)
            node_id: ID of prospective node
            node_type: Type of prospective node
            
        Returns:
            GeometricTriangleMetrics3D with centroid and distance measurements
        """
        # Calculate geometric centroid (center of mass of triangle)
        centroid_3d = (query_3d + current_3d + prospective_3d) / 3.0
        
        # Euclidean distance from centroid to query point in 3D space
        centroid_to_query_distance = np.linalg.norm(centroid_3d - query_3d)
        
        # Calculate edge lengths for additional geometric analysis
        edge_lengths = {
            'query_to_current': np.linalg.norm(query_3d - current_3d),
            'query_to_prospective': np.linalg.norm(query_3d - prospective_3d),
            'current_to_prospective': np.linalg.norm(current_3d - prospective_3d)
        }
        
        return GeometricTriangleMetrics3D(
            query_3d=query_3d,
            current_3d=current_3d,
            prospective_3d=prospective_3d,
            centroid_3d=centroid_3d,
            centroid_to_query_distance=centroid_to_query_distance,
            edge_lengths=edge_lengths,
            node_id=node_id,
            node_type=node_type
        )
    
    def retrieve(self, query: str, anchor_chunk: str) -> RetrievalResult:
        """
        Execute TRUE geometric triangulation retrieval in 3D space.
        
        Args:
            query: The search query
            anchor_chunk: Starting chunk for traversal
            
        Returns:
            RetrievalResult with 3D geometric triangulation results
        """
        start_time = time.time()
        
        self.logger.info(f"🔺 TriangulationGeometric3D: Starting from anchor {anchor_chunk}")
        
        # Step 1: Get query embedding and initialize PCA
        query_embedding = self._get_query_embedding(query)
        self._initialize_pca(query_embedding, anchor_chunk)
        
        # Get 3D coordinates for query and anchor
        query_3d = self.coords_3d_cache["__query__"]
        
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
            
            # Get 3D coordinates for current chunk
            current_3d = self._get_3d_coordinates(current_chunk, "chunk")
            if current_3d is None:
                self.logger.warning(f"   Could not get 3D coordinates for current chunk {current_chunk}")
                break
            
            # Get hybrid connections
            hybrid_nodes = self.get_hybrid_connections(current_chunk)
            
            if not hybrid_nodes:
                self.logger.debug(f"   No hybrid connections found for {current_chunk}")
                break
            
            # Calculate geometric triangle metrics for all candidates
            candidate_metrics = []
            
            for node_id, node_type, query_sim in hybrid_nodes:
                # Get 3D coordinates for prospective node
                prospective_3d = self._get_3d_coordinates(node_id, node_type)
                
                if prospective_3d is None:
                    continue
                
                # Calculate geometric triangle in 3D space
                metrics = self._calculate_geometric_triangle_3d(
                    query_3d, current_3d, prospective_3d, node_id, node_type
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
        
        self.logger.info(f"✅ TriangulationGeometric3D completed: {len(final_sentences)} sentences, "
                        f"{hop_count} hops in {processing_time:.3f}s")
        
        return RetrievalResult(
            algorithm_name="TriangulationGeometric3D",
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
                'extraction_strategy': 'geometric_3d_triangulation',
                'early_stop_triggered': early_stop_triggered,
                'triangles_calculated': len(triangle_metrics_history),
                'pca_explained_variance': self.pca_explained_variance,
                'projection_dimension': self.embedding_dimension
            },
            extraction_metadata={
                'total_extracted': len(extracted_sentences),
                'final_count': len(final_sentences),
                'extraction_points': len([node for node, granularity in zip(path_nodes, granularity_levels) 
                                        if granularity == GranularityLevel.CHUNK]),
                'geometric_triangles': [{
                    'node_id': m.node_id,
                    'node_type': m.node_type,
                    'centroid_distance': float(m.centroid_to_query_distance),
                    'edge_lengths': m.edge_lengths
                } for m in triangle_metrics_history]
            },
            sentence_sources=sentence_sources,
            query_similarities={sent: self.query_similarity_cache.get(self._find_sentence_id(sent), 0.0) 
                              for sent in final_sentences}
        )
