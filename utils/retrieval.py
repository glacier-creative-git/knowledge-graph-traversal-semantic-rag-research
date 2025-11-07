#!/usr/bin/env python3
"""
Refactored Retrieval Orchestrator
================================

Multi-algorithm retrieval orchestrator that provides access to all semantic retrieval algorithms.
Preserves existing functionality while enabling comprehensive algorithm comparison.
"""

import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity

from .algos import (
    BasicRetrievalAlgorithm,
    QueryTraversalAlgorithm,
    SSGTraversalAlgorithm,
    TriangulationAverageAlgorithm,
    TriangulationGeometric3DAlgorithm,
    TriangulationGeometricFullDimAlgorithm,
    LLMGuidedTraversalAlgorithm,
    RetrievalResult
)
from .traversal import TraversalPath, GranularityLevel
from .semantic_similarity_graph import SemanticSimilarityGraph
from .models import EmbeddingModel


class RetrievalOrchestrator:
    """
    Multi-algorithm retrieval orchestrator for semantic similarity graph traversal.
    Provides unified interface to all retrieval algorithms and enables benchmarking.
    """

    def __init__(self, semantic_similarity_graph: SemanticSimilarityGraph, config: Dict[str, Any],
                 logger: Optional[logging.Logger] = None):
        """Initialize orchestrator with semantic similarity graph and configuration."""
        self.ssg = semantic_similarity_graph
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.traversal_config = config['retrieval']['semantic_traversal']

        # Initialize embedding model for query encoding
        self.model_name = list(config['models']['embedding_models'])[0]
        self.embedding_model = EmbeddingModel(self.model_name, config['system']['device'], logger)

        # Query similarity cache for sharing across algorithms
        self.query_similarity_cache: Dict[str, float] = {}

        # Shared PCA projection for triangulation_geometric_3d algorithm (memory optimization)
        # This is initialized once and reused across all retrievals to save memory
        self.shared_pca_reducer = None
        self.shared_pca_coords_3d_cache: Dict[str, np.ndarray] = {}
        self.shared_pca_explained_variance = None

        # Available algorithms
        self.algorithms = {
            "basic_retrieval": BasicRetrievalAlgorithm,
            "query_traversal": QueryTraversalAlgorithm,
            "ssg_traversal": SSGTraversalAlgorithm,
            "triangulation_average": TriangulationAverageAlgorithm,
            "triangulation_geometric_3d": TriangulationGeometric3DAlgorithm,
            "triangulation_geometric_fulldim": TriangulationGeometricFullDimAlgorithm,
            "llm_guided_traversal": LLMGuidedTraversalAlgorithm
        }

        self.logger.info(f"RetrievalOrchestrator initialized with {len(self.algorithms)} algorithms")
    
    def retrieve(self, query: str, algorithm_name: str = "triangulation_average", 
                 algorithm_params: Optional[Dict] = None) -> RetrievalResult:
        """
        Execute retrieval using specified algorithm.
        
        Args:
            query: Search query string
            algorithm_name: Algorithm to use ("basic_retrieval", "query_traversal",
                          "ssg_traversal", "triangulation_average",
                          "triangulation_geometric_3d", "triangulation_geometric_fulldim")
            algorithm_params: Optional algorithm-specific parameters
            
        Returns:
            RetrievalResult with algorithm-specific results
        """
        if algorithm_name not in self.algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm_name}. Available: {list(self.algorithms.keys())}")
        
        start_time = time.time()
        
        self.logger.info(f"🔍 Executing {algorithm_name} for query: '{query[:50]}...'")
        
        # Step 1: Find anchor and cache query similarities
        query_embedding = self.embedding_model.encode_single(query)
        anchor_chunk, self.query_similarity_cache = self._find_anchor_and_cache_similarities(query_embedding)
        
        if not anchor_chunk:
            self.logger.warning("No anchor point found")
            return self._create_empty_result(query, algorithm_name, "no_anchors")
        
        # Step 2: Initialize and execute algorithm
        algorithm_class = self.algorithms[algorithm_name]
        algorithm_params = algorithm_params or {}

        # Pass shared resources to algorithms for memory efficiency
        if algorithm_name == "triangulation_geometric_3d":
            algorithm = algorithm_class(
                semantic_similarity_graph=self.ssg,
                config=self.config,
                query_similarity_cache=self.query_similarity_cache,
                logger=self.logger,
                shared_embedding_model=self.embedding_model,  # Share embedding model!
                shared_pca_reducer=self.shared_pca_reducer,
                shared_pca_coords_cache=self.shared_pca_coords_3d_cache,
                shared_pca_explained_variance=self.shared_pca_explained_variance
            )
            # Update shared PCA resources after initialization (first run will create them)
            self.shared_pca_reducer = algorithm.pca_reducer
            self.shared_pca_coords_3d_cache = algorithm.coords_3d_cache
            self.shared_pca_explained_variance = algorithm.pca_explained_variance
        else:
            algorithm = algorithm_class(
                semantic_similarity_graph=self.ssg,
                config=self.config,
                query_similarity_cache=self.query_similarity_cache,
                logger=self.logger,
                shared_embedding_model=self.embedding_model  # Share embedding model for ALL algorithms!
            )
        
        # Execute retrieval
        result = algorithm.retrieve(query, anchor_chunk)
        
        total_time = time.time() - start_time
        
        self.logger.info(f"✅ {algorithm_name} completed: {len(result.retrieved_content)} sentences, "
                        f"{result.total_hops} hops, {total_time:.3f}s total")
        
        return result
    
    def benchmark_all_algorithms(self, query: str, 
                                algorithm_params: Optional[Dict[str, Dict]] = None) -> Dict[str, RetrievalResult]:
        """
        Run all algorithms for comprehensive comparison.
        
        Args:
            query: Search query string
            algorithm_params: Optional algorithm-specific parameters for each algorithm
            
        Returns:
            Dictionary mapping algorithm names to their results
        """
        self.logger.info(f"🏁 Running benchmark for all algorithms on query: '{query[:50]}...'")
        
        algorithm_params = algorithm_params or {}
        results = {}
        
        # Use shared anchor and cache for fair comparison
        query_embedding = self.embedding_model.encode_single(query)
        anchor_chunk, self.query_similarity_cache = self._find_anchor_and_cache_similarities(query_embedding)
        
        if not anchor_chunk:
            self.logger.warning("No anchor point found for benchmarking")
            return {name: self._create_empty_result(query, name, "no_anchors") 
                   for name in self.algorithms.keys()}
        
        self.logger.info(f"   Using shared anchor: {anchor_chunk}")
        self.logger.info(f"   Cached similarities for {len(self.query_similarity_cache)} nodes")
        
        # Run each algorithm
        for algorithm_name in self.algorithms.keys():
            try:
                self.logger.info(f"   Running {algorithm_name}...")

                algorithm_class = self.algorithms[algorithm_name]
                params = algorithm_params.get(algorithm_name, {})

                # Pass shared resources to algorithms for memory efficiency
                if algorithm_name == "triangulation_geometric_3d":
                    algorithm = algorithm_class(
                        semantic_similarity_graph=self.ssg,
                        config=self.config,
                        query_similarity_cache=self.query_similarity_cache,
                        logger=self.logger,
                        shared_embedding_model=self.embedding_model,  # Share embedding model!
                        shared_pca_reducer=self.shared_pca_reducer,
                        shared_pca_coords_cache=self.shared_pca_coords_3d_cache,
                        shared_pca_explained_variance=self.shared_pca_explained_variance
                    )
                    # Update shared PCA resources after initialization (first run will create them)
                    self.shared_pca_reducer = algorithm.pca_reducer
                    self.shared_pca_coords_3d_cache = algorithm.coords_3d_cache
                    self.shared_pca_explained_variance = algorithm.pca_explained_variance
                else:
                    algorithm = algorithm_class(
                        semantic_similarity_graph=self.ssg,
                        config=self.config,
                        query_similarity_cache=self.query_similarity_cache,
                        logger=self.logger,
                        shared_embedding_model=self.embedding_model  # Share embedding model for ALL algorithms!
                    )
                
                result = algorithm.retrieve(query, anchor_chunk)
                results[algorithm_name] = result
                
                self.logger.info(f"     ✅ {algorithm_name}: {len(result.retrieved_content)} sentences, "
                               f"{result.total_hops} hops, {result.processing_time:.3f}s")
                
            except Exception as e:
                self.logger.error(f"     ❌ {algorithm_name} failed: {str(e)}")
                results[algorithm_name] = self._create_empty_result(query, algorithm_name, f"error: {str(e)}")
        
        self.logger.info(f"🏁 Benchmark completed for {len(results)} algorithms")
        return results
    
    def _find_anchor_and_cache_similarities(self, query_embedding: np.ndarray) -> Tuple[Optional[str], Dict[str, float]]:
        """
        Find anchor point and cache query similarities for ALL nodes.
        Returns: (anchor_chunk_id, query_similarity_cache)
        """
        query_similarity_cache = {}
        chunk_similarities = []
        
        # Diagnostic counters
        chunks_processed = 0
        chunks_with_embeddings = 0
        sentences_processed = 0
        sentences_with_embeddings = 0
        
        # Compute similarities to all chunks
        for chunk_id in self.ssg.chunks.keys():
            chunks_processed += 1
            chunk_embedding = self.ssg.get_chunk_embedding(chunk_id)
            if chunk_embedding is not None:
                chunks_with_embeddings += 1
                similarity = cosine_similarity([query_embedding], [chunk_embedding])[0][0]
                query_similarity_cache[chunk_id] = float(similarity)
                chunk_similarities.append((chunk_id, float(similarity)))
        
        # Compute similarities to all sentences
        for sentence_id in self.ssg.sentences.keys():
            sentences_processed += 1
            sentence_embedding = self.ssg.get_sentence_embedding(sentence_id)
            if sentence_embedding is not None:
                sentences_with_embeddings += 1
                similarity = cosine_similarity([query_embedding], [sentence_embedding])[0][0]
                query_similarity_cache[sentence_id] = float(similarity)

        self.logger.info(f"📊 Similarity Cache Analysis:")
        self.logger.info(f"   Chunks: {chunks_with_embeddings}/{chunks_processed} have embeddings")
        self.logger.info(f"   Sentences: {sentences_with_embeddings}/{sentences_processed} have embeddings")
        self.logger.info(f"   Total cached similarities: {len(query_similarity_cache)}")
        
        # Find best anchor chunk
        chunk_similarities.sort(key=lambda x: x[1], reverse=True)
        threshold = self.traversal_config.get('similarity_threshold', 0.3)
        
        anchor_chunk = None
        for chunk_id, sim in chunk_similarities:
            if sim >= threshold:
                anchor_chunk = chunk_id
                break
        
        if not anchor_chunk and chunk_similarities:
            # Fallback to highest similarity chunk
            anchor_chunk = chunk_similarities[0][0]
            self.logger.warning(f"No anchor above threshold {threshold}, using best: {anchor_chunk}")
        
        return anchor_chunk, query_similarity_cache
    
    def _create_empty_result(self, query: str, algorithm_name: str, error_reason: str) -> RetrievalResult:
        """Create an empty result for failed retrievals."""
        return RetrievalResult(
            algorithm_name=algorithm_name,
            traversal_path=TraversalPath([], [], [], 0, False, [error_reason]),
            retrieved_content=[],
            confidence_scores=[],
            query=query,
            total_hops=0,
            final_score=0.0,
            processing_time=0.0,
            metadata={"error": error_reason}
        )
    
    # Backward compatibility method - preserves existing interface
    def hybrid_semantic_traversal_retrieval(self, query: str) -> RetrievalResult:
        """
        Backward compatibility method that uses the triangulation average algorithm.
        Preserves the existing interface while using the new algorithm framework.
        """
        return self.retrieve(query, "triangulation_average")


# Factory function for easy initialization (backward compatibility)
def create_retrieval_engine(semantic_similarity_graph: SemanticSimilarityGraph, config: Dict[str, Any], 
                          logger: Optional[logging.Logger] = None) -> RetrievalOrchestrator:
    """Factory function to create a retrieval orchestrator."""
    return RetrievalOrchestrator(semantic_similarity_graph, config, logger)


# Legacy class name for backward compatibility
class HybridTraversalEngine:
    """Legacy class that redirects to RetrievalOrchestrator for backward compatibility."""
    
    def __init__(self, semantic_similarity_graph: SemanticSimilarityGraph, config: Dict[str, Any], 
                 logger: Optional[logging.Logger] = None):
        self.orchestrator = RetrievalOrchestrator(semantic_similarity_graph, config, logger)
    
    def __getattr__(self, name):
        return getattr(self.orchestrator, name)
