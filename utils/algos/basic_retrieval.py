#!/usr/bin/env python3
"""
Basic Retrieval Algorithm
========================

Traditional RAG approach using pure similarity ranking without graph traversal.
Selects top-k most similar chunks and extracts all sentences from them.
"""

import time
from typing import List, Dict, Any
from .base_algorithm import BaseRetrievalAlgorithm, RetrievalResult
from ..traversal import TraversalPath, GranularityLevel


class BasicRetrievalAlgorithm(BaseRetrievalAlgorithm):
    """Algorithm 1: Pure similarity-based RAG without traversal."""
    
    def __init__(self, knowledge_graph, config: Dict[str, Any], 
                 query_similarity_cache: Dict[str, float], logger=None):
        super().__init__(knowledge_graph, config, query_similarity_cache, logger)
        
        # Algorithm-specific parameters
        self.top_k_chunks = self.traversal_config.get('top_k_chunks', 5)
        
        self.logger.info(f"BasicRetrievalAlgorithm initialized with top_k={self.top_k_chunks}")
    
    def retrieve(self, query: str, anchor_chunk: str) -> RetrievalResult:
        """
        Retrieve using pure similarity ranking without graph traversal.
        
        Args:
            query: The search query
            anchor_chunk: Starting chunk (not used in basic retrieval)
            
        Returns:
            RetrievalResult with top-k most similar chunks
        """
        start_time = time.time()
        
        self.logger.info(f"🔍 BasicRetrievalAlgorithm: Processing query '{query[:50]}...'")
        
        # Step 1: Calculate similarities to all chunks (use cached values)
        chunk_similarities = []
        chunks_with_cache = 0
        
        for chunk_id in self.kg.chunks.keys():
            if chunk_id in self.query_similarity_cache:
                chunks_with_cache += 1
                similarity = self.query_similarity_cache[chunk_id]
                chunk_similarities.append((chunk_id, similarity))
        
        self.logger.info(f"   Found {chunks_with_cache} chunks with cached similarities")
        
        # Step 2: Select top-k most similar chunks
        chunk_similarities.sort(key=lambda x: x[1], reverse=True)
        top_chunks = chunk_similarities[:self.top_k_chunks]
        
        self.logger.info(f"   Selected top {len(top_chunks)} chunks")
        
        # Step 3: Extract sentences from selected chunks
        extracted_sentences = []
        selected_chunks = []
        extraction_metadata = {}
        
        for chunk_id, similarity in top_chunks:
            chunk_sentences = self.get_chunk_sentences(chunk_id)
            newly_extracted = self.deduplicate_sentences(chunk_sentences, extracted_sentences)
            
            extracted_sentences.extend(newly_extracted)
            selected_chunks.append(chunk_id)
            extraction_metadata[chunk_id] = {
                'similarity_score': similarity,
                'sentences_extracted': len(newly_extracted),
                'total_chunk_sentences': len(chunk_sentences)
            }
            
            self.logger.debug(f"     Chunk {chunk_id}: {len(newly_extracted)} new sentences (sim: {similarity:.3f})")
        
        # Step 4: Apply result limit
        final_sentences = extracted_sentences[:self.max_results]
        
        # Step 5: Calculate confidence scores and metadata
        confidence_scores = self.calculate_confidence_scores(final_sentences)
        sentence_sources = self.create_sentence_sources_mapping(final_sentences)
        
        # Create traversal path for consistency (no actual traversal)
        traversal_path = TraversalPath(
            nodes=selected_chunks,
            connection_types=[],  # No connections in basic retrieval
            granularity_levels=[GranularityLevel.CHUNK] * len(selected_chunks),
            total_hops=0,  # No traversal
            is_valid=True,
            validation_errors=[]
        )
        
        # Calculate final score as average of top chunk similarities
        final_score = sum(sim for _, sim in top_chunks) / len(top_chunks) if top_chunks else 0.0
        
        processing_time = time.time() - start_time
        
        self.logger.info(f"✅ BasicRetrievalAlgorithm completed: {len(final_sentences)} sentences in {processing_time:.3f}s")
        
        return RetrievalResult(
            algorithm_name="BasicRetrieval",
            traversal_path=traversal_path,
            retrieved_content=final_sentences,
            confidence_scores=confidence_scores,
            query=query,
            total_hops=0,
            final_score=final_score,
            processing_time=processing_time,
            metadata={
                'top_k_chunks': self.top_k_chunks,
                'chunks_processed': len(chunk_similarities),
                'chunks_selected': len(top_chunks),
                'extraction_metadata': extraction_metadata
            },
            extraction_metadata={
                'total_extracted': len(extracted_sentences),
                'final_count': len(final_sentences),
                'chunks_used': len(selected_chunks)
            },
            sentence_sources=sentence_sources,
            query_similarities={sent: self.query_similarity_cache.get(self._find_sentence_id(sent), 0.0) 
                              for sent in final_sentences}
        )
