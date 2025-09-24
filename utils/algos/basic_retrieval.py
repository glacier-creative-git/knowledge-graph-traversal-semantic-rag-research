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

        # Sort chunks by similarity (highest first)
        chunk_similarities.sort(key=lambda x: x[1], reverse=True)

        # Step 2: Content-quality-anchored early stopping
        early_stop_triggered = False

        # Step 3: Select chunks with content-quality-anchored early stopping
        extracted_sentences = []
        selected_chunks = []
        extraction_metadata = {}
        chunk_index = 0
        best_sentence_similarity = 0.0  # Track best sentence found so far

        # Keep selecting chunks until we reach target sentence count
        while len(extracted_sentences) < self.max_results and chunk_index < len(chunk_similarities):
            chunk_id, chunk_similarity = chunk_similarities[chunk_index]

            # Content-quality-anchored early stopping check
            if self.enable_early_stopping and len(extracted_sentences) >= 5:  # Need some sentences to compare
                if chunk_similarity < best_sentence_similarity:
                    early_stop_triggered = True
                    self.logger.info(f"🎯 CONTENT-QUALITY EARLY STOPPING: Next chunk similarity ({chunk_similarity:.3f}) < "
                                   f"best extracted sentence similarity ({best_sentence_similarity:.3f}). "
                                   f"Stopping with {len(extracted_sentences)} sentences.")
                    break

            chunk_sentences = self.get_chunk_sentences(chunk_id)
            newly_extracted = self.deduplicate_sentences(chunk_sentences, extracted_sentences)

            if newly_extracted:  # Only add chunk if it contributes new sentences
                extracted_sentences.extend(newly_extracted)
                selected_chunks.append(chunk_id)
                extraction_metadata[chunk_id] = {
                    'similarity_score': chunk_similarity,
                    'sentences_extracted': len(newly_extracted),
                    'total_chunk_sentences': len(chunk_sentences)
                }

                # Update best sentence similarity from newly extracted sentences
                for sentence in newly_extracted:
                    sentence_id = self._find_sentence_id(sentence)
                    if sentence_id and sentence_id in self.query_similarity_cache:
                        sentence_sim = self.query_similarity_cache[sentence_id]
                        best_sentence_similarity = max(best_sentence_similarity, sentence_sim)

                self.logger.debug(f"     Chunk {chunk_id}: {len(newly_extracted)} new sentences (chunk_sim: {chunk_similarity:.3f}, "
                                f"best_sentence_sim: {best_sentence_similarity:.3f})")

            chunk_index += 1
        
        self.logger.info(f"   Selected {len(selected_chunks)} chunks to get {len(extracted_sentences)} sentences")
        
        # Step 3: Apply result limit
        final_sentences = extracted_sentences[:self.max_results]
        
        # Step 4: Calculate confidence scores and metadata
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
        
        # Calculate final score as average of selected chunk similarities
        if selected_chunks:
            # Get similarities for selected chunks
            selected_similarities = [extraction_metadata[chunk_id]['similarity_score'] 
                                   for chunk_id in selected_chunks]
            final_score = sum(selected_similarities) / len(selected_similarities)
        else:
            final_score = 0.0
        
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
                'chunks_processed': len(chunk_similarities),
                'chunks_selected': len(selected_chunks),
                'extraction_metadata': extraction_metadata,
                'target_sentences': self.max_results,
                'adaptive_selection': True,
                'early_stop_triggered': early_stop_triggered
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
