#!/usr/bin/env python3
"""
Semantic Similarity Graph Traversal Algorithm
===============================================

Simple, clean algorithm that traverses based on chunk-to-chunk similarity
with exploration-potential early stopping and sentence-overlap prevention.

Key differences from Query Traversal:
1. Jumps based on current-chunk-to-prospective-chunk similarity (not query similarity)
2. Early stopping when no prospective chunk beats previous leap's similarity
3. Cannot traverse to chunks containing already-extracted sentences
"""

import time
from typing import List, Dict, Any, Set, Optional
from .base_algorithm import BaseRetrievalAlgorithm, RetrievalResult
from ..traversal import TraversalPath, GranularityLevel, ConnectionType


class SSGTraversalAlgorithm(BaseRetrievalAlgorithm):
    """Algorithm 3B: Chunk-centric traversal with exploration-potential early stopping."""

    def __init__(self, semantic_similarity_graph, config: Dict[str, Any],
                 query_similarity_cache: Dict[str, float], logger=None,
                 shared_embedding_model=None):
        super().__init__(semantic_similarity_graph, config, query_similarity_cache, logger, shared_embedding_model)

        self.logger.info(f"SSGTraversalAlgorithm initialized: max_hops={self.max_hops}, "
                         f"min_sentences={self.min_sentence_threshold}")

    def retrieve(self, query: str, anchor_chunk: str) -> RetrievalResult:
        """
        Traverse graph following nodes most similar to CURRENT chunk location.

        Args:
            query: The search query (used for sentence quality tracking only)
            anchor_chunk: Starting chunk for traversal

        Returns:
            RetrievalResult with chunk-centric traversal results
        """
        start_time = time.time()

        self.logger.info(f"🔍 SSGTraversalAlgorithm: Starting from anchor {anchor_chunk}")

        # Traversal state
        current_chunk = anchor_chunk
        extracted_sentences: List[str] = []
        path_nodes = [anchor_chunk]
        connection_types = []
        granularity_levels = [GranularityLevel.CHUNK]
        hop_count = 0
        early_stop_triggered = False
        previous_leap_similarity = 0.0  # Track similarity of previous leap

        # Extract sentences from anchor chunk initially
        anchor_sentences = self.get_chunk_sentences(anchor_chunk)
        extracted_sentences.extend(anchor_sentences)

        self.logger.info(f"   Extracted {len(anchor_sentences)} sentences from anchor")

        # Main traversal loop
        while len(extracted_sentences) < self.min_sentence_threshold and hop_count < self.max_hops:
            hop_count += 1

            self.logger.debug(f"🚶 Hop {hop_count}: Processing chunk {current_chunk}")

            # Get all connected chunks (not hybrid - chunks only for this algorithm)
            connected_chunks = self._get_connected_chunks_only(current_chunk)

            if not connected_chunks:
                self.logger.debug(f"   No connected chunks found for {current_chunk}")
                break

            # Calculate chunk-to-chunk similarities for all prospective chunks
            chunk_similarities = []
            for chunk_id in connected_chunks:
                # Skip chunks that contain sentences we've already extracted
                if self._chunk_contains_extracted_sentences(chunk_id, extracted_sentences):
                    self.logger.debug(f"   Skipping {chunk_id} - contains already extracted sentences")
                    continue

                chunk_sim = self.calculate_chunk_similarity(current_chunk, chunk_id)
                chunk_similarities.append((chunk_id, chunk_sim))
                self.logger.debug(f"   Chunk {chunk_id}: similarity={chunk_sim:.3f}")

            if not chunk_similarities:
                self.logger.debug("   No viable chunks (all contain extracted sentences)")
                break

            # Sort by chunk-to-chunk similarity (highest first)
            chunk_similarities.sort(key=lambda x: x[1], reverse=True)
            best_chunk_id, best_chunk_similarity = chunk_similarities[0]

            self.logger.info(f"   Best chunk: {best_chunk_id} (chunk_sim={best_chunk_similarity:.3f})")

            # Exploration-potential early stopping check
            if self.enable_early_stopping and hop_count > 1:  # Need at least one leap to compare
                if best_chunk_similarity <= previous_leap_similarity:
                    early_stop_triggered = True
                    self.logger.info(f"🎯 EXPLORATION-POTENTIAL EARLY STOPPING: "
                                     f"Best available chunk similarity ({best_chunk_similarity:.3f}) <= "
                                     f"previous leap similarity ({previous_leap_similarity:.3f}). "
                                     f"No better exploration potential remaining.")
                    break

            # TRAVERSE: Move to the best chunk
            self.logger.info(f"🚶 TRAVERSE: Moving to chunk {best_chunk_id}")
            current_chunk = best_chunk_id
            path_nodes.append(best_chunk_id)
            connection_types.append(ConnectionType.RAW_SIMILARITY)
            granularity_levels.append(GranularityLevel.CHUNK)

            # Extract sentences from new chunk
            chunk_sentences = self.get_chunk_sentences(best_chunk_id)
            newly_extracted = self.deduplicate_sentences(chunk_sentences, extracted_sentences)
            extracted_sentences.extend(newly_extracted)

            # Update previous leap similarity for next iteration's early stopping check
            previous_leap_similarity = best_chunk_similarity

            self.logger.info(f"📦 EXTRACTED: {len(newly_extracted)} new sentences from {best_chunk_id} "
                             f"(leap_similarity: {best_chunk_similarity:.3f})")

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

        self.logger.info(f"✅ SSGTraversalAlgorithm completed: {len(final_sentences)} sentences, "
                         f"{hop_count} hops in {processing_time:.3f}s "
                         f"{'(early stop)' if early_stop_triggered else ''}")

        return RetrievalResult(
            algorithm_name="SSGTraversal",
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
                'chunks_visited': len(path_nodes),
                'extraction_strategy': 'chunk_similarity_priority_with_sentence_overlap_prevention',
                'early_stop_triggered': early_stop_triggered,
                'final_leap_similarity': previous_leap_similarity
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

    def _get_connected_chunks_only(self, chunk_id: str) -> List[str]:
        """Get only connected chunks (no sentences) for chunk-centric traversal."""
        chunk = self.ssg.chunks.get(chunk_id)
        if not chunk:
            self.logger.warning(f"Chunk {chunk_id} not found in semantic similarity graph")
            return []

        # Get all connected chunks (intra and inter-document)
        all_connected_chunks = chunk.intra_doc_connections + chunk.inter_doc_connections
        return all_connected_chunks

    def _chunk_contains_extracted_sentences(self, chunk_id: str, extracted_sentences: List[str]) -> bool:
        """
        Check if chunk contains any sentences that have already been extracted.
        This is the key constraint that prevents us from traversing to chunks with duplicate content.
        """
        chunk_sentences = self.get_chunk_sentences(chunk_id)
        extracted_set = set(extracted_sentences)

        # Return True if ANY sentence in this chunk has already been extracted
        for sentence in chunk_sentences:
            if sentence in extracted_set:
                return True

        return False
