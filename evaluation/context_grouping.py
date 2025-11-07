#!/usr/bin/env python3
"""
Semantic Context Grouping Strategies
====================================

Advanced context grouping for synthetic dataset generation using semantic traversal principles.
Implements multiple configurable strategies that mirror retrieval algorithm capabilities,
creating self-validating benchmarks where context quality directly reflects algorithm performance.

Key Features:
- Semantic traversal-based context generation
- Configurable strategy weights and parameters
- Sentence-level deduplication for sliding window overlap
- Cycle prevention using visited chunk tracking
- Multiple traversal patterns for comprehensive algorithm testing

Strategies:
1. Intra-Document: Semantic coherence within single documents
2. Inter-Document: Cross-document similarity navigation
3. Theme-Based: Cross-document theme overlap traversal
4. Sequential Multi-Hop: Structured reading with configurable cross-document hops
5. Semantic Similarity Graph Similarity: Pure exploration without constraints
"""

import random
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass

# Local imports
from utils.semantic_similarity_graph import SemanticSimilarityGraph


@dataclass
class ContextGroup:
    """
    Represents a semantically coherent group of chunks for question generation.
    
    Contains the traversal metadata to enable validation that retrieval algorithms
    can reconstruct the same semantic navigation patterns.
    """
    chunks: List[str]                    # Chunk texts in traversal order
    chunk_ids: List[str]                 # Chunk IDs for traversal validation
    sentences: List[str]                 # Deduplicated sentences from all chunks
    strategy: str                        # Strategy used to create this group
    metadata: Dict[str, Any]             # Strategy-specific metadata
    traversal_path: List[str]            # Full traversal path for validation
    

class ContextGroupingStrategy:
    """
    Base class for context grouping strategies.
    
    Each strategy implements semantic traversal logic that mirrors
    the capabilities of retrieval algorithms being benchmarked.
    """
    
    def __init__(self, ssg: SemanticSimilarityGraph, config: Dict[str, Any], 
                 strategy_config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize strategy with semantic similarity graph and configuration.
        
        Args:
            ssg: Semantic similarity graph instance with chunks, similarities, and themes
            config: Global system configuration
            strategy_config: Strategy-specific configuration parameters
            logger: Optional logger instance
        """
        self.ssg = ssg
        self.config = config
        self.strategy_config = strategy_config
        self.logger = logger or logging.getLogger(__name__)
        
        # Strategy parameters
        self.max_sentences = strategy_config.get('max_sentences', 10)
        self.enabled = strategy_config.get('enabled', True)
        self.weight = strategy_config.get('weight', 1.0)
        
    def generate_context_groups(self, num_groups: int) -> List[ContextGroup]:
        """
        Generate the specified number of context groups using this strategy.
        
        Args:
            num_groups: Number of context groups to generate
            
        Returns:
            List of ContextGroup objects created by this strategy
        """
        raise NotImplementedError("Subclasses must implement generate_context_groups")
    
    def _get_random_anchor_chunk(self, exclude_chunks: Optional[Set[str]] = None,
                                min_quality: Optional[float] = None) -> Optional[str]:
        """
        Get a random chunk ID as anchor point, with optional quality filtering.

        Args:
            exclude_chunks: Set of chunk IDs to exclude from selection
            min_quality: Minimum quality score required (uses high-quality chunks if None)

        Returns:
            Random chunk ID meeting quality criteria, or None if no chunks available
        """
        # Determine quality threshold
        if min_quality is None:
            # Use high-quality chunks by default (0.8 is a good threshold for anchors)
            min_quality = 0.8

        # Start with all chunks
        available_chunks = list(self.ssg.chunks.keys())

        # Apply exclusion filter
        if exclude_chunks:
            available_chunks = [c for c in available_chunks if c not in exclude_chunks]

        # Apply quality filter if quality scores are available
        quality_filtered_chunks = []
        for chunk_id in available_chunks:
            chunk = self.ssg.chunks[chunk_id]
            if chunk.quality_score is not None and chunk.quality_score >= min_quality:
                quality_filtered_chunks.append(chunk_id)

        # Use quality-filtered chunks if available, otherwise fall back to all available
        final_chunks = quality_filtered_chunks if quality_filtered_chunks else available_chunks

        if final_chunks:
            selected_chunk = random.choice(final_chunks)
            if quality_filtered_chunks:
                chunk = self.ssg.chunks[selected_chunk]
                self.logger.debug(f"Selected high-quality anchor: {selected_chunk} (quality: {chunk.quality_score:.3f})")
            else:
                self.logger.debug(f"Selected anchor from fallback pool: {selected_chunk} (no quality filter)")
            return selected_chunk

        return None

    def _filter_chunks_by_quality(self, chunk_ids: List[str],
                                 min_quality: float = 0.7) -> List[str]:
        """
        Filter chunk IDs by quality score.

        Args:
            chunk_ids: List of chunk IDs to filter
            min_quality: Minimum quality score required

        Returns:
            List of chunk IDs meeting quality criteria
        """
        quality_filtered = []
        for chunk_id in chunk_ids:
            if chunk_id in self.ssg.chunks:
                chunk = self.ssg.chunks[chunk_id]
                if chunk.quality_score is not None and chunk.quality_score >= min_quality:
                    quality_filtered.append(chunk_id)
                elif chunk.quality_score is None:
                    # Include chunks without quality scores (for backward compatibility)
                    quality_filtered.append(chunk_id)

        return quality_filtered

    def _get_chunk_sentences(self, chunk_id: str) -> List[str]:
        """
        Extract actual sentence texts from a chunk using semantic similarity graph sentence ID mapping.
        
        Leverages the existing SSG infrastructure:
        chunk.sentence_ids -> ssg.sentences[sentence_id].sentence_text
        
        Args:
            chunk_id: Chunk identifier in semantic similarity graph
            
        Returns:
            List of sentence text strings from the chunk
        """
        if chunk_id not in self.ssg.chunks:
            self.logger.warning(f"Chunk {chunk_id} not found in semantic similarity graph")
            return []
        
        chunk = self.ssg.chunks[chunk_id]
        sentence_texts = []
        
        # Extract sentence texts using sentence ID references
        for sentence_id in chunk.sentence_ids:
            if sentence_id in self.ssg.sentences:
                sentence_obj = self.ssg.sentences[sentence_id]
                sentence_texts.append(sentence_obj.sentence_text)
            else:
                self.logger.warning(f"Sentence ID {sentence_id} not found in semantic similarity graph")
        
        return sentence_texts
    
    def _deduplicate_sentences(self, new_sentences: List[str], 
                             existing_sentences: List[str]) -> List[str]:
        """Remove sentences that already exist in the group."""
        existing_set = set(existing_sentences)
        return [s for s in new_sentences if s not in existing_set]
    
    def _get_most_similar_chunks(self, anchor_chunk_id: str,
                               exclude_chunks: Set[str],
                               cross_document_only: bool = False,
                               existing_sentences: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """
        Get chunks most similar to anchor using PRE-COMPUTED similarities from semantic similarity graph.

        Args:
            anchor_chunk_id: Reference chunk for similarity lookup
            exclude_chunks: Set of chunk IDs to exclude
            cross_document_only: If True, only return chunks from different documents
            existing_sentences: If provided, exclude chunks that contain any of these sentences

        Returns:
            List of (chunk_id, similarity_score) tuples, sorted by similarity
        """
        if anchor_chunk_id not in self.ssg.chunks:
            return []

        anchor_chunk = self.ssg.chunks[anchor_chunk_id]
        similarities = []

        # Get candidate chunks based on cross_document_only flag
        if cross_document_only:
            candidate_chunks = anchor_chunk.inter_doc_connections
        else:
            candidate_chunks = anchor_chunk.intra_doc_connections + anchor_chunk.inter_doc_connections

        # Look up PRE-COMPUTED similarity scores from semantic similarity graph
        for chunk_id in candidate_chunks:
            if chunk_id not in exclude_chunks:
                # If existing_sentences provided, filter out chunks with sentence overlap
                if existing_sentences and self._has_sentence_overlap(chunk_id, existing_sentences):
                    continue

                similarity_score = anchor_chunk.connection_scores.get(chunk_id, 0.0)
                similarities.append((chunk_id, similarity_score))

        return sorted(similarities, key=lambda x: x[1], reverse=True)
    
    def _calculate_chunk_similarity(self, chunk1_id: str, chunk2_id: str) -> float:
        """
        Get pre-computed similarity between two chunks from semantic similarity graph connections.
        
        Uses the existing connection_scores infrastructure instead of recalculating.
        """
        if chunk1_id not in self.ssg.chunks:
            return 0.0
            
        chunk1 = self.ssg.chunks[chunk1_id]
        return chunk1.connection_scores.get(chunk2_id, 0.0)
    
    def _has_sentence_overlap(self, chunk_id: str, existing_sentences: List[str]) -> bool:
        """Check if chunk contains any sentences already in the group."""
        chunk_sentences = self._get_chunk_sentences(chunk_id)
        existing_set = set(existing_sentences)
        return any(sentence in existing_set for sentence in chunk_sentences)


class IntraDocumentStrategy(ContextGroupingStrategy):
    """
    Intra-document semantic traversal strategy.
    
    Anchors to a random chunk within a document, then traverses to the most
    semantically similar chunks within the same document, skipping chunks
    that contain duplicate sentences from sliding window overlap.
    """
    
    def generate_context_groups(self, num_groups: int) -> List[ContextGroup]:
        """Generate intra-document context groups using semantic similarity."""
        context_groups = []
        
        for group_idx in range(num_groups):
            # Get random anchor chunk
            anchor_chunk_id = self._get_random_anchor_chunk()
            if not anchor_chunk_id:
                self.logger.warning(f"No anchor chunk available for group {group_idx}")
                break
                
            visited_chunks = {anchor_chunk_id}
            sentences = []
            traversal_path = [anchor_chunk_id]
            chunk_texts = []
            
            # Extract sentences from anchor
            anchor_sentences = self._get_chunk_sentences(anchor_chunk_id)
            sentences.extend(anchor_sentences)
            chunk_texts.append(self.ssg.chunks[anchor_chunk_id].chunk_text)
            
            self.logger.debug(f"🏁 Anchor chunk {anchor_chunk_id}: {len(anchor_sentences)} sentences")
            
            current_chunk = anchor_chunk_id
            anchor_doc = self.ssg.chunks[anchor_chunk_id].source_document
            
            # Traverse within document until max_sentences reached
            safety_counter = 0
            max_safety_hops = 50  # Prevent infinite loops
            
            while len(sentences) < self.max_sentences and safety_counter < max_safety_hops:
                safety_counter += 1
                
                # Get similar chunks within same document
                similar_chunks = self._get_most_similar_chunks(
                    current_chunk, visited_chunks, cross_document_only=False
                )
                
                # Filter to same document only
                same_doc_chunks = [
                    (chunk_id, sim) for chunk_id, sim in similar_chunks
                    if self.ssg.chunks[chunk_id].source_document == anchor_doc
                ]
                
                if not same_doc_chunks:
                    self.logger.debug(f"🔚 No more similar chunks in document at {len(sentences)} sentences")
                    break
                
                # Find next chunk without sentence overlap
                next_chunk = None
                for chunk_id, similarity in same_doc_chunks:
                    if not self._has_sentence_overlap(chunk_id, sentences):
                        next_chunk = chunk_id
                        break
                
                if not next_chunk:
                    self.logger.debug(f"🔚 No chunks without sentence overlap at {len(sentences)} sentences")
                    break
                
                # Add sentences from next chunk
                chunk_sentences = self._get_chunk_sentences(next_chunk)
                new_sentences = self._deduplicate_sentences(chunk_sentences, sentences)
                
                # Check if adding these sentences would exceed limit
                if len(sentences) + len(new_sentences) > self.max_sentences:
                    # Take only enough sentences to reach the limit
                    remaining_capacity = self.max_sentences - len(sentences)
                    new_sentences = new_sentences[:remaining_capacity]
                
                sentences.extend(new_sentences)
                
                # Update traversal state
                visited_chunks.add(next_chunk)
                traversal_path.append(next_chunk)
                chunk_texts.append(self.ssg.chunks[next_chunk].chunk_text)
                current_chunk = next_chunk
                
                self.logger.debug(f"➡️ Added chunk {next_chunk}: +{len(new_sentences)} sentences (total: {len(sentences)})")
            
            # Create context group with proper sentence tracking
            final_sentence_count = min(len(sentences), self.max_sentences)
            
            context_group = ContextGroup(
                chunks=chunk_texts,
                chunk_ids=traversal_path,
                sentences=sentences[:final_sentence_count],
                strategy="intra_document",
                metadata={
                    'anchor_document': anchor_doc,
                    'chunks_traversed': len(traversal_path),
                    'final_sentence_count': final_sentence_count,
                    'safety_hops_used': safety_counter,
                    'termination_reason': 'sentence_limit' if len(sentences) >= self.max_sentences else 'no_more_chunks'
                },
                traversal_path=traversal_path
            )
            
            self.logger.debug(f"✅ Intra-document group {group_idx}: {len(chunk_texts)} chunks, {final_sentence_count} sentences")
            context_groups.append(context_group)
        
        return context_groups


class InterDocumentStrategy(ContextGroupingStrategy):
    """
    Inter-document semantic traversal strategy.
    
    Anchors to a random chunk, then traverses to the most semantically similar
    chunks from OTHER documents, creating cross-domain semantic connections
    perfect for testing triangulation algorithm capabilities.
    """
    
    def generate_context_groups(self, num_groups: int) -> List[ContextGroup]:
        """Generate inter-document context groups using cross-document similarity."""
        context_groups = []
        
        for group_idx in range(num_groups):
            # Get random anchor chunk
            anchor_chunk_id = self._get_random_anchor_chunk()
            if not anchor_chunk_id:
                self.logger.warning(f"No anchor chunk available for group {group_idx}")
                break
                
            visited_chunks = {anchor_chunk_id}
            sentences = []
            traversal_path = [anchor_chunk_id]
            chunk_texts = []
            
            # Extract sentences from anchor
            anchor_sentences = self._get_chunk_sentences(anchor_chunk_id)
            sentences.extend(anchor_sentences)
            chunk_texts.append(self.ssg.chunks[anchor_chunk_id].chunk_text)
            
            self.logger.debug(f"🏁 Inter-doc anchor {anchor_chunk_id}: {len(anchor_sentences)} sentences")
            
            current_chunk = anchor_chunk_id
            
            # Traverse across documents until max_sentences reached
            safety_counter = 0
            max_safety_hops = 50  # Prevent infinite loops
            
            while len(sentences) < self.max_sentences and safety_counter < max_safety_hops:
                safety_counter += 1
                
                # Get similar chunks from OTHER documents only
                similar_chunks = self._get_most_similar_chunks(
                    current_chunk, visited_chunks, cross_document_only=True
                )
                
                if not similar_chunks:
                    self.logger.debug(f"🔚 No more cross-document chunks at {len(sentences)} sentences")
                    break
                
                # Find next chunk without sentence overlap
                next_chunk = None
                for chunk_id, similarity in similar_chunks:
                    if not self._has_sentence_overlap(chunk_id, sentences):
                        next_chunk = chunk_id
                        break
                
                if not next_chunk:
                    self.logger.debug(f"🔚 No cross-document chunks without overlap at {len(sentences)} sentences")
                    break
                
                # Add sentences from next chunk
                chunk_sentences = self._get_chunk_sentences(next_chunk)
                new_sentences = self._deduplicate_sentences(chunk_sentences, sentences)
                
                # Check if adding these sentences would exceed limit
                if len(sentences) + len(new_sentences) > self.max_sentences:
                    # Take only enough sentences to reach the limit
                    remaining_capacity = self.max_sentences - len(sentences)
                    new_sentences = new_sentences[:remaining_capacity]
                
                sentences.extend(new_sentences)
                
                # Update traversal state
                visited_chunks.add(next_chunk)
                traversal_path.append(next_chunk)
                chunk_texts.append(self.ssg.chunks[next_chunk].chunk_text)
                current_chunk = next_chunk
                
                self.logger.debug(f"➡️ Added cross-doc chunk {next_chunk}: +{len(new_sentences)} sentences (total: {len(sentences)})")
            
            # Calculate cross-document diversity
            documents_visited = set(
                self.ssg.chunks[chunk_id].source_document 
                for chunk_id in traversal_path
            )
            
            # Create context group with proper sentence tracking
            final_sentence_count = min(len(sentences), self.max_sentences)
            
            context_group = ContextGroup(
                chunks=chunk_texts,
                chunk_ids=traversal_path,
                sentences=sentences[:final_sentence_count],
                strategy="inter_document",
                metadata={
                    'documents_visited': list(documents_visited),
                    'cross_document_hops': len(traversal_path) - 1,
                    'final_sentence_count': final_sentence_count,
                    'safety_hops_used': safety_counter,
                    'termination_reason': 'sentence_limit' if len(sentences) >= self.max_sentences else 'no_more_chunks'
                },
                traversal_path=traversal_path
            )
            
            self.logger.debug(f"✅ Inter-document group {group_idx}: {len(chunk_texts)} chunks, {final_sentence_count} sentences, {len(documents_visited)} documents")
            context_groups.append(context_group)
        
        return context_groups


class ThemeBasedStrategy(ContextGroupingStrategy):
    """
    Theme-based cross-document traversal strategy.
    
    Anchors to a random chunk, identifies its themes, then traverses to chunks
    from other documents that share theme overlap, ranking by theme count
    and then by similarity. Falls back to inter-document if no themes overlap.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fallback_to_inter_document = self.strategy_config.get(
            'fallback_to_inter_document', True
        )
    
    def generate_context_groups(self, num_groups: int) -> List[ContextGroup]:
        """Generate theme-based context groups using theme overlap."""
        context_groups = []
        
        for group_idx in range(num_groups):
            # Get random anchor chunk
            anchor_chunk_id = self._get_random_anchor_chunk()
            if not anchor_chunk_id:
                self.logger.warning(f"No anchor chunk available for group {group_idx}")
                break
                
            visited_chunks = {anchor_chunk_id}
            sentences = []
            traversal_path = [anchor_chunk_id]
            chunk_texts = []
            
            # Extract sentences from anchor
            anchor_sentences = self._get_chunk_sentences(anchor_chunk_id)
            sentences.extend(anchor_sentences)
            chunk_texts.append(self.ssg.chunks[anchor_chunk_id].chunk_text)
            
            # Get anchor themes
            anchor_themes = self._get_chunk_themes(anchor_chunk_id)
            current_chunk = anchor_chunk_id
            theme_fallback_used = False
            
            self.logger.debug(f"🏁 Theme-based anchor {anchor_chunk_id}: {len(anchor_sentences)} sentences, themes: {anchor_themes}")
            
            # Traverse based on theme overlap until max_sentences reached
            safety_counter = 0
            max_safety_hops = 50  # Prevent infinite loops
            
            while len(sentences) < self.max_sentences and safety_counter < max_safety_hops:
                safety_counter += 1

                # Get current document for theme similarity lookup
                current_doc = self.ssg.chunks[current_chunk].source_document

                # Get theme-similar documents using new embedding-based approach
                theme_similar_docs = self.ssg.get_theme_similar_documents_by_title(current_doc)

                next_chunk = None
                theme_candidates_found = False

                # Try each theme-similar document in order of theme similarity
                for target_doc_id in theme_similar_docs:
                    # Convert doc_id back to document title for chunk matching
                    target_doc = self.ssg.get_document_title_by_id(target_doc_id)
                    if target_doc == current_doc:
                        continue  # Skip same document

                    # Get existing connections from current chunk (both intra and inter)
                    current_chunk_obj = self.ssg.chunks[current_chunk]
                    all_connections = current_chunk_obj.intra_doc_connections + current_chunk_obj.inter_doc_connections

                    # Filter connections to only include chunks from the target document
                    target_doc_connections = [
                        chunk_id for chunk_id in all_connections
                        if (chunk_id in self.ssg.chunks and
                            self.ssg.chunks[chunk_id].source_document == target_doc and
                            chunk_id not in visited_chunks)
                    ]

                    if not target_doc_connections:
                        continue  # No existing connections to this theme-similar document

                    # Find best connected chunk without sentence overlap
                    best_connected_chunk = None
                    best_similarity = 0.0

                    for connected_chunk_id in target_doc_connections:
                        if not self._has_sentence_overlap(connected_chunk_id, sentences):
                            # Get pre-computed similarity score from connection_scores
                            similarity = current_chunk_obj.connection_scores.get(connected_chunk_id, 0.0)
                            if similarity > best_similarity:
                                best_similarity = similarity
                                best_connected_chunk = connected_chunk_id

                    if best_connected_chunk:
                        next_chunk = best_connected_chunk
                        theme_candidates_found = True

                        # Convert current_doc title to doc_id for theme similarity score lookup
                        current_doc_id = None
                        for doc_id, doc in self.ssg.documents.items():
                            if doc.title == current_doc:
                                current_doc_id = doc_id
                                break
                        theme_sim_score = self.ssg.get_theme_similarity_score(current_doc_id, target_doc_id) if current_doc_id else 0.0

                        self.logger.debug(f"🎯 Theme bridge: {current_doc} -> {target_doc} "
                                        f"(theme_sim: {theme_sim_score:.3f}, chunk_sim: {best_similarity:.3f}, connected: True)")
                        break

                # Fallback to inter-document similarity if no theme candidates found
                if not next_chunk and self.fallback_to_inter_document:
                    fallback_candidates = self._get_most_similar_chunks(
                        current_chunk, visited_chunks, cross_document_only=True
                    )

                    if fallback_candidates:
                        for chunk_id, similarity in fallback_candidates:
                            if not self._has_sentence_overlap(chunk_id, sentences):
                                next_chunk = chunk_id
                                theme_fallback_used = True
                                self.logger.debug(f"🔄 Theme fallback: chunk similarity {similarity:.3f}")
                                break

                if not next_chunk:
                    self.logger.debug(f"🔚 No more theme or fallback candidates at {len(sentences)} sentences")
                    break
                
                # Add sentences from next chunk
                chunk_sentences = self._get_chunk_sentences(next_chunk)
                new_sentences = self._deduplicate_sentences(chunk_sentences, sentences)
                
                # Check if adding these sentences would exceed limit
                if len(sentences) + len(new_sentences) > self.max_sentences:
                    # Take only enough sentences to reach the limit
                    remaining_capacity = self.max_sentences - len(sentences)
                    new_sentences = new_sentences[:remaining_capacity]
                
                sentences.extend(new_sentences)
                
                # Update traversal state
                visited_chunks.add(next_chunk)
                traversal_path.append(next_chunk)
                chunk_texts.append(self.ssg.chunks[next_chunk].chunk_text)
                current_chunk = next_chunk
                
                self.logger.debug(f"➡️ Added theme chunk {next_chunk}: +{len(new_sentences)} sentences (total: {len(sentences)})")
            
            # Create context group with proper sentence tracking
            final_sentence_count = min(len(sentences), self.max_sentences)
            
            context_group = ContextGroup(
                chunks=chunk_texts,
                chunk_ids=traversal_path,
                sentences=sentences[:final_sentence_count],
                strategy="theme_based",
                metadata={
                    'anchor_themes': anchor_themes,
                    'theme_fallback_used': theme_fallback_used,
                    'final_sentence_count': final_sentence_count,
                    'safety_hops_used': safety_counter,
                    'termination_reason': 'sentence_limit' if len(sentences) >= self.max_sentences else 'no_more_chunks'
                },
                traversal_path=traversal_path
            )
            
            self.logger.debug(f"✅ Theme-based group {group_idx}: {len(chunk_texts)} chunks, {final_sentence_count} sentences")
            context_groups.append(context_group)
        
        return context_groups
    
    def _get_chunk_themes(self, chunk_id: str) -> List[str]:
        """Get themes associated with a chunk (inherited from document)."""
        if chunk_id not in self.ssg.chunks:
            return []

        chunk = self.ssg.chunks[chunk_id]

        # First try to get themes from chunk's inherited themes
        if hasattr(chunk, 'inherited_themes') and chunk.inherited_themes:
            return chunk.inherited_themes

        # Fallback to document themes if chunk doesn't have inherited themes
        doc_id = chunk.source_document
        if hasattr(self.ssg, 'documents') and doc_id in self.ssg.documents:
            doc = self.ssg.documents[doc_id]
            if hasattr(doc, 'doc_themes') and doc.doc_themes:
                return doc.doc_themes

        return []
    
    # Note: Old theme overlap method removed - now using embedding-based theme similarity


class SequentialMultiHopStrategy(ContextGroupingStrategy):
    """
    Sequential multi-hop reading strategy.
    
    Simulates structured reading patterns by traversing sequentially within documents
    for a configurable number of hops, then making strategic cross-document jumps.
    Creates predictable narrative flows perfect for algorithm validation.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_reading_hops = self.strategy_config.get('num_reading_hops', 3)
        self.num_paragraph_sentences = self.strategy_config.get('num_paragraph_sentences', 5)
        self.num_cross_doc_hops = self.strategy_config.get('num_cross_doc_hops', 3)
    
    def generate_context_groups(self, num_groups: int) -> List[ContextGroup]:
        """Generate sequential multi-hop context groups."""
        context_groups = []
        
        for group_idx in range(num_groups):
            # Get random anchor chunk
            anchor_chunk_id = self._get_random_anchor_chunk()
            if not anchor_chunk_id:
                break

            sentences = []
            traversal_path = [anchor_chunk_id]
            chunk_texts = []
            current_chunk = anchor_chunk_id

            # Log anchor selection with themes
            anchor_themes = self._get_chunk_themes(anchor_chunk_id)
            anchor_doc = self.ssg.chunks[anchor_chunk_id].source_document
            self.logger.debug(f"🏁 Sequential multi-hop anchor {anchor_chunk_id}: "
                            f"doc='{anchor_doc}', themes: {anchor_themes}")
            
            # Perform cross-document hops
            for hop in range(self.num_cross_doc_hops):
                if hop > 0:
                    # Theme-based cross-document hop with similarity fallback
                    next_chunk = self._theme_based_cross_document_hop(current_chunk, set(traversal_path))
                    if next_chunk:
                        current_chunk = next_chunk
                        traversal_path.append(current_chunk)
                
                # Sequential reading within current document
                current_doc = self.ssg.chunks[current_chunk].source_document
                paragraph_sentences = self._sequential_reading_within_document(
                    current_chunk, self.num_reading_hops, self.num_paragraph_sentences
                )

                # Log sequential reading progress
                sentences_before = len(sentences)
                sentences.extend(paragraph_sentences['sentences'])
                chunk_texts.extend(paragraph_sentences['chunk_texts'])

                # Add chunk IDs, but skip the first one if it's already in our path (prevents duplication)
                sequential_chunk_ids = paragraph_sentences['chunk_ids']
                if sequential_chunk_ids and sequential_chunk_ids[0] == current_chunk:
                    # Skip the first chunk since it's the starting chunk we already have
                    traversal_path.extend(sequential_chunk_ids[1:])
                else:
                    # Add all chunks if no duplication
                    traversal_path.extend(sequential_chunk_ids)

                sentences_added = len(sentences) - sentences_before
                self.logger.debug(f"📖 Sequential reading in '{current_doc}': "
                                f"{len(paragraph_sentences['chunk_ids'])} chunks, "
                                f"+{sentences_added} sentences (total: {len(sentences)})")
                
                # Update current chunk to last chunk in reading sequence
                if paragraph_sentences['chunk_ids']:
                    current_chunk = paragraph_sentences['chunk_ids'][-1]
            
            context_group = ContextGroup(
                chunks=chunk_texts,
                chunk_ids=traversal_path,
                sentences=sentences[:self.max_sentences],
                strategy="sequential_multi_hop",
                metadata={
                    'num_reading_hops': self.num_reading_hops,
                    'num_cross_doc_hops': self.num_cross_doc_hops,
                    'paragraphs_created': self.num_cross_doc_hops,
                    'target_sentences_per_paragraph': self.num_paragraph_sentences,
                    'final_sentence_count': len(sentences[:self.max_sentences])
                },
                traversal_path=traversal_path
            )

            # Summary log for this group
            final_sentence_count = len(sentences[:self.max_sentences])
            unique_chunks = len(set(traversal_path))  # Count unique chunks in case of duplicates
            self.logger.debug(f"✅ Sequential multi-hop group {group_idx}: {unique_chunks} chunks, {final_sentence_count} sentences")

            context_groups.append(context_group)
        
        return context_groups
    
    def _sequential_reading_within_document(self, start_chunk_id: str, 
                                          num_hops: int, 
                                          target_sentences: int) -> Dict[str, List]:
        """
        Perform sequential reading within a document.
        
        Reads forward or backward based on similarity to create narrative flow.
        """
        current_chunk = start_chunk_id
        doc_id = self.ssg.chunks[start_chunk_id].source_document
        sentences = []
        chunk_texts = []
        chunk_ids = []
        
        # Get all chunks in this document for sequential navigation
        doc_chunks = [
            chunk_id for chunk_id, chunk in self.ssg.chunks.items()
            if chunk.source_document == doc_id
        ]
        
        for hop in range(num_hops):
            if len(sentences) >= target_sentences:
                break
            
            # Add sentences from current chunk
            chunk_sentences = self._get_chunk_sentences(current_chunk)
            new_sentences = self._deduplicate_sentences(chunk_sentences, sentences)
            sentences.extend(new_sentences)
            
            if current_chunk not in chunk_ids:  # Avoid duplicate chunks
                chunk_texts.append(self.ssg.chunks[current_chunk].chunk_text)
                chunk_ids.append(current_chunk)
            
            if hop < num_hops - 1:  # Don't navigate on last hop
                # Find most similar neighboring chunk in document
                current_idx = doc_chunks.index(current_chunk) if current_chunk in doc_chunks else 0
                
                # Check before and after chunks
                candidates = []
                if current_idx > 0:
                    prev_chunk = doc_chunks[current_idx - 1]
                    if prev_chunk not in chunk_ids:
                        prev_sim = self._calculate_chunk_similarity(current_chunk, prev_chunk)
                        candidates.append((prev_chunk, prev_sim))
                
                if current_idx < len(doc_chunks) - 1:
                    next_chunk = doc_chunks[current_idx + 1]
                    if next_chunk not in chunk_ids:
                        next_sim = self._calculate_chunk_similarity(current_chunk, next_chunk)
                        candidates.append((next_chunk, next_sim))
                
                # Move to most similar neighboring chunk
                if candidates:
                    candidates.sort(key=lambda x: x[1], reverse=True)
                    current_chunk = candidates[0][0]
        
        return {
            'sentences': sentences,
            'chunk_texts': chunk_texts,
            'chunk_ids': chunk_ids
        }

    def _theme_based_cross_document_hop(self, current_chunk: str, visited_chunks: Set[str]) -> Optional[str]:
        """
        Perform theme-based cross-document hop with similarity fallback.

        Uses the same logic as ThemeBasedStrategy for consistent cross-document navigation:
        1. Get theme-similar documents for current chunk's document
        2. Find existing connections to chunks in those theme-similar documents
        3. Select best connection based on similarity scores
        4. Fallback to raw similarity if no theme-based connections found

        Args:
            current_chunk: Current chunk ID
            visited_chunks: Set of already visited chunk IDs to avoid

        Returns:
            Next chunk ID or None if no suitable chunk found
        """
        # Get current document for theme similarity lookup
        current_doc = self.ssg.chunks[current_chunk].source_document

        # Get theme-similar documents using embedding-based approach
        theme_similar_docs = self.ssg.get_theme_similar_documents_by_title(current_doc)

        next_chunk = None
        theme_candidates_found = False

        # Try each theme-similar document in order of theme similarity
        for target_doc_id in theme_similar_docs:
            # Convert doc_id back to document title for chunk matching
            target_doc = self.ssg.get_document_title_by_id(target_doc_id)
            if target_doc == current_doc:
                continue  # Skip same document

            # Get existing connections from current chunk
            current_chunk_obj = self.ssg.chunks[current_chunk]
            all_connections = current_chunk_obj.intra_doc_connections + current_chunk_obj.inter_doc_connections

            # Filter connections to only include chunks from the target theme-similar document
            target_doc_connections = [
                chunk_id for chunk_id in all_connections
                if (chunk_id in self.ssg.chunks and
                    self.ssg.chunks[chunk_id].source_document == target_doc and
                    chunk_id not in visited_chunks)
            ]

            if not target_doc_connections:
                continue  # No existing connections to this theme-similar document

            # Find best connected chunk using pre-computed similarity scores
            best_connected_chunk = None
            best_similarity = 0.0

            for connected_chunk_id in target_doc_connections:
                # Get pre-computed similarity score from connection_scores
                similarity = current_chunk_obj.connection_scores.get(connected_chunk_id, 0.0)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_connected_chunk = connected_chunk_id

            if best_connected_chunk:
                next_chunk = best_connected_chunk
                theme_candidates_found = True
                self.logger.debug(f"🎯 Sequential multi-hop: theme-based hop {current_doc} -> {target_doc} "
                                f"(chunk: {next_chunk}, chunk_sim: {best_similarity:.3f})")
                break

        # Fallback to raw cross-document similarity if no theme candidates found
        if not next_chunk:
            fallback_candidates = self._get_most_similar_chunks(
                current_chunk, visited_chunks, cross_document_only=True
            )
            if fallback_candidates:
                next_chunk = fallback_candidates[0][0]
                target_doc = self.ssg.chunks[next_chunk].source_document
                self.logger.debug(f"🔄 Sequential multi-hop: fallback hop {current_doc} -> {target_doc} "
                                f"(chunk: {next_chunk}, similarity: {fallback_candidates[0][1]:.3f})")

        return next_chunk

    def _get_chunk_themes(self, chunk_id: str) -> List[str]:
        """Get themes associated with a chunk (inherited from document)."""
        if chunk_id not in self.ssg.chunks:
            return []

        chunk = self.ssg.chunks[chunk_id]

        # First try to get themes from chunk's inherited themes
        if hasattr(chunk, 'inherited_themes') and chunk.inherited_themes:
            return chunk.inherited_themes

        # Fallback to document themes if chunk doesn't have inherited themes
        doc_id = chunk.source_document
        if hasattr(self.ssg, 'documents') and doc_id in self.ssg.documents:
            doc = self.ssg.documents[doc_id]
            if hasattr(doc, 'doc_themes') and doc.doc_themes:
                return doc.doc_themes

        return []


class SemanticSimilarityGraphSimilarityStrategy(ContextGroupingStrategy):
    """
    Pure semantic similarity graph similarity strategy.
    
    Performs unrestricted similarity-based traversal across the entire semantic similarity graph,
    allowing the algorithm to explore wherever semantic similarity leads,
    whether within documents or across them.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.allow_cross_document = self.strategy_config.get('allow_cross_document', True)
    
    def generate_context_groups(self, num_groups: int) -> List[ContextGroup]:
        """Generate pure similarity-based context groups."""
        context_groups = []
        
        for group_idx in range(num_groups):
            # Get random anchor chunk
            anchor_chunk_id = self._get_random_anchor_chunk()
            if not anchor_chunk_id:
                self.logger.warning(f"No anchor chunk available for group {group_idx}")
                break
                
            visited_chunks = {anchor_chunk_id}
            sentences = []
            traversal_path = [anchor_chunk_id]
            chunk_texts = []
            
            # Extract sentences from anchor
            anchor_sentences = self._get_chunk_sentences(anchor_chunk_id)
            sentences.extend(anchor_sentences)
            chunk_texts.append(self.ssg.chunks[anchor_chunk_id].chunk_text)
            
            self.logger.debug(f"🏁 SSG similarity anchor {anchor_chunk_id}: {len(anchor_sentences)} sentences")
            
            current_chunk = anchor_chunk_id
            
            # Pure similarity traversal until max_sentences reached
            safety_counter = 0
            max_safety_hops = 50  # Prevent infinite loops
            
            while len(sentences) < self.max_sentences and safety_counter < max_safety_hops:
                safety_counter += 1
                
                # Get most similar chunks (unrestricted) - exclude chunks with sentence overlap
                similar_chunks = self._get_most_similar_chunks(
                    current_chunk, visited_chunks,
                    cross_document_only=False,  # Allow both intra and inter-document
                    existing_sentences=sentences  # Prevent traversal to chunks with already-seen sentences
                )
                
                if not similar_chunks:
                    self.logger.debug(f"🔚 No more similar chunks at {len(sentences)} sentences")
                    break
                
                # Take the most similar chunk (sentence overlap already filtered)
                next_chunk = similar_chunks[0][0] if similar_chunks else None

                if not next_chunk:
                    self.logger.debug(f"🔚 No more similar chunks available at {len(sentences)} sentences")
                    break
                
                # Add sentences from next chunk
                chunk_sentences = self._get_chunk_sentences(next_chunk)
                new_sentences = self._deduplicate_sentences(chunk_sentences, sentences)
                
                # Check if adding these sentences would exceed limit
                if len(sentences) + len(new_sentences) > self.max_sentences:
                    # Take only enough sentences to reach the limit
                    remaining_capacity = self.max_sentences - len(sentences)
                    new_sentences = new_sentences[:remaining_capacity]
                
                sentences.extend(new_sentences)
                
                # Update traversal state
                visited_chunks.add(next_chunk)
                traversal_path.append(next_chunk)
                chunk_texts.append(self.ssg.chunks[next_chunk].chunk_text)
                current_chunk = next_chunk
                
                self.logger.debug(f"➡️ Added SSG chunk {next_chunk}: +{len(new_sentences)} sentences (total: {len(sentences)})")
            
            # Analyze traversal pattern
            documents_visited = set(
                self.ssg.chunks[chunk_id].source_document 
                for chunk_id in traversal_path
            )
            
            # Create context group with proper sentence tracking
            final_sentence_count = min(len(sentences), self.max_sentences)
            
            context_group = ContextGroup(
                chunks=chunk_texts,
                chunk_ids=traversal_path,
                sentences=sentences[:final_sentence_count],
                strategy="semantic_similarity_graph_similarity",
                metadata={
                    'documents_visited': list(documents_visited),
                    'cross_document_hops': sum(
                        1 for i in range(len(traversal_path) - 1)
                        if self.ssg.chunks[traversal_path[i]].source_document != 
                           self.ssg.chunks[traversal_path[i + 1]].source_document
                    ),
                    'final_sentence_count': final_sentence_count,
                    'safety_hops_used': safety_counter,
                    'termination_reason': 'sentence_limit' if len(sentences) >= self.max_sentences else 'no_more_chunks'
                },
                traversal_path=traversal_path
            )
            
            self.logger.debug(f"✅ SSG similarity group {group_idx}: {len(chunk_texts)} chunks, {final_sentence_count} sentences, {len(documents_visited)} documents")
            context_groups.append(context_group)
        
        return context_groups


class DeepEvalNativeStrategy(ContextGroupingStrategy):
    """
    DeepEval-native context extraction strategy.
    
    Uses simple random or sequential chunk extraction without sophisticated semantic traversal.
    Relies on DeepEval's FiltrationConfig quality scoring (clarity, self-containment) to ensure
    high-quality contexts, rather than using pre-computed SSG similarities.
    
    This strategy prioritizes:
    - Simplicity: Random/sequential chunk selection
    - Quality: DeepEval's LLM-based quality filtering
    - Diversity: Wide coverage across documents
    - Speed: No complex traversal calculations
    
    Perfect for baseline comparison against sophisticated semantic strategies.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extraction_mode = self.strategy_config.get('extraction_mode', 'random')  # 'random' or 'sequential'
        self.chunks_per_group = self.strategy_config.get('chunks_per_group', 3)
        self.ensure_document_diversity = self.strategy_config.get('ensure_document_diversity', True)
        
    def generate_context_groups(self, num_groups: int) -> List[ContextGroup]:
        """
        Generate context groups using simple extraction with DeepEval quality reliance.
        
        Unlike sophisticated strategies that use semantic similarity for traversal,
        this strategy simply extracts chunks and lets DeepEval's FiltrationConfig
        handle quality through LLM-based scoring of clarity and self-containment.
        
        Args:
            num_groups: Number of context groups to generate
            
        Returns:
            List of ContextGroup objects with simple extraction metadata
        """
        context_groups = []
        
        # Get all available chunks and apply quality filtering
        all_chunk_ids = list(self.ssg.chunks.keys())

        if not all_chunk_ids:
            self.logger.warning("No chunks available for DeepEval native strategy")
            return []

        # Apply quality filtering (prefer high-quality chunks for this strategy)
        quality_filtered_chunks = self._filter_chunks_by_quality(all_chunk_ids, min_quality=0.7)

        # Use quality-filtered chunks if available, otherwise fall back to all chunks
        working_chunks = quality_filtered_chunks if quality_filtered_chunks else all_chunk_ids

        self.logger.debug(f"🎯 DeepEval native: Using {len(working_chunks)}/{len(all_chunk_ids)} chunks "
                         f"({'quality-filtered' if quality_filtered_chunks else 'unfiltered'})")

        # Shuffle for randomness if in random mode
        if self.extraction_mode == 'random':
            random.shuffle(working_chunks)
            self.logger.debug(f"🎲 DeepEval native: Shuffled {len(working_chunks)} chunks for random extraction")
        
        # Track used chunks to avoid duplication
        used_chunks = set()
        
        for group_idx in range(num_groups):
            # Select chunks for this group
            group_chunk_ids = []
            group_documents = set()
            
            # Try to get chunks_per_group chunks
            attempts = 0
            max_attempts = len(all_chunk_ids)
            
            while len(group_chunk_ids) < self.chunks_per_group and attempts < max_attempts:
                # Get next chunk based on extraction mode
                if self.extraction_mode == 'random':
                    # Random selection from unused chunks (quality-filtered)
                    available_chunks = [c for c in working_chunks if c not in used_chunks]
                    if not available_chunks:
                        break
                    candidate_chunk = random.choice(available_chunks)
                else:  # sequential
                    # Sequential selection (quality-filtered)
                    candidate_idx = (group_idx * self.chunks_per_group + len(group_chunk_ids)) % len(working_chunks)
                    candidate_chunk = working_chunks[candidate_idx]
                    
                    if candidate_chunk in used_chunks:
                        attempts += 1
                        continue
                
                candidate_doc = self.ssg.chunks[candidate_chunk].source_document
                
                # Check document diversity if enabled
                if self.ensure_document_diversity and candidate_doc in group_documents:
                    attempts += 1
                    continue
                
                # Add chunk to group
                group_chunk_ids.append(candidate_chunk)
                group_documents.add(candidate_doc)
                used_chunks.add(candidate_chunk)
                
                attempts += 1
            
            # Extract data for context group
            chunks_texts = []
            sentences = []
            
            for chunk_id in group_chunk_ids:
                # Get chunk text
                chunk = self.ssg.chunks[chunk_id]
                chunks_texts.append(chunk.chunk_text)
                
                # Extract sentences from chunk (with deduplication)
                chunk_sentences = self._get_chunk_sentences(chunk_id)
                new_sentences = self._deduplicate_sentences(chunk_sentences, sentences)
                sentences.extend(new_sentences)
            
            # Truncate sentences to max_sentences limit
            final_sentences = sentences[:self.max_sentences]
            
            # Create context group with simple metadata
            context_group = ContextGroup(
                chunks=chunks_texts,
                chunk_ids=group_chunk_ids,
                sentences=final_sentences,
                strategy="deepeval_native",
                metadata={
                    'extraction_mode': self.extraction_mode,
                    'chunks_per_group_target': self.chunks_per_group,
                    'chunks_extracted': len(group_chunk_ids),
                    'documents_in_group': list(group_documents),
                    'document_diversity_enforced': self.ensure_document_diversity,
                    'final_sentence_count': len(final_sentences),
                    'quality_filtering': 'Relies on DeepEval FiltrationConfig with LLM-based quality scoring',
                    'note': 'Simple extraction strategy - no semantic traversal, relies on downstream quality filtering'
                },
                traversal_path=group_chunk_ids  # Linear path, no traversal
            )
            
            self.logger.debug(
                f"✅ DeepEval native group {group_idx}: {len(group_chunk_ids)} chunks "
                f"from {len(group_documents)} documents, {len(final_sentences)} sentences"
            )
            
            context_groups.append(context_group)
        
        # Log strategy summary
        total_chunks = sum(len(cg.chunk_ids) for cg in context_groups)
        total_docs = len(set(
            doc for cg in context_groups 
            for doc in cg.metadata.get('documents_in_group', [])
        ))
        
        self.logger.info(
            f"🎯 DeepEval native strategy generated {len(context_groups)} groups: "
            f"{total_chunks} chunks from {total_docs} documents (mode: {self.extraction_mode})"
        )
        
        return context_groups


class ContextGroupingOrchestrator:
    """
    Orchestrates multiple context grouping strategies according to configuration.
    
    Manages strategy selection, weighting, and aggregation to create diverse
    context groups for comprehensive algorithm testing.
    """
    
    def __init__(self, ssg: SemanticSimilarityGraph, config: Dict[str, Any], 
                 logger: Optional[logging.Logger] = None):
        """
        Initialize orchestrator with semantic similarity graph and configuration.
        
        Args:
            ssg: Semantic similarity graph instance
            config: Complete system configuration with context_strategies section
            logger: Optional logger instance
        """
        self.ssg = ssg
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Extract context strategies configuration
        self.context_strategies_config = config.get('context_strategies', {})
        
        # Initialize strategy instances
        self.strategies = self._initialize_strategies()
        
        self.logger.info(f"ContextGroupingOrchestrator initialized with {len(self.strategies)} strategies")
    
    def _initialize_strategies(self) -> Dict[str, ContextGroupingStrategy]:
        """Initialize all enabled strategy instances."""
        strategies = {}
        
        strategy_classes = {
            'intra_document': IntraDocumentStrategy,
            'theme_based': ThemeBasedStrategy,
            'sequential_multi_hop': SequentialMultiHopStrategy,
            'deepeval_native': DeepEvalNativeStrategy
            # 'inter_document': InterDocumentStrategy  # DEPRECATED: Redundant with ThemeBasedStrategy fallback
            # 'semantic_similarity_graph_similarity': SemanticSimilarityGraphSimilarityStrategy  # DEPRECATED: Redundant with inter_document
        }
        
        for strategy_name, strategy_class in strategy_classes.items():
            strategy_config = self.context_strategies_config.get(strategy_name, {})
            
            if strategy_config.get('enabled', False):
                strategy_instance = strategy_class(
                    self.ssg, self.config, strategy_config, self.logger
                )
                strategies[strategy_name] = strategy_instance
                self.logger.info(f"✅ Enabled strategy: {strategy_name} (weight: {strategy_config.get('weight', 1.0)})")
            else:
                self.logger.info(f"⏸️  Disabled strategy: {strategy_name}")
        
        return strategies
    
    def generate_context_groups(self, total_groups: int) -> List[ContextGroup]:
        """
        Generate context groups using all enabled strategies according to their weights.
        
        Args:
            total_groups: Total number of context groups to generate across all strategies
            
        Returns:
            List of ContextGroup objects from all strategies
        """
        if not self.strategies:
            self.logger.warning("No enabled strategies found")
            return []
        
        # Calculate groups per strategy based on weights
        groups_per_strategy = self._calculate_groups_per_strategy(total_groups)
        
        all_context_groups = []
        
        for strategy_name, num_groups in groups_per_strategy.items():
            if num_groups > 0:
                self.logger.info(f"🔄 Generating {num_groups} context groups using {strategy_name}")
                
                strategy = self.strategies[strategy_name]
                strategy_groups = strategy.generate_context_groups(num_groups)
                
                all_context_groups.extend(strategy_groups)
                
                self.logger.info(f"✅ Generated {len(strategy_groups)} context groups from {strategy_name}")
        
        # Randomize order to avoid strategy clustering
        random.shuffle(all_context_groups)
        
        self.logger.info(f"🎯 Total context groups generated: {len(all_context_groups)}")
        
        return all_context_groups
    
    def _calculate_groups_per_strategy(self, total_groups: int) -> Dict[str, int]:
        """Calculate number of groups each strategy should generate based on weights."""
        # Get weights for enabled strategies
        weights = {}
        for strategy_name, strategy in self.strategies.items():
            weights[strategy_name] = strategy.weight
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight == 0:
            # Equal distribution if no weights specified
            equal_weight = 1.0 / len(weights) if weights else 1.0
            weights = {name: equal_weight for name in weights}
            total_weight = sum(weights.values())
        
        # Calculate groups per strategy
        groups_per_strategy = {}
        allocated_groups = 0
        
        for strategy_name, weight in weights.items():
            proportion = weight / total_weight
            num_groups = int(total_groups * proportion)
            groups_per_strategy[strategy_name] = num_groups
            allocated_groups += num_groups
        
        # Distribute remaining groups to strategies with highest weights
        remaining_groups = total_groups - allocated_groups
        if remaining_groups > 0:
            sorted_strategies = sorted(weights.items(), key=lambda x: x[1], reverse=True)
            for i in range(remaining_groups):
                strategy_name = sorted_strategies[i % len(sorted_strategies)][0]
                groups_per_strategy[strategy_name] += 1
        
        return groups_per_strategy
    
    def get_strategy_statistics(self) -> Dict[str, Any]:
        """Get statistics about enabled strategies and their configuration."""
        return {
            'enabled_strategies': list(self.strategies.keys()),
            'strategy_weights': {
                name: strategy.weight for name, strategy in self.strategies.items()
            },
            'strategy_configs': {
                name: strategy.strategy_config for name, strategy in self.strategies.items()
            }
        }
