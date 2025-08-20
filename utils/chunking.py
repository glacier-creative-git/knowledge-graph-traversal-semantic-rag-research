#!/usr/bin/env python3
"""
Chunking Engine
==============

Handles different chunking strategies for the semantic RAG pipeline.
Currently implements sliding window strategy with easy expansion for future algorithms.
"""

import hashlib
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from wiki import WikipediaArticle


@dataclass
class ChunkInfo:
    """Information about a chunk."""
    chunk_id: str
    text: str
    source_article: str
    source_sentences: List[int]  # Indices of sentences from original article
    anchor_sentence_idx: int     # Index of the anchor sentence (first for sliding window)
    window_position: int         # Position of this window in the article
    total_windows: int          # Total windows in the source article


class ChunkEngine:
    """Engine for creating chunks from articles using various strategies."""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """Initialize the chunking engine."""
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.chunking_config = config['chunking']
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate chunking configuration."""
        strategy = self.chunking_config['strategy']
        
        if strategy == 'sliding_window':
            window_size = self.chunking_config['window_size']
            overlap = self.chunking_config['overlap']
            
            if window_size <= 0:
                raise ValueError("window_size must be positive")
            if overlap < 0:
                raise ValueError("overlap must be non-negative")
            if overlap >= window_size:
                raise ValueError("overlap must be less than window_size")
        else:
            raise ValueError(f"Unsupported chunking strategy: {strategy}")
    
    def create_chunks(self, articles: List[WikipediaArticle]) -> List[Dict[str, Any]]:
        """
        Create chunks from articles using the configured strategy.
        
        Args:
            articles: List of WikipediaArticle objects
            
        Returns:
            List of chunk dictionaries
        """
        strategy = self.chunking_config['strategy']
        
        if strategy == 'sliding_window':
            return self._create_sliding_window_chunks(articles)
        else:
            raise ValueError(f"Unsupported chunking strategy: {strategy}")
    
    def _create_sliding_window_chunks(self, articles: List[WikipediaArticle]) -> List[Dict[str, Any]]:
        """
        Create chunks using sliding window strategy.
        
        Forward-facing: anchored to the first sentence in each window.
        Window size: 3 sentences, Overlap: 1 sentence
        
        Example with sentences [A, B, C, D, E, F]:
        Window 1: [A, B, C] (anchor: A, position 0)
        Window 2: [C, D, E] (anchor: C, position 2) 
        Window 3: [E, F]    (anchor: E, position 4)
        """
        window_size = self.chunking_config['window_size']
        overlap = self.chunking_config['overlap']
        step_size = window_size - overlap
        
        self.logger.info(f"Creating sliding window chunks: window_size={window_size}, overlap={overlap}, step_size={step_size}")
        
        all_chunks = []
        total_articles = len(articles)
        
        for article_idx, article in enumerate(articles):
            self.logger.debug(f"Processing article {article_idx + 1}/{total_articles}: {article.title}")
            
            sentences = article.sentences
            if len(sentences) < window_size:
                self.logger.warning(f"Article '{article.title}' has only {len(sentences)} sentences, skipping (min required: {window_size})")
                continue
            
            # Calculate windows for this article
            windows = self._calculate_sliding_windows(sentences, window_size, step_size)
            
            # Create chunks for each window
            for window_idx, (start_idx, end_idx) in enumerate(windows):
                chunk = self._create_chunk_from_window(
                    article, sentences, start_idx, end_idx, window_idx, len(windows)
                )
                all_chunks.append(chunk)
        
        self.logger.info(f"Created {len(all_chunks)} chunks from {total_articles} articles")
        return all_chunks
    
    def _calculate_sliding_windows(self, sentences: List[str], window_size: int, step_size: int) -> List[Tuple[int, int]]:
        """
        Calculate sliding window positions.
        
        Args:
            sentences: List of sentences
            window_size: Size of each window
            step_size: Step size between windows
            
        Returns:
            List of (start_idx, end_idx) tuples for each window
        """
        windows = []
        start_idx = 0
        
        while start_idx < len(sentences):
            end_idx = min(start_idx + window_size, len(sentences))
            windows.append((start_idx, end_idx))
            
            # Break if we've reached the end
            if end_idx == len(sentences):
                break
            
            # Move to next window
            start_idx += step_size
        
        return windows
    
    def _create_chunk_from_window(self, article: WikipediaArticle, sentences: List[str], 
                                start_idx: int, end_idx: int, window_position: int, 
                                total_windows: int) -> Dict[str, Any]:
        """Create a chunk dictionary from a window of sentences."""
        
        # Get sentences in this window
        window_sentences = sentences[start_idx:end_idx]
        chunk_text = ' '.join(window_sentences)
        
        # Create unique chunk ID
        chunk_id = self._generate_chunk_id(article.title, start_idx, end_idx)
        
        # Source sentence indices (relative to original article)
        source_sentences = list(range(start_idx, end_idx))
        
        # Anchor sentence index (first sentence in window for forward-facing)
        anchor_sentence_idx = start_idx
        
        chunk_info = {
            'chunk_id': chunk_id,
            'text': chunk_text,
            'source_article': article.title,
            'source_sentences': source_sentences,
            'anchor_sentence_idx': anchor_sentence_idx,
            'window_position': window_position,
            'total_windows': total_windows,
            'window_size': len(window_sentences),
            'start_sentence_idx': start_idx,
            'end_sentence_idx': end_idx - 1  # Make end_idx inclusive for clarity
        }
        
        self.logger.debug(f"Created chunk {chunk_id}: sentences {start_idx}-{end_idx-1} from '{article.title}'")
        
        return chunk_info
    
    def _generate_chunk_id(self, article_title: str, start_idx: int, end_idx: int) -> str:
        """Generate a deterministic unique chunk ID."""
        # Create a readable but unique identifier
        safe_title = article_title.replace(' ', '_').replace('/', '_')[:50]  # Truncate long titles
        chunk_identifier = f"{safe_title}_window_{start_idx}_{end_idx}"
        
        # Add hash to ensure uniqueness (deterministic - no timestamp)
        hash_input = f"{article_title}_{start_idx}_{end_idx}"
        hash_suffix = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        
        return f"{chunk_identifier}_{hash_suffix}"
    
    def get_chunking_statistics(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about the chunking results."""
        if not chunks:
            return {}
        
        # Group chunks by source article
        articles_chunks = {}
        for chunk in chunks:
            article = chunk['source_article']
            if article not in articles_chunks:
                articles_chunks[article] = []
            articles_chunks[article].append(chunk)
        
        # Calculate statistics
        chunk_lengths = [len(chunk['text'].split()) for chunk in chunks]
        sentence_counts = [chunk['window_size'] for chunk in chunks]
        
        stats = {
            'total_chunks': len(chunks),
            'total_articles': len(articles_chunks),
            'avg_chunks_per_article': len(chunks) / len(articles_chunks) if articles_chunks else 0,
            'chunk_length_stats': {
                'mean_words': sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0,
                'min_words': min(chunk_lengths) if chunk_lengths else 0,
                'max_words': max(chunk_lengths) if chunk_lengths else 0
            },
            'sentence_count_stats': {
                'mean_sentences': sum(sentence_counts) / len(sentence_counts) if sentence_counts else 0,
                'min_sentences': min(sentence_counts) if sentence_counts else 0,
                'max_sentences': max(sentence_counts) if sentence_counts else 0
            },
            'chunking_config': {
                'strategy': self.chunking_config['strategy'],
                'window_size': self.chunking_config['window_size'],
                'overlap': self.chunking_config['overlap']
            }
        }
        
        return stats
    
    # Placeholder methods for future chunking strategies
    # These can be easily expanded later
    
    def _create_sentence_based_chunks(self, articles: List[WikipediaArticle]) -> List[Dict[str, Any]]:
        """
        Placeholder for sentence-based chunking strategy.
        
        This would create one chunk per sentence.
        """
        raise NotImplementedError("Sentence-based chunking not yet implemented")
    
    def _create_similarity_matrix_chunks(self, articles: List[WikipediaArticle]) -> List[Dict[str, Any]]:
        """
        Placeholder for similarity matrix-based chunking strategy.
        
        This would use semantic similarity to determine chunk boundaries.
        """
        raise NotImplementedError("Similarity matrix-based chunking not yet implemented")
    
    def _create_adaptive_distance_chunks(self, articles: List[WikipediaArticle]) -> List[Dict[str, Any]]:
        """
        Placeholder for adaptive distance chunking strategy.
        
        This would dynamically adjust chunk sizes based on content density.
        """
        raise NotImplementedError("Adaptive distance chunking not yet implemented")


def create_chunk_engine(config: Dict[str, Any], logger: Optional[logging.Logger] = None) -> ChunkEngine:
    """Factory function to create a ChunkEngine."""
    return ChunkEngine(config, logger)
