#!/usr/bin/env python3
"""
Semantic Reranking Module for RAG Retrieval Systems
=================================================

Implements sophisticated reranking strategies that decouple graph traversal
from query-specific relevance optimization. This module ensures fair comparison
between algorithms by standardizing post-retrieval processing while preserving
the unique traversal characteristics of each algorithm.

Key Design Principles:
- Separation of concerns: traversal algorithms focus on connectivity, reranker focuses on relevance
- Standardized output: all algorithms produce exactly 10 sentences after reranking
- Transparency: reranking decisions are logged and explicable
- Configurability: supports multiple reranking strategies
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RankedSentence:
    """Container for a ranked sentence with scoring metadata."""
    sentence_id: str
    content: str
    original_rank: int
    rerank_score: float
    score_components: Dict[str, float]
    source_chunk: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'sentence_id': self.sentence_id,
            'content': self.content[:100] + "..." if len(self.content) > 100 else self.content,
            'original_rank': self.original_rank,
            'rerank_score': self.rerank_score,
            'score_components': self.score_components,
            'source_chunk': self.source_chunk
        }


class BaseReranker(ABC):
    """Abstract base class for reranking strategies."""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
    
    @abstractmethod
    def rerank(self, sentences: List[str], query: str, metadata: Optional[Dict] = None) -> List[RankedSentence]:
        """Rerank sentences based on query relevance."""
        pass
    
    def _ensure_target_count(self, ranked_sentences: List[RankedSentence], target_count: int = 10) -> List[RankedSentence]:
        """Ensure exactly target_count sentences are returned."""
        if len(ranked_sentences) >= target_count:
            return ranked_sentences[:target_count]
        else:
            # If we have fewer than target_count, pad with duplicates (mark them clearly)
            padded = ranked_sentences[:]
            while len(padded) < target_count:
                if ranked_sentences:
                    # Duplicate the last sentence but mark it
                    duplicate = ranked_sentences[-1]
                    duplicate.score_components['is_padding'] = 1.0
                    duplicate.rerank_score *= 0.1  # Heavily penalize padding
                    padded.append(duplicate)
                else:
                    break
            return padded


class TFIDFReranker(BaseReranker):
    """TF-IDF based reranking using cosine similarity."""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        super().__init__(config, logger)
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=config.get('max_features', 5000),
            ngram_range=config.get('ngram_range', (1, 2))
        )
    
    def rerank(self, sentences: List[str], query: str, metadata: Optional[Dict] = None) -> List[RankedSentence]:
        """Rerank using TF-IDF cosine similarity."""
        if not sentences:
            return []
        
        # Prepare documents (query + sentences)
        documents = [query] + sentences
        
        try:
            # Fit TF-IDF on all documents
            tfidf_matrix = self.vectorizer.fit_transform(documents)
            
            # Calculate similarity between query (index 0) and all sentences
            query_vector = tfidf_matrix[0:1]
            sentence_vectors = tfidf_matrix[1:]
            
            similarities = cosine_similarity(query_vector, sentence_vectors)[0]
            
            # Create ranked sentences
            ranked_sentences = []
            for i, (sentence, similarity) in enumerate(zip(sentences, similarities)):
                score_components = {
                    'tfidf_similarity': float(similarity),
                    'position_penalty': 1.0 / (1.0 + i * 0.1)  # Slight preference for earlier positions
                }
                
                final_score = similarity * score_components['position_penalty']
                
                ranked_sentence = RankedSentence(
                    sentence_id=f"sentence_{i}",
                    content=sentence,
                    original_rank=i,
                    rerank_score=final_score,
                    score_components=score_components,
                    source_chunk=metadata.get('chunks', [f"chunk_{i}"])[i] if metadata else f"chunk_{i}"
                )
                ranked_sentences.append(ranked_sentence)
            
            # Sort by rerank score (descending)
            ranked_sentences.sort(key=lambda x: x.rerank_score, reverse=True)
            
            return self._ensure_target_count(ranked_sentences, self.config.get('target_count', 10))
            
        except Exception as e:
            self.logger.error(f"TF-IDF reranking failed: {e}")
            # Fallback: return original order
            return self._create_fallback_ranking(sentences, metadata)
    
    def _create_fallback_ranking(self, sentences: List[str], metadata: Optional[Dict] = None) -> List[RankedSentence]:
        """Create fallback ranking when reranking fails."""
        ranked_sentences = []
        for i, sentence in enumerate(sentences):
            ranked_sentence = RankedSentence(
                sentence_id=f"sentence_{i}",
                content=sentence,
                original_rank=i,
                rerank_score=1.0 - (i * 0.1),  # Simple decreasing score
                score_components={'fallback_score': 1.0},
                source_chunk=metadata.get('chunks', [f"chunk_{i}"])[i] if metadata else f"chunk_{i}"
            )
            ranked_sentences.append(ranked_sentence)
        
        return self._ensure_target_count(ranked_sentences, self.config.get('target_count', 10))


class SemanticReranker(BaseReranker):
    """Semantic similarity reranking using sentence transformers (placeholder for future implementation)."""
    
    def rerank(self, sentences: List[str], query: str, metadata: Optional[Dict] = None) -> List[RankedSentence]:
        """Placeholder for semantic reranking - falls back to TF-IDF for now."""
        self.logger.warning("SemanticReranker not fully implemented, falling back to TF-IDF")
        tfidf_reranker = TFIDFReranker(self.config, self.logger)
        return tfidf_reranker.rerank(sentences, query, metadata)


class HybridReranker(BaseReranker):
    """Hybrid reranker combining multiple signals."""
    
    def rerank(self, sentences: List[str], query: str, metadata: Optional[Dict] = None) -> List[RankedSentence]:
        """Combine TF-IDF similarity with other signals."""
        if not sentences:
            return []
        
        # Get TF-IDF baseline
        tfidf_reranker = TFIDFReranker(self.config, self.logger)
        tfidf_results = tfidf_reranker.rerank(sentences, query, metadata)
        
        # Apply additional signals
        for result in tfidf_results:
            # Length penalty (prefer moderate-length sentences)
            length_score = self._calculate_length_score(result.content)
            
            # Question word overlap bonus
            question_overlap = self._calculate_question_overlap(result.content, query)
            
            # Update score components
            result.score_components.update({
                'length_score': length_score,
                'question_overlap': question_overlap
            })
            
            # Recalculate final score
            weights = self.config.get('hybrid_weights', {
                'tfidf': 0.6,
                'length': 0.2,
                'question_overlap': 0.2
            })
            
            result.rerank_score = (
                result.score_components['tfidf_similarity'] * weights['tfidf'] +
                length_score * weights['length'] +
                question_overlap * weights['question_overlap']
            )
        
        # Re-sort by updated scores
        tfidf_results.sort(key=lambda x: x.rerank_score, reverse=True)
        
        return self._ensure_target_count(tfidf_results, self.config.get('target_count', 10))
    
    def _calculate_length_score(self, text: str) -> float:
        """Calculate score based on text length (prefer moderate lengths)."""
        word_count = len(text.split())
        # Optimal range: 15-50 words
        if 15 <= word_count <= 50:
            return 1.0
        elif word_count < 15:
            return word_count / 15.0
        else:
            return max(0.3, 50.0 / word_count)
    
    def _calculate_question_overlap(self, text: str, query: str) -> float:
        """Calculate bonus for question-relevant terms."""
        # Extract important words from query (excluding common words)
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        text_words = set(re.findall(r'\b\w+\b', text.lower()))
        
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'how', 'what', 'why', 'when', 'where'}
        
        query_content_words = query_words - stop_words
        overlap = len(query_content_words & text_words)
        
        return min(1.0, overlap / max(1, len(query_content_words)))


class RerankerOrchestrator:
    """Main orchestrator for reranking operations."""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.reranker = self._create_reranker()
    
    def _create_reranker(self) -> BaseReranker:
        """Create appropriate reranker based on configuration."""
        reranker_type = self.config.get('reranking', {}).get('strategy', 'tfidf')
        
        if reranker_type == 'tfidf':
            return TFIDFReranker(self.config.get('reranking', {}), self.logger)
        elif reranker_type == 'semantic':
            return SemanticReranker(self.config.get('reranking', {}), self.logger)
        elif reranker_type == 'hybrid':
            return HybridReranker(self.config.get('reranking', {}), self.logger)
        else:
            self.logger.warning(f"Unknown reranker type: {reranker_type}, defaulting to TF-IDF")
            return TFIDFReranker(self.config.get('reranking', {}), self.logger)
    
    def rerank_retrieval_result(self, retrieval_result: Any, query: str) -> Tuple[List[str], Dict[str, Any]]:
        """
        Rerank a retrieval result and return standardized output.
        
        Args:
            retrieval_result: Result from any retrieval algorithm
            query: The original query
            
        Returns:
            Tuple of (ranked_sentences, reranking_metadata)
        """
        try:
            # Extract sentences from retrieval result
            sentences = self._extract_sentences_from_result(retrieval_result)
            
            # Extract metadata for reranking
            metadata = self._extract_metadata_from_result(retrieval_result)
            
            # Perform reranking
            ranked_sentences = self.reranker.rerank(sentences, query, metadata)
            
            # Prepare output
            final_sentences = [rs.content for rs in ranked_sentences]
            reranking_metadata = {
                'original_sentence_count': len(sentences),
                'final_sentence_count': len(final_sentences),
                'reranking_strategy': self.config.get('reranking', {}).get('strategy', 'tfidf'),
                'ranking_details': [rs.to_dict() for rs in ranked_sentences[:5]]  # Top 5 for debugging
            }
            
            self.logger.debug(f"Reranked {len(sentences)} -> {len(final_sentences)} sentences using {reranking_metadata['reranking_strategy']}")
            
            return final_sentences, reranking_metadata
            
        except Exception as e:
            self.logger.error(f"Reranking failed: {e}")
            # Fallback: return original sentences
            sentences = self._extract_sentences_from_result(retrieval_result)
            return sentences[:10], {'error': str(e), 'fallback': True}
    
    def _extract_sentences_from_result(self, retrieval_result: Any) -> List[str]:
        """Extract sentences from retrieval result (algorithm-agnostic)."""
        sentences = []
        
        # Try different ways to extract sentences based on result structure
        if hasattr(retrieval_result, 'retrieved_content') and retrieval_result.retrieved_content:
            sentences = retrieval_result.retrieved_content
        elif hasattr(retrieval_result, 'sentences') and retrieval_result.sentences:
            sentences = retrieval_result.sentences
        elif hasattr(retrieval_result, 'content') and retrieval_result.content:
            sentences = retrieval_result.content
        else:
            # Last resort: try to find any list-like attribute
            for attr_name in dir(retrieval_result):
                attr = getattr(retrieval_result, attr_name)
                if isinstance(attr, list) and len(attr) > 0 and isinstance(attr[0], str):
                    sentences = attr
                    break
        
        # Ensure we have actual text content
        sentences = [s for s in sentences if s and isinstance(s, str) and len(s.strip()) > 10]
        
        return sentences
    
    def _extract_metadata_from_result(self, retrieval_result: Any) -> Dict[str, Any]:
        """Extract metadata from retrieval result for reranking context."""
        metadata = {}
        
        # Try to extract chunk information
        if hasattr(retrieval_result, 'traversal_path') and retrieval_result.traversal_path:
            metadata['chunks'] = retrieval_result.traversal_path.nodes
            metadata['connection_types'] = [str(ct) for ct in retrieval_result.traversal_path.connection_types]
        
        # Extract algorithm information
        if hasattr(retrieval_result, 'algorithm_name'):
            metadata['algorithm'] = retrieval_result.algorithm_name
        
        # Extract timing information
        if hasattr(retrieval_result, 'processing_time'):
            metadata['processing_time'] = retrieval_result.processing_time
        
        return metadata


# Factory function for easy integration
def create_reranker_orchestrator(config: Dict[str, Any], logger: Optional[logging.Logger] = None) -> RerankerOrchestrator:
    """Factory function to create a reranker orchestrator."""
    return RerankerOrchestrator(config, logger)
