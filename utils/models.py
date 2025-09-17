#!/usr/bin/env python3
"""
Enhanced Multi-Granularity Embedding Engine
==========================================

Handles multi-granularity embedding generation for the enhanced semantic RAG pipeline.
Supports chunks, sentences, and document summaries with unified caching and processing.
"""

import json
import hashlib
import logging
import time
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import nltk

# Download required NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


@dataclass
class MultiGranularityEmbeddingMetadata:
    """Metadata for cached multi-granularity embeddings."""
    model_name: str
    embedding_dimension: int
    granularity_counts: Dict[str, int]  # chunks: 100, sentences: 300, doc_summaries: 10
    created_at: str
    device: str
    processing_time: float
    config_hash: str
    granularity_config: Dict[str, Any]


@dataclass
class ChunkEmbedding:
    """Container for a chunk and its embedding (unchanged from original)."""
    chunk_id: str
    chunk_text: str
    embedding: np.ndarray
    source_article: str
    source_sentences: List[int]  # Indices of sentences in the chunk
    anchor_sentence_idx: int     # Index of the anchor sentence (first sentence for sliding window)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'chunk_id': self.chunk_id,
            'chunk_text': self.chunk_text,
            'embedding': self.embedding.tolist(),
            'source_article': self.source_article,
            'source_sentences': self.source_sentences,
            'anchor_sentence_idx': self.anchor_sentence_idx
        }


@dataclass
class SentenceEmbedding:
    """Container for a sentence and its embedding."""
    sentence_id: str
    sentence_text: str
    embedding: np.ndarray
    source_article: str
    sentence_index: int          # Global sentence index within article
    containing_chunks: List[str] # IDs of chunks that contain this sentence
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'sentence_id': self.sentence_id,
            'sentence_text': self.sentence_text,
            'embedding': self.embedding.tolist(),
            'source_article': self.source_article,
            'sentence_index': self.sentence_index,
            'containing_chunks': self.containing_chunks
        }


@dataclass
class DocumentSummaryEmbedding:
    """Container for a document summary and its embedding."""
    doc_id: str
    doc_title: str
    summary_text: str
    embedding: np.ndarray
    source_article: str
    summary_method: str          # extractive, ollama
    total_sentences: int         # Number of sentences in original document
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'doc_id': self.doc_id,
            'doc_title': self.doc_title,
            'summary_text': self.summary_text,
            'embedding': self.embedding.tolist(),
            'source_article': self.source_article,
            'summary_method': self.summary_method,
            'total_sentences': self.total_sentences
        }


class EmbeddingModel:
    """Wrapper for embedding models with unified interface (unchanged from original)."""
    
    def __init__(self, model_name: str, device: str = "cpu", logger: Optional[logging.Logger] = None):
        """Initialize embedding model."""
        self.model_name = model_name
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        self.model = None
        self.embedding_dimension = None
        
        # Validate device
        self._validate_device()
        
        # Load model
        self._load_model()
    
    def _validate_device(self):
        """Validate device string and availability."""
        valid_devices = ["cpu", "cuda", "mps"]
        
        if self.device not in valid_devices:
            raise ValueError(f"Invalid device '{self.device}'. Must be one of: {valid_devices}")
        
        # Check device availability
        if self.device == "cuda" and not torch.cuda.is_available():
            self.logger.warning(f"CUDA requested but not available, falling back to CPU")
            self.device = "cpu"
        elif self.device == "mps" and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            self.logger.warning(f"MPS requested but not available, falling back to CPU")
            self.device = "cpu"
        
        self.logger.info(f"Using device: {self.device}")
    
    def _load_model(self):
        """Load the embedding model."""
        try:
            self.logger.info(f"Loading embedding model: {self.model_name}")
            start_time = time.time()
            
            # Load sentence transformer model
            self.model = SentenceTransformer(self.model_name, device=self.device)
            
            # Get embedding dimension
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            
            load_time = time.time() - start_time
            self.logger.info(f"Model loaded successfully in {load_time:.2f}s")
            self.logger.info(f"Embedding dimension: {self.embedding_dimension}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    def encode_batch(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """
        Encode a batch of texts to embeddings.
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            
        Returns:
            Array of embeddings with shape (len(texts), embedding_dimension)
        """
        if not texts:
            return np.array([])
        
        try:
            self.logger.debug(f"Encoding {len(texts)} texts with batch size {batch_size}")
            
            # Use sentence-transformers encode method with batching
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                device=self.device
            )
            
            self.logger.debug(f"Generated embeddings shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Failed to encode batch: {e}")
            raise
    
    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text to embedding."""
        return self.encode_batch([text], batch_size=1, show_progress=False)[0]


class MultiGranularityEmbeddingEngine:
    """Enhanced engine for generating multi-granularity embeddings."""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """Initialize the multi-granularity embedding engine."""
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.device = config['system']['device']
        self.embeddings_dir = Path(config['directories']['embeddings'])
        self.granularity_config = config['models']['granularity_types']
        
        self.logger.info(f"MultiGranularityEmbeddingEngine initialized with device: {self.device}")
        
        # Create embeddings directory structure
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        (self.embeddings_dir / "raw").mkdir(exist_ok=True)
        
        # Log enabled granularity types
        enabled_types = [gt for gt, conf in self.granularity_config.items() if conf.get('enabled', True)]
        self.logger.info(f"Enabled granularity types: {enabled_types}")
    
    def generate_embeddings(self, chunks: List[Dict[str, Any]], articles: List[Any], 
                          force_recompute: bool = False) -> Dict[str, Dict[str, List[Any]]]:
        """
        Generate multi-granularity embeddings for all configured models.
        
        Args:
            chunks: List of chunk dictionaries from ChunkEngine
            articles: List of WikipediaArticle objects for document-level processing
            force_recompute: Whether to force recomputation even if cached
            
        Returns:
            Dictionary mapping model names to granularity dictionaries:
            {
                'model_name': {
                    'chunks': [ChunkEmbedding, ...],
                    'sentences': [SentenceEmbedding, ...],
                    'doc_summaries': [DocumentSummaryEmbedding, ...]
                }
            }
        """
        multi_granularity_embeddings = {}
        
        for model_name in self.config['models']['embedding_models']:
            self.logger.info(f"🔄 Processing multi-granularity embeddings for model: {model_name}")
            
            # Generate config hash for cache validation
            config_hash = self._generate_config_hash(model_name, chunks, articles)
            
            # Check cache
            cache_path = self._get_cache_path(model_name)
            if not force_recompute and self._is_cache_valid(cache_path, config_hash):
                self.logger.info(f"📂 Loading cached multi-granularity embeddings for {model_name}")
                multi_granularity_embeddings[model_name] = self._load_cached_embeddings(cache_path)
            else:
                self.logger.info(f"✨ Generating fresh multi-granularity embeddings for {model_name}")
                multi_granularity_embeddings[model_name] = self._generate_fresh_embeddings(
                    model_name, chunks, articles, config_hash, cache_path
                )
        
        return multi_granularity_embeddings
    
    def _generate_fresh_embeddings(self, model_name: str, chunks: List[Dict[str, Any]], 
                                 articles: List[Any], config_hash: str, cache_path: Path) -> Dict[str, List[Any]]:
        """Generate fresh multi-granularity embeddings for a model."""
        start_time = time.time()
        
        # Initialize model
        embedding_model = EmbeddingModel(model_name, self.device, self.logger)
        
        # Initialize result dictionary
        granularity_embeddings = {}
        
        # Generate chunks embeddings (existing logic)
        if self.granularity_config['chunks'].get('enabled', True):
            self.logger.info("🔨 Generating chunk embeddings...")
            granularity_embeddings['chunks'] = self._generate_chunk_embeddings(
                embedding_model, chunks
            )
        
        # Generate sentence embeddings (new)
        if self.granularity_config['sentences'].get('enabled', True):
            self.logger.info("📝 Generating sentence embeddings...")
            granularity_embeddings['sentences'] = self._generate_sentence_embeddings(
                embedding_model, articles, chunks
            )
        
        # Generate document summary embeddings (new)
        if self.granularity_config['doc_summaries'].get('enabled', True):
            self.logger.info("📄 Generating document summary embeddings...")
            granularity_embeddings['doc_summaries'] = self._generate_document_summary_embeddings(
                embedding_model, articles
            )
        
        processing_time = time.time() - start_time
        
        # Create metadata
        granularity_counts = {
            granularity_type: len(embeddings) 
            for granularity_type, embeddings in granularity_embeddings.items()
        }
        
        metadata = MultiGranularityEmbeddingMetadata(
            model_name=model_name,
            embedding_dimension=embedding_model.embedding_dimension,
            granularity_counts=granularity_counts,
            created_at=datetime.now().isoformat(),
            device=self.device,
            processing_time=processing_time,
            config_hash=config_hash,
            granularity_config=self.granularity_config
        )
        
        # Cache embeddings
        self._cache_embeddings(cache_path, granularity_embeddings, metadata)
        
        # Log results
        for granularity_type, count in granularity_counts.items():
            self.logger.info(f"   ✅ {granularity_type}: {count:,} embeddings")
        self.logger.info(f"🎯 Multi-granularity generation completed in {processing_time:.2f}s")
        
        return granularity_embeddings
    
    def _generate_chunk_embeddings(self, embedding_model: EmbeddingModel, 
                                 chunks: List[Dict[str, Any]]) -> List[ChunkEmbedding]:
        """Generate chunk embeddings (unchanged from original logic)."""
        texts = [chunk['text'] for chunk in chunks]
        batch_size = self.config['models']['embedding_batch_size']
        
        # Generate embeddings
        embeddings = embedding_model.encode_batch(texts, batch_size=batch_size)
        
        # Create ChunkEmbedding objects
        chunk_embeddings = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_embedding = ChunkEmbedding(
                chunk_id=chunk['chunk_id'],
                chunk_text=chunk['text'],
                embedding=embedding,
                source_article=chunk['source_article'],
                source_sentences=chunk['source_sentences'],
                anchor_sentence_idx=chunk['anchor_sentence_idx']
            )
            chunk_embeddings.append(chunk_embedding)
        
        return chunk_embeddings
    
    def _generate_sentence_embeddings(self, embedding_model: EmbeddingModel, 
                                    articles: List[Any], chunks: List[Dict[str, Any]]) -> List[SentenceEmbedding]:
        """Generate sentence embeddings from articles."""
        sentence_embeddings = []
        
        # Create chunk lookup for finding containing chunks
        chunk_lookup = {}  # sentence_index -> [chunk_ids]
        for chunk in chunks:
            for sentence_idx in chunk['source_sentences']:
                sentence_key = f"{chunk['source_article']}_{sentence_idx}"
                if sentence_key not in chunk_lookup:
                    chunk_lookup[sentence_key] = []
                chunk_lookup[sentence_key].append(chunk['chunk_id'])
        
        # Extract all sentences from articles
        all_sentences = []
        sentence_metadata = []
        
        for article in articles:
            for i, sentence in enumerate(article.sentences):
                sentence_key = f"{article.title}_{i}"
                containing_chunks = chunk_lookup.get(sentence_key, [])
                
                sentence_id = f"sent_{hashlib.md5((article.title + sentence).encode()).hexdigest()[:8]}"
                
                all_sentences.append(sentence)
                sentence_metadata.append({
                    'sentence_id': sentence_id,
                    'source_article': article.title,
                    'sentence_index': i,
                    'containing_chunks': containing_chunks
                })
        
        if not all_sentences:
            return sentence_embeddings
        
        # Generate embeddings for all sentences
        batch_size = self.config['models']['embedding_batch_size']
        embeddings = embedding_model.encode_batch(all_sentences, batch_size=batch_size)
        
        # Create SentenceEmbedding objects
        for sentence_text, embedding, metadata in zip(all_sentences, embeddings, sentence_metadata):
            sentence_embedding = SentenceEmbedding(
                sentence_id=metadata['sentence_id'],
                sentence_text=sentence_text,
                embedding=embedding,
                source_article=metadata['source_article'],
                sentence_index=metadata['sentence_index'],
                containing_chunks=metadata['containing_chunks']
            )
            sentence_embeddings.append(sentence_embedding)
        
        return sentence_embeddings
    
    def _generate_document_summary_embeddings(self, embedding_model: EmbeddingModel, 
                                            articles: List[Any]) -> List[DocumentSummaryEmbedding]:
        """Generate document summary embeddings."""
        doc_summary_embeddings = []
        
        # Extract summaries from articles
        all_summaries = []
        summary_metadata = []
        
        summary_config = self.granularity_config['doc_summaries']
        max_sentences = summary_config.get('max_sentences', 3)
        method = summary_config.get('method', 'extractive')
        
        for article in articles:
            # Create extractive summary (first N sentences)
            summary_sentences = article.sentences[:max_sentences]
            summary_text = ' '.join(summary_sentences)
            
            if not summary_text.strip():
                continue
            
            doc_id = f"doc_{hashlib.md5(article.title.encode()).hexdigest()[:8]}"
            
            all_summaries.append(summary_text)
            summary_metadata.append({
                'doc_id': doc_id,
                'doc_title': article.title,
                'source_article': article.title,
                'summary_method': method,
                'total_sentences': len(article.sentences)
            })
        
        if not all_summaries:
            return doc_summary_embeddings
        
        # Generate embeddings for all summaries
        batch_size = self.config['models']['embedding_batch_size']
        embeddings = embedding_model.encode_batch(all_summaries, batch_size=batch_size)
        
        # Create DocumentSummaryEmbedding objects
        for summary_text, embedding, metadata in zip(all_summaries, embeddings, summary_metadata):
            doc_summary_embedding = DocumentSummaryEmbedding(
                doc_id=metadata['doc_id'],
                doc_title=metadata['doc_title'],
                summary_text=summary_text,
                embedding=embedding,
                source_article=metadata['source_article'],
                summary_method=metadata['summary_method'],
                total_sentences=metadata['total_sentences']
            )
            doc_summary_embeddings.append(doc_summary_embedding)
        
        return doc_summary_embeddings
    
    def _get_cache_path(self, model_name: str) -> Path:
        """Get cache path for a model's multi-granularity embeddings."""
        safe_model_name = model_name.replace("/", "_").replace("-", "_")
        return self.embeddings_dir / "raw" / f"{safe_model_name}_multi_granularity.json"
    
    def _generate_config_hash(self, model_name: str, chunks: List[Dict[str, Any]], 
                            articles: List[Any]) -> str:
        """Generate hash of configuration and data for cache validation."""
        config_str = json.dumps({
            'model_name': model_name,
            'granularity_config': self.granularity_config,
            'chunking_strategy': self.config['chunking']['strategy'],
            'chunking_params': {
                'window_size': self.config['chunking']['window_size'],
                'overlap': self.config['chunking']['overlap']
            },
            'chunk_count': len(chunks),
            'article_count': len(articles),
            'first_chunk_text': chunks[0]['text'] if chunks else "",
            'first_article_title': articles[0].title if articles else ""
        }, sort_keys=True)
        
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_path: Path, expected_hash: str) -> bool:
        """Check if cached multi-granularity embeddings are valid."""
        if not cache_path.exists():
            return False
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            metadata = cache_data.get('metadata', {})
            cached_hash = metadata.get('config_hash', '')
            
            return cached_hash == expected_hash
            
        except Exception as e:
            self.logger.warning(f"Failed to validate multi-granularity cache: {e}")
            return False
    
    def _cache_embeddings(self, cache_path: Path, granularity_embeddings: Dict[str, List[Any]], 
                         metadata: MultiGranularityEmbeddingMetadata):
        """Cache multi-granularity embeddings to disk."""
        try:
            # Convert embeddings to dictionaries
            cache_data = {
                'metadata': asdict(metadata),
                'embeddings': {}
            }
            
            for granularity_type, embeddings in granularity_embeddings.items():
                cache_data['embeddings'][granularity_type] = [emb.to_dict() for emb in embeddings]
            
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"💾 Cached multi-granularity embeddings to {cache_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to cache multi-granularity embeddings: {e}")
            raise
    
    def _load_cached_embeddings(self, cache_path: Path) -> Dict[str, List[Any]]:
        """Load cached multi-granularity embeddings from disk."""
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            granularity_embeddings = {}
            
            for granularity_type, emb_data_list in cache_data['embeddings'].items():
                embeddings = []
                
                for emb_data in emb_data_list:
                    if granularity_type == 'chunks':
                        embedding = ChunkEmbedding(
                            chunk_id=emb_data['chunk_id'],
                            chunk_text=emb_data['chunk_text'],
                            embedding=np.array(emb_data['embedding']),
                            source_article=emb_data['source_article'],
                            source_sentences=emb_data['source_sentences'],
                            anchor_sentence_idx=emb_data['anchor_sentence_idx']
                        )
                    elif granularity_type == 'sentences':
                        embedding = SentenceEmbedding(
                            sentence_id=emb_data['sentence_id'],
                            sentence_text=emb_data['sentence_text'],
                            embedding=np.array(emb_data['embedding']),
                            source_article=emb_data['source_article'],
                            sentence_index=emb_data['sentence_index'],
                            containing_chunks=emb_data['containing_chunks']
                        )
                    elif granularity_type == 'doc_summaries':
                        embedding = DocumentSummaryEmbedding(
                            doc_id=emb_data['doc_id'],
                            doc_title=emb_data['doc_title'],
                            summary_text=emb_data['summary_text'],
                            embedding=np.array(emb_data['embedding']),
                            source_article=emb_data['source_article'],
                            summary_method=emb_data['summary_method'],
                            total_sentences=emb_data['total_sentences']
                        )
                    
                    embeddings.append(embedding)
                
                granularity_embeddings[granularity_type] = embeddings
            
            return granularity_embeddings
            
        except Exception as e:
            self.logger.error(f"Failed to load cached multi-granularity embeddings: {e}")
            raise
    
    def get_embedding_statistics(self, multi_granularity_embeddings: Dict[str, Dict[str, List[Any]]]) -> Dict[str, Any]:
        """Get statistics about generated multi-granularity embeddings."""
        stats = {}
        
        for model_name, granularity_embeddings in multi_granularity_embeddings.items():
            model_stats = {
                'granularity_types': {},
                'total_embeddings': 0
            }
            
            for granularity_type, embeddings in granularity_embeddings.items():
                if not embeddings:
                    continue
                
                # Get first embedding to determine dimension
                first_embedding = embeddings[0]
                embedding_vectors = np.array([emb.embedding for emb in embeddings])
                
                granularity_stats = {
                    'count': len(embeddings),
                    'embedding_dimension': embedding_vectors.shape[1],
                    'mean_norm': float(np.mean(np.linalg.norm(embedding_vectors, axis=1))),
                    'std_norm': float(np.std(np.linalg.norm(embedding_vectors, axis=1)))
                }
                
                # Add granularity-specific stats
                if granularity_type == 'chunks':
                    granularity_stats['sample_chunk_lengths'] = [
                        len(emb.chunk_text.split()) for emb in embeddings[:5]
                    ]
                elif granularity_type == 'sentences':
                    granularity_stats['sample_sentence_lengths'] = [
                        len(emb.sentence_text.split()) for emb in embeddings[:5]
                    ]
                elif granularity_type == 'doc_summaries':
                    granularity_stats['sample_summary_lengths'] = [
                        len(emb.summary_text.split()) for emb in embeddings[:5]
                    ]
                
                model_stats['granularity_types'][granularity_type] = granularity_stats
                model_stats['total_embeddings'] += len(embeddings)
            
            stats[model_name] = model_stats
        
        return stats
    
    def load_model_embeddings(self, model_name: str) -> Optional[Dict[str, List[Any]]]:
        """Load cached embeddings for a specific model.
        
        Args:
            model_name: Name of the embedding model
            
        Returns:
            Dictionary with granularity embeddings or None if not found
        """
        try:
            cache_path = self._get_cache_path(model_name)
            
            if not cache_path.exists():
                self.logger.info(f"No cached embeddings found for model: {model_name}")
                return None
            
            self.logger.info(f"Loading cached embeddings for model: {model_name}")
            embeddings = self._load_cached_embeddings(cache_path)
            
            self.logger.info(f"Successfully loaded {len(embeddings)} granularity types")
            for granularity_type, emb_list in embeddings.items():
                self.logger.info(f"  {granularity_type}: {len(emb_list)} embeddings")
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Failed to load model embeddings for {model_name}: {e}")
            return None


# Backwards compatibility: Alias the new class as EmbeddingEngine
EmbeddingEngine = MultiGranularityEmbeddingEngine