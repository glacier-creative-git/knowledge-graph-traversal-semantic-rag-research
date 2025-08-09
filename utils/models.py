#!/usr/bin/env python3
"""
Embedding Models Engine
======================

Handles multiple embedding model abstractions with unified interface for the semantic RAG pipeline.
Supports sentence-transformers and other embedding providers with intelligent caching.
"""

import json
import hashlib
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


@dataclass
class EmbeddingMetadata:
    """Metadata for cached embeddings."""
    model_name: str
    embedding_dimension: int
    total_chunks: int
    created_at: str
    device: str
    processing_time: float
    config_hash: str
    chunk_strategy: str
    chunk_params: Dict[str, Any]


@dataclass
class ChunkEmbedding:
    """Container for a chunk and its embedding."""
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


class EmbeddingModel:
    """Wrapper for embedding models with unified interface."""
    
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


class EmbeddingEngine:
    """Engine for generating and caching embeddings for multiple models."""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """Initialize the embedding engine."""
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.device = config['system']['device']
        self.embeddings_dir = Path(config['directories']['embeddings'])
        
        self.logger.info(f"EmbeddingEngine initialized with device: {self.device}")
        
        # Create embeddings directory structure
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        (self.embeddings_dir / "raw").mkdir(exist_ok=True)
    
    def generate_embeddings(self, chunks: List[Dict[str, Any]], force_recompute: bool = False) -> Dict[str, List[ChunkEmbedding]]:
        """
        Generate embeddings for chunks using all configured models.
        
        Args:
            chunks: List of chunk dictionaries from ChunkEngine
            force_recompute: Whether to force recomputation even if cached
            
        Returns:
            Dictionary mapping model names to lists of ChunkEmbedding objects
        """
        model_embeddings = {}
        
        for model_name in self.config['models']['embedding_models']:
            self.logger.info(f"Processing model: {model_name}")
            
            # Generate config hash for cache validation
            config_hash = self._generate_config_hash(model_name, chunks)
            
            # Check cache
            cache_path = self._get_cache_path(model_name)
            if not force_recompute and self._is_cache_valid(cache_path, config_hash):
                self.logger.info(f"Loading cached embeddings for {model_name}")
                model_embeddings[model_name] = self._load_cached_embeddings(cache_path)
            else:
                self.logger.info(f"Generating fresh embeddings for {model_name}")
                model_embeddings[model_name] = self._generate_fresh_embeddings(
                    model_name, chunks, config_hash, cache_path
                )
        
        return model_embeddings
    
    def _generate_fresh_embeddings(self, model_name: str, chunks: List[Dict[str, Any]], 
                                 config_hash: str, cache_path: Path) -> List[ChunkEmbedding]:
        """Generate fresh embeddings for a model."""
        start_time = time.time()
        
        # Initialize model
        embedding_model = EmbeddingModel(model_name, self.device, self.logger)
        
        # Extract texts for embedding
        texts = [chunk['text'] for chunk in chunks]
        batch_size = self.config['models']['embedding_batch_size']
        
        self.logger.info(f"Generating embeddings for {len(texts)} chunks")
        
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
        
        processing_time = time.time() - start_time
        
        # Create metadata
        metadata = EmbeddingMetadata(
            model_name=model_name,
            embedding_dimension=embedding_model.embedding_dimension,
            total_chunks=len(chunk_embeddings),
            created_at=datetime.now().isoformat(),
            device=self.device,
            processing_time=processing_time,
            config_hash=config_hash,
            chunk_strategy=self.config['chunking']['strategy'],
            chunk_params={
                'window_size': self.config['chunking']['window_size'],
                'overlap': self.config['chunking']['overlap']
            }
        )
        
        # Cache embeddings
        self._cache_embeddings(cache_path, chunk_embeddings, metadata)
        
        self.logger.info(f"Generated embeddings for {model_name}: {len(chunk_embeddings)} chunks in {processing_time:.2f}s")
        
        return chunk_embeddings
    
    def _get_cache_path(self, model_name: str) -> Path:
        """Get cache path for a model."""
        safe_model_name = model_name.replace("/", "_").replace("-", "_")
        return self.embeddings_dir / "raw" / f"{safe_model_name}.json"
    
    def _generate_config_hash(self, model_name: str, chunks: List[Dict[str, Any]]) -> str:
        """Generate hash of configuration and chunks for cache validation."""
        config_str = json.dumps({
            'model_name': model_name,
            'chunking_strategy': self.config['chunking']['strategy'],
            'chunking_params': {
                'window_size': self.config['chunking']['window_size'],
                'overlap': self.config['chunking']['overlap']
            },
            'chunk_count': len(chunks),
            'first_chunk_text': chunks[0]['text'] if chunks else "",
            'last_chunk_text': chunks[-1]['text'] if chunks else ""
        }, sort_keys=True)
        
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_path: Path, expected_hash: str) -> bool:
        """Check if cached embeddings are valid."""
        if not cache_path.exists():
            return False
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            metadata = cache_data.get('metadata', {})
            cached_hash = metadata.get('config_hash', '')
            
            return cached_hash == expected_hash
            
        except Exception as e:
            self.logger.warning(f"Failed to validate cache: {e}")
            return False
    
    def _cache_embeddings(self, cache_path: Path, embeddings: List[ChunkEmbedding], metadata: EmbeddingMetadata):
        """Cache embeddings to disk."""
        try:
            cache_data = {
                'metadata': asdict(metadata),
                'embeddings': [emb.to_dict() for emb in embeddings]
            }
            
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Cached embeddings to {cache_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to cache embeddings: {e}")
            raise
    
    def _load_cached_embeddings(self, cache_path: Path) -> List[ChunkEmbedding]:
        """Load cached embeddings from disk."""
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            embeddings = []
            for emb_data in cache_data['embeddings']:
                embedding = ChunkEmbedding(
                    chunk_id=emb_data['chunk_id'],
                    chunk_text=emb_data['chunk_text'],
                    embedding=np.array(emb_data['embedding']),
                    source_article=emb_data['source_article'],
                    source_sentences=emb_data['source_sentences'],
                    anchor_sentence_idx=emb_data['anchor_sentence_idx']
                )
                embeddings.append(embedding)
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Failed to load cached embeddings: {e}")
            raise
    
    def get_embedding_statistics(self, model_embeddings: Dict[str, List[ChunkEmbedding]]) -> Dict[str, Any]:
        """Get statistics about generated embeddings."""
        stats = {}
        
        for model_name, embeddings in model_embeddings.items():
            if not embeddings:
                continue
            
            embedding_vectors = np.array([emb.embedding for emb in embeddings])
            
            stats[model_name] = {
                'total_chunks': len(embeddings),
                'embedding_dimension': embedding_vectors.shape[1],
                'mean_norm': float(np.mean(np.linalg.norm(embedding_vectors, axis=1))),
                'std_norm': float(np.std(np.linalg.norm(embedding_vectors, axis=1))),
                'sample_chunk_lengths': [len(emb.chunk_text.split()) for emb in embeddings[:5]]
            }
        
        return stats
