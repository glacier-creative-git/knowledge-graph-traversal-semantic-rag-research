#!/usr/bin/env python3
"""
Enhanced Multi-Granularity Similarity Matrix Engine
=================================================

Handles multi-granularity similarity matrix computation and storage for the enhanced semantic RAG pipeline.
Computes sparse similarity matrices across multiple granularity types with configurable top-k constraints
for memory-efficient semantic traversal.

Architecture:
- chunk_to_chunk: Existing intra/inter-document similarities (enhanced)
- doc_to_doc: Document-level similarities via summary embeddings
- sentence_to_sentence: Fine-grained semantic and sequential relationships  
- cross_granularity: Relationships across different granularity levels
"""

import json
import hashlib
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict

import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from models import ChunkEmbedding, SentenceEmbedding, DocumentSummaryEmbedding


@dataclass
class MultiGranularitySimilarityMetadata:
    """Metadata for cached multi-granularity similarity matrices."""
    created_at: str
    granularity_counts: Dict[str, int]  # chunks: 100, sentences: 300, doc_summaries: 10
    embedding_models: List[str]
    similarity_metric: str
    granularity_config: Dict[str, Any]
    config_hash: str
    computation_time: float
    memory_usage_mb: float
    total_connections: int
    sparsity_statistics: Dict[str, Any]


@dataclass
class SimilarityConnection:
    """Represents a similarity connection between any two items (enhanced from original)."""
    source_idx: int
    target_idx: int
    similarity_score: float
    connection_type: str  # 'chunk_to_chunk_intra', 'chunk_to_chunk_inter', 'doc_to_doc', etc.
    source_id: str
    target_id: str
    granularity_type: str  # 'chunk_to_chunk', 'doc_to_doc', 'sentence_to_sentence', 'cross_granularity'


class MultiGranularitySimilarityEngine:
    """Enhanced engine for computing and managing multi-granularity similarity matrices."""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """Initialize the multi-granularity similarity engine."""
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.similarity_config = config['similarities']
        self.granularity_config = self.similarity_config['granularity_types']
        self.similarities_dir = Path(config['directories']['embeddings']) / "similarities"
        
        # Create similarities directory
        self.similarities_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate configuration
        self._validate_config()
        
        self.logger.info("🔗 Initialized multi-granularity similarity engine")
        self._log_configuration()
    
    def _validate_config(self):
        """Validate multi-granularity similarity configuration."""
        sim_config = self.similarity_config
        
        # Validate metric
        valid_metrics = ['cosine', 'dot_product', 'euclidean']
        if sim_config['similarity_metric'] not in valid_metrics:
            raise ValueError(f"Invalid similarity metric. Must be one of: {valid_metrics}")
        
        # Validate granularity type configurations
        for granularity_type, config in self.granularity_config.items():
            if not config.get('enabled', False):
                continue
                
            if granularity_type == 'chunk_to_chunk':
                if config['intra_document']['enabled'] and config['intra_document']['top_k'] <= 0:
                    raise ValueError("chunk_to_chunk intra_document top_k must be positive")
                if config['inter_document']['enabled'] and config['inter_document']['top_x'] <= 0:
                    raise ValueError("chunk_to_chunk inter_document top_x must be positive")
            elif granularity_type in ['doc_to_doc', 'sentence_to_sentence']:
                if config.get('top_k', 0) <= 0:
                    raise ValueError(f"{granularity_type} top_k must be positive")
            elif granularity_type == 'cross_granularity':
                if config['sentence_to_chunk']['enabled'] and config['sentence_to_chunk']['top_k'] <= 0:
                    raise ValueError("cross_granularity sentence_to_chunk top_k must be positive")
        
        # Validate batch size
        if sim_config['batch_size'] <= 0:
            raise ValueError("batch_size must be positive")
    
    def _log_configuration(self):
        """Log the multi-granularity configuration."""
        enabled_types = [gt for gt, conf in self.granularity_config.items() if conf.get('enabled', False)]
        self.logger.info(f"Enabled granularity types: {enabled_types}")
        
        for granularity_type in enabled_types:
            config = self.granularity_config[granularity_type]
            if granularity_type == 'chunk_to_chunk':
                self.logger.info(f"   {granularity_type}: intra_top_k={config['intra_document']['top_k']}, inter_top_x={config['inter_document']['top_x']}")
            elif granularity_type in ['doc_to_doc', 'sentence_to_sentence']:
                if 'semantic' in config:
                    self.logger.info(f"   {granularity_type}: semantic_top_k={config['semantic']['top_k']}, sequential={config.get('sequential', {}).get('enabled', False)}")
                else:
                    self.logger.info(f"   {granularity_type}: top_k={config['top_k']}")
            elif granularity_type == 'cross_granularity':
                self.logger.info(f"   {granularity_type}: sentence_to_chunk_top_k={config['sentence_to_chunk']['top_k']}")
    
    def compute_similarity_matrices(self, multi_granularity_embeddings: Dict[str, Dict[str, List[Any]]], 
                                  force_recompute: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Compute multi-granularity similarity matrices for all embedding models.
        
        Args:
            multi_granularity_embeddings: Dictionary mapping model names to granularity dictionaries
            force_recompute: Whether to force recomputation even if cached
            
        Returns:
            Dictionary mapping model names to multi-granularity similarity data
        """
        similarity_results = {}
        
        for model_name, granularity_embeddings in multi_granularity_embeddings.items():
            self.logger.info(f"🎯 Computing multi-granularity similarities for model: {model_name}")
            
            # Generate config hash for cache validation
            config_hash = self._generate_config_hash(model_name, granularity_embeddings)
            
            # Check cache
            cache_path = self._get_cache_path(model_name)
            if not force_recompute and self._is_cache_valid(cache_path, config_hash):
                self.logger.info(f"📂 Loading cached multi-granularity similarities for {model_name}")
                similarity_results[model_name] = self._load_cached_similarities(cache_path)
            else:
                self.logger.info(f"⚡ Computing fresh multi-granularity similarities for {model_name}")
                similarity_results[model_name] = self._compute_fresh_similarities(
                    model_name, granularity_embeddings, config_hash, cache_path
                )
        
        return similarity_results
    
    def _compute_fresh_similarities(self, model_name: str, granularity_embeddings: Dict[str, List[Any]],
                                  config_hash: str, cache_path: Path) -> Dict[str, Any]:
        """Compute fresh multi-granularity similarity matrices for a model."""
        start_time = time.time()
        
        # Log granularity counts
        granularity_counts = {gt: len(embeddings) for gt, embeddings in granularity_embeddings.items()}
        self.logger.info(f"Computing similarities for: {granularity_counts}")
        
        # Initialize storage for all matrices and connections
        all_matrices = {}
        all_connections = []
        all_index_maps = {}
        
        # Compute similarities for each enabled granularity type
        if self.granularity_config['chunk_to_chunk'].get('enabled', False):
            self.logger.info("🔗 Computing chunk-to-chunk similarities...")
            chunk_matrices, chunk_connections, chunk_index_map = self._compute_chunk_to_chunk_similarities(
                granularity_embeddings.get('chunks', [])
            )
            all_matrices.update(chunk_matrices)
            all_connections.extend(chunk_connections)
            all_index_maps['chunks'] = chunk_index_map
        
        if self.granularity_config['doc_to_doc'].get('enabled', False):
            self.logger.info("📄 Computing document-to-document similarities...")
            doc_matrices, doc_connections, doc_index_map = self._compute_doc_to_doc_similarities(
                granularity_embeddings.get('doc_summaries', [])
            )
            all_matrices.update(doc_matrices)
            all_connections.extend(doc_connections)
            all_index_maps['doc_summaries'] = doc_index_map
        
        if self.granularity_config['sentence_to_sentence'].get('enabled', False):
            self.logger.info("📝 Computing sentence-to-sentence similarities...")
            sentence_matrices, sentence_connections, sentence_index_map = self._compute_sentence_to_sentence_similarities(
                granularity_embeddings.get('sentences', [])
            )
            all_matrices.update(sentence_matrices)
            all_connections.extend(sentence_connections)
            all_index_maps['sentences'] = sentence_index_map
        
        if self.granularity_config['cross_granularity'].get('enabled', False):
            self.logger.info("🌉 Computing cross-granularity similarities...")
            cross_matrices, cross_connections = self._compute_cross_granularity_similarities(
                granularity_embeddings, all_index_maps
            )
            all_matrices.update(cross_matrices)
            all_connections.extend(cross_connections)
        
        computation_time = time.time() - start_time
        
        # Calculate statistics
        total_connections = len(all_connections)
        memory_usage_mb = (total_connections * 16) / (1024 * 1024)  # Estimated 16 bytes per connection
        
        # Calculate sparsity statistics
        sparsity_stats = self._calculate_sparsity_statistics(all_matrices, granularity_counts)
        
        # Create metadata
        metadata = MultiGranularitySimilarityMetadata(
            created_at=datetime.now().isoformat(),
            granularity_counts=granularity_counts,
            embedding_models=[model_name],
            similarity_metric=self.similarity_config['similarity_metric'],
            granularity_config=self.granularity_config,
            config_hash=config_hash,
            computation_time=computation_time,
            memory_usage_mb=memory_usage_mb,
            total_connections=total_connections,
            sparsity_statistics=sparsity_stats
        )
        
        # Package results
        similarity_data = {
            'metadata': metadata,
            'matrices': all_matrices,
            'connections': all_connections,
            'index_maps': all_index_maps
        }
        
        # Cache results
        self._cache_similarities(cache_path, similarity_data)
        
        # Log comprehensive statistics
        self._log_computation_results(metadata, all_matrices)
        
        return similarity_data
    
    def _compute_chunk_to_chunk_similarities(self, chunk_embeddings: List[ChunkEmbedding]) -> Tuple[Dict[str, sp.csr_matrix], List[SimilarityConnection], Dict[str, int]]:
        """Compute chunk-to-chunk similarities (enhanced from original logic)."""
        if not chunk_embeddings:
            return {}, [], {}
        
        # Create index mapping
        chunk_index_map = {emb.chunk_id: idx for idx, emb in enumerate(chunk_embeddings)}
        
        # Group embeddings by document for intra-document similarities
        doc_groups = self._group_embeddings_by_document(chunk_embeddings)
        
        # Initialize connection storage
        all_connections = []
        
        config = self.granularity_config['chunk_to_chunk']
        
        # Compute intra-document similarities
        if config['intra_document']['enabled']:
            intra_connections = self._compute_intra_document_similarities(
                doc_groups, chunk_embeddings, config['intra_document']
            )
            all_connections.extend(intra_connections)
        
        # Compute inter-document similarities
        if config['inter_document']['enabled']:
            inter_connections = self._compute_inter_document_similarities(
                doc_groups, chunk_embeddings, config['inter_document']
            )
            all_connections.extend(inter_connections)
        
        # Build sparse matrices
        matrices = self._build_chunk_sparse_matrices(all_connections, len(chunk_embeddings))
        
        return matrices, all_connections, chunk_index_map
    
    def _compute_doc_to_doc_similarities(self, doc_embeddings: List[DocumentSummaryEmbedding]) -> Tuple[Dict[str, sp.csr_matrix], List[SimilarityConnection], Dict[str, int]]:
        """Compute document-to-document similarities via summary embeddings."""
        if not doc_embeddings:
            return {}, [], {}
        
        # Create index mapping
        doc_index_map = {emb.doc_id: idx for idx, emb in enumerate(doc_embeddings)}
        
        config = self.granularity_config['doc_to_doc']
        top_k = config['top_k']
        min_threshold = config.get('min_threshold', 0.0)
        
        # Extract embeddings matrix
        embeddings_matrix = np.array([emb.embedding for emb in doc_embeddings])
        
        # Compute similarity matrix
        sim_matrix = self._compute_similarity_matrix(embeddings_matrix)
        
        # Extract top-k connections
        all_connections = []
        
        for i, source_emb in enumerate(doc_embeddings):
            # Get similarities for this document (excluding self-similarity)
            similarities = sim_matrix[i].copy()
            similarities[i] = -np.inf  # Exclude self
            
            # Apply minimum threshold
            similarities[similarities < min_threshold] = -np.inf
            
            # Get top-k indices
            valid_similarities = similarities[similarities > -np.inf]
            if len(valid_similarities) > 0:
                top_indices = np.argsort(similarities)[-top_k:][::-1]  # Descending order
                
                # Create connections
                for j in top_indices:
                    similarity_score = float(similarities[j])
                    if similarity_score > -np.inf:
                        connection = SimilarityConnection(
                            source_idx=i,
                            target_idx=j,
                            similarity_score=similarity_score,
                            connection_type='doc_to_doc',
                            source_id=source_emb.doc_id,
                            target_id=doc_embeddings[j].doc_id,
                            granularity_type='doc_to_doc'
                        )
                        all_connections.append(connection)
        
        # Build sparse matrix
        matrices = self._build_single_sparse_matrix(all_connections, len(doc_embeddings), 'doc_to_doc')
        
        return matrices, all_connections, doc_index_map
    
    def _compute_sentence_to_sentence_similarities(self, sentence_embeddings: List[SentenceEmbedding]) -> Tuple[Dict[str, sp.csr_matrix], List[SimilarityConnection], Dict[str, int]]:
        """Compute sentence-to-sentence similarities (semantic and sequential)."""
        if not sentence_embeddings:
            return {}, [], {}
        
        # Create index mapping
        sentence_index_map = {emb.sentence_id: idx for idx, emb in enumerate(sentence_embeddings)}
        
        all_connections = []
        matrices = {}
        
        config = self.granularity_config['sentence_to_sentence']
        
        # Compute semantic similarities
        if config['semantic']['enabled']:
            semantic_connections = self._compute_sentence_semantic_similarities(
                sentence_embeddings, config['semantic']
            )
            all_connections.extend(semantic_connections)
        
        # Compute sequential similarities (adjacent sentences)
        if config['sequential']['enabled']:
            sequential_connections = self._compute_sentence_sequential_similarities(
                sentence_embeddings
            )
            all_connections.extend(sequential_connections)
        
        # Build sparse matrices
        matrices = self._build_sentence_sparse_matrices(all_connections, len(sentence_embeddings))
        
        return matrices, all_connections, sentence_index_map
    
    def _compute_cross_granularity_similarities(self, granularity_embeddings: Dict[str, List[Any]], 
                                              index_maps: Dict[str, Dict[str, int]]) -> Tuple[Dict[str, sp.csr_matrix], List[SimilarityConnection]]:
        """Compute cross-granularity similarities between different granularity levels."""
        all_connections = []
        matrices = {}
        
        config = self.granularity_config['cross_granularity']
        
        # Sentence-to-chunk similarities
        if config['sentence_to_chunk']['enabled']:
            sentence_to_chunk_connections = self._compute_sentence_to_chunk_similarities(
                granularity_embeddings.get('sentences', []),
                granularity_embeddings.get('chunks', []),
                config['sentence_to_chunk']
            )
            all_connections.extend(sentence_to_chunk_connections)
        
        # Chunk-to-document similarities
        if config['chunk_to_doc']['enabled']:
            chunk_to_doc_connections = self._compute_chunk_to_doc_similarities(
                granularity_embeddings.get('chunks', []),
                granularity_embeddings.get('doc_summaries', [])
            )
            all_connections.extend(chunk_to_doc_connections)
        
        # Build cross-granularity matrices (more complex due to different dimensions)
        matrices = self._build_cross_granularity_matrices(all_connections, granularity_embeddings)
        
        return matrices, all_connections
    
    def _compute_similarity_matrix(self, embeddings_matrix: np.ndarray) -> np.ndarray:
        """Compute similarity matrix using configured metric."""
        metric = self.similarity_config['similarity_metric']
        
        if metric == 'cosine':
            return cosine_similarity(embeddings_matrix)
        elif metric == 'dot_product':
            return np.dot(embeddings_matrix, embeddings_matrix.T)
        elif metric == 'euclidean':
            from sklearn.metrics.pairwise import euclidean_distances
            dist_matrix = euclidean_distances(embeddings_matrix)
            return 1 / (1 + dist_matrix)  # Convert distance to similarity
        else:
            raise ValueError(f"Unsupported similarity metric: {metric}")
    
    def _group_embeddings_by_document(self, embeddings: List[ChunkEmbedding]) -> Dict[str, List[Tuple[int, ChunkEmbedding]]]:
        """Group embeddings by source document."""
        doc_groups = {}
        for idx, embedding in enumerate(embeddings):
            doc_name = embedding.source_article
            if doc_name not in doc_groups:
                doc_groups[doc_name] = []
            doc_groups[doc_name].append((idx, embedding))
        return doc_groups
    
    def _compute_intra_document_similarities(self, doc_groups: Dict[str, List[Tuple[int, ChunkEmbedding]]], 
                                           all_embeddings: List[ChunkEmbedding], 
                                           config: Dict[str, Any]) -> List[SimilarityConnection]:
        """Compute top-k similarities within each document."""
        top_k = config['top_k']
        min_threshold = config.get('min_threshold', 0.0)
        
        all_connections = []
        
        for doc_name, doc_embeddings in tqdm(doc_groups.items(), desc="Intra-document similarities"):
            if len(doc_embeddings) < 2:
                continue  # Skip documents with only one chunk
            
            # Extract embeddings for this document
            embeddings_matrix = np.array([emb.embedding for _, emb in doc_embeddings])
            
            # Compute similarities within document
            sim_matrix = self._compute_similarity_matrix(embeddings_matrix)
            
            # For each chunk in document, find top-k most similar (excluding self)
            for i, (source_idx, source_emb) in enumerate(doc_embeddings):
                # Get similarities for this chunk (excluding self-similarity)
                similarities = sim_matrix[i].copy()
                similarities[i] = -np.inf  # Exclude self
                
                # Apply minimum threshold
                similarities[similarities < min_threshold] = -np.inf
                
                # Get top-k indices
                valid_similarities = similarities[similarities > -np.inf]
                if len(valid_similarities) > 0:
                    top_indices = np.argsort(similarities)[-top_k:][::-1]  # Descending order
                    
                    # Create connections
                    for j in top_indices:
                        target_idx, target_emb = doc_embeddings[j]
                        similarity_score = float(similarities[j])
                        
                        if similarity_score > -np.inf:
                            connection = SimilarityConnection(
                                source_idx=source_idx,
                                target_idx=target_idx,
                                similarity_score=similarity_score,
                                connection_type='chunk_to_chunk_intra',
                                source_id=source_emb.chunk_id,
                                target_id=target_emb.chunk_id,
                                granularity_type='chunk_to_chunk'
                            )
                            all_connections.append(connection)
        
        return all_connections
    
    def _compute_inter_document_similarities(self, doc_groups: Dict[str, List[Tuple[int, ChunkEmbedding]]], 
                                           all_embeddings: List[ChunkEmbedding],
                                           config: Dict[str, Any]) -> List[SimilarityConnection]:
        """Compute top-x similarities across different documents."""
        top_x = config['top_x']
        min_threshold = config.get('min_threshold', 0.0)
        batch_size = self.similarity_config['batch_size']
        
        all_connections = []
        
        # Create mapping from chunk index to document
        chunk_to_doc = {}
        for doc_name, doc_embeddings in doc_groups.items():
            for idx, _ in doc_embeddings:
                chunk_to_doc[idx] = doc_name
        
        # Extract all embeddings for cross-document comparison
        all_embeddings_matrix = np.array([emb.embedding for emb in all_embeddings])
        
        # Process in batches to manage memory
        num_chunks = len(all_embeddings)
        for start_idx in tqdm(range(0, num_chunks, batch_size), desc="Inter-document similarities"):
            end_idx = min(start_idx + batch_size, num_chunks)
            batch_embeddings = all_embeddings_matrix[start_idx:end_idx]
            
            # Compute similarities between batch and all embeddings
            sim_matrix = self._compute_cross_similarity_matrix(batch_embeddings, all_embeddings_matrix)
            
            # For each chunk in batch, find top-x from different documents
            for batch_i, global_i in enumerate(range(start_idx, end_idx)):
                source_doc = chunk_to_doc[global_i]
                similarities = sim_matrix[batch_i].copy()
                
                # Mask out chunks from the same document and self
                for j in range(num_chunks):
                    if chunk_to_doc[j] == source_doc:
                        similarities[j] = -np.inf
                
                # Apply minimum threshold
                similarities[similarities < min_threshold] = -np.inf
                
                # Get top-x indices from different documents
                valid_similarities = similarities[similarities > -np.inf]
                if len(valid_similarities) > 0:
                    top_indices = np.argsort(similarities)[-top_x:][::-1]  # Descending order
                    
                    # Create connections
                    for target_idx in top_indices:
                        similarity_score = float(similarities[target_idx])
                        if similarity_score > -np.inf:
                            connection = SimilarityConnection(
                                source_idx=global_i,
                                target_idx=target_idx,
                                similarity_score=similarity_score,
                                connection_type='chunk_to_chunk_inter',
                                source_id=all_embeddings[global_i].chunk_id,
                                target_id=all_embeddings[target_idx].chunk_id,
                                granularity_type='chunk_to_chunk'
                            )
                            all_connections.append(connection)
        
        return all_connections
    
    def _compute_cross_similarity_matrix(self, batch_embeddings: np.ndarray, all_embeddings: np.ndarray) -> np.ndarray:
        """Compute similarity matrix between batch and all embeddings."""
        metric = self.similarity_config['similarity_metric']
        
        if metric == 'cosine':
            return cosine_similarity(batch_embeddings, all_embeddings)
        elif metric == 'dot_product':
            return np.dot(batch_embeddings, all_embeddings.T)
        elif metric == 'euclidean':
            from sklearn.metrics.pairwise import euclidean_distances
            dist_matrix = euclidean_distances(batch_embeddings, all_embeddings)
            return 1 / (1 + dist_matrix)
        else:
            raise ValueError(f"Unsupported similarity metric: {metric}")
    
    def _compute_sentence_semantic_similarities(self, sentence_embeddings: List[SentenceEmbedding], 
                                              config: Dict[str, Any]) -> List[SimilarityConnection]:
        """Compute semantic similarities between sentences."""
        if not sentence_embeddings:
            return []
        
        top_k = config['top_k']
        min_threshold = config.get('min_threshold', 0.0)
        
        # Extract embeddings matrix
        embeddings_matrix = np.array([emb.embedding for emb in sentence_embeddings])
        
        # Compute similarity matrix
        sim_matrix = self._compute_similarity_matrix(embeddings_matrix)
        
        all_connections = []
        
        for i, source_emb in enumerate(sentence_embeddings):
            # Get similarities for this sentence (excluding self-similarity)
            similarities = sim_matrix[i].copy()
            similarities[i] = -np.inf  # Exclude self
            
            # Apply minimum threshold
            similarities[similarities < min_threshold] = -np.inf
            
            # Get top-k indices
            valid_similarities = similarities[similarities > -np.inf]
            if len(valid_similarities) > 0:
                top_indices = np.argsort(similarities)[-top_k:][::-1]  # Descending order
                
                # Create connections
                for j in top_indices:
                    similarity_score = float(similarities[j])
                    if similarity_score > -np.inf:
                        connection = SimilarityConnection(
                            source_idx=i,
                            target_idx=j,
                            similarity_score=similarity_score,
                            connection_type='sentence_to_sentence_semantic',
                            source_id=source_emb.sentence_id,
                            target_id=sentence_embeddings[j].sentence_id,
                            granularity_type='sentence_to_sentence'
                        )
                        all_connections.append(connection)
        
        return all_connections
    
    def _compute_sentence_sequential_similarities(self, sentence_embeddings: List[SentenceEmbedding]) -> List[SimilarityConnection]:
        """Compute sequential similarities between adjacent sentences."""
        all_connections = []
        
        # Group sentences by document and sort by sentence index
        doc_sentences = {}
        for idx, emb in enumerate(sentence_embeddings):
            doc_name = emb.source_article
            if doc_name not in doc_sentences:
                doc_sentences[doc_name] = []
            doc_sentences[doc_name].append((idx, emb))
        
        # Sort sentences within each document by sentence index
        for doc_name in doc_sentences:
            doc_sentences[doc_name].sort(key=lambda x: x[1].sentence_index)
        
        # Create sequential connections
        for doc_name, sorted_sentences in doc_sentences.items():
            for i in range(len(sorted_sentences) - 1):
                source_idx, source_emb = sorted_sentences[i]
                target_idx, target_emb = sorted_sentences[i + 1]
                
                # Sequential connections have similarity score of 1.0 (perfect adjacency)
                connection = SimilarityConnection(
                    source_idx=source_idx,
                    target_idx=target_idx,
                    similarity_score=1.0,
                    connection_type='sentence_to_sentence_sequential',
                    source_id=source_emb.sentence_id,
                    target_id=target_emb.sentence_id,
                    granularity_type='sentence_to_sentence'
                )
                all_connections.append(connection)
        
        return all_connections
    
    def _compute_sentence_to_chunk_similarities(self, sentence_embeddings: List[SentenceEmbedding], 
                                              chunk_embeddings: List[ChunkEmbedding],
                                              config: Dict[str, Any]) -> List[SimilarityConnection]:
        """Compute similarities from sentences to their containing chunks."""
        if not sentence_embeddings or not chunk_embeddings:
            return []
        
        top_k = config['top_k']
        min_threshold = config.get('min_threshold', 0.0)
        
        # Create chunk lookup
        chunk_lookup = {chunk.chunk_id: (idx, chunk) for idx, chunk in enumerate(chunk_embeddings)}
        
        all_connections = []
        
        for sent_idx, sentence_emb in enumerate(sentence_embeddings):
            # Get containing chunks
            containing_chunks = sentence_emb.containing_chunks
            
            if not containing_chunks:
                continue
            
            # For sentences with multiple containing chunks, compute similarities to each
            chunk_similarities = []
            
            for chunk_id in containing_chunks:
                if chunk_id in chunk_lookup:
                    chunk_idx, chunk_emb = chunk_lookup[chunk_id]
                    
                    # Compute similarity between sentence and chunk embeddings
                    similarity = cosine_similarity(
                        [sentence_emb.embedding],
                        [chunk_emb.embedding]
                    )[0][0]
                    
                    if similarity >= min_threshold:
                        chunk_similarities.append((chunk_idx, chunk_emb, similarity))
            
            # Keep top-k most similar chunks
            chunk_similarities.sort(key=lambda x: x[2], reverse=True)
            
            for chunk_idx, chunk_emb, similarity in chunk_similarities[:top_k]:
                connection = SimilarityConnection(
                    source_idx=sent_idx,
                    target_idx=chunk_idx,
                    similarity_score=float(similarity),
                    connection_type='sentence_to_chunk',
                    source_id=sentence_emb.sentence_id,
                    target_id=chunk_emb.chunk_id,
                    granularity_type='cross_granularity'
                )
                all_connections.append(connection)
        
        return all_connections
    
    def _compute_chunk_to_doc_similarities(self, chunk_embeddings: List[ChunkEmbedding], 
                                         doc_embeddings: List[DocumentSummaryEmbedding]) -> List[SimilarityConnection]:
        """Compute similarities from chunks to their parent document summaries."""
        if not chunk_embeddings or not doc_embeddings:
            return []
        
        # Create document lookup
        doc_lookup = {doc.source_article: (idx, doc) for idx, doc in enumerate(doc_embeddings)}
        
        all_connections = []
        
        for chunk_idx, chunk_emb in enumerate(chunk_embeddings):
            source_article = chunk_emb.source_article
            
            if source_article in doc_lookup:
                doc_idx, doc_emb = doc_lookup[source_article]
                
                # Compute similarity between chunk and document summary
                similarity = cosine_similarity(
                    [chunk_emb.embedding],
                    [doc_emb.embedding]
                )[0][0]
                
                connection = SimilarityConnection(
                    source_idx=chunk_idx,
                    target_idx=doc_idx,
                    similarity_score=float(similarity),
                    connection_type='chunk_to_doc',
                    source_id=chunk_emb.chunk_id,
                    target_id=doc_emb.doc_id,
                    granularity_type='cross_granularity'
                )
                all_connections.append(connection)
        
        return all_connections
    
    def _build_chunk_sparse_matrices(self, connections: List[SimilarityConnection], num_chunks: int) -> Dict[str, sp.csr_matrix]:
        """Build sparse matrices for chunk-to-chunk connections."""
        # Separate connections by type
        intra_connections = [c for c in connections if c.connection_type == 'chunk_to_chunk_intra']
        inter_connections = [c for c in connections if c.connection_type == 'chunk_to_chunk_inter']
        
        matrices = {}
        
        # Build intra-document matrix
        if intra_connections:
            matrices['chunk_to_chunk_intra'] = self._build_matrix_from_connections(
                intra_connections, num_chunks, num_chunks
            )
        
        # Build inter-document matrix
        if inter_connections:
            matrices['chunk_to_chunk_inter'] = self._build_matrix_from_connections(
                inter_connections, num_chunks, num_chunks
            )
        
        # Build combined chunk matrix
        all_chunk_connections = intra_connections + inter_connections
        if all_chunk_connections:
            matrices['chunk_to_chunk_combined'] = self._build_matrix_from_connections(
                all_chunk_connections, num_chunks, num_chunks
            )
        
        return matrices
    
    def _build_single_sparse_matrix(self, connections: List[SimilarityConnection], 
                                   size: int, matrix_name: str) -> Dict[str, sp.csr_matrix]:
        """Build a single sparse matrix from connections."""
        matrices = {}
        
        if connections:
            matrices[matrix_name] = self._build_matrix_from_connections(connections, size, size)
        
        return matrices
    
    def _build_sentence_sparse_matrices(self, connections: List[SimilarityConnection], num_sentences: int) -> Dict[str, sp.csr_matrix]:
        """Build sparse matrices for sentence-to-sentence connections."""
        # Separate connections by type
        semantic_connections = [c for c in connections if c.connection_type == 'sentence_to_sentence_semantic']
        sequential_connections = [c for c in connections if c.connection_type == 'sentence_to_sentence_sequential']
        
        matrices = {}
        
        # Build semantic matrix
        if semantic_connections:
            matrices['sentence_to_sentence_semantic'] = self._build_matrix_from_connections(
                semantic_connections, num_sentences, num_sentences
            )
        
        # Build sequential matrix
        if sequential_connections:
            matrices['sentence_to_sentence_sequential'] = self._build_matrix_from_connections(
                sequential_connections, num_sentences, num_sentences
            )
        
        # Build combined sentence matrix
        all_sentence_connections = semantic_connections + sequential_connections
        if all_sentence_connections:
            matrices['sentence_to_sentence_combined'] = self._build_matrix_from_connections(
                all_sentence_connections, num_sentences, num_sentences
            )
        
        return matrices
    
    def _build_cross_granularity_matrices(self, connections: List[SimilarityConnection], 
                                        granularity_embeddings: Dict[str, List[Any]]) -> Dict[str, sp.csr_matrix]:
        """Build sparse matrices for cross-granularity connections."""
        matrices = {}
        
        # Get sizes for different granularity types
        num_sentences = len(granularity_embeddings.get('sentences', []))
        num_chunks = len(granularity_embeddings.get('chunks', []))
        num_docs = len(granularity_embeddings.get('doc_summaries', []))
        
        # Separate connections by type
        sentence_to_chunk_connections = [c for c in connections if c.connection_type == 'sentence_to_chunk']
        chunk_to_doc_connections = [c for c in connections if c.connection_type == 'chunk_to_doc']
        
        # Build sentence-to-chunk matrix
        if sentence_to_chunk_connections and num_sentences > 0 and num_chunks > 0:
            matrices['sentence_to_chunk'] = self._build_matrix_from_connections(
                sentence_to_chunk_connections, num_sentences, num_chunks
            )
        
        # Build chunk-to-doc matrix
        if chunk_to_doc_connections and num_chunks > 0 and num_docs > 0:
            matrices['chunk_to_doc'] = self._build_matrix_from_connections(
                chunk_to_doc_connections, num_chunks, num_docs
            )
        
        return matrices
    
    def _build_matrix_from_connections(self, connections: List[SimilarityConnection], 
                                     num_rows: int, num_cols: int) -> sp.csr_matrix:
        """Build a sparse matrix from connections."""
        if not connections:
            return sp.csr_matrix((num_rows, num_cols))
        
        rows = [c.source_idx for c in connections]
        cols = [c.target_idx for c in connections]
        data = [c.similarity_score for c in connections]
        
        matrix = sp.coo_matrix((data, (rows, cols)), shape=(num_rows, num_cols))
        return matrix.tocsr()
    
    def _calculate_sparsity_statistics(self, matrices: Dict[str, sp.csr_matrix], 
                                     granularity_counts: Dict[str, int]) -> Dict[str, Any]:
        """Calculate sparsity statistics for all matrices."""
        sparsity_stats = {}
        
        for matrix_name, matrix in matrices.items():
            total_possible = matrix.shape[0] * matrix.shape[1]
            stored_connections = matrix.nnz
            sparsity_ratio = stored_connections / total_possible if total_possible > 0 else 0
            
            sparsity_stats[matrix_name] = {
                'shape': matrix.shape,
                'stored_connections': stored_connections,
                'total_possible': total_possible,
                'sparsity_ratio': sparsity_ratio,
                'density': matrix.nnz / (matrix.shape[0] * matrix.shape[1]) if matrix.shape[0] * matrix.shape[1] > 0 else 0
            }
        
        return sparsity_stats
    
    def _log_computation_results(self, metadata: MultiGranularitySimilarityMetadata, 
                               matrices: Dict[str, sp.csr_matrix]):
        """Log comprehensive computation results."""
        self.logger.info(f"✅ Multi-granularity similarity computation completed:")
        self.logger.info(f"   Total connections: {metadata.total_connections:,}")
        self.logger.info(f"   Memory usage: {metadata.memory_usage_mb:.1f} MB")
        self.logger.info(f"   Computation time: {metadata.computation_time:.2f}s")
        
        # Log matrix-specific statistics
        for matrix_name, stats in metadata.sparsity_statistics.items():
            self.logger.info(f"   {matrix_name}: {stats['stored_connections']:,} connections, sparsity={stats['sparsity_ratio']:.6f}")
    
    def _get_cache_path(self, model_name: str) -> Path:
        """Get cache path for a model's multi-granularity similarities."""
        safe_model_name = model_name.replace("/", "_").replace("-", "_")
        return self.similarities_dir / f"{safe_model_name}_multi_granularity_similarities.npz"
    
    def _generate_config_hash(self, model_name: str, granularity_embeddings: Dict[str, List[Any]]) -> str:
        """Generate hash of configuration and embeddings for cache validation."""
        config_str = json.dumps({
            'model_name': model_name,
            'granularity_config': self.granularity_config,
            'similarity_config': self.similarity_config,
            'granularity_counts': {gt: len(embeddings) for gt, embeddings in granularity_embeddings.items()},
            'first_chunk_id': granularity_embeddings.get('chunks', [{}])[0].chunk_id if granularity_embeddings.get('chunks') else "",
            'first_sentence_id': granularity_embeddings.get('sentences', [{}])[0].sentence_id if granularity_embeddings.get('sentences') else ""
        }, sort_keys=True)
        
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_path: Path, expected_hash: str) -> bool:
        """Check if cached multi-granularity similarities are valid."""
        metadata_path = cache_path.with_suffix('.json')
        
        if not cache_path.exists() or not metadata_path.exists():
            return False
        
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            metadata = cache_data.get('metadata', {})
            cached_hash = metadata.get('config_hash', '')
            
            # Check if cache has the new multi-granularity format
            if 'matrix_names' not in cache_data:
                self.logger.info("Cache uses old format, will regenerate")
                return False
            
            return cached_hash == expected_hash
            
        except Exception as e:
            self.logger.warning(f"Failed to validate multi-granularity similarity cache: {e}")
            return False
    
    def _cache_similarities(self, cache_path: Path, similarity_data: Dict[str, Any]):
        """Cache multi-granularity similarity data to disk."""
        try:
            # Save sparse matrices as .npz with proper sparse matrix components
            matrices = similarity_data['matrices']
            matrix_data = {}
            
            for name, matrix in matrices.items():
                # Convert to CSR format if not already
                csr_matrix = matrix.tocsr()
                # Save the components needed to reconstruct the sparse matrix
                matrix_data[f"{name}_data"] = csr_matrix.data
                matrix_data[f"{name}_indices"] = csr_matrix.indices
                matrix_data[f"{name}_indptr"] = csr_matrix.indptr
                matrix_data[f"{name}_shape"] = np.array(csr_matrix.shape)
            
            np.savez_compressed(cache_path, **matrix_data)
            
            # Save metadata and other data as JSON
            metadata_path = cache_path.with_suffix('.json')
            cache_data = {
                'metadata': asdict(similarity_data['metadata']),
                'index_maps': similarity_data['index_maps'],
                'connection_summary': {
                    'total_connections': len(similarity_data['connections']),
                    'by_granularity_type': {
                        granularity_type: len([c for c in similarity_data['connections'] if c.granularity_type == granularity_type])
                        for granularity_type in set(c.granularity_type for c in similarity_data['connections'])
                    }
                },
                'matrix_names': list(matrices.keys())  # Store matrix names for loading
            }
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"💾 Cached multi-granularity similarities to {cache_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to cache multi-granularity similarities: {e}")
            raise
    
    def _load_cached_similarities(self, cache_path: Path) -> Dict[str, Any]:
        """Load cached multi-granularity similarity data from disk."""
        try:
            # Load metadata first to get matrix names
            metadata_path = cache_path.with_suffix('.json')
            with open(metadata_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Check if this is the new multi-granularity format
            if 'matrix_names' not in cache_data:
                self.logger.warning("Cache file uses old format, cannot load")
                raise ValueError("Cache file uses old format")
            
            # Load sparse matrix components
            matrices_data = np.load(cache_path, allow_pickle=True)
            matrices = {}
            
            # Reconstruct sparse matrices from saved components
            matrix_names = cache_data['matrix_names']
            for name in matrix_names:
                try:
                    # Load the components
                    data = matrices_data[f"{name}_data"]
                    indices = matrices_data[f"{name}_indices"]
                    indptr = matrices_data[f"{name}_indptr"]
                    shape = tuple(matrices_data[f"{name}_shape"])
                    
                    # Reconstruct the CSR matrix
                    matrices[name] = sp.csr_matrix((data, indices, indptr), shape=shape)
                    self.logger.debug(f"Loaded matrix '{name}' with shape {shape}")
                except KeyError as e:
                    self.logger.warning(f"Failed to load matrix '{name}': missing component {e}")
                    continue
                except Exception as e:
                    self.logger.warning(f"Failed to reconstruct matrix '{name}': {e}")
                    continue
            
            if not matrices:
                raise ValueError("No matrices could be loaded from cache")
            
            metadata = MultiGranularitySimilarityMetadata(**cache_data['metadata'])
            
            self.logger.info(f"📂 Successfully loaded {len(matrices)} multi-granularity similarity matrices from cache")
            
            return {
                'metadata': metadata,
                'matrices': matrices,
                'index_maps': cache_data['index_maps'],
                'connection_summary': cache_data['connection_summary']
            }
            
        except Exception as e:
            self.logger.error(f"Failed to load cached multi-granularity similarities: {e}")
            raise
    
    def get_similarity_statistics(self, similarity_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about computed multi-granularity similarities."""
        stats = {}
        
        for model_name, data in similarity_data.items():
            metadata = data['metadata']
            matrices = data['matrices']
            
            model_stats = {
                'granularity_counts': metadata.granularity_counts,
                'computation_time': metadata.computation_time,
                'memory_usage_mb': metadata.memory_usage_mb,
                'total_connections': metadata.total_connections,
                'sparsity_statistics': metadata.sparsity_statistics
            }
            
            # Add connection breakdown by granularity type
            if 'connection_summary' in data:
                model_stats['connections_by_granularity'] = data['connection_summary']['by_granularity_type']
            
            stats[model_name] = model_stats
        
        return stats


# Backwards compatibility: Alias the new class as SimilarityEngine
SimilarityEngine = MultiGranularitySimilarityEngine