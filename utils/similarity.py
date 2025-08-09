#!/usr/bin/env python3
"""
Similarity Matrix Engine
=======================

Handles similarity matrix computation and storage for the semantic RAG pipeline.
Computes sparse similarity matrices with configurable top-k intra-document and 
top-x inter-document connections for memory-efficient semantic traversal.
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

from models import ChunkEmbedding


@dataclass
class SimilarityMetadata:
    """Metadata for cached similarity matrices."""
    created_at: str
    total_chunks: int
    embedding_models: List[str]
    similarity_metric: str
    intra_document_top_k: int
    inter_document_top_x: int
    config_hash: str
    computation_time: float
    memory_usage_mb: float
    intra_doc_connections: int
    inter_doc_connections: int
    sparsity_ratio: float


@dataclass
class SimilarityConnection:
    """Represents a similarity connection between chunks."""
    source_chunk_idx: int
    target_chunk_idx: int
    similarity_score: float
    connection_type: str  # 'intra_document' or 'inter_document'
    source_chunk_id: str
    target_chunk_id: str


class SimilarityEngine:
    """Engine for computing and managing similarity matrices."""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """Initialize the similarity engine."""
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.similarity_config = config['similarities']
        self.similarities_dir = Path(config['directories']['embeddings']) / "similarities"
        
        # Create similarities directory
        self.similarities_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate similarity configuration."""
        sim_config = self.similarity_config
        
        # Validate metric
        valid_metrics = ['cosine', 'dot_product', 'euclidean']
        if sim_config['similarity_metric'] not in valid_metrics:
            raise ValueError(f"Invalid similarity metric. Must be one of: {valid_metrics}")
        
        # Validate top_k and top_x
        if sim_config['intra_document']['enabled']:
            top_k = sim_config['intra_document']['top_k']
            if top_k <= 0:
                raise ValueError("intra_document top_k must be positive")
        
        if sim_config['inter_document']['enabled']:
            top_x = sim_config['inter_document']['top_x']
            if top_x <= 0:
                raise ValueError("inter_document top_x must be positive")
        
        # Validate batch size
        if sim_config['batch_size'] <= 0:
            raise ValueError("batch_size must be positive")
    
    def compute_similarity_matrices(self, model_embeddings: Dict[str, List[ChunkEmbedding]], 
                                  force_recompute: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Compute similarity matrices for all embedding models.
        
        Args:
            model_embeddings: Dictionary mapping model names to chunk embeddings
            force_recompute: Whether to force recomputation even if cached
            
        Returns:
            Dictionary mapping model names to similarity data
        """
        similarity_results = {}
        
        for model_name, embeddings in model_embeddings.items():
            self.logger.info(f"Computing similarities for model: {model_name}")
            
            # Generate config hash for cache validation
            config_hash = self._generate_config_hash(model_name, embeddings)
            
            # Check cache
            cache_path = self._get_cache_path(model_name)
            if not force_recompute and self._is_cache_valid(cache_path, config_hash):
                self.logger.info(f"Loading cached similarities for {model_name}")
                similarity_results[model_name] = self._load_cached_similarities(cache_path)
            else:
                self.logger.info(f"Computing fresh similarities for {model_name}")
                similarity_results[model_name] = self._compute_fresh_similarities(
                    model_name, embeddings, config_hash, cache_path
                )
        
        return similarity_results
    
    def _compute_fresh_similarities(self, model_name: str, embeddings: List[ChunkEmbedding],
                                  config_hash: str, cache_path: Path) -> Dict[str, Any]:
        """Compute fresh similarity matrices for a model."""
        start_time = time.time()
        
        self.logger.info(f"Computing similarities for {len(embeddings)} chunks")
        
        # Group embeddings by document for intra-document similarities
        doc_groups = self._group_embeddings_by_document(embeddings)
        self.logger.info(f"Grouped into {len(doc_groups)} documents")
        
        # Initialize connection storage
        all_connections = []
        intra_doc_count = 0
        inter_doc_count = 0
        
        # Step 1: Compute intra-document similarities
        if self.similarity_config['intra_document']['enabled']:
            self.logger.info("🔗 Computing intra-document similarities...")
            intra_connections, intra_count = self._compute_intra_document_similarities(
                doc_groups, embeddings
            )
            all_connections.extend(intra_connections)
            intra_doc_count = intra_count
        
        # Step 2: Compute inter-document similarities
        if self.similarity_config['inter_document']['enabled']:
            self.logger.info("🌉 Computing inter-document similarities...")
            inter_connections, inter_count = self._compute_inter_document_similarities(
                doc_groups, embeddings
            )
            all_connections.extend(inter_connections)
            inter_doc_count = inter_count
        
        # Step 3: Build sparse matrices
        self.logger.info("🏗️  Building sparse similarity matrices...")
        similarity_matrices = self._build_sparse_matrices(all_connections, len(embeddings))
        
        # Calculate statistics
        total_possible = len(embeddings) * (len(embeddings) - 1)  # Exclude self-similarities
        total_stored = len(all_connections)
        sparsity_ratio = total_stored / total_possible if total_possible > 0 else 0
        
        computation_time = time.time() - start_time
        
        # Estimate memory usage (rough approximation)
        memory_usage_mb = (total_stored * 12) / (1024 * 1024)  # 12 bytes per connection (int + int + float)
        
        # Create metadata
        metadata = SimilarityMetadata(
            created_at=datetime.now().isoformat(),
            total_chunks=len(embeddings),
            embedding_models=[model_name],
            similarity_metric=self.similarity_config['similarity_metric'],
            intra_document_top_k=self.similarity_config['intra_document']['top_k'],
            inter_document_top_x=self.similarity_config['inter_document']['top_x'],
            config_hash=config_hash,
            computation_time=computation_time,
            memory_usage_mb=memory_usage_mb,
            intra_doc_connections=intra_doc_count,
            inter_doc_connections=inter_doc_count,
            sparsity_ratio=sparsity_ratio
        )
        
        # Package results
        similarity_data = {
            'metadata': metadata,
            'matrices': similarity_matrices,
            'connections': all_connections,
            'chunk_index_map': {emb.chunk_id: idx for idx, emb in enumerate(embeddings)}
        }
        
        # Cache results
        self._cache_similarities(cache_path, similarity_data)
        
        # Log statistics
        self.logger.info(f"✅ Similarity computation completed:")
        self.logger.info(f"   Total connections: {total_stored:,}")
        self.logger.info(f"   Intra-document: {intra_doc_count:,}")
        self.logger.info(f"   Inter-document: {inter_doc_count:,}")
        self.logger.info(f"   Sparsity ratio: {sparsity_ratio:.6f}")
        self.logger.info(f"   Memory usage: {memory_usage_mb:.1f} MB")
        self.logger.info(f"   Computation time: {computation_time:.2f}s")
        
        return similarity_data
    
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
                                           all_embeddings: List[ChunkEmbedding]) -> Tuple[List[SimilarityConnection], int]:
        """Compute top-k similarities within each document."""
        top_k = self.similarity_config['intra_document']['top_k']
        metric = self.similarity_config['similarity_metric']
        
        all_connections = []
        total_connections = 0
        
        for doc_name, doc_embeddings in tqdm(doc_groups.items(), desc="Intra-document similarities"):
            if len(doc_embeddings) < 2:
                continue  # Skip documents with only one chunk
            
            # Extract embeddings for this document
            indices = [idx for idx, _ in doc_embeddings]
            embeddings_matrix = np.array([emb.embedding for _, emb in doc_embeddings])
            
            # Compute similarities within document
            if metric == 'cosine':
                sim_matrix = cosine_similarity(embeddings_matrix)
            elif metric == 'dot_product':
                sim_matrix = np.dot(embeddings_matrix, embeddings_matrix.T)
            elif metric == 'euclidean':
                # Convert to similarity (higher = more similar)
                from sklearn.metrics.pairwise import euclidean_distances
                dist_matrix = euclidean_distances(embeddings_matrix)
                sim_matrix = 1 / (1 + dist_matrix)  # Convert distance to similarity
            
            # For each chunk in document, find top-k most similar (excluding self)
            for i, (source_idx, source_emb) in enumerate(doc_embeddings):
                # Get similarities for this chunk (excluding self-similarity)
                similarities = sim_matrix[i].copy()
                similarities[i] = -np.inf  # Exclude self
                
                # Get top-k indices
                top_indices = np.argsort(similarities)[-top_k:][::-1]  # Descending order
                
                # Create connections
                for j in top_indices:
                    target_idx, target_emb = doc_embeddings[j]
                    similarity_score = float(similarities[j])
                    
                    if similarity_score > -np.inf:  # Valid similarity
                        connection = SimilarityConnection(
                            source_chunk_idx=source_idx,
                            target_chunk_idx=target_idx,
                            similarity_score=similarity_score,
                            connection_type='intra_document',
                            source_chunk_id=source_emb.chunk_id,
                            target_chunk_id=target_emb.chunk_id
                        )
                        all_connections.append(connection)
                        total_connections += 1
        
        return all_connections, total_connections
    
    def _compute_inter_document_similarities(self, doc_groups: Dict[str, List[Tuple[int, ChunkEmbedding]]], 
                                           all_embeddings: List[ChunkEmbedding]) -> Tuple[List[SimilarityConnection], int]:
        """Compute top-x similarities across different documents."""
        top_x = self.similarity_config['inter_document']['top_x']
        metric = self.similarity_config['similarity_metric']
        batch_size = self.similarity_config['batch_size']
        
        all_connections = []
        total_connections = 0
        
        # Create mapping from chunk index to document
        chunk_to_doc = {}
        for doc_name, doc_embeddings in doc_groups.items():
            for idx, _ in doc_embeddings:
                chunk_to_doc[idx] = doc_name
        
        # Extract all embeddings for cross-document comparison
        all_embeddings_matrix = np.array([emb.embedding for emb in all_embeddings])
        
        self.logger.info(f"Computing inter-document similarities with batch_size={batch_size}")
        
        # Process in batches to manage memory
        num_chunks = len(all_embeddings)
        for start_idx in tqdm(range(0, num_chunks, batch_size), desc="Inter-document similarities"):
            end_idx = min(start_idx + batch_size, num_chunks)
            batch_embeddings = all_embeddings_matrix[start_idx:end_idx]
            
            # Compute similarities between batch and all embeddings
            if metric == 'cosine':
                sim_matrix = cosine_similarity(batch_embeddings, all_embeddings_matrix)
            elif metric == 'dot_product':
                sim_matrix = np.dot(batch_embeddings, all_embeddings_matrix.T)
            elif metric == 'euclidean':
                from sklearn.metrics.pairwise import euclidean_distances
                dist_matrix = euclidean_distances(batch_embeddings, all_embeddings_matrix)
                sim_matrix = 1 / (1 + dist_matrix)
            
            # For each chunk in batch, find top-x from different documents
            for batch_i, global_i in enumerate(range(start_idx, end_idx)):
                source_doc = chunk_to_doc[global_i]
                similarities = sim_matrix[batch_i].copy()
                
                # Mask out chunks from the same document and self
                for j in range(num_chunks):
                    if chunk_to_doc[j] == source_doc:
                        similarities[j] = -np.inf
                
                # Get top-x indices from different documents
                valid_similarities = similarities[similarities > -np.inf]
                if len(valid_similarities) > 0:
                    top_indices = np.argsort(similarities)[-top_x:][::-1]  # Descending order
                    
                    # Create connections
                    for target_idx in top_indices:
                        similarity_score = float(similarities[target_idx])
                        if similarity_score > -np.inf:
                            connection = SimilarityConnection(
                                source_chunk_idx=global_i,
                                target_chunk_idx=target_idx,
                                similarity_score=similarity_score,
                                connection_type='inter_document',
                                source_chunk_id=all_embeddings[global_i].chunk_id,
                                target_chunk_id=all_embeddings[target_idx].chunk_id
                            )
                            all_connections.append(connection)
                            total_connections += 1
        
        return all_connections, total_connections
    
    def _build_sparse_matrices(self, connections: List[SimilarityConnection], num_chunks: int) -> Dict[str, sp.csr_matrix]:
        """Build sparse matrices from connections."""
        # Separate connections by type
        intra_connections = [c for c in connections if c.connection_type == 'intra_document']
        inter_connections = [c for c in connections if c.connection_type == 'inter_document']
        
        matrices = {}
        
        # Build intra-document matrix
        if intra_connections:
            rows = [c.source_chunk_idx for c in intra_connections]
            cols = [c.target_chunk_idx for c in intra_connections]
            data = [c.similarity_score for c in intra_connections]
            
            intra_matrix = sp.coo_matrix((data, (rows, cols)), shape=(num_chunks, num_chunks))
            matrices['intra_document'] = intra_matrix.tocsr()
        
        # Build inter-document matrix
        if inter_connections:
            rows = [c.source_chunk_idx for c in inter_connections]
            cols = [c.target_chunk_idx for c in inter_connections]
            data = [c.similarity_score for c in inter_connections]
            
            inter_matrix = sp.coo_matrix((data, (rows, cols)), shape=(num_chunks, num_chunks))
            matrices['inter_document'] = inter_matrix.tocsr()
        
        # Build combined matrix
        all_rows = [c.source_chunk_idx for c in connections]
        all_cols = [c.target_chunk_idx for c in connections]
        all_data = [c.similarity_score for c in connections]
        
        combined_matrix = sp.coo_matrix((all_data, (all_rows, all_cols)), shape=(num_chunks, num_chunks))
        matrices['combined'] = combined_matrix.tocsr()
        
        return matrices
    
    def _get_cache_path(self, model_name: str) -> Path:
        """Get cache path for a model's similarities."""
        safe_model_name = model_name.replace("/", "_").replace("-", "_")
        return self.similarities_dir / f"{safe_model_name}_similarities.npz"
    
    def _generate_config_hash(self, model_name: str, embeddings: List[ChunkEmbedding]) -> str:
        """Generate hash of configuration and embeddings for cache validation."""
        config_str = json.dumps({
            'model_name': model_name,
            'similarity_config': self.similarity_config,
            'embedding_count': len(embeddings),
            'first_chunk_id': embeddings[0].chunk_id if embeddings else "",
            'last_chunk_id': embeddings[-1].chunk_id if embeddings else ""
        }, sort_keys=True)
        
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_path: Path, expected_hash: str) -> bool:
        """Check if cached similarities are valid."""
        metadata_path = cache_path.with_suffix('.json')
        
        if not cache_path.exists() or not metadata_path.exists():
            return False
        
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            metadata = cache_data.get('metadata', {})
            cached_hash = metadata.get('config_hash', '')
            
            # Check if cache has the new format (matrix_names)
            if 'matrix_names' not in cache_data:
                self.logger.info("Cache uses old format, will regenerate")
                return False
            
            return cached_hash == expected_hash
            
        except Exception as e:
            self.logger.warning(f"Failed to validate similarity cache: {e}")
            return False
    
    def _cache_similarities(self, cache_path: Path, similarity_data: Dict[str, Any]):
        """Cache similarity data to disk."""
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
                'chunk_index_map': similarity_data['chunk_index_map'],
                'connection_summary': {
                    'total_connections': len(similarity_data['connections']),
                    'intra_document': len([c for c in similarity_data['connections'] if c.connection_type == 'intra_document']),
                    'inter_document': len([c for c in similarity_data['connections'] if c.connection_type == 'inter_document'])
                },
                'matrix_names': list(matrices.keys())  # Store matrix names for loading
            }
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Cached similarities to {cache_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to cache similarities: {e}")
            raise
    
    def _load_cached_similarities(self, cache_path: Path) -> Dict[str, Any]:
        """Load cached similarity data from disk."""
        try:
            # Load metadata first to get matrix names
            metadata_path = cache_path.with_suffix('.json')
            with open(metadata_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Check if this is the new format
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
            
            metadata = SimilarityMetadata(**cache_data['metadata'])
            
            self.logger.info(f"Successfully loaded {len(matrices)} similarity matrices from cache")
            
            return {
                'metadata': metadata,
                'matrices': matrices,
                'chunk_index_map': cache_data['chunk_index_map'],
                'connection_summary': cache_data['connection_summary']
            }
            
        except Exception as e:
            self.logger.error(f"Failed to load cached similarities: {e}")
            raise
    
    def get_similarity_statistics(self, similarity_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about computed similarities."""
        stats = {}
        
        for model_name, data in similarity_data.items():
            metadata = data['metadata']
            matrices = data['matrices']
            
            model_stats = {
                'total_chunks': metadata.total_chunks,
                'computation_time': metadata.computation_time,
                'memory_usage_mb': metadata.memory_usage_mb,
                'sparsity_ratio': metadata.sparsity_ratio,
                'connections': {
                    'intra_document': metadata.intra_doc_connections,
                    'inter_document': metadata.inter_doc_connections,
                    'total': metadata.intra_doc_connections + metadata.inter_doc_connections
                }
            }
            
            # Add matrix-specific statistics
            if 'combined' in matrices:
                combined_matrix = matrices['combined']
                model_stats['matrix_stats'] = {
                    'shape': combined_matrix.shape,
                    'nnz': combined_matrix.nnz,
                    'density': combined_matrix.nnz / (combined_matrix.shape[0] * combined_matrix.shape[1])
                }
            
            stats[model_name] = model_stats
        
        return stats
