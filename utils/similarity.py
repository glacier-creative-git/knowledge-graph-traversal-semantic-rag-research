#!/usr/bin/env python3
"""
Simplified Similarity Engine
===========================

Focused chunk-to-chunk similarity computation only.
No multi-granularity complexity - just clean, fast chunk navigation.
"""

import json
import hashlib
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
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
    chunk_count: int
    embedding_models: List[str]
    similarity_metric: str
    config_hash: str
    computation_time: float
    memory_usage_mb: float
    total_connections: int
    intra_doc_connections: int
    inter_doc_connections: int


@dataclass
class ChunkConnection:
    """Represents a connection between two chunks."""
    source_chunk_id: str
    target_chunk_id: str
    similarity_score: float
    connection_type: str  # 'intra_document' or 'inter_document'
    source_document: str
    target_document: str


class SimilarityEngine:
    """Simplified engine for computing chunk-to-chunk similarities only."""

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """Initialize the similarity engine."""
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.similarity_config = config['similarities']
        self.chunk_config = self.similarity_config['granularity_types']['chunk_to_chunk']
        self.similarities_dir = Path(config['directories']['embeddings']) / "similarities"

        # Create similarities directory
        self.similarities_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("🔗 Initialized simplified chunk-to-chunk similarity engine")

    def compute_similarity_matrices(self, embeddings: Dict[str, Dict[str, List[Any]]],
                                    force_recompute: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Compute chunk-to-chunk similarity matrices for all embedding models.

        Args:
            embeddings: Multi-granularity embeddings (we only use chunks)
            force_recompute: Whether to force recomputation even if cached

        Returns:
            Dictionary mapping model names to similarity data
        """
        similarity_results = {}

        for model_name, granularity_embeddings in embeddings.items():
            # Only process chunk embeddings
            chunk_embeddings = granularity_embeddings.get('chunks', [])
            if not chunk_embeddings:
                self.logger.warning(f"No chunk embeddings found for model {model_name}")
                continue

            self.logger.info(f"🎯 Computing chunk similarities for model: {model_name}")

            # Generate config hash for cache validation
            config_hash = self._generate_config_hash(model_name, chunk_embeddings)

            # Check cache
            cache_path = self._get_cache_path(model_name)
            if not force_recompute and self._is_cache_valid(cache_path, config_hash):
                self.logger.info(f"📂 Loading cached similarities for {model_name}")
                similarity_results[model_name] = self._load_cached_similarities(cache_path)
            else:
                self.logger.info(f"⚡ Computing fresh similarities for {model_name}")
                similarity_results[model_name] = self._compute_fresh_similarities(
                    model_name, chunk_embeddings, config_hash, cache_path
                )

        return similarity_results

    def _compute_fresh_similarities(self, model_name: str, chunk_embeddings: List[ChunkEmbedding],
                                    config_hash: str, cache_path: Path) -> Dict[str, Any]:
        """Compute fresh chunk-to-chunk similarities for a model."""
        start_time = time.time()

        self.logger.info(f"Computing similarities for {len(chunk_embeddings)} chunks")

        # Group chunks by document for intra-document processing
        doc_groups = self._group_chunks_by_document(chunk_embeddings)

        # Initialize connection storage
        all_connections = []

        # Step 1: Compute intra-document similarities (within same document)
        if self.chunk_config['intra_document']['enabled']:
            intra_connections = self._compute_intra_document_similarities(
                doc_groups, chunk_embeddings
            )
            all_connections.extend(intra_connections)
            self.logger.info(f"✅ Built {len(intra_connections)} intra-document connections")

        # Step 2: Compute inter-document similarities (across different documents)
        if self.chunk_config['inter_document']['enabled']:
            inter_connections = self._compute_inter_document_similarities(
                doc_groups, chunk_embeddings
            )
            all_connections.extend(inter_connections)
            self.logger.info(f"✅ Built {len(inter_connections)} inter-document connections")

        # Build sparse matrices for storage
        matrices = self._build_sparse_matrices(all_connections, len(chunk_embeddings), chunk_embeddings)

        computation_time = time.time() - start_time

        # Create metadata
        metadata = SimilarityMetadata(
            created_at=datetime.now().isoformat(),
            chunk_count=len(chunk_embeddings),
            embedding_models=[model_name],
            similarity_metric=self.similarity_config['similarity_metric'],
            config_hash=config_hash,
            computation_time=computation_time,
            memory_usage_mb=(len(all_connections) * 32) / (1024 * 1024),  # Rough estimate
            total_connections=len(all_connections),
            intra_doc_connections=len([c for c in all_connections if c.connection_type == 'intra_document']),
            inter_doc_connections=len([c for c in all_connections if c.connection_type == 'inter_document'])
        )

        # Package results
        similarity_data = {
            'metadata': metadata,
            'matrices': matrices,
            'connections': all_connections,
            'chunk_index_map': {emb.chunk_id: idx for idx, emb in enumerate(chunk_embeddings)}
        }

        # Cache results
        self._cache_similarities(cache_path, similarity_data)

        # Log results
        self.logger.info(f"✅ Similarity computation completed in {computation_time:.2f}s")
        self.logger.info(f"   Total connections: {len(all_connections):,}")
        self.logger.info(f"   Intra-document: {metadata.intra_doc_connections:,}")
        self.logger.info(f"   Inter-document: {metadata.inter_doc_connections:,}")

        return similarity_data

    def _group_chunks_by_document(self, chunk_embeddings: List[ChunkEmbedding]) -> Dict[
        str, List[Tuple[int, ChunkEmbedding]]]:
        """Group chunk embeddings by source document."""
        doc_groups = {}
        for idx, chunk_emb in enumerate(chunk_embeddings):
            doc_name = chunk_emb.source_article
            if doc_name not in doc_groups:
                doc_groups[doc_name] = []
            doc_groups[doc_name].append((idx, chunk_emb))
        return doc_groups

    def _compute_intra_document_similarities(self, doc_groups: Dict[str, List[Tuple[int, ChunkEmbedding]]],
                                             all_embeddings: List[ChunkEmbedding]) -> List[ChunkConnection]:
        """Compute similarities within each document."""
        connections = []
        config = self.chunk_config['intra_document']
        top_k = config['top_k']
        min_threshold = config.get('min_threshold', 0.0)

        for doc_name, doc_chunks in tqdm(doc_groups.items(), desc="Intra-document similarities"):
            if len(doc_chunks) < 2:
                continue  # Skip documents with only one chunk

            # Extract embeddings for this document
            embeddings_matrix = np.array([chunk_emb.embedding for _, chunk_emb in doc_chunks])

            # Compute similarity matrix for this document
            sim_matrix = cosine_similarity(embeddings_matrix)

            # For each chunk, find top-k most similar chunks in same document
            for i, (source_idx, source_chunk) in enumerate(doc_chunks):
                similarities = sim_matrix[i].copy()
                similarities[i] = -1.0  # Exclude self

                # Apply minimum threshold
                similarities[similarities < min_threshold] = -1.0

                # Get top-k indices
                valid_indices = np.where(similarities > -1.0)[0]
                if len(valid_indices) > 0:
                    top_indices = np.argsort(similarities)[-top_k:][::-1]  # Descending order

                    for j in top_indices:
                        if similarities[j] > -1.0:
                            target_idx, target_chunk = doc_chunks[j]

                            connection = ChunkConnection(
                                source_chunk_id=source_chunk.chunk_id,
                                target_chunk_id=target_chunk.chunk_id,
                                similarity_score=float(similarities[j]),
                                connection_type='intra_document',
                                source_document=doc_name,
                                target_document=doc_name
                            )
                            connections.append(connection)

        return connections

    def _compute_inter_document_similarities(self, doc_groups: Dict[str, List[Tuple[int, ChunkEmbedding]]],
                                             all_embeddings: List[ChunkEmbedding]) -> List[ChunkConnection]:
        """Compute similarities across different documents."""
        connections = []
        config = self.chunk_config['inter_document']
        top_x = config['top_x']
        min_threshold = config.get('min_threshold', 0.0)
        batch_size = self.similarity_config.get('batch_size', 1000)

        # Create mapping from chunk index to document
        chunk_to_doc = {}
        for doc_name, doc_chunks in doc_groups.items():
            for idx, _ in doc_chunks:
                chunk_to_doc[idx] = doc_name

        # Extract all embeddings for cross-document comparison
        all_embeddings_matrix = np.array([emb.embedding for emb in all_embeddings])

        # Process in batches for memory efficiency
        num_chunks = len(all_embeddings)
        for start_idx in tqdm(range(0, num_chunks, batch_size), desc="Inter-document similarities"):
            end_idx = min(start_idx + batch_size, num_chunks)
            batch_embeddings = all_embeddings_matrix[start_idx:end_idx]

            # Compute similarities between batch and all embeddings
            sim_matrix = cosine_similarity(batch_embeddings, all_embeddings_matrix)

            # For each chunk in batch, find top-x from different documents
            for batch_i, global_i in enumerate(range(start_idx, end_idx)):
                source_doc = chunk_to_doc[global_i]
                source_chunk = all_embeddings[global_i]
                similarities = sim_matrix[batch_i].copy()

                # Mask out chunks from the same document
                for j in range(num_chunks):
                    if chunk_to_doc[j] == source_doc:
                        similarities[j] = -1.0

                # Apply minimum threshold
                similarities[similarities < min_threshold] = -1.0

                # Get top-x indices from different documents
                valid_indices = np.where(similarities > -1.0)[0]
                if len(valid_indices) > 0:
                    top_indices = np.argsort(similarities)[-top_x:][::-1]  # Descending order

                    for target_idx in top_indices:
                        if similarities[target_idx] > -1.0:
                            target_chunk = all_embeddings[target_idx]
                            target_doc = chunk_to_doc[target_idx]

                            connection = ChunkConnection(
                                source_chunk_id=source_chunk.chunk_id,
                                target_chunk_id=target_chunk.chunk_id,
                                similarity_score=float(similarities[target_idx]),
                                connection_type='inter_document',
                                source_document=source_doc,
                                target_document=target_doc
                            )
                            connections.append(connection)

        return connections

    def _build_sparse_matrices(self, connections: List[ChunkConnection],
                               num_chunks: int, chunk_embeddings: List[ChunkEmbedding]) -> Dict[str, sp.csr_matrix]:
        """Build sparse matrices from connections."""
        matrices = {}

        # Create chunk ID to index mapping
        chunk_id_to_idx = {emb.chunk_id: idx for idx, emb in enumerate(chunk_embeddings)}

        # Separate connections by type
        intra_connections = [c for c in connections if c.connection_type == 'intra_document']
        inter_connections = [c for c in connections if c.connection_type == 'inter_document']

        # Build intra-document matrix
        if intra_connections:
            matrices['intra_document'] = self._build_matrix_from_connections(
                intra_connections, num_chunks, chunk_id_to_idx
            )

        # Build inter-document matrix
        if inter_connections:
            matrices['inter_document'] = self._build_matrix_from_connections(
                inter_connections, num_chunks, chunk_id_to_idx
            )

        # Build combined matrix
        all_connections = intra_connections + inter_connections
        if all_connections:
            matrices['combined'] = self._build_matrix_from_connections(
                all_connections, num_chunks, chunk_id_to_idx
            )

        return matrices

    def _build_matrix_from_connections(self, connections: List[ChunkConnection],
                                       num_chunks: int, chunk_id_to_idx: Dict[str, int]) -> sp.csr_matrix:
        """Build a sparse matrix from connections."""
        if not connections:
            return sp.csr_matrix((num_chunks, num_chunks))

        rows = []
        cols = []
        data = []

        for conn in connections:
            if conn.source_chunk_id in chunk_id_to_idx and conn.target_chunk_id in chunk_id_to_idx:
                source_idx = chunk_id_to_idx[conn.source_chunk_id]
                target_idx = chunk_id_to_idx[conn.target_chunk_id]

                rows.append(source_idx)
                cols.append(target_idx)
                data.append(conn.similarity_score)

        matrix = sp.coo_matrix((data, (rows, cols)), shape=(num_chunks, num_chunks))
        return matrix.tocsr()

    def _generate_config_hash(self, model_name: str, chunk_embeddings: List[ChunkEmbedding]) -> str:
        """Generate hash of configuration and data for cache validation."""
        config_str = json.dumps({
            'model_name': model_name,
            'chunk_config': self.chunk_config,
            'similarity_metric': self.similarity_config['similarity_metric'],
            'chunk_count': len(chunk_embeddings),
            'first_chunk_id': chunk_embeddings[0].chunk_id if chunk_embeddings else ""
        }, sort_keys=True)

        return hashlib.md5(config_str.encode()).hexdigest()

    def _get_cache_path(self, model_name: str) -> Path:
        """Get cache path for a model's similarities."""
        safe_model_name = model_name.replace("/", "_").replace("-", "_")
        return self.similarities_dir / f"{safe_model_name}_chunk_similarities.npz"

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

            return cached_hash == expected_hash

        except Exception as e:
            self.logger.warning(f"Failed to validate cache: {e}")
            return False

    def _cache_similarities(self, cache_path: Path, similarity_data: Dict[str, Any]):
        """Cache similarity data to disk."""
        try:
            # Save sparse matrices as .npz
            matrices = similarity_data['matrices']
            matrix_data = {}

            for name, matrix in matrices.items():
                csr_matrix = matrix.tocsr()
                matrix_data[f"{name}_data"] = csr_matrix.data
                matrix_data[f"{name}_indices"] = csr_matrix.indices
                matrix_data[f"{name}_indptr"] = csr_matrix.indptr
                matrix_data[f"{name}_shape"] = np.array(csr_matrix.shape)

            np.savez_compressed(cache_path, **matrix_data)

            # Save metadata and connections as JSON
            metadata_path = cache_path.with_suffix('.json')
            cache_data = {
                'metadata': asdict(similarity_data['metadata']),
                'chunk_index_map': similarity_data['chunk_index_map'],
                'connections': [
                    {
                        'source_chunk_id': conn.source_chunk_id,
                        'target_chunk_id': conn.target_chunk_id,
                        'similarity_score': float(conn.similarity_score),
                        'connection_type': conn.connection_type,
                        'source_document': conn.source_document,
                        'target_document': conn.target_document
                    } for conn in similarity_data['connections']
                ],
                'matrix_names': list(matrices.keys())
            }

            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"💾 Cached similarities to {cache_path}")

        except Exception as e:
            self.logger.error(f"Failed to cache similarities: {e}")
            raise

    def _load_cached_similarities(self, cache_path: Path) -> Dict[str, Any]:
        """Load cached similarities from disk."""
        try:
            # Load metadata first
            metadata_path = cache_path.with_suffix('.json')
            with open(metadata_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)

            # Load sparse matrices
            matrices_data = np.load(cache_path, allow_pickle=True)
            matrices = {}

            matrix_names = cache_data['matrix_names']
            for name in matrix_names:
                try:
                    data = matrices_data[f"{name}_data"]
                    indices = matrices_data[f"{name}_indices"]
                    indptr = matrices_data[f"{name}_indptr"]
                    shape = tuple(matrices_data[f"{name}_shape"])

                    matrices[name] = sp.csr_matrix((data, indices, indptr), shape=shape)
                except KeyError as e:
                    self.logger.warning(f"Failed to load matrix '{name}': missing component {e}")
                    continue

            # Reconstruct connections
            connections = []
            for conn_data in cache_data['connections']:
                connection = ChunkConnection(
                    source_chunk_id=conn_data['source_chunk_id'],
                    target_chunk_id=conn_data['target_chunk_id'],
                    similarity_score=conn_data['similarity_score'],
                    connection_type=conn_data['connection_type'],
                    source_document=conn_data['source_document'],
                    target_document=conn_data['target_document']
                )
                connections.append(connection)

            metadata = SimilarityMetadata(**cache_data['metadata'])

            self.logger.info(
                f"📂 Successfully loaded {len(matrices)} matrices and {len(connections)} connections from cache")

            return {
                'metadata': metadata,
                'matrices': matrices,
                'connections': connections,
                'chunk_index_map': cache_data['chunk_index_map']
            }

        except Exception as e:
            self.logger.error(f"Failed to load cached similarities: {e}")
            raise

    def get_similarity_statistics(self, similarity_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about computed similarities."""
        stats = {}

        for model_name, data in similarity_data.items():
            metadata = data['metadata']

            model_stats = {
                'chunk_count': metadata.chunk_count,
                'computation_time': metadata.computation_time,
                'memory_usage_mb': metadata.memory_usage_mb,
                'total_connections': metadata.total_connections,
                'intra_doc_connections': metadata.intra_doc_connections,
                'inter_doc_connections': metadata.inter_doc_connections
            }

            stats[model_name] = model_stats

        return stats