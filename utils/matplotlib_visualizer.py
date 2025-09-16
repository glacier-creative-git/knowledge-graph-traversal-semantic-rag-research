#!/usr/bin/env python3
"""
Knowledge Graph Traversal 2D Heatmap Visualizer with Matplotlib
==============================================================

Creates publication-ready 2D heatmap visualizations showing semantic similarity
matrices as "chess boards" with traversal paths drawn between chunks.
Adapted from perfect reference examples to work with algorithm results.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import ConnectionPatch, Circle
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import warnings
import yaml
from pathlib import Path

from .algos.base_algorithm import RetrievalResult
from .traversal import TraversalPath, GranularityLevel, ConnectionType
from .knowledge_graph import KnowledgeGraph


@dataclass
class DocumentHeatmapInfo:
    """Information about a document's heatmap visualization"""
    doc_id: str
    similarity_matrix: np.ndarray
    chunks_in_doc: List[str]  # Chunk IDs in this document
    chunk_to_matrix_idx: Dict[str, int]  # Map chunk ID to matrix index
    ax: plt.Axes
    bbox: Any  # Axes bounding box
    title: str


@dataclass
class TraversalStep:
    """Simplified traversal step for visualization"""
    step_number: int
    chunk_id: str
    doc_id: str
    connection_type: str
    relevance_score: float
    is_early_stop_point: bool = False


class KnowledgeGraphMatplotlibVisualizer:
    """Create 2D heatmap visualizations of knowledge graph traversal"""

    def __init__(self, knowledge_graph: KnowledgeGraph, figure_size: Tuple[int, int] = (20, 8), dpi: int = 150):
        self.kg = knowledge_graph
        self.figure_size = figure_size
        self.dpi = dpi
        
        # Load config for visualization parameters
        self._load_config()
    
    def _load_config(self):
        """Load configuration from config.yaml"""
        try:
            # Find config.yaml in project root (assume we're in utils/)
            config_path = Path(__file__).parent.parent / "config.yaml"
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.window_buffer_size = config.get('chunking', {}).get('window_visualization_buffer_size', 3)
        except Exception as e:
            print(f"Warning: Could not load config, using default buffer size: {e}")
            self.window_buffer_size = 3

    def visualize_retrieval_result(self, result: RetrievalResult, query: str,
                                   max_documents: int = 6) -> plt.Figure:
        """
        Create 2D heatmap visualization of algorithm traversal results.
        Matches the style of the perfect reference examples.
        """
        print(f"🎨 Creating 2D heatmap visualization for {result.algorithm_name}")

        # Extract traversal information from the result
        traversal_steps = self._extract_traversal_steps(result)
        self._current_steps = traversal_steps  # Store for reference during heatmap building
        print(f"Extracted {len(traversal_steps)} traversal steps")
        
        if not traversal_steps:
            print("⚠️ No traversal steps found - creating basic visualization")
            return self._create_basic_visualization(result, query)

        # Get documents involved in traversal
        involved_docs = self._get_involved_documents(traversal_steps)
        involved_docs = involved_docs[:max_documents]  # Limit for visualization clarity

        print(f"📄 Visualizing traversal across {len(involved_docs)} documents: {involved_docs}")

        # Build similarity matrices for each document using cached embeddings
        doc_heatmap_infos = self._build_document_heatmaps(involved_docs)

        if not doc_heatmap_infos:
            print("❌ Could not build heatmaps - falling back to basic visualization")
            return self._create_basic_visualization(result, query)

        # Create the figure with heatmaps (matching reference style)
        fig = self._create_heatmap_figure(doc_heatmap_infos, result, query)

        # Draw traversal path (like reference examples)
        self._draw_traversal_path(fig, traversal_steps, doc_heatmap_infos)

        print(f"✅ 2D visualization created successfully")
        return fig

    def _extract_traversal_steps(self, result: RetrievalResult) -> List[TraversalStep]:
        """Extract traversal steps from the result - robust approach"""
        steps = []

        if not result.traversal_path or not result.traversal_path.nodes:
            # For BasicRetrieval or algorithms without traversal paths,
            # create steps from the metadata if available
            if hasattr(result, 'metadata') and result.metadata:
                extraction_metadata = result.metadata.get('extraction_metadata', {})
                for i, (chunk_id, chunk_metadata) in enumerate(extraction_metadata.items()):
                    doc_id = self._get_chunk_document(chunk_id)
                    steps.append(TraversalStep(
                        step_number=i,
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        connection_type='similarity_based',
                        relevance_score=chunk_metadata.get('similarity_score', 0.5),
                        is_early_stop_point=False
                    ))
            return steps

        path = result.traversal_path

        # Check for early stopping from metadata
        early_stop_triggered = False
        if hasattr(result, 'metadata') and result.metadata:
            early_stop_triggered = result.metadata.get('early_stop_triggered', False)

        # Process traversal path nodes
        for i, node_id in enumerate(path.nodes):
            granularity = path.granularity_levels[i] if i < len(path.granularity_levels) else GranularityLevel.CHUNK
            connection_type = path.connection_types[i - 1] if i > 0 and i - 1 < len(path.connection_types) else ConnectionType.RAW_SIMILARITY

            # Only include chunk-level nodes for heatmap visualization
            if granularity == GranularityLevel.CHUNK:
                doc_id = self._get_chunk_document(node_id)
                
                # Calculate relevance score using cached similarities or confidence scores
                relevance_score = self._calculate_node_relevance(node_id, result)

                # Check if this is the early stop point
                is_early_stop_point = (early_stop_triggered and i == len(path.nodes) - 1)

                steps.append(TraversalStep(
                    step_number=i,
                    chunk_id=node_id,
                    doc_id=doc_id,
                    connection_type=connection_type.value if hasattr(connection_type, 'value') else str(connection_type),
                    relevance_score=relevance_score,
                    is_early_stop_point=is_early_stop_point
                ))

        return steps

    def _get_involved_documents(self, steps: List[TraversalStep]) -> List[str]:
        """Get list of documents involved in traversal, in order of first appearance"""
        seen_docs = set()
        doc_order = []

        for step in steps:
            if step.doc_id and step.doc_id != "unknown" and step.doc_id not in seen_docs:
                doc_order.append(step.doc_id)
                seen_docs.add(step.doc_id)

        print(f"Documents involved in traversal: {doc_order}")
        return doc_order
    
    def _get_current_traversal_steps(self) -> List[TraversalStep]:
        """Get current traversal steps for reference during heatmap building"""
        # This will be set by the calling function
        return getattr(self, '_current_steps', [])
    
    def _extract_chunk_index_from_id(self, chunk_id: str) -> Optional[int]:
        """Extract the starting index from a chunk ID for sequential ordering.
        
        Expected format: 'Document_name_window_START_END_hash'
        Returns the START index for sequential ordering.
        """
        try:
            parts = chunk_id.split('_')
            if 'window' in parts:
                window_idx = parts.index('window')
                if window_idx + 1 < len(parts):
                    start_idx = int(parts[window_idx + 1])
                    return start_idx
        except (ValueError, IndexError) as e:
            print(f"Warning: Could not extract chunk index from {chunk_id}: {e}")
        return None
    
    def _get_sequential_chunk_window(self, doc_id: str, traversed_chunks: List[str]) -> List[str]:
        """Get sequential chunk window for a document based on traversed chunks.
        
        Args:
            doc_id: Document identifier
            traversed_chunks: List of chunk IDs that were traversed in this document
            
        Returns:
            List of chunk IDs in sequential order within the window
        """
        # Get all chunks for this document from knowledge graph
        all_doc_chunks = []
        for chunk_id, chunk_obj in self.kg.chunks.items():
            if self._get_chunk_document(chunk_id) == doc_id:
                chunk_index = self._extract_chunk_index_from_id(chunk_id)
                if chunk_index is not None:
                    all_doc_chunks.append((chunk_index, chunk_id))
        
        if not all_doc_chunks:
            return []
        
        # Sort by sequential index
        all_doc_chunks.sort(key=lambda x: x[0])
        
        # Find min and max indices of traversed chunks
        traversed_indices = []
        for chunk_id in traversed_chunks:
            chunk_index = self._extract_chunk_index_from_id(chunk_id)
            if chunk_index is not None:
                traversed_indices.append(chunk_index)
        
        if not traversed_indices:
            # Fallback: return first few chunks if no traversed chunks found
            return [chunk_id for _, chunk_id in all_doc_chunks[:15]]
        
        min_traversed_idx = min(traversed_indices)
        max_traversed_idx = max(traversed_indices)
        
        # Expand window by buffer size in each direction
        window_start = min_traversed_idx - self.window_buffer_size
        window_end = max_traversed_idx + self.window_buffer_size
        
        # Select chunks within the window, maintaining sequential order
        windowed_chunks = []
        for chunk_index, chunk_id in all_doc_chunks:
            if window_start <= chunk_index <= window_end:
                windowed_chunks.append(chunk_id)
        
        print(f"   Sequential window: indices {window_start} to {window_end} (traversed: {min_traversed_idx}-{max_traversed_idx})")
        return windowed_chunks
        
    def _get_actual_chunk_indices(self, chunk_ids: List[str]) -> List[int]:
        """Extract actual document indices from chunk IDs for accurate axis labeling.
        
        Args:
            chunk_ids: List of chunk IDs in the current visualization
            
        Returns:
            List of actual document indices corresponding to the chunk IDs
        """
        indices = []
        for chunk_id in chunk_ids:
            chunk_index = self._extract_chunk_index_from_id(chunk_id)
            if chunk_index is not None:
                indices.append(chunk_index)
            else:
                # Fallback: use position in list if extraction fails
                indices.append(len(indices))
        return indices
    
    def _get_all_document_chunks(self, doc_id: str) -> List[str]:
        """Get all chunks for a document in sequential order for global visualization.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            List of all chunk IDs in sequential order for this document
        """
        # Get all chunks for this document from knowledge graph
        all_doc_chunks = []
        for chunk_id, chunk_obj in self.kg.chunks.items():
            if self._get_chunk_document(chunk_id) == doc_id:
                chunk_index = self._extract_chunk_index_from_id(chunk_id)
                if chunk_index is not None:
                    all_doc_chunks.append((chunk_index, chunk_id))
        
        if not all_doc_chunks:
            return []
        
        # Sort by sequential index and return chunk IDs
        all_doc_chunks.sort(key=lambda x: x[0])
        return [chunk_id for _, chunk_id in all_doc_chunks]
    
    def _detect_reading_sessions(self, steps: List[TraversalStep]) -> List[List[TraversalStep]]:
        """Detect continuous reading sessions for sequential window visualization.
        
        A reading session is a continuous sequence of steps within the same document.
        Document changes and large index gaps trigger new sessions, but we preserve
        cross-document connection information for visualization.
        
        Args:
            steps: List of traversal steps
            
        Returns:
            List of reading sessions, where each session is a list of steps
        """
        if not steps:
            return []
        
        sessions = []
        current_session = [steps[0]]
        
        for i in range(1, len(steps)):
            current_step = steps[i]
            previous_step = steps[i-1]
            
            # Start new session if:
            # 1. Different document, OR  
            # 2. Same document but large gap in chunk indices (>8 chunks apart)
            start_new_session = False
            
            if current_step.doc_id != previous_step.doc_id:
                start_new_session = True
            else:
                # Check index gap within same document
                curr_idx = self._extract_chunk_index_from_id(current_step.chunk_id)
                prev_idx = self._extract_chunk_index_from_id(previous_step.chunk_id)
                
                if curr_idx is not None and prev_idx is not None:
                    index_gap = abs(curr_idx - prev_idx)
                    if index_gap > 8:  # Reduced threshold for better session continuity
                        start_new_session = True
            
            if start_new_session:
                sessions.append(current_session)
                current_session = [current_step]
            else:
                current_session.append(current_step)
        
        # Add the last session
        if current_session:
            sessions.append(current_session)
        
        return sessions

    def _build_document_heatmaps(self, doc_ids: List[str]) -> List[DocumentHeatmapInfo]:
        """Build similarity matrices for each document using sequential windowed approach.
        
        Creates windows around traversed chunks, maintaining sequential document order
        to demonstrate algorithm reading patterns.
        """
        heatmap_infos = []

        for doc_id in doc_ids:
            print(f"🔍 Building sequential heatmap for document: '{doc_id}'")
            
            # Find all chunks that were traversed in this document
            traversed_chunks = []
            for step in self._get_current_traversal_steps():
                if step.doc_id == doc_id:
                    traversed_chunks.append(step.chunk_id)
            
            print(f"   Found {len(traversed_chunks)} traversed chunks")
            
            # Get sequential window of chunks around traversed ones
            doc_chunks = self._get_sequential_chunk_window(doc_id, traversed_chunks)
            
            print(f"   Sequential window contains {len(doc_chunks)} chunks")
            if len(doc_chunks) > 0:
                print(f"   Sample chunk IDs: {doc_chunks[:3]}...")

            # Need at least 2 chunks for meaningful heatmap
            if len(doc_chunks) < 2:
                print(f"⚠️ Document {doc_id} has fewer than 2 chunks in window, skipping")
                continue

            # Get embeddings for chunks in sequential order
            chunk_embeddings = []
            valid_chunks = []

            for chunk_id in doc_chunks:
                embedding = self._get_chunk_embedding(chunk_id)
                if embedding is not None:
                    chunk_embeddings.append(embedding)
                    valid_chunks.append(chunk_id)

            if len(valid_chunks) < 2:
                print(f"⚠️ Document {doc_id} has only {len(valid_chunks)} chunks with embeddings, creating minimal visualization")
                # Create a minimal 2x2 matrix for single chunk case
                if len(valid_chunks) == 1:
                    # Duplicate the single chunk to create a 2x2 matrix
                    chunk_embeddings.append(chunk_embeddings[0])
                    valid_chunks.append(valid_chunks[0] + "_duplicate")
                else:
                    continue

            # Build similarity matrix using sequential chunks (maintains existing color scheme)
            embeddings_array = np.array(chunk_embeddings)
            # Normalize embeddings for proper cosine similarity
            norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            # Avoid division by zero
            norms = np.where(norms == 0, 1, norms)
            embeddings_array = embeddings_array / norms
            similarity_matrix = np.dot(embeddings_array, embeddings_array.T)
            
            print(f"   Built {similarity_matrix.shape} sequential similarity matrix")

            # Create mapping from chunk ID to matrix index (sequential order preserved)
            chunk_to_matrix_idx = {chunk_id: i for i, chunk_id in enumerate(valid_chunks)}

            heatmap_infos.append(DocumentHeatmapInfo(
                doc_id=doc_id,
                similarity_matrix=similarity_matrix,
                chunks_in_doc=valid_chunks,
                chunk_to_matrix_idx=chunk_to_matrix_idx,
                ax=None,  # Will be set when creating the figure
                bbox=None,  # Will be set when creating the figure
                title=f"Document {doc_id}"
            ))

        return heatmap_infos
    
    def create_global_visualization(self, result: RetrievalResult, query: str,
                                   max_documents: int = 6) -> plt.Figure:
        """Create global visualization showing full documents with complete traversal paths.
        
        This method provides strategic overview by showing entire documents and how
        algorithms navigate across the full document landscape.
        
        Args:
            result: RetrievalResult from any algorithm
            query: Original query string
            max_documents: Maximum number of documents to show
            
        Returns:
            Matplotlib Figure showing global document structure and traversal
        """
        print(f"🌍 Creating global visualization for {result.algorithm_name}")
        
        # Extract traversal information
        traversal_steps = self._extract_traversal_steps(result)
        self._current_steps = traversal_steps
        print(f"Global view: {len(traversal_steps)} total steps")
        
        if not traversal_steps:
            return self._create_basic_visualization(result, query)
        
        # Get documents involved in traversal
        involved_docs = self._get_involved_documents(traversal_steps)
        involved_docs = involved_docs[:max_documents]
        print(f"📚 Global view across {len(involved_docs)} documents: {involved_docs}")
        
        # Build global document heatmaps (full documents)
        heatmap_infos = self._build_global_document_heatmaps(involved_docs)
        
        if not heatmap_infos:
            return self._create_basic_visualization(result, query)
        
        # Create figure with global view styling
        fig = self._create_heatmap_figure(heatmap_infos, result, query)
        
        # Draw complete traversal path across full documents
        self._draw_traversal_path(fig, traversal_steps, heatmap_infos)
        
        print(f"✅ Global visualization created successfully")
        return fig
    
    def create_sequential_window_visualization(self, result: RetrievalResult, query: str) -> plt.Figure:
        """Create sequential window visualization showing reading sessions chronologically.
        
        This method provides tactical analysis by showing individual reading sessions
        as separate panels, arranged left-to-right in temporal order.
        
        Args:
            result: RetrievalResult from any algorithm
            query: Original query string
            
        Returns:
            Matplotlib Figure showing sequential reading sessions
        """
        print(f"📖 Creating sequential window visualization for {result.algorithm_name}")
        
        # Extract traversal information
        traversal_steps = self._extract_traversal_steps(result)
        self._current_steps = traversal_steps
        print(f"Sequential view: {len(traversal_steps)} total steps")
        
        if not traversal_steps:
            return self._create_basic_visualization(result, query)
        
        # Detect reading sessions
        reading_sessions = self._detect_reading_sessions(traversal_steps)
        print(f"📑 Detected {len(reading_sessions)} reading sessions")
        
        # Build heatmaps for each reading session
        session_heatmaps = self._build_session_heatmaps(reading_sessions)
        
        if not session_heatmaps:
            return self._create_basic_visualization(result, query)
        
        # Create figure with sequential session layout
        fig = self._create_sequential_session_figure(session_heatmaps, result, query)
        
        # Draw session-specific traversal paths
        self._draw_sequential_session_paths(fig, reading_sessions, session_heatmaps)
        
        print(f"✅ Sequential window visualization created successfully")
        return fig
    
    def _build_global_document_heatmaps(self, doc_ids: List[str]) -> List[DocumentHeatmapInfo]:
        """Build similarity matrices for complete documents (global view).
        
        Creates full document matrices without windowing to show complete
        document architecture and traversal patterns.
        
        Args:
            doc_ids: List of document identifiers
            
        Returns:
            List of DocumentHeatmapInfo objects for global visualization
        """
        heatmap_infos = []
        
        for doc_id in doc_ids:
            print(f"🔍 Building global heatmap for document: '{doc_id}'")
            
            # Get ALL chunks for this document in sequential order
            doc_chunks = self._get_all_document_chunks(doc_id)
            
            print(f"   Global document contains {len(doc_chunks)} total chunks")
            
            if len(doc_chunks) < 2:
                print(f"⚠️ Document {doc_id} has fewer than 2 chunks, skipping")
                continue
            
            # Get embeddings for all chunks
            chunk_embeddings = []
            valid_chunks = []
            
            for chunk_id in doc_chunks:
                embedding = self._get_chunk_embedding(chunk_id)
                if embedding is not None:
                    chunk_embeddings.append(embedding)
                    valid_chunks.append(chunk_id)
            
            if len(valid_chunks) < 2:
                print(f"⚠️ Document {doc_id} has only {len(valid_chunks)} chunks with embeddings")
                continue
            
            # Build complete similarity matrix
            embeddings_array = np.array(chunk_embeddings)
            norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            embeddings_array = embeddings_array / norms
            similarity_matrix = np.dot(embeddings_array, embeddings_array.T)
            
            print(f"   Built global {similarity_matrix.shape} similarity matrix")
            
            chunk_to_matrix_idx = {chunk_id: i for i, chunk_id in enumerate(valid_chunks)}
            
            heatmap_infos.append(DocumentHeatmapInfo(
                doc_id=doc_id,
                similarity_matrix=similarity_matrix,
                chunks_in_doc=valid_chunks,
                chunk_to_matrix_idx=chunk_to_matrix_idx,
                ax=None,
                bbox=None,
                title=f"Document {doc_id} (Global View)"
            ))
        
        return heatmap_infos
    
    def _build_session_heatmaps(self, reading_sessions: List[List[TraversalStep]]) -> List[DocumentHeatmapInfo]:
        """Build similarity matrices for each reading session.
        
        Creates windowed matrices for individual reading sessions,
        maintaining sequential order within each session.
        
        Args:
            reading_sessions: List of reading sessions (each session is a list of steps)
            
        Returns:
            List of DocumentHeatmapInfo objects for session visualization
        """
        session_heatmaps = []
        
        for session_idx, session_steps in enumerate(reading_sessions):
            if not session_steps:
                continue
                
            # Determine document and chunk range for this session
            doc_id = session_steps[0].doc_id
            session_chunk_ids = [step.chunk_id for step in session_steps]
            
            print(f"📄 Building session {session_idx + 1} heatmap for '{doc_id}'")
            print(f"   Session spans {len(session_chunk_ids)} chunks")
            
            # Get windowed chunks around this session
            windowed_chunks = self._get_sequential_chunk_window(doc_id, session_chunk_ids)
            
            if len(windowed_chunks) < 2:
                print(f"⚠️ Session {session_idx + 1} has insufficient chunks, skipping")
                continue
            
            # Build similarity matrix for session window
            chunk_embeddings = []
            valid_chunks = []
            
            for chunk_id in windowed_chunks:
                embedding = self._get_chunk_embedding(chunk_id)
                if embedding is not None:
                    chunk_embeddings.append(embedding)
                    valid_chunks.append(chunk_id)
            
            if len(valid_chunks) < 2:
                continue
            
            # Compute similarity matrix
            embeddings_array = np.array(chunk_embeddings)
            norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            embeddings_array = embeddings_array / norms
            similarity_matrix = np.dot(embeddings_array, embeddings_array.T)
            
            print(f"   Built session {similarity_matrix.shape} matrix")
            
            chunk_to_matrix_idx = {chunk_id: i for i, chunk_id in enumerate(valid_chunks)}
            
            # Get chunk indices for title
            session_indices = [self._extract_chunk_index_from_id(cid) for cid in session_chunk_ids]
            session_indices = [idx for idx in session_indices if idx is not None]
            if session_indices:
                idx_range = f"chunks {min(session_indices)}-{max(session_indices)}"
            else:
                idx_range = f"{len(session_chunk_ids)} chunks"
            
            session_heatmaps.append(DocumentHeatmapInfo(
                doc_id=f"Session {session_idx + 1}",
                similarity_matrix=similarity_matrix,
                chunks_in_doc=valid_chunks,
                chunk_to_matrix_idx=chunk_to_matrix_idx,
                ax=None,
                bbox=None,
                title=f"Session {session_idx + 1}: {doc_id} ({idx_range})"
            ))
        
        return session_heatmaps
    
    def _create_sequential_session_figure(self, session_heatmaps: List[DocumentHeatmapInfo],
                                         result: RetrievalResult, query: str) -> plt.Figure:
        """Create figure layout for sequential session visualization.
        
        Arranges reading sessions chronologically from left to right,
        with each session as a separate panel.
        
        Args:
            session_heatmaps: List of DocumentHeatmapInfo for each session
            result: RetrievalResult for metadata
            query: Original query string
            
        Returns:
            Matplotlib Figure with sequential session layout
        """
        num_sessions = len(session_heatmaps)
        
        # Calculate figure width based on number of sessions
        session_width = 4  # Width per session panel
        fig_width = max(self.figure_size[0], num_sessions * session_width)
        
        # Create figure with appropriate styling
        plt.style.use('default')
        fig = plt.figure(figsize=(fig_width, self.figure_size[1]), facecolor='white', dpi=self.dpi)
        
        # Create grid layout - colorbar at top, sessions below
        gs = gridspec.GridSpec(2, num_sessions, figure=fig,
                               height_ratios=[0.08, 1],
                               hspace=0.05, wspace=0.35)  # More space between sessions
        
        # Create horizontal colorbar at top
        cbar_ax = fig.add_subplot(gs[0, :])
        
        # Session visualization parameters
        vmin, vmax = 0, 1
        cmap = 'RdYlBu_r'  # Maintain consistent color scheme
        
        # Create heatmap for each session
        for i, heatmap_info in enumerate(session_heatmaps):
            ax = fig.add_subplot(gs[1, i])
            heatmap_info.ax = ax
            heatmap_info.bbox = ax.get_position()
            
            # Create heatmap with consistent styling
            im = ax.imshow(heatmap_info.similarity_matrix,
                           cmap=cmap,
                           aspect='equal',
                           vmin=vmin, vmax=vmax,
                           interpolation='nearest')
            
            # Set title and labels for session
            ax.set_title(heatmap_info.title, fontsize=12, fontweight='bold', pad=15)
            ax.set_xlabel('Chunk Index', fontsize=10)
            if i == 0:  # Only leftmost session gets y-label
                ax.set_ylabel('Chunk Index', fontsize=10)
            
            # Get actual document indices for session
            actual_indices = self._get_actual_chunk_indices(heatmap_info.chunks_in_doc)
            n_chunks = len(heatmap_info.chunks_in_doc)
            
            # Set ticks with actual document positions
            if n_chunks <= 12:
                ax.set_xticks(range(n_chunks))
                ax.set_yticks(range(n_chunks))
                ax.set_xticklabels(actual_indices, rotation=45, fontsize=8)
                ax.set_yticklabels(actual_indices, fontsize=8)
            else:
                # For larger sessions, show fewer ticks
                tick_positions = np.linspace(0, n_chunks - 1, min(8, n_chunks), dtype=int)
                selected_indices = [actual_indices[i] for i in tick_positions]
                ax.set_xticks(tick_positions)
                ax.set_yticks(tick_positions)
                ax.set_xticklabels(selected_indices, rotation=45, fontsize=8)
                ax.set_yticklabels(selected_indices, fontsize=8)
        
        # Add shared colorbar
        if session_heatmaps:
            cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
            cbar.set_label('Chunk Similarity Score', fontsize=12, fontweight='bold')
            cbar_ax.xaxis.set_label_position('top')
        
        # Add comprehensive title
        title = (f"{result.algorithm_name} Sequential Reading Sessions\n"
                 f"Query: '{query[:60]}...' | "
                 f"Retrieved: {len(result.retrieved_content)} sentences | "
                 f"Score: {result.final_score:.3f} | "
                 f"Sessions: {num_sessions}")
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.95)
        
        return fig
    
    def _draw_sequential_session_paths(self, fig: plt.Figure, 
                                     reading_sessions: List[List[TraversalStep]],
                                     session_heatmaps: List[DocumentHeatmapInfo]):
        """Draw traversal paths within and between reading sessions.
        
        Uses consistent marker system from windowed approach with cross-document
        connections between sessions. Maintains global step numbering across sessions.
        
        Args:
            fig: Matplotlib figure
            reading_sessions: List of reading sessions
            session_heatmaps: Corresponding heatmap info for each session
        """
        if len(reading_sessions) != len(session_heatmaps):
            print("Warning: Mismatch between sessions and heatmaps")
            return
        
        print(f"Drawing sequential paths across {len(reading_sessions)} sessions")
        
        # Flatten all steps to maintain global numbering
        all_steps = []
        session_step_mapping = {}  # Maps (session_idx, step_idx) to global_step_idx
        
        global_step_idx = 0
        for session_idx, session_steps in enumerate(reading_sessions):
            for step_idx, step in enumerate(session_steps):
                all_steps.append(step)
                session_step_mapping[(session_idx, step_idx)] = global_step_idx
                global_step_idx += 1
        
        # Draw paths within each session using windowed marker system
        for session_idx, (session_steps, heatmap_info) in enumerate(zip(reading_sessions, session_heatmaps)):
            if not session_steps or not heatmap_info.ax:
                continue
            
            ax = heatmap_info.ax
            print(f"   Session {session_idx + 1}: {len(session_steps)} steps")
            
            # Draw step markers using windowed approach
            for step_idx, step in enumerate(session_steps):
                if step.chunk_id not in heatmap_info.chunk_to_matrix_idx:
                    continue
                
                matrix_idx = heatmap_info.chunk_to_matrix_idx[step.chunk_id]
                global_step_number = session_step_mapping[(session_idx, step_idx)]
                
                # Use windowed marker system (consistent with original approach)
                if global_step_number == 0:
                    # Global anchor point (gold star)
                    marker_color = 'gold'
                    marker_size = 400
                    edge_color = 'black'
                    edge_width = 3
                    marker_symbol = 'star'
                elif step.is_early_stop_point:
                    # Early stopping point (red)
                    marker_color = 'red'
                    marker_size = 350
                    edge_color = 'darkred'
                    edge_width = 3
                    marker_symbol = 'o'
                else:
                    # Regular traversal step (green scale based on relevance)
                    relevance = max(0, min(1, step.relevance_score))
                    green_intensity = 0.3 + (relevance * 0.7)
                    marker_color = (1.0 - green_intensity, 1.0, 1.0 - green_intensity)
                    marker_size = 200 + (relevance * 200)
                    edge_color = 'darkgreen'
                    edge_width = 2
                    marker_symbol = 'o'
                
                # Draw marker on diagonal
                if marker_symbol == 'star':
                    ax.scatter([matrix_idx], [matrix_idx], s=marker_size, marker='*',
                               c=[marker_color], edgecolors=edge_color, linewidths=edge_width, zorder=10)
                else:
                    ax.scatter([matrix_idx], [matrix_idx], s=marker_size,
                               c=[marker_color], edgecolors=edge_color, linewidths=edge_width, zorder=10)
                
                # Add global step number (continuous across sessions)
                ax.text(matrix_idx, matrix_idx, str(global_step_number),
                        ha='center', va='center', fontsize=11, fontweight='bold',
                        color='black', zorder=11)
            
            # Draw connections within session (same document)
            for step_idx in range(len(session_steps) - 1):
                current_step = session_steps[step_idx]
                next_step = session_steps[step_idx + 1]
                
                if (current_step.chunk_id in heatmap_info.chunk_to_matrix_idx and
                    next_step.chunk_id in heatmap_info.chunk_to_matrix_idx):
                    
                    current_idx = heatmap_info.chunk_to_matrix_idx[current_step.chunk_id]
                    next_idx = heatmap_info.chunk_to_matrix_idx[next_step.chunk_id]
                    
                    # Within-session connection (dotted green)
                    ax.annotate('', xy=(next_idx, next_idx),
                                xytext=(current_idx, current_idx),
                                arrowprops=dict(arrowstyle='->',
                                                color='green',
                                                linestyle=':',
                                                linewidth=2,
                                                alpha=0.6),
                                zorder=5)
        
        # Draw cross-session connections (between different sessions)
        for session_idx in range(len(reading_sessions) - 1):
            current_session = reading_sessions[session_idx]
            next_session = reading_sessions[session_idx + 1]
            current_heatmap = session_heatmaps[session_idx]
            next_heatmap = session_heatmaps[session_idx + 1]
            
            if (current_session and next_session and 
                current_heatmap.ax and next_heatmap.ax):
                
                # Get last step of current session and first step of next session
                last_step = current_session[-1]
                first_step = next_session[0]
                
                if (last_step.chunk_id in current_heatmap.chunk_to_matrix_idx and
                    first_step.chunk_id in next_heatmap.chunk_to_matrix_idx):
                    
                    current_idx = current_heatmap.chunk_to_matrix_idx[last_step.chunk_id]
                    next_idx = next_heatmap.chunk_to_matrix_idx[first_step.chunk_id]
                    
                    # Determine connection type based on document change
                    if last_step.doc_id != first_step.doc_id:
                        # Cross-document connection (purple dashed)
                        line_color = 'purple'
                        line_style = '--'
                        line_width = 3
                        alpha = 0.8
                        connection_label = 'Cross-Document'
                    else:
                        # Same document but different session (blue dashed)
                        line_color = 'blue'
                        line_style = '--'
                        line_width = 2
                        alpha = 0.7
                        connection_label = 'Hierarchical'
                    
                    # Draw cross-session connection using ConnectionPatch
                    conn = ConnectionPatch(
                        xyA=(current_idx, current_idx), coordsA='data', axesA=current_heatmap.ax,
                        xyB=(next_idx, next_idx), coordsB='data', axesB=next_heatmap.ax,
                        arrowstyle='->',
                        linestyle=line_style,
                        linewidth=line_width,
                        color=line_color,
                        alpha=alpha,
                        zorder=5
                    )
                    fig.add_artist(conn)
                    print(f"   Cross-session connection: Session {session_idx + 1} -> {session_idx + 2} ({connection_label})")
        
        # Add updated legend matching windowed approach
        legend_elements = [
            plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='gold',
                       markersize=15, markeredgecolor='black', markeredgewidth=2,
                       linestyle='None', label='Anchor Point'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen',
                       markersize=12, markeredgecolor='darkgreen', markeredgewidth=2,
                       linestyle='None', label='Traversal Step'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                       markersize=12, markeredgecolor='darkred', markeredgewidth=2,
                       linestyle='None', label='Early Stop Point'),
            plt.Line2D([0], [0], color='purple', linestyle='--', linewidth=3,
                       label='Cross-Document'),
            plt.Line2D([0], [0], color='blue', linestyle='--', linewidth=2,
                       label='Hierarchical'),
            plt.Line2D([0], [0], color='green', linestyle=':', linewidth=2,
                       label='Within Session')
        ]
        
        fig.legend(handles=legend_elements, loc='lower center',
                   bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=True,
                   fancybox=True, shadow=True, fontsize=10)

    def _create_heatmap_figure(self, heatmap_infos: List[DocumentHeatmapInfo],
                               result: RetrievalResult, query: str) -> plt.Figure:
        """Create the figure with side-by-side heatmaps (matching reference style)"""

        num_docs = len(heatmap_infos)

        # Calculate figure width based on number of documents (like reference)
        fig_width = max(self.figure_size[0], num_docs * 4)

        # Set matplotlib style for publication quality (like reference examples)
        plt.style.use('default')

        fig = plt.figure(figsize=(fig_width, self.figure_size[1]), facecolor='white', dpi=self.dpi)

        # Create grid layout - space at top for colorbar (like reference)
        gs = gridspec.GridSpec(2, num_docs, figure=fig,
                               height_ratios=[0.08, 1],
                               hspace=0.05, wspace=0.25)

        # Create horizontal colorbar at top
        cbar_ax = fig.add_subplot(gs[0, :])

        # Create heatmaps with proper colormap (matching reference colors)
        vmin, vmax = 0, 1
        cmap = 'RdYlBu_r'  # Same as reference examples

        for i, heatmap_info in enumerate(heatmap_infos):
            ax = fig.add_subplot(gs[1, i])
            heatmap_info.ax = ax
            heatmap_info.bbox = ax.get_position()

            # Create heatmap (same style as reference)
            im = ax.imshow(heatmap_info.similarity_matrix,
                           cmap=cmap,
                           aspect='equal',
                           vmin=vmin, vmax=vmax,
                           interpolation='nearest')

            # Set title and labels (matching reference style)
            ax.set_title(heatmap_info.title, fontsize=14, fontweight='bold', pad=15)
            ax.set_xlabel('Chunk Index (Document Position)', fontsize=12)
            if i == 0:  # Only leftmost plot gets y-label
                ax.set_ylabel('Chunk Index (Document Position)', fontsize=12)

            # Get actual document indices for accurate labeling
            actual_indices = self._get_actual_chunk_indices(heatmap_info.chunks_in_doc)
            n_chunks = len(heatmap_info.chunks_in_doc)
            
            # Set ticks to show actual document positions
            if n_chunks <= 15:
                ax.set_xticks(range(n_chunks))
                ax.set_yticks(range(n_chunks))
                ax.set_xticklabels(actual_indices, rotation=45)
                ax.set_yticklabels(actual_indices)
            else:
                # For larger matrices, show fewer ticks but still use actual indices
                tick_positions = np.linspace(0, n_chunks - 1, min(10, n_chunks), dtype=int)
                selected_indices = [actual_indices[i] for i in tick_positions]
                ax.set_xticks(tick_positions)
                ax.set_yticks(tick_positions)
                ax.set_xticklabels(selected_indices, rotation=45)
                ax.set_yticklabels(selected_indices)

        # Add colorbar (matching reference style)
        if heatmap_infos:
            cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
            cbar.set_label('Chunk Similarity Score', fontsize=12, fontweight='bold')
            cbar_ax.xaxis.set_label_position('top')

        # Add title (matching reference style)
        title = (f"{result.algorithm_name} Traversal Path Visualization\n"
                 f"Query: '{query[:80]}...' | "
                 f"Retrieved: {len(result.retrieved_content)} sentences | "
                 f"Score: {result.final_score:.3f}")
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.95)

        return fig

    def _draw_traversal_path(self, fig: plt.Figure, steps: List[TraversalStep],
                             heatmap_infos: List[DocumentHeatmapInfo]):
        """Draw the traversal path on the heatmaps (like reference examples)"""

        if len(steps) < 1:
            return

        print(f"🎯 Drawing traversal path for {len(steps)} steps")

        # Create mapping from doc_id to heatmap_info
        doc_to_heatmap = {info.doc_id: info for info in heatmap_infos}

        # Draw step markers (matching reference style)
        for step in steps:
            heatmap_info = doc_to_heatmap.get(step.doc_id)
            if not heatmap_info or step.chunk_id not in heatmap_info.chunk_to_matrix_idx:
                continue

            matrix_idx = heatmap_info.chunk_to_matrix_idx[step.chunk_id]
            ax = heatmap_info.ax

            # Determine marker properties (like reference examples)
            if step.step_number == 0:
                # Anchor point (gold star like reference)
                marker_color = 'gold'
                marker_size = 400
                edge_color = 'black'
                edge_width = 3
                marker_symbol = 'star'
            elif step.is_early_stop_point:
                # Early stopping point (red like reference)
                marker_color = 'red'
                marker_size = 350
                edge_color = 'darkred'
                edge_width = 3
                marker_symbol = 'o'
            else:
                # Regular traversal step (green scale like reference)
                relevance = max(0, min(1, step.relevance_score))
                green_intensity = 0.3 + (relevance * 0.7)  # 0.3 to 1.0
                marker_color = (1.0 - green_intensity, 1.0, 1.0 - green_intensity)
                marker_size = 200 + (relevance * 200)  # Size based on relevance
                edge_color = 'darkgreen'
                edge_width = 2
                marker_symbol = 'o'

            # Draw marker on diagonal (chunk similarity to itself, like reference)
            if marker_symbol == 'star':
                ax.scatter([matrix_idx], [matrix_idx], s=marker_size, marker='*',
                           c=[marker_color], edgecolors=edge_color, linewidths=edge_width, zorder=10)
            else:
                ax.scatter([matrix_idx], [matrix_idx], s=marker_size,
                           c=[marker_color], edgecolors=edge_color, linewidths=edge_width, zorder=10)

            # Add step number text (like reference)
            ax.text(matrix_idx, matrix_idx, str(step.step_number),
                    ha='center', va='center', fontsize=11, fontweight='bold',
                    color='black', zorder=11)

        # Draw connections between steps (like reference examples)
        for i in range(len(steps) - 1):
            current_step = steps[i]
            next_step = steps[i + 1]

            current_heatmap = doc_to_heatmap.get(current_step.doc_id)
            next_heatmap = doc_to_heatmap.get(next_step.doc_id)

            if not current_heatmap or not next_heatmap:
                continue

            if (current_step.chunk_id not in current_heatmap.chunk_to_matrix_idx or
                    next_step.chunk_id not in next_heatmap.chunk_to_matrix_idx):
                continue

            current_idx = current_heatmap.chunk_to_matrix_idx[current_step.chunk_id]
            next_idx = next_heatmap.chunk_to_matrix_idx[next_step.chunk_id]

            # Determine line properties based on connection type (like reference)
            if next_step.connection_type in ['cross_document', 'theme_bridge']:
                line_color = 'purple'
                line_width = 3
                line_style = '-'  # Solid for cross-document
                alpha = 0.8
            elif next_step.connection_type == 'hierarchical':
                line_color = 'blue'
                line_width = 2
                line_style = '--'  # Dashed for hierarchical
                alpha = 0.7
            else:
                line_color = 'green'
                line_width = 2
                line_style = ':'  # Dotted for within-document
                alpha = 0.6

            # Draw connection
            if current_step.doc_id != next_step.doc_id:
                # Cross-document connection using ConnectionPatch (like reference)
                conn = ConnectionPatch(
                    xyA=(current_idx, current_idx), coordsA='data', axesA=current_heatmap.ax,
                    xyB=(next_idx, next_idx), coordsB='data', axesB=next_heatmap.ax,
                    arrowstyle='->',
                    linestyle=line_style,
                    linewidth=line_width,
                    color=line_color,
                    alpha=alpha,
                    zorder=5
                )
                fig.add_artist(conn)
                print(f"   🔗 Cross-document connection: {current_step.doc_id} -> {next_step.doc_id}")
            else:
                # Same document connection
                current_heatmap.ax.annotate('', xy=(next_idx, next_idx),
                                            xytext=(current_idx, current_idx),
                                            arrowprops=dict(arrowstyle='->',
                                                            color=line_color,
                                                            linestyle=line_style,
                                                            linewidth=line_width,
                                                            alpha=alpha),
                                            zorder=5)

        # Add legend (matching reference style)
        self._add_legend(fig)

    def _add_legend(self, fig: plt.Figure):
        """Add legend explaining the visualization elements (matching reference style)"""

        legend_elements = [
            plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='gold',
                       markersize=15, markeredgecolor='black', markeredgewidth=2,
                       linestyle='None', label='Anchor Point'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen',
                       markersize=12, markeredgecolor='darkgreen', markeredgewidth=2,
                       linestyle='None', label='Traversal Step'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                       markersize=12, markeredgecolor='darkred', markeredgewidth=2,
                       linestyle='None', label='Early Stop Point'),
            plt.Line2D([0], [0], color='purple', linestyle='-', linewidth=3,
                       label='Cross-Document'),
            plt.Line2D([0], [0], color='blue', linestyle='--', linewidth=2,
                       label='Hierarchical'),
            plt.Line2D([0], [0], color='green', linestyle=':', linewidth=2,
                       label='Within Document')
        ]

        fig.legend(handles=legend_elements, loc='lower center',
                   bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=True,
                   fancybox=True, shadow=True, fontsize=10)

    def _create_basic_visualization(self, result: RetrievalResult, query: str) -> plt.Figure:
        """Create a basic visualization when no traversal path is available"""

        fig, ax = plt.subplots(figsize=self.figure_size, facecolor='white', dpi=self.dpi)

        # Create a simple text display of results (like reference fallback)
        ax.text(0.5, 0.7, f"Algorithm: {result.algorithm_name}",
                ha='center', va='center', fontsize=16, fontweight='bold')
        ax.text(0.5, 0.6, f"Query: {query[:100]}...",
                ha='center', va='center', fontsize=12)
        ax.text(0.5, 0.5, f"Retrieved Sentences: {len(result.retrieved_content)}",
                ha='center', va='center', fontsize=12)
        ax.text(0.5, 0.4, f"Final Score: {result.final_score:.3f}",
                ha='center', va='center', fontsize=12)
        ax.text(0.5, 0.3, f"Processing Time: {result.processing_time:.3f}s",
                ha='center', va='center', fontsize=12)

        if hasattr(result, 'metadata') and result.metadata:
            metadata_items = []
            for k, v in list(result.metadata.items())[:5]:  # Show max 5 items
                if isinstance(v, dict):
                    metadata_items.append(f"{k}: {len(v)} items")
                else:
                    metadata_items.append(f"{k}: {v}")
            
            metadata_text = "\n".join(metadata_items)
            ax.text(0.5, 0.2, f"Metadata:\n{metadata_text}",
                    ha='center', va='center', fontsize=10)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{result.algorithm_name} Results Summary", fontsize=18, fontweight='bold')
        ax.axis('off')

        return fig

    def _get_chunk_document(self, chunk_id: str) -> str:
        """Get document ID for a chunk (robust approach)"""
        # Try to get the chunk object from knowledge graph
        chunk_obj = self.kg.chunks.get(chunk_id)
        if chunk_obj and hasattr(chunk_obj, 'source_document'):
            return chunk_obj.source_document
        
        # Fallback: try to extract from chunk ID and convert underscores back to spaces
        if '_' in chunk_id:
            parts = chunk_id.split('_')
            # Find the 'window' keyword to separate document name from chunk info
            if 'window' in parts:
                window_idx = parts.index('window')
                doc_parts = parts[:window_idx]
                # Convert underscores back to spaces and handle parentheses
                doc_name = ' '.join(doc_parts).replace('(', '(').replace(')', ')')
                return doc_name
            else:
                # Fallback to everything except the last hash part
                doc_parts = parts[:-1] if len(parts) > 1 else parts
                doc_name = ' '.join(doc_parts).replace('(', '(').replace(')', ')')
                return doc_name
        
        return "unknown"

    def _get_chunk_embedding(self, chunk_id: str) -> Optional[np.ndarray]:
        """Get embedding for a chunk using cached embeddings from knowledge graph"""
        # Use the knowledge graph's embedding cache system
        return self.kg.get_chunk_embedding(chunk_id)

    def _calculate_node_relevance(self, node_id: str, result: RetrievalResult) -> float:
        """Calculate relevance score for a node"""
        # Try to use query similarities if available
        if hasattr(result, 'query_similarities') and result.query_similarities:
            if node_id in result.query_similarities:
                return result.query_similarities[node_id]
            
            # For chunks, try to find the highest similarity among its sentences
            chunk_sentences = self.kg.get_chunk_sentences(node_id)
            if chunk_sentences:
                max_similarity = 0.0
                for sentence in chunk_sentences:
                    sentence_text = sentence.sentence_text if hasattr(sentence, 'sentence_text') else str(sentence)
                    similarity = result.query_similarities.get(sentence_text, 0.0)
                    max_similarity = max(max_similarity, similarity)
                return max_similarity
        
        # Fallback: use metadata if available
        if hasattr(result, 'metadata') and result.metadata:
            extraction_metadata = result.metadata.get('extraction_metadata', {})
            if node_id in extraction_metadata:
                return extraction_metadata[node_id].get('similarity_score', 0.5)
        
        return 0.5  # Default relevance


def create_heatmap_visualization(result: RetrievalResult, query: str,
                                 knowledge_graph: KnowledgeGraph,
                                 figure_size: Tuple[int, int] = (20, 8),
                                 max_documents: int = 6,
                                 visualization_type: str = "windowed") -> plt.Figure:
    """
    Main entry point for creating 2D heatmap visualizations of algorithm results.
    Supports multiple visualization approaches for different analytical needs.

    Args:
        result: RetrievalResult from any algorithm
        query: Original query string
        knowledge_graph: The knowledge graph instance
        figure_size: Figure size as (width, height)
        max_documents: Maximum number of documents to show (for windowed/global views)
        visualization_type: Type of visualization:
            - "windowed": Original windowed approach (default, backward compatible)
            - "global": Full document view showing complete traversal patterns
            - "sequential": Reading sessions arranged chronologically

    Returns:
        Matplotlib Figure ready for display or saving
    """
    visualizer = KnowledgeGraphMatplotlibVisualizer(knowledge_graph, figure_size)
    
    if visualization_type == "global":
        return visualizer.create_global_visualization(result, query, max_documents)
    elif visualization_type == "sequential":
        return visualizer.create_sequential_window_visualization(result, query)
    else:  # Default to windowed for backward compatibility
        return visualizer.visualize_retrieval_result(result, query, max_documents)


def create_global_visualization(result: RetrievalResult, query: str,
                              knowledge_graph: KnowledgeGraph,
                              figure_size: Tuple[int, int] = (20, 8),
                              max_documents: int = 6) -> plt.Figure:
    """
    Create global visualization showing full documents with complete traversal paths.
    
    Provides strategic overview by displaying entire document architecture and
    how algorithms navigate across the complete document landscape.
    
    Args:
        result: RetrievalResult from any algorithm
        query: Original query string
        knowledge_graph: The knowledge graph instance
        figure_size: Figure size as (width, height)
        max_documents: Maximum number of documents to show
        
    Returns:
        Matplotlib Figure showing global document structure and traversal
    """
    visualizer = KnowledgeGraphMatplotlibVisualizer(knowledge_graph, figure_size)
    return visualizer.create_global_visualization(result, query, max_documents)


def create_sequential_visualization(result: RetrievalResult, query: str,
                                  knowledge_graph: KnowledgeGraph,
                                  figure_size: Tuple[int, int] = (20, 8)) -> plt.Figure:
    """
    Create sequential window visualization showing reading sessions chronologically.
    
    Provides tactical analysis by displaying individual reading sessions as
    separate panels, arranged left-to-right in temporal order.
    
    Args:
        result: RetrievalResult from any algorithm
        query: Original query string
        knowledge_graph: The knowledge graph instance
        figure_size: Figure size as (width, height)
        
    Returns:
        Matplotlib Figure showing sequential reading sessions
    """
    visualizer = KnowledgeGraphMatplotlibVisualizer(knowledge_graph, figure_size)
    return visualizer.create_sequential_window_visualization(result, query)


# Example usage function
def example_usage():
    """Example of how to use the visualizer"""

    print("Example usage:")
    print("from utils.matplotlib_visualizer import create_heatmap_visualization")
    print("")
    print("# After running an algorithm:")
    print("result = retrieval_orchestrator.retrieve(query, 'kg_traversal')")
    print("fig = create_heatmap_visualization(")
    print("    result=result,")
    print("    query=query,")
    print("    knowledge_graph=kg,")
    print("    figure_size=(20, 8),")
    print("    max_documents=6")
    print(")")
    print("plt.tight_layout()")
    print("plt.show()  # Display the plot")
    print("# fig.savefig('traversal_heatmap.png', dpi=300, bbox_inches='tight')  # Save to file")


if __name__ == "__main__":
    example_usage()
