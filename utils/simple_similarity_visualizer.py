#!/usr/bin/env python3
"""
Simple Similarity Matrix Visualizer
===================================

Uses the exact same visualization code as matplotlib_visualizer.py but with
just a numpy similarity matrix - no semantic similarity graph or retrieval algorithms needed.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SimpleDocumentHeatmap:
    """Simplified version of DocumentHeatmapInfo for basic similarity matrix visualization"""
    doc_id: str
    similarity_matrix: np.ndarray
    chunk_labels: List[str]  # Simple labels for chunks (e.g., "Chunk 0", "Chunk 1", etc.)
    ax: plt.Axes = None
    bbox: Any = None
    title: str = ""


def create_simple_similarity_visualization(similarity_matrix: np.ndarray,
                                         title: str = "Similarity Matrix",
                                         chunk_labels: Optional[List[str]] = None,
                                         figure_size: Tuple[int, int] = (12, 8),
                                         dpi: int = 150,
                                         document_boundaries: Optional[List[int]] = None) -> plt.Figure:
    """
    Create similarity matrix visualization using the exact same styling as matplotlib_visualizer.py

    Args:
        similarity_matrix: 2D numpy array of similarity values (should be symmetric)
        title: Title for the visualization
        chunk_labels: Optional labels for chunks. If None, will use "Chunk 0", "Chunk 1", etc.
        figure_size: Figure size as (width, height)
        dpi: DPI for the figure
        document_boundaries: Optional list of chunk indices where documents end (for boundary lines)

    Returns:
        Matplotlib Figure ready for display or saving
    """

    print(f"🎨 Creating similarity matrix visualization: {similarity_matrix.shape}")

    # Validate input
    if len(similarity_matrix.shape) != 2:
        raise ValueError("similarity_matrix must be 2D")
    if similarity_matrix.shape[0] != similarity_matrix.shape[1]:
        raise ValueError("similarity_matrix must be square")

    n_chunks = similarity_matrix.shape[0]

    # Generate chunk labels if not provided
    if chunk_labels is None:
        chunk_labels = [f"Chunk {i}" for i in range(n_chunks)]
    elif len(chunk_labels) != n_chunks:
        raise ValueError(f"chunk_labels length ({len(chunk_labels)}) must match matrix size ({n_chunks})")

    # Create DocumentHeatmapInfo object (using the same structure as the original)
    heatmap_info = SimpleDocumentHeatmap(
        doc_id="Document",
        similarity_matrix=similarity_matrix,
        chunk_labels=chunk_labels,
        title=title
    )

    # Create the figure using the same styling as matplotlib_visualizer.py
    fig = _create_simple_heatmap_figure([heatmap_info], figure_size, dpi, document_boundaries)

    print(f"✅ Similarity visualization created successfully")
    return fig


def _create_simple_heatmap_figure(heatmap_infos: List[SimpleDocumentHeatmap],
                                figure_size: Tuple[int, int],
                                dpi: int,
                                document_boundaries: Optional[List[int]] = None) -> plt.Figure:
    """
    Create the figure with heatmap - uses EXACT same styling as matplotlib_visualizer.py
    """

    num_docs = len(heatmap_infos)

    # Calculate figure width based on number of documents (same as original)
    fig_width = max(figure_size[0], num_docs * 4)

    # Set matplotlib style for publication quality (same as original)
    plt.style.use('default')

    fig = plt.figure(figsize=(fig_width, figure_size[1]), facecolor='white', dpi=dpi)

    # Create grid layout with more space for title and colorbar
    gs = gridspec.GridSpec(2, num_docs, figure=fig,
                          height_ratios=[0.08, 1],
                          hspace=0.2, wspace=0.25,
                          top=0.78, bottom=0.15)  # More space for title (0.82 -> 0.78)

    # Create horizontal colorbar at top (same as original)
    cbar_ax = fig.add_subplot(gs[0, :])

    # Create heatmaps with proper colormap (same as original)
    vmin, vmax = 0, 1
    cmap = 'RdYlBu_r'  # Same as original

    for i, heatmap_info in enumerate(heatmap_infos):
        ax = fig.add_subplot(gs[1, i])
        heatmap_info.ax = ax
        heatmap_info.bbox = ax.get_position()

        # Create heatmap (same style as original)
        im = ax.imshow(heatmap_info.similarity_matrix,
                      cmap=cmap,
                      aspect='equal',
                      vmin=vmin, vmax=vmax,
                      interpolation='nearest')

        # Set title and labels (same style as original)
        ax.set_title(heatmap_info.title, fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Chunk Index', fontsize=12)
        if i == 0:  # Only leftmost plot gets y-label
            ax.set_ylabel('Chunk Index', fontsize=12)

        # Set ticks and labels
        n_chunks = len(heatmap_info.chunk_labels)

        if n_chunks <= 15:
            # Show all ticks for smaller matrices
            ax.set_xticks(range(n_chunks))
            ax.set_yticks(range(n_chunks))
            ax.set_xticklabels(range(n_chunks), rotation=45)
            ax.set_yticklabels(range(n_chunks))
        else:
            # For larger matrices, show fewer ticks
            tick_positions = np.linspace(0, n_chunks - 1, min(10, n_chunks), dtype=int)
            ax.set_xticks(tick_positions)
            ax.set_yticks(tick_positions)
            ax.set_xticklabels(tick_positions, rotation=45)
            ax.set_yticklabels(tick_positions)

    # Add colorbar (same style as original)
    if heatmap_infos:
        cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Chunk Similarity Score', fontsize=12, fontweight='bold')
        cbar_ax.xaxis.set_label_position('top')

    # Add document boundary lines if provided
    if document_boundaries and heatmap_infos:
        ax = heatmap_infos[0].ax  # Get the main heatmap axis

        print(f"📏 Adding document boundary lines at: {document_boundaries}")

        for boundary in document_boundaries:
            # Add vertical dashed line
            ax.axvline(x=boundary - 0.5, color='white', linestyle='--', linewidth=2, alpha=0.8)
            # Add horizontal dashed line
            ax.axhline(y=boundary - 0.5, color='white', linestyle='--', linewidth=2, alpha=0.8)

    return fig


def create_multiple_similarity_visualization(similarity_matrices: List[np.ndarray],
                                           titles: List[str],
                                           figure_size: Tuple[int, int] = (20, 8),
                                           dpi: int = 150) -> plt.Figure:
    """
    Create visualization with multiple similarity matrices side-by-side

    Args:
        similarity_matrices: List of 2D numpy arrays
        titles: List of titles for each matrix
        figure_size: Figure size as (width, height)
        dpi: DPI for the figure

    Returns:
        Matplotlib Figure with multiple heatmaps
    """

    if len(similarity_matrices) != len(titles):
        raise ValueError("Number of matrices must match number of titles")

    print(f"🎨 Creating multi-matrix visualization: {len(similarity_matrices)} matrices")

    # Create heatmap info objects
    heatmap_infos = []
    for i, (matrix, title) in enumerate(zip(similarity_matrices, titles)):
        n_chunks = matrix.shape[0]
        chunk_labels = [f"Chunk {j}" for j in range(n_chunks)]

        heatmap_infos.append(SimpleDocumentHeatmap(
            doc_id=f"Doc_{i}",
            similarity_matrix=matrix,
            chunk_labels=chunk_labels,
            title=title
        ))

    # Create the figure
    fig = _create_simple_heatmap_figure(heatmap_infos, figure_size, dpi)

    print(f"✅ Multi-matrix visualization created successfully")
    return fig


# Example usage
if __name__ == "__main__":
    # Create a simple test similarity matrix
    n = 10
    np.random.seed(42)

    # Create a similarity matrix with some structure
    similarity_matrix = np.random.rand(n, n)
    similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2  # Make symmetric
    np.fill_diagonal(similarity_matrix, 1.0)  # Diagonal should be 1

    print("Example usage:")
    print("similarity_matrix shape:", similarity_matrix.shape)

    # Create visualization
    fig = create_simple_similarity_visualization(
        similarity_matrix=similarity_matrix,
        title="Example Similarity Matrix",
        figure_size=(10, 8)
    )

    plt.tight_layout()
    plt.show()