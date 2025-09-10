#!/usr/bin/env python3
"""
Knowledge Graph Traversal 2D Heatmap Visualizer with Matplotlib
==============================================================

Creates publication-ready 2D heatmap visualizations showing semantic similarity
matrices as "chess boards" with traversal paths drawn between chunks.
Handles early stopping elegantly by coloring final chunks specially.
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

    def visualize_retrieval_result(self, result: RetrievalResult, query: str,
                                   max_documents: int = 6) -> plt.Figure:
        """
        Create 2D heatmap visualization of algorithm traversal results.

        Args:
            result: RetrievalResult from any of the four algorithms
            query: Original query string
            max_documents: Maximum number of documents to show side-by-side

        Returns:
            Matplotlib Figure ready for display or saving
        """
        print(f"Creating 2D heatmap visualization for {result.algorithm_name}")

        # Extract traversal information
        traversal_steps = self._extract_traversal_steps(result)

        if not traversal_steps:
            print("No traversal steps found - creating basic visualization")
            return self._create_basic_visualization(result, query)

        # Get documents involved in traversal
        involved_docs = self._get_involved_documents(traversal_steps)
        involved_docs = involved_docs[:max_documents]  # Limit for visualization clarity

        print(f"Visualizing traversal across {len(involved_docs)} documents")

        # Build similarity matrices for each document
        doc_heatmap_infos = self._build_document_heatmaps(involved_docs)

        if not doc_heatmap_infos:
            print("Could not build heatmaps - falling back to basic visualization")
            return self._create_basic_visualization(result, query)

        # Create the figure with heatmaps
        fig = self._create_heatmap_figure(doc_heatmap_infos, result, query)

        # Draw traversal path
        self._draw_traversal_path(fig, traversal_steps, doc_heatmap_infos)

        return fig

    def _extract_traversal_steps(self, result: RetrievalResult) -> List[TraversalStep]:
        """Extract traversal steps from the result"""
        steps = []

        if not result.traversal_path or not result.traversal_path.nodes:
            return steps

        path = result.traversal_path

        # Check for early stopping - look at metadata
        early_stop_triggered = False
        if hasattr(result, 'metadata') and result.metadata:
            early_stop_triggered = result.metadata.get('early_stop_triggered', False)

        for i, node_id in enumerate(path.nodes):
            granularity = path.granularity_levels[i] if i < len(path.granularity_levels) else GranularityLevel.CHUNK
            connection_type = path.connection_types[i - 1] if i > 0 and i - 1 < len(
                path.connection_types) else ConnectionType.RAW_SIMILARITY

            # Only include chunk-level nodes for heatmap visualization
            if granularity == GranularityLevel.CHUNK:
                # Get document ID for this chunk
                doc_id = self._get_chunk_document(node_id)

                # Calculate relevance score
                relevance_score = 0.7  # Default
                if hasattr(result, 'query_similarities') and result.query_similarities:
                    # Try to find a sentence from this chunk in the similarities
                    chunk_sentences = self.kg.get_chunk_sentences(node_id)
                    if chunk_sentences:
                        sentence_similarities = [
                            result.query_similarities.get(sent.sentence_text, 0.5)
                            for sent in chunk_sentences
                        ]
                        relevance_score = max(sentence_similarities) if sentence_similarities else 0.7

                # Check if this is the early stop point (last step and early stop triggered)
                is_early_stop_point = (early_stop_triggered and i == len(path.nodes) - 1)

                steps.append(TraversalStep(
                    step_number=i,
                    chunk_id=node_id,
                    doc_id=doc_id,
                    connection_type=connection_type.value,
                    relevance_score=relevance_score,
                    is_early_stop_point=is_early_stop_point
                ))

        return steps

    def _get_involved_documents(self, steps: List[TraversalStep]) -> List[str]:
        """Get list of documents involved in traversal, in order of first appearance"""
        seen_docs = set()
        doc_order = []

        for step in steps:
            if step.doc_id not in seen_docs:
                doc_order.append(step.doc_id)
                seen_docs.add(step.doc_id)

        return doc_order

    def _build_document_heatmaps(self, doc_ids: List[str]) -> List[DocumentHeatmapInfo]:
        """Build similarity matrices for each document"""
        heatmap_infos = []

        for doc_id in doc_ids:
            # Get all chunks for this document
            doc_chunks = [chunk_id for chunk_id, chunk_obj in self.kg.chunks.items()
                          if self._get_chunk_document(chunk_id) == doc_id]

            if len(doc_chunks) < 2:
                print(f"Document {doc_id} has fewer than 2 chunks, skipping")
                continue

            # Get embeddings for chunks in this document
            chunk_embeddings = []
            valid_chunks = []

            for chunk_id in doc_chunks:
                chunk_obj = self.kg.chunks.get(chunk_id)
                if chunk_obj and hasattr(chunk_obj, 'embedding'):
                    chunk_embeddings.append(chunk_obj.embedding)
                    valid_chunks.append(chunk_id)

            if len(valid_chunks) < 2:
                print(f"Document {doc_id} has fewer than 2 valid chunks with embeddings")
                continue

            # Build similarity matrix
            embeddings_array = np.array(chunk_embeddings)
            similarity_matrix = np.dot(embeddings_array, embeddings_array.T)

            # Create mapping from chunk ID to matrix index
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

    def _create_heatmap_figure(self, heatmap_infos: List[DocumentHeatmapInfo],
                               result: RetrievalResult, query: str) -> plt.Figure:
        """Create the figure with side-by-side heatmaps"""

        num_docs = len(heatmap_infos)

        # Calculate figure width based on number of documents
        fig_width = max(self.figure_size[0], num_docs * 4)

        fig = plt.figure(figsize=(fig_width, self.figure_size[1]), facecolor='white', dpi=self.dpi)

        # Create grid layout - small space at top for colorbar
        gs = gridspec.GridSpec(2, num_docs, figure=fig,
                               height_ratios=[0.08, 1],
                               hspace=0.05, wspace=0.2)

        # Create horizontal colorbar at top
        cbar_ax = fig.add_subplot(gs[0, :])

        # Create heatmaps
        for i, heatmap_info in enumerate(heatmap_infos):
            ax = fig.add_subplot(gs[1, i])
            heatmap_info.ax = ax
            heatmap_info.bbox = ax.get_position()

            # Create heatmap using same colors as your benchmark
            im = ax.imshow(heatmap_info.similarity_matrix,
                           cmap='RdYlBu_r',  # Red-Yellow-Blue colormap
                           aspect='equal',
                           vmin=0, vmax=1,
                           interpolation='nearest')

            # Set title and labels
            ax.set_title(heatmap_info.title, fontsize=14, fontweight='bold', pad=15)
            ax.set_xlabel('Chunk Index', fontsize=12)
            if i == 0:  # Only leftmost plot gets y-label
                ax.set_ylabel('Chunk Index', fontsize=12)

            # Set ticks
            n_chunks = len(heatmap_info.chunks_in_doc)
            if n_chunks <= 10:
                ax.set_xticks(range(n_chunks))
                ax.set_yticks(range(n_chunks))
                ax.set_xticklabels(range(n_chunks))
                ax.set_yticklabels(range(n_chunks))
            else:
                # For larger matrices, show fewer ticks
                tick_positions = np.linspace(0, n_chunks - 1, min(10, n_chunks), dtype=int)
                ax.set_xticks(tick_positions)
                ax.set_yticks(tick_positions)
                ax.set_xticklabels(tick_positions)
                ax.set_yticklabels(tick_positions)

        # Add colorbar
        if heatmap_infos:
            cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
            cbar.set_label('Chunk Similarity Score', fontsize=12, fontweight='bold')
            cbar_ax.xaxis.set_label_position('top')

        # Add title
        title = (f"{result.algorithm_name} Traversal Path Visualization\n"
                 f"Query: '{query[:80]}...' | "
                 f"Retrieved: {len(result.retrieved_content)} sentences | "
                 f"Score: {result.final_score:.3f}")
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.95)

        return fig

    def _draw_traversal_path(self, fig: plt.Figure, steps: List[TraversalStep],
                             heatmap_infos: List[DocumentHeatmapInfo]):
        """Draw the traversal path on the heatmaps"""

        if len(steps) < 1:
            return

        print(f"Drawing traversal path for {len(steps)} steps")

        # Create mapping from doc_id to heatmap_info
        doc_to_heatmap = {info.doc_id: info for info in heatmap_infos}

        # Draw step markers
        for step in steps:
            heatmap_info = doc_to_heatmap.get(step.doc_id)
            if not heatmap_info or step.chunk_id not in heatmap_info.chunk_to_matrix_idx:
                continue

            matrix_idx = heatmap_info.chunk_to_matrix_idx[step.chunk_id]
            ax = heatmap_info.ax

            # Determine marker properties
            if step.step_number == 0:
                # Anchor point
                marker_color = 'gold'
                marker_size = 400  # Larger for anchor
                edge_color = 'black'
                edge_width = 3
                marker_symbol = 'star'
            elif step.is_early_stop_point:
                # Early stopping point - special color
                marker_color = 'red'
                marker_size = 350
                edge_color = 'darkred'
                edge_width = 3
                marker_symbol = 'o'
            else:
                # Regular traversal step
                # Color based on relevance score
                relevance = max(0, min(1, step.relevance_score))
                green_intensity = 0.3 + (relevance * 0.7)  # 0.3 to 1.0
                marker_color = (1.0 - green_intensity, 1.0, 1.0 - green_intensity)  # Light green to dark green
                marker_size = 200 + (relevance * 200)  # Size 200-400 based on relevance
                edge_color = 'darkgreen'
                edge_width = 2
                marker_symbol = 'o'

            # Draw marker on diagonal (chunk similarity to itself)
            if marker_symbol == 'star':
                ax.scatter([matrix_idx], [matrix_idx], s=marker_size, marker='*',
                           c=[marker_color], edgecolors=edge_color, linewidths=edge_width, zorder=10)
            else:
                ax.scatter([matrix_idx], [matrix_idx], s=marker_size,
                           c=[marker_color], edgecolors=edge_color, linewidths=edge_width, zorder=10)

            # Add step number text
            ax.text(matrix_idx, matrix_idx, str(step.step_number),
                    ha='center', va='center', fontsize=11, fontweight='bold',
                    color='black', zorder=11)

        # Draw connections between steps
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

            # Determine line properties based on connection type
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
                # Cross-document connection using ConnectionPatch
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
                print(f"Drew cross-document connection: {current_step.doc_id} -> {next_step.doc_id}")
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

        # Add legend
        self._add_legend(fig)

    def _add_legend(self, fig: plt.Figure):
        """Add legend explaining the visualization elements"""

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

        # Create a simple text display of results
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
            metadata_text = "\n".join([f"{k}: {v}" for k, v in result.metadata.items()])
            ax.text(0.5, 0.2, f"Metadata:\n{metadata_text}",
                    ha='center', va='center', fontsize=10)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{result.algorithm_name} Results Summary", fontsize=18, fontweight='bold')
        ax.axis('off')

        return fig

    def _get_chunk_document(self, chunk_id: str) -> str:
        """Get document ID for a chunk"""
        chunk_obj = self.kg.chunks.get(chunk_id)
        if chunk_obj and hasattr(chunk_obj, 'source_document'):
            return chunk_obj.source_document
        elif chunk_obj and hasattr(chunk_obj, 'document_id'):
            return chunk_obj.document_id
        else:
            # Fallback: extract from chunk ID if it contains document info
            if '_' in chunk_id:
                parts = chunk_id.split('_')
                return '_'.join(parts[:-3]) if len(parts) > 3 else parts[0]
            return "unknown"


def create_heatmap_visualization(result: RetrievalResult, query: str,
                                 knowledge_graph: KnowledgeGraph,
                                 figure_size: Tuple[int, int] = (20, 8),
                                 max_documents: int = 6) -> plt.Figure:
    """
    Main entry point for creating 2D heatmap visualizations of algorithm results.

    Args:
        result: RetrievalResult from any algorithm
        query: Original query string
        knowledge_graph: The knowledge graph instance
        figure_size: Figure size as (width, height)
        max_documents: Maximum number of documents to show side-by-side

    Returns:
        Matplotlib Figure ready for display or saving
    """
    visualizer = KnowledgeGraphMatplotlibVisualizer(knowledge_graph, figure_size)
    return visualizer.visualize_retrieval_result(result, query, max_documents)


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