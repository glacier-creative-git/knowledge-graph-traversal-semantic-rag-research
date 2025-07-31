"""
Matrix Visualization Module
==========================

This module provides visualization tools for similarity matrices and enhancement data.
Based on the research visualization techniques for semantic chunking analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class MatrixVisualizer:
    """
    Comprehensive visualization toolkit for similarity matrices and chunking analysis.
    """

    def __init__(self, style: str = "seaborn-v0_8"):
        """
        Initialize the visualizer with a specified plotting style.

        Args:
            style: Matplotlib style to use for plots
        """
        plt.style.use(style)
        self.figure_size = (12, 10)
        self.dpi = 300

    def plot_similarity_matrix(self, matrix: np.ndarray,
                               title: str = "Similarity Matrix",
                               sentences: Optional[List[str]] = None,
                               save_path: Optional[str] = None) -> None:
        """
        Create a heatmap visualization of a similarity matrix.

        Args:
            matrix: Similarity matrix to visualize
            title: Title for the plot
            sentences: Optional list of sentences for hover text
            save_path: Optional path to save the figure
        """
        plt.figure(figsize=self.figure_size)

        # Create the heatmap
        sns.heatmap(matrix,
                    annot=False,
                    cmap='viridis',
                    square=True,
                    cbar_kws={'label': 'Similarity Score'},
                    linewidths=0.1)

        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Sentence Index', fontsize=12)
        plt.ylabel('Sentence Index', fontsize=12)

        # Add sentence count info
        n_sentences = matrix.shape[0]
        plt.figtext(0.02, 0.02, f'Document: {n_sentences} sentences',
                    fontsize=10, style='italic')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        plt.show()

    def plot_enhancement_comparison(self, original_matrix: np.ndarray,
                                    enhanced_matrix: np.ndarray,
                                    enhancement_data: Dict[str, np.ndarray],
                                    save_path: Optional[str] = None) -> None:
        """
        Create a 2x2 comparison plot showing the enhancement process.

        Args:
            original_matrix: Original similarity matrix
            enhanced_matrix: Enhanced similarity matrix
            enhancement_data: Dictionary containing enhancement arrays
            save_path: Optional path to save the figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        # Original matrix
        sns.heatmap(original_matrix, ax=axes[0, 0], cmap='viridis',
                    square=True, cbar=True, annot=False)
        axes[0, 0].set_title('Original Similarity Matrix', fontweight='bold')

        # Enhanced matrix
        sns.heatmap(enhanced_matrix, ax=axes[0, 1], cmap='viridis',
                    square=True, cbar=True, annot=False)
        axes[0, 1].set_title('Enhanced Similarity Matrix', fontweight='bold')

        # Edge magnitude
        if 'edge_magnitude' in enhancement_data:
            sns.heatmap(enhancement_data['edge_magnitude'], ax=axes[1, 0],
                        cmap='hot', square=True, cbar=True, annot=False)
            axes[1, 0].set_title('Edge Magnitude\n(Boundary Detection)', fontweight='bold')

        # Block measure
        if 'block_measure' in enhancement_data:
            sns.heatmap(enhancement_data['block_measure'], ax=axes[1, 1],
                        cmap='plasma', square=True, cbar=True, annot=False)
            axes[1, 1].set_title('Block Measure\n(Coherence Detection)', fontweight='bold')

        plt.suptitle('Similarity Matrix Enhancement Process',
                     fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        plt.show()

    def plot_interactive_similarity_matrix(self, matrix: np.ndarray,
                                           sentences: List[str],
                                           title: str = "Interactive Similarity Matrix") -> None:
        """
        Create an interactive Plotly heatmap with sentence hover information.

        Args:
            matrix: Similarity matrix to visualize
            sentences: List of sentences for hover text
            title: Title for the plot
        """
        # Truncate sentences for hover text
        hover_sentences = [s[:100] + "..." if len(s) > 100 else s for s in sentences]

        # Create hover text matrix
        hover_text = []
        for i in range(len(sentences)):
            hover_row = []
            for j in range(len(sentences)):
                hover_row.append(
                    f"Sentence {i + 1} ↔ Sentence {j + 1}<br>"
                    f"Similarity: {matrix[i, j]:.3f}<br><br>"
                    f"Sent {i + 1}: {hover_sentences[i]}<br>"
                    f"Sent {j + 1}: {hover_sentences[j]}"
                )
            hover_text.append(hover_row)

        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            hovertemplate='%{text}<extra></extra>',
            text=hover_text,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Similarity Score")
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Sentence Index",
            yaxis_title="Sentence Index",
            width=800,
            height=800
        )

        fig.show()

    def plot_chunking_boundaries(self, similarity_matrix: np.ndarray,
                                 chunk_boundaries: List[int],
                                 sentences: List[str],
                                 title: str = "Chunk Boundaries Visualization",
                                 save_path: Optional[str] = None) -> None:
        """
        Visualize chunk boundaries overlaid on the similarity matrix.

        Args:
            similarity_matrix: The similarity matrix
            chunk_boundaries: List of sentence indices where chunks split
            sentences: List of sentences
            title: Title for the plot
            save_path: Optional path to save the figure
        """
        plt.figure(figsize=self.figure_size)

        # Create the base heatmap
        ax = sns.heatmap(similarity_matrix,
                         cmap='viridis',
                         square=True,
                         cbar_kws={'label': 'Similarity Score'})

        # Add boundary lines
        for boundary in chunk_boundaries:
            # Vertical line
            ax.axvline(x=boundary, color='red', linewidth=3, alpha=0.7)
            # Horizontal line
            ax.axhline(y=boundary, color='red', linewidth=3, alpha=0.7)

        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Sentence Index', fontsize=12)
        plt.ylabel('Sentence Index', fontsize=12)

        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color='red', lw=3, label='Chunk Boundaries')]
        plt.legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        plt.show()

    def plot_semantic_flow(self, similarity_matrix: np.ndarray,
                           sentences: List[str],
                           save_path: Optional[str] = None) -> None:
        """
        Plot the semantic flow through a document by showing adjacent sentence similarities.

        Args:
            similarity_matrix: The similarity matrix
            sentences: List of sentences
            save_path: Optional path to save the figure
        """
        # Calculate adjacent similarities
        n_sentences = len(sentences)
        adjacent_similarities = []
        positions = []

        for i in range(n_sentences - 1):
            similarity = similarity_matrix[i, i + 1]
            adjacent_similarities.append(similarity)
            position = (i + 0.5) / n_sentences
            positions.append(position)

        plt.figure(figsize=(14, 6))

        # Plot the semantic flow
        plt.plot(positions, adjacent_similarities,
                 color='blue', linewidth=2, alpha=0.7, label='Semantic Flow')

        # Fill area under the curve
        plt.fill_between(positions, adjacent_similarities, alpha=0.3, color='blue')

        # Add markers for low points (potential boundaries)
        low_threshold = np.percentile(adjacent_similarities, 25)
        low_points = [(pos, sim) for pos, sim in zip(positions, adjacent_similarities)
                      if sim <= low_threshold]

        if low_points:
            low_x, low_y = zip(*low_points)
            plt.scatter(low_x, low_y, color='red', s=100,
                        label='Potential Boundaries', zorder=5)

        plt.title('Document Semantic Flow\n(Lower values indicate potential chunk boundaries)',
                  fontsize=14, fontweight='bold')
        plt.xlabel('Relative Position in Document', fontsize=12)
        plt.ylabel('Adjacent Sentence Similarity', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add sentence indices on x-axis
        sentence_positions = [i / n_sentences for i in range(0, n_sentences, max(1, n_sentences // 10))]
        sentence_labels = [str(int(pos * n_sentences) + 1) for pos in sentence_positions]
        plt.xticks(sentence_positions, sentence_labels)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        plt.show()

    def create_research_figure(self, matrices: Dict[str, np.ndarray],
                               titles: Dict[str, str],
                               save_path: Optional[str] = None) -> None:
        """
        Create a publication-ready figure showing multiple matrices.

        This recreates the style of your research visualization with
        Original, Enhanced, Edge Magnitude, and Block Measure matrices.

        Args:
            matrices: Dictionary of matrix_name -> matrix_array
            titles: Dictionary of matrix_name -> title_string
            save_path: Optional path to save the figure
        """
        n_matrices = len(matrices)
        cols = 2
        rows = (n_matrices + 1) // 2

        fig, axes = plt.subplots(rows, cols, figsize=(16, 8 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)

        matrix_names = list(matrices.keys())

        for idx, (name, matrix) in enumerate(matrices.items()):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col]

            # Choose colormap based on matrix type
            if 'edge' in name.lower():
                cmap = 'hot'
            elif 'block' in name.lower():
                cmap = 'plasma'
            else:
                cmap = 'viridis'

            sns.heatmap(matrix, ax=ax, cmap=cmap, square=True,
                        cbar=True, annot=False, cbar_kws={'shrink': 0.8})

            ax.set_title(titles.get(name, name), fontweight='bold', fontsize=14)

            # Add sentence count information
            n_sentences = matrix.shape[0]
            ax.text(0.02, 0.98, f'{n_sentences} sentences',
                    transform=ax.transAxes, fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                    verticalalignment='top')

        # Remove empty subplots
        for idx in range(len(matrices), rows * cols):
            row = idx // cols
            col = idx % cols
            fig.delaxes(axes[row, col])

        plt.suptitle('Similarity Matrix Analysis for Semantic Chunking',
                     fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        plt.show()


# Utility functions for quick visualizations
def quick_matrix_plot(matrix: np.ndarray, title: str = "Similarity Matrix") -> None:
    """Quick utility function to plot a matrix."""
    visualizer = MatrixVisualizer()
    visualizer.plot_similarity_matrix(matrix, title)


def quick_comparison_plot(original: np.ndarray, enhanced: np.ndarray,
                          enhancement_data: Dict[str, np.ndarray]) -> None:
    """Quick utility function to compare matrices."""
    visualizer = MatrixVisualizer()
    visualizer.plot_enhancement_comparison(original, enhanced, enhancement_data)


# Example usage for testing
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    n_sentences = 10

    # Create a sample similarity matrix with block structure
    matrix = np.random.rand(n_sentences, n_sentences)
    matrix = (matrix + matrix.T) / 2  # Make symmetric
    np.fill_diagonal(matrix, 1.0)  # Perfect self-similarity

    # Create enhancement data
    enhancement_data = {
        'edge_magnitude': np.random.rand(n_sentences, n_sentences),
        'block_measure': np.random.rand(n_sentences, n_sentences)
    }

    # Test the visualizer
    visualizer = MatrixVisualizer()

    print("Testing Matrix Visualizer...")
    visualizer.plot_similarity_matrix(matrix, "Test Similarity Matrix")
    visualizer.plot_enhancement_comparison(matrix, matrix * 0.9, enhancement_data)

    print("Matrix Visualizer module ready for use!")