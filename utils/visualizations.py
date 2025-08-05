"""
Visualization Utilities for Semantic Graph RAG
==============================================

Combined 2D matplotlib and 3D plotly visualizations for semantic graph traversal.
Extracted and modularized from matplotlib_vis.py and plotly_vis.py.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from .rag_system import TraversalStep, SemanticGraphRAG


class SemanticGraphVisualizer:
    """
    Combined visualizer for both 2D and 3D semantic graph traversal visualizations
    """

    def __init__(self, figure_size_2d: Tuple[int, int] = (24, 8),
                 figure_size_3d: Tuple[int, int] = (12, 10),
                 dpi: int = 150):
        self.figure_size_2d = figure_size_2d
        self.figure_size_3d = figure_size_3d
        self.dpi = dpi

    def create_2d_visualization(self, rag_system: SemanticGraphRAG,
                                question: str,
                                traversal_steps: List[TraversalStep],
                                max_steps: int = 15,
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Create 2D matplotlib heatmap visualization of semantic graph traversal

        Args:
            rag_system: The RAG system with similarity matrices
            question: The query question
            traversal_steps: List of traversal steps
            max_steps: Maximum steps to show in visualization
            save_path: Optional path to save the figure

        Returns:
            Matplotlib figure
        """
        if not rag_system.similarity_matrices:
            print("⚠️ No similarity matrices available for visualization")
            return plt.Figure()

        # Get traversed documents in order
        traversed_doc_ids = []
        seen_docs = set()
        for step in traversal_steps:
            doc_id = step.sentence_info.doc_id
            if doc_id not in seen_docs:
                traversed_doc_ids.append(doc_id)
                seen_docs.add(doc_id)

        num_docs = len(traversed_doc_ids)
        print(f"🎨 Creating 2D visualization for {num_docs} documents")

        # Create figure
        fig = plt.figure(figsize=(6 * num_docs, 8), facecolor='white', dpi=self.dpi)

        # Create grid layout
        gs = gridspec.GridSpec(2, num_docs, figure=fig,
                               height_ratios=[0.08, 1],
                               hspace=0.05, wspace=0.15)

        # Colorbar at top
        cbar_ax = fig.add_subplot(gs[0, :])

        axes = []
        heatmap_extents = []

        # Create heatmaps for each traversed document
        for plot_position, original_doc_id in enumerate(traversed_doc_ids):
            ax = fig.add_subplot(gs[1, plot_position])
            axes.append(ax)

            # Get similarity matrix
            similarity_matrix = rag_system.similarity_matrices[original_doc_id]

            # Create heatmap
            im = ax.imshow(similarity_matrix,
                           cmap='RdYlBu_r',
                           aspect='equal',
                           vmin=0, vmax=1,
                           interpolation='nearest')

            # Set title
            context_title = f"Document {plot_position + 1}"
            if original_doc_id < len(rag_system.selected_contexts):
                context_id = rag_system.selected_contexts[original_doc_id]['id'][-8:]
                context_title += f"\n({context_id})"

            ax.set_title(context_title, fontsize=14, fontweight='bold', pad=15)
            ax.set_xlabel('Sentence Index', fontsize=12)
            ax.set_ylabel('Sentence Index', fontsize=12)

            # Set ticks
            n_sentences = similarity_matrix.shape[0]
            max_ticks = min(n_sentences, 10)
            if n_sentences <= 10:
                ax.set_xticks(range(n_sentences))
                ax.set_yticks(range(n_sentences))
            else:
                tick_positions = np.linspace(0, n_sentences - 1, max_ticks, dtype=int)
                ax.set_xticks(tick_positions)
                ax.set_yticks(tick_positions)

            # Store extent information
            bbox = ax.get_position()
            heatmap_extents.append({
                'ax': ax,
                'bbox': bbox,
                'original_doc_id': original_doc_id,
                'plot_position': plot_position,
                'matrix_size': n_sentences
            })

        # Add colorbar
        cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Similarity Score', fontsize=12, fontweight='bold')
        cbar_ax.xaxis.set_label_position('top')

        # Draw traversal path
        self._draw_2d_traversal_path(fig, traversal_steps, heatmap_extents,
                                     traversed_doc_ids, rag_system, max_steps)

        # Add title
        fig.suptitle(f"Semantic Graph Traversal: {question[:60]}...",
                     fontsize=16, fontweight='bold', y=0.95)

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def create_3d_visualization(self, question: str,
                                traversal_steps: List[TraversalStep],
                                method: str = "pca",
                                max_steps: int = 20) -> go.Figure:
        """
        Create 3D plotly visualization of semantic graph traversal

        Args:
            question: The query question
            traversal_steps: List of traversal steps
            method: Dimensionality reduction method ("pca" or "tsne")
            max_steps: Maximum steps to show

        Returns:
            Plotly figure
        """
        if len(traversal_steps) < 2:
            print("⚠️ Not enough traversal steps for 3D visualization")
            return go.Figure()

        # Limit steps for visualization
        steps_to_show = traversal_steps[:max_steps]

        # Get embeddings and reduce dimensionality
        all_embeddings = np.array([step.sentence_info.embedding for step in steps_to_show])

        if method == "pca":
            reducer = PCA(n_components=3)
            coords_3d = reducer.fit_transform(all_embeddings)
        else:  # t-SNE
            perplexity = min(30, len(all_embeddings) - 1)
            if perplexity < 5:
                perplexity = max(1, len(all_embeddings) - 1)
            reducer = TSNE(n_components=3, random_state=42, perplexity=perplexity)
            coords_3d = reducer.fit_transform(all_embeddings)

        # Prepare plotting data
        plot_data = []
        for i, step in enumerate(steps_to_show):
            plot_data.append({
                'x': coords_3d[i, 0],
                'y': coords_3d[i, 1],
                'z': coords_3d[i, 2],
                'doc_id': step.sentence_info.doc_id,
                'paragraph_id': step.sentence_info.paragraph_id,
                'connection_type': step.connection_type,
                'relevance_score': step.relevance_score,
                'distance_from_anchor': step.distance_from_anchor,
                'context_id': step.sentence_info.source_context_id,
                'text_preview': step.sentence_info.text[:100] + "..." if len(
                    step.sentence_info.text) > 100 else step.sentence_info.text,
                'step_number': step.step_number
            })

        df = pd.DataFrame(plot_data)

        # Color mapping
        connection_type_colors = {
            'anchor': 'red',
            'same_paragraph': 'blue',
            'neighboring_paragraph': 'green',
            'distant_paragraph': 'orange',
            'cross_document': 'purple'
        }

        # Create 3D scatter plot
        fig = go.Figure()

        for connection_type in df['connection_type'].unique():
            mask = df['connection_type'] == connection_type
            subset = df[mask]

            fig.add_trace(go.Scatter3d(
                x=subset['x'],
                y=subset['y'],
                z=subset['z'],
                mode='markers+text',
                marker=dict(
                    size=subset['relevance_score'] * 20 + 8,
                    color=connection_type_colors.get(connection_type, 'gray'),
                    opacity=0.8,
                    line=dict(width=2, color='black')
                ),
                text=subset['step_number'],
                textposition="middle center",
                name=connection_type.replace('_', ' ').title(),
                hovertemplate=(
                        "<b>Step %{text}</b><br>" +
                        "Document: %{customdata[0]}<br>" +
                        "Paragraph: %{customdata[1]}<br>" +
                        "Relevance: %{customdata[2]:.3f}<br>" +
                        "Connection: %{customdata[3]}<br>" +
                        "Text: %{customdata[4]}<br>" +
                        "<extra></extra>"
                ),
                customdata=list(zip(subset['doc_id'], subset['paragraph_id'],
                                    subset['relevance_score'], subset['connection_type'],
                                    subset['text_preview']))
            ))

        # Add traversal path lines
        for i in range(len(coords_3d) - 1):
            current_step = steps_to_show[i]
            next_step = steps_to_show[i + 1]

            line_color = connection_type_colors.get(next_step.connection_type, 'gray')
            line_width = 4 if next_step.connection_type == 'cross_document' else 2

            fig.add_trace(go.Scatter3d(
                x=[coords_3d[i, 0], coords_3d[i + 1, 0]],
                y=[coords_3d[i, 1], coords_3d[i + 1, 1]],
                z=[coords_3d[i, 2], coords_3d[i + 1, 2]],
                mode='lines',
                line=dict(color=line_color, width=line_width),
                showlegend=False,
                hovertemplate=f"Step {current_step.step_number} → Step {next_step.step_number}<extra></extra>"
            ))

        # Update layout
        fig.update_layout(
            title=f"3D Semantic Graph Traversal<br>{question[:80]}...",
            scene=dict(
                xaxis_title=f"Dimension 1 ({method.upper()})",
                yaxis_title=f"Dimension 2 ({method.upper()})",
                zaxis_title=f"Dimension 3 ({method.upper()})",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            height=800,
            font=dict(size=12)
        )

        return fig

    def create_analysis_charts(self, analysis: Dict) -> go.Figure:
        """
        Create pattern analysis charts

        Args:
            analysis: Analysis dictionary from RAG system

        Returns:
            Plotly figure with analysis charts
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Connection Type Distribution', 'Document Distribution',
                            'Traversal Summary', 'Key Metrics'),
            specs=[[{'type': 'pie'}, {'type': 'bar'}],
                   [{'type': 'table'}, {'type': 'indicator'}]]
        )

        # Connection type pie chart
        conn_types = list(analysis['connection_type_percentages'].keys())
        conn_percentages = list(analysis['connection_type_percentages'].values())

        fig.add_trace(go.Pie(
            labels=[t.replace('_', ' ').title() for t in conn_types],
            values=conn_percentages,
            name="Connection Types"
        ), row=1, col=1)

        # Document distribution bar chart
        docs = list(analysis['document_distribution'].keys())
        doc_counts = list(analysis['document_distribution'].values())

        fig.add_trace(go.Bar(
            x=[f"Doc {d + 1}" for d in docs],
            y=doc_counts,
            name="Document Distribution"
        ), row=1, col=2)

        # Summary table
        summary_data = [
            ["Total Steps", analysis['total_steps']],
            ["Cross-document Rate", f"{analysis['cross_document_rate']:.1f}%"],
            ["Number of Contexts", analysis['num_contexts']],
        ]

        fig.add_trace(go.Table(
            header=dict(values=["Metric", "Value"]),
            cells=dict(values=list(zip(*summary_data)))
        ), row=2, col=1)

        # Key metric indicator
        fig.add_trace(go.Indicator(
            mode="number+gauge",
            value=analysis['cross_document_rate'],
            title={"text": "Cross-Document<br>Traversal Rate %"},
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': "darkblue"},
                   'steps': [{'range': [0, 25], 'color': "lightgray"},
                             {'range': [25, 50], 'color': "gray"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                                 'thickness': 0.75, 'value': 75}}
        ), row=2, col=2)

        fig.update_layout(height=800, showlegend=False,
                          title_text=f"Traversal Analysis: {analysis.get('question', 'Unknown')[:60]}...")

        return fig

    def _draw_2d_traversal_path(self, fig: plt.Figure, traversal_steps: List[TraversalStep],
                                heatmap_extents: List[Dict], doc_traversal_order: List[int],
                                rag_system: SemanticGraphRAG, max_steps: int = 15):
        """Draw traversal path on 2D heatmaps"""
        if len(traversal_steps) < 2:
            return

        # Create mapping from doc_id to plot position
        doc_id_to_plot_position = {doc_id: i for i, doc_id in enumerate(doc_traversal_order)}

        # Group steps by document for sentence indexing
        doc_sentence_counts = defaultdict(int)
        for sentence_info in rag_system.sentences_info:
            doc_sentence_counts[sentence_info.doc_id] += 1

        # Map global sentence IDs to local document sentence IDs
        doc_sentence_mapping = {}
        for doc_id in doc_sentence_counts:
            doc_sentences = [s for s in rag_system.sentences_info if s.doc_id == doc_id]
            doc_sentence_mapping[doc_id] = {s.sentence_id: i for i, s in enumerate(doc_sentences)}

        # Draw step markers
        drawn_steps = []
        steps_shown = 0

        for step in traversal_steps:
            if steps_shown >= max_steps:
                break

            original_doc_id = step.sentence_info.doc_id
            global_sent_id = step.sentence_info.sentence_id

            # Skip if document not in plot
            if original_doc_id not in doc_id_to_plot_position:
                continue

            # Get local sentence ID
            if (original_doc_id in doc_sentence_mapping and
                    global_sent_id in doc_sentence_mapping[original_doc_id]):
                local_sent_id = doc_sentence_mapping[original_doc_id][global_sent_id]
            else:
                continue

            # Find corresponding heatmap
            heatmap_info = None
            for extent in heatmap_extents:
                if extent['original_doc_id'] == original_doc_id:
                    heatmap_info = extent
                    break

            if not heatmap_info or local_sent_id >= heatmap_info['matrix_size']:
                continue

            ax = heatmap_info['ax']

            # Draw step marker on diagonal
            row, col = local_sent_id, local_sent_id

            if step.connection_type == 'anchor':
                marker_color = 'gold'
                marker_size = 15
            else:
                relevance = max(0, min(1, step.relevance_score))
                marker_color = plt.cm.Greens(0.3 + relevance * 0.7)
                marker_size = 8 + (relevance * 7)

            ax.scatter([col], [row], s=marker_size ** 2, c=[marker_color],
                       edgecolors='black', linewidths=1, zorder=10)

            # Add step number
            ax.text(col, row, str(steps_shown),
                    ha='center', va='center',
                    fontsize=9, fontweight='bold',
                    color='black', zorder=11)

            drawn_steps.append((step, original_doc_id, row, col, ax))
            steps_shown += 1

        # Draw connections between steps
        for i in range(len(drawn_steps) - 1):
            current_step, current_doc, current_row, current_col, current_ax = drawn_steps[i]
            next_step, next_doc, next_row, next_col, next_ax = drawn_steps[i + 1]

            is_cross_document = (current_doc != next_doc)

            if is_cross_document:
                # Cross-document connection
                from matplotlib.patches import ConnectionPatch
                conn = ConnectionPatch(
                    xyA=(current_col, current_row), coordsA='data', axesA=current_ax,
                    xyB=(next_col, next_row), coordsB='data', axesB=next_ax,
                    arrowstyle='-',
                    linestyle='--',
                    linewidth=2,
                    color='red',
                    alpha=0.8,
                    zorder=5
                )
                fig.add_artist(conn)
            else:
                # Same document connection
                current_ax.plot([current_col, next_col], [current_row, next_row],
                                color='orange', linestyle='-',
                                linewidth=1.5, alpha=0.7, zorder=5)


# Convenience functions for easy notebook usage
def create_2d_visualization(rag_system: SemanticGraphRAG,
                            question: str,
                            traversal_steps: List[TraversalStep],
                            save_path: Optional[str] = None) -> plt.Figure:
    """Quick function to create 2D visualization"""
    visualizer = SemanticGraphVisualizer()
    return visualizer.create_2d_visualization(rag_system, question, traversal_steps, save_path=save_path)


def create_3d_visualization(question: str,
                            traversal_steps: List[TraversalStep],
                            method: str = "pca") -> go.Figure:
    """Quick function to create 3D visualization"""
    visualizer = SemanticGraphVisualizer()
    return visualizer.create_3d_visualization(question, traversal_steps, method)


def create_analysis_charts(analysis: Dict) -> go.Figure:
    """Quick function to create analysis charts"""
    visualizer = SemanticGraphVisualizer()
    return visualizer.create_analysis_charts(analysis)