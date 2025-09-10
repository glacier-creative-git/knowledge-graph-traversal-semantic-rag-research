#!/usr/bin/env python3
"""
Knowledge Graph Traversal 3D Visualizer with Plotly
==================================================

Creates beautiful 3D visualizations of semantic graph traversal for all four algorithms.
Shows complete traversal paths including all visited nodes, not just final results.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import time

from .algos.base_algorithm import RetrievalResult
from .traversal import TraversalPath, GranularityLevel, ConnectionType
from .knowledge_graph import KnowledgeGraph


@dataclass
class VisualizationNode:
    """Node information for visualization"""
    node_id: str
    node_type: str  # 'chunk' or 'sentence'
    text: str
    embedding: np.ndarray
    step_number: int
    relevance_score: float
    connection_type: str
    distance_from_anchor: int
    document_id: str
    is_final_result: bool


class KnowledgeGraphPlotlyVisualizer:
    """Create 3D visualizations of knowledge graph traversal using Plotly"""

    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.kg = knowledge_graph

    def visualize_retrieval_result(self, result: RetrievalResult, query: str,
                                   method: str = "pca", max_nodes: int = 50,
                                   show_all_visited: bool = True) -> go.Figure:
        """
        Create 3D visualization of algorithm traversal results.

        Args:
            result: RetrievalResult from any of the four algorithms
            query: Original query string
            method: Dimensionality reduction method ('pca' or 'tsne')
            max_nodes: Maximum nodes to visualize (for performance)
            show_all_visited: If True, show all visited nodes; if False, only final results
        """
        print(f"Creating 3D visualization for {result.algorithm_name}")

        # Handle different algorithm types
        if result.algorithm_name == "BasicRetrieval":
            return self._visualize_basic_retrieval(result, query, method)
        else:
            return self._visualize_traversal_algorithm(result, query, method, max_nodes, show_all_visited)

    def _visualize_basic_retrieval(self, result: RetrievalResult, query: str, method: str) -> go.Figure:
        """Visualize BasicRetrieval results (no traversal path)"""

        # Get embeddings for final results
        nodes = []
        query_embedding = None

        # Add query as a special node
        if hasattr(self.kg, 'encode_query'):  # If KG has query encoding method
            query_embedding = self.kg.encode_query(query)
        else:
            # Fallback: use the embedding model from one of the retrieved chunks
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
            query_embedding = model.encode([query])[0]

        nodes.append(VisualizationNode(
            node_id="QUERY",
            node_type="query",
            text=query,
            embedding=query_embedding,
            step_number=-1,
            relevance_score=1.0,
            connection_type="query",
            distance_from_anchor=0,
            document_id="QUERY",
            is_final_result=False
        ))

        # Add final result sentences
        for i, sentence in enumerate(result.retrieved_content):
            # Find the sentence in the knowledge graph
            sentence_id = self._find_sentence_id(sentence)
            if sentence_id:
                sentence_obj = self.kg.sentences.get(sentence_id)
                if sentence_obj and hasattr(sentence_obj, 'embedding'):
                    nodes.append(VisualizationNode(
                        node_id=sentence_id,
                        node_type="sentence",
                        text=sentence,
                        embedding=sentence_obj.embedding,
                        step_number=i,
                        relevance_score=result.confidence_scores[i] if i < len(result.confidence_scores) else 0.5,
                        connection_type="final_result",
                        distance_from_anchor=0,
                        document_id=sentence_obj.source_document if hasattr(sentence_obj,
                                                                            'source_document') else "unknown",
                        is_final_result=True
                    ))

        return self._create_3d_plot(nodes, result, query, method)

    def _visualize_traversal_algorithm(self, result: RetrievalResult, query: str,
                                       method: str, max_nodes: int, show_all_visited: bool) -> go.Figure:
        """Visualize traversal algorithms (QueryTraversal, KGTraversal, TriangulationCentroid)"""

        nodes = []

        # Add query node
        query_embedding = self._get_query_embedding(query)
        nodes.append(VisualizationNode(
            node_id="QUERY",
            node_type="query",
            text=query,
            embedding=query_embedding,
            step_number=-1,
            relevance_score=1.0,
            connection_type="query",
            distance_from_anchor=0,
            document_id="QUERY",
            is_final_result=False
        ))

        # Extract all visited nodes from traversal path
        if show_all_visited and result.traversal_path:
            visited_nodes = self._extract_all_visited_nodes(result)
            nodes.extend(visited_nodes[:max_nodes])  # Limit for performance

        # Mark final results
        final_sentences = set(result.retrieved_content)
        for node in nodes:
            if node.text in final_sentences:
                node.is_final_result = True

        return self._create_3d_plot(nodes, result, query, method)

    def _extract_all_visited_nodes(self, result: RetrievalResult) -> List[VisualizationNode]:
        """Extract all nodes visited during traversal from the traversal path"""
        nodes = []

        if not result.traversal_path or not result.traversal_path.nodes:
            return nodes

        path = result.traversal_path

        for i, node_id in enumerate(path.nodes):
            # Determine if this is a chunk or sentence based on granularity
            granularity = path.granularity_levels[i] if i < len(path.granularity_levels) else GranularityLevel.CHUNK
            connection_type = path.connection_types[i - 1] if i > 0 and i - 1 < len(
                path.connection_types) else ConnectionType.RAW_SIMILARITY

            # Get node information from knowledge graph
            if granularity == GranularityLevel.SENTENCE:
                sentence_obj = self.kg.sentences.get(node_id)
                if sentence_obj and hasattr(sentence_obj, 'embedding'):
                    # Calculate relevance score if available
                    relevance_score = 0.5  # Default
                    if hasattr(result, 'query_similarities') and result.query_similarities:
                        relevance_score = result.query_similarities.get(sentence_obj.sentence_text, 0.5)

                    nodes.append(VisualizationNode(
                        node_id=node_id,
                        node_type="sentence",
                        text=sentence_obj.sentence_text,
                        embedding=sentence_obj.embedding,
                        step_number=i,
                        relevance_score=relevance_score,
                        connection_type=connection_type.value,
                        distance_from_anchor=i,
                        document_id=sentence_obj.source_document if hasattr(sentence_obj,
                                                                            'source_document') else "unknown",
                        is_final_result=False
                    ))
            else:  # CHUNK
                chunk_obj = self.kg.chunks.get(node_id)
                if chunk_obj and hasattr(chunk_obj, 'embedding'):
                    # For chunks, we'll show them but they represent extraction points
                    nodes.append(VisualizationNode(
                        node_id=node_id,
                        node_type="chunk",
                        text=chunk_obj.chunk_text[:100] + "..." if len(
                            chunk_obj.chunk_text) > 100 else chunk_obj.chunk_text,
                        embedding=chunk_obj.embedding,
                        step_number=i,
                        relevance_score=0.7,  # Chunks get medium relevance
                        connection_type=connection_type.value,
                        distance_from_anchor=i,
                        document_id=chunk_obj.source_document if hasattr(chunk_obj, 'source_document') else "unknown",
                        is_final_result=False
                    ))

        return nodes

    def _create_3d_plot(self, nodes: List[VisualizationNode], result: RetrievalResult,
                        query: str, method: str) -> go.Figure:
        """Create the 3D scatter plot"""

        if len(nodes) < 2:
            print("Not enough nodes for visualization")
            return go.Figure()

        # Get embeddings and reduce dimensionality
        embeddings = np.array([node.embedding for node in nodes])

        if method == "pca":
            reducer = PCA(n_components=3)
            coords_3d = reducer.fit_transform(embeddings)
            explained_variance = reducer.explained_variance_ratio_
            subtitle = f"PCA (explained variance: {explained_variance.sum():.1%})"
        else:  # t-SNE
            perplexity = min(30, len(embeddings) - 1)
            perplexity = max(5, perplexity)
            reducer = TSNE(n_components=3, random_state=42, perplexity=perplexity)
            coords_3d = reducer.fit_transform(embeddings)
            subtitle = "t-SNE"

        # Create DataFrame for plotting
        plot_data = []
        for i, node in enumerate(nodes):
            plot_data.append({
                'x': coords_3d[i, 0],
                'y': coords_3d[i, 1],
                'z': coords_3d[i, 2],
                'node_type': node.node_type,
                'connection_type': node.connection_type,
                'step_number': node.step_number,
                'relevance_score': node.relevance_score,
                'is_final_result': node.is_final_result,
                'text_preview': node.text[:150] + "..." if len(node.text) > 150 else node.text,
                'document_id': node.document_id,
                'distance_from_anchor': node.distance_from_anchor
            })

        df = pd.DataFrame(plot_data)

        # Create the 3D scatter plot
        fig = go.Figure()

        # Color mapping for different node types and states
        color_map = {
            'query': 'gold',
            'anchor': 'red',
            'chunk': 'lightblue',
            'sentence': 'lightgreen',
            'final_result': 'darkgreen'
        }

        # Plot different node types
        for node_type in df['node_type'].unique():
            if node_type == 'query':
                # Special handling for query node
                subset = df[df['node_type'] == node_type]
                fig.add_trace(go.Scatter3d(
                    x=subset['x'], y=subset['y'], z=subset['z'],
                    mode='markers+text',
                    marker=dict(
                        size=20,
                        color='gold',
                        symbol='star',
                        line=dict(width=3, color='black')
                    ),
                    text=['QUERY'],
                    textposition="middle center",
                    name="Query",
                    hovertemplate="<b>QUERY</b><br>%{customdata[0]}<extra></extra>",
                    customdata=list(zip(subset['text_preview']))
                ))
            else:
                # Regular nodes - separate final results from intermediate nodes
                node_subset = df[df['node_type'] == node_type]

                # Final results
                final_subset = node_subset[node_subset['is_final_result'] == True]
                if not final_subset.empty:
                    fig.add_trace(go.Scatter3d(
                        x=final_subset['x'], y=final_subset['y'], z=final_subset['z'],
                        mode='markers+text',
                        marker=dict(
                            size=final_subset['relevance_score'] * 15 + 8,
                            color='darkgreen',
                            opacity=0.9,
                            line=dict(width=2, color='black')
                        ),
                        text=final_subset['step_number'],
                        textposition="middle center",
                        name=f"Final {node_type.title()}s",
                        hovertemplate=(
                                f"<b>Final {node_type.title()}</b><br>" +
                                "Step: %{text}<br>" +
                                "Relevance: %{customdata[0]:.3f}<br>" +
                                "Connection: %{customdata[1]}<br>" +
                                "Document: %{customdata[2]}<br>" +
                                "Text: %{customdata[3]}<br>" +
                                "<extra></extra>"
                        ),
                        customdata=list(zip(
                            final_subset['relevance_score'],
                            final_subset['connection_type'],
                            final_subset['document_id'],
                            final_subset['text_preview']
                        ))
                    ))

                # Intermediate nodes
                intermediate_subset = node_subset[node_subset['is_final_result'] == False]
                if not intermediate_subset.empty:
                    fig.add_trace(go.Scatter3d(
                        x=intermediate_subset['x'], y=intermediate_subset['y'], z=intermediate_subset['z'],
                        mode='markers+text',
                        marker=dict(
                            size=intermediate_subset['relevance_score'] * 10 + 5,
                            color=color_map.get(node_type, 'gray'),
                            opacity=0.6,
                            line=dict(width=1, color='gray')
                        ),
                        text=intermediate_subset['step_number'],
                        textposition="middle center",
                        name=f"Visited {node_type.title()}s",
                        hovertemplate=(
                                f"<b>Visited {node_type.title()}</b><br>" +
                                "Step: %{text}<br>" +
                                "Relevance: %{customdata[0]:.3f}<br>" +
                                "Connection: %{customdata[1]}<br>" +
                                "Document: %{customdata[2]}<br>" +
                                "Text: %{customdata[3]}<br>" +
                                "<extra></extra>"
                        ),
                        customdata=list(zip(
                            intermediate_subset['relevance_score'],
                            intermediate_subset['connection_type'],
                            intermediate_subset['document_id'],
                            intermediate_subset['text_preview']
                        ))
                    ))

        # Add traversal path lines
        self._add_traversal_path_lines(fig, nodes, coords_3d, result)

        # Update layout
        fig.update_layout(
            title=f"{result.algorithm_name} Semantic Traversal Visualization<br>" +
                  f"Query: '{query[:80]}...'<br>" +
                  f"Retrieved: {len(result.retrieved_content)} sentences | " +
                  f"Traversed: {len(nodes)} nodes | " +
                  f"Score: {result.final_score:.3f}<br>" +
                  f"<i>{subtitle}</i>",
            scene=dict(
                xaxis_title=f"Dimension 1 ({method.upper()})",
                yaxis_title=f"Dimension 2 ({method.upper()})",
                zaxis_title=f"Dimension 3 ({method.upper()})",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            height=900,
            font=dict(size=12),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        return fig

    def _add_traversal_path_lines(self, fig: go.Figure, nodes: List[VisualizationNode],
                                  coords_3d: np.ndarray, result: RetrievalResult):
        """Add lines showing the traversal path"""

        if result.algorithm_name == "BasicRetrieval":
            # For BasicRetrieval, connect query to all final results
            query_idx = next(i for i, node in enumerate(nodes) if node.node_type == "query")
            final_result_indices = [i for i, node in enumerate(nodes) if node.is_final_result]

            for final_idx in final_result_indices:
                fig.add_trace(go.Scatter3d(
                    x=[coords_3d[query_idx, 0], coords_3d[final_idx, 0]],
                    y=[coords_3d[query_idx, 1], coords_3d[final_idx, 1]],
                    z=[coords_3d[query_idx, 2], coords_3d[final_idx, 2]],
                    mode='lines',
                    line=dict(color='orange', width=3, dash='dot'),
                    showlegend=False,
                    hovertemplate="Query → Final Result<extra></extra>"
                ))
        else:
            # For traversal algorithms, connect nodes in step order
            step_ordered_nodes = [(i, node) for i, node in enumerate(nodes) if node.step_number >= 0]
            step_ordered_nodes.sort(key=lambda x: x[1].step_number)

            for i in range(len(step_ordered_nodes) - 1):
                current_idx, current_node = step_ordered_nodes[i]
                next_idx, next_node = step_ordered_nodes[i + 1]

                # Color based on connection type
                if next_node.connection_type in ['cross_document', 'theme_bridge']:
                    line_color = 'red'
                    line_width = 4
                    line_dash = 'solid'
                elif next_node.connection_type == 'hierarchical':
                    line_color = 'purple'
                    line_width = 3
                    line_dash = 'dash'
                else:
                    line_color = 'blue'
                    line_width = 2
                    line_dash = 'dot'

                fig.add_trace(go.Scatter3d(
                    x=[coords_3d[current_idx, 0], coords_3d[next_idx, 0]],
                    y=[coords_3d[current_idx, 1], coords_3d[next_idx, 1]],
                    z=[coords_3d[current_idx, 2], coords_3d[next_idx, 2]],
                    mode='lines',
                    line=dict(color=line_color, width=line_width, dash=line_dash),
                    showlegend=False,
                    hovertemplate=f"Step {current_node.step_number} → {next_node.step_number}<br>" +
                                  f"Connection: {next_node.connection_type}<extra></extra>"
                ))

    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Get embedding for the query"""
        if hasattr(self.kg, 'encode_query'):
            return self.kg.encode_query(query)
        else:
            # Fallback to using sentence transformers
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
            return model.encode([query])[0]

    def _find_sentence_id(self, sentence_text: str) -> Optional[str]:
        """Find sentence ID by text content"""
        for sentence_id, sentence_obj in self.kg.sentences.items():
            if hasattr(sentence_obj, 'sentence_text') and sentence_obj.sentence_text == sentence_text:
                return sentence_id
        return None


def create_algorithm_visualization(result: RetrievalResult, query: str,
                                   knowledge_graph: KnowledgeGraph,
                                   method: str = "pca", max_nodes: int = 50,
                                   show_all_visited: bool = True) -> go.Figure:
    """
    Main entry point for creating 3D visualizations of algorithm results.

    Args:
        result: RetrievalResult from any algorithm
        query: Original query string
        knowledge_graph: The knowledge graph instance
        method: Dimensionality reduction method ('pca' or 'tsne')
        max_nodes: Maximum nodes to show for performance
        show_all_visited: Whether to show all visited nodes or just final results

    Returns:
        Plotly Figure ready for display or saving
    """
    visualizer = KnowledgeGraphPlotlyVisualizer(knowledge_graph)
    return visualizer.visualize_retrieval_result(
        result, query, method, max_nodes, show_all_visited
    )


# Example usage function
def example_usage():
    """Example of how to use the visualizer"""

    # Assuming you have a retrieval result and knowledge graph
    # This would typically come from running one of your algorithms

    print("Example usage:")
    print("from utils.plotly_visualizer import create_algorithm_visualization")
    print("")
    print("# After running an algorithm:")
    print("result = retrieval_orchestrator.retrieve(query, 'query_traversal')")
    print("fig = create_algorithm_visualization(")
    print("    result=result,")
    print("    query=query,")
    print("    knowledge_graph=kg,")
    print("    method='pca',  # or 'tsne'")
    print("    max_nodes=50,")
    print("    show_all_visited=True")
    print(")")
    print("fig.show()  # Display interactive plot")
    print("# fig.write_html('visualization.html')  # Save to file")


if __name__ == "__main__":
    example_usage()