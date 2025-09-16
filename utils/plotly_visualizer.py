#!/usr/bin/env python3
"""
Knowledge Graph Traversal 3D Visualizer with Plotly
==================================================

Creates beautiful 3D visualizations of semantic graph traversal for all four algorithms.
Adapted from perfect reference examples to work with algorithm results and cached embeddings.
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
    node_type: str  # 'chunk', 'sentence', or 'query'
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
        Matches the style and functionality of the perfect reference examples.
        """
        print(f"🎨 Creating 3D visualization for {result.algorithm_name}")

        # Extract visualization nodes from the result and knowledge graph
        nodes = self._extract_visualization_nodes(result, query, show_all_visited)
        
        if len(nodes) < 2:
            print(f"⚠️ Only {len(nodes)} nodes found - creating basic visualization")
            return self._create_basic_plotly_visualization(result, query)

        # Limit nodes for performance (like reference examples)
        if len(nodes) > max_nodes:
            # Keep query, first few traversal steps, and all final results
            query_nodes = [n for n in nodes if n.node_type == 'query']
            final_result_nodes = [n for n in nodes if n.is_final_result]
            traversal_nodes = [n for n in nodes if n.node_type != 'query' and not n.is_final_result]
            
            # Keep first traversal nodes up to the limit
            max_traversal = max_nodes - len(query_nodes) - len(final_result_nodes)
            traversal_nodes = traversal_nodes[:max_traversal]
            
            nodes = query_nodes + traversal_nodes + final_result_nodes
            print(f"🎯 Limited to {len(nodes)} nodes for performance")

        # Create the 3D plot (like reference examples)
        return self._create_3d_plot(nodes, result, query, method)

    def _extract_visualization_nodes(self, result: RetrievalResult, query: str, 
                                   show_all_visited: bool) -> List[VisualizationNode]:
        """Extract all nodes for visualization from result and knowledge graph"""
        nodes = []

        # Add query as a special node (like reference examples)
        query_embedding = self._get_query_embedding(query)
        if query_embedding is not None:
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

        # Get final result nodes (sentences that were actually retrieved)
        final_result_nodes = self._extract_final_result_nodes(result)
        
        # Get traversal nodes if available and requested
        traversal_nodes = []
        if show_all_visited and result.traversal_path and result.traversal_path.nodes:
            traversal_nodes = self._extract_traversal_nodes(result)

        # Combine all nodes
        nodes.extend(traversal_nodes)
        nodes.extend(final_result_nodes)

        # Mark final results
        final_sentences = set(result.retrieved_content)
        for node in nodes:
            if node.text in final_sentences:
                node.is_final_result = True

        print(f"📊 Extracted {len(nodes)} nodes: {len([n for n in nodes if n.node_type == 'query'])} query, "
              f"{len([n for n in nodes if n.is_final_result])} final results, "
              f"{len([n for n in nodes if not n.is_final_result and n.node_type != 'query'])} traversal")

        return nodes

    def _extract_final_result_nodes(self, result: RetrievalResult) -> List[VisualizationNode]:
        """Extract nodes for the final retrieved content"""
        nodes = []

        for i, sentence_text in enumerate(result.retrieved_content):
            # Try to find this sentence in the knowledge graph
            sentence_id = self._find_sentence_id(sentence_text)
            embedding = None
            document_id = "unknown"

            if sentence_id:
                # Get sentence object and its embedding
                sentence_obj = self.kg.sentences.get(sentence_id)
                if sentence_obj:
                    embedding = self._get_sentence_embedding(sentence_id)
                    document_id = self._get_sentence_document(sentence_obj)
            
            if embedding is None:
                # Fallback: encode the sentence text directly
                embedding = self._encode_text(sentence_text)

            # Calculate relevance score
            relevance_score = result.confidence_scores[i] if i < len(result.confidence_scores) else 0.5
            if hasattr(result, 'query_similarities') and result.query_similarities:
                relevance_score = result.query_similarities.get(sentence_text, relevance_score)

            nodes.append(VisualizationNode(
                node_id=sentence_id or f"sentence_{i}",
                node_type="sentence",
                text=sentence_text,
                embedding=embedding,
                step_number=i,
                relevance_score=relevance_score,
                connection_type="final_result",
                distance_from_anchor=0,
                document_id=document_id,
                is_final_result=True
            ))

        return nodes

    def _extract_traversal_nodes(self, result: RetrievalResult) -> List[VisualizationNode]:
        """Extract nodes from the traversal path"""
        nodes = []

        if not result.traversal_path or not result.traversal_path.nodes:
            return nodes

        path = result.traversal_path

        for i, node_id in enumerate(path.nodes):
            granularity = path.granularity_levels[i] if i < len(path.granularity_levels) else GranularityLevel.CHUNK
            connection_type = path.connection_types[i - 1] if i > 0 and i - 1 < len(path.connection_types) else ConnectionType.RAW_SIMILARITY

            embedding = None
            text = ""
            document_id = "unknown"
            node_type = "chunk"

            if granularity == GranularityLevel.SENTENCE:
                # This is a sentence node
                sentence_obj = self.kg.sentences.get(node_id)
                if sentence_obj:
                    embedding = self._get_sentence_embedding(node_id)
                    text = sentence_obj.sentence_text if hasattr(sentence_obj, 'sentence_text') else str(sentence_obj)
                    document_id = self._get_sentence_document(sentence_obj)
                    node_type = "sentence"
            else:
                # This is a chunk node
                chunk_obj = self.kg.chunks.get(node_id)
                if chunk_obj:
                    embedding = self._get_chunk_embedding(node_id)
                    text = chunk_obj.chunk_text if hasattr(chunk_obj, 'chunk_text') else str(chunk_obj)
                    # Truncate long chunk text for display
                    if len(text) > 100:
                        text = text[:100] + "..."
                    document_id = self._get_chunk_document(node_id)
                    node_type = "chunk"

            if embedding is None:
                # Skip nodes without embeddings
                continue

            # Calculate relevance score
            relevance_score = self._calculate_node_relevance(node_id, result)

            nodes.append(VisualizationNode(
                node_id=node_id,
                node_type=node_type,
                text=text,
                embedding=embedding,
                step_number=i,
                relevance_score=relevance_score,
                connection_type=connection_type.value if hasattr(connection_type, 'value') else str(connection_type),
                distance_from_anchor=i,
                document_id=document_id,
                is_final_result=False
            ))

        return nodes

    def _create_3d_plot(self, nodes: List[VisualizationNode], result: RetrievalResult,
                        query: str, method: str) -> go.Figure:
        """Create the 3D scatter plot (matching reference examples style)"""

        if len(nodes) < 2:
            print("❌ Not enough nodes for 3D visualization")
            return self._create_basic_plotly_visualization(result, query)

        # Get embeddings and reduce dimensionality (like reference examples)
        embeddings = np.array([node.embedding for node in nodes])

        # Normalize embeddings for better visualization
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        if method == "pca":
            reducer = PCA(n_components=3, random_state=42)
            coords_3d = reducer.fit_transform(embeddings)
            explained_variance = reducer.explained_variance_ratio_
            subtitle = f"PCA (explained variance: {explained_variance.sum():.1%})"
        else:  # t-SNE
            perplexity = min(30, len(embeddings) - 1)
            perplexity = max(5, perplexity)
            reducer = TSNE(n_components=3, random_state=42, perplexity=perplexity, n_iter=1000)
            coords_3d = reducer.fit_transform(embeddings)
            subtitle = "t-SNE"

        print(f"🔬 Applied {method.upper()} dimensionality reduction to {len(embeddings)} embeddings")

        # Create the figure (like reference examples)
        fig = go.Figure()

        # Group nodes by type and status for different visual treatment
        node_groups = self._group_nodes_for_visualization(nodes, coords_3d)

        # Plot each group with appropriate styling (like reference examples)
        for group_name, group_data in node_groups.items():
            if not group_data['nodes']:
                continue

            self._add_node_group_trace(fig, group_name, group_data)

        # Add traversal path lines (like reference examples)
        self._add_traversal_path_lines(fig, nodes, coords_3d, result)

        # Update layout (matching reference style)
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
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showspikes=False, showgrid=True, gridcolor='lightgray'),
                yaxis=dict(showspikes=False, showgrid=True, gridcolor='lightgray'),
                zaxis=dict(showspikes=False, showgrid=True, gridcolor='lightgray')
            ),
            height=900,
            font=dict(size=12),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='gray',
                borderwidth=1
            ),
            margin=dict(l=0, r=0, t=150, b=0)
        )

        print(f"✅ 3D visualization created successfully")
        return fig

    def _group_nodes_for_visualization(self, nodes: List[VisualizationNode], 
                                     coords_3d: np.ndarray) -> Dict[str, Dict]:
        """Group nodes for different visual treatment"""
        groups = {
            'query': {'nodes': [], 'coords': [], 'colors': [], 'sizes': [], 'symbols': []},
            'final_sentences': {'nodes': [], 'coords': [], 'colors': [], 'sizes': [], 'symbols': []},
            'final_chunks': {'nodes': [], 'coords': [], 'colors': [], 'sizes': [], 'symbols': []},
            'traversal_sentences': {'nodes': [], 'coords': [], 'colors': [], 'sizes': [], 'symbols': []},
            'traversal_chunks': {'nodes': [], 'coords': [], 'colors': [], 'sizes': [], 'symbols': []}
        }

        for i, node in enumerate(nodes):
            coord = coords_3d[i]
            
            if node.node_type == 'query':
                groups['query']['nodes'].append(node)
                groups['query']['coords'].append(coord)
                groups['query']['colors'].append('gold')
                groups['query']['sizes'].append(25)
                groups['query']['symbols'].append('diamond')  # Using diamond instead of star
            elif node.is_final_result:
                if node.node_type == 'sentence':
                    groups['final_sentences']['nodes'].append(node)
                    groups['final_sentences']['coords'].append(coord)
                    groups['final_sentences']['colors'].append('darkgreen')
                    groups['final_sentences']['sizes'].append(15 + node.relevance_score * 10)
                    groups['final_sentences']['symbols'].append('circle')
                else:
                    groups['final_chunks']['nodes'].append(node)
                    groups['final_chunks']['coords'].append(coord)
                    groups['final_chunks']['colors'].append('green')
                    groups['final_chunks']['sizes'].append(12 + node.relevance_score * 8)
                    groups['final_chunks']['symbols'].append('square')
            else:
                # Traversal nodes
                if node.node_type == 'sentence':
                    groups['traversal_sentences']['nodes'].append(node)
                    groups['traversal_sentences']['coords'].append(coord)
                    # Color based on relevance score
                    intensity = 0.3 + (node.relevance_score * 0.7)
                    color = f'rgba(0, {int(255 * intensity)}, 0, 0.6)'
                    groups['traversal_sentences']['colors'].append(color)
                    groups['traversal_sentences']['sizes'].append(8 + node.relevance_score * 6)
                    groups['traversal_sentences']['symbols'].append('circle')
                else:
                    groups['traversal_chunks']['nodes'].append(node)
                    groups['traversal_chunks']['coords'].append(coord)
                    groups['traversal_chunks']['colors'].append('lightblue')
                    groups['traversal_chunks']['sizes'].append(6 + node.relevance_score * 4)
                    groups['traversal_chunks']['symbols'].append('square')

        return groups

    def _add_node_group_trace(self, fig: go.Figure, group_name: str, group_data: Dict):
        """Add a trace for a group of nodes"""
        nodes = group_data['nodes']
        coords = np.array(group_data['coords'])
        colors = group_data['colors']
        sizes = group_data['sizes']
        symbols = group_data['symbols']

        if len(nodes) == 0:
            return

        # Create hover text
        hover_texts = []
        for node in nodes:
            hover_text = (
                f"<b>{group_name.replace('_', ' ').title()}</b><br>" +
                f"Step: {node.step_number}<br>" +
                f"Relevance: {node.relevance_score:.3f}<br>" +
                f"Connection: {node.connection_type}<br>" +
                f"Document: {node.document_id}<br>" +
                f"Text: {node.text[:150]}..."
            )
            hover_texts.append(hover_text)

        # Create step number labels for display
        step_labels = [str(node.step_number) if node.step_number >= 0 else 'Q' for node in nodes]

        # Determine marker properties
        if group_name == 'query':
            marker_symbol = 'diamond'  # Using diamond instead of star (star not supported)
            show_text = True
        else:
            marker_symbol = 'circle' if 'sentence' in group_name else 'square'
            show_text = True

        # Add trace
        trace = go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode='markers+text' if show_text else 'markers',
            marker=dict(
                size=sizes,
                color=colors,
                symbol=marker_symbol,
                line=dict(width=2, color='black' if group_name == 'query' else 'gray'),
                opacity=1.0 if 'final' in group_name or group_name == 'query' else 0.7
            ),
            text=step_labels if show_text else None,
            textposition="middle center",
            textfont=dict(size=10, color='black'),
            name=group_name.replace('_', ' ').title(),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover_texts
        )

        fig.add_trace(trace)

    def _add_traversal_path_lines(self, fig: go.Figure, nodes: List[VisualizationNode],
                                  coords_3d: np.ndarray, result: RetrievalResult):
        """Add lines showing the traversal path (like reference examples)"""

        if result.algorithm_name == "BasicRetrieval":
            # For BasicRetrieval, connect query to all final results
            query_indices = [i for i, node in enumerate(nodes) if node.node_type == "query"]
            final_result_indices = [i for i, node in enumerate(nodes) if node.is_final_result]

            if query_indices:
                query_idx = query_indices[0]
                for final_idx in final_result_indices:
                    fig.add_trace(go.Scatter3d(
                        x=[coords_3d[query_idx, 0], coords_3d[final_idx, 0]],
                        y=[coords_3d[query_idx, 1], coords_3d[final_idx, 1]],
                        z=[coords_3d[query_idx, 2], coords_3d[final_idx, 2]],
                        mode='lines',
                        line=dict(color='orange', width=3, dash='dot'),
                        showlegend=False,
                        hovertemplate="Query → Final Result<extra></extra>",
                        name="Connection"
                    ))
        else:
            # For traversal algorithms, connect nodes in step order
            step_ordered_nodes = [(i, node) for i, node in enumerate(nodes) 
                                if node.step_number >= 0 and node.node_type != 'query']
            step_ordered_nodes.sort(key=lambda x: x[1].step_number)

            for i in range(len(step_ordered_nodes) - 1):
                current_idx, current_node = step_ordered_nodes[i]
                next_idx, next_node = step_ordered_nodes[i + 1]

                # Color based on connection type (like reference examples)
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
                                  f"Connection: {next_node.connection_type}<extra></extra>",
                    name="Traversal Path"
                ))

    def _create_basic_plotly_visualization(self, result: RetrievalResult, query: str) -> go.Figure:
        """Create a basic visualization when 3D plot is not possible"""
        fig = go.Figure()

        # Create a simple text display
        fig.add_annotation(
            text=f"<b>{result.algorithm_name} Results</b><br><br>" +
                 f"Query: {query[:100]}...<br>" +
                 f"Retrieved Sentences: {len(result.retrieved_content)}<br>" +
                 f"Final Score: {result.final_score:.3f}<br>" +
                 f"Processing Time: {result.processing_time:.3f}s",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16),
            align="center"
        )

        fig.update_layout(
            title=f"{result.algorithm_name} - Basic Visualization",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=600
        )

        return fig

    # Utility methods for data extraction
    def _get_query_embedding(self, query: str) -> Optional[np.ndarray]:
        """Get embedding for the query using available methods"""
        # Try to use the knowledge graph's encoding method if available
        if hasattr(self.kg, 'encode_query'):
            return self.kg.encode_query(query)
        elif hasattr(self.kg, 'embedding_model'):
            return self.kg.embedding_model.encode([query])[0]
        else:
            # Fallback: use cached model to avoid repeated loading
            return self._encode_text(query)

    def _encode_text(self, text: str) -> np.ndarray:
        """Fallback text encoding method using KG's model if available"""
        try:
            # Try to use the knowledge graph's existing model first
            if hasattr(self.kg, 'embedding_model'):
                return self.kg.embedding_model.encode([text])[0]
            else:
                # Fallback: create model once and reuse
                if not hasattr(self, '_fallback_model'):
                    from sentence_transformers import SentenceTransformer
                    self._fallback_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
                return self._fallback_model.encode([text])[0]
        except Exception as e:
            print(f"⚠️ Warning: Could not encode text '{text[:50]}...': {e}")
            # Return a dummy embedding if all else fails
            return np.random.normal(0, 1, 768)  # Fixed dimension to 768

    def _get_sentence_embedding(self, sentence_id: str) -> Optional[np.ndarray]:
        """Get cached embedding for a sentence"""
        sentence_obj = self.kg.sentences.get(sentence_id)
        if sentence_obj:
            for attr in ['embedding', 'embeddings', 'vector']:
                if hasattr(sentence_obj, attr):
                    embedding = getattr(sentence_obj, attr)
                    if embedding is not None:
                        return np.array(embedding)
        return None

    def _get_chunk_embedding(self, chunk_id: str) -> Optional[np.ndarray]:
        """Get cached embedding for a chunk"""
        chunk_obj = self.kg.chunks.get(chunk_id)
        if chunk_obj:
            for attr in ['embedding', 'embeddings', 'vector']:
                if hasattr(chunk_obj, attr):
                    embedding = getattr(chunk_obj, attr)
                    if embedding is not None:
                        return np.array(embedding)
        return None

    def _find_sentence_id(self, sentence_text: str) -> Optional[str]:
        """Find sentence ID by text content"""
        for sentence_id, sentence_obj in self.kg.sentences.items():
            if hasattr(sentence_obj, 'sentence_text') and sentence_obj.sentence_text == sentence_text:
                return sentence_id
        return None

    def _get_sentence_document(self, sentence_obj) -> str:
        """Get document ID for a sentence object"""
        for attr in ['source_document', 'document_id', 'doc_id']:
            if hasattr(sentence_obj, attr):
                doc_id = getattr(sentence_obj, attr)
                if doc_id:
                    return str(doc_id)
        return "unknown"

    def _get_chunk_document(self, chunk_id: str) -> str:
        """Get document ID for a chunk"""
        chunk_obj = self.kg.chunks.get(chunk_id)
        if chunk_obj:
            for attr in ['source_document', 'document_id', 'doc_id']:
                if hasattr(chunk_obj, attr):
                    doc_id = getattr(chunk_obj, attr)
                    if doc_id:
                        return str(doc_id)
        
        # Fallback: extract from chunk ID
        if '_' in chunk_id:
            parts = chunk_id.split('_')
            if len(parts) >= 2:
                return '_'.join(parts[:-1])
            return parts[0]
        
        return "unknown"

    def _calculate_node_relevance(self, node_id: str, result: RetrievalResult) -> float:
        """Calculate relevance score for a node"""
        # Try query similarities first
        if hasattr(result, 'query_similarities') and result.query_similarities:
            if node_id in result.query_similarities:
                return result.query_similarities[node_id]
            
            # For chunks, find max similarity among sentences
            if hasattr(self.kg, 'get_chunk_sentences'):
                chunk_sentences = self.kg.get_chunk_sentences(node_id)
                if chunk_sentences:
                    max_similarity = 0.0
                    for sentence in chunk_sentences:
                        sentence_text = sentence.sentence_text if hasattr(sentence, 'sentence_text') else str(sentence)
                        similarity = result.query_similarities.get(sentence_text, 0.0)
                        max_similarity = max(max_similarity, similarity)
                    return max_similarity
        
        # Try metadata
        if hasattr(result, 'metadata') and result.metadata:
            extraction_metadata = result.metadata.get('extraction_metadata', {})
            if node_id in extraction_metadata:
                return extraction_metadata[node_id].get('similarity_score', 0.5)
        
        return 0.5  # Default


def create_algorithm_visualization(result: RetrievalResult, query: str,
                                   knowledge_graph: KnowledgeGraph,
                                   method: str = "pca", max_nodes: int = 50,
                                   show_all_visited: bool = True) -> go.Figure:
    """
    Main entry point for creating 3D visualizations of algorithm results.
    Matches the style and functionality of the perfect reference examples.

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
