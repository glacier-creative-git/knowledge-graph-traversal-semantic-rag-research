"""
3D Semantic Graph Traversal Visualizer
=====================================

This script creates a 3D visualization of semantic graph traversal across multiple documents,
where each document is represented as a 2D similarity heatmap positioned in 3D space.

Key Features:
- Pre-renders heatmaps using matplotlib
- Positions them as textured surfaces in Plotly 3D
- Draws precise traversal paths between exact sentence coordinates
- Handles variable document sizes and densities
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import tempfile
import os
from typing import List, Tuple, Dict
from dataclasses import dataclass
import io
import base64
from PIL import Image

@dataclass
class SentenceNode:
    """Represents a sentence with its document and position info"""
    doc_id: int
    sentence_id: int
    text: str
    embedding: np.ndarray

@dataclass
class TraversalStep:
    """Represents a step in the semantic traversal"""
    node: SentenceNode
    step_number: int
    connection_type: str  # 'anchor', 'intra_doc', 'cross_doc'
    similarity_score: float
    coordinates_2d: Tuple[int, int]  # Position in heatmap
    coordinates_3d: Tuple[float, float, float]  # Position in 3D space

class SemanticGraph3DVisualizer:
    """Creates 3D visualizations of semantic graph traversal"""

    def __init__(self, layer_spacing: float = 6.0, heatmap_size: float = 4.0):
        self.layer_spacing = layer_spacing  # Distance between documents (horizontal)
        self.heatmap_size = heatmap_size    # Size of each heatmap in 3D space
        self.documents = []
        self.similarity_matrices = []
        self.traversal_path = []

    def generate_synthetic_documents(self, num_docs: int = 3, sentences_per_doc: int = 5) -> List[Dict]:
        """Generate synthetic documents with semantic overlap for testing"""

        # Base topics for creating semantic relationships
        topics = {
            'technology': ['computer', 'software', 'algorithm', 'data', 'system'],
            'science': ['research', 'experiment', 'theory', 'discovery', 'analysis'],
            'business': ['company', 'market', 'strategy', 'growth', 'revenue']
        }

        topic_keys = list(topics.keys())
        documents = []

        for doc_id in range(num_docs):
            # Each document focuses on a primary topic but has some overlap
            primary_topic = topic_keys[doc_id % len(topic_keys)]
            secondary_topic = topic_keys[(doc_id + 1) % len(topic_keys)]

            sentences = []
            for sent_id in range(sentences_per_doc):
                if sent_id < 3:  # First 3 sentences focus on primary topic
                    words = np.random.choice(topics[primary_topic], 3, replace=False)
                else:  # Last 2 sentences introduce secondary topic (creates cross-doc connections)
                    words = np.random.choice(topics[secondary_topic], 3, replace=False)

                sentence = f"The {words[0]} involves {words[1]} and {words[2]} concepts."
                sentences.append(sentence)

            documents.append({
                'doc_id': doc_id,
                'title': f'Document {doc_id + 1}: {primary_topic.title()}',
                'sentences': sentences
            })

        return documents

    def create_embeddings(self, documents: List[Dict]) -> List[List[np.ndarray]]:
        """Create simple embeddings based on word overlap (for testing)"""
        # Create a simple vocabulary from all sentences
        all_words = set()
        for doc in documents:
            for sentence in doc['sentences']:
                words = sentence.lower().split()
                all_words.update(words)

        vocab = list(all_words)
        vocab_size = len(vocab)

        # Create word-to-index mapping
        word_to_idx = {word: idx for idx, word in enumerate(vocab)}

        # Generate embeddings for each sentence
        document_embeddings = []
        for doc in documents:
            doc_embeddings = []
            for sentence in doc['sentences']:
                # Create bag-of-words embedding
                embedding = np.zeros(vocab_size)
                words = sentence.lower().split()
                for word in words:
                    if word in word_to_idx:
                        embedding[word_to_idx[word]] = 1.0

                # Add some noise for more realistic similarities
                embedding += np.random.normal(0, 0.1, vocab_size)
                # Normalize
                embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
                doc_embeddings.append(embedding)

            document_embeddings.append(doc_embeddings)

        return document_embeddings

    def build_similarity_matrices(self, document_embeddings: List[List[np.ndarray]]) -> List[np.ndarray]:
        """Build similarity matrices for each document"""
        matrices = []

        for doc_embeddings in document_embeddings:
            n_sentences = len(doc_embeddings)
            similarity_matrix = np.zeros((n_sentences, n_sentences))

            for i in range(n_sentences):
                for j in range(n_sentences):
                    if i == j:
                        similarity_matrix[i, j] = 1.0
                    else:
                        # Dot product similarity
                        sim = np.dot(doc_embeddings[i], doc_embeddings[j])
                        similarity_matrix[i, j] = max(0, sim)  # Ensure non-negative

            matrices.append(similarity_matrix)

        return matrices

    def simulate_traversal(self, documents: List[Dict], document_embeddings: List[List[np.ndarray]],
                          query: str = "computer algorithm data") -> List[TraversalStep]:
        """Simulate a semantic graph traversal based on a query"""

        # Create query embedding
        vocab = set()
        for doc in documents:
            for sentence in doc['sentences']:
                vocab.update(sentence.lower().split())
        vocab = list(vocab)
        word_to_idx = {word: idx for idx, word in enumerate(vocab)}

        query_embedding = np.zeros(len(vocab))
        query_words = query.lower().split()
        for word in query_words:
            if word in word_to_idx:
                query_embedding[word_to_idx[word]] = 1.0
        query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)

        # Find anchor points (most similar sentences to query across all documents)
        anchor_candidates = []
        for doc_id, doc_embeddings in enumerate(document_embeddings):
            for sent_id, sent_embedding in enumerate(doc_embeddings):
                similarity = np.dot(query_embedding, sent_embedding)
                anchor_candidates.append((doc_id, sent_id, similarity))

        # Sort by similarity and pick top anchor
        anchor_candidates.sort(key=lambda x: x[2], reverse=True)

        # Simulate traversal path
        traversal_steps = []
        step_number = 0
        visited = set()

        # Start with the best anchor
        doc_id, sent_id, sim_score = anchor_candidates[0]
        anchor_step = TraversalStep(
            node=SentenceNode(doc_id, sent_id, documents[doc_id]['sentences'][sent_id],
                            document_embeddings[doc_id][sent_id]),
            step_number=step_number,
            connection_type='anchor',
            similarity_score=sim_score,
            coordinates_2d=(sent_id, sent_id),  # Diagonal position in heatmap
            coordinates_3d=self.map_2d_to_3d(sent_id, sent_id, doc_id, len(documents[doc_id]['sentences']))
        )
        traversal_steps.append(anchor_step)
        visited.add((doc_id, sent_id))
        step_number += 1

        # Add a few more steps within the same document
        current_doc = doc_id
        for i in range(2):  # Add 2 more intra-document steps
            best_score = -1
            best_next = None

            # Find best unvisited sentence in current document
            for next_sent_id in range(len(documents[current_doc]['sentences'])):
                if (current_doc, next_sent_id) not in visited:
                    sim_score = self.similarity_matrices[current_doc][sent_id, next_sent_id]
                    if sim_score > best_score:
                        best_score = sim_score
                        best_next = next_sent_id

            if best_next is not None:
                step = TraversalStep(
                    node=SentenceNode(current_doc, best_next,
                                    documents[current_doc]['sentences'][best_next],
                                    document_embeddings[current_doc][best_next]),
                    step_number=step_number,
                    connection_type='intra_doc',
                    similarity_score=best_score,
                    coordinates_2d=(sent_id, best_next),
                    coordinates_3d=self.map_2d_to_3d(sent_id, best_next, current_doc,
                                                   len(documents[current_doc]['sentences']))
                )
                traversal_steps.append(step)
                visited.add((current_doc, best_next))
                sent_id = best_next
                step_number += 1

        # Add cross-document jump
        if len(documents) > 1:
            # Find best cross-document connection
            current_embedding = document_embeddings[current_doc][sent_id]
            best_cross_score = -1
            best_cross_doc = None
            best_cross_sent = None

            for other_doc_id in range(len(documents)):
                if other_doc_id != current_doc:
                    for other_sent_id, other_embedding in enumerate(document_embeddings[other_doc_id]):
                        if (other_doc_id, other_sent_id) not in visited:
                            sim_score = np.dot(current_embedding, other_embedding)
                            if sim_score > best_cross_score:
                                best_cross_score = sim_score
                                best_cross_doc = other_doc_id
                                best_cross_sent = other_sent_id

            if best_cross_doc is not None:
                step = TraversalStep(
                    node=SentenceNode(best_cross_doc, best_cross_sent,
                                    documents[best_cross_doc]['sentences'][best_cross_sent],
                                    document_embeddings[best_cross_doc][best_cross_sent]),
                    step_number=step_number,
                    connection_type='cross_doc',
                    similarity_score=best_cross_score,
                    coordinates_2d=(best_cross_sent, best_cross_sent),
                    coordinates_3d=self.map_2d_to_3d(best_cross_sent, best_cross_sent, best_cross_doc,
                                                   len(documents[best_cross_doc]['sentences']))
                )
                traversal_steps.append(step)

        return traversal_steps

    def map_2d_to_3d(self, row: int, col: int, doc_id: int, matrix_size: int) -> Tuple[float, float, float]:
        """Map 2D heatmap coordinates to 3D world coordinates (VERTICAL standing documents)"""

        # Calculate the position within the heatmap (normalized to [0, 1])
        x_norm = col / (matrix_size - 1) if matrix_size > 1 else 0.5
        z_norm = row / (matrix_size - 1) if matrix_size > 1 else 0.5

        # VERTICAL STANDING LAYOUT: Documents stand up in X-Z plane
        x_3d = doc_id * self.layer_spacing + (x_norm - 0.5) * self.heatmap_size
        y_3d = 0  # All documents at same Y position (depth)
        z_3d = (z_norm - 0.5) * self.heatmap_size  # Height dimension

        return (x_3d, y_3d, z_3d)

    def render_heatmap(self, similarity_matrix: np.ndarray, title: str = "Similarity Matrix") -> str:
        """Render a similarity matrix as a PNG and return as base64 string"""

        fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')

        # Create the heatmap with a beautiful colormap
        im = ax.imshow(similarity_matrix, cmap='viridis', aspect='equal',
                      interpolation='nearest', vmin=0, vmax=1)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Similarity Score', rotation=270, labelpad=20)

        # Styling
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Sentence Index', fontsize=12)
        ax.set_ylabel('Sentence Index', fontsize=12)

        # Add grid
        ax.set_xticks(range(similarity_matrix.shape[1]))
        ax.set_yticks(range(similarity_matrix.shape[0]))
        ax.grid(True, alpha=0.3, linewidth=0.5)

        plt.tight_layout()

        # Save to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        buffer.seek(0)

        # Convert to base64
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        plt.close(fig)
        return image_base64

    def create_3d_visualization(self, documents: List[Dict], traversal_steps: List[TraversalStep],
                               query: str = "Query") -> go.Figure:
        """Create the main 3D visualization with ROTATED standing documents and proper coloring"""

        fig = go.Figure()

        # Find min/max similarity values across all documents for consistent coloring
        all_similarities = []
        for matrix in self.similarity_matrices:
            all_similarities.extend(matrix.flatten())
        sim_min, sim_max = min(all_similarities), max(all_similarities)

        # Add discrete heatmap surfaces for each document (ROTATED STANDING)
        for doc_id, (doc, similarity_matrix) in enumerate(zip(documents, self.similarity_matrices)):

            n_sentences = len(doc['sentences'])
            doc_center_x = doc_id * self.layer_spacing

            # Create individual squares for each cell to get discrete appearance
            for i in range(n_sentences):
                for j in range(n_sentences):
                    similarity_val = similarity_matrix[i, j]

                    # Calculate square boundaries for this cell
                    cell_size = self.heatmap_size / n_sentences

                    # ROTATED: Documents face each other in Y-Z plane
                    # X = document position (fixed for each doc)
                    # Y = column offset (width of heatmap)
                    # Z = row offset (height of heatmap)
                    x_pos = doc_center_x  # Fixed X position for this document
                    y_min = (j - (n_sentences-1)/2) * cell_size - cell_size/2
                    y_max = (j - (n_sentences-1)/2) * cell_size + cell_size/2
                    z_min = (i - (n_sentences-1)/2) * cell_size - cell_size/2
                    z_max = (i - (n_sentences-1)/2) * cell_size + cell_size/2

                    # Create square mesh for this cell (in Y-Z plane)
                    fig.add_trace(go.Mesh3d(
                        x=[x_pos, x_pos, x_pos, x_pos, x_pos-0.01, x_pos-0.01, x_pos-0.01, x_pos-0.01],
                        y=[y_min, y_max, y_max, y_min, y_min, y_max, y_max, y_min],
                        z=[z_min, z_min, z_max, z_max, z_min, z_min, z_max, z_max],
                        i=[0, 0, 0, 4, 4, 4, 2, 2],
                        j=[1, 2, 3, 5, 6, 7, 6, 3],
                        k=[2, 3, 0, 6, 7, 4, 7, 0],
                        intensity=[similarity_val] * 8,
                        colorscale='Viridis',
                        cmin=sim_min,  # Set explicit color range
                        cmax=sim_max,
                        showscale=True if doc_id == 0 and i == 0 and j == 0 else False,
                        colorbar=dict(
                            title="Similarity",
                            titleside="right",
                            tickmode="linear",
                            tick0=sim_min,
                            dtick=(sim_max-sim_min)/5
                        ) if doc_id == 0 and i == 0 and j == 0 else None,
                        hovertemplate=(
                            f'<b>Document {doc_id + 1}</b><br>' +
                            f'Sentence {i+1} → {j+1}<br>' +
                            f'Similarity: {similarity_val:.3f}<br>' +
                            '<extra></extra>'
                        ),
                        showlegend=False,
                        opacity=0.9
                    ))

            # Add document labels
            fig.add_trace(go.Scatter3d(
                x=[doc_center_x],
                y=[0],
                z=[self.heatmap_size/2 + 0.5],
                mode='text',
                text=[f'<b>Doc {doc_id + 1}</b>'],
                textfont=dict(size=16, color='black'),
                showlegend=False,
                hoverinfo='skip'
            ))

        # Add traversal path (updated for rotated layout)
        if len(traversal_steps) > 1:
            # Extract coordinates for the path (need to update mapping for rotation)
            x_path = []
            y_path = []
            z_path = []

            for step in traversal_steps:
                # Recalculate coordinates for rotated documents
                doc_id = step.node.doc_id
                sent_row = step.node.sentence_id  # This might need adjustment based on your coordinate system
                sent_col = step.node.sentence_id  # This might need adjustment

                n_sentences = len(documents[doc_id]['sentences'])
                cell_size = self.heatmap_size / n_sentences

                x_coord = doc_id * self.layer_spacing + 0.1  # Slightly in front of document
                y_coord = (sent_col - (n_sentences-1)/2) * cell_size
                z_coord = (sent_row - (n_sentences-1)/2) * cell_size

                x_path.append(x_coord)
                y_path.append(y_coord)
                z_path.append(z_coord)

            # Create hover text for path points
            hover_text = []
            colors = []
            for step in traversal_steps:
                text = (f"<b>Step {step.step_number}</b><br>"
                       f"Type: {step.connection_type.replace('_', ' ').title()}<br>"
                       f"Doc {step.node.doc_id + 1}, Sentence {step.node.sentence_id + 1}<br>"
                       f"Similarity: {step.similarity_score:.3f}<br>"
                       f"Text: {step.node.text[:60]}...")
                hover_text.append(text)

                # Color code by connection type
                if step.connection_type == 'anchor':
                    colors.append('gold')
                elif step.connection_type == 'cross_doc':
                    colors.append('red')
                else:
                    colors.append('orange')

            # Add traversal line segments with different colors for cross-doc jumps
            for i in range(len(traversal_steps) - 1):
                current_step = traversal_steps[i]
                next_step = traversal_steps[i + 1]

                # Determine line style based on connection type
                if next_step.connection_type == 'cross_doc':
                    line_color = 'red'
                    line_width = 12
                    line_dash = 'solid'
                    opacity = 1.0
                else:
                    line_color = 'orange'
                    line_width = 8
                    line_dash = 'dash'
                    opacity = 0.8

                # Add individual line segment
                fig.add_trace(go.Scatter3d(
                    x=[x_path[i], x_path[i+1]],
                    y=[y_path[i], y_path[i+1]],
                    z=[z_path[i], z_path[i+1]],
                    mode='lines',
                    line=dict(color=line_color, width=line_width, dash=line_dash),
                    opacity=opacity,
                    showlegend=False,
                    hoverinfo='skip'
                ))

            # Add step markers
            fig.add_trace(go.Scatter3d(
                x=x_path, y=y_path, z=z_path,
                mode='markers+text',
                marker=dict(size=12, color=colors, line=dict(width=3, color='white')),
                text=[str(step.step_number) for step in traversal_steps],
                textfont=dict(size=12, color='white'),
                textposition='middle center',
                name='Traversal Steps',
                hovertext=hover_text,
                hovertemplate='%{hovertext}<extra></extra>'
            ))

        # Add query point (positioned in front of all documents)
        query_x = -self.layer_spacing * 0.5  # In front of first document
        query_y = 0
        query_z = 0
        fig.add_trace(go.Scatter3d(
            x=[query_x], y=[query_y], z=[query_z],
            mode='markers+text',
            marker=dict(size=20, color='gold', symbol='diamond',
                        line=dict(width=3, color='darkgoldenrod')),
            text=['QUERY'],
            textposition='top center',
            textfont=dict(size=18, color='darkgoldenrod'),
            name='Query Point',
            hovertemplate=f'<b>Query</b><br>{query}<extra></extra>'
        ))

        # Draw line from query to first traversal step
        if traversal_steps and x_path:
            fig.add_trace(go.Scatter3d(
                x=[query_x, x_path[0]],
                y=[query_y, y_path[0]],
                z=[query_z, z_path[0]],
                mode='lines',
                line=dict(color='gold', width=10, dash='dot'),
                showlegend=False,
                hoverinfo='skip'
            ))

        # Update layout for rotated standing documents
        fig.update_layout(
            title=dict(
                text=f'3D Semantic Graph Traversal - Rotated Documents<br><span style="font-size:14px">Query: "{query}"</span>',
                x=0.5,
                font=dict(size=20)
            ),
            scene=dict(
                xaxis_title='Document Position →',
                yaxis_title='← Document Width →',
                zaxis_title='↑ Document Height',
                camera=dict(
                    eye=dict(x=2.0, y=1.5, z=1.0)  # Better angle for rotated documents
                ),
                aspectmode='manual',
                aspectratio=dict(x=2, y=1, z=1),  # Adjusted for rotated documents
                xaxis=dict(showgrid=True, gridcolor='lightgray'),
                yaxis=dict(showgrid=True, gridcolor='lightgray'),
                zaxis=dict(showgrid=True, gridcolor='lightgray'),
            ),
            width=1200,
            height=800,
            font=dict(size=12),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        return fig

    def run_full_pipeline(self, query: str = "computer algorithm data analysis") -> go.Figure:
        """Run the complete pipeline from data generation to visualization"""

        print("🚀 Starting 3D Semantic Graph Visualization Pipeline")
        print("="*60)

        # Step 1: Generate synthetic documents
        print("📄 Generating synthetic documents...")
        documents = self.generate_synthetic_documents(num_docs=3, sentences_per_doc=5)
        self.documents = documents

        for i, doc in enumerate(documents):
            print(f"   Document {i+1}: {doc['title']}")
            for j, sent in enumerate(doc['sentences']):
                print(f"     Sentence {j+1}: {sent}")

        # Step 2: Create embeddings
        print("\n🧠 Creating embeddings...")
        document_embeddings = self.create_embeddings(documents)

        # Step 3: Build similarity matrices
        print("📊 Building similarity matrices...")
        self.similarity_matrices = self.build_similarity_matrices(document_embeddings)

        for i, matrix in enumerate(self.similarity_matrices):
            print(f"   Document {i+1} similarity matrix shape: {matrix.shape}")
            print(f"   Average similarity: {np.mean(matrix):.3f}")

        # Step 4: Simulate traversal
        print(f"\n🐍 Simulating traversal for query: '{query}'")
        traversal_steps = self.simulate_traversal(documents, document_embeddings, query)

        print(f"   Traversal path has {len(traversal_steps)} steps:")
        for step in traversal_steps:
            print(f"     Step {step.step_number}: {step.connection_type} -> "
                  f"Doc {step.node.doc_id+1}, Sent {step.node.sentence_id+1} "
                  f"(sim: {step.similarity_score:.3f})")

        # Step 5: Create 3D visualization
        print("\n🎨 Creating 3D visualization...")
        fig = self.create_3d_visualization(documents, traversal_steps, query)

        print("✅ Pipeline complete!")
        return fig

# Example usage
if __name__ == "__main__":
    # Create visualizer with horizontal spacing
    visualizer = SemanticGraph3DVisualizer(layer_spacing=6.0, heatmap_size=4.0)

    # Run pipeline
    fig = visualizer.run_full_pipeline(query="computer algorithm system analysis")

    # Display
    fig.show()

    # Optional: Save as HTML
    # fig.write_html("semantic_graph_3d_horizontal.html")

    print("\n💡 Visualization Tips:")
    print("   • Documents now stand vertically like pieces of paper!")
    print("   • Each colored square represents similarity between two sentences")
    print("   • Rotate the view to see different perspectives")
    print("   • Hover over squares to see exact similarity values")
    print("   • Red lines = cross-document jumps, Orange dashed = intra-document")
    print("   • Gold diamond shows the query anchor point")
    print("   • Traversal coordinates should now be precisely mapped!")