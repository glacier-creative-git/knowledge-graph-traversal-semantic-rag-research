"""
Building Block 1: Similarity Matrices
=====================================

This module implements the core similarity matrix building block for semantic chunking.
Based on research into multi-window similarity matrices with enhancement techniques.

Key Mathematical Innovations:
- Multi-window similarity matrices (1, 2, 3 sentence windows)
- Matrix enhancement techniques for boundary detection
- Edge detection and block measure analysis
- Normalization and preprocessing methods
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SimilarityMatrixConfig:
    """Configuration for similarity matrix creation and enhancement"""
    window_sizes: List[int] = None
    enhancement_enabled: bool = True
    normalization_method: str = "l2"  # "l2", "minmax", "zscore"
    edge_detection_sigma: float = 1.0
    structure_tensor_sigma: float = 1.0

    def __post_init__(self):
        if self.window_sizes is None:
            self.window_sizes = [1, 2, 3]


class SimilarityMatrixBuilder:
    """
    Core class for building and enhancing similarity matrices for semantic chunking.

    This implements the fundamental mathematical building block that underpins
    multiple chunking algorithms.
    """

    def __init__(self, config: SimilarityMatrixConfig = None):
        self.config = config or SimilarityMatrixConfig()

    def create_basic_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Create a basic similarity matrix using dot product similarity.

        Args:
            embeddings: (n_sentences, embedding_dim) array of sentence embeddings

        Returns:
            (n_sentences, n_sentences) similarity matrix
        """
        # Normalize embeddings for cosine similarity
        normalized_embeddings = self._normalize_embeddings(embeddings)

        # Compute dot product similarity (cosine similarity for normalized vectors)
        similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)

        logger.debug(f"Created basic similarity matrix of shape {similarity_matrix.shape}")
        logger.debug(f"Similarity range: {similarity_matrix.min():.4f} - {similarity_matrix.max():.4f}")

        return similarity_matrix

    def create_multi_window_matrices(self, sentences: List[str],
                                     embedding_function) -> Dict[int, np.ndarray]:
        """
        Create similarity matrices for multiple window sizes.

        This is the core innovation: instead of just sentence-to-sentence similarity,
        we compute similarities between overlapping windows of different sizes.

        Args:
            sentences: List of sentence strings
            embedding_function: Function that takes text and returns embeddings

        Returns:
            Dictionary mapping window_size -> similarity_matrix
        """
        results = {}
        n_sentences = len(sentences)

        for window_size in self.config.window_sizes:
            logger.info(f"Processing {window_size}-sentence windows")

            # Create overlapping windows
            windows = self._create_padded_windows(sentences, window_size)
            logger.info(f"Created {len(windows)} windows for size {window_size}")

            # Get embeddings for windows
            embeddings = embedding_function(windows)

            # Create similarity matrix for this window size
            similarity_matrix = np.zeros((n_sentences, n_sentences))

            # Calculate window-to-window similarities
            for i in range(len(windows)):
                for j in range(len(windows)):
                    if i < n_sentences and j < n_sentences:
                        # Compute similarity between window i and window j
                        sim_score = self._cosine_similarity(embeddings[i], embeddings[j])
                        similarity_matrix[i, j] = sim_score

            results[window_size] = similarity_matrix

        return results

    def enhance_similarity_matrix(self, similarity_matrix: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Apply enhancement techniques to improve boundary detection.

        This implements your research innovations for matrix enhancement:
        - Edge detection for boundary identification
        - Block measure analysis for coherent regions
        - Structure tensor analysis

        Args:
            similarity_matrix: Basic similarity matrix

        Returns:
            enhanced_matrix, enhancement_data
        """
        if not self.config.enhancement_enabled:
            return similarity_matrix, {}

        logger.info("Applying matrix enhancement techniques")

        # Start with the original matrix
        enhanced = similarity_matrix.copy()

        # Apply enhancement techniques (simplified versions of your innovations)
        enhancement_data = {}

        # 1. Edge detection for boundary identification
        edge_magnitude = self._detect_edges(enhanced)
        enhancement_data['edge_magnitude'] = edge_magnitude

        # 2. Block measure analysis
        block_measure = self._calculate_block_measure(enhanced)
        enhancement_data['block_measure'] = block_measure

        # 3. Combine enhancements
        # Weight the original similarity with edge and block information
        alpha, beta, gamma = 0.7, 0.2, 0.1  # Hyperparameters for combination

        enhanced = (alpha * enhanced +
                    beta * (1 - edge_magnitude) +  # Lower similarity at edges
                    gamma * block_measure)  # Higher similarity in blocks

        # Normalize the enhanced matrix
        enhanced = self._normalize_matrix(enhanced)

        logger.debug(f"Enhanced matrix range: {enhanced.min():.4f} - {enhanced.max():.4f}")

        return enhanced, enhancement_data

    def _create_padded_windows(self, sentences: List[str], window_size: int) -> List[str]:
        """Create overlapping windows with padding for consistent matrix dimensions."""
        windows = []
        n_sentences = len(sentences)

        for i in range(n_sentences):
            # Create window centered at position i
            start_idx = max(0, i - window_size // 2)
            end_idx = min(n_sentences, start_idx + window_size)

            # Adjust start if we're near the end
            if end_idx - start_idx < window_size:
                start_idx = max(0, end_idx - window_size)

            window_sentences = sentences[start_idx:end_idx]

            # Pad if necessary (for edge cases)
            while len(window_sentences) < window_size and len(window_sentences) > 0:
                if start_idx == 0:
                    window_sentences.append(window_sentences[-1])  # Pad with last sentence
                else:
                    window_sentences.insert(0, window_sentences[0])  # Pad with first sentence

            windows.append(' '.join(window_sentences))

        return windows

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings based on the configured method."""
        if self.config.normalization_method == "l2":
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            return embeddings / norms
        elif self.config.normalization_method == "minmax":
            min_vals = embeddings.min(axis=1, keepdims=True)
            max_vals = embeddings.max(axis=1, keepdims=True)
            return (embeddings - min_vals) / (max_vals - min_vals + 1e-8)
        elif self.config.normalization_method == "zscore":
            mean_vals = embeddings.mean(axis=1, keepdims=True)
            std_vals = embeddings.std(axis=1, keepdims=True)
            return (embeddings - mean_vals) / (std_vals + 1e-8)
        else:
            return embeddings

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return np.dot(vec1, vec2) / (norm1 * norm2)

    def _detect_edges(self, matrix: np.ndarray) -> np.ndarray:
        """
        Detect edges in the similarity matrix for boundary identification.

        Simplified version of edge detection - in practice, you might use
        more sophisticated techniques like Sobel filters or gradient analysis.
        """
        # Simple gradient-based edge detection
        grad_x = np.gradient(matrix, axis=1)
        grad_y = np.gradient(matrix, axis=0)

        edge_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

        # Normalize to [0, 1]
        if edge_magnitude.max() > 0:
            edge_magnitude = edge_magnitude / edge_magnitude.max()

        return edge_magnitude

    def _calculate_block_measure(self, matrix: np.ndarray) -> np.ndarray:
        """
        Calculate block measure for identifying coherent regions.

        This is a simplified version of structure tensor analysis
        for identifying block-like structures in the similarity matrix.
        """
        # Simple block detection using local variance
        kernel_size = 3
        block_measure = np.zeros_like(matrix)

        for i in range(kernel_size // 2, matrix.shape[0] - kernel_size // 2):
            for j in range(kernel_size // 2, matrix.shape[1] - kernel_size // 2):
                # Extract local neighborhood
                neighborhood = matrix[i - kernel_size // 2:i + kernel_size // 2 + 1,
                               j - kernel_size // 2:j + kernel_size // 2 + 1]

                # Calculate local coherence (inverse of variance)
                local_var = np.var(neighborhood)
                block_measure[i, j] = 1.0 / (1.0 + local_var)

        return block_measure

    def _normalize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Normalize matrix to [0, 1] range."""
        min_val = matrix.min()
        max_val = matrix.max()

        if max_val - min_val == 0:
            return matrix

        return (matrix - min_val) / (max_val - min_val)


# Example usage and testing functions
def create_sample_embeddings(n_sentences: int, embedding_dim: int = 384) -> np.ndarray:
    """Create sample embeddings for testing purposes."""
    np.random.seed(42)  # For reproducibility
    return np.random.randn(n_sentences, embedding_dim)


def demo_similarity_matrices():
    """Demonstrate the similarity matrix building block."""
    print("=== Similarity Matrix Building Block Demo ===\n")

    # Sample sentences for demonstration
    sample_sentences = [
        "The cat sat on the mat.",
        "A feline rested on the rug.",
        "Machine learning is a subset of artificial intelligence.",
        "AI systems can learn from data.",
        "The weather today is sunny and warm.",
        "It's a beautiful day outside.",
        "Python is a popular programming language.",
        "Many data scientists use Python for analysis."
    ]

    print(f"Sample document with {len(sample_sentences)} sentences:")
    for i, sentence in enumerate(sample_sentences):
        print(f"  {i + 1}. {sentence}")
    print()

    # Create sample embeddings (in practice, you'd use a real embedding model)
    embeddings = create_sample_embeddings(len(sample_sentences))

    # Initialize the similarity matrix builder
    config = SimilarityMatrixConfig(window_sizes=[1, 2, 3])
    builder = SimilarityMatrixBuilder(config)

    print("1. Creating basic similarity matrix...")
    basic_matrix = builder.create_basic_similarity_matrix(embeddings)
    print(f"   Shape: {basic_matrix.shape}")
    print(f"   Range: [{basic_matrix.min():.3f}, {basic_matrix.max():.3f}]")
    print()

    print("2. Enhancing similarity matrix...")
    enhanced_matrix, enhancement_data = builder.enhance_similarity_matrix(basic_matrix)
    print(f"   Enhanced range: [{enhanced_matrix.min():.3f}, {enhanced_matrix.max():.3f}]")
    print(f"   Enhancement data keys: {list(enhancement_data.keys())}")
    print()

    print("3. Matrix comparison (first 4x4 block):")
    print("   Basic matrix:")
    print(f"   {basic_matrix[:4, :4]}")
    print("   Enhanced matrix:")
    print(f"   {enhanced_matrix[:4, :4]}")

    return basic_matrix, enhanced_matrix, enhancement_data


if __name__ == "__main__":
    # Run the demonstration
    basic_matrix, enhanced_matrix, enhancement_data = demo_similarity_matrices()