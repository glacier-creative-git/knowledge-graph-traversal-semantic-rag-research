"""
Building Block 2: Distance Calculations
=======================================

This module implements distance calculation methods for semantic chunking.
These distance metrics form the foundation for boundary detection and optimization.

Key Mathematical Components:
- Euclidean distance matrices (for adaptive distance algorithms)
- Cosine distance calculations (for split distance algorithms)
- Peak detection for boundary identification
- Distance-based boundary detection methods
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from scipy.signal import find_peaks
from scipy.spatial.distance import pdist, squareform
import logging

logger = logging.getLogger(__name__)


@dataclass
class DistanceConfig:
    """Configuration for distance calculations and peak detection"""
    distance_metric: str = "euclidean"  # "euclidean", "cosine", "manhattan"
    normalize_distances: bool = True
    peak_detection_distance: int = 2  # Minimum distance between peaks
    peak_detection_prominence: float = 0.1  # Minimum prominence for peaks
    boundary_threshold_percentile: float = 75  # Percentile for boundary detection


class DistanceCalculator:
    """
    Core class for calculating distances between embeddings and detecting boundaries.

    This implements the mathematical foundations used in multiple chunking algorithms
    for identifying natural split points in documents.
    """

    def __init__(self, config: DistanceConfig = None):
        self.config = config or DistanceConfig()

    def calculate_euclidean_distances(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Calculate normalized Euclidean distance matrix.

        Used in adaptive distance algorithms for measuring semantic distance
        between sentence embeddings.

        Args:
            embeddings: (n_sentences, embedding_dim) array

        Returns:
            (n_sentences, n_sentences) distance matrix
        """
        logger.debug(f"Calculating Euclidean distances for {embeddings.shape[0]} embeddings")

        # Calculate pairwise Euclidean distances
        distances = np.zeros((len(embeddings), len(embeddings)))

        for i in range(len(embeddings)):
            for j in range(len(embeddings)):
                # Euclidean distance formula: sqrt(sum((a-b)^2))
                diff = embeddings[i] - embeddings[j]
                distances[i, j] = np.sqrt(np.sum(diff ** 2))

        # Normalize distances if requested
        if self.config.normalize_distances:
            max_distance = distances.max()
            if max_distance > 0:
                distances = distances / max_distance

        logger.debug(f"Euclidean distance range: {distances.min():.4f} - {distances.max():.4f}")
        return distances

    def calculate_cosine_distances(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Calculate cosine distance matrix (1 - cosine similarity).

        Used in split distance algorithms for spectral clustering and
        boundary detection based on semantic dissimilarity.

        Args:
            embeddings: (n_sentences, embedding_dim) array

        Returns:
            (n_sentences, n_sentences) distance matrix
        """
        logger.debug(f"Calculating cosine distances for {embeddings.shape[0]} embeddings")

        # Normalize embeddings for cosine similarity calculation
        normalized_embeddings = self._normalize_embeddings(embeddings)

        # Calculate cosine similarities
        similarities = np.dot(normalized_embeddings, normalized_embeddings.T)

        # Convert to distances: distance = 1 - similarity
        distances = 1 - similarities

        # Ensure non-negative distances (numerical stability)
        distances = np.maximum(distances, 0)

        logger.debug(f"Cosine distance range: {distances.min():.4f} - {distances.max():.4f}")
        return distances

    def calculate_manhattan_distances(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Calculate Manhattan (L1) distance matrix.

        Alternative distance metric that can be useful for certain types
        of semantic analysis.

        Args:
            embeddings: (n_sentences, embedding_dim) array

        Returns:
            (n_sentences, n_sentences) distance matrix
        """
        logger.debug(f"Calculating Manhattan distances for {embeddings.shape[0]} embeddings")

        distances = np.zeros((len(embeddings), len(embeddings)))

        for i in range(len(embeddings)):
            for j in range(len(embeddings)):
                # Manhattan distance: sum(|a-b|)
                distances[i, j] = np.sum(np.abs(embeddings[i] - embeddings[j]))

        # Normalize if requested
        if self.config.normalize_distances:
            max_distance = distances.max()
            if max_distance > 0:
                distances = distances / max_distance

        logger.debug(f"Manhattan distance range: {distances.min():.4f} - {distances.max():.4f}")
        return distances

    def calculate_distance_matrix(self, embeddings: np.ndarray,
                                  metric: Optional[str] = None) -> np.ndarray:
        """
        Calculate distance matrix using the specified metric.

        Args:
            embeddings: (n_sentences, embedding_dim) array
            metric: Distance metric to use (overrides config if provided)

        Returns:
            Distance matrix
        """
        metric = metric or self.config.distance_metric

        if metric == "euclidean":
            return self.calculate_euclidean_distances(embeddings)
        elif metric == "cosine":
            return self.calculate_cosine_distances(embeddings)
        elif metric == "manhattan":
            return self.calculate_manhattan_distances(embeddings)
        else:
            raise ValueError(f"Unknown distance metric: {metric}")

    def find_distance_peaks(self, distances: Union[np.ndarray, List[float]]) -> Tuple[np.ndarray, Dict]:
        """
        Find peaks in distance patterns for boundary detection.

        This is a core technique used to identify natural split points
        where semantic distance is locally maximum.

        Args:
            distances: 1D array of distances (e.g., adjacent sentence distances)

        Returns:
            (peak_indices, peak_properties)
        """
        distances_array = np.array(distances)

        # Find peaks using scipy's find_peaks
        peaks, properties = find_peaks(
            distances_array,
            distance=self.config.peak_detection_distance,
            prominence=self.config.peak_detection_prominence
        )

        logger.debug(f"Found {len(peaks)} peaks in distance pattern")

        return peaks, properties

    def detect_boundaries_from_distances(self, distance_matrix: np.ndarray) -> List[int]:
        """
        Detect potential chunk boundaries from a distance matrix.

        This analyzes adjacent sentence distances and identifies locations
        where semantic distance is significantly high, indicating natural
        boundaries between topics or concepts.

        Args:
            distance_matrix: (n_sentences, n_sentences) distance matrix

        Returns:
            List of sentence indices where boundaries should be placed
        """
        n_sentences = distance_matrix.shape[0]

        # Calculate distances between adjacent sentences
        adjacent_distances = []
        for i in range(n_sentences - 1):
            adjacent_distances.append(distance_matrix[i, i + 1])

        # Find peaks in adjacent distances
        peaks, properties = self.find_distance_peaks(adjacent_distances)

        # Convert peak indices to sentence boundaries
        # Peak at index i means boundary between sentence i and i+1
        boundaries = [peak + 1 for peak in peaks]  # +1 because we want the boundary position

        # Alternative method: threshold-based boundary detection
        threshold = np.percentile(adjacent_distances, self.config.boundary_threshold_percentile)
        threshold_boundaries = []
        for i, distance in enumerate(adjacent_distances):
            if distance >= threshold:
                threshold_boundaries.append(i + 1)

        # Combine both methods and remove duplicates
        all_boundaries = sorted(set(boundaries + threshold_boundaries))

        # Filter out boundaries too close to start/end
        filtered_boundaries = [b for b in all_boundaries if 1 < b < n_sentences - 1]

        logger.info(f"Detected {len(filtered_boundaries)} potential boundaries: {filtered_boundaries}")

        return filtered_boundaries

    def calculate_row_wise_distances(self, distance_matrix: np.ndarray) -> np.ndarray:
        """
        Calculate row-wise mean distances for spectral analysis.

        This computes the mean distance from each sentence to all others,
        which can be used for identifying outliers and natural groupings.

        Args:
            distance_matrix: (n_sentences, n_sentences) distance matrix

        Returns:
            1D array of mean distances for each sentence
        """
        # Calculate mean distance from each sentence to all others
        row_distances = np.mean(distance_matrix, axis=1)

        logger.debug(f"Row-wise distances range: {row_distances.min():.4f} - {row_distances.max():.4f}")

        return row_distances

    def calculate_semantic_flow(self, distance_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate semantic flow through the document.

        This analyzes how semantic distance changes as we move through
        the document, helping identify transition points and coherent regions.

        Args:
            distance_matrix: (n_sentences, n_sentences) distance matrix

        Returns:
            (positions, flow_values) where positions are relative document positions
            and flow_values are smoothed distance measures
        """
        n_sentences = distance_matrix.shape[0]

        # Calculate adjacent distances
        adjacent_distances = []
        positions = []

        for i in range(n_sentences - 1):
            distance = distance_matrix[i, i + 1]
            adjacent_distances.append(distance)
            position = (i + 0.5) / n_sentences  # Relative position
            positions.append(position)

        # Apply smoothing to reduce noise
        flow_values = self._smooth_signal(np.array(adjacent_distances))

        return np.array(positions), flow_values

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings to unit vectors for cosine similarity."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return embeddings / norms

    def _smooth_signal(self, signal: np.ndarray, window_size: int = 3) -> np.ndarray:
        """Apply simple moving average smoothing to a signal."""
        if len(signal) <= window_size:
            return signal

        smoothed = np.convolve(signal, np.ones(window_size) / window_size, mode='same')
        return smoothed


class BoundaryDetector:
    """
    Specialized class for detecting chunk boundaries using distance analysis.

    This implements advanced boundary detection techniques that combine
    multiple distance-based approaches for robust boundary identification.
    """

    def __init__(self, distance_calculator: DistanceCalculator = None):
        self.distance_calc = distance_calculator or DistanceCalculator()

    def detect_boundaries_multi_method(self, embeddings: np.ndarray,
                                       target_chunks: int) -> Dict[str, List[int]]:
        """
        Detect boundaries using multiple distance-based methods.

        Args:
            embeddings: Sentence embeddings
            target_chunks: Desired number of chunks

        Returns:
            Dictionary mapping method_name -> boundary_list
        """
        results = {}

        # Method 1: Euclidean distance peaks
        euclidean_matrix = self.distance_calc.calculate_euclidean_distances(embeddings)
        euclidean_boundaries = self.distance_calc.detect_boundaries_from_distances(euclidean_matrix)
        results['euclidean_peaks'] = euclidean_boundaries

        # Method 2: Cosine distance peaks
        cosine_matrix = self.distance_calc.calculate_cosine_distances(embeddings)
        cosine_boundaries = self.distance_calc.detect_boundaries_from_distances(cosine_matrix)
        results['cosine_peaks'] = cosine_boundaries

        # Method 3: Row-wise distance analysis
        row_distances = self.distance_calc.calculate_row_wise_distances(euclidean_matrix)
        row_peaks, _ = self.distance_calc.find_distance_peaks(row_distances)
        results['row_distance_peaks'] = row_peaks.tolist()

        # Method 4: Combined approach
        combined_boundaries = self._combine_boundary_methods(results, target_chunks)
        results['combined'] = combined_boundaries

        return results

    def _combine_boundary_methods(self, boundary_results: Dict[str, List[int]],
                                  target_chunks: int) -> List[int]:
        """
        Combine multiple boundary detection methods to get the best boundaries.

        Args:
            boundary_results: Results from different detection methods
            target_chunks: Desired number of chunks

        Returns:
            List of final boundary positions
        """
        # Collect all boundary candidates with weights
        boundary_votes = {}

        for method, boundaries in boundary_results.items():
            if method == 'combined':  # Skip to avoid recursion
                continue

            weight = 1.0
            if 'euclidean' in method:
                weight = 1.2  # Slightly prefer Euclidean
            elif 'cosine' in method:
                weight = 1.1

            for boundary in boundaries:
                if boundary not in boundary_votes:
                    boundary_votes[boundary] = 0
                boundary_votes[boundary] += weight

        # Sort boundaries by vote weight
        sorted_boundaries = sorted(boundary_votes.items(), key=lambda x: x[1], reverse=True)

        # Select top boundaries up to target_chunks - 1
        max_boundaries = target_chunks - 1
        selected_boundaries = [b[0] for b in sorted_boundaries[:max_boundaries]]

        return sorted(selected_boundaries)


# Utility functions for distance analysis
def calculate_distance_statistics(distance_matrix: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive statistics for a distance matrix."""
    stats = {
        'mean_distance': float(np.mean(distance_matrix)),
        'std_distance': float(np.std(distance_matrix)),
        'min_distance': float(np.min(distance_matrix)),
        'max_distance': float(np.max(distance_matrix)),
        'median_distance': float(np.median(distance_matrix)),
        'sparsity': float(np.sum(distance_matrix == 0) / distance_matrix.size)
    }
    return stats


def demo_distance_calculations():
    """Demonstrate the distance calculation building block."""
    print("=== Distance Calculations Building Block Demo ===\n")

    # Create sample embeddings with clear structure
    np.random.seed(42)
    n_sentences = 8
    embedding_dim = 100

    # Create structured embeddings (two distinct topics)
    embeddings = []
    for i in range(n_sentences):
        if i < 4:  # First topic
            base = np.array([1.0, 0.0])
        else:  # Second topic
            base = np.array([0.0, 1.0])

        # Add random components
        noise = np.random.normal(0, 0.1, embedding_dim - 2)
        full_embedding = np.concatenate([base, noise])
        embeddings.append(full_embedding)

    embeddings = np.array(embeddings)

    print(f"Created {n_sentences} embeddings with {embedding_dim} dimensions")
    print("Structure: Sentences 0-3 (Topic A), Sentences 4-7 (Topic B)")
    print()

    # Initialize distance calculator
    config = DistanceConfig(normalize_distances=True)
    calc = DistanceCalculator(config)

    # Calculate different distance matrices
    print("1. Calculating Euclidean distances...")
    euclidean_distances = calc.calculate_euclidean_distances(embeddings)
    euclidean_stats = calculate_distance_statistics(euclidean_distances)
    print(f"   Mean distance: {euclidean_stats['mean_distance']:.3f}")
    print(f"   Distance range: [{euclidean_stats['min_distance']:.3f}, {euclidean_stats['max_distance']:.3f}]")
    print()

    print("2. Calculating cosine distances...")
    cosine_distances = calc.calculate_cosine_distances(embeddings)
    cosine_stats = calculate_distance_statistics(cosine_distances)
    print(f"   Mean distance: {cosine_stats['mean_distance']:.3f}")
    print(f"   Distance range: [{cosine_stats['min_distance']:.3f}, {cosine_stats['max_distance']:.3f}]")
    print()

    # Detect boundaries
    print("3. Detecting boundaries from distances...")
    euclidean_boundaries = calc.detect_boundaries_from_distances(euclidean_distances)
    cosine_boundaries = calc.detect_boundaries_from_distances(cosine_distances)

    print(f"   Euclidean method boundaries: {euclidean_boundaries}")
    print(f"   Cosine method boundaries: {cosine_boundaries}")
    print()

    # Calculate semantic flow
    print("4. Analyzing semantic flow...")
    positions, flow_values = calc.calculate_semantic_flow(euclidean_distances)
    print(f"   Flow analysis complete: {len(flow_values)} flow points")
    print(f"   Flow range: [{flow_values.min():.3f}, {flow_values.max():.3f}]")
    print()

    # Multi-method boundary detection
    print("5. Multi-method boundary detection...")
    detector = BoundaryDetector(calc)
    boundary_results = detector.detect_boundaries_multi_method(embeddings, target_chunks=2)

    for method, boundaries in boundary_results.items():
        print(f"   {method}: {boundaries}")
    print()

    print("✅ Distance calculations building block complete!")
    print("   Ready for use in chunking algorithms.")

    return euclidean_distances, cosine_distances, boundary_results


if __name__ == "__main__":
    # Run demonstration
    euclidean_dist, cosine_dist, boundaries = demo_distance_calculations()