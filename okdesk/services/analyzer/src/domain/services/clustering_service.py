"""Clustering service for grouping similar issues."""

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ClusteringService:
    """
    Service for clustering embeddings using various algorithms.

    Supports HDBSCAN and K-means clustering.
    """

    def cluster_hdbscan(
        self, embeddings: np.ndarray, min_cluster_size: int = 5, min_samples: int = 3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cluster embeddings using HDBSCAN algorithm.

        HDBSCAN automatically determines the number of clusters and handles noise.

        Args:
            embeddings: numpy array of shape (n_samples, n_features)
            min_cluster_size: Minimum size of clusters
            min_samples: Minimum number of samples in neighborhood

        Returns:
            Tuple of (labels, centroids)
            - labels: Cluster labels for each sample (-1 for outliers)
            - centroids: Centroid vectors for each cluster
        """
        import hdbscan

        logger.debug(f"Running HDBSCAN on {len(embeddings)} samples...")
        logger.debug(f"Parameters: min_cluster_size={min_cluster_size}, min_samples={min_samples}, metric='cosine'")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='cosine',  # Use cosine for normalized embeddings (better for semantic similarity)
            cluster_selection_method='eom',  # Excess of Mass
        )

        labels = clusterer.fit_predict(embeddings)

        # Count clusters
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_outliers = (labels == -1).sum()
        logger.debug(f"HDBSCAN found {n_clusters} clusters, {n_outliers} outliers")

        # Compute centroids for each cluster
        logger.debug("Computing cluster centroids...")
        centroids = self._compute_centroids(embeddings, labels)

        return labels, centroids

    def cluster_kmeans(
        self, embeddings: np.ndarray, n_clusters: Optional[int] = None, max_k: int = 20
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cluster embeddings using K-means algorithm.

        If n_clusters is not specified, finds optimal K using silhouette score.

        Args:
            embeddings: numpy array of shape (n_samples, n_features)
            n_clusters: Number of clusters (None for auto-detection)
            max_k: Maximum K to try for auto-detection

        Returns:
            Tuple of (labels, centroids)
            - labels: Cluster labels for each sample
            - centroids: Centroid vectors for each cluster
        """
        from sklearn.cluster import KMeans

        if n_clusters is None:
            logger.debug("Auto-detecting optimal number of clusters...")
            n_clusters = self._find_optimal_k(embeddings, max_k)
            logger.debug(f"Optimal K selected: {n_clusters}")

        logger.debug(f"Running K-means with {n_clusters} clusters...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        centroids = kmeans.cluster_centers_

        return labels, centroids

    def compute_distances(
        self, embeddings: np.ndarray, centroids: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine distances from embeddings to their cluster centroids.

        Args:
            embeddings: numpy array of shape (n_samples, n_features)
            centroids: numpy array of shape (n_clusters, n_features)
            labels: Cluster labels for each embedding

        Returns:
            numpy array of distances (0-1, where 0=identical, 1=opposite)
        """
        distances = np.zeros(len(embeddings))

        for i, (emb, label) in enumerate(zip(embeddings, labels)):
            if label == -1:  # Outlier
                distances[i] = 1.0
            else:
                centroid = centroids[label]
                # Cosine distance = 1 - cosine_similarity
                distances[i] = 1 - self._cosine_similarity(emb, centroid)

        return distances

    def _compute_centroids(self, embeddings: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Compute cluster centroids as mean of embeddings in each cluster.

        Args:
            embeddings: numpy array of shape (n_samples, n_features)
            labels: Cluster labels for each embedding

        Returns:
            numpy array of centroids of shape (n_clusters, n_features)
        """
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)  # Exclude outliers

        centroids = []
        for label in sorted(unique_labels):
            mask = labels == label
            centroid = embeddings[mask].mean(axis=0)
            # Normalize centroid
            centroid = centroid / np.linalg.norm(centroid)
            centroids.append(centroid)

        return np.array(centroids)

    def _find_optimal_k(self, embeddings: np.ndarray, max_k: int) -> int:
        """
        Find optimal number of clusters using silhouette score.

        Args:
            embeddings: numpy array of shape (n_samples, n_features)
            max_k: Maximum K to try

        Returns:
            Optimal number of clusters
        """
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        best_k = 2
        best_score = -1

        max_k = min(max_k, len(embeddings) - 1)
        logger.info(f"Testing K from 2 to {max_k} to find optimal number of clusters...")

        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            score = silhouette_score(embeddings, labels)

            logger.debug(f"K={k}: silhouette_score={score:.4f}")

            if score > best_score:
                best_score = score
                best_k = k

        logger.info(f"Optimal K found: {best_k} (score={best_score:.4f})")
        return best_k

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity (0 to 1)
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))
