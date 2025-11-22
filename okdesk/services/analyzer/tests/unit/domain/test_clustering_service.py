"""Unit tests for ClusteringService."""

import numpy as np
import pytest

from domain.services.clustering_service import ClusteringService


class TestClusteringService:
    """Test suite for ClusteringService."""

    @pytest.fixture
    def clustering_service(self):
        """Create ClusteringService instance."""
        return ClusteringService()

    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings for testing."""
        # Create 3 clusters with 5 points each
        np.random.seed(42)
        cluster1 = np.random.randn(5, 10) + np.array([10, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        cluster2 = np.random.randn(5, 10) + np.array([0, 10, 0, 0, 0, 0, 0, 0, 0, 0])
        cluster3 = np.random.randn(5, 10) + np.array([0, 0, 10, 0, 0, 0, 0, 0, 0, 0])

        return np.vstack([cluster1, cluster2, cluster3])

    def test_cluster_hdbscan(self, clustering_service, sample_embeddings):
        """Test HDBSCAN clustering."""
        labels, centroids = clustering_service.cluster_hdbscan(
            sample_embeddings, min_cluster_size=3, min_samples=2
        )

        # Check that we got labels
        assert len(labels) == len(sample_embeddings)
        assert isinstance(labels, np.ndarray)

        # Check centroids shape
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        assert centroids.shape[0] == num_clusters
        assert centroids.shape[1] == sample_embeddings.shape[1]

    def test_cluster_kmeans(self, clustering_service, sample_embeddings):
        """Test K-means clustering with specified K."""
        n_clusters = 3
        labels, centroids = clustering_service.cluster_kmeans(
            sample_embeddings, n_clusters=n_clusters
        )

        # Check labels
        assert len(labels) == len(sample_embeddings)
        assert len(set(labels)) == n_clusters

        # Check centroids
        assert centroids.shape == (n_clusters, sample_embeddings.shape[1])

    def test_cluster_kmeans_auto_k(self, clustering_service, sample_embeddings):
        """Test K-means clustering with auto K detection."""
        labels, centroids = clustering_service.cluster_kmeans(
            sample_embeddings, n_clusters=None, max_k=5
        )

        # Check that K was determined
        n_clusters = len(set(labels))
        assert 2 <= n_clusters <= 5
        assert centroids.shape[0] == n_clusters

    def test_compute_distances(self, clustering_service, sample_embeddings):
        """Test distance computation."""
        labels, centroids = clustering_service.cluster_kmeans(
            sample_embeddings, n_clusters=3
        )

        distances = clustering_service.compute_distances(
            sample_embeddings, centroids, labels
        )

        # Check distances shape
        assert len(distances) == len(sample_embeddings)

        # Check distances are valid (0-1 for cosine distance)
        assert np.all(distances >= 0)
        assert np.all(distances <= 1)

    def test_cosine_similarity(self, clustering_service):
        """Test cosine similarity calculation."""
        vec1 = np.array([1, 0, 0])
        vec2 = np.array([1, 0, 0])
        vec3 = np.array([0, 1, 0])

        # Identical vectors
        sim = clustering_service._cosine_similarity(vec1, vec2)
        assert np.isclose(sim, 1.0)

        # Orthogonal vectors
        sim = clustering_service._cosine_similarity(vec1, vec3)
        assert np.isclose(sim, 0.0)
