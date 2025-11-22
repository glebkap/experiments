"""Unit tests for domain models."""

from datetime import datetime
from uuid import uuid4

import numpy as np
import pytest

from domain.models.cluster import Cluster
from domain.models.embedding import Embedding
from domain.models.issue import Issue
from domain.models.message import Message
from domain.models.preprocessed_issue import PreprocessedIssue


class TestIssue:
    """Test suite for Issue model."""

    def test_create_issue(self):
        """Test issue creation."""
        issue = Issue(
            id=uuid4(),
            external_id="TEST-123",
            source_id=uuid4(),
            title="Test Title",
            description="Test Description",
            status="opened",
            priority=1,
            created_at=datetime.utcnow(),
            updated_at=None,
        )

        assert issue.title == "Test Title"
        assert issue.status == "opened"

    def test_get_combined_text(self):
        """Test combining title and description."""
        issue = Issue(
            id=uuid4(),
            external_id="TEST-123",
            source_id=uuid4(),
            title="Test Title",
            description="Test Description",
            status="opened",
            priority=1,
            created_at=datetime.utcnow(),
            updated_at=None,
        )

        combined = issue.get_combined_text()
        assert "Test Title" in combined
        assert "Test Description" in combined

    def test_get_combined_text_none_fields(self):
        """Test combining when fields are None."""
        issue = Issue(
            id=uuid4(),
            external_id="TEST-123",
            source_id=uuid4(),
            title=None,
            description="Only Description",
            status="opened",
            priority=None,
            created_at=datetime.utcnow(),
            updated_at=None,
        )

        combined = issue.get_combined_text()
        assert combined == "Only Description"


class TestEmbedding:
    """Test suite for Embedding model."""

    def test_create_embedding(self):
        """Test embedding creation."""
        vector = np.random.randn(1024)
        embedding = Embedding(
            issue_id=uuid4(),
            vector=vector,
            dimension=1024,
        )

        assert embedding.dimension == 1024
        assert embedding.vector.shape == (1024,)

    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        vec1 = np.array([1, 0, 0])
        vec2 = np.array([1, 0, 0])
        vec3 = np.array([0, 1, 0])

        emb1 = Embedding(issue_id=uuid4(), vector=vec1, dimension=3)
        emb2 = Embedding(issue_id=uuid4(), vector=vec2, dimension=3)
        emb3 = Embedding(issue_id=uuid4(), vector=vec3, dimension=3)

        # Identical vectors
        assert np.isclose(emb1.cosine_similarity(emb2), 1.0)

        # Orthogonal vectors
        assert np.isclose(emb1.cosine_similarity(emb3), 0.0)


class TestCluster:
    """Test suite for Cluster model."""

    def test_create_cluster(self):
        """Test cluster creation."""
        centroid = np.random.randn(1024)
        cluster = Cluster(
            id=uuid4(),
            cluster_label=0,
            name="Test Cluster",
            description=None,
            centroid_embedding=centroid,
            size=10,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        assert cluster.size == 10
        assert cluster.centroid_embedding.shape == (1024,)
        assert cluster.get_dimension() == 1024

    def test_is_outlier_cluster(self):
        """Test outlier detection."""
        centroid = np.random.randn(1024)
        cluster = Cluster(
            id=uuid4(),
            cluster_label=-1,
            name=None,
            description=None,
            centroid_embedding=centroid,
            size=0,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        assert cluster.is_outlier_cluster()

    def test_centroid_to_list(self):
        """Test centroid conversion to list."""
        centroid = np.array([1.0, 2.0, 3.0])
        cluster = Cluster(
            id=uuid4(),
            cluster_label=0,
            name="Test",
            description=None,
            centroid_embedding=centroid,
            size=5,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        centroid_list = cluster.centroid_to_list()
        assert centroid_list == [1.0, 2.0, 3.0]
        assert isinstance(centroid_list, list)


class TestPreprocessedIssue:
    """Test suite for PreprocessedIssue model."""

    def test_create_preprocessed_issue(self):
        """Test preprocessed issue creation."""
        content = "preprocessed text content"
        preprocessed = PreprocessedIssue(
            id=uuid4(),
            content=content,
            processed_at=datetime.utcnow(),
        )

        assert preprocessed.content == content
        assert preprocessed.get_word_count() == 3
        assert preprocessed.get_char_count() == len(content)

    def test_word_count(self):
        """Test word count calculation."""
        preprocessed = PreprocessedIssue(
            id=uuid4(),
            content="one two three four five",
            processed_at=datetime.utcnow(),
        )

        assert preprocessed.get_word_count() == 5

    def test_validate_empty_content_raises_error(self):
        """Test validation with empty content raises error."""
        with pytest.raises(ValueError, match="content cannot be empty"):
            PreprocessedIssue(
                id=uuid4(),
                content="",
                processed_at=datetime.utcnow(),
            )

    def test_validate_whitespace_content_raises_error(self):
        """Test validation with whitespace-only content raises error."""
        with pytest.raises(ValueError, match="content cannot be empty"):
            PreprocessedIssue(
                id=uuid4(),
                content="   ",
                processed_at=datetime.utcnow(),
            )
