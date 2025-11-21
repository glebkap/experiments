"""Unit tests for domain-database mappers."""

from datetime import datetime
from uuid import uuid4

from src.domain.models import (
    Source,
    SourceType,
    Issue,
    IssueStatus,
    Message,
    AuthorType,
    ImportJob,
    ImportStatus,
)
from src.infrastructure.persistence.mappers import (
    SourceMapper,
    IssueMapper,
    MessageMapper,
    ImportJobMapper,
)
from src.infrastructure.persistence.models import (
    SourceModel,
    IssueModel,
    MessageModel,
    ImportModel,
)


class TestSourceMapper:
    """Tests for SourceMapper."""

    def test_to_domain(self):
        """Test converting SQLAlchemy model to domain entity."""
        model = SourceModel(
            id=uuid4(),
            name="Test Source",
            type=SourceType.OKDESK,
            config={"api_key": "test123"},
            created_at=datetime(2024, 11, 21, 10, 0, 0),
        )

        entity = SourceMapper.to_domain(model)

        assert isinstance(entity, Source)
        assert entity.id == model.id
        assert entity.name == "Test Source"
        assert entity.type == SourceType.OKDESK
        assert entity.config == {"api_key": "test123"}
        assert entity.created_at == model.created_at

    def test_to_model(self):
        """Test converting domain entity to SQLAlchemy model."""
        entity = Source(
            id=uuid4(),
            name="Test Source",
            type=SourceType.TELEGRAM,
            config=None,
            created_at=datetime(2024, 11, 21, 10, 0, 0),
        )

        model = SourceMapper.to_model(entity)

        assert isinstance(model, SourceModel)
        assert model.id == entity.id
        assert model.name == "Test Source"
        assert model.type == SourceType.TELEGRAM
        assert model.config is None

    def test_to_domain_to_model_round_trip(self):
        """Test round trip: model -> domain -> model."""
        original_model = SourceModel(
            id=uuid4(),
            name="Round Trip",
            type=SourceType.OKDESK,
            config={"test": "data"},
            created_at=datetime(2024, 11, 21),
        )

        entity = SourceMapper.to_domain(original_model)
        new_model = SourceMapper.to_model(entity)

        assert new_model.id == original_model.id
        assert new_model.name == original_model.name
        assert new_model.type == original_model.type


class TestIssueMapper:
    """Tests for IssueMapper."""

    def test_to_domain_full_data(self):
        """Test converting model with all fields populated."""
        model = IssueModel(
            id=uuid4(),
            external_id="EXT-123",
            source_id=uuid4(),
            title="Test Issue",
            description="Description",
            status=IssueStatus.OPENED,
            priority=2,
            created_at=datetime(2024, 11, 21, 10, 0),
            updated_at=datetime(2024, 11, 21, 11, 0),
            completed_at=None,
        )

        entity = IssueMapper.to_domain(model)

        assert isinstance(entity, Issue)
        assert entity.external_id == "EXT-123"
        assert entity.status == IssueStatus.OPENED
        assert entity.priority == 2

    def test_to_domain_minimal_data(self):
        """Test converting model with minimal fields."""
        model = IssueModel(
            id=uuid4(),
            external_id="EXT-456",
            source_id=uuid4(),
            title=None,
            description=None,
            status=None,
            priority=None,
            created_at=None,
            updated_at=None,
            completed_at=None,
        )

        entity = IssueMapper.to_domain(model)

        assert isinstance(entity, Issue)
        assert entity.external_id == "EXT-456"
        assert entity.title is None
        assert entity.status is None

    def test_to_model(self):
        """Test converting domain entity to model."""
        entity = Issue(
            id=uuid4(),
            external_id="DOM-789",
            source_id=uuid4(),
            title="Domain Issue",
            description="Test",
            status=IssueStatus.COMPLETED,
            priority=1,
            created_at=datetime(2024, 11, 20),
            updated_at=datetime(2024, 11, 21),
            completed_at=datetime(2024, 11, 21, 12, 0),
        )

        model = IssueMapper.to_model(entity)

        assert isinstance(model, IssueModel)
        assert model.external_id == "DOM-789"
        assert model.status == IssueStatus.COMPLETED
        assert model.completed_at is not None


class TestMessageMapper:
    """Tests for MessageMapper."""

    def test_to_domain_complete_message(self):
        """Test converting complete message."""
        model = MessageModel(
            id=uuid4(),
            issue_id=uuid4(),
            external_id="MSG-001",
            author_id="user-123",
            author_name="John Doe",
            author_type=AuthorType.EMPLOYEE,
            content="Test message content",
            is_public=True,
            published_at=datetime(2024, 11, 21, 10, 30),
        )

        entity = MessageMapper.to_domain(model)

        assert isinstance(entity, Message)
        assert entity.external_id == "MSG-001"
        assert entity.author_name == "John Doe"
        assert entity.author_type == AuthorType.EMPLOYEE
        assert entity.content == "Test message content"
        assert entity.is_public is True

    def test_to_domain_minimal_message(self):
        """Test converting minimal message."""
        model = MessageModel(
            id=uuid4(),
            issue_id=uuid4(),
            external_id="MSG-002",
            author_id=None,
            author_name=None,
            author_type=None,
            content="Minimal message",
            is_public=False,
            published_at=None,
        )

        entity = MessageMapper.to_domain(model)

        assert isinstance(entity, Message)
        assert entity.author_id is None
        assert entity.author_type is None
        assert entity.is_public is False

    def test_to_model(self):
        """Test converting domain entity to model."""
        entity = Message(
            id=uuid4(),
            issue_id=uuid4(),
            external_id="MSG-003",
            author_id="auth-456",
            author_name="Jane Smith",
            author_type=AuthorType.CONTACT,
            content="Another message",
            is_public=True,
            published_at=datetime(2024, 11, 21, 14, 0),
        )

        model = MessageMapper.to_model(entity)

        assert isinstance(model, MessageModel)
        assert model.external_id == "MSG-003"
        assert model.author_type == AuthorType.CONTACT


class TestImportJobMapper:
    """Tests for ImportJobMapper."""

    def test_to_domain_in_progress(self):
        """Test converting import job in progress."""
        model = ImportModel(
            id=uuid4(),
            source_id=uuid4(),
            filename="test.jsonl",
            file_path="/data/test.jsonl",
            started_at=datetime(2024, 11, 21, 10, 0),
            completed_at=None,
            status=ImportStatus.IN_PROGRESS,
            stats=None,
            error_message=None,
        )

        entity = ImportJobMapper.to_domain(model)

        assert isinstance(entity, ImportJob)
        assert entity.status == ImportStatus.IN_PROGRESS
        assert entity.completed_at is None
        assert entity.stats is None

    def test_to_domain_completed(self):
        """Test converting completed import job."""
        stats = {"total": 100, "new": 80, "updated": 20}
        model = ImportModel(
            id=uuid4(),
            source_id=uuid4(),
            filename="completed.jsonl",
            file_path="/data/completed.jsonl",
            started_at=datetime(2024, 11, 21, 10, 0),
            completed_at=datetime(2024, 11, 21, 10, 30),
            status=ImportStatus.COMPLETED,
            stats=stats,
            error_message=None,
        )

        entity = ImportJobMapper.to_domain(model)

        assert isinstance(entity, ImportJob)
        assert entity.status == ImportStatus.COMPLETED
        assert entity.completed_at is not None
        assert entity.stats == stats

    def test_to_domain_failed(self):
        """Test converting failed import job."""
        model = ImportModel(
            id=uuid4(),
            source_id=uuid4(),
            filename="failed.jsonl",
            file_path="/data/failed.jsonl",
            started_at=datetime(2024, 11, 21, 10, 0),
            completed_at=datetime(2024, 11, 21, 10, 5),
            status=ImportStatus.FAILED,
            stats=None,
            error_message="File not found",
        )

        entity = ImportJobMapper.to_domain(model)

        assert isinstance(entity, ImportJob)
        assert entity.status == ImportStatus.FAILED
        assert entity.error_message == "File not found"

    def test_to_model(self):
        """Test converting domain entity to model."""
        entity = ImportJob(
            id=uuid4(),
            source_id=uuid4(),
            filename="new.jsonl",
            file_path="/tmp/new.jsonl",
            started_at=datetime(2024, 11, 21, 15, 0),
            completed_at=None,
            status=ImportStatus.IN_PROGRESS,
            stats=None,
            error_message=None,
        )

        model = ImportJobMapper.to_model(entity)

        assert isinstance(model, ImportModel)
        assert model.filename == "new.jsonl"
        assert model.status == ImportStatus.IN_PROGRESS
