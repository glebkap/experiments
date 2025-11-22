"""Pytest configuration and fixtures."""

import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


import pytest
from uuid import uuid4
from datetime import datetime

from domain.models.issue import Issue
from domain.models.preprocessed_issue import PreprocessedIssue


@pytest.fixture
def sample_issue():
    """Create a sample issue for testing."""
    return Issue(
        id=uuid4(),
        external_id="TEST-123",
        source_id=uuid4(),
        title="Test Issue Title",
        description="<p>This is a <b>test</b> description with HTML</p>",
        status="opened",
        priority=1,
        created_at=datetime.utcnow(),
        updated_at=None,
    )


@pytest.fixture
def sample_preprocessed_issue():
    """Create a sample preprocessed issue for testing."""
    return PreprocessedIssue(
        issue_id=uuid4(),
        content="test issue title test description html",
        processed_at=datetime.utcnow(),
    )
