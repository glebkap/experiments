"""Helper utilities for logging and debugging."""

import textwrap
from typing import Optional

from ....config import settings


def format_issue_content(
    content: str,
    max_length: Optional[int] = None,
    indent: int = 4,
) -> str:
    """
    Format issue content for readable logging.

    Args:
        content: Issue content to format
        max_length: Maximum length to show (None = use settings)
        indent: Number of spaces to indent each line

    Returns:
        Formatted content string
    """
    if not content:
        return "<empty>"

    # Use settings if max_length not provided
    if max_length is None:
        max_length = settings.debug_log_max_content_length

    # Truncate if needed
    if max_length > 0 and len(content) > max_length:
        content = content[:max_length] + "..."

    # Wrap text for readability (80 chars minus indent)
    wrapper = textwrap.TextWrapper(
        width=80 - indent,
        initial_indent=" " * indent,
        subsequent_indent=" " * indent,
        break_long_words=False,
        break_on_hyphens=False,
    )

    # Split by newlines and wrap each line
    lines = content.split("\n")
    formatted_lines = []

    for line in lines:
        if line.strip():  # Skip empty lines
            wrapped = wrapper.fill(line)
            formatted_lines.append(wrapped)
        else:
            formatted_lines.append(" " * indent + "")  # Preserve empty lines

    return "\n".join(formatted_lines)


def format_issue_debug_info(
    issue_id: str,
    title: Optional[str],
    description: Optional[str],
    status: Optional[str] = None,
    index: Optional[int] = None,
    total: Optional[int] = None,
) -> str:
    """
    Format issue information for debug logging.

    Args:
        issue_id: Issue UUID
        title: Issue title
        description: Issue description
        status: Issue status
        index: Current index (1-based)
        total: Total number of issues

    Returns:
        Formatted debug string with issue details
    """
    lines = []

    # Header
    header = f"Issue {issue_id}"
    if index and total:
        header += f" ({index}/{total})"
    lines.append("=" * 80)
    lines.append(header)
    lines.append("=" * 80)

    # Status
    if status:
        lines.append(f"Status: {status}")
        lines.append("-" * 80)

    # Title
    if title:
        lines.append("Title:")
        if settings.debug_log_issue_content:
            lines.append(format_issue_content(title))
        else:
            preview = title[:100] if len(title) > 100 else title
            lines.append(f"    {preview}{'...' if len(title) > 100 else ''}")
    else:
        lines.append("Title: <empty>")

    lines.append("-" * 80)

    # Description
    if description:
        lines.append(f"Description (length={len(description)}):")
        if settings.debug_log_issue_content:
            lines.append(format_issue_content(description))
        else:
            preview = description[:200] if len(description) > 200 else description
            lines.append(f"    {preview}{'...' if len(description) > 200 else ''}")
    else:
        lines.append("Description: <empty>")

    lines.append("=" * 80)

    return "\n".join(lines)


def format_preprocessed_content_debug(
    issue_id: str,
    original_title_len: int,
    original_desc_len: int,
    preprocessed_content: str,
    index: Optional[int] = None,
    total: Optional[int] = None,
) -> str:
    """
    Format preprocessed content for debug logging.

    Args:
        issue_id: Issue UUID
        original_title_len: Length of original title
        original_desc_len: Length of original description
        preprocessed_content: Preprocessed content
        index: Current index (1-based)
        total: Total number of issues

    Returns:
        Formatted debug string with preprocessing results
    """
    lines = []

    # Header
    header = f"Preprocessed Issue {issue_id}"
    if index and total:
        header += f" ({index}/{total})"
    lines.append("=" * 80)
    lines.append(header)
    lines.append("=" * 80)

    # Stats
    lines.append(f"Original lengths: title={original_title_len}, description={original_desc_len}")
    lines.append(f"Preprocessed length: {len(preprocessed_content)}")
    compression_ratio = len(preprocessed_content) / (original_title_len + original_desc_len) * 100
    lines.append(f"Compression: {compression_ratio:.1f}%")
    lines.append("-" * 80)

    # Content
    if settings.debug_log_issue_content:
        lines.append("Preprocessed content:")
        lines.append(format_issue_content(preprocessed_content))
    else:
        preview = preprocessed_content[:200] if len(preprocessed_content) > 200 else preprocessed_content
        lines.append("Preprocessed content (preview):")
        lines.append(f"    {preview}{'...' if len(preprocessed_content) > 200 else ''}")

    lines.append("=" * 80)

    return "\n".join(lines)
