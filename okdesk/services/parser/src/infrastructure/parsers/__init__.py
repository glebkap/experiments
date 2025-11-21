"""Parsers for external data sources."""

from .okdesk_parser import OKDeskParser
from .telegram_parser import TelegramParser

__all__ = [
    "OKDeskParser",
    "TelegramParser",
]
