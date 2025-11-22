"""Pymorphy2 wrapper for lemmatization."""

import logging

import pymorphy2

logger = logging.getLogger(__name__)


class PyMorphyWrapper:
    """
    Wrapper for pymorphy2.MorphAnalyzer.

    Lazy initialization pattern to avoid loading morphological dictionary
    at module import time.
    """

    def __init__(self):
        """Initialize pymorphy2 analyzer."""
        logger.info("Initializing pymorphy2 MorphAnalyzer (this may take a few seconds)...")
        self.morph = pymorphy2.MorphAnalyzer()
        logger.info("Pymorphy2 initialized successfully")

    def parse(self, word: str):
        """
        Parse word and return morphological analysis.

        Args:
            word: Word to parse

        Returns:
            Parse result from pymorphy2
        """
        return self.morph.parse(word)

    def normal_form(self, word: str) -> str:
        """
        Get normal form (lemma) of a word.

        Args:
            word: Word to lemmatize

        Returns:
            Normal form of the word
        """
        parsed = self.morph.parse(word)[0]
        return parsed.normal_form
