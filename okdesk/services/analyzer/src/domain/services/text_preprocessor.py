"""Text preprocessing service for cleaning and normalizing text."""

import re
from typing import Set

from bs4 import BeautifulSoup


class TextPreprocessor:
    """
    Service for preprocessing text: cleaning HTML, normalization, lemmatization.

    This is a domain service that contains business logic for text preprocessing.
    """

    def __init__(self):
        """Initialize preprocessor with stop words."""
        self.stop_words = self._load_stop_words()
        # Pymorphy2 will be injected from infrastructure layer
        self._morph = None

    def set_morph_analyzer(self, morph):
        """
        Set pymorphy2 analyzer (dependency injection from infrastructure).

        Args:
            morph: pymorphy2.MorphAnalyzer instance
        """
        self._morph = morph

    def preprocess(self, text: str) -> str:
        """
        Full text preprocessing pipeline.

        Steps:
        1. Clean HTML
        2. Normalize patterns (URLs, emails, phones)
        3. Convert to lowercase
        4. Remove extra whitespace
        5. Lemmatize (if morph analyzer available)

        Args:
            text: Raw text to preprocess

        Returns:
            Preprocessed text
        """
        if not text:
            return ""

        # 1. Clean HTML
        text = self._clean_html(text)

        # 2. Normalize patterns
        text = self._normalize_patterns(text)

        # 3. Lowercase and cleanup
        text = text.lower().strip()

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # 4. Lemmatize
        if self._morph:
            text = self._lemmatize(text)

        # 5. Remove stop words
        text = self._remove_stop_words(text)

        return text.strip()

    def preprocess_issue(self, title: str | None, description: str | None) -> str:
        """
        Preprocess issue title and description.

        Args:
            title: Issue title (can be None)
            description: Issue description (can be None)

        Returns:
            Combined preprocessed text
        """
        title_clean = self.preprocess(title) if title else ""
        desc_clean = self.preprocess(description) if description else ""

        if title_clean and desc_clean:
            return f"{title_clean}\n\n{desc_clean}"
        return title_clean or desc_clean

    def _clean_html(self, text: str) -> str:
        """
        Remove HTML/XML tags from text.

        Args:
            text: Text with HTML

        Returns:
            Plain text
        """
        soup = BeautifulSoup(text, "lxml")
        return soup.get_text(separator=" ")

    def _normalize_patterns(self, text: str) -> str:
        """
        Normalize special patterns (URLs, emails, phones).

        Args:
            text: Text to normalize

        Returns:
            Text with normalized patterns
        """
        # URL
        text = re.sub(r'https?://\S+', '[URL]', text)

        # Email
        text = re.sub(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text
        )

        # Phone (простой паттерн)
        text = re.sub(r'\+?\d[\d\s\-\(\)]{7,}\d', '[PHONE]', text)

        # Version numbers (v1.2.3 -> версия)
        text = re.sub(r'v?\d+\.\d+(\.\d+)?', '[VERSION]', text)

        return text

    def _lemmatize(self, text: str) -> str:
        """
        Lemmatize text using pymorphy2.

        Args:
            text: Text to lemmatize

        Returns:
            Lemmatized text
        """
        if not self._morph:
            return text

        words = text.split()
        lemmas = []

        for word in words:
            # Keep special tokens unchanged
            if word.startswith('[') and word.endswith(']'):
                lemmas.append(word)
            else:
                # Lemmatize word
                parsed = self._morph.parse(word)[0]
                lemmas.append(parsed.normal_form)

        return " ".join(lemmas)

    def _remove_stop_words(self, text: str) -> str:
        """
        Remove stop words from text.

        Args:
            text: Text with stop words

        Returns:
            Text without stop words
        """
        words = text.split()
        filtered = [w for w in words if w not in self.stop_words and len(w) > 2]
        return " ".join(filtered)

    def _load_stop_words(self) -> Set[str]:
        """
        Load Russian stop words.

        Returns:
            Set of stop words
        """
        # Базовый набор русских стоп-слов
        return {
            "и",
            "в",
            "на",
            "с",
            "по",
            "для",
            "к",
            "от",
            "из",
            "о",
            "что",
            "это",
            "как",
            "же",
            "бы",
            "ли",
            "до",
            "про",
            "при",
            "за",
            "над",
            "под",
            "или",
            "а",
            "но",
            "да",
            "нет",
            "не",
            "ни",
            "то",
            "так",
            "вот",
            "быть",
            "есть",
            "был",
            "была",
            "были",
            "будет",
            "будут",
            "мой",
            "моя",
            "мое",
            "мои",
            "твой",
            "твоя",
            "твое",
            "твои",
            "наш",
            "наша",
            "наше",
            "наши",
            "ваш",
            "ваша",
            "ваше",
            "ваши",
            "его",
            "ее",
            "их",
            "этот",
            "эта",
            "это",
            "эти",
            "тот",
            "та",
            "те",
            "весь",
            "вся",
            "все",
            "кто",
            "где",
            "когда",
            "куда",
            "откуда",
            "почему",
            "зачем",
            "сколько",
            "который",
            "которая",
            "которое",
            "которые",
        }
