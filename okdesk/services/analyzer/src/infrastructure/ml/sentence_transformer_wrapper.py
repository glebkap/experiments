"""SentenceTransformer wrapper for embedding generation."""

import logging
from typing import List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from ...domain.services.embedding_generator import EmbeddingGenerator

logger = logging.getLogger(__name__)


class SentenceTransformerWrapper(EmbeddingGenerator):
    """
    Wrapper around SentenceTransformer library for generating embeddings.

    Uses pre-trained multilingual models optimized for semantic similarity.
    Supports CPU, CUDA (NVIDIA GPU), MPS (Apple Silicon), and auto device detection.
    """

    def __init__(self, model_name: str, device: str = "cpu"):
        """
        Initialize SentenceTransformer model.

        Args:
            model_name: HuggingFace model name (e.g., 'intfloat/multilingual-e5-large')
            device: Device to use ('cpu', 'cuda', 'mps', or 'auto')

        Raises:
            RuntimeError: If requested device is not available
            Exception: If model loading fails
        """
        # Auto-detect best available device
        if device == "auto":
            device = self._detect_best_device()
            logger.info(f"Auto-detected device: {device}")

        # Validate device availability
        device = self._validate_and_prepare_device(device)

        logger.info(f"Loading SentenceTransformer model: {model_name}")
        logger.info(f"Target device: {device}")

        self.model_name = model_name
        self.device = device

        try:
            self.model = SentenceTransformer(model_name, device=device)
            self.dimension = self.model.get_sentence_embedding_dimension()

            # Log device info
            if device == "cuda":
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                logger.info(
                    f"Successfully loaded model {model_name} on CUDA "
                    f"(GPU: {gpu_name}, Memory: {gpu_memory:.1f}GB, dim={self.dimension})"
                )
            elif device == "mps":
                logger.info(
                    f"Successfully loaded model {model_name} on Apple Silicon MPS "
                    f"(dim={self.dimension})"
                )
            else:
                logger.info(
                    f"Successfully loaded model {model_name} on CPU "
                    f"(dim={self.dimension})"
                )

        except Exception as e:
            logger.error(f"Failed to load model {model_name} on {device}: {e}")
            raise

    @staticmethod
    def _detect_best_device() -> str:
        """
        Auto-detect the best available device.

        Priority: cuda > mps > cpu

        Returns:
            Best available device string
        """
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    @staticmethod
    def _validate_and_prepare_device(device: str) -> str:
        """
        Validate device availability and prepare for use.

        Args:
            device: Requested device ('cpu', 'cuda', or 'mps')

        Returns:
            Validated device string

        Raises:
            RuntimeError: If requested device is not available
        """
        if device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA device requested but CUDA is not available. "
                    "Install PyTorch with CUDA support or use device='cpu'"
                )
            # Log CUDA info
            logger.info(f"CUDA available: {torch.cuda.device_count()} device(s)")
            for i in range(torch.cuda.device_count()):
                logger.info(f"  Device {i}: {torch.cuda.get_device_name(i)}")

        elif device == "mps":
            if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                raise RuntimeError(
                    "MPS device requested but MPS is not available. "
                    "Use device='cpu' or upgrade to macOS 12.3+ with Apple Silicon"
                )
            logger.info("Apple Silicon MPS available")

        elif device == "cpu":
            logger.info("Using CPU for embeddings (consider using GPU for better performance)")

        return device

    def generate_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of text strings
            batch_size: Batch size for encoding

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        if not texts:
            logger.warning("Empty text list provided")
            return np.array([])

        logger.debug(f"Generating embeddings for {len(texts)} texts")

        try:
            # For E5 models, prefix text with "query: " or "passage: "
            # Use "passage: " for indexing/storing, "query: " for searching
            if "e5" in self.model_name.lower():
                texts = [f"passage: {text}" for text in texts]

            # Generate embeddings with progress bar
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=len(texts) > 100,
                convert_to_numpy=True,
                normalize_embeddings=True,  # Normalize for cosine similarity
            )

            logger.info(f"Generated embeddings with shape {embeddings.shape}")

            return embeddings

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise

    def generate_single(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Input text string

        Returns:
            numpy array of shape (embedding_dim,)
        """
        embeddings = self.generate_batch([text], batch_size=1)
        return embeddings[0]

    def get_dimension(self) -> int:
        """
        Get embedding dimension.

        Returns:
            Embedding vector dimension
        """
        return self.dimension

    def encode_query(self, query_text: str) -> np.ndarray:
        """
        Encode text for similarity search queries.

        For E5 models, this prefixes with "query: " instead of "passage: ".

        Args:
            query_text: Query text

        Returns:
            numpy array of shape (embedding_dim,)
        """
        logger.debug(f"Encoding query: {query_text[:100]}...")

        try:
            # For E5 models, use "query: " prefix for search
            if "e5" in self.model_name.lower():
                query_text = f"query: {query_text}"

            embedding = self.model.encode(
                query_text,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

            return embedding

        except Exception as e:
            logger.error(f"Failed to encode query: {e}")
            raise
