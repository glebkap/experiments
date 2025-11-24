"""ChromaDB client for vector storage and similarity search."""

import logging
from typing import List, Optional, Tuple
from uuid import UUID

import chromadb
import numpy as np
from chromadb.api.models.Collection import Collection

from ...domain.services.vector_db_service import VectorDBService

logger = logging.getLogger(__name__)


class ChromaDBClient(VectorDBService):
    """
    ChromaDB client implementation for vector database operations.

    Handles storage and retrieval of embeddings for semantic similarity search.
    """

    def __init__(self, host: str, port: int, collection_name: str):
        """
        Initialize ChromaDB client.

        Args:
            host: ChromaDB server host
            port: ChromaDB server port
            collection_name: Name of the collection to use
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name

        logger.info(
            f"Connecting to ChromaDB at {host}:{port}, collection={collection_name}"
        )

        try:
            # Create ChromaDB client
            self.client = chromadb.HttpClient(host=host, port=port)

            # Get or create collection
            self.collection: Collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},  # Use cosine similarity
            )

            logger.info(
                f"Successfully connected to ChromaDB, collection has {self.collection.count()} items"
            )

        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {e}")
            raise

    async def save_embeddings(
        self,
        issue_ids: List[UUID],
        embeddings: np.ndarray,
        documents: List[str],
    ) -> None:
        """
        Save embeddings to ChromaDB.

        Args:
            issue_ids: List of issue UUIDs
            embeddings: numpy array of shape (n_samples, embedding_dim)
            documents: List of preprocessed text content
        """
        if len(issue_ids) != len(embeddings) or len(issue_ids) != len(documents):
            raise ValueError(
                f"Mismatched lengths: {len(issue_ids)} IDs, {len(embeddings)} embeddings, {len(documents)} documents"
            )

        if len(issue_ids) == 0:
            logger.debug("No embeddings to save")
            return

        logger.debug(f"Saving {len(issue_ids)} embeddings to ChromaDB")

        try:
            # Convert UUIDs to strings for ChromaDB IDs
            ids = [str(issue_id) for issue_id in issue_ids]

            # Convert numpy array to list of lists
            embeddings_list = embeddings.tolist()

            # Create metadata for each embedding
            metadatas = [{"text": text} for text in documents]

            # Add to collection (upsert will update existing entries)
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents,  # Store full text for retrieval
            )

            logger.info(
                f"Successfully saved {len(issue_ids)} embeddings to ChromaDB"
            )

        except Exception as e:
            logger.error(f"Failed to save embeddings to ChromaDB: {e}")
            raise

    async def search_similar(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        min_similarity: float = 0.0,
    ) -> List[Tuple[UUID, float, str]]:
        """
        Search for similar issues using vector similarity.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold (0-1)

        Returns:
            List of tuples (issue_id, similarity_score, text)
        """
        logger.debug(f"Searching for top {top_k} similar issues")

        try:
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                include=["documents", "distances", "metadatas"],
            )

            # Parse results
            similar_issues = []

            if results["ids"] and len(results["ids"]) > 0:
                for i in range(len(results["ids"][0])):
                    issue_id_str = results["ids"][0][i]
                    distance = results["distances"][0][i]
                    text = results["documents"][0][i]

                    # Convert cosine distance to similarity (1 - distance)
                    similarity = 1.0 - distance

                    # Filter by minimum similarity
                    if similarity >= min_similarity:
                        issue_id = UUID(issue_id_str)
                        similar_issues.append((issue_id, similarity, text))

            logger.info(f"Found {len(similar_issues)} similar issues")

            return similar_issues

        except Exception as e:
            logger.error(f"Failed to search ChromaDB: {e}")
            raise

    async def get_all_embeddings(self) -> Tuple[List[UUID], np.ndarray]:
        """
        Retrieve all embeddings from ChromaDB.

        Returns:
            Tuple of (issue_ids, embeddings_array)
        """
        logger.debug("Retrieving all embeddings from ChromaDB")

        try:
            # Get all items from collection
            results = self.collection.get(include=["embeddings"])

            if not results["ids"]:
                logger.info("No embeddings found in ChromaDB")
                return [], np.array([])

            # Parse results
            issue_ids = [UUID(id_str) for id_str in results["ids"]]
            embeddings = np.array(results["embeddings"], dtype=np.float32)

            logger.info(
                f"Retrieved {len(issue_ids)} embeddings with shape {embeddings.shape}"
            )

            return issue_ids, embeddings

        except Exception as e:
            logger.error(f"Failed to retrieve embeddings from ChromaDB: {e}")
            raise

    async def delete_embedding(self, issue_id: UUID) -> None:
        """
        Delete embedding by issue ID.

        Args:
            issue_id: Issue UUID
        """
        logger.debug(f"Deleting embedding for issue {issue_id}")

        try:
            self.collection.delete(ids=[str(issue_id)])
            logger.info(f"Deleted embedding for issue {issue_id}")

        except Exception as e:
            logger.error(f"Failed to delete embedding: {e}")
            raise

    async def search_similar_by_id(
        self, issue_id: UUID, top_k: int = 10, min_similarity: float = 0.7
    ) -> List:
        """
        Find similar issues by embedding ID.

        Args:
            issue_id: Reference issue ID
            top_k: Number of similar issues to return
            min_similarity: Minimum similarity threshold (0-1)

        Returns:
            List of similar issue DTOs
        """
        logger.debug(f"Searching similar issues by ID: {issue_id}")

        try:
            # Get the embedding for the reference issue
            result = self.collection.get(
                ids=[str(issue_id)],
                include=["embeddings"]
            )

            if not result["ids"] or not result["embeddings"]:
                raise ValueError(f"No embedding found for issue {issue_id}")

            query_embedding = result["embeddings"][0]

            # Search using the embedding
            return await self.search_similar(
                query_embedding=query_embedding,
                top_k=top_k,
                min_similarity=min_similarity
            )

        except Exception as e:
            logger.error(f"Failed to search similar by ID: {e}")
            raise

    def count_total(self) -> int:
        """
        Count total number of embeddings.

        Returns:
            Total count
        """
        try:
            count = self.collection.count()
            logger.debug(f"Total embeddings in ChromaDB: {count}")
            return count
        except Exception as e:
            logger.error(f"Failed to count embeddings in ChromaDB: {e}", exc_info=True)
            return 0

    def exists(self, issue_id: UUID) -> bool:
        """
        Check if embedding exists for issue.

        Args:
            issue_id: Issue UUID

        Returns:
            True if embedding exists
        """
        result = self.collection.get(ids=[str(issue_id)])
        return len(result["ids"]) > 0

    def clear_all(self) -> int:
        """
        Delete all embeddings from the collection.

        WARNING: This will delete the entire collection and recreate it!
        Used when changing embedding model or dimension.

        Returns:
            Number of deleted embeddings
        """
        logger.warning("⚠️  Clearing ALL embeddings from ChromaDB collection!")
        logger.warning(f"⚠️  Collection: '{self.collection_name}'")

        try:
            # Get count before deletion
            count = self.collection.count()
            logger.warning(f"⚠️  Deleting {count} embeddings...")

            # Delete the collection and recreate it
            self.client.delete_collection(name=self.collection_name)

            # Recreate collection with same metadata
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

            logger.warning(f"✅ Cleared {count} embeddings from ChromaDB")
            return count

        except Exception as e:
            logger.error(f"Failed to clear embeddings from ChromaDB: {e}", exc_info=True)
            raise

    def get_collection_info(self) -> dict:
        """
        Get information about the current collection.

        Returns:
            Dictionary with collection metadata
        """
        try:
            count = self.collection.count()
            logger.debug(f"ChromaDB collection '{self.collection_name}' has {count} items")

            return {
                "name": self.collection_name,
                "count": count,
                "metadata": self.collection.metadata,
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}", exc_info=True)
            return {
                "name": self.collection_name,
                "count": 0,
                "metadata": {},
            }
