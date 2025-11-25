"""HTTP API client for interacting with backend services."""

import time
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import ValidationError as PydanticValidationError

from support_cli.config import Settings

from .exceptions import (
    APIError,
    ConnectionError,
    NotFoundError,
    ServerError,
    ValidationError,
)
from .models import (
    ClusterDetails,
    ClusteringInfo,
    ExportResponse,
    ImportHistory,
    ImportResponse,
    ImportStatus,
    IssueDetails,
    IssueListItem,
    PipelineResponse,
    PipelineStatus,
    SearchResult,
    StatsResponse,
)


class APIClient:
    """HTTP client for backend API."""

    def __init__(self, settings: Settings) -> None:
        """Initialize API client.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.base_url = str(settings.api_gateway_url).rstrip("/")
        self.timeout = settings.request_timeout
        self.max_retries = settings.max_retries

        self.client = httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout,
            follow_redirects=True,
        )

    def __enter__(self) -> "APIClient":
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.close()

    def close(self) -> None:
        """Close HTTP client."""
        self.client.close()

    def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs: Any,
    ) -> httpx.Response:
        """Make HTTP request with retries and error handling.

        Args:
            method: HTTP method (GET, POST, etc)
            endpoint: API endpoint path
            **kwargs: Additional arguments for httpx.request

        Returns:
            httpx.Response: HTTP response

        Raises:
            ConnectionError: If connection fails
            APIError: For other API errors
        """
        url = f"{self.base_url}{endpoint}"

        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.request(method, url, **kwargs)
                self._handle_error(response)
                return response
            except httpx.ConnectError as e:
                if attempt == self.max_retries:
                    raise ConnectionError(f"Failed to connect to {url}: {e}")
                time.sleep(2**attempt)  # Exponential backoff
            except httpx.TimeoutException as e:
                if attempt == self.max_retries:
                    raise ConnectionError(f"Request timeout for {url}: {e}")
                time.sleep(2**attempt)

        raise ConnectionError(f"Failed to connect to {url} after {self.max_retries} retries")

    def _handle_error(self, response: httpx.Response) -> None:
        """Handle HTTP error responses.

        Args:
            response: HTTP response

        Raises:
            ValidationError: For 4xx errors
            NotFoundError: For 404 errors
            ServerError: For 5xx errors
        """
        if response.is_success:
            return

        try:
            error_data = response.json()
            message = error_data.get("detail", response.text)
        except Exception:
            message = response.text

        if response.status_code == 404:
            raise NotFoundError(message, response.status_code)
        elif 400 <= response.status_code < 500:
            raise ValidationError(message, response.status_code)
        elif 500 <= response.status_code < 600:
            raise ServerError(message, response.status_code)
        else:
            raise APIError(message, response.status_code)

    def health_check(self) -> bool:
        """Check API health.

        Returns:
            bool: True if API is healthy
        """
        try:
            response = self._request("GET", "/api/v1/health")
            return response.status_code == 200
        except Exception:
            return False

    # Import endpoints
    def import_okdesk(self, file_path: Path) -> ImportResponse:
        """Import OKDesk JSONL file.

        Args:
            file_path: Path to JSONL file

        Returns:
            ImportResponse: Import response with ID and status
        """
        with open(file_path, "rb") as f:
            files = {"file": (file_path.name, f, "application/x-ndjson")}
            response = self._request("POST", "/api/v1/import/okdesk", files=files)
            return ImportResponse.model_validate(response.json())

    def import_telegram(self, file_path: Path) -> ImportResponse:
        """Import Telegram JSON file.

        Args:
            file_path: Path to JSON file

        Returns:
            ImportResponse: Import response with ID and status
        """
        with open(file_path, "rb") as f:
            files = {"file": (file_path.name, f, "application/json")}
            response = self._request("POST", "/api/v1/import/telegram", files=files)
            return ImportResponse.model_validate(response.json())

    def get_import_status(self, import_id: str) -> ImportStatus:
        """Get import status by ID.

        Args:
            import_id: Import ID

        Returns:
            ImportStatus: Import status details
        """
        response = self._request("GET", f"/api/v1/import/{import_id}")
        return ImportStatus.model_validate(response.json())

    def list_imports(self, limit: int = 50) -> list[ImportHistory]:
        """Get import history.

        Args:
            limit: Maximum number of records

        Returns:
            list[ImportHistory]: List of imports
        """
        response = self._request("GET", "/api/v1/imports", params={"limit": limit})
        data = response.json()
        return [ImportHistory.model_validate(item) for item in data]

    # Pipeline endpoints
    def process_pipeline(
        self,
        batch_size: int = 100,
        device: str = "cpu",
    ) -> PipelineResponse:
        """Start pipeline processing.

        Args:
            batch_size: Batch size for processing
            device: Device to use (cpu/cuda)

        Returns:
            PipelineResponse: Processing result
        """
        payload = {"batch_size": batch_size, "device": device}
        response = self._request("POST", "/api/v1/analyzer/pipeline/process", json=payload)
        return PipelineResponse.model_validate(response.json())

    def get_pipeline_status(self) -> PipelineStatus:
        """Get pipeline processing status.

        Returns:
            PipelineStatus: Pipeline status
        """
        response = self._request("GET", "/api/v1/analyzer/pipeline/status")
        return PipelineStatus.model_validate(response.json())

    # Clustering endpoints
    def run_clustering(
        self,
        method: Literal["hdbscan", "kmeans"] = "hdbscan",
        min_cluster_size: int = 5,
        use_llm: bool = False,
    ) -> ClusteringInfo:
        """Run clustering algorithm.

        Args:
            method: Clustering method
            min_cluster_size: Minimum cluster size
            use_llm: Use LLM to generate cluster names

        Returns:
            ClusteringInfo: Clustering results
        """
        payload = {
            "method": method,
            "min_cluster_size": min_cluster_size,
            "use_llm": use_llm,
        }
        response = self._request("POST", "/api/v1/analyzer/clustering/run", json=payload)
        return ClusteringInfo.model_validate(response.json())

    def get_clustering_info(self) -> ClusteringInfo:
        """Get clustering information.

        Returns:
            ClusteringInfo: Clustering info
        """
        response = self._request("GET", "/api/v1/analyzer/clustering/info")
        return ClusteringInfo.model_validate(response.json())

    def get_cluster_issues(
        self,
        cluster_id: str,
        limit: int = 10,
    ) -> ClusterDetails:
        """Get issues in a cluster.

        Args:
            cluster_id: Cluster ID
            limit: Maximum number of issues

        Returns:
            ClusterDetails: Cluster details with issues
        """
        response = self._request(
            "GET",
            f"/api/v1/analyzer/clusters/{cluster_id}/issues",
            params={"limit": limit},
        )
        return ClusterDetails.model_validate(response.json())

    # Search endpoints
    def search_similar(
        self,
        query: str,
        top_k: int = 10,
        min_similarity: float = 0.7,
    ) -> list[SearchResult]:
        """Semantic search for similar issues.

        Args:
            query: Search query
            top_k: Number of results
            min_similarity: Minimum similarity threshold

        Returns:
            list[SearchResult]: Search results
        """
        payload = {
            "query": query,
            "top_k": top_k,
            "min_similarity": min_similarity,
        }
        response = self._request("POST", "/api/v1/analyzer/search/similar", json=payload)
        data = response.json()
        results = data.get("results", [])
        return [SearchResult.model_validate(item) for item in results]

    def search_fulltext(
        self,
        query: str,
        limit: int = 50,
        offset: int = 0,
    ) -> list[IssueListItem]:
        """Full-text search.

        Args:
            query: Search query
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            list[IssueListItem]: Search results
        """
        params = {"q": query, "limit": limit, "offset": offset}
        response = self._request("GET", "/api/v1/analyzer/search/fulltext", params=params)
        data = response.json()
        return [IssueListItem.model_validate(item) for item in data]

    def list_issues(
        self,
        status: str | None = None,
        source_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[IssueListItem]:
        """List issues with filters.

        Args:
            status: Filter by status
            source_id: Filter by source ID
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            list[IssueListItem]: List of issues
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        if source_id:
            params["source_id"] = source_id

        response = self._request("GET", "/api/v1/analyzer/issues", params=params)
        data = response.json()
        return [IssueListItem.model_validate(item) for item in data]

    def get_issue(self, issue_id: str) -> IssueDetails:
        """Get issue details.

        Args:
            issue_id: Issue ID

        Returns:
            IssueDetails: Issue details
        """
        response = self._request("GET", f"/api/v1/analyzer/issues/{issue_id}")
        return IssueDetails.model_validate(response.json())

    # Stats endpoints
    def get_stats_processing(self) -> StatsResponse:
        """Get processing statistics.

        Returns:
            StatsResponse: Processing stats
        """
        response = self._request("GET", "/api/v1/analyzer/stats/processing")
        return StatsResponse.model_validate(response.json())

    def get_stats_clusters(self) -> StatsResponse:
        """Get cluster statistics.

        Returns:
            StatsResponse: Cluster stats
        """
        response = self._request("GET", "/api/v1/analyzer/stats/clusters")
        return StatsResponse.model_validate(response.json())

    def get_stats_sources(self) -> StatsResponse:
        """Get source statistics.

        Returns:
            StatsResponse: Source stats
        """
        response = self._request("GET", "/api/v1/analyzer/stats/sources")
        return StatsResponse.model_validate(response.json())

    def get_stats_timeline(
        self,
        from_date: str | None = None,
        to_date: str | None = None,
        granularity: Literal["day", "week", "month"] = "day",
    ) -> StatsResponse:
        """Get timeline statistics.

        Args:
            from_date: Start date (ISO format)
            to_date: End date (ISO format)
            granularity: Time granularity

        Returns:
            StatsResponse: Timeline stats
        """
        params: dict[str, Any] = {"granularity": granularity}
        if from_date:
            params["from_date"] = from_date
        if to_date:
            params["to_date"] = to_date

        response = self._request("GET", "/api/v1/analyzer/stats/timeline", params=params)
        return StatsResponse.model_validate(response.json())

    # Export endpoints
    def export_data(
        self,
        format: Literal["csv", "json"] = "csv",
        filters: dict[str, Any] | None = None,
    ) -> ExportResponse:
        """Export data with filters.

        Args:
            format: Export format
            filters: Export filters

        Returns:
            ExportResponse: Export result
        """
        payload = {"format": format, "filters": filters or {}}
        response = self._request("POST", "/api/v1/analyzer/export", json=payload)
        return ExportResponse.model_validate(response.json())
