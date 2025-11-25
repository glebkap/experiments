"""Custom exceptions for API client."""


class APIError(Exception):
    """Base exception for API-related errors."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        """Initialize API error.

        Args:
            message: Error message
            status_code: HTTP status code if applicable
        """
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class ConnectionError(APIError):
    """Exception raised when connection to API fails."""

    def __init__(self, message: str = "Failed to connect to API") -> None:
        """Initialize connection error.

        Args:
            message: Error message
        """
        super().__init__(message, status_code=None)


class ValidationError(APIError):
    """Exception raised when request validation fails."""

    def __init__(self, message: str, status_code: int = 422) -> None:
        """Initialize validation error.

        Args:
            message: Error message
            status_code: HTTP status code (default: 422)
        """
        super().__init__(message, status_code)


class NotFoundError(APIError):
    """Exception raised when resource is not found."""

    def __init__(self, message: str, status_code: int = 404) -> None:
        """Initialize not found error.

        Args:
            message: Error message
            status_code: HTTP status code (default: 404)
        """
        super().__init__(message, status_code)


class ServerError(APIError):
    """Exception raised when server returns 5xx error."""

    def __init__(self, message: str, status_code: int = 500) -> None:
        """Initialize server error.

        Args:
            message: Error message
            status_code: HTTP status code (default: 500)
        """
        super().__init__(message, status_code)
