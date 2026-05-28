"""Report-layer exceptions."""
from __future__ import annotations


class ReportInputError(Exception):
    """Raised when the report cannot be built because of missing/invalid inputs."""
