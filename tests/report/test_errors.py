"""ReportInputError contract tests."""
from __future__ import annotations

import pytest

from roro.report.errors import ReportInputError


def test_report_input_error_is_exception() -> None:
    assert issubclass(ReportInputError, Exception)


def test_report_input_error_message() -> None:
    with pytest.raises(ReportInputError, match="boom"):
        raise ReportInputError("boom")
