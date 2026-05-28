"""RoRo visualization layer: CSV-in / HTML-out interactive report pipeline."""
from roro.report.bundle import DataBundle
from roro.report.errors import ReportInputError
from roro.report.orchestrate import build_report

__all__ = ["DataBundle", "ReportInputError", "build_report"]
