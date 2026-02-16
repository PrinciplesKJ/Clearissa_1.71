"""
Clearissa - format_detector.py
-------------------------------
Unified format detection for time-series fluorescence data.

This module provides detection and parsing for:
1. Standard format (metadata lines followed by a "Well" column)
2. Minimal format Type A ("Well", "Time [min]", data columns)
3. Minimal format Type B ("Time [min]", data columns; no "Well")

It ensures a consistent :class:`pandas.DataFrame` structure across
all processors that consume plate reader data.

Author: Križan Jurinović
Date: October 2025
"""

import re
import logging
import pandas as pd
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class DataFormat:
    """Enumeration of supported data formats."""

    STANDARD = "standard"              # Full format with metadata section
    MINIMAL_WELL = "minimal_well"      # Minimal format with "Well" column
    MINIMAL_TIME = "minimal_time"      # Minimal format without "Well" column
    UNKNOWN = "unknown"


def detect_data_format(filepath: str) -> Tuple[str, int]:
    """Detect the on-disk format of a CSV file.

    The function inspects the first non-empty lines of ``filepath`` to
    determine whether the file uses the standard format with metadata
    rows, or one of the minimal formats.

    Parameters
    ----------
    filepath : str
        Path to the CSV file to analyse.

    Returns
    -------
    tuple of (str, int)
        format_type : str
            One of ``"standard"``, ``"minimal_well"``, ``"minimal_time"``,
            or ``"unknown"``.
        skip_rows : int
            Number of leading metadata rows to skip before the data header
            when reading the file with :func:`pandas.read_csv`.

    Notes
    -----
    Detection strategy
    ~~~~~~~~~~~~~~~~~~
    1. Read up to the first 20 non-empty lines of the file.
    2. Find the first line whose first token starts with ``"well"`` or
       ``"time"`` (case-insensitive).
    3. If this header line is on the first row, treat the file as a
       minimal format:

       * If the header starts with ``"Well"``, classify as
         :data:`DataFormat.MINIMAL_WELL`.
       * Otherwise, classify as :data:`DataFormat.MINIMAL_TIME`.

    4. If the header line appears after at least one earlier line,
       classify as :data:`DataFormat.STANDARD` and treat the preceding
       lines as metadata. ``skip_rows`` is then set to the index of the
       header line.

    If no header starting with "Well" or "Time" is found in the first
    20 non-empty lines, the format is reported as
    :data:`DataFormat.UNKNOWN` with ``skip_rows = 0``.
    """
    logger.debug("***********************************************")
    logger.debug("[FORMAT DETECT] Analysing file: %s", filepath)
    logger.debug("***********************************************")

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            # Read up to 20 non-empty lines for header detection.
            lines = [line.strip() for line in f.readlines()[:20] if line.strip()]
    except Exception as e:  # pragma: no cover - filesystem error path
        logger.error("[FORMAT DETECT] Failed to read file: %s", e)
        return DataFormat.UNKNOWN, 0

    if not lines:
        logger.error("[FORMAT DETECT] File is empty")
        return DataFormat.UNKNOWN, 0

    logger.debug("[FORMAT DETECT] First 5 lines:")
    for i, line in enumerate(lines[:5]):
        logger.debug("  Line %d: %s", i, line[:80])

    # Search for data header row
    data_header_idx: Optional[int] = None
    header_type: Optional[str] = None

    for idx, line in enumerate(lines):
        line_lower = line.lower()

        # Check if line starts with "Well"
        if line_lower.startswith("well"):
            data_header_idx = idx
            header_type = "well"
            logger.debug("[FORMAT DETECT] Found 'Well' at line %d: %s", idx, line[:80])
            break

        # Check if line starts with "Time"
        if line_lower.startswith("time"):
            data_header_idx = idx
            header_type = "time"
            logger.debug("[FORMAT DETECT] Found 'Time' at line %d: %s", idx, line[:80])
            break

    if data_header_idx is None:
        logger.warning("[FORMAT DETECT] No 'Well' or 'Time' header found in first 20 lines")
        return DataFormat.UNKNOWN, 0

    # Determine format based on header position and type
    if data_header_idx == 0:
        # First line is data header - minimal format
        if header_type == "well":
            format_type = DataFormat.MINIMAL_WELL
            logger.info("[FORMAT DETECT] - MINIMAL FORMAT (Type A: with Well column)")
        else:
            format_type = DataFormat.MINIMAL_TIME
            logger.info("[FORMAT DETECT] - MINIMAL FORMAT (Type B: without Well column)")
        skip_rows = 0
    else:
        # Metadata lines exist before data header - standard format
        format_type = DataFormat.STANDARD
        skip_rows = data_header_idx
        logger.info("[FORMAT DETECT] - STANDARD FORMAT (with %d metadata lines)", skip_rows)

    logger.debug("***********************************************")
    logger.debug("[FORMAT DETECT] Result: %s, skip_rows=%d", format_type, skip_rows)
    logger.debug("***********************************************")

    return format_type, skip_rows


def load_and_standardize_dataframe(filepath: str) -> Tuple[Optional[pd.DataFrame], str]:
    """Load a CSV file and standardise it to a common structure.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    tuple of (pandas.DataFrame or None, str)
        df : pandas.DataFrame or None
            Standardised DataFrame with columns ``["Well", "Time [min]", ...]``.
            Returns ``None`` if loading or standardisation fails.
        format_type : str
            Detected format type, as returned by :func:`detect_data_format`.

    Notes
    -----
    Output structure
    ~~~~~~~~~~~~~~~~
    All returned DataFrames have the following structure:

    * ``"Well"`` as the first column
    * ``"Time [min]"`` as the second column
    * One or more data columns in the remaining positions

    For :data:`DataFormat.MINIMAL_TIME` (minimal format without a "Well"
    column), the function inserts a synthetic ``"Well"`` column filled with
    the string ``"Global"``.

    Time column detection and renaming
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    After loading, the function searches for a time column whose header
    begins with ``"time"`` (case-insensitive). If such a column is found
    and its name is not already ``"Time [min]"``, it is renamed to this
    standard form. If no time-like column can be located, the function
    returns ``None``.
    """
    logger.info("=" * 60)
    logger.info("[LOAD & STANDARDIZE] Starting for: %s", filepath)
    logger.info("=" * 60)

    # Detect format
    format_type, skip_rows = detect_data_format(filepath)

    if format_type == DataFormat.UNKNOWN:
        logger.error("[LOAD & STANDARDIZE] Unrecognised data format")
        return None, format_type

    try:
        # Load CSV with appropriate settings
        logger.debug("[LOAD & STANDARDIZE] Reading CSV (skip_rows=%d)...", skip_rows)
        df = pd.read_csv(filepath, skiprows=skip_rows)

        # Normalise column names for matching
        df.columns = df.columns.str.strip()
        logger.debug("[LOAD & STANDARDIZE] Loaded shape: %s", df.shape)
        logger.debug("[LOAD & STANDARDIZE] Raw columns: %s", list(df.columns)[:5])

    except Exception as e:  # pragma: no cover - filesystem error path
        logger.error("[LOAD & STANDARDIZE] Failed to read CSV: %s", e)
        return None, format_type

    # Standardise based on format type
    if format_type == DataFormat.MINIMAL_TIME:
        # Insert synthetic "Well" column for minimal-time format
        logger.debug("[LOAD & STANDARDIZE] Inserting synthetic 'Well' column")
        df.insert(0, "Well", "Global")
        logger.info("[LOAD & STANDARDIZE] Added 'Well' column with value 'Global'")

    # Verify required columns exist
    if "Well" not in df.columns:
        logger.error("[LOAD & STANDARDIZE] Missing 'Well' column after standardisation")
        return None, format_type

    # Look for time column (flexible matching)
    time_col: Optional[str] = None
    time_pattern = re.compile(r"^time.*", re.IGNORECASE)
    for col in df.columns:
        if time_pattern.match(str(col)):
            time_col = col
            break

    if not time_col:
        logger.error("[LOAD & STANDARDIZE] No time column found")
        return None, format_type

    # Rename time column to standard name if needed
    if time_col != "Time [min]":
        logger.debug("[LOAD & STANDARDIZE] Renaming '%s' -> 'Time [min]'", time_col)
        df.rename(columns={time_col: "Time [min]"}, inplace=True)

    # Ensure proper column order: Well, Time [min], data columns
    cols = ["Well", "Time [min]"] + [c for c in df.columns if c not in ["Well", "Time [min]"]]
    df = df[cols]

    logger.info("[LOAD & STANDARDIZE] Success")
    logger.info("  - Format: %s", format_type)
    logger.info("  - Final shape: %s", df.shape)
    logger.info("  - Final columns: %s", list(df.columns)[:5])
    logger.info("=" * 60)

    return df, format_type

