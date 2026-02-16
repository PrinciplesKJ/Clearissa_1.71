"""
Clearissa - data_processing_utils.py
-------------------------------------
Core utilities for loading, cleaning, and transforming experimental data.

This module provides:
- CSV/Excel file import with automatic format detection
- Header assignment and time column normalisation
- Injection marker removal and overflow detection
- Data merging across multiple experimental files
- DataFrame persistence for session recovery
- Plot object creation for data visualisation

All functions support flexible input formats from plate reader instruments
and handle both simple and complex time notation.

Author: Krizan Jurinovic
"""

import os
import sys
import json
import pickle
import re
import logging

from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

# Add resource_utils import for proper path handling in frozen executables
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from resource_utils import get_data_path

from pyqtgraph.Qt import QtCore
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt5.QtWidgets import QMessageBox

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# UI message helpers
# ---------------------------------------------------------------------------

def show_warning_message(message: str):
    """Display a warning message dialogue via PyQt5."""
    msg_box = QMessageBox()
    msg_box.setIcon(QMessageBox.Warning)
    msg_box.setText("Warning")
    msg_box.setInformativeText(message)
    msg_box.setWindowTitle("Warning")
    msg_box.exec_()


def show_error_message(message: str):
    """Display an error message dialogue via PyQt5."""
    msg_box = QMessageBox()
    msg_box.setIcon(QMessageBox.Critical)
    msg_box.setText("An error has occurred.")
    msg_box.setInformativeText(message)
    msg_box.setWindowTitle("Error")
    msg_box.exec_()


# ---------------------------------------------------------------------------
# Time conversion helpers (module-level for reuse)
# ---------------------------------------------------------------------------

def _convert_value_to_minutes(value, unit: str) -> Optional[float]:
    """Convert a numeric value from the given unit to minutes."""
    try:
        value = float(value)
    except (ValueError, TypeError):
        logger.error("Cannot parse numeric time value: %s", value)
        return None

    if unit == 's':
        return round(value / 60, 2)
    elif unit == 'h':
        return round(value * 60, 2)
    return round(value, 2)


def _convert_complex_time_to_minutes(time_str: str) -> Optional[float]:
    """
    Convert complex time strings to minutes.

    Examples: '1 h 30 min' -> 90.0, '45 s' -> 0.75
    """
    parts = time_str.strip().split()
    total_minutes = 0.0

    for i in range(0, len(parts), 2):
        try:
            val = float(parts[i])
            unit = parts[i + 1].lower() if i + 1 < len(parts) else 'min'
            partial = _convert_value_to_minutes(val, unit)
            if partial is None:
                return None
            total_minutes += partial
        except (ValueError, IndexError):
            logger.error("Error parsing complex time string: '%s'", time_str)
            return None

    return round(total_minutes, 2)


def _detect_time_column(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """
    Identify time column by header or monotonic numeric pattern.

    Returns
    -------
    tuple of (column_name, unit) or (None, None) if detection fails.
    Unit is one of: 's', 'min', 'h', 'complex', or None.
    """
    df.columns = [str(c).strip() for c in df.columns]

    # Header-based search
    header_candidates = [
        col for col in df.columns
        if str(col).strip().lower().startswith(("time", "elapsed time", "t"))
    ]

    if len(header_candidates) == 1:
        time_column = header_candidates[0]
    elif len(header_candidates) > 1:
        logger.error("Multiple candidate time columns found: %s", header_candidates)
        return None, None
    else:
        # Fallback: look for monotonic numeric column
        numeric_candidates = []
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                series = df[col].dropna()
                if series.nunique() >= 3 and series.is_monotonic_increasing:
                    numeric_candidates.append(col)

        if len(numeric_candidates) == 1:
            time_column = numeric_candidates[0]
            logger.warning("No time header found; using monotonic column '%s'", time_column)
        else:
            logger.error(
                "Cannot identify unique time column (header=%s, numeric=%s)",
                header_candidates, numeric_candidates
            )
            return None, None

    # Extract unit from column name brackets
    unit_regex = re.compile(
        r"\[\s*(?P<unit>"
        r"s(ec(ond)?s?|s)?|"
        r"m(in(s|ute)?s?)?|"
        r"h(r|our)?s?"
        r")\s*]",
        re.IGNORECASE
    )

    match = unit_regex.search(str(time_column))
    if match:
        raw_unit = match.group("unit").lower()
        unit = ("s" if raw_unit.startswith("s") else
                "h" if raw_unit.startswith("h") else
                "min")
        return time_column, unit

    # Check cell values for unit tokens
    if df[time_column].astype(str).str.contains(r"(?:h|min|s)", case=False).any():
        return time_column, "complex"

    # Default assumption: minutes
    return time_column, "min"


# ---------------------------------------------------------------------------
# Header and format detection
# ---------------------------------------------------------------------------

def assign_headers(df: pd.DataFrame, minimal_structure: bool = False) -> pd.DataFrame:
    """
    Assign column headers from the first two rows of a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with headers in the first two rows.
    minimal_structure : bool, optional
        If True, preserve existing column names without processing.

    Returns
    -------
    pandas.DataFrame
        DataFrame with assigned headers and header rows removed.
    """
    if minimal_structure:
        return df

    if df.shape[0] < 2:
        msg = "DataFrame too short (rows=%d); need at least 2 for header extraction" % df.shape[0]
        logger.error(msg)
        raise ValueError(msg)

    time_pattern = re.compile(r"^time(?:\s*\[.*])?$", re.IGNORECASE)

    raw_headers = df.iloc[0].astype(str).str.strip().tolist()
    second_row = df.iloc[1].astype(str).str.strip().tolist()

    # Override any Time column detected in row 1
    for idx, val in enumerate(second_row):
        if time_pattern.match(val):
            raw_headers[idx] = val

    df2 = df.copy()
    df2.columns = raw_headers
    df2 = df2.iloc[2:].reset_index(drop=True)
    return df2


def check_csv_format(file_path: str) -> pd.DataFrame:
    """
    Determine CSV separator by detecting consecutive semicolons.

    Returns DataFrame with no header row assigned.
    """
    try:
        with open(file_path, 'r') as file:
            content = file.read()

        if ';;;' in content:
            logger.info("Semicolon separator detected for file: %s", file_path)
            return pd.read_csv(file_path, sep=";", header=None)
        else:
            return pd.read_csv(file_path, sep=",", header=None)
    except Exception as e:
        logger.error("Error reading file '%s': %s", file_path, e)
        raise


def extract_info_tags(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Extract experiment (Test Name) and info (ID1 row) tags from DataFrame.

    Returns
    -------
    tuple of (experiment, info)

    Raises
    ------
    ValueError
        If DataFrame is empty or required patterns are not found.
    """
    if df.empty:
        raise ValueError("DataFrame is empty.")

    # Locate 'Test Name:' pattern
    search_pattern = r'(?i)^test\s*name:\s*(.*)$'
    experiment_matches = df.iloc[:, 0].astype(str).str.extract(search_pattern, expand=True)
    valid_rows = experiment_matches.dropna()

    if not valid_rows.empty:
        experiment = valid_rows.iloc[0, 0].strip()
    else:
        logger.warning("No row matching 'Test Name:' found; using first non-empty value as fallback.")
        fallback_series = df.iloc[:, 0].dropna().astype(str)
        if not fallback_series.empty:
            experiment = fallback_series.iloc[0].strip()
        else:
            raise ValueError("No suitable experiment tag found.")

    # Locate row starting with 'ID1'
    matching_rows = df[df.iloc[:, 0].astype(str).str.startswith('ID1', na=False)]
    if matching_rows.empty:
        raise ValueError("No row found starting with 'ID1' in the first column.")

    info_row = matching_rows.iloc[0].dropna()
    info = ' '.join(map(str, info_row.values))

    return experiment, info


def check_datetime(df: pd.DataFrame) -> Optional[datetime]:
    """
    Locate date/time lines in the first 10 rows and parse them.

    Returns a datetime object or None if parsing fails.
    """
    potential_formats = [
        "Date: %d/%m/%Y Time: %H:%M:%S",
        "Date: %Y-%m-%d Time: %H:%M:%S",
        "Date: %Y/%d/%m Time: %H:%M:%S",
        "Date: %d.%m.%Y Time: %H:%M:%S",
    ]
    date_value, time_value = None, None

    for row_index in range(min(10, len(df))):
        row = df.iloc[row_index]
        for cell in row:
            if isinstance(cell, str):
                if cell.startswith("Date:"):
                    date_value = cell
                elif cell.startswith("Time:"):
                    time_value = cell
            if date_value and time_value:
                break
        if date_value and time_value:
            break

    if not date_value or not time_value:
        logger.error("Could not locate both Date and Time lines in header rows.")
        return None

    combined = "%s %s" % (date_value, time_value)
    for fmt in potential_formats:
        try:
            return datetime.strptime(combined, fmt)
        except ValueError:
            continue

    logger.error("Failed to parse datetime from '%s' with known formats.", combined)
    return None


def has_minimal_format(df: pd.DataFrame) -> bool:
    """
    Check if the DataFrame is in minimal format.

    Minimal format has:
    - First column starting with 'time' or 'well'
    - If 'well', second column starts with 'time'
    - Remaining columns match well pattern (A01-G12) or have data suffixes
    """
    well_or_data_pattern = re.compile(r'^([a-g][0-9]{1,2}|c[0-9]{2}).*', re.IGNORECASE)

    def _check_structure(name_list: list) -> bool:
        if len(name_list) < 2:
            return False

        norm_names = [str(n).strip().lower() for n in name_list]

        if norm_names[0] == "well":
            if len(norm_names) < 3 or not norm_names[1].startswith("time"):
                return False
            return all(well_or_data_pattern.match(col) for col in norm_names[2:])

        if not norm_names[0].startswith("time"):
            return False

        return all(well_or_data_pattern.match(col) for col in norm_names[1:])

    # Check DataFrame columns first, then first row
    if _check_structure(list(df.columns)):
        return True

    if not df.empty and _check_structure([str(x) for x in df.iloc[0]]):
        return True

    return False


# ---------------------------------------------------------------------------
# Data trimming and cleaning
# ---------------------------------------------------------------------------

def trim_datatable(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Truncate DataFrame to rows starting from where any column contains 'Well'.
    Remove empty rows, drop 'Group' columns, then transpose.

    Returns truncated and transposed DataFrame, or None if 'Well' not found.
    """
    try:
        mask_well = df.apply(
            lambda col: col.map(lambda x: isinstance(x, str) and x.strip().lower() == 'well')
        )
        well_indices = df.index[mask_well.any(axis=1)].tolist()

        if not well_indices:
            logger.error("No row with 'Well' found in the DataFrame.")
            return None

        if len(well_indices) > 1:
            logger.warning("Multiple 'Well' rows found; using the first.")

        truncated_df = df.iloc[well_indices[0]:].copy()
        truncated_df.dropna(axis=0, how='all', inplace=True)

        # Drop columns containing 'group' in the header row
        header_row = truncated_df.iloc[0].astype(str).str.strip().str.lower()
        group_cols = header_row[header_row.str.contains("group", case=False, na=False)].index
        if len(group_cols) > 0:
            truncated_df.drop(columns=group_cols, inplace=True)

        return truncated_df.T

    except Exception as e:
        logger.error("Error trimming data table: %s", e)
        return None


def remove_injection_marker(data_frame: pd.DataFrame) -> Tuple[pd.DataFrame, List]:
    """
    Remove rows containing injection markers and extract injection timepoints.

    Returns
    -------
    tuple of (cleaned DataFrame, list of injection timepoints in minutes)
    """
    if data_frame.empty:
        return data_frame, []

    if 'Time [min]' not in data_frame.columns:
        raise ValueError("Missing 'Time [min]' column")

    inj_rows = data_frame.apply(
        lambda row: row.astype(str).str.contains('Inj.', case=False).any(), axis=1
    )
    if not inj_rows.any():
        return data_frame, []

    injection_timepoints = data_frame.loc[inj_rows, 'Time [min]'].tolist()
    data_frame = data_frame[~inj_rows].copy()
    data_frame = data_frame.replace('Inj.', '', regex=True).infer_objects()
    data_frame['Time [min]'] = pd.to_numeric(data_frame['Time [min]'], errors='coerce')

    logger.info("Removed injection marker rows; timepoints: %s", injection_timepoints)
    return data_frame, injection_timepoints


def process_time_column(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Detect and convert time column to standardised minutes format.

    Detection strategy:
    1. Search for column headers containing 'time', 'elapsed time', or 't'
    2. Fallback to monotonic numeric columns if no header match
    3. Extract units from brackets: [s], [min], [h]
    4. Handle complex formats like '1 h 30 min'
    5. Default to minutes if no unit specified

    Returns DataFrame with 'Time [min]' column, or None on failure.
    """
    time_col, time_unit = _detect_time_column(df)
    if not time_col:
        logger.error("Time column detection failed")
        return None

    logger.info("Converting column '%s' (unit: %s) to minutes", time_col, time_unit)

    if time_unit == 'complex':
        df[time_col] = df[time_col].astype(str).apply(_convert_complex_time_to_minutes)
    else:
        df[time_col] = df[time_col].apply(lambda x: _convert_value_to_minutes(x, time_unit))

    if str(time_col) != 'Time [min]':
        df.rename(columns={time_col: 'Time [min]'}, inplace=True)

    df['Time [min]'] = df['Time [min]'].round(2)
    return df


def check_overflow_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect and replace instrument overflow values (260000) with NaN.
    """
    try:
        affected_columns = []

        for col in df.columns[2:]:
            mask = (df[col] == 260000)
            if isinstance(mask, pd.Series) and mask.any():
                affected_columns.append(col)
                df.loc[mask, col] = np.nan

        if affected_columns:
            logger.warning(
                "Overflow values (260000) detected in %d column(s): %s",
                len(affected_columns), ', '.join(affected_columns[:10])
            )
            show_warning_message(
                "Overflow values (260000) detected and replaced with NaN.\n\n"
                "Affected columns (%d): %s%s" % (
                    len(affected_columns),
                    ', '.join(affected_columns[:10]),
                    '...' if len(affected_columns) > 10 else ''
                )
            )

        return df
    except Exception as e:
        logger.error("Overflow check failed: %s", e)
        raise


def remove_empty_well_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns (from the third onward) that contain only NaN values.
    """
    columns_to_drop = []

    for col in df.columns[2:]:
        if df[col].notna().sum() == 0:
            logger.warning("No valid datapoints in well '%s'. Dropping column.", col)
            QMessageBox().warning(
                None, "Warning",
                "No valid datapoints found in well '%s'. Dropping this well." % col
            )
            columns_to_drop.append(col)

    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
    return df


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------

def _process_minimal_file(df: pd.DataFrame, file: str) -> Optional[Dict[str, Any]]:
    """
    Process a single file detected as minimal format.

    Returns a dict with keys: dataframe, experiment, info, timestamp, channels, minimal
    or None on failure.
    """
    # Determine if headers are in columns or first row
    norm_cols = [str(c).strip().lower() for c in df.columns]
    headers_in_first_row = not (norm_cols[0] == "well" or norm_cols[0].startswith("time"))

    if headers_in_first_row:
        # Check first row matches minimal pattern
        norm_first_row = [str(x).strip().lower() for x in df.iloc[0]]
        if not (norm_first_row[0] == "well" or norm_first_row[0].startswith("time")):
            return None
        processed_df = df.copy()
        processed_df.columns = processed_df.iloc[0]
        processed_df = processed_df.iloc[1:].reset_index(drop=True)
    else:
        processed_df = df.copy()

    # Determine if a Well column is present
    first_col = str(processed_df.columns[0]).strip().lower()
    has_well_column = (first_col == 'well')

    # Process time column
    processed_df = process_time_column(processed_df)
    if processed_df is None:
        logger.error("Time column processing failed for minimal format file '%s'", file)
        return None

    # Determine channels
    if has_well_column:
        channels = processed_df['Well'].unique().tolist()
    else:
        if 'Well' not in processed_df.columns:
            processed_df.insert(0, 'Well', 'Default')
        channels = ['Default']

    # Convert data columns to numeric
    for col in processed_df.columns:
        if col not in ['Time [min]', 'Well']:
            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')

    # Remove injection markers if present
    try:
        processed_df, _ = remove_injection_marker(processed_df)
    except Exception as e:
        logger.warning("Injection marker removal failed for '%s': %s", file, e)

    return {
        'dataframe': processed_df,
        'experiment': None,
        'info': None,
        'timestamp': None,
        'channels': channels,
        'minimal': True
    }


def _process_standard_file(df: pd.DataFrame, file: str) -> Optional[Dict[str, Any]]:
    """
    Process a single file detected as standard (metadata-rich) format.

    Returns a dict with keys: dataframe, experiment, info, timestamp, channels, minimal
    or None on failure.
    """
    # Extract metadata
    try:
        experiment, info = extract_info_tags(df)
    except Exception as e:
        logger.warning("Could not extract info tags from '%s': %s", file, e)
        experiment, info = None, None

    try:
        timestamp = check_datetime(df)
    except Exception as e:
        logger.warning("Timestamp extraction failed for '%s': %s", file, e)
        timestamp = None

    # Trim and assign headers
    try:
        processed_df = trim_datatable(df)
        if processed_df is None:
            raise ValueError("trim_datatable returned None")
        processed_df = assign_headers(processed_df)
    except Exception as e:
        logger.error("Error processing data structure in '%s': %s", file, e)
        return None

    # Process time column
    processed_df = process_time_column(processed_df)
    if processed_df is None:
        logger.error("Time column processing failed for standard format file '%s'", file)
        return None

    # Remove injection markers
    try:
        processed_df, _ = remove_injection_marker(processed_df)
    except Exception as e:
        logger.warning("Injection marker removal failed for '%s': %s", file, e)

    # Clean empty wells and get channels
    processed_df = remove_empty_well_columns(processed_df)
    channels = list(processed_df.iloc[:, 0].unique()) if not processed_df.empty else []

    return {
        'dataframe': processed_df,
        'experiment': experiment,
        'info': info,
        'timestamp': timestamp,
        'channels': channels,
        'minimal': False
    }


def load_data(path: str) -> Dict[str, Any]:
    """
    Load CSV/Excel file(s) from a single file path or folder path.

    Parameters
    ----------
    path : str
        Path to a single file or a folder containing files.

    Returns
    -------
    dict
        Keyed by filename, each value containing:
        dataframe, experiment, info, timestamp, channels, minimal
    """
    dataframes_dict = {}

    if path is None:
        logger.error("No path provided.")
        return dataframes_dict

    # Resolve file list
    if os.path.isfile(path):
        files_to_process = [(os.path.dirname(path), os.path.basename(path))]
    elif os.path.isdir(path):
        try:
            supported_extensions = (".csv", ".xls", ".xlsx")
            filenames = [f for f in os.listdir(path) if f.lower().endswith(supported_extensions)]
            files_to_process = [(path, f) for f in filenames]
        except Exception as e:
            logger.error("Failed listing files in '%s': %s", path, e)
            return dataframes_dict

        if not files_to_process:
            logger.warning("No supported files found in folder '%s'.", path)
            return dataframes_dict
    else:
        logger.error("Path '%s' is neither a file nor a directory.", path)
        return dataframes_dict

    # Process each file
    for file_idx, (folder_path, file) in enumerate(files_to_process, 1):
        file_path = os.path.join(folder_path, file)
        logger.info("Processing file %d/%d: %s", file_idx, len(files_to_process), file)

        try:
            # Load raw data
            if file.lower().endswith(".csv"):
                df = check_csv_format(file_path)
            else:
                df = pd.read_excel(file_path, header=0)
        except Exception as e:
            logger.error("Failed to load '%s': %s", file, e)
            continue

        try:
            if df.empty:
                logger.warning("File '%s' loaded but is empty.", file)
                continue

            # Detect format and process accordingly
            if has_minimal_format(df):
                result = _process_minimal_file(df, file)
            else:
                result = _process_standard_file(df, file)

            if result is None:
                logger.error("Processing failed for file '%s'", file)
                continue

            dataframes_dict[file] = result
            logger.info(
                "File '%s' processed: format=%s, shape=%s, channels=%d",
                file,
                'minimal' if result['minimal'] else 'standard',
                result['dataframe'].shape,
                len(result['channels'])
            )

        except Exception as e:
            logger.error("Error processing file '%s': %s", file, e)
            continue

    logger.info("Loading complete: %d/%d files processed successfully",
                len(dataframes_dict), len(files_to_process))
    return dataframes_dict


# ---------------------------------------------------------------------------
# Data merging
# ---------------------------------------------------------------------------

def _merge_minimal_dataframes(df_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Merge minimal format dataframes.

    Files without a Well column are merged on Time [min] with filename suffixes.
    Files with a Well column are concatenated directly.
    """
    merged_dataframe = None

    for filename, data_info in df_dict.items():
        df = data_info['dataframe'].copy()

        if 'Time [min]' not in df.columns:
            logger.error("File '%s' missing 'Time [min]' column, skipping.", filename)
            continue

        has_well_column = 'Well' in df.columns

        # Convert data columns to numeric
        for col in df.columns:
            if col not in ['Time [min]', 'Well']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if has_well_column:
            # Native structure with Well column - concatenate
            if merged_dataframe is None:
                merged_dataframe = df
            else:
                merged_dataframe = pd.concat([merged_dataframe, df], axis=0, ignore_index=True)
        else:
            # Pure minimal format - merge on time with filename suffixes
            data_cols = [c for c in df.columns if c != 'Time [min]']
            rename_dict = {col: "%s_%s" % (col, filename) for col in data_cols}
            df = df.rename(columns=rename_dict)

            if merged_dataframe is None:
                merged_dataframe = df
            else:
                merged_dataframe = pd.merge(merged_dataframe, df, on='Time [min]', how='outer')

    if merged_dataframe is None:
        raise ValueError("No valid dataframes to merge")

    # Sort and finalise
    if 'Well' in merged_dataframe.columns:
        merged_dataframe.sort_values(by=['Well', 'Time [min]'], inplace=True)
    else:
        merged_dataframe.sort_values(by='Time [min]', inplace=True)

    merged_dataframe.reset_index(drop=True, inplace=True)
    merged_dataframe['Time [min]'] = merged_dataframe['Time [min]'].round(2)

    return merged_dataframe


def _merge_standard_dataframes(df_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Merge standard format dataframes using timestamp-based time offsets.

    Files are sorted by timestamp and subsequent files have their time values
    offset relative to the first file.
    """
    sorted_items = sorted(df_dict.items(), key=lambda x: x[1]['timestamp'])
    timestamps = [item[1]['timestamp'] for item in sorted_items]
    reference_timestamp = timestamps[0]

    sorted_dataframes = []
    for i, (name, info) in enumerate(sorted_items):
        df = info['dataframe'].copy()

        # Convert to numeric except Well
        for col in df.columns:
            if col != 'Well':
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Track file source for stable sorting
        df['_file_index'] = i
        df['_original_row'] = range(len(df))

        sorted_dataframes.append(df)

    # Apply time offsets to subsequent files
    for i in range(1, len(sorted_dataframes)):
        time_diff = (timestamps[i] - reference_timestamp).total_seconds() / 60
        sorted_dataframes[i]['Time [min]'] += time_diff
        logger.info("File %d: applied time offset of +%.2f minutes", i + 1, time_diff)

    # Concatenate and sort
    merged_dataframe = pd.concat(sorted_dataframes, axis=0, ignore_index=True)

    merged_dataframe.sort_values(
        by=['Well', 'Time [min]', '_file_index', '_original_row'],
        inplace=True
    )
    merged_dataframe.reset_index(drop=True, inplace=True)
    merged_dataframe.drop(columns=['_file_index', '_original_row'], inplace=True)

    # Final numeric conversion and overflow check
    for col in merged_dataframe.columns:
        if col != 'Well':
            merged_dataframe[col] = pd.to_numeric(merged_dataframe[col], errors='coerce')

    merged_dataframe = check_overflow_values(merged_dataframe)
    merged_dataframe['Time [min]'] = merged_dataframe['Time [min]'].round(2)

    # Check for time discontinuities
    wells_with_issues = []
    for well in merged_dataframe['Well'].unique():
        well_data = merged_dataframe[merged_dataframe['Well'] == well]
        time_diffs = well_data['Time [min]'].diff()
        unusual_jumps = time_diffs[(time_diffs < 0) | (time_diffs > 100)]
        if len(unusual_jumps) > 0:
            wells_with_issues.append(well)

    if wells_with_issues:
        logger.warning(
            "Time discontinuities detected in %d well(s): %s",
            len(wells_with_issues), wells_with_issues
        )

    return merged_dataframe


def merge_data(df_dict: Dict[str, Dict[str, Any]], has_minimal_format: bool = None) -> pd.DataFrame:
    """
    Merge dataframes from multiple files into a single DataFrame.

    Parameters
    ----------
    df_dict : dict
        Dictionary keyed by filename, each value containing dataframe and metadata.
    has_minimal_format : bool, optional
        If None, auto-detected from the first file entry.

    Returns
    -------
    pandas.DataFrame
        Merged DataFrame with consistent Time [min] and Well columns.

    Raises
    ------
    Exception
        If merge fails for any reason.
    """
    try:
        logger.info("Starting data merge for %d file(s)", len(df_dict))

        # Auto-detect format if not specified
        if has_minimal_format is None:
            first_entry = next(iter(df_dict.values()), None)
            has_minimal_format = first_entry.get('minimal', False) if first_entry else False

        logger.info("Merge format: %s", 'minimal' if has_minimal_format else 'standard')

        if has_minimal_format:
            merged = _merge_minimal_dataframes(df_dict)
        else:
            merged = _merge_standard_dataframes(df_dict)

        logger.info("Merge complete: final shape %s", merged.shape)
        return merged

    except Exception as e:
        error_msg = "Data merge failed: %s" % e
        logger.error(error_msg, exc_info=True)
        show_error_message(error_msg)
        raise


# ---------------------------------------------------------------------------
# Data transformation utilities
# ---------------------------------------------------------------------------

def calculate_mean_over_time_range(df: pd.DataFrame, start_time: float, end_time: float) -> pd.DataFrame:
    """
    Calculate the mean of each well channel over [start_time, end_time],
    ignoring NaN values, and replace those rows with the calculated mean.
    """
    selected_data = df[(df['Time [min]'] >= start_time) & (df['Time [min]'] <= end_time)]
    if selected_data.empty:
        raise ValueError("No data in the specified time range.")

    selected_data.iloc[:, 2:] = selected_data.iloc[:, 2:].apply(pd.to_numeric, errors='coerce')

    mean_values = selected_data.iloc[:, 2:].mean(skipna=True)
    if mean_values.isnull().any():
        raise ValueError("NaN values found in the calculated means even after ignoring NaNs.")

    df_update = df.copy()
    mask = (df['Time [min]'] >= start_time) & (df['Time [min]'] <= end_time)
    for col in mean_values.index:
        df_update.loc[mask, col] = mean_values[col]

    return df_update


def find_closest_timepoint(df: pd.DataFrame, target_time: float) -> float:
    """Find the time value in 'Time [min]' closest to target_time."""
    time_diff = df['Time [min]'] - target_time
    return df.loc[time_diff.abs().idxmin(), 'Time [min]']


def adjust_data_offset(df: pd.DataFrame, timepoint: float) -> pd.DataFrame:
    """
    Adjust offset of each well so that values at the specified timepoint become zero.
    """
    timepoint_index = df[df['Time [min]'] > timepoint].index.min()
    if pd.isna(timepoint_index):
        logger.warning("Specified timepoint is beyond the data range. No offset adjustment.")
        return df

    for col in df.columns[2:]:
        offset = df.at[timepoint_index, col]
        if offset < 0:
            df.loc[timepoint_index:, col] += abs(offset)
        else:
            df.loc[timepoint_index:, col] -= offset
    return df


def filter_rows_by_string(df: pd.DataFrame, column_name: str, search_string: str) -> pd.DataFrame:
    """Return rows where search_string appears in the specified column."""
    return df[df[column_name].str.contains(search_string, na=False)]


def standardise_minimal_headers(df: pd.DataFrame, log: logging.Logger = None) -> pd.DataFrame:
    """
    Rename columns to 'Well' and 'Time [min]' by fuzzy matching.

    Raises ValueError if columns cannot be uniquely identified.
    """
    _log = log or logger
    df = df.copy()

    well_col_candidates = [col for col in df.columns if re.search(r'well', str(col), re.IGNORECASE)]
    time_col_candidates = [col for col in df.columns if re.search(r'time.*min', str(col), re.IGNORECASE)]

    if len(well_col_candidates) != 1:
        raise ValueError("Cannot uniquely identify 'Well' column. Candidates: %s" % well_col_candidates)
    if len(time_col_candidates) != 1:
        raise ValueError("Cannot uniquely identify 'Time [min]' column. Candidates: %s" % time_col_candidates)

    rename_map = {well_col_candidates[0]: "Well", time_col_candidates[0]: "Time [min]"}
    df.rename(columns=rename_map, inplace=True)
    _log.debug("Renamed columns: %s", rename_map)
    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def create_plot_object(df: pd.DataFrame, **kwargs) -> Optional[pg.PlotWidget]:
    """
    Create a PlotWidget from a DataFrame containing time-series data.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with 'Time [min]' and optionally 'Well' columns.
    **kwargs
        line_size, scatter_state, show_legend, title, show_x_label,
        show_y_label, black_white_state, grid_state, channel_map

    Returns
    -------
    pg.PlotWidget or None
    """
    from core.common.plot_style import (
        configure_pyqtgraph_widget, get_trace_color,
        get_time_label, get_signal_label
    )

    if df is None or df.empty:
        logger.error("No data to plot: DataFrame is None or empty.")
        return None
    if "Time [min]" not in df.columns:
        logger.error("'Time [min]' column not found. Cannot plot.")
        return None

    # Extract settings
    line_size = kwargs.get("line_size", 1.8)
    scatter_state = kwargs.get("scatter_state", False)
    show_legend = kwargs.get("show_legend", True)
    title = kwargs.get("title", None)
    show_x_label = kwargs.get("show_x_label", True)
    show_y_label = kwargs.get("show_y_label", True)
    black_white = kwargs.get("black_white_state", True)
    grid_state = kwargs.get("grid_state", False)
    channel_map = kwargs.get("channel_map", {})

    # Create and configure widget
    plot_widget = pg.PlotWidget()
    x_label = get_time_label() if show_x_label else ""
    y_label = get_signal_label() if show_y_label else ""

    configure_pyqtgraph_widget(
        plot_widget,
        x_label=x_label,
        y_label=y_label,
        title=title,
        enable_grid=grid_state,
        background='w' if black_white else 'k'
    )

    symbols = ["o", "s", "t", "d", "+", "x", "star", "p", "h"]

    # Add legend
    if show_legend:
        try:
            legend = plot_widget.addLegend(offset=(5, 5), labelTextSize='8pt')
            if legend:
                legend.setBrush(pg.mkBrush('w'))
                legend.setPen(pg.mkPen(color='k', width=1))
        except Exception as e:
            logger.error("Failed to create legend: %s", e)

    # Plot traces
    try:
        if "Well" in df.columns:
            _plot_with_well_column(df, plot_widget, symbols, channel_map,
                                   scatter_state, line_size, title)
        else:
            _plot_without_well_column(df, plot_widget, symbols,
                                      scatter_state, line_size, title)
    except Exception as e:
        logger.error("Error creating plot items: %s", e)

    # Axis visibility
    try:
        if not show_x_label:
            plot_widget.getPlotItem().hideAxis("bottom")
        if not show_y_label:
            plot_widget.getPlotItem().hideAxis("left")
    except Exception as e:
        logger.debug("Axis visibility update failed: %s", e)

    return plot_widget


def _plot_with_well_column(df, plot_widget, symbols, channel_map,
                           scatter_state, line_size, title):
    """Plot traces when DataFrame has a 'Well' column (multi-channel data)."""
    from core.common.plot_style import get_trace_color

    if len(df.columns) <= 2:
        return

    unique_wells = [w for w in df["Well"].unique() if pd.notna(w)]
    if not unique_wells:
        return

    well_columns = df.columns[2:].tolist()
    trace_idx = 0

    for channel in unique_wells:
        for well_col_name in well_columns:
            sub_df = df[df["Well"] == channel]
            x = pd.to_numeric(sub_df["Time [min]"], errors="coerce")
            y = pd.to_numeric(sub_df[well_col_name], errors="coerce")
            mask = ~x.isna() & ~y.isna()
            if not mask.any():
                continue

            x_vals = x[mask].values
            y_vals = y[mask].values
            colour = pg.mkColor(get_trace_color(trace_idx))
            symbol = symbols[trace_idx % len(symbols)]
            channel_display = channel_map.get(channel, channel) if channel_map else channel
            trace_label = "%s (%s)" % (well_col_name, channel_display)

            _add_trace(plot_widget, x_vals, y_vals, colour, symbol,
                       trace_label, scatter_state, line_size)
            trace_idx += 1

    if not title:
        channel_names = [channel_map.get(ch, ch) if channel_map else ch for ch in unique_wells]
        plot_widget.setTitle("Channels: %s" % ', '.join(map(str, channel_names)))


def _plot_without_well_column(df, plot_widget, symbols, scatter_state, line_size, title):
    """Plot traces when DataFrame has no 'Well' column (simple time-series)."""
    from core.common.plot_style import get_trace_color

    data_cols = [c for c in df.columns if c != "Time [min]"]
    if not data_cols:
        return

    for idx, col in enumerate(data_cols):
        colour = pg.mkColor(get_trace_color(idx))
        symbol = symbols[idx % len(symbols)]
        x = pd.to_numeric(df["Time [min]"], errors="coerce")
        y = pd.to_numeric(df[col], errors="coerce")
        mask = ~x.isna() & ~y.isna()
        if not mask.any():
            continue

        _add_trace(plot_widget, x[mask].values, y[mask].values, colour, symbol,
                   str(col), scatter_state, line_size)

    if not title:
        plot_widget.setTitle("Channel Data")


def _add_trace(plot_widget, x_vals, y_vals, colour, symbol, label,
               scatter_state, line_size):
    """Add a single trace to the plot widget."""
    if scatter_state:
        plot_widget.plot(
            x_vals, y_vals,
            pen=None,
            symbol=symbol,
            symbolPen=colour,
            symbolBrush=colour,
            symbolSize=3,
            name=label
        )
    else:
        plot_widget.plot(
            x_vals, y_vals,
            pen=pg.mkPen(colour, width=line_size),
            symbol=None,
            symbolSize=0,
            name=label
        )


def toggle_scatter(plot_widget, state):
    """
    Toggle scatter mode for existing PlotDataItems in the plot.
    Preserves colours by applying symbol pens/brushes that match the line pen.
    """
    try:
        scatter = (state == QtCore.Qt.Checked) or (state is True)

        try:
            data_items = plot_widget.getPlotItem().listDataItems()
        except Exception:
            data_items = [
                it for it in plot_widget.getPlotItem().items
                if isinstance(it, pg.PlotDataItem)
            ]

        for item in data_items:
            try:
                if scatter:
                    colour_qt = _get_item_colour(item)
                    item.setSymbol('o')
                    item.setSymbolSize(5)
                    item.setSymbolPen(pg.mkPen(colour_qt, width=1))
                    item.setSymbolBrush(pg.mkBrush(colour_qt))
                else:
                    item.setSymbol(None)
            except Exception as e:
                logger.warning("toggle_scatter: item update failed: %s", e)

        # Toggle standalone ScatterPlotItems
        try:
            for item in plot_widget.getPlotItem().items:
                if isinstance(item, pg.ScatterPlotItem):
                    item.setVisible(scatter)
        except Exception:
            pass

    except Exception as e:
        logger.error("toggle_scatter failed: %s", e)


def _get_item_colour(item):
    """Extract colour from a PlotDataItem's pen or symbol pen."""
    pen = item.opts.get('pen') if hasattr(item, 'opts') else None
    try:
        if pen is not None and hasattr(pen, 'color'):
            return pen.color()
    except Exception:
        pass

    sp = item.opts.get('symbolPen') if hasattr(item, 'opts') else None
    try:
        if sp is not None and hasattr(sp, 'color'):
            return sp.color()
    except Exception:
        pass

    return pg.mkColor(120, 120, 120)


# ---------------------------------------------------------------------------
# Session persistence
# ---------------------------------------------------------------------------

def save_recent_data(df: pd.DataFrame, dict_: dict):
    """Serialise and save the recent data to user data directory."""
    try:
        file_path = get_data_path('config/last_run_data.pkl')
        with open(file_path, 'wb') as file:
            pickle.dump({'csvdict': dict_, 'merged_dataframe': df}, file)
        logger.info("Data saved to %s", file_path)
    except Exception as e:
        logger.error("Error saving data: %s", e)


def load_lastrun_data() -> Tuple[Optional[pd.DataFrame], Optional[dict]]:
    """
    Load previously saved session data from user data directory.

    Returns (DataFrame, data_dict) or (None, None) if not found.
    """
    try:
        file_path = get_data_path('config/last_run_data.pkl')
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            return data.get('merged_dataframe'), data.get('csvdict')
    except FileNotFoundError:
        logger.info("No previous session data found.")
        return None, None
    except Exception as e:
        logger.error("Error loading session data: %s", e)
        return None, None


# ---------------------------------------------------------------------------
# Calibration file management
# ---------------------------------------------------------------------------

def check_and_sort_calibration_file(calibration_data_filename=None):
    """
    Ensure calibration file exists, handle invalid JSON, and sort by species name.

    Parameters
    ----------
    calibration_data_filename : str or Path, optional
        Path to calibration file. If None, uses default location.
    """
    if calibration_data_filename is None:
        calibration_data_filename = get_data_path("config/calibration_data.json")
    else:
        calibration_data_filename = Path(calibration_data_filename)

    Path(calibration_data_filename).parent.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(calibration_data_filename):
        reply = QMessageBox.question(
            None, "File Not Found",
            "Calibration data file '%s' not found.\nCreate a new empty calibration file?" % calibration_data_filename,
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            try:
                with open(calibration_data_filename, 'w') as f:
                    json.dump([], f, indent=2)
                logger.info("Created new empty calibration file.")
            except Exception as e:
                logger.error("Could not create calibration file: %s", e)
        return

    try:
        with open(calibration_data_filename, 'r') as f:
            calibration_data = json.load(f)

        if not isinstance(calibration_data, list):
            reply = QMessageBox.question(
                None, "Invalid Data Format",
                "Calibration file is invalid.\nReset to empty list?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                with open(calibration_data_filename, 'w') as f:
                    json.dump([], f, indent=2)
            return

        calibration_data.sort(key=lambda x: x.get('species', '').lower())
        with open(calibration_data_filename, 'w') as f:
            json.dump(calibration_data, f, indent=2)

    except json.JSONDecodeError:
        reply = QMessageBox.question(
            None, "Invalid JSON Format",
            "Calibration file contains invalid JSON.\nReset to empty list?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            try:
                with open(calibration_data_filename, 'w') as f:
                    json.dump([], f, indent=2)
            except Exception as e:
                logger.error("Failed to reset calibration file: %s", e)
    except Exception as e:
        QMessageBox.warning(None, "File Error", "Error loading calibration data: %s" % e)
        logger.warning("Error loading calibration data: %s", e)
