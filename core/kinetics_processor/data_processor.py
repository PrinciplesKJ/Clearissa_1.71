"""
Data Processing Module for Kinetics Processor
----------------------------------------------
Handles data loading, replicate detection, and data transformations.

Author: Krizan Jurinovic
Date: November 2025
"""

import logging
import re
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QMessageBox
from .replicate_manager import ReplicateManager

logger = logging.getLogger(__name__)

# Floating-point tolerance for time comparisons
# Updated from 1e-9 to 1e-6 (0.06 milliseconds when time in minutes)
# This provides robust boundary matching without being overly permissive
FLOAT_TOLERANCE = 1e-6


def normalise_time_column(df, time_col, reference_time):
    """
    Normalise time column to start from zero by subtracting a reference time.

    This utility function provides a consistent way to normalise time across
    all modules (plotting, fitting, export). Using a centralised function
    ensures consistent behaviour and reduces code duplication.

    **IMPORTANT:** This function modifies the DataFrame IN-PLACE. Callers should
    use df.copy() before calling if the original data must be preserved.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing time-series data (will be modified in-place)
    time_col : str
        Name of the time column to normalise
    reference_time : float
        Reference time to subtract (typically the start of the time window)

    Returns
    -------
    pd.DataFrame
        The same DataFrame reference (modified in-place) for convenience

    Notes
    -----
    - The input DataFrame is modified IN-PLACE for efficiency
    - Returns the same DataFrame reference for method chaining
    - Logs the normalisation operation for traceability
    - Validates that normalisation produces expected result (starting near 0)

    Examples
    --------
    >>> df_copy = df.copy()  # IMPORTANT: Make copy first!
    >>> df_normalised = normalise_time_column(df_copy, 'Time', 10.0)
    >>> # Time column now starts at 0 instead of 10.0
    """
    if time_col not in df.columns:
        logger.warning("Time column '%s' not found in DataFrame for normalisation", time_col)
        return df

    original_min = df[time_col].min()
    df[time_col] = df[time_col] - reference_time
    normalised_min = df[time_col].min()

    logger.debug("Normalised time column '%s': shifted by %.4f min (from %.4f to %.4f)",
                time_col, reference_time, original_min, normalised_min)

    # Verify normalisation worked correctly with stricter threshold
    # Updated from 0.01 min to 1e-4 min for better validation
    if abs(normalised_min) > 1e-4:
        logger.warning("Time normalisation verification failed: expected ~0, got %.6f (tolerance: 1e-4)",
                      normalised_min)

    return df


def filter_time_window(df, time_col, t_start, t_end, tolerance=FLOAT_TOLERANCE):
    """
    Filter DataFrame to include only data within the specified time window.

    Uses floating-point tolerance to handle boundary precision issues and
    ensures consistent boundary behaviour across all modules.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing time-series data
    time_col : str
        Name of the time column
    t_start : float
        Start time (inclusive with tolerance)
    t_end : float
        End time (inclusive with tolerance)
    tolerance : float, optional
        Floating-point tolerance for boundary comparisons (default: 1e-9)

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only rows within [t_start, t_end]

    Notes
    -----
    - Uses >= and <= with tolerance to avoid floating-point precision issues
    - Returns an empty DataFrame if no data points fall within the window
    - Logs warning if the window contains no data

    Examples
    --------
    >>> df_windowed = filter_time_window(df, 'Time', 10.0, 60.0)
    >>> # Returns only rows where Time is between 10.0 and 60.0 minutes
    """
    if time_col not in df.columns:
        logger.error("Time column '%s' not found in DataFrame", time_col)
        return df

    # Apply window with tolerance
    mask = (df[time_col] >= t_start - tolerance) & (df[time_col] <= t_end + tolerance)
    df_windowed = df[mask].copy()

    n_before = len(df)
    n_after = len(df_windowed)

    if n_after == 0:
        logger.warning("Time window [%.4f, %.4f] contains no data points", t_start, t_end)
    else:
        logger.debug("Filtered time window: %d -> %d rows (%.1f%% retained)",
                    n_before, n_after, 100 * n_after / n_before if n_before > 0 else 0)

    return df_windowed


def validate_time_coordinates(df, time_col, expected_start, expected_end,
                              tolerance=1e-4, context=""):
    """
    Validate that DataFrame time coordinates match expectations.

    This function checks for common time coordinate issues that could indicate
    data pipeline problems such as incorrect normalisation or filtering.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate
    time_col : str
        Name of time column
    expected_start : float
        Expected start time (minutes)
    expected_end : float
        Expected end time (minutes)
    tolerance : float, optional
        Tolerance for start/end matching (default: 1e-4 min = 0.006 sec)
    context : str, optional
        Context string for logging (e.g., "fitting worker", "export")

    Returns
    -------
    bool
        True if valid, False if issues detected

    Warnings Logged
    ---------------
    - Time column not monotonic increasing
    - Start/end times differ from expected by more than tolerance
    - Time column contains NaN or inf values
    """
    if time_col not in df.columns:
        logger.error("%s: Time column '%s' not found", context, time_col)
        return False

    is_valid = True
    actual_start = df[time_col].min()
    actual_end = df[time_col].max()

    # Check monotonic (sort order)
    if not df[time_col].is_monotonic_increasing:
        logger.warning("%s: Time column not monotonic increasing", context)
        is_valid = False

    # Check start time
    if abs(actual_start - expected_start) > tolerance:
        logger.warning("%s: Start time mismatch - expected %.6f, got %.6f (diff: %.6f)",
                      context, expected_start, actual_start, actual_start - expected_start)
        is_valid = False

    # Check end time
    if abs(actual_end - expected_end) > tolerance:
        logger.warning("%s: End time mismatch - expected %.6f, got %.6f (diff: %.6f)",
                      context, expected_end, actual_end, actual_end - expected_end)
        is_valid = False

    # Check for invalid values
    if df[time_col].isna().any():
        n_nan = df[time_col].isna().sum()
        logger.error("%s: Time column contains %d NaN values", context, n_nan)
        is_valid = False

    if np.isinf(df[time_col]).any():
        n_inf = np.isinf(df[time_col]).sum()
        logger.error("%s: Time column contains %d inf values", context, n_inf)
        is_valid = False

    if is_valid:
        logger.debug("%s: Time coordinates validated successfully [%.6f, %.6f]",
                    context, actual_start, actual_end)

    return is_valid


def validate_array_alignment(time_array, data_array, fitted_array=None, context=""):
    """
    Validate that time and data arrays are properly aligned.

    This function ensures arrays used in fitting and plotting have compatible
    shapes and NaN patterns, preventing subtle bugs from array mismatches.

    Parameters
    ----------
    time_array : np.ndarray
        Time coordinate array
    data_array : np.ndarray
        Data value array
    fitted_array : np.ndarray, optional
        Fitted curve array (if available)
    context : str, optional
        Context string for logging

    Returns
    -------
    bool
        True if arrays are properly aligned, False otherwise

    Checks Performed
    ----------------
    - All arrays have same length
    - NaN patterns are compatible
    - Time array is sorted
    - No length mismatches between data and fitted curves
    """
    n_time = len(time_array)
    n_data = len(data_array)

    if n_time != n_data:
        logger.error("%s: Array length mismatch - time: %d, data: %d",
                    context, n_time, n_data)
        return False

    if fitted_array is not None:
        n_fitted = len(fitted_array)
        if n_fitted != n_time:
            logger.error("%s: Fitted array length mismatch - time: %d, fitted: %d",
                        context, n_time, n_fitted)
            return False

    # Check for corresponding NaN values (optional - log warning if different)
    time_valid = np.isfinite(time_array)
    data_valid = np.isfinite(data_array)

    if not np.array_equal(time_valid, data_valid):
        n_time_valid = time_valid.sum()
        n_data_valid = data_valid.sum()
        logger.debug("%s: Valid point counts differ - time: %d, data: %d",
                    context, n_time_valid, n_data_valid)

    # Check time is sorted
    if not np.all(time_array[:-1] <= time_array[1:]):
        logger.warning("%s: Time array is not sorted", context)
        return False

    logger.debug("%s: Array alignment validated - length: %d", context, n_time)
    return True


class DataProcessor:
    """
    Handles all data processing operations for kinetics analysis.

    Responsibilities:
    - Load and validate data frames
    - Detect and process replicates
    - Apply data transformations (offsets, windowing)
    - Calculate replicate statistics
    """

    def __init__(self, parent_widget):
        """
        Initialise the data processor.

        Parameters
        ----------
        parent_widget : QWidget
            Parent widget for displaying message boxes
        """
        self.parent = parent_widget
        self.replicate_info = {}
        self.replicate_stats_df = None
        self.replicate_manager = ReplicateManager(parent_widget)
        self.user_defined_groups = {}  # Store user-defined replicate groups

    @staticmethod
    def convert_time_to_seconds(time_min):
        """
        Convert time from minutes to seconds.

        Parameters
        ----------
        time_min : float or np.ndarray
            Time value(s) in minutes

        Returns
        -------
        float or np.ndarray
            Time value(s) in seconds
        """
        return time_min * 60.0

    @staticmethod
    def convert_time_to_minutes(time_sec):
        """
        Convert time from seconds to minutes.

        Parameters
        ----------
        time_sec : float or np.ndarray
            Time value(s) in seconds

        Returns
        -------
        float or np.ndarray
            Time value(s) in minutes
        """
        return time_sec / 60.0

    def load_dataframe_directly(self, df, label, time_col_hint=None):
        """
        Load and validate a DataFrame for kinetics analysis.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing time-series data with 'Well' and time columns
        label : str
            Label describing the dataset source
        time_col_hint : str, optional
            Hint for the time column name

        Returns
        -------
        dict
            Dictionary containing validated data and metadata:
            - 'data_df': processed DataFrame
            - 'time_col': name of time column
            - 'filename': label for the dataset
            - 'time_range': tuple of (min_time, max_time)
        """
        if df is None or df.empty:
            QMessageBox.warning(
                self.parent,
                "Invalid Data",
                "Cannot load empty DataFrame."
            )
            return None

        # Validate required columns
        if 'Well' not in df.columns:
            QMessageBox.warning(
                self.parent,
                "Invalid Data",
                "DataFrame must contain a 'Well' column."
            )
            return None

        # Find time column
        time_col = time_col_hint
        if time_col is None or time_col not in df.columns:
            for col in df.columns:
                if 'time' in col.lower():
                    time_col = col
                    break

        if time_col is None:
            QMessageBox.warning(
                self.parent,
                "Invalid Data",
                "DataFrame must contain a time column (e.g., 'Time [min]')."
            )
            return None

        # Copy and store the DataFrame
        data_df = df.copy()

        logger.info("Loaded DataFrame directly: %s", label)
        logger.info("  Shape: %d rows x %d columns", df.shape[0], df.shape[1])
        logger.info("  Time column: %s", time_col)

        # Calculate time range
        max_time = data_df[time_col].max()
        min_time = data_df[time_col].min()

        logger.info("Data time range: %.2f to %.2f", min_time, max_time)

        return {
            'data_df': data_df,
            'time_col': time_col,
            'filename': label,
            'time_range': (min_time, max_time)
        }

    def process_replicates(self, df, time_col, show_dialogue=False, gui_handler=None):
        """
        Process replicates - either using user-defined groups or auto-detection.

        Parameters
        ----------
        df : pd.DataFrame
            Data frame containing experimental data
        time_col : str
            Name of the time column
        show_dialogue : bool
            Whether to show the replicate management dialogue
        gui_handler : KineticsGUI, optional
            GUI handler for accessing button styles

        Returns
        -------
        pd.DataFrame
            Original data frame (unmodified)
        """
        if df is None or time_col not in df.columns:
            logger.warning("Cannot initialise trace list without valid data and a time column.")
            return df

        # Get all data columns (exclude time & fitted curves)
        data_cols = [c for c in df.columns if c != time_col and not c.endswith("_fitted")]

        # If dialogue requested, show replicate manager
        if show_dialogue:
            # First, auto-detect groups as suggestions
            auto_detected = self._auto_detect_groups(data_cols)

            # Show dialogue
            user_groups = self.replicate_manager.show_replicate_dialogue(
                data_cols,
                auto_detected,
                gui_handler
            )

            if user_groups:
                self.user_defined_groups = user_groups
                logger.info("User defined %d replicate groups", len(user_groups))

        # Calculate statistics based on user-defined groups or auto-detection
        if self.user_defined_groups:
            # Use user-defined groups
            result = self.replicate_manager.calculate_group_statistics(
                df, time_col, self.user_defined_groups
            )
            if result:
                self.replicate_stats_df, self.replicate_info = result
                logger.info("Using user-defined replicate groups (%d groups)",
                           len(self.user_defined_groups))
            else:
                self.replicate_stats_df = None
                self.replicate_info = {}
        else:
            # Fall back to auto-detection (silent)
            self.replicate_stats_df = self._detect_and_calculate_replicate_stats(df, time_col)
            logger.info("Using auto-detected replicate groups")

        if self.replicate_stats_df is not None and not self.replicate_stats_df.empty:
            logger.info("Replicate statistics available for %d groups",
                       len(self.replicate_info))

        return df

    def _auto_detect_groups(self, data_cols):
        """
        Automatically detect replicate groups based on well column patterns.

        Parameters
        ----------
        data_cols : list
            List of data column names

        Returns
        -------
        dict
            Dictionary of detected groups {group_name: [traces]}
        """
        replicate_groups = {}

        for col in data_cols:
            # Extract numeric suffix (well plate column identifier)
            match = re.search(r'(\d+)$', col)

            if match:
                well_column = match.group(1)
                group_key = f"Col{well_column}"

                if group_key not in replicate_groups:
                    replicate_groups[group_key] = []
                replicate_groups[group_key].append(col)

        # Filter to only groups with 2+ members
        replicate_groups = {k: v for k, v in replicate_groups.items() if len(v) >= 2}

        logger.debug("Auto-detected %d replicate groups", len(replicate_groups))
        return replicate_groups

    def _detect_and_calculate_replicate_stats(self, df, time_col):
        """
        Silently detect replicate groups and calculate statistics.

        Replicate detection logic:
        - Well plate replicates are identified by the same column (numeric suffix)
        - A01, B01, C01 are replicates (all in column "01")
        - A02, B02 are replicates (all in column "02")
        - Different numeric suffixes indicate different well columns

        Parameters
        ----------
        df : pd.DataFrame
            Data frame containing experimental data
        time_col : str
            Name of the time column

        Returns
        -------
        pd.DataFrame or None
            DataFrame with columns: Time, Col01_mean, Col01_std, Col01_sem, ...
            Returns None if no replicates detected
        """
        # Get data columns
        data_cols = [c for c in df.columns if c != time_col and not c.endswith("_fitted")]

        if len(data_cols) < 2:
            logger.debug("Too few traces to detect replicates")
            return None

        # Group detection: extract numeric well column identifier (trailing digits)
        replicate_groups = {}

        for col in data_cols:
            # Extract numeric suffix (well plate column identifier)
            # Examples: "A01" -> "01", "B02" -> "02", "Well_C_03" -> "03"
            match = re.search(r'(\d+)$', col)

            if match:
                well_column = match.group(1)
                if well_column not in replicate_groups:
                    replicate_groups[well_column] = []
                replicate_groups[well_column].append(col)

        # Filter to only groups with 2+ members (actual replicates)
        replicate_groups = {k: v for k, v in replicate_groups.items() if len(v) >= 2}

        if not replicate_groups:
            logger.debug("No replicate groups detected")
            return None

        logger.info("Detected %d replicate groups (well columns): %s",
                   len(replicate_groups), list(replicate_groups.keys()))

        # Calculate statistics for each group
        stats_df = df[[time_col]].copy()

        for well_col, members in replicate_groups.items():
            # Get data for all members in this well column
            group_data = df[members]

            # Create a descriptive group name
            group_name = f"Col{well_col}"

            # Calculate mean, std, sem
            stats_df[f"{group_name}_mean"] = group_data.mean(axis=1)
            stats_df[f"{group_name}_std"] = group_data.std(axis=1, ddof=1)
            stats_df[f"{group_name}_sem"] = group_data.sem(axis=1)
            stats_df[f"{group_name}_n"] = group_data.count(axis=1)

            # Store mapping in replicate_info
            self.replicate_info[group_name] = {
                'columns': members,
                'mean_col': f"{group_name}_mean",
                'std_col': f"{group_name}_std",
                'sem_col': f"{group_name}_sem"
            }

            logger.debug("Replicate group '%s': %s", group_name, members)

        return stats_df


    def validate_time_window(self, df, time_col, t_start, t_end):
        """
        Validate time window parameters.

        Parameters
        ----------
        df : pd.DataFrame
            Data frame containing experimental data
        time_col : str
            Name of the time column
        t_start : float
            Start time
        t_end : float
            End time

        Returns
        -------
        bool
            True if valid, False otherwise
        """
        if t_start >= t_end:
            logger.error("Start time must be less than end time")
            return False

        if df is None or df.empty:
            logger.error("No data available for time window validation")
            return False

        # Check if any data points fall within the window
        mask = (df[time_col] >= t_start) & (df[time_col] <= t_end)
        if not mask.any():
            logger.warning("No data points in specified time window")
            return False

        return True

    def apply_time_window(self, df, time_col, t_start, t_end):
        """
        Apply time window filter to data frame.

        Parameters
        ----------
        df : pd.DataFrame
            Data frame containing experimental data
        time_col : str
            Name of the time column
        t_start : float
            Start time
        t_end : float
            End time

        Returns
        -------
        pd.DataFrame
            Windowed data frame
        """
        # Ensure monotonic time
        if not df[time_col].is_monotonic_increasing:
            logger.debug("Time column not monotonic - sorting by %s", time_col)
            df = df.sort_values(time_col).reset_index(drop=True)

        # Apply window using centralised utility with floating-point tolerance
        df_windowed = filter_time_window(df, time_col, t_start, t_end)

        return df_windowed

    def manage_replicates_interactive(self, df, time_col, gui_handler=None,
                                      colour_palette=None):
        """
        Launch interactive replicate management dialogue.

        Parameters
        ----------
        df : pd.DataFrame
            Data frame containing experimental data
        time_col : str
            Name of the time column
        gui_handler : KineticsGUI, optional
            GUI handler for accessing button styles
        colour_palette : list, optional
            List of hex colour strings for graph colour preview

        Returns
        -------
        bool
            True if user defined groups, False if cancelled
        """
        if df is None or time_col not in df.columns:
            logger.warning("Cannot manage replicates without valid data")
            return False

        # Get all data columns
        data_cols = [c for c in df.columns if c != time_col and not c.endswith("_fitted")]

        # Auto-detect groups as suggestions (only if user hasn't defined any)
        if self.user_defined_groups:
            # Use existing user-defined groups
            initial_groups = self.user_defined_groups
        else:
            # Auto-detect groups as suggestions
            auto_detected = self._auto_detect_groups(data_cols)
            initial_groups = auto_detected if auto_detected else {}

        # Show dialogue
        user_groups = self.replicate_manager.show_replicate_dialogue(
            data_cols,
            initial_groups,
            gui_handler,
            colour_palette=colour_palette
        )

        if user_groups is not None:
            self.user_defined_groups = user_groups

            # Recalculate statistics with new groups
            if self.user_defined_groups:
                result = self.replicate_manager.calculate_group_statistics(
                    df, time_col, self.user_defined_groups
                )
                if result:
                    self.replicate_stats_df, self.replicate_info = result
                    logger.info("Updated replicate groups (%d groups)", len(self.user_defined_groups))
            else:
                # User cleared all groups - use auto-detection
                self.replicate_stats_df = self._detect_and_calculate_replicate_stats(df, time_col)
                logger.info("Cleared user groups - using auto-detection")

            return True

        return False

    def get_replicate_groups(self):
        """
        Get current replicate group definitions.

        Returns
        -------
        dict
            Dictionary of replicate groups {group_name: [traces]}
        """
        return self.user_defined_groups.copy()
