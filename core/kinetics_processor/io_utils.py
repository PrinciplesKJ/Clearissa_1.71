"""
IO Utilities for Kinetics Processor
------------------------------------
Handles data loading, file operations, and export functionality.

Author: Križan Jurinović
Date: October 2025
"""

import os
import sys
import re
import logging
import base64
import pickle
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from PyQt5.QtWidgets import QFileDialog, QMessageBox

# Add resource_utils import for proper path handling in frozen executables
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from resource_utils import get_data_path

# Import centralised time normalisation utilities
from .data_processor import normalise_time_column, filter_time_window

logger = logging.getLogger(__name__)


class DataFormatError(Exception):
    """Exception raised for data format errors."""
    pass


class KineticsIOUtils:
    """Handles all file I/O operations for kinetics data."""

    # Regex definitions for extracting well names and time columns
    _WELL_REGEX = re.compile(r"^[A-Ha-h][0-9]{2}(?:_.*)?$")
    _TIME_KEYWORDS = ("time", "t", "timestamp")

    def __init__(self, parent_widget):
        """
        Initialise IO utilities.

        Args:
            parent_widget: Parent widget for displaying dialogs
        """
        self.parent = parent_widget
        self.filename = ""
        self.time_col = "Time"
        self.detected_format = None
        self.current_dataset_hash = None  # Unique identifier for loaded dataset

    def calculate_dataset_hash(self, df, filename=""):
        """
        Calculate a SHA-256 hash for a dataset based on shape, columns, and sampled rows.

        Used to detect whether the same data has been re-loaded, so that groups
        and fitted results can be preserved or cleared accordingly.
        """
        try:
            hasher = hashlib.sha256()

            # Include filename
            hasher.update(filename.encode('utf-8'))

            # Include shape
            hasher.update(str(df.shape).encode('utf-8'))

            # Include column names
            hasher.update(','.join(df.columns).encode('utf-8'))

            # Include first and last 10 rows for efficiency
            # (full dataset hash would be slow for large files)
            head_data = df.head(10).to_numpy()
            tail_data = df.tail(10).to_numpy()
            hasher.update(head_data.tobytes())
            hasher.update(tail_data.tobytes())

            # Include sum of all numeric values as additional fingerprint
            numeric_sum = df.select_dtypes(include=[np.number]).sum().sum()
            hasher.update(str(numeric_sum).encode('utf-8'))

            hash_value = hasher.hexdigest()
            logger.debug("Dataset hash calculated: %s (file: %s, shape: %s)",
                        hash_value[:16] + "...", filename, df.shape)

            return hash_value

        except Exception as e:
            logger.error("Failed to calculate dataset hash: %s", e)
            # Return a random hash if calculation fails
            return hashlib.sha256(str(np.random.random()).encode()).hexdigest()

    def load_data(self):
        """
        Load CSV file with automatic format detection using unified detector.

        Returns:
            pd.DataFrame or None: Cleaned numeric DataFrame with standardised structure
        """
        file_path = self._prompt_user_for_file()
        if not file_path:
            logger.info("Data load cancelled by user.")
            return None

        logger.info("=" * 60)
        logger.info("[KINETICS LOAD] Loading data from: %s", file_path)
        logger.info("=" * 60)

        try:
            # Detect format by checking first line
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip().lower()

            # Determine skip_rows based on whether first line starts with 'well' or 'time'
            if first_line.startswith('well') or first_line.startswith('time'):
                skip_rows = 0
                format_desc = "Minimal format"
            else:
                skip_rows = 0  # Will need to find data start
                format_desc = "Standard format with metadata"

            self.detected_format = format_desc

            logger.info("[KINETICS LOAD] Detected format: %s", format_desc)

            # Load CSV with appropriate skip rows
            raw_df = pd.read_csv(file_path, skiprows=skip_rows)
            raw_df.columns = raw_df.columns.str.strip()

            logger.debug("[KINETICS LOAD] Loaded shape: %s", raw_df.shape)
            logger.debug("[KINETICS LOAD] Columns: %s", list(raw_df.columns)[:5])

        except DataFormatError as exc:
            logger.error("[KINETICS LOAD] Format error: %s", exc)
            self._show_critical("Format Error", str(exc))
            return None
        except Exception as exc:
            logger.exception("[KINETICS LOAD] Error reading CSV: %s", exc)
            self._show_critical("Load Error", f"Error reading CSV file:\n{exc}")
            return None

        try:
            # Standardise DataFrame structure
            numeric_df = self._standardise_dataframe(raw_df, format_desc)

            if numeric_df is None or numeric_df.empty:
                raise DataFormatError(
                    "No valid numeric data found after processing.\n\n"
                    "Please check that your file contains:\n"
                    "- A valid 'Time' column\n"
                    "- One or more columns with numeric values\n"
                    "- No corrupted or non-numeric entries in data columns"
                )

            self.filename = os.path.basename(file_path)

            # Calculate dataset hash for identity tracking
            self.current_dataset_hash = self.calculate_dataset_hash(numeric_df, self.filename)

            logger.info("[KINETICS LOAD] - SUCCESS")
            logger.info("  - File: %s", self.filename)
            logger.info("  - Format: %s", format_desc)
            logger.info("  - Shape: %s", numeric_df.shape)
            logger.info("  - Time column: %s", self.time_col)
            logger.info("  - Dataset ID: %s...", self.current_dataset_hash[:16])
            logger.info("=" * 60)

            return numeric_df

        except Exception as exc:
            logger.exception("[KINETICS LOAD] Error processing data: %s", exc)
            self._show_critical("Processing Error", f"Error processing data:\n{exc}")
            return None

    def _standardise_dataframe(self, df: pd.DataFrame, format_type: str) -> pd.DataFrame:
        """
        Standardise DataFrame to consistent structure for kinetics analysis.

        Args:
            df: Raw DataFrame from CSV
            format_type: Detected format type

        Returns:
            Standardised DataFrame with time column first, then data columns
        """
        logger.debug("[KINETICS STANDARDISE] Processing format: %s", format_type)

        # Handle minimal format without Well column (add synthetic Well column if needed)
        first_col = str(df.columns[0]).strip().lower()
        if first_col.startswith('time') and 'well' not in [str(c).strip().lower() for c in df.columns]:
            logger.debug("[KINETICS STANDARDISE] Adding synthetic 'Well' column")
            df.insert(0, 'Well', 'Global')

        # Detect and process Well column
        well_col = self._detect_well_column(df)
        time_col = self._detect_time_column(df)

        numeric_df = df.copy()

        # Remove Well column (kinetics works with individual traces)
        if well_col:
            numeric_df.drop(columns=well_col, inplace=True)
            logger.debug("[KINETICS STANDARDIZE] Dropped well column '%s'", well_col)

        # Reorder columns: time first, then data
        columns_order = [c for c in numeric_df.columns if c != time_col]
        numeric_df = numeric_df[[time_col] + columns_order]

        # Convert to numeric
        numeric_df[time_col] = pd.to_numeric(numeric_df[time_col], errors="coerce")
        self.time_col = time_col

        for col in columns_order:
            numeric_df[col] = pd.to_numeric(numeric_df[col], errors="coerce")

        # Clean up
        numeric_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        numeric_df.dropna(how="all", axis=0, inplace=True)
        numeric_df.dropna(how="all", axis=1, inplace=True)

        logger.debug("[KINETICS STANDARDIZE] Final shape: %s", numeric_df.shape)
        logger.debug("[KINETICS STANDARDIZE] Data columns: %d", len(columns_order))

        return numeric_df

    def _prompt_user_for_file(self) -> str:
        """Open the file-dialog and return the chosen path or an empty string."""
        file_path, _ = QFileDialog.getOpenFileName(
            self.parent,
            "Select CSV file with converted values for kinetic analysis",
            "",
            "CSV Files (*.csv)",
        )
        return file_path

    def _detect_well_column(self, df: pd.DataFrame):
        """Return the column name that looks like *Well* (or *None*)."""
        for col in df.columns:
            if col.strip().lower().startswith("well"):
                logger.debug("[KINETICS DETECT] Well column by name: '%s'", col)
                return col

        candidate_cols = []
        for col in df.columns:
            series = df[col].dropna().astype(str)
            if not series.empty and series.map(self._WELL_REGEX.match).all():
                candidate_cols.append(col)

        if len(candidate_cols) > 1:
            logger.warning("[KINETICS DETECT] Multiple well columns found: %s - using first", candidate_cols)
        return candidate_cols[0] if candidate_cols else None

    def _detect_time_column(self, df: pd.DataFrame) -> str:
        """Return the name of the column that represents time (raise if not found)."""
        for col in df.columns:
            if any(key in col.lower() for key in self._TIME_KEYWORDS):
                logger.debug("[KINETICS DETECT] Time column by name: '%s'", col)
                return col

        for col in df.columns:
            numeric_count = pd.to_numeric(df[col], errors="coerce").notna().sum()
            if numeric_count == len(df):
                logger.debug("[KINETICS DETECT] Time column heuristically: '%s'", col)
                return col

        raise DataFormatError("Unable to detect a valid time column. Make sure your CSV contains one.")

    def _show_critical(self, title: str, message: str) -> None:
        """Show critical error dialog."""
        QMessageBox.critical(self.parent, title, message)

    def export_hd_graph(self, figure):
        """Export the graph as a high-resolution figure (vector or 600 dpi PNG)."""
        file_dialog = QFileDialog(self.parent)
        file_path, _ = file_dialog.getSaveFileName(
            self.parent,
            "Save Graph",
            "",
            "Vector Files (*.pdf *.svg *.eps);;PNG Image (*.png)"
        )

        if file_path:
            try:
                ext = file_path.split('.')[-1].lower()
                if ext in ['pdf', 'svg', 'eps']:
                    figure.savefig(file_path, bbox_inches='tight')
                elif ext == 'png':
                    figure.savefig(file_path, dpi=600, bbox_inches='tight')
                else:
                    QMessageBox.warning(self.parent, "Unsupported Format", f"Unsupported file extension: .{ext}")
                    return

                logger.info("Graph exported as '%s'", file_path)
            except Exception as e:
                logger.error("Graph export failed: %s", e)
                QMessageBox.critical(self.parent, "Export Error", f"Graph export failed: {e}")
        else:
            logger.info("Export cancelled.")

    def export_results_with_plot_file(self, plot_file_path, results_html, model_info,
                                      experiment_info, mean_rate_constant, fitted_df, time_col,
                                      replicate_stats_df=None, replicate_info=None,
                                      time_window_start=None, time_window_end=None,
                                      is_catalytic=False):
        """
        Export analysis results to a timestamped folder.

        Args:
            plot_file_path: Path to temporary plot image file
            results_html: HTML string with fitting results table
            model_info: HTML string with model information
            experiment_info: User-provided experiment notes
            mean_rate_constant: Tuple of (mean_kf, mean_kr)
            fitted_df: DataFrame with fitted results
            time_col: Name of time column
            replicate_stats_df: Optional DataFrame with replicate statistics (mean, std, sem, n)
            replicate_info: Optional dict with replicate group metadata
            time_window_start: Optional start time of analysis window (for normalisation)
            time_window_end: Optional end time of analysis window (for validation)
            is_catalytic: Boolean indicating if catalytic model is being used
        """
        # Ask user to select parent directory
        parent_dir = QFileDialog.getExistingDirectory(
            self.parent,
            "Select folder to save results",
            "",
            QFileDialog.ShowDirsOnly
        )
        
        if not parent_dir:
            logger.info("Export cancelled by user.")
            # Clean up temporary plot file
            try:
                os.remove(plot_file_path)
            except:
                pass
            return

        # Create timestamped folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_folder = os.path.join(parent_dir, f"kinetics_results_{timestamp}")
        
        try:
            os.makedirs(export_folder, exist_ok=True)
            logger.info(f"Created export folder: {export_folder}")

            # 1. Export fitted data to CSV with 4-decimal formatting
            csv_path = os.path.join(export_folder, "fitted_data.csv")
            time_reference = None  # Store reference time for consistency

            if fitted_df is not None and not fitted_df.empty:
                # CRITICAL: Normalise time to start from ZERO for export
                # Use the FIRST DATA POINT in the filtered window
                # Store this reference to use for all exported files
                formatted_df = fitted_df.copy()

                # Diagnostic logging before normalisation
                logger.info("CSV export - Before normalisation: %d rows, time range %.4f to %.4f",
                           len(formatted_df),
                           formatted_df[time_col].min() if time_col in formatted_df.columns else 0,
                           formatted_df[time_col].max() if time_col in formatted_df.columns else 0)

                if time_col in formatted_df.columns:
                    # Store the reference time for use with other exports
                    time_reference = formatted_df[time_col].min()
                    logger.info("Normalising to first data point (%.4f min) so export starts at time=0", time_reference)

                    # Normalise using centralised utility
                    normalise_time_column(formatted_df, time_col, time_reference)

                    # Verify normalisation and log first few time values
                    actual_min = formatted_df[time_col].min()
                    first_five_times = formatted_df[time_col].head(10).tolist()
                    logger.info("CSV export - After normalisation: first 10 time values: %s",
                               [f"{t:.4f}" for t in first_five_times])

                    if abs(actual_min) > 0.01:
                        logger.error("Time normalisation failed - time starts at %.4f instead of 0", actual_min)
                    else:
                        logger.info("Time normalised for fitted_data.csv: shifted by %.2f min, now starts at %.4f",
                                   time_reference, actual_min)
                else:
                    logger.warning("Time column '%s' not found in fitted_df for normalisation", time_col)

                # Format all numeric columns to 4 decimal places
                for col in formatted_df.columns:
                    if formatted_df[col].dtype in ['float64', 'float32']:
                        formatted_df[col] = formatted_df[col].round(4)

                # CRITICAL: Verify data integrity before CSV export
                logger.info("CSV export - Final check before writing: %d rows, %d columns",
                           len(formatted_df), len(formatted_df.columns))
                logger.info("CSV export - DataFrame index: %s", formatted_df.index.tolist()[:10])
                logger.info("CSV export - Time column is sorted: %s",
                           formatted_df[time_col].is_monotonic_increasing if time_col in formatted_df.columns else "N/A")

                # Export to CSV
                formatted_df.to_csv(csv_path, index=False, float_format='%.4f')

                # Verify what was actually written by reading back first few rows
                verification_df = pd.read_csv(csv_path, nrows=10)
                if time_col in verification_df.columns:
                    logger.info("CSV export - VERIFICATION: First 10 time values in written file: %s",
                               verification_df[time_col].tolist())
                else:
                    logger.error("CSV export - VERIFICATION: Time column '%s' not found in written CSV!", time_col)

                logger.info("Exported fitted data to CSV with 4-decimal formatting: %s", csv_path)

            # 2. Export group statistics to separate CSV if available
            if replicate_stats_df is not None and not replicate_stats_df.empty:
                group_csv_path = os.path.join(export_folder, "group_statistics.csv")
                # Format group stats with 4 decimals and normalise time
                formatted_group_df = replicate_stats_df.copy()

                # CRITICAL FIX: Use SAME time reference as fitted_data.csv
                # This ensures time columns align across all exported files
                if time_col in formatted_group_df.columns and time_reference is not None:
                    logger.info("Normalising group stats to same reference as fitted data (%.4f min)", time_reference)

                    # Normalise using centralised utility with SAME reference
                    normalise_time_column(formatted_group_df, time_col, time_reference)

                    # Verify normalisation
                    actual_min_group = formatted_group_df[time_col].min()
                    if abs(actual_min_group) > 0.01:
                        logger.error("Group stats time normalisation failed - time starts at %.4f", actual_min_group)
                    else:
                        logger.info("Time normalised for group_statistics.csv: shifted by %.2f min, now starts at %.4f",
                                   time_reference, actual_min_group)
                elif time_reference is None:
                    logger.warning("No time reference available - group stats will not be normalised")

                for col in formatted_group_df.columns:
                    if formatted_group_df[col].dtype in ['float64', 'float32']:
                        formatted_group_df[col] = formatted_group_df[col].round(4)
                formatted_group_df.to_csv(group_csv_path, index=False, float_format='%.4f')
                logger.info("Exported group statistics to CSV: %s", group_csv_path)

            # 3. Export to Excel with multiple sheets
            self._export_to_excel(export_folder, fitted_df, time_col, replicate_stats_df, replicate_info,
                                 time_window_start)

            # 4. Generate HTML report
            self._generate_html_report(
                export_folder, plot_file_path, results_html, model_info,
                experiment_info, mean_rate_constant, fitted_df, time_col, timestamp,
                replicate_stats_df, is_catalytic
            )

            # 5. Copy the plot image to export folder
            import shutil
            plot_export_path = os.path.join(export_folder, "kinetics_plot.png")
            shutil.copy(plot_file_path, plot_export_path)
            logger.info("Copied plot to: %s", plot_export_path)

            # Clean up temporary file
            try:
                os.remove(plot_file_path)
            except:
                pass

            # Build success message based on what was exported
            contents_list = [
                "• fitted_data.csv - All data with fitted curves (4 decimal precision)",
                "• kinetics_analysis.xlsx - Multi-sheet Excel workbook"
            ]

            if replicate_stats_df is not None and not replicate_stats_df.empty:
                contents_list.insert(1, "• group_statistics.csv - Replicate group means, errors, and sample sizes")

            contents_list.extend([
                "• analysis_report.html - Complete analysis report",
                "• kinetics_plot.png - High-resolution plot"
            ])

            # Show success message
            QMessageBox.information(
                self.parent,
                "Export Complete",
                f"Results exported successfully to:\n\n{export_folder}\n\n"
                f"Contents:\n" + "\n".join(contents_list)
            )
            logger.info("Export completed successfully")
            
        except Exception as e:
            logger.error(f"Export failed: {e}", exc_info=True)
            QMessageBox.critical(
                self.parent,
                "Export Error",
                f"Failed to export results:\n{e}"
            )

    def _export_to_excel(self, export_folder, fitted_df, time_col, replicate_stats_df=None, replicate_info=None,
                        time_window_start=None):
        """
        Export data to Excel with multiple sheets (All Data, Group Statistics,
        Fitted Curves, Summary). All numeric values rounded to 4 decimal places.

        Args:
            export_folder: Folder to save Excel file
            fitted_df: DataFrame with all data
            time_col: Name of time column
            replicate_stats_df: Optional DataFrame with replicate statistics
            replicate_info: Optional dict with replicate group metadata
            time_window_start: Optional start time of analysis window (for normalisation)
        """
        try:
            # Check if openpyxl is available
            try:
                import openpyxl
            except ImportError:
                logger.warning("openpyxl not available - skipping Excel export")
                return
            
            excel_path = os.path.join(export_folder, "kinetics_analysis.xlsx")

            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                sheets_created = 0  # Track number of sheets created

                # Sheet 1: All data (formatted to 4 decimals, time normalised)
                if fitted_df is not None and not fitted_df.empty:
                    formatted_df = fitted_df.copy()

                    # Normalise time to start from zero using first data point
                    # CRITICAL: Store this reference for consistent normalisation across all sheets
                    time_reference = None
                    if time_col in formatted_df.columns:
                        time_reference = formatted_df[time_col].min()
                        logger.info("Excel 'All Data' sheet: normalising to first data point (%.4f min)", time_reference)

                        # Normalise using centralised utility
                        normalise_time_column(formatted_df, time_col, time_reference)
                        actual_min = formatted_df[time_col].min()

                        # Log first few time values for verification
                        first_times = formatted_df[time_col].head(10).tolist()
                        logger.info("Excel 'All Data' sheet: first 10 time values after normalisation: %s",
                                   [f"{t:.4f}" for t in first_times])

                        logger.info("Excel 'All Data' sheet: time normalised by %.2f min, now starts at %.4f",
                                   time_reference, actual_min)

                    for col in formatted_df.columns:
                        if formatted_df[col].dtype in ['float64', 'float32']:
                            formatted_df[col] = formatted_df[col].round(4)
                    formatted_df.to_excel(writer, sheet_name='All Data', index=False)
                    sheets_created += 1

                    # Sheet 2: Group Statistics (use SAME reference time as All Data)
                    if replicate_stats_df is not None and not replicate_stats_df.empty:
                        formatted_group_df = replicate_stats_df.copy()

                        # CRITICAL FIX: Use same time reference as All Data sheet
                        # This ensures time columns align across all sheets
                        if time_col in formatted_group_df.columns and time_reference is not None:
                            logger.info("Excel 'Group Statistics' sheet: normalising to same reference (%.4f min)",
                                       time_reference)

                            # Normalise using centralised utility with SAME reference
                            normalise_time_column(formatted_group_df, time_col, time_reference)

                        for col in formatted_group_df.columns:
                            if formatted_group_df[col].dtype in ['float64', 'float32']:
                                formatted_group_df[col] = formatted_group_df[col].round(4)

                        # Add explanatory header row if groups are present
                        formatted_group_df.to_excel(writer, sheet_name='Group Statistics', index=False, startrow=1)

                        # Get workbook and sheet to add header
                        workbook = writer.book
                        worksheet = writer.sheets['Group Statistics']
                        worksheet.insert_rows(0)
                        worksheet['A1'] = "Replicate Group Statistics - Mean +/- SEM for each group"
                        worksheet['A1'].font = openpyxl.styles.Font(bold=True, size=12)
                        sheets_created += 1

                        logger.info("Exported group statistics with %d groups", len(replicate_info) if replicate_info else 0)
                    else:
                        # Fallback: try to extract from fitted_df if replicate_stats_df not provided
                        mean_cols = [time_col] + [c for c in formatted_df.columns
                                                  if 'Mean' in c or 'Std' in c or 'SEM' in c]
                        if len(mean_cols) > 1:
                            means_df = formatted_df[mean_cols].copy()
                            for col in means_df.columns:
                                if means_df[col].dtype in ['float64', 'float32']:
                                    means_df[col] = means_df[col].round(4)
                            means_df.to_excel(writer, sheet_name='Group Statistics', index=False)
                            sheets_created += 1

                    # Sheet 3: Fitted curves only (formatted to 4 decimals)
                    # CRITICAL: Use normalised formatted_df, not original fitted_df
                    fitted_cols = [time_col] + [c for c in formatted_df.columns if c.endswith('_fitted')]
                    if len(fitted_cols) > 1:
                        fitted_only_df = formatted_df[fitted_cols].copy()

                        # Log first few time values for verification
                        if time_col in fitted_only_df.columns:
                            first_times = fitted_only_df[time_col].head(10).tolist()
                            logger.info("Excel 'Fitted Curves' sheet: first 10 time values: %s",
                                       [f"{t:.4f}" for t in first_times])

                        for col in fitted_only_df.columns:
                            if fitted_only_df[col].dtype in ['float64', 'float32']:
                                fitted_only_df[col] = fitted_only_df[col].round(4)
                        fitted_only_df.to_excel(writer, sheet_name='Fitted Curves', index=False)
                        sheets_created += 1

                    # Sheet 4: Create summary statistics (formatted to 4 decimals)
                    summary_data = self._calculate_summary_statistics(fitted_df, time_col)
                    if summary_data:
                        summary_df = pd.DataFrame(summary_data)
                        for col in summary_df.columns:
                            if summary_df[col].dtype in ['float64', 'float32']:
                                summary_df[col] = summary_df[col].round(4)
                        summary_df.to_excel(writer, sheet_name='Summary', index=False)
                        sheets_created += 1

                # If no sheets were created, create a placeholder sheet to avoid Excel error
                if sheets_created == 0:
                    logger.warning("No data available for Excel export - creating placeholder sheet")
                    placeholder_df = pd.DataFrame({
                        'Message': ['No kinetics data available for export'],
                        'Info': ['Please load and fit data before exporting']
                    })
                    placeholder_df.to_excel(writer, sheet_name='Info', index=False)

            logger.info("Exported to Excel with 4-decimal formatting: %s", excel_path)
            
        except Exception as e:
            logger.error(f"Excel export failed: {e}", exc_info=True)

    def _calculate_summary_statistics(self, fitted_df, time_col):
        """
        Calculate summary statistics for each trace.
        
        Args:
            fitted_df: DataFrame with fitted data
            time_col: Name of time column
            
        Returns:
            List of dictionaries with summary statistics
        """
        summary = []
        
        for col in fitted_df.columns:
            if col == time_col or col.endswith('_fitted') or col.endswith(' Std') or col.endswith(' SEM'):
                continue
            
            data = fitted_df[col].dropna()
            if len(data) == 0:
                continue
            
            summary.append({
                'Trace': col,
                'N Points': len(data),
                'Mean': data.mean(),
                'Std Dev': data.std(),
                'Min': data.min(),
                'Max': data.max(),
                'Final Value': data.iloc[-1] if len(data) > 0 else np.nan
            })
        
        return summary

    def _generate_html_report(self, export_folder, plot_file_path, results_html,
                              model_info, experiment_info, mean_rate_constant,
                              fitted_df, time_col, timestamp, replicate_stats_df=None,
                              is_catalytic=False):
        """
        Generate HTML analysis report with embedded plot.

        Args:
            export_folder: Folder to save HTML file
            plot_file_path: Path to plot image
            results_html: HTML with fitting results
            model_info: HTML with model information
            experiment_info: User experiment notes (not used)
            mean_rate_constant: Tuple of rate constants (only used for bimolecular)
            fitted_df: DataFrame with fitted data
            time_col: Time column name
            timestamp: Export timestamp
            replicate_stats_df: Optional DataFrame with replicate statistics
            is_catalytic: Boolean indicating if catalytic model is being used
        """
        html_path = os.path.join(export_folder, "analysis_report.html")
        
        # Read and encode plot image
        try:
            with open(plot_file_path, 'rb') as f:
                plot_data = base64.b64encode(f.read()).decode('utf-8')
            plot_img_tag = f'<img src="data:image/png;base64,{plot_data}" style="max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; padding: 5px;">'
        except:
            plot_img_tag = '<p style="color: red;">Plot image not available</p>'


        # Build HTML content
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kinetics Analysis Report - {timestamp}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 8px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin: 0 0 10px 0;
            font-size: 32px;
        }}
        .header p {{
            margin: 5px 0;
            font-size: 14px;
            opacity: 0.9;
        }}
        .section {{
            background: white;
            padding: 25px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        .section h2 {{
            margin-top: 0;
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        .section h3 {{
            color: #764ba2;
            margin-top: 20px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: 600;
            color: #495057;
        }}
        tr:hover {{
            background-color: #f8f9fa;
        }}
        .metadata {{
            background: #f8f9fa;
            padding: 15px;
            border-left: 4px solid #667eea;
            margin: 15px 0;
            border-radius: 4px;
        }}
        .metadata p {{
            margin: 8px 0;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #e0e0e0;
            color: #666;
            font-size: 12px;
        }}
        .plot-container {{
            text-align: center;
            margin: 20px 0;
            padding: 15px;
            background: #fafafa;
            border-radius: 8px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Kinetics Analysis Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime("%d %B %Y at %H:%M:%S")}</p>
        <p><strong>Data file:</strong> {self.filename if self.filename else 'N/A'}</p>
    </div>
    
    <div class="section">
        <h2>Experimental Data Plot</h2>
        <div class="plot-container">
            {plot_img_tag}
        </div>
    </div>
    
    <div class="section">
        <h2>Fitting Results</h2>
        {results_html if results_html else '<p>No fitting results available</p>'}
    </div>
    
    <div class="footer">
        <p>Generated by Clearissa Kinetics Processor</p>
    </div>
</body>
</html>
"""
        
        # Write HTML file
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Generated HTML report: {html_path}")

    def save_state(self, state_dict):
        """
        Serialise processor state to a pickle file for session recovery.

        Parameters
        ----------
        state_dict : dict
            Complete state dictionary (data, parameters, UI settings, etc.).
        """
        try:
            # Use proper path resolution for user data directory
            state_file = get_data_path("config/kinetics_last_state.pkl")

            with open(state_file, 'wb') as f:
                pickle.dump(state_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

            logger.info(f"State saved to {state_file}")

        except Exception as e:
            logger.error(f"Failed to save state: {e}", exc_info=True)

    def load_state(self):
        """
        Deserialise processor state from pickle file.

        Returns
        -------
        dict or None
            State dictionary, or None if no file exists or loading fails.
        """
        try:
            # Use proper path resolution for user data directory
            state_file = get_data_path("config/kinetics_last_state.pkl")

            with open(state_file, 'rb') as f:
                state_dict = pickle.load(f)

            logger.info(f"State loaded from {state_file}")
            return state_dict

        except Exception as e:
            logger.error(f"Failed to load state: {e}", exc_info=True)
            return None
