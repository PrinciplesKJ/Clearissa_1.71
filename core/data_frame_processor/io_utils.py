"""
I/O utilities for the DataFrameProcessor.

Handles data export, loading from previous runs, and saving or restoring
user preferences. All file-related and persistent state operations
are isolated here for clarity and reuse.
"""

import logging
import os
import sys
from pathlib import Path
from tokenize import blank_re

import pandas as pd
import re
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtCore import QSettings
from core.common.data_processing_utils import load_lastrun_data
from core.common.ui_theme import Colors

# Add parent directory to path to import resource_utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from resource_utils import get_data_path


class IOUtils:
    """Provides file export, settings management, and persistence for DataFrameProcessor."""

    def __init__(self, processor):
        self.p = processor
        self.logger = logging.getLogger(__name__)

    # -------------------------------------------------------------------------
    # LOAD LAST RUN
    # -------------------------------------------------------------------------
    def load_last_run(self):
        """Load the last processed dataset and associated metadata."""
        try:
            df, csvdict = load_lastrun_data()
            if df is None or df.empty:
                self.logger.warning("No last-run data available.")
                return None, None
            self.logger.info("Loaded last run data: %s", str(df.shape))
            return df, csvdict
        except Exception as e:
            self.logger.error("Failed to load last run data: %s", str(e))
            QMessageBox.critical(None, "Error", f"Failed to load data: {str(e)}")
            return None, None

    # -------------------------------------------------------------------------
    # EXPORT DATA
    # -------------------------------------------------------------------------
    def export_filtered_data(self):
        """Export the currently filtered dataset to CSV with user-configurable options."""
        self.logger.info("Export initiated from View Data tab.")

        # Get current selections
        start_time, end_time = self.p.get_start_end_times()
        ch_dict = getattr(self.p, "selected_channels_checkboxes", {})
        wells = getattr(self.p, "select_wells_checkboxes", {})

        # Get selected channels
        from PyQt5.QtWidgets import QLabel
        if ch_dict and isinstance(list(ch_dict.values())[0], QLabel):
            selected_channels = [
                ch for ch, label in ch_dict.items()
                if getattr(label, "_selected", False)
            ]
        else:
            selected_channels = [ch for ch, chk in ch_dict.items() if chk.isChecked()]

        # Filter the data
        df = self.p.dataops.filter_data(start_time, end_time, ch_dict, wells)
        if df is None or df.empty:
            QMessageBox.information(None, "No Data", "No data available for export.")
            self.logger.info("Export aborted: no filtered data.")
            return

        # Normalize time to minutes
        df, time_col = self._normalise_time_to_minutes(df)

        # Log export parameters
        self.logger.info(f"Export parameters - Channels: {selected_channels}, "
                        f"Wells: {list(wells.keys())}, Time range: {start_time:.2f}-{end_time:.2f}")


        # Ask user to choose export structure
        structure_choice = self._prompt_export_structure()
        if structure_choice is None:
            self.logger.info("Export cancelled by user during structure selection.")
            return

        # Check if time is subset and ask about rebasing
        time_rebased = False
        if self._is_time_subset(df, time_col):
            time_rebase_choice = self._prompt_time_rebasing()
            if time_rebase_choice is None:
                self.logger.info("Export cancelled by user during time rebasing selection.")
                return
            time_rebased = time_rebase_choice

        # Apply time rebasing if requested
        if time_rebased:
            df = self._rebase_time_to_zero(df, time_col)
            self.logger.info("Time rebased to start at zero.")

        # Clean and prepare data
        df[time_col] = pd.to_numeric(df[time_col], errors="coerce").round(3)
        df = df.dropna(subset=[time_col])
        df.sort_values(by=[time_col], inplace=True)

        # Export based on structure choice
        if structure_choice == "native":
            self._export_native_structure(df, time_col, start_time, end_time, time_rebased)
        else:  # minimal
            self._export_minimal_structure(df, time_col, selected_channels, start_time, end_time, time_rebased)

    def _prompt_export_structure(self):
        """
        Prompt user to select export structure format.

        Returns:
            str: 'native' or 'minimal', or None if cancelled
        """
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton

        dialog = QDialog()
        dialog.setWindowTitle("Select Export Structure")
        dialog.setMinimumWidth(400)

        layout = QVBoxLayout(dialog)

        # Header
        header = QLabel("<b>Choose the dataframe structure for export:</b>")
        layout.addWidget(header)

        # Option 1: Native structure
        native_label = QLabel(
            "<b>a) Native Structure</b><br>"
            "Format: <i>Well | Time [min] | Data columns...</i><br>"
            "The 'Well' column contains channel/well identifiers.<br>"
            "Single file with all selected channels."
        )
        native_label.setWordWrap(True)
        native_label.setStyleSheet("padding: 10px; background-color: {Colors.INFO_PANEL_INTERNAL_BG}; border-radius: 5px; margin: 5px;")
        layout.addWidget(native_label)

        native_btn = QPushButton("Export as Native Structure")
        native_btn.clicked.connect(lambda: dialog.done(1))
        layout.addWidget(native_btn)

        # Option 2: Minimal structure
        minimal_label = QLabel(
            "<b>b) Minimal Structure</b><br>"
            "Format: <i>Time [min] | Data columns...</i><br>"
            "The 'Well' column is omitted.<br>"
            "If multiple channels selected, one file per channel will be created."
        )
        minimal_label.setWordWrap(True)
        minimal_label.setStyleSheet("padding: 10px; background-color: {Colors.INFO_PANEL_HMSD_BG}; border-radius: 5px; margin: 5px;")
        layout.addWidget(minimal_label)

        minimal_btn = QPushButton("Export as Minimal Structure")
        minimal_btn.clicked.connect(lambda: dialog.done(2))
        layout.addWidget(minimal_btn)

        # Cancel button
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(lambda: dialog.done(0))
        layout.addWidget(cancel_btn)

        result = dialog.exec_()

        if result == 1:
            return "native"
        elif result == 2:
            return "minimal"
        else:
            return None

    def _prompt_time_rebasing(self):
        """
        Prompt user whether to rebase time to zero.

        Returns:
            bool: True if rebase, False if keep original, None if cancelled
        """
        reply = QMessageBox.question(
            None,
            "Time Adjustment",
            "A subset of the time range is selected.\n\n"
            "Would you like to rebase time to start at zero in the exported file?",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
        )

        if reply == QMessageBox.Yes:
            return True
        elif reply == QMessageBox.No:
            return False
        else:
            return None

    def _is_time_subset(self, df, time_col):
        """
        Check if the current time range is a subset of the total available time.

        Returns:
            bool: True if subset, False if full range
        """
        if self.p.merged_dataframe is None or self.p.merged_dataframe.empty:
            return False

        # Get full time range from merged dataframe
        full_df = self.p.merged_dataframe.copy()
        if time_col not in full_df.columns:
            full_df, _ = self._normalise_time_to_minutes(full_df)

        full_min = full_df[time_col].min()
        full_max = full_df[time_col].max()

        # Get current selection range
        current_min = df[time_col].min()
        current_max = df[time_col].max()

        # Check if it's a subset (with small tolerance for floating point comparison)
        tolerance = 0.01
        is_subset = (current_min > full_min + tolerance) or (current_max < full_max - tolerance)

        return is_subset

    def _rebase_time_to_zero(self, df, time_col):
        """
        Rebase time column to start at zero.

        Args:
            df: DataFrame to modify
            time_col: Name of time column

        Returns:
            pd.DataFrame: Modified dataframe with rebased time
        """
        try:
            t0 = df[time_col].min()
            df = df.copy()
            df[time_col] = df[time_col] - t0
            return df
        except Exception as e:
            self.logger.error(f"Failed to rebase time: {e}")
            return df

    def _normalise_time_to_minutes(self, df):
        """
        Ensure time column exists and is properly named.

        Args:
            df: DataFrame to process

        Returns:
            tuple: (DataFrame, time_column_name)
        """
        # By the time data reaches export, it should already have 'Time [min]'
        if 'Time [min]' in df.columns:
            return df, 'Time [min]'

        # Fallback: look for any time column
        time_cols = [col for col in df.columns if 'time' in str(col).lower()]

        if not time_cols:
            raise ValueError("No time column found in dataframe for export")

        return df, time_cols[0]

    def _extract_experiment_name(self):
        """
        Extract a clean experiment name for use in filenames.

        Returns:
            str: Sanitized experiment name or 'data_export' as fallback
        """
        csvdict = getattr(self.p, "csvdict", None)

        if not csvdict:
            return "data_export"

        # Try to get experiment name from first file entry
        for file_data in csvdict.values():
            if isinstance(file_data, dict):
                experiment = file_data.get('experiment')
                if experiment and experiment not in [None, 'N/A', '']:
                    # Sanitize for filename
                    return self._sanitize_filename(str(experiment))

        # Fallback to a generic name
        return "data_export"

    def _export_native_structure(self, df, time_col, start_time, end_time, time_rebased):
        """
        Export data in native structure format (with Well column).

        Args:
            df: DataFrame to export
            time_col: Name of time column
            start_time: Start time for filename
            end_time: End time for filename
            time_rebased: Whether time was rebased
        """
        exp_name = self._extract_experiment_name()
        time_suffix = "rebased" if time_rebased else f"{start_time:.2f}-{end_time:.2f}"
        suggested = f"{exp_name}_native_{time_suffix}.csv"

        path, _ = QFileDialog.getSaveFileName(None, "Export Data (Native Structure)", suggested, "CSV Files (*.csv)")
        if not path:
            self.logger.info("Export cancelled by user during file selection.")
            return

        try:
            df.to_csv(path, index=False, encoding='utf-8')
            QMessageBox.information(None, "Export Successful", f"Data exported to:\n{path}")
            self.logger.info(f"Native structure export successful: {path}")
        except Exception as e:
            QMessageBox.critical(None, "Export Error", f"Failed to export data:\n{str(e)}")
            self.logger.error(f"Export failed: {e}")

    def _export_minimal_structure(self, df, time_col, selected_channels, start_time, end_time, time_rebased):
        """
        Export data in minimal structure format (without Well column).
        If multiple channels selected, create one file per channel.

        Args:
            df: DataFrame to export
            time_col: Name of time column
            selected_channels: List of selected channel names
            start_time: Start time for filename
            end_time: End time for filename
            time_rebased: Whether time was rebased
        """
        exp_name = self._extract_experiment_name()
        time_suffix = "rebased" if time_rebased else f"{start_time:.2f}-{end_time:.2f}"

        if len(selected_channels) == 1:
            # Single channel - single file
            channel = selected_channels[0]
            channel_clean = self._sanitize_filename(self.p.channel_map.get(channel, channel))
            suggested = f"{exp_name}_{channel_clean}_{time_suffix}.csv"

            path, _ = QFileDialog.getSaveFileName(None, "Export Data (Minimal Structure)", suggested, "CSV Files (*.csv)")
            if not path:
                self.logger.info("Export cancelled by user during file selection.")
                return

            # Create minimal structure (drop Well column)
            df_export = df[df['Well'] == channel].copy()
            df_export = df_export.drop(columns=['Well'])

            try:
                df_export.to_csv(path, index=False, encoding='utf-8')
                QMessageBox.information(None, "Export Successful", f"Data exported to:\n{path}")
                self.logger.info(f"Minimal structure export successful: {path}")
            except Exception as e:
                QMessageBox.critical(None, "Export Error", f"Failed to export data:\n{str(e)}")
                self.logger.error(f"Export failed: {e}")

        else:
            # Multiple channels - ask for directory and create one file per channel
            directory = QFileDialog.getExistingDirectory(None, "Select Directory for Multiple Channel Export")
            if not directory:
                self.logger.info("Export cancelled by user during directory selection.")
                return

            import os
            exported_files = []

            for channel in selected_channels:
                channel_clean = self._sanitize_filename(self.p.channel_map.get(channel, channel))
                filename = f"{exp_name}_{channel_clean}_{time_suffix}.csv"
                path = os.path.join(directory, filename)

                # Create minimal structure for this channel
                df_channel = df[df['Well'] == channel].copy()
                df_channel = df_channel.drop(columns=['Well'])

                try:
                    df_channel.to_csv(path, index=False, encoding='utf-8')
                    exported_files.append(filename)
                    self.logger.info(f"Exported channel {channel} to: {path}")
                except Exception as e:
                    self.logger.error(f"Failed to export channel {channel}: {e}")

            if exported_files:
                files_list = "\n".join(exported_files)
                QMessageBox.information(
                    None,
                    "Export Successful",
                    f"Exported {len(exported_files)} file(s) to:\n{directory}\n\nFiles:\n{files_list}"
                )
                self.logger.info(f"Multi-channel minimal export successful: {len(exported_files)} files")
            else:
                QMessageBox.warning(None, "Export Warning", "No files were exported successfully.")

    def _sanitize_filename(self, name):
        """
        Sanitize a string for use in filenames (ASCII-safe).

        Args:
            name: String to sanitize

        Returns:
            str: Sanitized filename-safe string
        """
        import re
        # Remove or replace non-ASCII characters
        name = str(name).encode('ascii', 'ignore').decode('ascii')
        # Replace spaces and special chars with underscores
        name = re.sub(r'[^\w.-]', '_', name)
        # Remove multiple underscores
        name = re.sub(r'_+', '_', name)
        # Remove leading/trailing underscores
        name = name.strip('_')
        return name if name else "channel"

    # -------------------------------------------------------------------------
    # SETTINGS SAVE/LOAD
    # -------------------------------------------------------------------------
    def save_settings(self):
        """Persist GUI selections and plotting state to QSettings with experiment-specific well mappings."""
        try:
            s = QSettings("Clearissa", "DataFrameProcessor")
            s.beginGroup("UserPreferences")

            # Get experiment identifier for this dataset
            exp_id = self._get_experiment_identifier()
            self.logger.info(f"Saving settings for experiment: {exp_id}")

            # Channels (label-based)
            if hasattr(self.p, "selected_channels_checkboxes"):
                ch_selected = [
                    c for c, lbl in self.p.selected_channels_checkboxes.items()
                    if getattr(lbl, "_selected", False)
                ]
                s.setValue("selected_channels", ch_selected)

            # Save well selection state using WellSelectionManager (JSON format)
            config_dir = get_data_path() / 'config'
            config_dir.mkdir(parents=True, exist_ok=True)
            well_state_file = config_dir / f'well_selection_{exp_id}.json'
            self.p.well_selection_manager.save_state(well_state_file)
            self.logger.info(f"Saved well selection state for {exp_id}")

            # Unit selections
            if hasattr(self.p, "original_unit_combo") and hasattr(self.p, "display_unit_combo"):
                s.setValue("original_unit", self.p.original_unit_combo.currentText())
                s.setValue("display_unit", self.p.display_unit_combo.currentText())

            # Time range selections
            if hasattr(self.p, "start_time_spinbox") and hasattr(self.p, "end_time_spinbox"):
                s.setValue("start_time", self.p.start_time_spinbox.value())
                s.setValue("end_time", self.p.end_time_spinbox.value())

            # Plot states
            s.setValue("scatter_state", getattr(self.p, "scatter_state", False))
            s.setValue("black_white_state", getattr(self.p, "black_white_state", True))
            s.setValue("grid_state", getattr(self.p, "grid_state", False))
            s.setValue("legend_state", getattr(self.p, "legend_state", True))

            s.endGroup()
            self.logger.info(f"User settings saved successfully for experiment: {exp_id}")
        except Exception as e:
            self.logger.error("Failed to save settings: %s", e)

    def load_settings(self):
        """Restore user preferences from QSettings with experiment-specific well mappings."""
        try:
            s = QSettings("Clearissa", "DataFrameProcessor")
            s.beginGroup("UserPreferences")

            # Get experiment identifier for current dataset
            exp_id = self._get_experiment_identifier()
            self.logger.info(f"Loading settings for experiment: {exp_id}")

            # Restore channels
            ch_sel = s.value("selected_channels", [])
            if ch_sel and hasattr(self.p, "selected_channels_checkboxes"):
                for ch in ch_sel:
                    if ch in self.p.selected_channels_checkboxes:
                        lbl = self.p.selected_channels_checkboxes[ch]
                        lbl._selected = True
                        colour = self.p.channel_colours.get(ch, "#8BC34A")
                        text_colour = self.p.plotutils.get_contrast_text_colour(colour)
                        lbl.setStyleSheet(
                            f"QLabel {{background-color:{colour}; color:{text_colour}; "
                            f"font-weight:bold; border:2px solid {colour}; border-radius:3px; padding:5px;}}"
                        )

            # Get valid wells from current data before loading well selections
            valid_wells = set(self.p.columns) if hasattr(self.p, 'columns') else set()
            self.logger.info(f"Valid wells in current dataset: {len(valid_wells)} wells")

            # Load well selection state from JSON file
            config_dir = get_data_path() / 'config'
            well_state_file = config_dir / f'well_selection_{exp_id}.json'

            if os.path.exists(well_state_file):
                self.logger.info(f"Loading well selection state from {well_state_file}")
                success = self.p.well_selection_manager.load_state(well_state_file)

                # Validate loaded wells against current data
                if success:
                    invalid_wells = []
                    for well_id in list(self.p.well_selection_manager.well_states.keys()):
                        if well_id not in valid_wells:
                            # Remove wells that don't exist in current data
                            if self.p.well_selection_manager.well_states[well_id]["category"] != "unassigned":
                                invalid_wells.append(well_id)
                            del self.p.well_selection_manager.well_states[well_id]

                    if invalid_wells:
                        self.logger.warning(
                            f"Removed {len(invalid_wells)} invalid wells from loaded settings: {invalid_wells[:10]}...")
            else:
                self.logger.info(f"No saved well selection state found for experiment: {exp_id}")

            # Restore unit selections (block signals to avoid triggering multiple plot updates)
            if hasattr(self.p, "original_unit_combo"):
                orig_unit = s.value("original_unit", "Minutes")
                idx = self.p.original_unit_combo.findText(orig_unit)
                if idx >= 0:
                    try:
                        self.p.original_unit_combo.blockSignals(True)
                        self.p.original_unit_combo.setCurrentIndex(idx)
                    finally:
                        self.p.original_unit_combo.blockSignals(False)

            if hasattr(self.p, "display_unit_combo"):
                disp_unit = s.value("display_unit", "Minutes")
                idx = self.p.display_unit_combo.findText(disp_unit)
                if idx >= 0:
                    try:
                        self.p.display_unit_combo.blockSignals(True)
                        self.p.display_unit_combo.setCurrentIndex(idx)
                    finally:
                        self.p.display_unit_combo.blockSignals(False)

            # Restore time range selections (block signals to avoid triggering plot updates)
            if hasattr(self.p, "start_time_spinbox") and hasattr(self.p, "end_time_spinbox"):
                start_time = s.value("start_time", type=float)
                end_time = s.value("end_time", type=float)

                # Only restore if values are valid and within data range
                if start_time is not None and end_time is not None:
                    try:
                        self.p.start_time_spinbox.blockSignals(True)
                        self.p.end_time_spinbox.blockSignals(True)

                        # Validate against current data range
                        min_val = self.p.start_time_spinbox.minimum()
                        max_val = self.p.end_time_spinbox.maximum()

                        if min_val <= start_time <= max_val:
                            self.p.start_time_spinbox.setValue(start_time)
                        if min_val <= end_time <= max_val:
                            self.p.end_time_spinbox.setValue(end_time)

                        self.logger.info(f"Restored time range: {start_time:.2f} - {end_time:.2f}")
                    finally:
                        self.p.start_time_spinbox.blockSignals(False)
                        self.p.end_time_spinbox.blockSignals(False)

            # Restore plot states
            self.p.scatter_state = s.value("scatter_state", False, type=bool)
            self.p.black_white_state = s.value("black_white_state", True, type=bool)
            self.p.grid_state = s.value("grid_state", False, type=bool)
            self.p.legend_state = s.value("legend_state", True, type=bool)

            # Update well button colours after loading (with safety check for deleted widgets)
            try:
                # Use _refresh_well_display() for modern well selection system
                if hasattr(self.p, '_refresh_well_display'):
                    self.p._refresh_well_display()
                else:
                    # Fallback to legacy method
                    self.p.update_well_button_colours()
            except RuntimeError as e:
                self.logger.warning("Could not update well button colours (widgets may not be fully initialised): %s", e)

            s.endGroup()
            self.logger.info(f"User settings loaded successfully for experiment: {exp_id}")
        except Exception as e:
            self.logger.error("Failed to load settings: %s", e)
    # -------------------------------------------------------------------------
    # HELPERS
    # -------------------------------------------------------------------------
    def _get_experiment_identifier(self):
        """
        Generate a unique experiment identifier from csvdict metadata.
        This identifier is used to map well selections to specific experiments.
        """
        def find_first_key(target, d):
            if target in d:
                return d[target]
            for v in d.values():
                if isinstance(v, dict):
                    res = find_first_key(target, v)
                    if res:
                        return res
            return None

        csvdict = getattr(self.p, "csvdict", None)
        if not csvdict:
            return "default_experiment"

        # Try to find experiment and info fields
        experiment_val = None
        info_val = None

        for v in csvdict.values():
            if isinstance(v, dict):
                if experiment_val is None:
                    experiment_val = find_first_key("experiment", v)
                if info_val is None:
                    info_val = find_first_key("info", v)
                if experiment_val and info_val:
                    break

        # Create identifier from experiment and info
        if experiment_val and info_val:
            identifier = f"{experiment_val}_{info_val}"
        elif info_val:
            identifier = str(info_val)
        elif experiment_val:
            identifier = str(experiment_val)
        else:
            identifier = "default_experiment"

        # Sanitize identifier
        identifier = str(identifier).strip()
        identifier = re.sub(r"^ID1:\s*", "", identifier)
        identifier = re.sub(r"\s+", "_", identifier)
        identifier = re.sub(r"[^\w.-]", "", identifier)

        return identifier


    # -------------------------------------------------------------------------
    # STATE SAVE
    # -------------------------------------------------------------------------
    def save_state(self):
        """Save current user state on close or tab switch."""
        self.save_settings()
