"""
Clearissa - settings_manager.py
--------------------------------
Persistent settings management for application state.

This module provides:
- Cross-session state persistence using QSettings
- Well selection and channel configuration storage
- Plot state and display preferences
- Automatic restoration on application restart

Author: Križan Jurinović
Date: October 2025
"""

from PyQt5.QtCore import QSettings
import logging


class SettingsManager:
    """
    Central manager for persistent user settings across sessions.

    Manages storage and retrieval of:
    - Selected wells (data, blank, positive control)
    - Channel selections
    - Time unit preferences
    - Plot display states (scatter, grid, colour schemes)

    Notes
    -----
    Uses QSettings for platform-independent storage.
    Settings are automatically persisted to registry (Windows)
    or configuration files (Linux/macOS).
    """

    def __init__(self):
        """
        Initialise settings manager.

        Creates QSettings instance for application-wide persistence.
        """
        # Create a single QSettings object used for the entire application
        self._settings = QSettings("MyCompany", "MyApp")
        self.logger = logging.getLogger(__name__)

    def save_data_frame_processor(self, dfproc):
        """
        Persist DataFrameProcessor state to settings.

        Parameters
        ----------
        dfproc : DataFrameProcessor
            Processor instance whose state should be saved.

        Notes
        -----
        Saved state includes:
        - Selected wells and control classifications
        - Active channels
        - Time unit selections
        - Plot display preferences
        """
        self.logger.info("SettingsManager: Saving DataFrameProcessor state...")
        self._settings.beginGroup("DataFrameProcessor")

        # Channels
        selected_channels = [
            ch for ch, chk in dfproc.selected_channels_checkboxes.items()
            if chk.isChecked()
        ]
        self._settings.setValue("selected_channels", selected_channels)

        # Well selection dictionaries
        self._settings.setValue("data_wells", list(dfproc.select_wells_checkboxes.keys()))
        self._settings.setValue("blank_wells", list(dfproc.select_blank_checkboxes.keys()))
        self._settings.setValue("pos_ctrl_wells", list(dfproc.select_posctrl_checkboxes.keys()))

        # Plotting states & combos
        self._settings.setValue("original_unit", dfproc.original_unit_combo.currentText())
        self._settings.setValue("display_unit", dfproc.display_unit_combo.currentText())
        self._settings.setValue("scatter_state", dfproc.scatter_state)
        self._settings.setValue("black_white_state", dfproc.black_white_state)
        self._settings.setValue("grid_state", dfproc.grid_state)

        self._settings.endGroup()
        self.logger.info("SettingsManager: DataFrameProcessor state saved.")

    def load_data_frame_processor(self, dfproc):
        """
        Load DataFrameProcessor state from settings.

        Parameters
        ----------
        dfproc : DataFrameProcessor
            Processor instance whose state should be loaded.

        Notes
        -----
        - Missing wells in the current dataset are skipped, and the user is warned.
        - Channel and well selections are restored, along with plot and unit settings.
        """
        self.logger.info("SettingsManager: Loading DataFrameProcessor state...")
        dfproc.select_wells_checkboxes.clear()
        dfproc.select_blank_checkboxes.clear()
        dfproc.select_posctrl_checkboxes.clear()

        self._settings.beginGroup("DataFrameProcessor")
        saved_channels = self._settings.value("selected_channels", [])
        saved_data_wells = self._settings.value("data_wells", [])
        saved_blank_wells = self._settings.value("blank_wells", [])
        saved_pos_wells = self._settings.value("pos_ctrl_wells", [])

        # Convert from string to list if needed
        if isinstance(saved_channels, str):
            saved_channels = [saved_channels]
        if isinstance(saved_data_wells, str):
            saved_data_wells = [saved_data_wells]
        if isinstance(saved_blank_wells, str):
            saved_blank_wells = [saved_blank_wells]
        if isinstance(saved_pos_wells, str):
            saved_pos_wells = [saved_pos_wells]

        original_unit = self._settings.value("original_unit", "Minutes")
        display_unit = self._settings.value("display_unit", "Minutes")
        scatter_state = self._settings.value("scatter_state", False, type=bool)
        black_white_state = self._settings.value("black_white_state", True, type=bool)
        grid_state = self._settings.value("grid_state", False, type=bool)

        self._settings.endGroup()

        # Re-apply channels
        for ch in saved_channels:
            if ch in dfproc.selected_channels_checkboxes:
                dfproc.selected_channels_checkboxes[ch].setChecked(True)
            else:
                self.logger.warning("Saved channel '%s' missing in current data", ch)

        # Re-apply well selections, skipping missing ones
        missing_data_wells = []
        missing_blank_wells = []
        missing_pos_wells = []
        for w in saved_data_wells:
            if w in dfproc.well_buttons:
                dfproc.select_wells_checkboxes[w].setChecked(True)
            else:
                missing_data_wells.append(w)

        for w in saved_blank_wells:
            if w in dfproc.columns:
                dfproc.select_blank_checkboxes[w] = True
            else:
                missing_blank_wells.append(w)

        for w in saved_pos_wells:
            if w in dfproc.columns:
                dfproc.select_posctrl_checkboxes[w] = True
            else:
                missing_pos_wells.append(w)

        # Warn user about missing wells
        all_missing_msgs = []
        if missing_data_wells:
            all_missing_msgs.append("Data wells: " + ", ".join(missing_data_wells))
        if missing_blank_wells:
            all_missing_msgs.append("Blank wells: " + ", ".join(missing_blank_wells))
        if missing_pos_wells:
            all_missing_msgs.append("Positive control wells: " + ", ".join(missing_pos_wells))

        if all_missing_msgs:
            from PyQt5.QtWidgets import QMessageBox
            msg_text = (
                "Some saved wells from the last session do not exist in the current dataset.\n"
                + "\n".join(all_missing_msgs) + "\nThey were removed from your selections."
            )
            self.logger.warning(msg_text)
            QMessageBox.warning(None, "Missing wells", msg_text)

        # Re-apply combos and states
        if original_unit in ["Minutes", "Seconds"]:
            dfproc.original_unit_combo.setCurrentText(original_unit)
        else:
            self.logger.warning("Unrecognised original_unit '%s'; defaulting to 'Minutes'", original_unit)
            dfproc.original_unit_combo.setCurrentText("Minutes")

        if display_unit in ["Seconds", "Minutes", "Hours", "Days", "Weeks"]:
            dfproc.display_unit_combo.setCurrentText(display_unit)
        else:
            self.logger.warning("Unrecognised display_unit '%s'; defaulting to 'Minutes'", display_unit)
            dfproc.display_unit_combo.setCurrentText("Minutes")

        dfproc.scatter_state = scatter_state
        dfproc.black_white_state = black_white_state
        dfproc.grid_state = grid_state

        self.logger.info("SettingsManager: DataFrameProcessor state loaded successfully.")
