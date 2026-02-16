# -*- coding: utf-8 -*-
"""
Clearissa - Channel Configuration Manager
==========================================

Provides the user interface and persistence layer for FRET channel definitions
used throughout the Clearissa application for automatic channel detection and
data conversion workflows.

Module Overview
---------------
This module manages channel configuration sets that define the excitation/emission
wavelength pairs for common FRET experimental setups. These configurations are
used by:

1. **Channel Detection** (core.convert_data_tab.channel_detector)
   - Automatically identifies donor, acceptor, and FRET channels from well labels
   - Matches experimental data columns against configured channel patterns

2. **Data Conversion** (core.convert_data_tab.conversions)
   - Selects appropriate channels for TMSD, HMSD, GATE, and mass-action modes
   - Enables multi-channel FRET analysis with automatic fallback logic

3. **Data Processing** (core.data_frame_processor)
   - Initializes ConvertDataTab with channel definitions
   - Provides channel mapping for visualization and export

Channel Set Structure
---------------------
Each channel set defines 6 FRET channel configurations:

- **Donor 1**: Primary donor channel (excitation-ex/emission-em)
- **Acceptor 1**: Primary acceptor channel (direct excitation)
- **FRET 1**: Primary FRET channel (donor excitation → acceptor emission)
- **Donor 2**: Secondary donor channel (for dual-FRET experiments)
- **Acceptor 2**: Secondary acceptor channel
- **FRET 2**: Secondary FRET channel

Format: "excitation_wavelength-bandwidth/emission_wavelength-bandwidth"
Example: "488-14/535-30" means excite at 488±14nm, detect at 535±30nm

Integration Points
------------------
**Saved to**: config/channel_settings.json
**Loaded by**:
  - ConvertDataTab.__init__() → passes to ChannelDetector
  - DataFrameProcessor.create_convert_tab() → extracts active_channels

**Signal**: channelsUpdated(dict) - emitted when user applies changes
  - Currently unused but available for real-time updates

Usage Example
-------------
>>> from core.common.options import OptionsPanel
>>> options = OptionsPanel()
>>> options.apply_changes()  # Save and activate current set
>>> channels = options.active_channels
>>> print(channels["Donor 1"])
'488-14/535-30'

Author: Križan Jurinović
Date: January 2025
Last Updated: January 2025
"""

import copy
import os
import sys
import json
import logging
import re

# Add resource_utils import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resource_utils import get_data_path
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QLineEdit,
    QPushButton, QFrame, QGroupBox, QMessageBox
)

logger = logging.getLogger(__name__)


# ==================== INDUSTRIAL THEME CONSTANTS ====================

CARD_BACKGROUND = "#F4F5F7"
CARD_BORDER_COLOUR = "#9DA3AE"
MAIN_BACKGROUND = "#DEE1E6"
CONTENT_BACKGROUND = "#E9ECF1"
SECTION_BACKGROUND = "#D6DAE0"
TEXT_PRIMARY = "#111827"
TEXT_SECONDARY = "#374151"
TEXT_TERTIARY = "#6B7280"
ACCENT_BLUE = "#0F6CBD"
ACCENT_GREEN = "#17813E"
ACCENT_ORANGE = "#C95A00"
ACCENT_RED = "#B91C1C"

# Wavelength color mapping for visual feedback
WAVELENGTH_COLORS = {
    'blue': '#2563EB',      # 450-495 nm
    'cyan': '#06B6D4',      # 495-520 nm
    'green': '#16A34A',     # 520-565 nm
    'yellow': '#CA8A04',    # 565-590 nm
    'orange': '#EA580C',    # 590-625 nm
    'red': '#DC2626',       # 625-700 nm
    'far_red': '#991B1B',   # 700+ nm
}

# ==================== HELPER FUNCTIONS ====================

def get_wavelength_color(wavelength_nm):
    """
    Map wavelength to visual color for UI indication.

    Parameters
    ----------
    wavelength_nm : int or str
        Wavelength in nanometers

    Returns
    -------
    str
        Hex color code representing the wavelength range
    """
    try:
        wl = int(wavelength_nm)
        if wl < 495:
            return WAVELENGTH_COLORS['blue']
        elif wl < 520:
            return WAVELENGTH_COLORS['cyan']
        elif wl < 565:
            return WAVELENGTH_COLORS['green']
        elif wl < 590:
            return WAVELENGTH_COLORS['yellow']
        elif wl < 625:
            return WAVELENGTH_COLORS['orange']
        elif wl < 700:
            return WAVELENGTH_COLORS['red']
        else:
            return WAVELENGTH_COLORS['far_red']
    except (ValueError, TypeError):
        return TEXT_TERTIARY


def extract_wavelength(channel_string):
    """
    Extract primary wavelength from channel definition string.

    Parameters
    ----------
    channel_string : str
        Format: "excitation-bandwidth/emission-bandwidth"
        Example: "488-14/535-30"

    Returns
    -------
    tuple of (int, int) or (None, None)
        (excitation_wavelength, emission_wavelength) or (None, None) if parsing fails
    """
    try:
        pattern = r'(\d+)-\d+/(\d+)-\d+'
        match = re.match(pattern, channel_string.strip())
        if match:
            return int(match.group(1)), int(match.group(2))
    except (ValueError, AttributeError):
        pass
    return None, None


# ==================== INDUSTRIAL STYLED WIDGETS ====================

class IndustrialCard(QFrame):
    """
    Industrial-themed container with hard corners and technical styling.
    Matches the convert data tab aesthetic.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {CARD_BACKGROUND};
                border: 1px solid {CARD_BORDER_COLOUR};
                border-radius: 0px;
            }}
        """)
        self.setFrameShape(QFrame.StyledPanel)


class ChannelInputRow(QWidget):
    """
    Specialized widget for channel configuration input with wavelength color indication.
    """

    valueChanged = pyqtSignal(str, str)  # (parameter_name, new_value)

    def __init__(self, parameter_name, initial_value, parent=None):
        super().__init__(parent)
        self.parameter_name = parameter_name

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Channel type indicator with icon
        self.indicator = QLabel()
        self.indicator.setFixedSize(24, 24)
        self.indicator.setAlignment(Qt.AlignCenter)
        self.indicator.setStyleSheet(f"""
            QLabel {{
                background-color: {SECTION_BACKGROUND};
                border: 1px solid {CARD_BORDER_COLOUR};
                border-radius: 0px;
                font-weight: bold;
                font-size: 9pt;
                color: {TEXT_PRIMARY};
            }}
        """)

        # Set indicator symbol based on channel type
        if "Donor" in parameter_name:
            self.indicator.setText("D")
            self.indicator.setToolTip("Donor Channel - Direct excitation of donor fluorophore")
        elif "Acceptor" in parameter_name:
            self.indicator.setText("A")
            self.indicator.setToolTip("Acceptor Channel - Direct excitation of acceptor fluorophore")
        elif "FRET" in parameter_name:
            self.indicator.setText("F")
            self.indicator.setToolTip("FRET Channel - Donor excitation → Acceptor emission")

        # Input field with industrial styling
        self.input_field = QLineEdit(initial_value)
        self.input_field.setFixedHeight(26)
        self.input_field.setStyleSheet(f"""
            QLineEdit {{
                background-color: {CARD_BACKGROUND};
                border: 1px solid {CARD_BORDER_COLOUR};
                border-radius: 0px;
                padding: 3px 6px;
                font-size: 8pt;
                color: {TEXT_PRIMARY};
                font-family: 'Consolas', 'Courier New', monospace;
            }}
            QLineEdit:focus {{
                border-color: {ACCENT_BLUE};
            }}
            QLineEdit:hover {{
                border-color: {TEXT_SECONDARY};
            }}
        """)
        self.input_field.textChanged.connect(self._on_text_changed)
        self.input_field.setPlaceholderText("ex-bw/em-bw")

        # Wavelength color indicator (excitation)
        self.ex_color_indicator = QLabel()
        self.ex_color_indicator.setFixedSize(20, 20)
        self.ex_color_indicator.setToolTip("Excitation wavelength")
        self.ex_color_indicator.setStyleSheet(f"""
            QLabel {{
                border: 1px solid {CARD_BORDER_COLOUR};
                border-radius: 0px;
            }}
        """)

        # Wavelength color indicator (emission)
        self.em_color_indicator = QLabel()
        self.em_color_indicator.setFixedSize(20, 20)
        self.em_color_indicator.setToolTip("Emission wavelength")
        self.em_color_indicator.setStyleSheet(f"""
            QLabel {{
                border: 1px solid {CARD_BORDER_COLOUR};
                border-radius: 0px;
            }}
        """)

        layout.addWidget(self.indicator)
        layout.addWidget(self.input_field, 1)
        layout.addWidget(QLabel("Ex:"))
        layout.addWidget(self.ex_color_indicator)
        layout.addWidget(QLabel("Em:"))
        layout.addWidget(self.em_color_indicator)

        # Initialize color indicators
        self._update_color_indicators(initial_value)

    def _on_text_changed(self, text):
        """Handle text changes and update color indicators."""
        self._update_color_indicators(text)
        self.valueChanged.emit(self.parameter_name, text)

    def _update_color_indicators(self, channel_string):
        """Update wavelength color indicators based on channel definition."""
        ex_wl, em_wl = extract_wavelength(channel_string)

        if ex_wl:
            ex_color = get_wavelength_color(ex_wl)
            self.ex_color_indicator.setStyleSheet(f"""
                QLabel {{
                    background-color: {ex_color};
                    border: 1px solid {CARD_BORDER_COLOUR};
                    border-radius: 0px;
                }}
            """)
            self.ex_color_indicator.setToolTip(f"Excitation: {ex_wl} nm")
        else:
            self.ex_color_indicator.setStyleSheet(f"""
                QLabel {{
                    background-color: {SECTION_BACKGROUND};
                    border: 1px solid {CARD_BORDER_COLOUR};
                    border-radius: 0px;
                }}
            """)
            self.ex_color_indicator.setToolTip("Invalid format")

        if em_wl:
            em_color = get_wavelength_color(em_wl)
            self.em_color_indicator.setStyleSheet(f"""
                QLabel {{
                    background-color: {em_color};
                    border: 1px solid {CARD_BORDER_COLOUR};
                    border-radius: 0px;
                }}
            """)
            self.em_color_indicator.setToolTip(f"Emission: {em_wl} nm")
        else:
            self.em_color_indicator.setStyleSheet(f"""
                QLabel {{
                    background-color: {SECTION_BACKGROUND};
                    border: 1px solid {CARD_BORDER_COLOUR};
                    border-radius: 0px;
                }}
            """)
            self.em_color_indicator.setToolTip("Invalid format")

    def set_value(self, value):
        """Set the input field value programmatically."""
        self.input_field.setText(value)

    def get_value(self):
        """Get the current input field value."""
        return self.input_field.text()


# ==================== MAIN OPTIONS PANEL ====================
class OptionsPanel(QWidget):
    """
    Configuration interface for FRET channel definitions.

    Manages multiple channel configuration sets stored in
    config/channel_settings.json. The active set is used for automatic
    channel detection and data conversion.

    Signals
    -------
    channelsUpdated : pyqtSignal(dict)
        Emitted when user applies changes.
    """

    channelsUpdated = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)

        # File and default sets - use proper user data directory
        self.settings_file = get_data_path("config/channel_settings.json")
        self.default_channel_sets = {
            "Set 1": {
                "Donor 1": "488-14/535-30",
                "Acceptor 1": "540-20/590-30",
                "FRET 1": "488-14/583-30",
                "Donor 2": "540-20/590-30",
                "Acceptor 2": "625-30/680-30",
                "FRET 2": "546-40/690-40",
            },
            "Set 2": {
                "Donor 1": "500-20/550-30",
                "Acceptor 1": "560-20/600-30",
                "FRET 1": "500-20/600-30",
                "Donor 2": "610-30/700-40",
                "Acceptor 2": "720-40/800-50",
                "FRET 2": "610-30/800-50",
            },
        }

        # Load from file or use defaults
        loaded = self._load_channel_settings()
        self.channel_sets = loaded.get("channels", copy.deepcopy(self.default_channel_sets))
        self.current_channel_set = list(self.channel_sets.keys())[0]
        self.active_channels = self.channel_sets[self.current_channel_set]
        self.parameter_fields = {}

        # Build the UI
        self._build_ui()

    # -----------------------------------------------------------------------
    # UI Construction
    # -----------------------------------------------------------------------
    def _build_ui(self):
        """Construct the channel configuration layout."""
        # Main background
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QColor(MAIN_BACKGROUND))
        self.setPalette(palette)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # --- Header Section ---
        header_layout = QHBoxLayout()
        header_layout.setSpacing(10)

        # Settings icon
        icon_label = QLabel("⚙")
        icon_label.setStyleSheet(f"""
            QLabel {{
                font-size: 22pt;
                color: {ACCENT_BLUE};
                background-color: transparent;
            }}
        """)

        header_text_layout = QVBoxLayout()
        header_text_layout.setSpacing(2)

        header = QLabel("Channel Configuration")
        header.setStyleSheet(f"""
            QLabel {{
                font-size: 14pt;
                font-weight: bold;
                color: {TEXT_PRIMARY};
                background-color: transparent;
            }}
        """)

        desc = QLabel("Define FRET channel wavelengths for automatic detection and data conversion")
        desc.setStyleSheet(f"""
            QLabel {{
                color: {TEXT_SECONDARY};
                font-size: 8pt;
                background-color: transparent;
            }}
        """)
        desc.setWordWrap(True)

        header_text_layout.addWidget(header)
        header_text_layout.addWidget(desc)

        header_layout.addWidget(icon_label)
        header_layout.addLayout(header_text_layout, 1)
        layout.addLayout(header_layout)

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Plain)
        separator.setStyleSheet(f"background-color: {CARD_BORDER_COLOUR}; max-height: 1px;")
        layout.addWidget(separator)

        # --- Channel Set Selection Card ---
        selector_card = IndustrialCard()
        selector_layout = QVBoxLayout(selector_card)
        selector_layout.setContentsMargins(15, 15, 15, 15)
        selector_layout.setSpacing(10)

        selector_header = QLabel("Active Channel Set")
        selector_header.setStyleSheet(f"""
            QLabel {{
                font-weight: 600;
                font-size: 9pt;
                color: {TEXT_SECONDARY};
                background-color: transparent;
            }}
        """)

        selector_row = QHBoxLayout()
        selector_row.setSpacing(10)

        self.set_selector = QComboBox()
        self.set_selector.addItems(self.channel_sets.keys())
        self.set_selector.setFixedHeight(28)
        self.set_selector.setStyleSheet(f"""
            QComboBox {{
                background-color: {CARD_BACKGROUND};
                border: 1px solid {CARD_BORDER_COLOUR};
                border-radius: 0px;
                padding: 4px 8px;
                font-size: 8pt;
                color: {TEXT_PRIMARY};
            }}
            QComboBox:focus {{
                border-color: {ACCENT_BLUE};
            }}
            QComboBox:hover {{
                border-color: {TEXT_SECONDARY};
            }}
            QComboBox::drop-down {{
                border: none;
                width: 20px;
            }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid {TEXT_SECONDARY};
                width: 0px;
                height: 0px;
                margin-right: 6px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {CARD_BACKGROUND};
                border: 1px solid {ACCENT_BLUE};
                border-radius: 0px;
                selection-background-color: {ACCENT_BLUE};
                selection-color: white;
                padding: 4px;
            }}
        """)
        self.set_selector.currentTextChanged.connect(self.update_ui_for_selected_set)

        selector_row.addWidget(self.set_selector, 1)

        selector_layout.addWidget(selector_header)
        selector_layout.addLayout(selector_row)
        layout.addWidget(selector_card)

        # --- Channel Parameters Card ---
        params_card = IndustrialCard()
        params_layout = QVBoxLayout(params_card)
        params_layout.setContentsMargins(15, 15, 15, 15)
        params_layout.setSpacing(12)

        params_header = QLabel("FRET Channel Definitions")
        params_header.setStyleSheet(f"""
            QLabel {{
                font-weight: 600;
                font-size: 9pt;
                color: {TEXT_SECONDARY};
                background-color: transparent;
            }}
        """)
        params_layout.addWidget(params_header)

        # Info panel
        info_panel = QLabel(
            "Format: excitation-bandwidth/emission-bandwidth (e.g., 488-14/535-30)\n"
            "Each tag must appear as a substring of the channel labels in your data."
        )
        info_panel.setStyleSheet(f"""
            QLabel {{
                background-color: #E4EFFA;
                border: 1px solid #6D9CCF;
                border-radius: 0px;
                padding: 6px;
                font-size: 7pt;
                color: #1D4C7F;
            }}
        """)
        params_layout.addWidget(info_panel)

        # Primary FRET channels group
        primary_group = QGroupBox("Primary FRET Configuration")
        primary_group.setStyleSheet(f"""
            QGroupBox {{
                background-color: transparent;
                border: 1px solid {CARD_BORDER_COLOUR};
                border-radius: 0px;
                margin-top: 8px;
                padding: 8px 6px 6px 6px;
                font-weight: 600;
                font-size: 8pt;
                color: {TEXT_SECONDARY};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 8px;
                padding: 0 4px 0 4px;
                background-color: {CARD_BACKGROUND};
            }}
        """)
        primary_layout = QVBoxLayout(primary_group)
        primary_layout.setSpacing(8)

        # Secondary FRET channels group
        secondary_group = QGroupBox("Secondary FRET Configuration")
        secondary_group.setStyleSheet(f"""
            QGroupBox {{
                background-color: transparent;
                border: 1px solid {CARD_BORDER_COLOUR};
                border-radius: 0px;
                margin-top: 8px;
                padding: 8px 6px 6px 6px;
                font-weight: 600;
                font-size: 8pt;
                color: {TEXT_SECONDARY};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 8px;
                padding: 0 4px 0 4px;
                background-color: {CARD_BACKGROUND};
            }}
        """)
        secondary_layout = QVBoxLayout(secondary_group)
        secondary_layout.setSpacing(8)

        # Create channel input rows
        for param, value in self.channel_sets[self.current_channel_set].items():
            # Create label
            label = QLabel(param + ":")
            label.setStyleSheet(f"""
                QLabel {{
                    font-size: 8pt;
                    font-weight: 600;
                    color: {TEXT_SECONDARY};
                    background-color: transparent;
                }}
            """)

            # Create input row
            input_row = ChannelInputRow(param, value)
            input_row.valueChanged.connect(self.update_parameter)
            self.parameter_fields[param] = input_row

            # Add to appropriate group
            row_layout = QHBoxLayout()
            row_layout.setSpacing(10)
            row_layout.addWidget(label)
            row_layout.addWidget(input_row, 1)

            if "1" in param:
                primary_layout.addLayout(row_layout)
            else:
                secondary_layout.addLayout(row_layout)

        params_layout.addWidget(primary_group)
        params_layout.addWidget(secondary_group)
        layout.addWidget(params_card)

        # --- Action Buttons ---
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)

        save_btn = QPushButton("✓ Save and Apply")
        save_btn.setFixedHeight(32)
        save_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {ACCENT_GREEN};
                color: white;
                border: none;
                border-radius: 0px;
                padding: 8px 16px;
                font-weight: 600;
                font-size: 9pt;
            }}
            QPushButton:hover {{
                background-color: #0f5c27;
            }}
            QPushButton:pressed {{
                background-color: #0d4a20;
            }}
        """)
        save_btn.clicked.connect(self.apply_changes)

        reset_btn = QPushButton("↺ Reset to Defaults")
        reset_btn.setFixedHeight(32)
        reset_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {CARD_BACKGROUND};
                color: {TEXT_SECONDARY};
                border: 1px solid {CARD_BORDER_COLOUR};
                border-radius: 0px;
                padding: 8px 16px;
                font-weight: 500;
                font-size: 9pt;
            }}
            QPushButton:hover {{
                background-color: {CONTENT_BACKGROUND};
                border-color: {ACCENT_BLUE};
            }}
            QPushButton:pressed {{
                background-color: {SECTION_BACKGROUND};
            }}
        """)
        reset_btn.clicked.connect(self.reset_all_sets)

        button_layout.addStretch()
        button_layout.addWidget(save_btn)
        button_layout.addWidget(reset_btn)
        layout.addLayout(button_layout)

        layout.addStretch()

    # -----------------------------------------------------------------------
    # Functional Logic
    # -----------------------------------------------------------------------
    def update_ui_for_selected_set(self, selected_set):
        """
        Refresh visible parameters when changing channel set.

        Updates all parameter input fields to display values from
        the newly selected channel set.

        Parameters
        ----------
        selected_set : str
            Name of the channel set to display

        Notes
        -----
        This only updates the UI - changes are not saved until
        the user clicks "Save and Apply".
        """
        self.current_channel_set = selected_set
        for key, input_row in self.parameter_fields.items():
            input_row.input_field.blockSignals(True)
            input_row.set_value(self.channel_sets[self.current_channel_set][key])
            input_row.input_field.blockSignals(False)

    def update_parameter(self, key, value):
        """
        Update in-memory setting for a specific parameter.

        Parameters
        ----------
        key : str
            Parameter name (e.g., "Donor 1", "FRET 1")
        value : str
            Channel definition string (e.g., "488-14/535-30")

        Notes
        -----
        Changes are stored in memory but not persisted until apply_changes()
        is called. This allows users to modify multiple parameters before
        committing changes.
        """
        self.channel_sets[self.current_channel_set][key] = value

    def _load_channel_settings(self):
        """
        Load channel settings from JSON file or use defaults.

        Returns
        -------
        dict
            Dictionary with structure: {"channels": {set_name: {param: value}}}

        Notes
        -----
        - If file doesn't exist, returns default channel sets
        - If file is corrupted, logs warning and returns defaults
        - File location: config/channel_settings.json (in user data directory)
        """
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.info("Loaded channel settings from file.")
                return data
            except (json.JSONDecodeError, IOError):
                logger.warning("Error reading channel settings file; using defaults.")
        return {"channels": copy.deepcopy(self.default_channel_sets)}

    def _save_channel_settings(self):
        """
        Write current channel settings to JSON file.

        Creates the config directory if it doesn't exist. Logs errors
        if file cannot be written.

        File Structure
        --------------
        {
          "channels": {
            "Set 1": {
              "Donor 1": "488-14/535-30",
              ...
            },
            ...
          }
        }

        Notes
        -----
        This method is called by apply_changes() and reset_all_sets().
        It persists the current state of all channel sets, not just
        the active one.
        """
        os.makedirs(os.path.dirname(self.settings_file), exist_ok=True)
        data = {"channels": self.channel_sets}
        try:
            with open(self.settings_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.info("Saved channel settings to disk.")
        except IOError as e:
            logger.error(f"Failed to save channel settings: {e}")

    def apply_changes(self):
        """
        Persist current channel settings and activate them.

        This method:
        1. Updates the active_channels attribute to the current set
        2. Saves all channel sets to disk (config/channel_settings.json)
        3. Logs the applied configuration
        4. Emits channelsUpdated signal with active channel dictionary

        Notes
        -----
        This is the only method that actually commits changes to disk.
        All UI modifications are temporary until this is called.

        The emitted signal can be connected to update other components
        in real-time, though currently this is not utilised. Channel
        settings are read fresh when ConvertDataTab is created.

        See Also
        --------
        DataFrameProcessor.create_convert_tab : Reads active_channels
        ChannelDetector.__init__ : Loads from config file
        """
        self.active_channels = self.channel_sets[self.current_channel_set]
        self._save_channel_settings()
        logger.info("Applied Channel Settings:")
        for k, v in self.active_channels.items():
            logger.info(f"  {k}: {v}")
        self.channelsUpdated.emit(self.active_channels)

        # Confirm to the user
        lines = [f"  {k}:  {v}" for k, v in self.active_channels.items()]
        QMessageBox.information(
            self,
            "Channel Settings Updated",
            "Channel configuration saved and applied.\n\n"
            + "\n".join(lines)
            + "\n\nThe Convert Data tab will use these settings "
            "on the next conversion."
        )

    def reload_from_file(self):
        """
        Reload channel sets from disk, preserving UI consistency.

        Called after external code (e.g. auto-save in ConvertDataTab) writes
        directly to the settings file. Merges any new sets into the in-memory
        state without discarding unsaved UI edits for the current set.
        """
        loaded = self._load_channel_settings()
        file_sets = loaded.get("channels", {})
        for set_name, params in file_sets.items():
            if set_name not in self.channel_sets:
                self.channel_sets[set_name] = params
                logger.info("Loaded new channel set from file: %s", set_name)
        # Refresh the set selector if present
        if hasattr(self, 'set_selector'):
            current_text = self.set_selector.currentText()
            self.set_selector.blockSignals(True)
            self.set_selector.clear()
            self.set_selector.addItems(list(self.channel_sets.keys()))
            idx = self.set_selector.findText(current_text)
            if idx >= 0:
                self.set_selector.setCurrentIndex(idx)
            self.set_selector.blockSignals(False)

    def reset_all_sets(self):
        """
        Revert all channel sets to factory defaults.

        This method:
        1. Replaces all channel sets with default configurations
        2. Updates the UI to show the current set's default values
        3. Saves the defaults to disk immediately

        Notes
        -----
        This operation is immediate and cannot be undone. All custom
        channel configurations will be lost.

        Default sets are based on common fluorophore pairs:
        - Set 1: GFP/RFP-like (488/540nm excitation)
        - Set 2: YFP/mCherry-like (500/610nm excitation)

        Users will need to click "Save and Apply" after this to
        activate the reset defaults (they are saved but not active).
        """
        self.channel_sets = copy.deepcopy(self.default_channel_sets)
        self.update_ui_for_selected_set(self.current_channel_set)
        self._save_channel_settings()
        logger.info("Reset all channel sets to default values.")

    # -----------------------------------------------------------------------
    # Utility Methods
    # -----------------------------------------------------------------------
    def get_active_channel(self, channel_name):
        """
        Get a specific channel definition from active set.

        Parameters
        ----------
        channel_name : str
            Name of channel (e.g., "Donor 1", "FRET 1")

        Returns
        -------
        str or None
            Channel definition string (e.g., "488-14/535-30") or None if not found

        Examples
        --------
        >>> options.get_active_channel("Donor 1")
        '488-14/535-30'
        """
        return self.active_channels.get(channel_name)

    def validate_channel_format(self, channel_string):
        """
        Validate channel definition format.

        Parameters
        ----------
        channel_string : str
            Channel definition to validate

        Returns
        -------
        bool
            True if format is valid, False otherwise

        Notes
        -----
        Expected format: "excitation-bandwidth/emission-bandwidth"
        Example: "488-14/535-30"

        This is a basic validation - it doesn't check if wavelengths
        are physically reasonable.
        """
        # Pattern: number-number/number-number
        pattern = r'^\d+-\d+/\d+-\d+$'
        return bool(re.match(pattern, channel_string.strip()))

    def get_all_channel_names(self):
        """
        Get list of all channel parameter names.

        Returns
        -------
        list of str
            Channel parameter names: ["Donor 1", "Acceptor 1", "FRET 1", ...]
        """
        if self.active_channels:
            return list(self.active_channels.keys())
        return ["Donor 1", "Acceptor 1", "FRET 1", "Donor 2", "Acceptor 2", "FRET 2"]

    def get_channel_set_names(self):
        """
        Get list of all available channel set names.

        Returns
        -------
        list of str
            Names of all configured channel sets
        """
        return list(self.channel_sets.keys())

