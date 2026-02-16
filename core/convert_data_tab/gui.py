"""
Convert Data Tab - PyQt5 GUI Component
======================================

Data Structure Specification for Input Tables
----------------------------------------------

Each input dataset (e.g. `selected_data`, `pos_ctrl_data`, `blank_ctrl_data`)
is a Pandas DataFrame in long format representing fluorescence time series
measurements across multiple channels and wells.

Canonical column layout:
    Well          str   - Channel identifier (e.g. 'D488', 'D583')
    Time [min]    float - Timepoint in minutes
    A01, A02, ... float - Data columns representing fluorescence per well

Each 'Well' group defines one channel. Within each group, the time column 
defines the x-axis, and remaining numeric columns define independent traces.
"""

from __future__ import annotations

from pathlib import Path
import logging
import json
import os
import sys
import re
from typing import Any, Dict, Optional, List
from datetime import datetime
import pandas as pd
import numpy as np
import pyqtgraph as pg

logger = logging.getLogger(__name__)

# Add resource_utils import for proper path handling in frozen executables
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from resource_utils import get_data_path

from PyQt5.QtCore import Qt, QEvent, QObject, QSettings, pyqtSignal, QTimer
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QLineEdit,
    QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QMessageBox,
    QFileDialog, QComboBox, QTextEdit, QToolTip,
    QStackedWidget, QSizePolicy, QCompleter,
    QTableWidget, QHeaderView, QTableWidgetItem, QAbstractScrollArea,
    QTabWidget, QFrame, QDialog
)

from .species import SpeciesManager
from .params import parse_params, ConvertParams
from . import calibration, io_utils
from .conversion_TMSD import convert_tmsd_to_conc
from .conversion_Internal_TMSD import convert_internal_tmsd_to_conc
from .conversion_normalised_sd import convert_normalised_sd
from .conversion_HMSD_FRET import convert_fret_onestep_to_conc
from .conversion_catalytic_FRET import convert_catalytic_fret_2x2
from .conversion_TMSD_HP_Quenching import convert_reverted_readout_to_conc
from .plothelper import PlotDashboard
from .channel_detector import ChannelDetector
from .widget_factory import WidgetFactory
from .state_manager import StateManager
from .matrix_renderer import MatrixRenderer
from core.common.ui_theme import Colors, UITheme
from .ui_components import ComboBoxFactory
from .ui_components import create_info_button, create_section_separator
from .sidebar_builder import SidebarBuilder


MODE_WIDGET_RULES = {
    "tmsd": {
        "neg_ctrl": False,
        "inj1": True,
        "inj2": False,
        "b1l": False,
        "species_box": False,
    },
    "internal_tmsd": {
        "neg_ctrl": True,
        "inj1": True,
        "inj2": False,
        "b1l": False,
        "species_box": False,
    },
    "normalised_sd": {
        "neg_ctrl": True,
        "inj1": False,
        "inj2": False,
        "b1l": False,
        "species_box": False,
    },
    "hmsd": {"neg_ctrl": False, "inj1": False, "inj2": False, "b1l": False, "species_box": False},
    "mass": {"neg_ctrl": False, "inj1": False, "inj2": False, "b1l": True, "species_box": True},
    "hp_quenching": {"neg_ctrl": True, "inj1": False, "inj2": False, "b1l": False, "species_box": False},
}


class ModeInfoDialog(QDialog):
    """Dialog window to display conversion mode information and documentation."""

    def __init__(self, mode: str, parent=None):
        super().__init__(parent)
        self.mode = mode
        self._setup_ui()

    def _setup_ui(self):
        """Build the dialog UI."""
        from .ui_components import ModeInfoProvider

        mode_names = {
            "tmsd": "Basic TMSD",
            "internal_tmsd": "Internal Toehold TMSD",
            "normalised_sd": "Normalised SD",
            "hmsd": "One-step HMSD",
            "mass": "Mass-action Catalytic"
        }

        self.setWindowTitle(f"Method Information: {mode_names.get(self.mode, self.mode.upper())}")
        self.setMinimumWidth(800)
        self.setMinimumHeight(600)
        self.resize(900, 700)  # Default size

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Title
        title = QLabel(mode_names.get(self.mode, self.mode.upper()))
        title.setStyleSheet(f"font-size: 14pt; font-weight: bold; color: {Colors.TEXT_PRIMARY};")
        layout.addWidget(title)

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Plain)
        separator.setStyleSheet(f"background-color: {Colors.CARD_BORDER}; max-height: 2px;")
        layout.addWidget(separator)

        # Info text - using monospace font for better formatting of equations/code
        info_text = QTextEdit()
        info_text.setReadOnly(True)

        # Get the mode info text
        mode_text = ModeInfoProvider.get_mode_info_text(self.mode)
        logger.debug(f"ModeInfoDialog for mode '{self.mode}': retrieved {len(mode_text)} characters")

        info_text.setPlainText(mode_text)
        info_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {Colors.CARD_BACKGROUND};
                border: 1px solid {Colors.CARD_BORDER};
                border-radius: 4px;
                padding: 12px;
                font-size: 9pt;
                font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                color: {Colors.TEXT_PRIMARY};
                line-height: 1.4;
            }}
        """)
        layout.addWidget(info_text)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colors.ACCENT_BLUE};
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 24px;
                font-size: 10pt;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: #0c5591;
            }}
        """)
        close_btn.clicked.connect(self.accept)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)


class SpeciesSelectionManager:
    """
    Centralized manager for species selection across all conversion modes.
    Maps GUI combo boxes to backend calibration records consistently.
    """

    def __init__(self, parent):
        self.parent = parent
        self.logger = logging.getLogger(__name__)

        self.mode_mappings = {
            "tmsd": {
                # No calibration species required - beta calculated empirically from experimental data
            },
            "internal_tmsd": {
                # No calibration species required - slope calculated from positive control wells
            },
            "normalised_sd": {
                # No calibration species required
            },
            "hmsd": {
                # No calibration species required - beta calculated from donor and product controls
            },
            "mass": {
                ("Donor", "B1B2"): "B1B2-Donor",
                ("Donor", "B2"): "B2-Donor",
                ("Donor", "B1AA"): "B1AA-Donor",
                ("Donor", "B1L"): "B1L-Donor",
                ("FRET", "B1B2"): "B1B2-FRET",
                ("FRET", "B2"): "B2-FRET",
                ("FRET", "B1AA"): "B1AA-FRET",
                ("FRET", "B1L"): "B1L-FRET",
            }
        }

    def get_combo_widget(self, mode: str, key) -> Optional[QComboBox]:
        """Retrieve combo widget for a given mode and key."""
        if mode == "mass":
            return self.parent.mass_combos.get(key)
        return None

    def collect_species_map(self, mode: str) -> Dict[str, Dict[str, Any]]:
        """
        Collect all species selections for the given mode and return
        a dictionary mapping backend keys to calibration records.
        """
        species_map = {}

        if mode not in self.mode_mappings:
            self.logger.warning(f"Unknown mode '{mode}'")
            return species_map

        mappings = self.mode_mappings[mode]

        for gui_key, backend_key in mappings.items():
            combo = self.get_combo_widget(mode, gui_key)
            if combo is None:
                continue

            selected_species = combo.currentText().strip()
            if not selected_species or selected_species == "NONE":
                continue

            try:
                cal_data = self.parent.get_species_calibration(selected_species)
                if cal_data and cal_data.get("slope") is not None:
                    species_map[backend_key] = cal_data
                    self.logger.info(
                        f"Species mapping: {backend_key} --- '{selected_species}' "
                        f"(slope={cal_data.get('slope'):.4f})"
                    )
                else:
                    self.logger.warning(
                        f"No calibration for '{selected_species}' (backend_key={backend_key})"
                    )
            except Exception as e:
                self.logger.error(f"Failed to retrieve calibration for '{selected_species}': {e}")

        return species_map

    def validate_required_species(self, mode: str, species_map: Dict) -> List[str]:
        """Check if all required species for the mode are present."""
        required = set(self.mode_mappings.get(mode, {}).values())
        present = set(species_map.keys())
        missing = required - present

        if missing:
            self.logger.warning(f"Mode '{mode}' missing required species: {sorted(missing)}")

        return sorted(list(missing))

    def get_required_species_list(self, mode: str) -> List[str]:
        """Return list of all required backend keys for a mode."""
        return sorted(list(self.mode_mappings.get(mode, {}).values()))


class ConvertDataTab(QWidget):
    """
    PyQt5 widget for converting raw experimental data into concentration values.
    Supports multiple conversion modes: TMSD, HMSD, and mass-action catalytic.
    """

    # Signal emitted when data is forwarded to Kinetics Processor
    # (dataframe, label, metadata dict with optional per_trace_Z0)
    forwarded_to_kinetics = pyqtSignal(object, str, object)

    def __init__(
        self,
        parent: Optional[QWidget],
        selected_data: Any,
        blank_ctrl_data: Any,
        pos_ctrl_data: Any,
        donor_data: Any = None,
        acceptor_data: Any = None,
        neg_ctrl_data: Any = None,
        blocked_ctrl_data: Any = None,
        detected_inj_timepoints: Any = None,
        donor1_channel: Optional[str] = None,
        acceptor1_channel: Optional[str] = None,
        fret1_channel: Optional[str] = None,
        donor2_channel: Optional[str] = None,
        acceptor2_channel: Optional[str] = None,
        fret2_channel: Optional[str] = None,
        csv_info: Any = None,
        experiment_id: Optional[str] = None,
    ):
        super().__init__(parent)
        self.master = parent

        # Store experiment ID for experiment-specific settings
        self.experiment_id = experiment_id
        if experiment_id:
            logger.info(f"ConvertDataTab initialised with experiment_id: {experiment_id}")
        else:
            logger.info("ConvertDataTab initialised without experiment_id (will use default settings)")

        self.species_manager = SpeciesSelectionManager(self)
        # Initialise state manager and matrix renderer
        self.state_manager = StateManager(self)
        self.matrix_renderer = MatrixRenderer(self)

        # Registry for storing converted datasets
        self.converted_data_registry = {}
        self.parent_tab_widget = None  # Set by main window for tab switching

        self.selected_data = selected_data
        self.blank_ctrl_data = blank_ctrl_data
        self.pos_ctrl_data = pos_ctrl_data
        self.donor_data = donor_data
        self.acceptor_data = acceptor_data
        self.neg_ctrl_data = neg_ctrl_data
        self.blocked_ctrl_data = blocked_ctrl_data
        self.detected_inj_timepoints = detected_inj_timepoints

        self.donor1_channel = donor1_channel
        self.acceptor1_channel = acceptor1_channel
        self.fret1_channel = fret1_channel
        self.donor2_channel = donor2_channel
        self.acceptor2_channel = acceptor2_channel
        self.fret2_channel = fret2_channel

        self.csv_info = csv_info

        self.detected_channels = {}
        self.selected_donor_channel = None
        self.selected_acceptor_channel = None
        self.selected_fret_channel = None
        self._detect_and_log_channels()

        # Use proper path resolution for calibration data
        self.calibration_data_path = Path(get_data_path("config/calibration_data.json"))
        self.calibration_records = []

        self._current_result = None

        self.is_basic_tmsd = False
        self.is_fretconserve = False
        self.is_mass_action = False

        self.start_with_donor = True

        self.init_start_time_entry = QLineEdit()
        self.init_end_time_entry = QLineEdit()
        self.neg_ctrl_start_time_entry = QLineEdit()
        self.neg_ctrl_end_time_entry = QLineEdit()
        self.reaction_trigger_timepoint = QLineEdit()
        self.volume_200ul_timepoint = QLineEdit()
        self.injection_for_nuking_entry = QLineEdit()
        self.injection_for_nuking_entry_2 = QLineEdit()
        self.b1l_initial_conc_entry = QLineEdit()
        self.b1l_initial_conc_entry.setText("10")
        self.b1l_initial_conc_entry.setPlaceholderText("nM")
        self.b1l_initial_conc_entry.setToolTip(
            "Experimental B1-L starting concentration [nM].\n"
            "This is the actual [B1L]_0 in your sample wells.\n"
            "May differ from the B1L control concentration used for calibration."
        )

        self.c_ref_entry = QLineEdit()
        self.c_ref_entry.setText("10")
        self.c_ref_entry.setPlaceholderText("nM")
        self.c_ref_entry.setToolTip("Positive control reference concentration (default: 10 nM)")

        # Calibration mode selector for mass-action mode
        self.calibration_mode_combo = QComboBox()
        self.calibration_mode_combo.addItem("Pre-calibrated (stored slopes)", "precalibrated")
        self.calibration_mode_combo.addItem("Runtime (plate controls)", "runtime")
        self.calibration_mode_combo.addItem("Both (compare methods)", "both")
        self.calibration_mode_combo.setToolTip(
            "Calibration mode:\n"
            "- Pre-calibrated: Use slopes from calibration database\n"
            "- Runtime: Estimate slopes from plate-matched controls\n"
            "- Both: Run both methods and compare results"
        )

        # Apply consistent styling to all input fields
        for entry in [self.init_start_time_entry, self.init_end_time_entry,
                      self.neg_ctrl_start_time_entry, self.neg_ctrl_end_time_entry,
                      self.reaction_trigger_timepoint, self.volume_200ul_timepoint,
                      self.injection_for_nuking_entry, self.injection_for_nuking_entry_2,
                      self.b1l_initial_conc_entry, self.c_ref_entry]:
            entry.setStyleSheet(UITheme.get_lineedit_style(max_width=70))

        self.calibration_mode_combo.setStyleSheet(UITheme.get_combobox_style())

        self.pos_ctrl_concentration_entries = []
        self.neg_ctrl_concentration_entries = []
        self.donor_ctrl_concentration_entry = None
        self.acceptor_ctrl_concentration_entry = None
        self.b1l_ctrl_concentration_entry = None

        self.plot_widget_top: QWidget | None = None
        self.plot_widget_bottom: QWidget | None = None
        self.plot_mode = 'line'  # Default plot mode: 'line' or 'scatter'

        import time
        start_time = time.time()

        self._load_calibration()
        logger.info(f"_load_calibration took {time.time() - start_time:.3f}s")

        start_time = time.time()
        self._setup_ui()
        logger.info(f"_setup_ui took {time.time() - start_time:.3f}s")

        start_time = time.time()
        self._wire_events()
        logger.info(f"_wire_events took {time.time() - start_time:.3f}s")

        start_time = time.time()
        self.check_for_missing_don_acc_data()
        logger.info(f"check_for_missing_don_acc_data took {time.time() - start_time:.3f}s")

        start_time = time.time()
        self.load_state()
        logger.info(f"load_state took {time.time() - start_time:.3f}s")

        # Apply initial visibility rules based on current mode
        initial_mode = self.approach_combo.currentData()
        if initial_mode:
            self._apply_mode_visibility_rules(initial_mode)

    def _datawell_columns(self, df) -> list[str]:
        """Return data-well column names, excluding 'Well' and 'Time [min]'."""
        try:
            cols = list(getattr(df, "columns", []))
            return cols[2:] if len(cols) > 2 else []
        except Exception:
            return []

    def _mode_flags_from_combo(self) -> None:
        """Update mode flags based on current approach selection."""
        mode = self.approach_combo.currentData()
        self.is_basic_tmsd = (mode == "tmsd")
        self.is_fretconserve = (mode == "hmsd")
        self.is_mass_action = (mode == "mass")

    def _apply_mode_visibility_rules(self, mode: str) -> None:
        """Show/hide parameter inputs based on conversion mode requirements."""
        rules = MODE_WIDGET_RULES.get(mode, {})

        # Negative control fields
        need_neg_ctrl = rules.get("neg_ctrl", False)
        if hasattr(self, 'neg_ctrl_window_label'):
            self.neg_ctrl_window_label.setVisible(need_neg_ctrl)
        self.neg_ctrl_start_time_entry.setVisible(need_neg_ctrl)
        self.neg_ctrl_end_time_entry.setVisible(need_neg_ctrl)
        if hasattr(self, 'neg_ctrl_separator'):
            self.neg_ctrl_separator.setVisible(need_neg_ctrl)

        # Negative control concentration entries (only for hp_quenching mode)
        need_neg_ctrl_conc = (mode == "hp_quenching")
        if hasattr(self, 'neg_ctrl_conc_label'):
            self.neg_ctrl_conc_label.setVisible(need_neg_ctrl_conc)
        for entry in self.neg_ctrl_concentration_entries:
            entry.setVisible(need_neg_ctrl_conc)
        for label in getattr(self, 'neg_ctrl_concentration_labels', []):
            label.setVisible(need_neg_ctrl_conc)

        # Injection 1 field (nuking)
        need_inj1 = rules.get("inj1", False)
        self.inj1_label.setVisible(need_inj1)
        self.injection_for_nuking_entry.setVisible(need_inj1)
        if need_inj1:
            if mode in ("tmsd", "internal_tmsd"):
                self.inj1_label.setText("Nuking time")
            else:
                self.inj1_label.setText("Injection time")

        # Injection 2 field
        need_inj2 = rules.get("inj2", False)
        self.inj2_label.setVisible(need_inj2)
        self.injection_for_nuking_entry_2.setVisible(need_inj2)

        # B1-L initial concentration
        need_b1l = rules.get("b1l", False)
        self.b1l_label.setVisible(need_b1l)
        self.b1l_initial_conc_entry.setVisible(need_b1l)

        # Calibration mode controls (only for mass-action mode)
        need_calib_mode = (mode == "mass")
        if hasattr(self, 'calibration_mode_label'):
            self.calibration_mode_label.setVisible(need_calib_mode)
        if hasattr(self, 'calibration_mode_combo'):
            self.calibration_mode_combo.setVisible(need_calib_mode)

        # Species selection box and method configuration header
        need_species = rules.get("species_box", True)
        if hasattr(self, 'species_box'):
            self.species_box.setVisible(need_species)
        # Hide "Method configuration" header when species box is hidden
        if hasattr(self, 'method_config_title_label'):
            self.method_config_title_label.setVisible(need_species)

        # Coefficients and bleed-through tables (only for mass-action mode)
        need_matrix = (mode == "mass")
        if hasattr(self, 'coeff_table'):
            self.coeff_table.setVisible(need_matrix)
        if hasattr(self, 'bleed_table'):
            self.bleed_table.setVisible(need_matrix)
        if hasattr(self, 'coeff_label'):
            self.coeff_label.setVisible(need_matrix)
        if hasattr(self, 'slopes_label'):
            self.slopes_label.setVisible(need_matrix)
        if hasattr(self, 'matrix_header'):
            self.matrix_header.setVisible(need_matrix)

    def _setup_ui(self) -> None:
        """Build the main UI layout with consistent styling."""
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QColor(Colors.MAIN_BACKGROUND))
        self.setPalette(palette)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        root.addLayout(layout)

        left_card = QFrame(self)
        left_card.setObjectName("graphDeck")
        left_card.setStyleSheet(
            f"QFrame#graphDeck {{ background-color: {Colors.CARD_BACKGROUND}; border: 1px solid {Colors.CARD_BORDER}; border-radius: 0px; }}"
        )
        left_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        left_card_layout = QVBoxLayout(left_card)
        left_card_layout.setContentsMargins(8, 8, 8, 8)
        left_card_layout.setSpacing(6)

        self.left_tab_widget = QTabWidget(left_card)
        self.left_tab_widget.setTabsClosable(True)
        self.left_tab_widget.tabCloseRequested.connect(self._on_tab_close_requested)
        self.left_tab_widget.setStyleSheet(f"""
            QTabWidget::pane {{
                background-color: {Colors.CARD_BACKGROUND};
                border: none;
            }}
            QTabBar::tab {{
                background-color: {Colors.CONTENT_BACKGROUND};
                border: 1px solid {Colors.CARD_BORDER};
                border-bottom: none;
                border-top-left-radius: 0px;
                border-top-right-radius: 0px;
                padding: 6px 12px;
                margin-right: 2px;
                font-size: 9px;
                color: {Colors.TEXT_SECONDARY};
            }}
            QTabBar::tab:selected {{
                background-color: {Colors.CARD_BACKGROUND};
                color: {Colors.ACCENT_BLUE};
                font-weight: 600;
            }}
            QTabBar::tab:hover:!selected {{
                background-color: {Colors.INFO_PANEL_INTERNAL_BG};
            }}
        """)
        self.left_tab_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.left_tab_widget.setMinimumWidth(220)
        self._add_initial_data_tabs()
        left_card_layout.addWidget(self.left_tab_widget)
        layout.addWidget(left_card, 1)

        self.sidebar = SidebarBuilder(self).build()
        layout.addWidget(self.sidebar, 0)

        try:
            self._on_mode_change(self.approach_combo.currentIndex())
        except Exception as e:
            logger.debug(f"Initial mode application skipped: {e}")


    def _create_matrix_table(self) -> QTableWidget:
        """Create a formatted table widget for coefficient/slope display."""
        return WidgetFactory.create_matrix_table()

    def _populate_approach_combo(self, combo: QComboBox) -> None:
        """Populate conversion mode combo box."""
        ComboBoxFactory.create_approach_combo(combo)

    def _make_sorted_combo(self, *, hint=None, prefer=None) -> QComboBox:
        """Create a sorted combo box with species from calibration data."""
        cb = ComboBoxFactory.create_species_combo(self, hint=hint, prefer=prefer)
        cb.setMaximumWidth(220)
        cb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        return cb

    def _build_species_group(self) -> QWidget:
        """Build the stacked method configuration UI for all modes."""
        box = QWidget(self)
        box.setStyleSheet("background-color: transparent;")
        v = QVBoxLayout(box)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(6)

        # Method configuration header (will be hidden for modes without species selection)
        self.method_config_header_layout = self._build_method_header()
        v.addLayout(self.method_config_header_layout)

        self.species_stack = QStackedWidget(self)
        self.species_stack.setStyleSheet("background-color: transparent;")
        # Use Maximum size policy to allow shrinking to fit current page
        self.species_stack.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        # Connect signal to adjust size when page changes
        self.species_stack.currentChanged.connect(lambda: self.species_stack.updateGeometry())

        # ========== EMPTY PAGE (for modes that don't need species selection) ==========
        # Used for: tmsd, internal_tmsd, normalised_sd, hmsd
        empty_page = QWidget()
        empty_page.setStyleSheet("background-color: transparent;")
        empty_page.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.species_stack.addWidget(empty_page)

        # ========== MASS PAGE ==========
        mass_page = QWidget()
        mass_page.setStyleSheet("background-color: transparent;")
        mass_page.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        mass_layout = QVBoxLayout(mass_page)
        mass_layout.setContentsMargins(0, 0, 0, 0)
        mass_layout.setSpacing(4)

        # Species selection grid
        mass_grid = QGridLayout()
        mass_grid.setHorizontalSpacing(4)
        mass_grid.setVerticalSpacing(3)
        rows_m = ["Donor", "FRET"]
        cols_m = ["B1B2", "B1L", "B1AA", "B2"]
        self.mass_combos: Dict[tuple[str, str], QComboBox] = {}

        for r, row in enumerate(rows_m):
            label = QLabel(row)
            label.setStyleSheet(f"font-weight: bold; font-size: 8pt; color: {Colors.TEXT_SECONDARY}; background-color: transparent;")
            mass_grid.addWidget(label, r + 1, 0)

        for c, col in enumerate(cols_m):
            label = QLabel(col)
            label.setStyleSheet(f"font-weight: bold; font-size: 8pt; color: {Colors.TEXT_SECONDARY}; background-color: transparent;")
            mass_grid.addWidget(label, 0, c + 1)

        for r, row in enumerate(rows_m):
            for c, col in enumerate(cols_m):
                cb = self._make_sorted_combo(prefer=["donor", "acceptor", "fret", "blocked"])
                cb.setMinimumWidth(85)
                cb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
                self.mass_combos[(row, col)] = cb
                mass_grid.addWidget(cb, r + 1, c + 1)

        mass_layout.addLayout(mass_grid)
        mass_layout.addStretch()
        self.species_stack.addWidget(mass_page)


        v.addWidget(self.species_stack)

        return box

    def _build_method_header(self) -> QGridLayout:
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 4)
        grid.setHorizontalSpacing(4)
        self.method_config_title_label = QLabel("Method configuration", self)
        self.method_config_title_label.setStyleSheet(UITheme.get_label_style_secondary())
        grid.addWidget(self.method_config_title_label, 0, 0, 1, 1)
        return grid

    def _show_mode_info_dialog(self):
        """Display mode information dialog when info button is clicked."""
        mode = self.approach_combo.currentData()
        if mode:
            dialog = ModeInfoDialog(mode, self)
            dialog.exec_()

    def _get_mode_info_text(self, mode: str) -> str:
        """Return detailed information text for each conversion mode."""
        return ModeInfoProvider.get_mode_info_text(mode)

    def _on_mode_change(self, idx: int) -> None:
        """Handle mode change with explicit flags and GUI updates."""
        try:
            mode = self.approach_combo.itemData(idx)
            if mode is None:
                logger.warning(f"Invalid mode index: {idx}")
                return

            self._mode_flags_from_combo()

            page_for = {"tmsd": 0, "internal_tmsd": 0, "normalised_sd": 0, "hmsd": 0, "mass": 1}
            page_idx = page_for.get(mode, 0)
            if hasattr(self, 'species_stack') and self.species_stack is not None:
                self.species_stack.setCurrentIndex(page_idx)
                # Force layout recalculation for dynamic sizing
                current_page = self.species_stack.currentWidget()
                if current_page:
                    current_page.updateGeometry()
                self.species_stack.updateGeometry()


            self._apply_mode_visibility_rules(mode)

            if self.s():
                try:
                    params = parse_params(self.load_parameters())
                    self._refresh_matrices_if_available(params)
                except Exception as e:
                    logger.debug(f"Mode-change refresh skipped: {e}")

        except Exception as e:
            logger.error(f"Error in _on_mode_change: {e}")

    def s(self) -> bool:
        """Check if all required widgets are initialised and ready."""
        required_attrs = [
            'approach_combo', 'species_stack', 'init_start_time_entry',
            'init_end_time_entry', 'reaction_trigger_timepoint',
            'injection_for_nuking_entry', 'injection_for_nuking_entry_2',
            'pos_ctrl_concentration_entries'
        ]
        for attr in required_attrs:
            if not hasattr(self, attr) or getattr(self, attr) is None:
                return False
        return True

    def _wire_events(self) -> None:
        """Connect UI signals to handlers."""
        self.plot_mode_button.clicked.connect(self._on_plot_mode_toggle)
        self.convert_button.clicked.connect(self._on_convert)
        self.export_button.clicked.connect(self._on_export)
        self.approach_combo.currentIndexChanged.connect(self._on_mode_change)
        self.approach_combo.currentIndexChanged.connect(self.update_species_selection_state)
        self.c_ref_change_button.clicked.connect(self._on_c_ref_change)

        # Forward to kinetics connections
        self.dataset_combo.currentIndexChanged.connect(self._on_dataset_selected)
        self.forward_button.clicked.connect(self._forward_to_kinetics)

        # Initialize c_ref display
        self._update_c_ref_display()

    def _update_c_ref_display(self) -> None:
        """Update the c_ref display label to show the current value."""
        c_ref_value = self.c_ref_entry.text().strip() or "10"
        self.c_ref_display_label.setText(f"Positive control reference: {c_ref_value} nM")


    def _on_c_ref_change(self) -> None:
        """Handle the c_ref change button click."""
        from PyQt5.QtWidgets import QInputDialog

        current_value = self.c_ref_entry.text().strip() or "10"

        value, ok = QInputDialog.getDouble(
            self,
            "Change Reference Concentration",
            "Enter positive control reference concentration (nM):",
            float(current_value),
            0.01,  # minimum
            1000.0,  # maximum
            2  # decimals
        )

        if ok:
            self.c_ref_entry.setText(str(value))
            self._update_c_ref_display()

    def load_parameters(self) -> ConvertParams:
        """
        Read and validate user input, returning a fully structured ConvertParams object.
        This is the single source of truth for parameter extraction.
        """
        def _to_float(name: str, entry: QLineEdit) -> float:
            txt = entry.text().strip()
            if not txt:
                raise ValueError(f"{name} is required")
            try:
                return float(txt)
            except Exception:
                raise ValueError(f"Invalid value for {name}: '{txt}'")

        def _validate_concentration(value: float, name: str, typical_min: float = 0.1, typical_max: float = 100.0) -> float:
            """Validate concentration value and prompt user if it seems unrealistic."""
            if value < 0:
                raise ValueError(f"{name} cannot be negative: {value}")

            if value > typical_max:
                suggestion_div10 = value / 10.0
                suggestion_div100 = value / 100.0

                msg = (
                    f"Unusual {name}: {value:.2f} nM\n\n"
                    f"Typical range: {typical_min}-{typical_max} nM\n"
                    f"Your value is {value/typical_max:.1f}x higher than typical.\n\n"
                    f"Possible causes:\n"
                    f"  - Wrong units (uM instead of nM?)\n"
                    f"  - Typo in decimal placement\n\n"
                    f"Common corrections:\n"
                    f"  - {suggestion_div10:.2f} nM (div 10)\n"
                    f"  - {suggestion_div100:.2f} nM (div 100)\n\n"
                    f"Keep original value {value:.2f} nM?"
                )

                reply = QMessageBox.question(
                    self,
                    "Unusual Concentration Value",
                    msg,
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )

                if reply == QMessageBox.No:
                    raise ValueError(
                        f"{name} seems unrealistic ({value:.2f} nM). "
                        f"Please check the value and re-enter."
                    )

            elif value < typical_min and value > 0:
                msg = (
                    f"Unusual {name}: {value:.4f} nM\n\n"
                    f"Typical range: {typical_min}-{typical_max} nM\n"
                    f"Your value is {typical_min/value:.1f}x lower than typical.\n\n"
                    f"This may indicate:\n"
                    f"  - Very low concentration experiment\n"
                    f"  - Decimal point error\n\n"
                    f"Proceed with {value:.4f} nM?"
                )

                reply = QMessageBox.question(
                    self,
                    "Low Concentration Value",
                    msg,
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )

                if reply == QMessageBox.No:
                    raise ValueError(
                        f"{name} seems unusually low ({value:.4f} nM). "
                        f"Please verify and re-enter."
                    )

            return value

        def _to_float_optional(entry: QLineEdit) -> Optional[float]:
            txt = entry.text().strip()
            if not txt:
                return None
            try:
                return float(txt)
            except Exception:
                raise ValueError(f"Invalid numeric value: '{txt}'")

        mode = self.approach_combo.currentData()
        if not mode:
            raise ValueError("Please select a conversion mode")

        init_start = _to_float("Init start", self.init_start_time_entry)
        init_end = _to_float("Init end", self.init_end_time_entry)
        trigger = _to_float("Trigger time", self.reaction_trigger_timepoint)

        neg_ctrl_start = _to_float_optional(self.neg_ctrl_start_time_entry)
        neg_ctrl_end = _to_float_optional(self.neg_ctrl_end_time_entry)

        injection_primary = _to_float_optional(self.injection_for_nuking_entry)
        injection_secondary = _to_float_optional(self.injection_for_nuking_entry_2)

        b1l_initial_conc = None
        if mode == "mass":
            try:
                b1l_initial_conc = _to_float("B1-L Initial Concentration", self.b1l_initial_conc_entry)
            except ValueError as e:
                raise ValueError(f"Mass-action mode requires B1-L initial concentration: {e}")

        # Read c_ref value (positive control reference concentration)
        try:
            c_ref = _to_float("Positive Control Reference", self.c_ref_entry)
            if c_ref <= 0:
                raise ValueError("Positive Control Reference must be positive")
        except ValueError:
            # If not provided or invalid, default to 10.0 nM
            c_ref = 10.0
            logger.warning("Invalid or missing C_ref value, using default: 10.0 nM")

        pos_ctrl_concentrations = []
        for i, entry in enumerate(self.pos_ctrl_concentration_entries, start=1):
            txt = entry.text().strip()
            if txt:
                try:
                    pos_ctrl_concentrations.append(float(txt))
                except Exception:
                    raise ValueError(f"Invalid positive control concentration at index {i}: '{txt}'")

        donor_ctrl_concentrations = []
        if self.donor_data is not None and (not hasattr(self.donor_data, 'empty') or not self.donor_data.empty):
            if self.donor_ctrl_concentration_entry is not None:
                txt = self.donor_ctrl_concentration_entry.text().strip()
                if txt:
                    try:
                        donor_ctrl_concentrations.append(float(txt))
                    except Exception:
                        raise ValueError(f"Invalid donor control concentration: '{txt}'")

        acceptor_ctrl_concentrations = []
        if self.acceptor_data is not None and (
                not hasattr(self.acceptor_data, 'empty') or not self.acceptor_data.empty):
            if self.acceptor_ctrl_concentration_entry is not None:
                txt = self.acceptor_ctrl_concentration_entry.text().strip()
                if txt:
                    try:
                        acceptor_ctrl_concentrations.append(float(txt))
                    except Exception:
                        raise ValueError(f"Invalid acceptor control concentration: '{txt}'")

        # B1L control concentration (for runtime slope calibration)
        # This is separate from b1l_initial_conc (experimental starting concentration)
        b1l_ctrl_concentration = None
        if self.blocked_ctrl_data is not None and (
                not hasattr(self.blocked_ctrl_data, 'empty') or not self.blocked_ctrl_data.empty):
            if self.b1l_ctrl_concentration_entry is not None:
                txt = self.b1l_ctrl_concentration_entry.text().strip()
                if txt:
                    try:
                        b1l_ctrl_concentration = float(txt)
                    except Exception:
                        raise ValueError(f"Invalid B1L control concentration: '{txt}'")

        neg_ctrl_concentrations = []
        for i, entry in enumerate(self.neg_ctrl_concentration_entries, start=1):
            txt = entry.text().strip()
            if txt:
                try:
                    neg_ctrl_concentrations.append(float(txt))
                except Exception:
                    raise ValueError(f"Invalid negative control concentration at index {i}: '{txt}'")

        start_with_donor = bool(getattr(self, "start_with_donor", True))


        species_map = self.species_manager.collect_species_map(mode)

        missing = self.species_manager.validate_required_species(mode, species_map)
        if missing:
            required_list = self.species_manager.get_required_species_list(mode)

            missing_str = '\n  - '.join(missing)
            required_str = ', '.join(required_list)

            error_msg = (
                f"Missing Required Calibrations for '{mode.upper()}' Mode\n\n"
                f"Required species: {required_list}\n\n"
                f"Missing calibrations:\n  - {missing_str}\n\n"
                f"To fix this issue:\n\n"
                f"1. Open the 'Calibration' section in the Convert Data tab\n"
                f"2. For each missing species:\n"
                f"   - Select the species from the dropdown menu\n"
                f"   - The slope and intercept will load automatically\n"
                f"   - Or manually enter calibration values if needed\n\n"
                f"3. Ensure all required species are assigned\n"
                f"4. Try conversion again\n\n"
                f"Note: Calibration data is loaded from:\n"
                f"      config/calibration_data.json"
            )

            raise ValueError(error_msg)

        logger.info("Collected %d species calibrations for mode %s", len(species_map), mode.upper())

        params = ConvertParams(
            mode=mode,
            init_start=init_start,
            init_end=init_end,
            trigger=trigger,
            neg_ctrl_start=neg_ctrl_start,
            neg_ctrl_end=neg_ctrl_end,
            injection_primary=injection_primary,
            injection_secondary=injection_secondary,
            beta_DF=None,
            beta_AF=None,
            beta_BF=None,
            pos_ctrl_concentrations=pos_ctrl_concentrations,
            donor_ctrl_concentrations=donor_ctrl_concentrations,
            acceptor_ctrl_concentrations=acceptor_ctrl_concentrations,
            neg_ctrl_concentrations=neg_ctrl_concentrations,
            c_ref=c_ref,
            start_with_donor=start_with_donor,
        )

        object.__setattr__(params, 'species', species_map)

        # Get calibration mode (default to "precalibrated" for backward compatibility)
        calibration_mode = "precalibrated"
        if hasattr(self, 'calibration_mode_combo') and self.calibration_mode_combo is not None:
            calibration_mode = self.calibration_mode_combo.currentData() or "precalibrated"

        # B1-L initial concentration is the experimental starting concentration
        # B1-L control concentration is for slope calibration (may differ from experimental)
        object.__setattr__(params, 'extra', {
            "selected_channels": {
                "donor": getattr(self, "selected_donor_channel", None),
                "acceptor": getattr(self, "selected_acceptor_channel", None),
                "fret": getattr(self, "selected_fret_channel", None),
                "set": (self.detected_channels or {}).get("resolved_set"),
            },
            "channel_detection": self.detected_channels or {},
            "b1l_initial_conc": b1l_initial_conc,
            "calibration_mode": calibration_mode,
            "b1l_ctrl_concentration": b1l_ctrl_concentration,
            "b1aa_ctrl_concentration": acceptor_ctrl_concentrations[0] if acceptor_ctrl_concentrations else None,
        })

        return params

    def _load_calibration(self) -> None:
        """Load calibration data from JSON file."""
        try:
            self.calibration_records = calibration.load(self.calibration_data_path)
            logger.info(f"Loaded {len(self.calibration_records)} calibration records")
        except calibration.CalibrationError as e:
            logger.warning(f"Calibration data issue: {e}")
            self.calibration_records = []
        except Exception as e:
            logger.error(f"Unexpected error loading calibration data: {e}")
            self.calibration_records = []

    def get_species_calibration(self, species_name: str) -> Dict[str, Any]:
        """Retrieve calibration data (slope and intercept) for a given species."""
        if species_name == "NONE" or not species_name:
            return {"species": species_name, "slope": None, "intercept": None}

        try:
            from .calibration import build_index, lookup
            idx = build_index(self.calibration_records)
            rec = lookup(idx, species_name)
            if rec:
                return {
                    "species": species_name,
                    "slope": rec.get("slope"),
                    "intercept": rec.get("intercept")
                }
        except Exception as e:
            logger.debug(f"Could not retrieve calibration for '{species_name}': {e}")

        return {"species": species_name, "slope": None, "intercept": None}

    def update_species_selection_state(self) -> None:
        """Update UI state based on current species selections."""
        try:
            self._mode_flags_from_combo()

            if self.s():
                params = parse_params(self.load_parameters())
                self._refresh_matrices_if_available(params)
        except Exception as e:
            logger.debug(f"update_species_selection_state skipped: {e}")

    def _check_and_show_posctrl_warning(self, result: dict) -> None:
        """Check for positive control warning in result and display if present."""
        if 'summary' in result:
            posctrl_warning = result['summary'].get('posctrl_warning')
            if posctrl_warning:
                QMessageBox.warning(
                    self,
                    "Positive Control Normalisation Warning",
                    posctrl_warning
                )

    def _on_convert(self) -> None:
        """Execute conversion based on selected mode."""
        try:
            params = self.load_parameters()
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Input", str(e))
            return

        mode = params.mode

        # Validate negative control for modes that require it
        if mode in ("internal_tmsd", "normalised_sd", "hp_quenching"):
            if self.neg_ctrl_data is None or (hasattr(self.neg_ctrl_data, 'empty') and self.neg_ctrl_data.empty):
                QMessageBox.critical(
                    self,
                    "Missing Negative Control",
                    f"The '{mode.replace('_', ' ').title()}' conversion mode requires negative control data.\n\n"
                    f"Please go back to the Data Frame Processor and select wells for the "
                    f"'Negative Control' category before proceeding with this conversion.\n\n"
                    f"The negative control is used as a baseline reference for calculating "
                    f"the initial fluorescence level."
                )
                return

        try:
            if mode == "tmsd":
                logger.info("Starting TMSD conversion")
                result = convert_tmsd_to_conc(
                    selected_data=self.selected_data,
                    pos_ctrl_data=self.pos_ctrl_data,
                    blank_ctrl_data=self.blank_ctrl_data,
                    params=params,
                )

                self._current_result = result
                self._show_result(result)
                self.export_button.setEnabled(True)

                self._register_conversion(result, params)

                # Check for positive control warning
                self._check_and_show_posctrl_warning(result)

                if 'summary' in result:
                    summary = result['summary']
                    n_success = summary.get('n_successful', 0)
                    n_failed = len(summary.get('failed_wells', []))

                    msg = f"TMSD conversion complete.\n\n{n_success} wells processed successfully."

                    if n_failed > 0:
                        failed_wells = summary.get('failed_wells', [])
                        msg += f"\n\nWarning: {n_failed} well(s) failed."
                        if n_failed <= 5:
                            msg += f"\n  {', '.join(failed_wells)}"

                    QMessageBox.information(self, "Conversion Complete", msg)

                logger.info("TMSD conversion completed successfully")

            elif mode == "internal_tmsd":
                logger.info("Starting Internal TMSD conversion")
                result = convert_internal_tmsd_to_conc(
                    selected_data=self.selected_data,
                    pos_ctrl_data=self.pos_ctrl_data,
                    blank_ctrl_data=self.blank_ctrl_data,
                    neg_ctrl_data=self.neg_ctrl_data,
                    params=params,
                )

                self._current_result = result
                self._show_result(result)
                self.export_button.setEnabled(True)

                self._register_conversion(result, params)

                # Check for positive control warning
                self._check_and_show_posctrl_warning(result)

                if 'summary' in result:
                    summary = result['summary']
                    n_success = summary.get('n_successful', 0)
                    n_failed = len(summary.get('failed_wells', []))

                    msg = f"Internal TMSD conversion complete.\n\n{n_success} wells processed successfully."

                    if n_failed > 0:
                        failed_wells = summary.get('failed_wells', [])
                        msg += f"\n\nWarning: {n_failed} well(s) failed."
                        if n_failed <= 5:
                            msg += f"\n  {', '.join(failed_wells)}"

                    QMessageBox.information(self, "Conversion Complete", msg)

                logger.info("Internal TMSD conversion completed successfully")

            elif mode == "normalised_sd":
                logger.info("Starting Normalised SD conversion (relative fluorescence)")
                result = convert_normalised_sd(
                    selected_data=self.selected_data,
                    pos_ctrl_data=self.pos_ctrl_data,
                    blank_ctrl_data=self.blank_ctrl_data,
                    neg_ctrl_data=self.neg_ctrl_data,
                    params=params,
                )

                self._current_result = result
                self._show_result(result)
                self.export_button.setEnabled(True)

                self._register_conversion(result, params)

                # Check for positive control warning
                self._check_and_show_posctrl_warning(result)

                if 'summary' in result:
                    summary = result['summary']
                    n_success = summary.get('n_successful', 0)
                    n_failed = summary.get('n_failed', 0)

                    msg = f"Normalised SD conversion complete.\n\n{n_success} wells processed successfully."

                    # Check for wells with high negative fractions (over-correction)
                    wells_high_neg = summary.get('wells_with_high_negatives', [])
                    if wells_high_neg:
                        msg += f"\n\nWarning: {len(wells_high_neg)} well(s) show high negative fractions"
                        msg += "\n(possible over-correction from constant baseline)."

                    if n_failed > 0:
                        failed_wells = summary.get('failed_wells', [])
                        msg += f"\n\nWarning: {n_failed} well(s) failed."
                        if n_failed <= 5:
                            msg += f"\n  {', '.join(failed_wells)}"

                    QMessageBox.information(self, "Conversion Complete", msg)

                logger.info("Normalised SD conversion completed successfully")

            elif mode == "hmsd":
                logger.info("Starting HMSD (Handhold) one-step FRET conversion")
                result = convert_fret_onestep_to_conc(
                    selected_data=self.selected_data,
                    pos_ctrl_data=self.pos_ctrl_data,
                    blank_ctrl_data=self.blank_ctrl_data,
                    donor_data=self.donor_data,
                    acceptor_data=self.acceptor_data,
                    params=params,
                )

                logger.info(f"Conversion returned {len(result)} output keys:")
                for key in result.keys():
                    if key != 'summary':
                        logger.info(f"  - {key}: {result[key].shape if hasattr(result[key], 'shape') else 'N/A'}")

                self._current_result = result
                self._show_result(result)
                self.export_button.setEnabled(True)

                self._register_conversion(result, params)

                # Check for positive control warning
                self._check_and_show_posctrl_warning(result)

                if 'summary' in result:
                    summary = result['summary']
                    n_success = summary.get('n_successful', 0)
                    n_failed = len(summary.get('failed_wells', []))

                    msg = f"HMSD conversion complete.\n\n{n_success} wells processed successfully."

                    if n_failed > 0:
                        failed_wells = summary.get('failed_wells', [])
                        msg += f"\n\nWarning: {n_failed} well(s) failed."
                        if n_failed <= 5:
                            msg += f"\n  {', '.join(failed_wells)}"

                    QMessageBox.information(self, "Conversion Complete", msg)

                if 'FRET Raw' in result and 'FRET Blank Subtracted' in result and 'FRET Normalised' in result:
                    logger.info("FRET diagnostic traces available for plotting:")
                    logger.info("  - FRET Raw: Original fluorescence")
                    logger.info("  - FRET Blank Subtracted: Background-corrected")
                    logger.info("  - FRET Normalised: Positive control normalised")

                if 'FRET Positive Control' in result:
                    logger.info("FRET Positive Control: Photobleaching/drift-corrected normalisation trajectory")

                logger.info("HMSD conversion completed successfully")

            elif mode == "mass":
                logger.info("Starting mass-action catalytic conversion")
                logger.info("Calibration mode: %s", params.extra.get('calibration_mode', 'precalibrated'))

                result = convert_catalytic_fret_2x2(
                    selected_data=self.selected_data,
                    pos_ctrl_data=self.pos_ctrl_data,
                    blank_data=self.blank_ctrl_data,
                    donor_data=self.donor_data,
                    acceptor_data=self.acceptor_data,
                    params=params,
                    b1l_ctrl_data=self.blocked_ctrl_data,  # B1-L quenched acceptor control
                    b1aa_ctrl_data=self.acceptor_data,     # B1-AA activated acceptor control
                )

                self._current_result = result
                self._show_result(result)
                self.export_button.setEnabled(True)

                self._register_conversion(result, params)

                # Check for positive control warning
                self._check_and_show_posctrl_warning(result)

                if 'summary' in result:
                    summary = result['summary']
                    n_success = summary.get('n_successful', 0)
                    n_failed = len(summary.get('failed_wells', []))

                    msg = f"Mass-action catalytic conversion complete.\n\n{n_success} wells processed successfully."

                    if n_failed > 0:
                        failed_wells = summary.get('failed_wells', [])
                        msg += f"\n\nWarning: {n_failed} well(s) failed."
                        if n_failed <= 5:
                            msg += f"\n  {', '.join(failed_wells)}"

                    QMessageBox.information(self, "Conversion Complete", msg)

                logger.info("Mass-action conversion completed successfully")

            elif mode == "hp_quenching":
                logger.info("Starting Hairpin Quenching TMSD conversion")
                result = convert_reverted_readout_to_conc(
                    selected_data=self.selected_data,
                    pos_ctrl_data=self.pos_ctrl_data,
                    neg_ctrl_data=self.neg_ctrl_data,
                    blank_ctrl_data=self.blank_ctrl_data,
                    params=params,
                )

                self._current_result = result
                self._show_result(result)
                self.export_button.setEnabled(True)

                self._register_conversion(result, params)

                self._check_and_show_posctrl_warning(result)

                if 'summary' in result:
                    summary = result['summary']
                    n_success = summary.get('n_successful', 0)
                    n_failed = len(summary.get('failed_wells', []))

                    msg = f"Hairpin Quenching TMSD conversion complete.\n\n{n_success} wells processed successfully."
                    if summary.get('beta') is not None:
                        msg += f"\n\nBeta (HI/H brightness ratio) = {summary['beta']:.4f}"

                    if n_failed > 0:
                        failed_wells = summary.get('failed_wells', [])
                        msg += f"\n\nWarning: {n_failed} well(s) failed."
                        if n_failed <= 5:
                            msg += f"\n  {', '.join(failed_wells)}"

                    QMessageBox.information(self, "Conversion Complete", msg)

                logger.info("Hairpin Quenching TMSD conversion completed successfully")

            else:
                QMessageBox.critical(self, "Unsupported mode", f"Unknown mode: {mode}")
                return

        except Exception as e:
            logger.exception("Conversion failed")
            QMessageBox.critical(self, "Conversion", f"Conversion failed:\n\n{str(e)}")
            return

        try:
            self._refresh_matrices_if_available(params)
        except Exception as e:
            logger.debug(f"Matrix refresh failed: {e}")

    def _refresh_matrices_if_available(self, params) -> None:
        """Render coefficient and bleed-through matrices using MatrixRenderer."""
        try:
            self.matrix_renderer.refresh_matrices_for_mode(params.mode)
        except Exception as e:
            logger.debug(f"Matrix refresh failed: {e}")

    def _on_plot_mode_toggle(self) -> None:
        """Toggle between scatter and line plot modes and update all result tabs."""
        is_scatter = self.plot_mode_button.isChecked()
        new_mode = 'scatter' if is_scatter else 'line'

        self.plot_mode_button.setText("Scatter" if is_scatter else "Line")
        self.plot_mode = new_mode

        logger.info(f"Plot mode toggled to: {new_mode}")

        if hasattr(self, 'left_tab_widget'):
            for i in range(self.left_tab_widget.count()):
                widget = self.left_tab_widget.widget(i)
                if isinstance(widget, PlotDashboard):
                    widget.update_plot_mode(new_mode)
                    logger.info(f"Updated plot mode for tab {i}: {self.left_tab_widget.tabText(i)}")

    def _on_export(self) -> None:
        """Export all conversion results and metadata to a structured folder."""
        if self._current_result is None:
            QMessageBox.information(self, "Export", "Nothing to export yet.")
            return

        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select Export Folder",
            "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )

        if not folder_path:
            return

        try:
            params = self.load_parameters()

            export_dir = io_utils.export_conversion_package(
                result=self._current_result,
                params=params,
                folder_path=folder_path
            )

            QMessageBox.information(
                self,
                "Export Successful",
                f"Successfully exported conversion package to:\n\n{export_dir}\n\n"
                f"Package includes:\n"
                f"- All result DataFrames (CSV)\n"
                f"- Summary statistics (JSON)\n"
                f"- Input parameters and metadata (JSON)\n"
                f"- Human-readable README (TXT)"
            )

            logger.info(f"Export completed successfully to {export_dir}")

        except Exception as e:
            logger.exception("Export failed")
            QMessageBox.critical(
                self,
                "Export Failed",
                f"Failed to export results:\n\n{str(e)}\n\n"
                f"Check the log file for details."
            )

    def _show_result(self, result: Any) -> None:
        """Display conversion results with plots in a new tab."""
        try:
            import pandas as pd

            if isinstance(result, dict):
                keys = ", ".join(result.keys())
                # Result display removed - status shown in plot tabs
                # self.result_display.setPlainText(f"Received {len(result)} result tables: {keys}")

                dataframes_to_plot = []
                titles_to_use = []

                # Keys to skip (not time-series data, cannot be plotted)
                skip_keys = {'summary', 'Diagnostics', 'calibration_comparison'}

                for key, value in result.items():
                    if key in skip_keys:
                        continue

                    # Skip scalar values (cannot be plotted)
                    if isinstance(value, (int, float, str, bool)):
                        continue

                    if isinstance(value, pd.DataFrame):
                        dataframes_to_plot.append(value)
                        titles_to_use.append(key)

                    elif isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, pd.DataFrame):
                                dataframes_to_plot.append(sub_value)
                                titles_to_use.append(f"{key}_{sub_key}")

                if dataframes_to_plot:
                    self.plot_df_in_new_tab(*dataframes_to_plot, titles=titles_to_use)

            elif hasattr(result, "head"):
                # head_str = pd.DataFrame(result).head(10).to_string(index=False)
                # self.result_display.setPlainText(head_str)
                self.plot_df_in_new_tab(result, titles="Conversion Result")
            # else:
                # self.result_display.setPlainText(str(result))
        except Exception as e:
            logger.error(f"Error displaying results: {e}")
            # self.result_display.setPlainText(str(result))

    def plot_df_in_new_tab(self, *dataframes, titles=None):
        """Create an interactive dashboard tab for the supplied DataFrames."""
        if not dataframes:
            return

        if titles is None:
            titles = [f"DataFrame {i + 1}" for i in range(len(dataframes))]
        elif isinstance(titles, str):
            titles = [titles] * len(dataframes)
        elif isinstance(titles, list):
            if len(titles) != len(dataframes):
                raise ValueError("Number of titles must match number of DataFrames.")
        else:
            raise TypeError("Titles must be a string, a list of strings, or None.")

        display_titles = titles

        try:
            dashboard = PlotDashboard(list(dataframes), display_titles, self, plot_mode=self.plot_mode)

            if display_titles:
                first_title = display_titles[0]
                tab_name = first_title[:12] + "..." if len(first_title) > 12 else first_title
            else:
                tab_name = "Results"

            self.left_tab_widget.addTab(dashboard, tab_name)
            self.left_tab_widget.setCurrentWidget(dashboard)

            logger.info(f"Created dashboard tab '{tab_name}' with {len(dataframes)} datasets")

        except Exception as e:
            logger.error(f"Failed to create plot dashboard: {e}")
            error_tab = QWidget()
            error_layout = QVBoxLayout(error_tab)
            error_label = QLabel(f"Error creating dashboard: {e}")
            error_label.setWordWrap(True)
            error_label.setStyleSheet(f"color: {Colors.ACCENT_RED}; padding: 20px;")
            error_layout.addWidget(error_label)
            self.left_tab_widget.addTab(error_tab, "Error")

    def check_for_missing_don_acc_data(self) -> None:
        """Warn if FRET-like modes are selected but donor/acceptor data were not provided."""
        if self.donor_data is None or self.acceptor_data is None:
            logger.info("Donor or acceptor data missing - FRET modes may be limited")

    def refresh_channels(self, _channels: dict = None) -> None:
        """
        Re-detect channels from the latest saved configuration.

        Intended as a slot for ``OptionsPanel.channelsUpdated``. The signal
        payload is accepted but unused — detection always reads from disk so
        that the full scoring/matching logic runs against the loaded data.
        """
        logger.info("Channel configuration changed — re-detecting channels")
        self._detect_and_log_channels()

    def _detect_and_log_channels(self) -> None:
        """Detect available channels using ChannelDetector."""
        try:
            cfg_path = get_data_path("config/channel_settings.json")
            detector = ChannelDetector(cfg_path)

            result = detector.detect_channels(
                self.selected_data,
                self.pos_ctrl_data,
                self.blank_ctrl_data,
                self.donor_data,
                self.acceptor_data
            )

            self.selected_donor_channel = result.selected_donor
            self.selected_acceptor_channel = result.selected_acceptor
            self.selected_fret_channel = result.selected_fret
            self.detected_channels = result.to_dict()

            logger.info(
                f"Channel detection: Set {result.resolved_set} ({result.resolved_set_name}), "
                f"Donor={result.donor_info['matches']} matches, "
                f"Acceptor={result.acceptor_info['matches']} matches, "
                f"FRET={result.fret_info['matches']} matches"
            )

            self._check_and_offer_save_channels(result, detector, cfg_path)

        except Exception as e:
            logger.error(f"Channel detection failed: {e}")
            self.selected_donor_channel = None
            self.selected_acceptor_channel = None
            self.selected_fret_channel = None
            self.detected_channels = {
                "resolved_set": None,
                "resolved_set_name": None,
                "Donor": {"selected": None, "matches": 0, "examples": []},
                "Acceptor": {"selected": None, "matches": 0, "examples": []},
                "FRET": {"selected": None, "matches": 0, "examples": []},
            }

    def _check_and_offer_save_channels(self, result, detector, cfg_path) -> None:
        """Check if detected channels are new and offer to save them to configuration."""
        if not all([result.selected_donor, result.selected_acceptor, result.selected_fret]):
            return

        is_known = self._is_channel_config_known(
            result.selected_donor,
            result.selected_acceptor,
            result.selected_fret,
            detector.channel_sets
        )

        if is_known:
            return

        msg = (
            f"New Channel Configuration Detected!\n\n"
            f"Clearissa detected the following channels in your data:\n\n"
            f"  - Donor:    {result.selected_donor}\n"
            f"  - Acceptor: {result.selected_acceptor}\n"
            f"  - FRET:     {result.selected_fret}\n\n"
            f"This configuration is not in your saved channel settings.\n\n"
            f"Would you like to save it for future experiments?\n"
            f"This will make channel detection faster next time."
        )

        reply = QMessageBox.question(
            self,
            "Save New Channel Configuration?",
            msg,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )

        if reply == QMessageBox.Yes:
            self._save_new_channel_config(
                result.selected_donor,
                result.selected_acceptor,
                result.selected_fret,
                cfg_path
            )

    def _is_channel_config_known(self, donor, acceptor, fret, channel_sets) -> bool:
        """Check if a channel configuration already exists in saved settings."""
        for set_name, channels in channel_sets.items():
            if (channels.get("Donor 1") == donor and
                channels.get("Acceptor 1") == acceptor and
                channels.get("FRET 1") == fret):
                return True

            if (channels.get("Donor 2") == donor and
                channels.get("Acceptor 2") == acceptor and
                channels.get("FRET 2") == fret):
                return True

        return False

    def _save_new_channel_config(self, donor, acceptor, fret, cfg_path) -> None:
        """Save a new channel configuration to the settings file."""
        try:
            import json
            from pathlib import Path

            if cfg_path.exists():
                with open(cfg_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                data = {"channels": {}}

            channels = data.get("channels", {})

            existing_sets = [k for k in channels.keys() if k.startswith("Set ")]
            set_numbers = [int(s.replace("Set ", "")) for s in existing_sets if s.replace("Set ", "").isdigit()]
            next_set_num = max(set_numbers) + 1 if set_numbers else 1
            new_set_name = f"Set {next_set_num}"

            channels[new_set_name] = {
                "Donor 1": donor,
                "Acceptor 1": acceptor,
                "FRET 1": fret,
                "Donor 2": "",
                "Acceptor 2": "",
                "FRET 2": ""
            }

            data["channels"] = channels

            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cfg_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved new channel configuration as '{new_set_name}'")

            # Sync the OptionsPanel in-memory state so it doesn't overwrite
            # this new set on the next "Save and Apply"
            main_app = self.parent()
            if hasattr(main_app, 'optionsproc'):
                main_app.optionsproc.reload_from_file()

            QMessageBox.information(
                self,
                "Channel Configuration Saved",
                f"Successfully saved as '{new_set_name}'\n\n"
                f"This configuration will be automatically detected\n"
                f"in future experiments with the same channel setup.\n\n"
                f"You can edit channel settings in:\n"
                f"config/channel_settings.json"
            )

        except Exception as e:
            logger.error(f"Failed to save channel configuration: {e}")
            QMessageBox.warning(
                self,
                "Save Failed",
                f"Could not save channel configuration:\n\n{str(e)}\n\n"
                f"You can manually add it to:\n"
                f"config/channel_settings.json"
            )

    def save_state(self) -> None:
        """Save current state using StateManager."""
        try:
            self.state_manager.save_state()
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def load_state(self) -> None:
        """Load state using StateManager."""
        try:
            snapshot = self.state_manager.load_state()
            if snapshot:
                self.state_manager.restore_state(snapshot)
        except Exception as e:
            logger.error(f"Failed to load state: {e}")

    def closeEvent(self, event) -> None:
        """Clean up resources when the tab is closed."""
        try:
            self.save_state()

            if self.plot_widget_top is not None:
                try:
                    self.plot_widget_top.deleteLater()
                except:
                    pass

            if self.plot_widget_bottom is not None:
                try:
                    self.plot_widget_bottom.deleteLater()
                except:
                    pass

        except Exception as e:
            logger.error(f"Error during closeEvent: {e}")
        finally:
            event.accept()

    def _add_initial_data_tabs(self) -> None:
        """Add overview tab showing all input data."""
        try:
            overview_tab = QWidget()
            overview_tab.setStyleSheet(f"QWidget {{ background-color: transparent; }}")
            overview_layout = QVBoxLayout(overview_tab)
            overview_layout.setContentsMargins(8, 8, 8, 8)
            overview_layout.setSpacing(8)

            header = QWidget()
            header_layout = QHBoxLayout(header)
            header_layout.setContentsMargins(0, 0, 0, 8)
            title_label = QLabel("Input Data Overview")
            title_label.setStyleSheet(f"font-size: 10pt; font-weight: 600; color: {Colors.TEXT_PRIMARY}; background-color: transparent;")
            header_layout.addWidget(title_label)
            header_layout.addStretch()
            overview_layout.addWidget(header)

            graphics_layout = pg.GraphicsLayoutWidget()
            graphics_layout.setBackground('w')

            graphics_layout.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            graphics_layout.setMinimumHeight(200)
            graphics_layout.setMinimumWidth(200)

            overview_layout.addWidget(graphics_layout, 1)

            # Colourblind-friendly academic palette
            academic_colours = [
                '#E6194B',  # Red
                '#3CB44B',  # Green
                '#0082C8',  # Blue
                '#F58231',  # Orange
                '#911EB4',  # Purple
                '#46F0F0',  # Cyan
                '#F032E6',  # Magenta
                '#D2F53C',  # Lime
            ]

            datasets = []
            dataset_idx = 0
            if isinstance(self.selected_data, pd.DataFrame) and not self.selected_data.empty:
                datasets.append(("Selected Data", self.selected_data, academic_colours[dataset_idx % len(academic_colours)]))
                dataset_idx += 1
            if isinstance(self.pos_ctrl_data, pd.DataFrame) and not self.pos_ctrl_data.empty:
                datasets.append(("Positive Control", self.pos_ctrl_data, academic_colours[dataset_idx % len(academic_colours)]))
                dataset_idx += 1
            if isinstance(self.blank_ctrl_data, pd.DataFrame) and not self.blank_ctrl_data.empty:
                datasets.append(("Blank", self.blank_ctrl_data, academic_colours[dataset_idx % len(academic_colours)]))
                dataset_idx += 1
            if isinstance(self.neg_ctrl_data, pd.DataFrame) and not self.neg_ctrl_data.empty:
                datasets.append(("Negative Control", self.neg_ctrl_data, academic_colours[dataset_idx % len(academic_colours)]))
                dataset_idx += 1

            if not datasets:
                lbl = QLabel("No input data available.")
                lbl.setStyleSheet(f"color: {Colors.TEXT_TERTIARY}; background-color: transparent;")
                overview_layout.addWidget(lbl)
                self.left_tab_widget.addTab(overview_tab, "Input Data")
                return

            num_datasets = len(datasets)
            cols = 2 if num_datasets > 2 else num_datasets
            for idx, (title, df, colour) in enumerate(datasets):
                if idx > 0 and idx % cols == 0:
                    graphics_layout.nextRow()
                plot = graphics_layout.addPlot()

                plot.setTitle(title, color='#000', size='11pt')

                label_style = {'color': '#000', 'font-size': '10pt', 'font-weight': 'bold', 'font-family': 'Arial'}
                plot.setLabel('left', 'Fluorescence (AFU)', **label_style)
                plot.setLabel('bottom', 'Time (min)', **label_style)

                plot.showGrid(x=True, y=True, alpha=0.15)

                axis_pen = pg.mkPen('k', width=2)
                plot.getAxis('bottom').setPen(axis_pen)
                plot.getAxis('left').setPen(axis_pen)
                plot.getAxis('bottom').setTextPen('k')
                plot.getAxis('left').setTextPen('k')

                time_col = next((c for c in ['Time [min]', 'Time', 'time', 't'] if c in df.columns), None)
                if time_col is None:
                    logger.warning(f"No time column found in {title}")
                    continue

                df = df.copy()
                df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
                df = df.dropna(subset=[time_col]).sort_values(by=time_col)

                if 'Well' not in df.columns:
                    wells = [None]
                else:
                    wells = sorted(df['Well'].dropna().unique())

                trace_idx = 0
                for w_i, well_name in enumerate(wells):
                    if well_name is not None:
                        sub_df = df[df['Well'] == well_name]
                    else:
                        sub_df = df

                    y_cols = [c for c in sub_df.columns if c not in ['Well', time_col, 'Content']]
                    for i, y_col in enumerate(y_cols):
                        ydata = pd.to_numeric(sub_df[y_col], errors='coerce')
                        xdata = pd.to_numeric(sub_df[time_col], errors='coerce')

                        valid = np.isfinite(xdata) & np.isfinite(ydata)
                        if not valid.any():
                            continue

                        x_np = xdata[valid].to_numpy()
                        y_np = ydata[valid].to_numpy()

                        trace_colour = academic_colours[trace_idx % len(academic_colours)]
                        colour_pen = pg.mkPen(trace_colour, width=2.5)
                        plot.plot(x_np, y_np, pen=colour_pen, name=f"{well_name} - {y_col}" if well_name else y_col)
                        trace_idx += 1

                if len(plot.listDataItems()) <= 12:
                    legend = plot.addLegend(offset=(8, 8))
                    legend.setLabelTextColor('#000')

            # Colored indicators removed for cleaner interface
            self.left_tab_widget.addTab(overview_tab, "Input Data")
            self.left_tab_widget.setCurrentIndex(0)

        except Exception as e:
            logger.error(f"Failed to create input data overview: {e}")
            fallback = QWidget()
            layout = QVBoxLayout(fallback)
            err = QLabel(f"Error creating overview: {e}")
            err.setStyleSheet(f"color: {Colors.ACCENT_RED}; padding: 8px;")
            layout.addWidget(err)
            self.left_tab_widget.addTab(fallback, "Input Data")

    def _on_tab_close_requested(self, index: int) -> None:
        """Handle tab close requests from the user."""
        try:
            if index >= 0 and index < self.left_tab_widget.count():
                self.left_tab_widget.removeTab(index)
        except Exception as e:
            logger.debug(f"Failed to close tab at index {index}: {e}")

    # =========================================================================
    # FORWARDING TO KINETICS PROCESSOR
    # =========================================================================

    def _register_conversion(self, result_dict: dict, params: ConvertParams) -> None:
        """Register a successful conversion in the data registry."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = params.mode.upper()

        # For HMSD mode, extract per-well initial donor concentrations
        initial_concentrations = {}
        if mode == "HMSD" and "Initial Donor [S1-T]_0 [nM]" in result_dict:
            s1t0_df = result_dict["Initial Donor [S1-T]_0 [nM]"]
            # Extract first value from each well column (values are constant per well)
            for col in s1t0_df.columns:
                if col not in ("Well", "Time [min]") and not col.lower().startswith("time"):
                    try:
                        # Get first non-NaN value from the column
                        val = s1t0_df[col].dropna().iloc[0] if len(s1t0_df[col].dropna()) > 0 else None
                        if val is not None:
                            initial_concentrations[col] = float(val)
                    except (IndexError, ValueError) as e:
                        logger.warning(f"Could not extract initial concentration for {col}: {e}")

            if initial_concentrations:
                logger.info(f"Extracted {len(initial_concentrations)} initial donor concentrations for HMSD")

        for species_name, df in result_dict.items():
            if species_name == "summary" or not isinstance(df, pd.DataFrame):
                continue

            label = f"{mode}_{species_name}_{timestamp}"

            self.converted_data_registry[label] = {
                'dataframe': df.copy(),
                'species': species_name,
                'mode': mode,
                'timestamp': timestamp,
                'params': params,
                'initial_concentrations': initial_concentrations if mode == "HMSD" else {},
            }

            logger.info(f"Registered conversion: {label}")

        self._update_dataset_combo()

    def _update_dataset_combo(self) -> None:
        """Refresh the dataset selection dropdown."""
        self.dataset_combo.blockSignals(True)
        self.dataset_combo.clear()

        if not self.converted_data_registry:
            self.dataset_combo.addItem("No datasets available")
            self.dataset_combo.setEnabled(False)
            self.forward_button.setEnabled(False)
            self.forward_info_label.setText("Convert data first to enable forwarding")
        else:
            sorted_keys = sorted(
                self.converted_data_registry.keys(),
                key=lambda k: self.converted_data_registry[k]['timestamp'],
                reverse=True
            )

            for key in sorted_keys:
                entry = self.converted_data_registry[key]
                display_text = f"{entry['mode']} - {entry['species']} ({entry['timestamp']})"
                self.dataset_combo.addItem(display_text, key)

            self.dataset_combo.setEnabled(True)
            self.forward_button.setEnabled(True)
            self.forward_info_label.setText(
                f"{len(self.converted_data_registry)} dataset(s) available"
            )

        self.dataset_combo.blockSignals(False)

    def _on_dataset_selected(self, index: int) -> None:
        """Handle dataset selection change."""
        if index >= 0 and self.dataset_combo.currentData():
            key = self.dataset_combo.currentData()
            entry = self.converted_data_registry.get(key)
            if entry:
                rows, cols = entry['dataframe'].shape
                self.forward_info_label.setText(
                    f"Selected: {rows} timepoints x {cols-2} wells"
                )

    def _forward_to_kinetics(self) -> None:
        """Forward selected dataset to Kinetics Processor."""
        if not self.dataset_combo.currentData():
            QMessageBox.warning(
                self,
                "No Dataset Selected",
                "Please select a dataset to forward."
            )
            return

        key = self.dataset_combo.currentData()
        entry = self.converted_data_registry.get(key)

        if not entry:
            QMessageBox.critical(
                self,
                "Error",
                f"Dataset '{key}' not found in registry."
            )
            return

        df = entry['dataframe']
        label = f"{entry['mode']}_{entry['species']}"

        if df.empty:
            QMessageBox.warning(
                self,
                "Empty Dataset",
                "Cannot forward an empty dataset."
            )
            return

        # Transform DataFrame for Kinetics Processor
        # The Convert Data tab DataFrames have format:
        #   Column 0: 'Well' - contains species name (e.g., 'AB', 'B1B2') repeated for each timepoint
        #   Column 1: 'Time [min]' - time values
        #   Columns 2+: Individual well data (A01, A02, etc.)
        #
        # The Kinetics Processor expects:
        #   Column 0: 'Time [min]' - time values
        #   Columns 1+: Individual well data (A01, A02, etc.)
        #
        # So we need to drop the 'Well' column which just contains the species name
        df_for_kinetics = df.copy()
        if 'Well' in df_for_kinetics.columns:
            # The 'Well' column contains the species name, not well identifiers
            # Drop it as the Kinetics Processor doesn't need it
            df_for_kinetics = df_for_kinetics.drop(columns=['Well'])

        # Build metadata dict for kinetics processor
        metadata = {
            'mode': entry.get('mode', ''),
            'species': entry.get('species', ''),
        }

        # For HMSD mode, include per-trace initial donor concentrations
        # These will be used to pre-populate the manual per-trace Z0 table
        initial_concs = entry.get('initial_concentrations', {})
        if initial_concs:
            metadata['per_trace_Z0'] = initial_concs
            logger.info(
                f"Including {len(initial_concs)} initial concentrations in metadata"
            )

        self.forwarded_to_kinetics.emit(df_for_kinetics, label, metadata)

        logger.info(
            f"Forwarded dataset '{label}' to Kinetics Processor "
            f"({df_for_kinetics.shape[0]} rows x {df_for_kinetics.shape[1]} cols)"
        )

        if self.parent_tab_widget:
            kinetics_index = self._find_kinetics_tab_index()
            if kinetics_index >= 0:
                self.parent_tab_widget.setCurrentIndex(kinetics_index)

        # Show appropriate message based on whether initial concentrations are included
        if initial_concs:
            QMessageBox.information(
                self,
                "Data Forwarded",
                f"Dataset '{label}' has been sent to the Kinetics Processor.\n\n"
                f"Initial donor concentrations ([S1-T]_0) for {len(initial_concs)} traces "
                f"have been pre-populated in manual per-trace mode.\n\n"
                f"Switched to Kinetics tab for immediate analysis."
            )
        else:
            QMessageBox.information(
                self,
                "Data Forwarded",
                f"Dataset '{label}' has been sent to the Kinetics Processor.\n\n"
                f"Switched to Kinetics tab for immediate analysis."
            )

    def _find_kinetics_tab_index(self) -> int:
        """Find the index of the Kinetics Processor tab."""
        if not self.parent_tab_widget:
            return -1

        for i in range(self.parent_tab_widget.count()):
            if "Kinetics" in self.parent_tab_widget.tabText(i):
                return i
        return -1
