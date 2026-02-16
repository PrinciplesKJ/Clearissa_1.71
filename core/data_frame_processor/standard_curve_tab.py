"""
Clearissa - standard_curve_tab.py
----------------------------------
Standard curve generation and calibration management.

This module provides:
- Interactive calibration data entry and management
- Zero-intercept linear regression fitting for standard curves
- Blank subtraction
- Species classification (donor, acceptor, FRET, blocked)
- Calibration data persistence and export
- High-resolution plot export

Note: Calibration uses zero-intercept regression (y = mx), which is
appropriate for fluorescence measurements where zero concentration
should correspond to zero signal after blank subtraction.

Author: Križan Jurinović
Date: October 2025
"""

import os
import sys
import re
import json
import base64
from datetime import datetime

# Add resource_utils import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resource_utils import get_data_path

import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter
from scipy.stats import linregress

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtSvg import QSvgRenderer
from PyQt5.QtGui import QDoubleValidator, QColor, QBrush
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QFrame, QGroupBox, QListWidgetItem,
    QPushButton, QTextEdit, QLabel, QLineEdit, QListWidget, QMessageBox,
    QDialog, QGridLayout, QFormLayout, QDialogButtonBox, QFileDialog
)
from core.common.data_processing_utils import remove_empty_well_columns
from core.common.plot_style import configure_pyqtgraph_widget, get_trace_color, get_concentration_label
from core.common.ui_theme import Colors

import logging

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------
WELL_COLUMN_INDEX = 0
TIME_COLUMN_INDEX = 1
DATA_START_COLUMN_INDEX = 2

MIN_CALIBRATION_POINTS = 3
DEFAULT_CONCENTRATION = "10"

EXPORT_IMAGE_WIDTH = 3000
EXPORT_IMAGE_HEIGHT = 2000

# High-resolution figure export settings
PUB_FIGURE_WIDTH_MM = 85  # mm
PUB_FIGURE_HEIGHT_MM = 70  # mm
PUB_FIGURE_DPI = 600
PUB_FONT_FAMILY = 'sans-serif'  # Helvetica/Arial
PUB_FONT_SIZE_AXIS_LABEL = 8  # pt
PUB_FONT_SIZE_TICK_LABEL = 7  # pt
PUB_FONT_SIZE_EQUATION = 7  # pt
PUB_LINE_WIDTH = 0.6  # pt
PUB_REGRESSION_LINE_WIDTH = 0.7  # pt
PUB_MARKER_SIZE = 2.0  # pt (radius)
PUB_MARKER_COLOR = '#2C2C2C'  # Dark grey/black
PUB_REGRESSION_COLOR = '#000000'  # Black

# -------------------------------------------------------------------
# Classification helpers
# -------------------------------------------------------------------
CATEGORY_COLOURS = {
    "DONOR": QColor(0, 128, 0),  # green
    "ACCEPTOR": QColor(255, 140, 0),  # orange
    "FRET": QColor(0, 102, 204),  # blue
    "BLOCKED": QColor(150, 75, 0),  # brown
}


def classify_entry(species: str, comments: str) -> str:
    """
    Classify calibration entry by category.

    Parameters
    ----------
    species : str
        Species name or description.
    comments : str
        Additional comments or notes.

    Returns
    -------
    str
        Category label: DONOR, ACCEPTOR, FRET, BLOCKED, or UNKNOWN.

    Notes
    -----
    Classification is case-insensitive and searches both species
    and comments fields for keywords.
    """
    text = f"{species or ''} {comments or ''}".lower()

    if "fret" in text:
        return "FRET"
    if "donor" in text:
        return "DONOR"
    if "acceptor" in text:
        return "ACCEPTOR"
    if any(k in text for k in ["blocked", "block", "quenched", "quench", "dark"]):
        return "BLOCKED"

    return "UNKNOWN"


# -------------------------------------------------------------------
# Calibration Data Manager
# -------------------------------------------------------------------
def load_calibration_data(filename):
    """Load calibration data from JSON, returning a list of calibration entries."""
    if not os.path.exists(filename):
        logger.warning(f"Calibration data file '{filename}' not found. Returning empty list.")
        return []
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError as e:
        logger.warning(f"Could not decode JSON in '{filename}': {e}")
        return []


def save_calibration_data(filename, data):
    """Save the list of calibration dictionaries to JSON."""
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Calibration data saved successfully to '{filename}'")
    except Exception as e:
        logger.error(f"Failed to save data to '{filename}': {e}")
        raise


# -------------------------------------------------------------------
# Calibration Utilities
# -------------------------------------------------------------------
def subtract_blank(sample_df, blank_ctrl_df, ctrl_col_index=DATA_START_COLUMN_INDEX):
    """
    Subtracts blank data from the sample DataFrame.
    If unavailable or index out of range, returns the original sample DataFrame.
    """
    if blank_ctrl_df is None:
        logger.warning("Blank data not provided. Skipping subtraction.")
        return sample_df

    if sample_df.shape[1] <= ctrl_col_index or blank_ctrl_df.shape[1] <= ctrl_col_index:
        logger.warning("Blank column index out of range. Skipping subtraction.")
        return sample_df

    blank_column = blank_ctrl_df.iloc[:, ctrl_col_index]
    df_copy = sample_df.copy()
    df_copy.iloc[:, ctrl_col_index:] = df_copy.iloc[:, ctrl_col_index:].sub(blank_column, axis=0)
    return df_copy


def calculate_regression(concentrations, afu_values):
    """
    Performs a linear regression of AFU vs. concentrations (with intercept).
    Returns slope, intercept, r_value, p_value, std_err.

    Note: This is kept for reference/legacy purposes. The primary calibration
    now uses zero-intercept regression via calculate_zero_intercept_slope().
    """
    if len(concentrations) != len(afu_values):
        raise ValueError("Mismatch in number of concentrations vs. A.F.U. values.")

    conc = np.array(concentrations, dtype=float)
    afu = np.array(afu_values, dtype=float)
    slope, intercept, r_value, p_value, std_err = linregress(conc, afu)
    return slope, intercept, r_value, p_value, std_err


def calculate_zero_intercept_slope(concentrations, afu_values):
    """
    Calculate slope for regression forced through origin (zero intercept).
    This is the primary calibration method.

    Parameters
    ----------
    concentrations : array-like
        Concentration values
    afu_values : array-like
        AFU values (raw, not adjusted)

    Returns
    -------
    tuple
        (slope, r_squared, std_err) for zero-intercept regression

    Notes
    -----
    For zero-intercept regression:
    - slope = Σ(x*y) / Σ(x²)
    - R² = 1 - SS_res / SS_tot where SS_tot = Σ(y²) for zero-intercept models
    - std_err = sqrt(SS_res / (n-1)) / sqrt(Σ(x²))
    """
    conc = np.array(concentrations, dtype=float)
    afu = np.array(afu_values, dtype=float)
    n = len(conc)

    # Calculate slope for regression through origin: slope = sum(x*y) / sum(x^2)
    sum_x2 = np.sum(conc ** 2)
    slope = np.sum(conc * afu) / sum_x2

    # Calculate R-squared for regression through origin
    y_pred = slope * conc
    ss_res = np.sum((afu - y_pred) ** 2)
    ss_tot = np.sum(afu ** 2)  # For zero-intercept, total sum of squares is sum(y^2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Calculate standard error of slope
    # For zero-intercept: std_err = sqrt(SS_res / (n-1)) / sqrt(Σ(x²))
    if n > 1:
        mse = ss_res / (n - 1)
        std_err = np.sqrt(mse / sum_x2)
    else:
        std_err = 0.0

    return slope, r_squared, std_err



def sanitise_for_windows(name):
    """Replace Windows-illegal filename characters with underscores."""
    if not name.strip():
        return "Unnamed_Species"
    return re.sub(r'[<>:"/\\|?*]+', '_', name.strip())


# -------------------------------------------------------------------
# Main Widget: StandardCurveTab
# -------------------------------------------------------------------
class StandardCurveTab(QWidget):
    """
    A tab for creating/viewing a standard calibration curve with split view.
    """
    saved_concentration_values = {}

    def __init__(self, parent, selected_data, blank_ctrl_data):
        super().__init__(parent)
        self.parent = parent
        self.selected_data = selected_data
        self.blank_ctrl_data = blank_ctrl_data

        self.selected_data = remove_empty_well_columns(self.selected_data)

        self.standard_curve_entry_values = {}
        self.standard_curve_objects = None
        self.calibration_data_filename = get_data_path("config/calibration_data.json")

        self._init_styles()
        self.setup_ui()
        self.populate_existing_calibration_data()

    def _init_styles(self):
        """Define basic style sheets."""
        self.label_style = """
            QLabel {
                background-color: {Colors.SECTION_BACKGROUND};
                border: none;
                border-radius: 5px;
                padding: 5px;
            }
        """
        self.entry_label_style = self.label_style + "font-weight: bold; font-size: 12px;"
        self.frame_style = """
            QFrame {
                border: 1px solid {Colors.BORDER_LIGHT};
                border-radius: 5px;
                background-color: {Colors.SECTION_BACKGROUND};
                font-size: 12px;
            }
        """
        self.error_entry_style = """
            QLineEdit {
                border: 2px solid red;
                background-color: #FFE6E6;
            }
        """

    def setup_ui(self):
        """Set up the overall layout with split plot view."""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Left column - Input controls
        left_frame = QFrame()
        left_frame.setFrameShape(QFrame.StyledPanel)
        left_layout = QVBoxLayout(left_frame)
        left_layout.setSpacing(5)

        species_comments_box = self._create_species_comments_form()
        left_layout.addWidget(species_comments_box)

        concentration_box = self._create_concentration_entries()
        left_layout.addWidget(concentration_box)

        button_frame = self._create_button_panel()
        left_layout.addWidget(button_frame)

        existing_calibration_group = self._create_calibration_list()
        left_layout.addWidget(existing_calibration_group)

        # Right column - Split plot view
        right_frame = QFrame()
        right_frame.setFrameShape(QFrame.StyledPanel)
        right_layout = QVBoxLayout(right_frame)

        # Split plot area with equal sizing
        plot_splitter = QHBoxLayout()
        plot_splitter.setSpacing(10)

        # Current calibration section (left side)
        current_section = QFrame()
        current_section.setFrameShape(QFrame.StyledPanel)
        current_section_layout = QVBoxLayout(current_section)
        current_section_layout.setContentsMargins(5, 5, 5, 5)
        current_section_layout.setSpacing(5)

        current_plot_group = QGroupBox("Current Calibration")
        current_plot_group.setStyleSheet(self.frame_style)
        current_plot_layout = QVBoxLayout(current_plot_group)
        current_plot_layout.setContentsMargins(5, 5, 5, 5)
        self.current_calibration_plot = pg.PlotWidget()
        self._setup_plot_widget(self.current_calibration_plot, "Create a calibration to view plot")
        self.current_calibration_plot.setMinimumHeight(450)
        current_plot_layout.addWidget(self.current_calibration_plot)

        # Export button for current calibration
        self.export_current_button = QPushButton("Export Publication Figure")
        self.export_current_button.setEnabled(False)
        self.export_current_button.clicked.connect(self.export_current_calibration_figure)
        current_plot_layout.addWidget(self.export_current_button)

        current_section_layout.addWidget(current_plot_group, 3)  # Give more weight to plot

        # Current calibration details
        current_details_group = QGroupBox("Current Details")
        current_details_group.setStyleSheet(self.frame_style)
        current_details_layout = QVBoxLayout(current_details_group)
        current_details_layout.setContentsMargins(5, 5, 5, 5)
        self.current_calibration_details = QTextEdit()
        self.current_calibration_details.setReadOnly(True)
        self.current_calibration_details.setMaximumHeight(150)
        self.current_calibration_details.setMinimumHeight(100)
        self.current_calibration_details.setPlaceholderText("Calibration details will appear here...")
        current_details_layout.addWidget(self.current_calibration_details)
        current_section_layout.addWidget(current_details_group, 1)  # Less weight than plot

        plot_splitter.addWidget(current_section, 1)  # 50% width

        # Historical calibration section (right side)
        historical_section = QFrame()
        historical_section.setFrameShape(QFrame.StyledPanel)
        historical_section_layout = QVBoxLayout(historical_section)
        historical_section_layout.setContentsMargins(5, 5, 5, 5)
        historical_section_layout.setSpacing(5)

        historical_plot_group = QGroupBox("Historical Calibration")
        historical_plot_group.setStyleSheet(self.frame_style)
        historical_plot_layout = QVBoxLayout(historical_plot_group)
        historical_plot_layout.setContentsMargins(5, 5, 5, 5)
        self.historical_calibration_plot = pg.PlotWidget()
        self._setup_plot_widget(self.historical_calibration_plot, "Select a calibration to view")
        self.historical_calibration_plot.setMinimumHeight(450)
        historical_plot_layout.addWidget(self.historical_calibration_plot)

        # Export button for historical calibration
        self.export_historical_button = QPushButton("Export Publication Figure")
        self.export_historical_button.setEnabled(False)
        self.export_historical_button.clicked.connect(self.export_historical_calibration_figure)
        historical_plot_layout.addWidget(self.export_historical_button)

        historical_section_layout.addWidget(historical_plot_group, 3)  # Give more weight to plot

        # Historical calibration details
        historical_details_group = QGroupBox("Historical Details")
        historical_details_group.setStyleSheet(self.frame_style)
        historical_details_layout = QVBoxLayout(historical_details_group)
        historical_details_layout.setContentsMargins(5, 5, 5, 5)
        self.historical_calibration_details = QTextEdit()
        self.historical_calibration_details.setReadOnly(True)
        self.historical_calibration_details.setMaximumHeight(150)
        self.historical_calibration_details.setMinimumHeight(100)
        self.historical_calibration_details.setPlaceholderText("Select a calibration to view details...")
        historical_details_layout.addWidget(self.historical_calibration_details)
        historical_section_layout.addWidget(historical_details_group, 1)  # Less weight than plot

        plot_splitter.addWidget(historical_section, 1)  # 50% width

        right_layout.addLayout(plot_splitter)

        main_layout.addWidget(left_frame, 1)
        main_layout.addWidget(right_frame, 3)

    def _setup_plot_widget(self, plot_widget, empty_message=""):
        """Configure a plot widget with Clearissa styling."""
        configure_pyqtgraph_widget(
            plot_widget,
            x_label=get_concentration_label(),
            y_label="Signal [AFU]",
            title=None,
            enable_grid=False,
            background='w'
        )
        if empty_message:
            text_item = pg.TextItem(empty_message, color='grey', anchor=(0.5, 0.5))
            plot_widget.addItem(text_item)
            text_item.setPos(0.5, 0.5)

    def _create_button_panel(self):
        """Create the button control panel."""
        button_frame = QFrame()
        button_layout = QVBoxLayout(button_frame)
        button_layout.setSpacing(5)

        calibrate_button = QPushButton("Create Standard Curve")
        calibrate_button.clicked.connect(self.perform_calibration_workflow)
        button_layout.addWidget(calibrate_button)

        self.save_calibration_button = QPushButton("Save Calibration")
        self.save_calibration_button.setEnabled(False)
        self.save_calibration_button.clicked.connect(self.save_calibration_workflow)
        button_layout.addWidget(self.save_calibration_button)

        delete_calibration_button = QPushButton("Delete Selected Calibration")
        delete_calibration_button.clicked.connect(self.delete_selected_calibration)
        button_layout.addWidget(delete_calibration_button)

        edit_datasets_button = QPushButton("Edit Datasets")
        edit_datasets_button.clicked.connect(self.edit_datasets)
        button_layout.addWidget(edit_datasets_button)

        reset_values_button = QPushButton("Reset to Default Values")
        reset_values_button.clicked.connect(self.reset_concentration_values)
        button_layout.addWidget(reset_values_button)

        return button_frame

    def _create_calibration_list(self):
        """Create the existing calibrations list widget."""
        existing_calibration_group = QGroupBox("Existing Calibrations")
        existing_calibration_group.setStyleSheet(self.frame_style)
        existing_calibration_layout = QVBoxLayout(existing_calibration_group)

        self.listbox_existing_calibration = QListWidget()
        self.listbox_existing_calibration.itemSelectionChanged.connect(self.view_selected_calibration)
        existing_calibration_layout.addWidget(self.listbox_existing_calibration)

        return existing_calibration_group

    # -----------------------------------------------------------------
    # UI Helpers
    # -----------------------------------------------------------------
    def _create_species_comments_form(self):
        """Group box containing species and comments fields."""
        group_box = QGroupBox("Species and Comments")
        group_box.setStyleSheet(self.frame_style)

        form_layout = QFormLayout()

        species_label = QLabel("Species:")
        species_label.setStyleSheet(self.entry_label_style)
        species_entry = QLineEdit()
        species_entry.setPlaceholderText("Enter species...")
        form_layout.addRow(species_label, species_entry)
        self.standard_curve_entry_values['species'] = species_entry

        comments_label = QLabel("Comments:")
        comments_label.setStyleSheet(self.entry_label_style)
        comments_entry = QLineEdit()
        comments_entry.setPlaceholderText("Enter comments...")
        form_layout.addRow(comments_label, comments_entry)
        self.standard_curve_entry_values['comments'] = comments_entry

        group_box.setLayout(form_layout)
        return group_box

    def _create_concentration_entries(self):
        """Group box with QLineEdits for each column in self.selected_data."""
        group_box = QGroupBox("Concentration")
        group_box.setStyleSheet(self.frame_style)

        grid_layout = QGridLayout()
        grid_layout.setSpacing(5)

        row, col = 0, 0
        max_cols = 2
        for col_name in self.selected_data.columns[DATA_START_COLUMN_INDEX:]:
            label = QLabel(f"Well {col_name}:")
            label.setStyleSheet(self.entry_label_style)
            grid_layout.addWidget(label, row, col)

            entry = QLineEdit()
            entry.setPlaceholderText("Enter concentration")
            default_value = StandardCurveTab.saved_concentration_values.get(col_name, DEFAULT_CONCENTRATION)
            entry.setText(default_value)
            entry.setValidator(QDoubleValidator(0.0, 1e9, 6))
            entry.textChanged.connect(lambda text, w=col_name: self.update_saved_concentration(w, text))
            grid_layout.addWidget(entry, row, col + 1)
            self.standard_curve_entry_values[col_name] = entry

            col += 2
            if col >= max_cols * 2:
                col = 0
                row += 1

        group_box.setLayout(grid_layout)
        return group_box

    def update_saved_concentration(self, well, text):
        """Keep track of user-entered concentration values."""
        StandardCurveTab.saved_concentration_values[well] = text

    def reset_concentration_values(self):
        """Reset all concentration entries to default."""
        for key, widget in self.standard_curve_entry_values.items():
            if key not in ['species', 'comments']:
                StandardCurveTab.saved_concentration_values[key] = DEFAULT_CONCENTRATION
                widget.setText(DEFAULT_CONCENTRATION)
                widget.setStyleSheet("")  # Clear any error styling
        QMessageBox.information(self, "Reset", "Concentration values have been reset to default.")

    # -----------------------------------------------------------------
    # Calibration Workflow
    # -----------------------------------------------------------------
    def perform_calibration_workflow(self):
        """
        Main calibration workflow: gather data, validate, calculate, plot.
        """
        logger.info("Performing calibration workflow...")

        wells = list(self.selected_data.columns[DATA_START_COLUMN_INDEX:])
        concentrations = []
        invalid_wells = []

        # 1) Gather and validate concentrations
        logger.debug("Reading user-entered concentrations...")
        for w in wells:
            entry_widget = self.standard_curve_entry_values[w]
            text_val = entry_widget.text()
            try:
                c_val = float(text_val)
                concentrations.append(c_val)
                entry_widget.setStyleSheet("")  # Clear error styling
            except ValueError:
                logger.warning(f"Invalid concentration input for well '{w}'. Value: '{text_val}'")
                entry_widget.setStyleSheet(self.error_entry_style)
                invalid_wells.append(w)

        if invalid_wells:
            QMessageBox.warning(
                self,
                "Invalid Input",
                f"Invalid concentration for wells: {', '.join(invalid_wells)}\n"
                f"Please enter valid numeric values."
            )
            return

        # 2) Subtract blank
        logger.debug("Subtracting blank...")
        adjusted_df = subtract_blank(
            self.selected_data,
            self.blank_ctrl_data,
            ctrl_col_index=DATA_START_COLUMN_INDEX
        )

        afu_values = adjusted_df.iloc[:, DATA_START_COLUMN_INDEX:].apply(
            pd.to_numeric, errors='coerce'
        ).mean().values

        # 3) Filter zero values and track which wells
        logger.debug("Checking for zero A.F.U. values...")
        filtered_data = []
        removed_wells = []

        for i, (well, conc, afu) in enumerate(zip(wells, concentrations, afu_values)):
            if afu == 0:
                removed_wells.append(well)
                logger.warning(f"Zero A.F.U. value detected for well '{well}'. Omitting from calibration.")
            else:
                filtered_data.append((conc, afu))

        # Show which wells were removed
        if removed_wells:
            QMessageBox.information(
                self,
                "Zero Values Detected",
                f"The following wells had zero A.F.U. values and were omitted:\n"
                f"{', '.join(removed_wells)}\n\n"
                f"Remaining data points: {len(filtered_data)}"
            )

        # 4) Check minimum data points
        if len(filtered_data) < MIN_CALIBRATION_POINTS:
            logger.error(f"Insufficient data points: {len(filtered_data)} (minimum {MIN_CALIBRATION_POINTS} required)")
            QMessageBox.critical(
                self,
                "Insufficient Data",
                f"Only {len(filtered_data)} valid data points remain after filtering.\n"
                f"A minimum of {MIN_CALIBRATION_POINTS} points is required for calibration."
            )
            return

        # Separate concentrations and AFU values
        filtered_concentrations = [x[0] for x in filtered_data]
        filtered_afu_values = [x[1] for x in filtered_data]

        # 5) Calculate zero-intercept regression (primary method)
        try:
            slope, r_squared, std_err = calculate_zero_intercept_slope(
                filtered_concentrations, filtered_afu_values
            )
            logger.info("Zero-intercept regression calculation successful.")
        except Exception as e:
            logger.exception("Error calculating zero-intercept regression.")
            QMessageBox.critical(self, "Regression Error", str(e))
            return

        # 6) Update current calibration plot
        species = self.standard_curve_entry_values['species'].text()
        logger.debug("Updating calibration plot...")
        self._update_current_plot(
            filtered_concentrations,
            filtered_afu_values,
            slope,
            species
        )

        # 7) Store results and update current details
        comments = self.standard_curve_entry_values['comments'].text()
        result_text = (
            f"Species: {species}\n"
            f"Comments: {comments}\n"
            f"Slope (zero-intercept): {slope:.2f}\n"
            f"R-squared: {r_squared:.4f}\n"
            f"Std Error: {std_err:.2f}\n"
            f"Data Points: {len(filtered_data)}"
        )
        self.current_calibration_details.setPlainText(result_text)

        self.standard_curve_objects = {
            'slope': slope,
            'intercept': 0.0,  # Zero intercept
            'r_squared': r_squared,
            'std_err': std_err,
            'species': species,
            'comments': comments,
            'concentrations': filtered_concentrations,
            'afu_values': filtered_afu_values
        }

        # 8) Enable save button if species provided
        if species.strip():
            self.save_calibration_button.setEnabled(True)
        logger.info("Calibration workflow complete.")

    def _update_current_plot(self, concentrations, afu_values, slope, species=""):
        """
        Update the current calibration plot with zero-intercept regression.
        Clears and redraws rather than destroying widget.
        """
        # Clear existing plot
        self.current_calibration_plot.clear()

        # Re-apply plot configuration
        title = species if species.strip() else "Standard Curve"
        configure_pyqtgraph_widget(
            self.current_calibration_plot,
            x_label=get_concentration_label(),
            y_label="Signal [AFU]",
            title=title,
            enable_grid=False,
            background='w'
        )

        # Plot scatter points (navy colour, plus symbols)
        scatter = pg.ScatterPlotItem(
            x=concentrations,
            y=afu_values,
            symbol='+',
            size=8,
            pen=pg.mkPen(get_trace_color(0), width=1.5),
            brush=pg.mkBrush(get_trace_color(0))
        )
        self.current_calibration_plot.addItem(scatter)

        # Plot regression line through origin
        x_vals = np.array(concentrations)
        x_line = np.linspace(0, x_vals.max(), 100)
        y_line = slope * x_line  # Zero-intercept regression
        regression_line = pg.PlotDataItem(
            x=x_line,
            y=y_line,
            pen=pg.mkPen('k', width=2.0, style=Qt.DashLine)  # Dashed line for fitted model
        )
        self.current_calibration_plot.addItem(regression_line)

        # Set axes to start at zero
        self.current_calibration_plot.setXRange(0, max(concentrations) * 1.1, padding=0)
        self.current_calibration_plot.setYRange(0, max(afu_values) * 1.1, padding=0)

        # Enable export button
        self.export_current_button.setEnabled(True)


    def _update_historical_plot(self, calibration):
        """
        Update the historical calibration plot from saved calibration data.
        Uses zero-intercept regression for display.
        """
        # Clear existing plot
        self.historical_calibration_plot.clear()

        # Get species name for title
        species = calibration.get('species', 'Historical Calibration')

        # Re-apply plot configuration
        configure_pyqtgraph_widget(
            self.historical_calibration_plot,
            x_label=get_concentration_label(),
            y_label="Signal [AFU]",
            title=species,
            enable_grid=False,
            background='w'
        )

        # Get original data
        x_scatter = calibration.get('concentrations', [])
        y_scatter = calibration.get('afu_values', [])

        # Use stored zero_intercept_slope if available, otherwise calculate
        if 'zero_intercept_slope' in calibration:
            display_slope = calibration['zero_intercept_slope']
        elif x_scatter and y_scatter:
            display_slope, _, _ = calculate_zero_intercept_slope(x_scatter, y_scatter)
        else:
            display_slope = calibration.get('slope', 0)

        # Generate regression line through origin
        if x_scatter:
            x_scatter_array = np.array(x_scatter)
            x_line = np.linspace(0, x_scatter_array.max(), 100)
            y_line = display_slope * x_line
        else:
            x_line, y_line = [], []

        # Plot scatter points (navy colour, plus symbols)
        if x_scatter and y_scatter:
            scatter = pg.ScatterPlotItem(
                x=x_scatter,
                y=y_scatter,
                symbol='+',
                size=8,
                pen=pg.mkPen(get_trace_color(0), width=1.5),
                brush=pg.mkBrush(get_trace_color(0))
            )
            self.historical_calibration_plot.addItem(scatter)

        # Plot regression line
        if x_line is not None and len(x_line) > 0:
            regression_line = pg.PlotDataItem(
                x=x_line,
                y=y_line,
                pen=pg.mkPen('k', width=2.0, style=Qt.DashLine)  # Dashed line for fitted model
            )
            self.historical_calibration_plot.addItem(regression_line)

        # Set axes to start at zero
        if x_scatter and y_scatter:
            self.historical_calibration_plot.setXRange(0, max(x_scatter) * 1.1, padding=0)
            self.historical_calibration_plot.setYRange(0, max(y_scatter) * 1.1, padding=0)

        # Enable export button
        self.export_historical_button.setEnabled(True)


    def save_calibration_workflow(self):
        """Save calibration with plot data embedded in JSON."""
        if not self.standard_curve_objects:
            QMessageBox.warning(self, "No Data", "No calibration data to save.")
            return

        logger.info("Saving calibration data...")
        data_obj = self.standard_curve_objects
        species = data_obj['species'].strip()
        if not species:
            QMessageBox.warning(self, "Error", "Species name is required to save calibration.")
            return

        # Create species folder
        folder_name = sanitise_for_windows(species)
        os.makedirs(folder_name, exist_ok=True)

        # Generate filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"{species}_{timestamp}.csv"
        png_filename = f"{species}_{timestamp}.png"

        csv_path = os.path.join(folder_name, csv_filename)
        png_path = os.path.join(folder_name, png_filename)

        # Save CSV
        df = pd.DataFrame({
            'Concentration': data_obj['concentrations'],
            'A.F.U.': data_obj['afu_values']
        })
        df.to_csv(csv_path, index=False)
        logger.info(f"Data saved as CSV: {csv_path}")

        # Export high-quality PNG
        exporter = ImageExporter(self.current_calibration_plot.plotItem)
        exporter.parameters()['width'] = EXPORT_IMAGE_WIDTH
        exporter.parameters()['height'] = EXPORT_IMAGE_HEIGHT
        exporter.export(png_path)
        logger.info(f"High-quality graph saved as PNG: {png_path}")

        # Prepare plot data for embedding
        concentrations = data_obj['concentrations']
        afu_values = data_obj['afu_values']
        slope = data_obj['slope']

        # Create scatter points list
        scatter_points = [[float(c), float(a)] for c, a in zip(concentrations, afu_values)]

        # Create regression line through origin (just endpoints for minimal storage)
        x_min, x_max = min(concentrations), max(concentrations)
        regression_line = [
            [float(x_min), float(slope * x_min)],
            [float(x_max), float(slope * x_max)]
        ]

        # Prepare JSON entry with plot data (zero-intercept calibration)
        calibration_entry = {
            'concentrations': [round(c, 4) for c in concentrations],
            'afu_values': [round(a, 4) for a in afu_values],
            'slope': round(slope, 4),
            'intercept': 0.0,  # Zero intercept
            'zero_intercept_slope': round(slope, 4),
            'r_squared': round(data_obj['r_squared'], 4),
            'zero_intercept_r_squared': round(data_obj['r_squared'], 4),
            'std_err': round(data_obj['std_err'], 4),
            'species': data_obj['species'],
            'comments': data_obj['comments'],
            'plot_data': {
                'scatter_points': scatter_points,
                'regression_line': regression_line
            }
        }

        # Update calibration_data.json
        cal_data = load_calibration_data(self.calibration_data_filename)
        cal_data.append(calibration_entry)
        save_calibration_data(self.calibration_data_filename, cal_data)

        QMessageBox.information(
            self,
            "Save Successful",
            f"Calibration saved successfully!\n\n"
            f"Files saved in: {folder_name}/\n"
            f"- {csv_filename}\n"
            f"- {png_filename}"
        )

        # Refresh calibration list
        self.populate_existing_calibration_data()

        # Open folder (Windows only)
        try:
            os.startfile(folder_name)
        except Exception as e:
            logger.warning(f"Could not open folder automatically: {e}")

    # -----------------------------------------------------------------
    # Existing Calibrations
    # -----------------------------------------------------------------
    def populate_existing_calibration_data(self):
        """Load and display existing calibrations in list, sorted and coloured."""
        cal_list = load_calibration_data(self.calibration_data_filename)
        self.listbox_existing_calibration.clear()

        # Build (original_index, entry) pairs and sort by species
        indexed = list(enumerate(cal_list))
        indexed_sorted = sorted(indexed, key=lambda p: (p[1].get('species', '') or '').lower())

        for orig_idx, entry in indexed_sorted:
            species = entry.get('species', 'Unknown Species') or 'Unknown Species'
            comments = entry.get('comments', '') or ''
            category = classify_entry(species, comments)

            item = QListWidgetItem(species)

            # Bold font for visibility
            bold_font = QtGui.QFont()
            bold_font.setBold(True)
            item.setFont(bold_font)

            # Store original index
            item.setData(QtCore.Qt.UserRole, orig_idx)

            # Colour by category
            if category in CATEGORY_COLOURS:
                item.setForeground(QBrush(CATEGORY_COLOURS[category]))

            item.setToolTip(f"Category: {category}")

            self.listbox_existing_calibration.addItem(item)

    def view_selected_calibration(self):
        """Display details and plot of selected calibration."""
        selected_items = self.listbox_existing_calibration.selectedItems()
        if not selected_items:
            logger.warning("No valid calibration selected.")
            self.historical_calibration_details.clear()
            self.historical_calibration_plot.clear()
            return

        cal_list = load_calibration_data(self.calibration_data_filename)
        orig_idx = selected_items[0].data(QtCore.Qt.UserRole)
        if orig_idx is None or not (0 <= orig_idx < len(cal_list)):
            logger.warning("Stored index invalid for selected calibration.")
            self.historical_calibration_details.clear()
            self.historical_calibration_plot.clear()
            return

        calibration = cal_list[orig_idx]
        self.populate_historical_details(calibration)
        self._update_historical_plot(calibration)

    def populate_historical_details(self, calibration):
        """Show selected calibration's details in historical panel."""
        # Prefer zero_intercept_slope if available
        if 'zero_intercept_slope' in calibration:
            slope_value = calibration['zero_intercept_slope']
            r_squared_value = calibration.get('zero_intercept_r_squared', calibration.get('r_squared', 'N/A'))
        else:
            slope_value = calibration.get('slope', 'N/A')
            r_squared_value = calibration.get('r_squared', 'N/A')

        details = (
            f"Species: {calibration.get('species', 'N/A')}\n"
            f"Comments: {calibration.get('comments', 'N/A')}\n"
            f"Slope (zero-intercept): {slope_value}\n"
            f"R-squared: {r_squared_value}\n"
            f"Std Error: {calibration.get('std_err', 'N/A')}\n"
            f"Data Points: {len(calibration.get('concentrations', []))}"
        )
        self.historical_calibration_details.setPlainText(details)

    def delete_selected_calibration(self):
        """Remove selected calibration from JSON with confirmation."""
        selected_items = self.listbox_existing_calibration.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Error", "No calibration selected.")
            return

        cal_list = load_calibration_data(self.calibration_data_filename)
        orig_idx = selected_items[0].data(QtCore.Qt.UserRole)
        if orig_idx is None or not (0 <= orig_idx < len(cal_list)):
            QMessageBox.warning(self, "Error", "Invalid selection.")
            return

        selected_cal = cal_list[orig_idx]
        species_name = selected_cal.get('species', 'Unknown')

        confirm = QMessageBox.question(
            self,
            "Delete Confirmation",
            f"Are you sure you want to delete '{species_name}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        if confirm == QMessageBox.Yes:
            updated_data = [c for i, c in enumerate(cal_list) if i != orig_idx]
            save_calibration_data(self.calibration_data_filename, updated_data)
            logger.info(f"Calibration entry for species '{species_name}' deleted.")
            self.populate_existing_calibration_data()
            self.historical_calibration_details.clear()
            self.historical_calibration_plot.clear()

    def edit_datasets(self):
        """Open dialog to edit selected calibration entry."""
        selected_items = self.listbox_existing_calibration.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Error", "No calibration selected.")
            return

        cal_list = load_calibration_data(self.calibration_data_filename)
        orig_idx = selected_items[0].data(QtCore.Qt.UserRole)
        if orig_idx is None or not (0 <= orig_idx < len(cal_list)):
            QMessageBox.warning(self, "Error", "Invalid selection.")
            return

        selected_calibration = cal_list[orig_idx]
        edit_dialog = EditDatasetsDialog(self, selected_calibration)
        if edit_dialog.exec_() == QDialog.Accepted:
            updated_calibration = edit_dialog.get_updated_calibration()

            # Preserve plot_data if it exists in original
            if 'plot_data' in selected_calibration and 'plot_data' not in updated_calibration:
                updated_calibration['plot_data'] = selected_calibration['plot_data']

            cal_list[orig_idx] = updated_calibration
            save_calibration_data(self.calibration_data_filename, cal_list)
            logger.info(f"Calibration entry for species '{updated_calibration.get('species')}' updated.")
            self.populate_existing_calibration_data()
            self.view_selected_calibration()

    # -----------------------------------------------------------------
    # High-Resolution Figure Export
    # -----------------------------------------------------------------
    def export_current_calibration_figure(self):
        """Export current calibration as a high-resolution figure."""
        if not self.standard_curve_objects:
            QMessageBox.warning(self, "No Data", "No calibration data to export.")
            return

        # Get data from current calibration
        concentrations = self.standard_curve_objects['concentrations']
        afu_values = self.standard_curve_objects['afu_values']
        slope = self.standard_curve_objects['slope']
        r_squared = self.standard_curve_objects['r_squared']
        species = self.standard_curve_objects.get('species', 'Calibration')

        self._export_publication_figure(
            concentrations, afu_values, slope, r_squared, species
        )

    def export_historical_calibration_figure(self):
        """Export historical calibration as a high-resolution figure."""
        selected_items = self.listbox_existing_calibration.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select a calibration to export.")
            return

        cal_list = load_calibration_data(self.calibration_data_filename)
        orig_idx = selected_items[0].data(QtCore.Qt.UserRole)

        if orig_idx is None or not (0 <= orig_idx < len(cal_list)):
            QMessageBox.warning(self, "Error", "Invalid selection.")
            return

        calibration = cal_list[orig_idx]

        # Extract data
        concentrations = calibration.get('concentrations', [])
        afu_values = calibration.get('afu_values', [])
        species = calibration.get('species', 'Calibration')

        if not concentrations or not afu_values:
            QMessageBox.warning(self, "No Data", "Selected calibration has no data to export.")
            return

        # Use zero_intercept_slope if available, otherwise calculate
        if 'zero_intercept_slope' in calibration:
            slope = calibration['zero_intercept_slope']
            r_squared = calibration.get('zero_intercept_r_squared', calibration.get('r_squared', 0))
        else:
            slope, r_squared, _ = calculate_zero_intercept_slope(concentrations, afu_values)

        self._export_publication_figure(
            concentrations, afu_values, slope, r_squared, species
        )

    def _export_publication_figure(self, concentrations, afu_values, slope, r_squared, species):
        """
        Create and export a high-resolution calibration figure with zero-intercept regression.

        Specifications:
        - Canvas: 85mm × 70mm @ 600 DPI
        - Font: Sans-serif, 8pt labels, 7pt ticks/equation
        - Line thickness: 0.6-0.7pt
        - Markers: 2.5pt solid circles, dark grey/black
        - White background, no bounding box
        """
        # Prompt user for save location
        default_filename = f"{sanitise_for_windows(species)}_calibration.png"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Publication Figure",
            default_filename,
            "PNG Image (*.png);;PDF Document (*.pdf);;SVG Vector (*.svg);;All Files (*)"
        )

        if not file_path:
            return  # User cancelled

        # Determine file format from extension
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in ['.png', '.pdf', '.svg']:
            file_ext = '.png'
            file_path += '.png'

        # Configure matplotlib for high-resolution output
        plt.rcParams.update({
            'font.family': PUB_FONT_FAMILY,
            'font.size': PUB_FONT_SIZE_TICK_LABEL,
            'axes.labelsize': PUB_FONT_SIZE_AXIS_LABEL,
            'axes.linewidth': PUB_LINE_WIDTH,
            'xtick.major.width': PUB_LINE_WIDTH,
            'ytick.major.width': PUB_LINE_WIDTH,
            'xtick.labelsize': PUB_FONT_SIZE_TICK_LABEL,
            'ytick.labelsize': PUB_FONT_SIZE_TICK_LABEL,
            'lines.linewidth': PUB_REGRESSION_LINE_WIDTH,
            'lines.markersize': PUB_MARKER_SIZE * 2,  # matplotlib uses diameter
            'savefig.dpi': PUB_FIGURE_DPI,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.05,
        })

        # Create figure with exact dimensions
        fig_width_inches = PUB_FIGURE_WIDTH_MM / 25.4  # Convert mm to inches
        fig_height_inches = PUB_FIGURE_HEIGHT_MM / 25.4

        fig, ax = plt.subplots(figsize=(fig_width_inches, fig_height_inches))

        # Plot data points with plus symbols
        ax.scatter(
            concentrations,
            afu_values,
            s=(PUB_MARKER_SIZE * 2) ** 2,  # s is area in points^2
            color=PUB_MARKER_COLOR,
            marker='+',
            edgecolors='none',
            zorder=3
        )

        # Plot regression line through origin
        x_vals = np.array(concentrations)
        x_line = np.linspace(0, x_vals.max(), 100)
        y_line = slope * x_line  # Zero-intercept regression

        ax.plot(
            x_line,
            y_line,
            color=PUB_REGRESSION_COLOR,
            linewidth=PUB_REGRESSION_LINE_WIDTH,
            linestyle='--',  # Dashed line for fitted model
            zorder=2
        )

        # Add equation and R² to plot (top-left corner)
        equation_text = f'y = {slope:.2f}x\nR² = {r_squared:.4f}'

        # Position text in top-left corner
        ax.text(
            0.05, 0.95,
            equation_text,
            transform=ax.transAxes,
            fontsize=PUB_FONT_SIZE_EQUATION,
            verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='none')
        )

        # Set labels
        ax.set_xlabel(get_concentration_label(), fontsize=PUB_FONT_SIZE_AXIS_LABEL)
        ax.set_ylabel('Signal [AFU]', fontsize=PUB_FONT_SIZE_AXIS_LABEL)

        # Set axes to start at zero
        ax.set_xlim(0, max(concentrations) * 1.1)
        ax.set_ylim(0, max(afu_values) * 1.1)


        # White background
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')

        # Remove top and right spines (optional - cleaner look)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Ensure axis lines are proper width
        for spine in ax.spines.values():
            spine.set_linewidth(PUB_LINE_WIDTH)

        # Tight layout to minimize margins
        plt.tight_layout()

        # Save figure
        try:
            if file_ext == '.png':
                plt.savefig(file_path, dpi=PUB_FIGURE_DPI, format='png', facecolor='white')
            elif file_ext == '.pdf':
                plt.savefig(file_path, format='pdf', facecolor='white')
            elif file_ext == '.svg':
                plt.savefig(file_path, format='svg', facecolor='white')

            plt.close(fig)

            QMessageBox.information(
                self,
                "Export Successful",
                f"Figure saved to:\n{file_path}\n\n"
                f"Specifications:\n"
                f"- Size: {PUB_FIGURE_WIDTH_MM}mm × {PUB_FIGURE_HEIGHT_MM}mm\n"
                f"- Resolution: {PUB_FIGURE_DPI} DPI\n"
                f"- Format: {file_ext.upper()}"
            )
            logger.info(f"Publication figure exported: {file_path}")

        except Exception as e:
            plt.close(fig)
            QMessageBox.critical(
                self,
                "Export Failed",
                f"Failed to export figure:\n{str(e)}"
            )
            logger.error(f"Export failed: {e}", exc_info=True)


# -------------------------------------------------------------------
# Edit Datasets Dialog
# -------------------------------------------------------------------
class EditDatasetsDialog(QDialog):
    """Dialog to edit an existing calibration entry with validation."""

    def __init__(self, parent, calibration):
        super().__init__(parent)
        self.setWindowTitle("Edit Datasets")
        self.calibration = calibration

        self.error_style = """
            QLineEdit {
                border: 2px solid red;
                background-color: #FFE6E6;
            }
        """

        layout = QVBoxLayout(self)

        # Species / Comments
        species_comments_layout = QHBoxLayout()
        species_label = QLabel("Species:")
        self.species_entry = QLineEdit(calibration.get('species', ''))
        comments_label = QLabel("Comments:")
        self.comments_entry = QLineEdit(calibration.get('comments', ''))
        species_comments_layout.addWidget(species_label)
        species_comments_layout.addWidget(self.species_entry)
        species_comments_layout.addWidget(comments_label)
        species_comments_layout.addWidget(self.comments_entry)
        layout.addLayout(species_comments_layout)

        # Concentrations & A.F.U.
        concentrations_afu_layout = QVBoxLayout()
        concentrations_label = QLabel("Concentrations (comma-separated):")
        self.concentrations_entry = QLineEdit(
            ', '.join(str(conc) for conc in calibration.get('concentrations', []))
        )
        self.concentrations_entry.textChanged.connect(self.validate_inputs)

        afu_label = QLabel("A.F.U. Values (comma-separated):")
        self.afu_entry = QLineEdit(
            ', '.join(str(afu) for afu in calibration.get('afu_values', []))
        )
        self.afu_entry.textChanged.connect(self.validate_inputs)

        concentrations_afu_layout.addWidget(concentrations_label)
        concentrations_afu_layout.addWidget(self.concentrations_entry)
        concentrations_afu_layout.addWidget(afu_label)
        concentrations_afu_layout.addWidget(self.afu_entry)
        layout.addLayout(concentrations_afu_layout)

        # Calibration results
        calibration_results_layout = QFormLayout()

        self.slope_entry = QLineEdit(str(calibration.get('slope', '')))
        self.slope_entry.textChanged.connect(self.validate_inputs)

        self.intercept_entry = QLineEdit(str(calibration.get('intercept', '')))
        self.intercept_entry.textChanged.connect(self.validate_inputs)

        self.r_squared_entry = QLineEdit(str(calibration.get('r_squared', '')))
        self.r_squared_entry.textChanged.connect(self.validate_inputs)

        self.p_value_entry = QLineEdit(str(calibration.get('p_value', '')))
        self.p_value_entry.textChanged.connect(self.validate_inputs)

        self.std_err_entry = QLineEdit(str(calibration.get('std_err', '')))
        self.std_err_entry.textChanged.connect(self.validate_inputs)

        calibration_results_layout.addRow("Slope:", self.slope_entry)
        calibration_results_layout.addRow("Intercept:", self.intercept_entry)
        calibration_results_layout.addRow("R-squared:", self.r_squared_entry)
        calibration_results_layout.addRow("P-value:", self.p_value_entry)
        calibration_results_layout.addRow("Std Err:", self.std_err_entry)
        layout.addLayout(calibration_results_layout)

        # Validation message
        self.validation_label = QLabel("")
        self.validation_label.setStyleSheet("QLabel { color: red; font-weight: bold; }")
        self.validation_label.setWordWrap(True)
        layout.addWidget(self.validation_label)

        # OK/Cancel buttons
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

        # Initial validation
        self.validate_inputs()

    def validate_inputs(self):
        """Validate all input fields and enable/disable OK button."""
        is_valid = True
        error_messages = []
        conc_list = []
        afu_list = []

        # Validate concentrations
        try:
            conc_list = [float(c.strip()) for c in self.concentrations_entry.text().split(',') if c.strip()]
            if len(conc_list) == 0:
                raise ValueError("At least one concentration required")
            self.concentrations_entry.setStyleSheet("")
        except ValueError as e:
            is_valid = False
            self.concentrations_entry.setStyleSheet(self.error_style)
            error_messages.append("Concentrations: Invalid numeric values")

        # Validate AFU values
        try:
            afu_list = [float(a.strip()) for a in self.afu_entry.text().split(',') if a.strip()]
            if len(afu_list) == 0:
                raise ValueError("At least one A.F.U. value required")
            self.afu_entry.setStyleSheet("")
        except ValueError as e:
            is_valid = False
            self.afu_entry.setStyleSheet(self.error_style)
            error_messages.append("A.F.U. Values: Invalid numeric values")

        # Check length match
        if len(conc_list) > 0 and len(afu_list) > 0 and len(conc_list) != len(afu_list):
            is_valid = False
            error_messages.append("Concentrations and A.F.U. values must have same length")

        # Validate slope
        try:
            float(self.slope_entry.text())
            self.slope_entry.setStyleSheet("")
        except ValueError:
            is_valid = False
            self.slope_entry.setStyleSheet(self.error_style)
            error_messages.append("Slope: Invalid numeric value")

        # Validate intercept
        try:
            float(self.intercept_entry.text())
            self.intercept_entry.setStyleSheet("")
        except ValueError:
            is_valid = False
            self.intercept_entry.setStyleSheet(self.error_style)
            error_messages.append("Intercept: Invalid numeric value")

        # Validate R-squared
        try:
            float(self.r_squared_entry.text())
            self.r_squared_entry.setStyleSheet("")
        except ValueError:
            is_valid = False
            self.r_squared_entry.setStyleSheet(self.error_style)
            error_messages.append("R-squared: Invalid numeric value")

        # Validate P-value
        try:
            float(self.p_value_entry.text())
            self.p_value_entry.setStyleSheet("")
        except ValueError:
            is_valid = False
            self.p_value_entry.setStyleSheet(self.error_style)
            error_messages.append("P-value: Invalid numeric value")

        # Validate Std Err
        try:
            float(self.std_err_entry.text())
            self.std_err_entry.setStyleSheet("")
        except ValueError:
            is_valid = False
            self.std_err_entry.setStyleSheet(self.error_style)
            error_messages.append("Std Err: Invalid numeric value")

        # Update validation label and OK button
        if is_valid:
            self.validation_label.setText("")
            self.button_box.button(QDialogButtonBox.Ok).setEnabled(True)
        else:
            self.validation_label.setText("Errors:\n" + "\n".join(error_messages))
            self.button_box.button(QDialogButtonBox.Ok).setEnabled(False)

        return is_valid

    def get_updated_calibration(self):
        """Return dict with updated calibration info."""
        updated_calibration = self.calibration.copy()
        updated_calibration['species'] = self.species_entry.text()
        updated_calibration['comments'] = self.comments_entry.text()

        updated_calibration['concentrations'] = [
            float(conc.strip()) for conc in self.concentrations_entry.text().split(',')
            if conc.strip()
        ]
        updated_calibration['afu_values'] = [
            float(afu.strip()) for afu in self.afu_entry.text().split(',')
            if afu.strip()
        ]
        updated_calibration['slope'] = float(self.slope_entry.text())
        updated_calibration['intercept'] = float(self.intercept_entry.text())
        updated_calibration['r_squared'] = float(self.r_squared_entry.text())
        updated_calibration['p_value'] = float(self.p_value_entry.text())
        updated_calibration['std_err'] = float(self.std_err_entry.text())

        return updated_calibration