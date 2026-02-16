"""
Clearissa - kinetics_processor/main.py
---------------------------------------
Core kinetics data processing, model fitting, and analysis interface.

Author: Križan Jurinović
Date: October 2025
"""

import logging
import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QMessageBox,
    QHeaderView
)
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui

from .io_utils import KineticsIOUtils
from .gui import KineticsGUI
from .data_processor import DataProcessor
from .endpoint_detector import EndpointDetector
from .state_manager import StateManager
from .report_generator import ReportGenerator
from .fitting_engine import FittingEngine
from .plot_manager import PlotManager, infer_y_axis_label
from .results_formatter import format_results_html

logger = logging.getLogger(__name__)


class KineticsProcessor(QWidget):
    """
    Main interface for kinetic data analysis and model fitting.

    This class coordinates the GUI, I/O operations, and model fitting routines
    for kinetic reaction analysis. Supports multiple reaction mechanisms
    including TMSD, Catalytic turnover, and user-defined custom reactions.

    Parameters
    ----------
    parent : QWidget, optional
        Parent widget for Qt hierarchy management.

    Attributes
    ----------
    data_df : pandas.DataFrame or None
        Loaded experimental time-series data.
    fitted_df : pandas.DataFrame or None
        Simulated data from fitted kinetic models.
    plot_widget : pg.PlotWidget
        PyQtGraph widget for interactive plotting.
    models : KineticModels
        Instance managing kinetic model equations and fitting routines.

    Notes
    -----
    The processor maintains state between sessions via pickle serialisation.
    All time units are internally converted to seconds for numerical stability,
    while display units remain in minutes for user convenience.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # Set up the pyqtgraph plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.showGrid(x=False, y=False)

        # Set size policy to expand in both directions
        from PyQt5.QtWidgets import QSizePolicy
        self.plot_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.plot_widget.setMinimumHeight(200)  # Reduced minimum to fit small screens

        # Configure axes styling (matching replicate average plot)
        # Get axis references
        axis_left = self.plot_widget.getAxis('left')
        axis_bottom = self.plot_widget.getAxis('bottom')

        # Configure text colour to black
        axis_left.setTextPen('k')
        axis_bottom.setTextPen('k')

        # Disable SI prefixes to prevent "kmin" notation
        axis_left.enableAutoSIPrefix(False)
        axis_bottom.enableAutoSIPrefix(False)

        # Set axis labels with bold font
        font_size_str = '10pt'
        font = {'color': 'k', 'font-size': font_size_str, 'font-weight': 'bold'}
        self.plot_widget.setLabel('left', 'Signal', **font)
        self.plot_widget.setLabel('bottom', 'Time (min)', **font)

        # Configure tick font
        tick_font = QtGui.QFont()
        tick_font.setPointSize(10)
        tick_font.setBold(True)

        # Set tick styling
        tick_length = 4
        axis_bottom.setStyle(tickLength=-tick_length, tickFont=tick_font, tickTextOffset=10)
        axis_left.setStyle(tickLength=-tick_length, tickFont=tick_font, tickTextOffset=10)

        # Set pens AFTER all tick/style configuration to prevent overriding
        # Use full opacity black (0, 0, 0, 255) to ensure visibility
        axis_pen = pg.mkPen(color=(0, 0, 0, 255), width=1.5)
        tick_pen = pg.mkPen(color=(0, 0, 0, 255), width=1)

        axis_bottom.setPen(axis_pen)
        axis_bottom.setTickPen(tick_pen)

        axis_left.setPen(axis_pen)
        axis_left.setTickPen(tick_pen)

        # Add legend positioned on the right-hand side
        # Using negative x-offset to anchor legend to right edge of plot
        self.legend = self.plot_widget.addLegend(offset=(-10, 10))
        self.legend.setVisible(True)
        # Set legend to anchor to top-right corner for right-side placement
        self.legend.anchor(itemPos=(1, 0), parentPos=(1, 0), offset=(-10, 10))

        # Initialise component handlers
        self.io_utils = KineticsIOUtils(self)
        self.gui = None  # Will be initialised in init_ui

        # Initialise new modular components
        self.data_processor = DataProcessor(self)
        self.endpoint_detector = EndpointDetector(self)
        self.state_manager = None  # Will be initialised after io_utils
        self.report_generator = ReportGenerator(parent_widget=self)
        self.plot_manager = PlotManager(parent_widget=self)

        # Scientifically-validated colorblind-friendly colour palette
        # Based on Paul Tol's "Bright" and "Vibrant" palettes (SRON/EPS/TN/09-002)
        # optimised for MAXIMUM VISUAL DISTINCTION while remaining colorblind-safe.
        #
        # This palette is specifically designed to:
        # 1. Maximise visual contrast between adjacent traces (high saturation)
        # 2. Remain distinguishable for deuteranopia, protanopia, tritanopia
        # 3. Work in both print and digital media
        # 4. Meet scientific publication standards (Nature, Science, Cell)
        #
        # The palette maintains consistent assignment across datasets:
        # - Groups sorted alphanumerically before colour assignment
        # - Same group name always gets same colour across experiments
        # - Enables reliable cross-dataset comparisons
        #
        # References:
        # - Paul Tol (2012): Colour Schemes, SRON Technical Note SRON/EPS/TN/09-002
        # - Tested with CVD simulators (Coblis, Color Oracle)
        # - Recommended by https://personal.sron.nl/~pault/
        self.default_colours = [
            # Paul Tol "Bright" palette (7 colours - maximum distinction)
            '#4477AA',  # Blue (strong, excellent contrast)
            '#EE6677',  # Red (warm, highly visible, bold)
            '#228833',  # Green (natural, accessible to all CVD types)
            '#CCBB44',  # Yellow (bright, high contrast on white)
            '#66CCEE',  # Cyan (cool, distinctive from blue)
            '#AA3377',  # Purple (rich, stands out clearly)
            '#BBBBBB',  # Grey (neutral, good for reference lines)

            # Paul Tol "Vibrant" extended (13 additional colours)
            '#EE7733',  # Orange (warm, highly saturated)
            '#0077BB',  # Strong Blue (deeper than first blue)
            '#33BBEE',  # Bright Cyan (lighter variant)
            '#EE3377',  # Magenta (vibrant, eye-catching)
            '#CC3311',  # Dark Red
            '#009988',  # Teal (unique hue, excellent distinction)
            '#BBBBBB',  # Grey (repeated for balance)
            '#000000',  # Black (maximum contrast, good for data)
            '#DDCC77',  # Sand (warm neutral, distinct from yellow)
            '#117733',  # Forest Green (darker variant)
            '#882255',  # Wine (deep purple-red)
            '#44AA88',  # Sea Green (between green and teal)
            '#AA4499'   # Orchid (distinct purple variant)
        ]

        # Data storage
        self.data_df = None
        self.fitted_df = None
        self.time_col = "Time"
        self.filename = ""
        self._previous_dataset_hash = None  # Track dataset changes to preserve groups
        self._replicate_groups_by_hash = {}  # Store groups per dataset: {hash: {groups_dict}}

        # UI element references (will be populated by GUI)
        self.bimolecular_radio = None
        self.catalytic_radio = None
        self.stacked_widget = None
        self.bimolecular_widget = None
        self.ode_widget = None
        self.start_time_spinbox = None
        self.end_time_spinbox = None
        self.results_browser = None

        # UI element dictionaries for models
        self.initial_I_entry = None
        self.initial_SN_entry = None
        self.initial_guess_entry = None

        # Catalytic turnover model parameters
        self.catalytic_template_T_spinbox = None
        self.fluorescence_full_scale_nM_spinbox = None
        self.catalytic_S10_guess_spinbox = None
        self.catalytic_k_guess_spinbox = None
        self.catalytic_K_guess_spinbox = None


        # Experiment metadata
        self.experiment_title = ""
        self.experiment_info = ""
        self._last_mean_rate_constant = (0, 0)

        # Trace settings
        self.trace_settings = {}
        self.show_legend = True  # Default to showing legend
        self.replicate_info = {}  # Initialise replicate info dictionary
        self.replicate_stats_df = None  # Separate dataframe for replicate statistics (export only)

        # Store fitted parameters per trace for later use
        self.fitted_parameters = {}  # Dictionary to store fitted parameters: {trace_name: {'k_f': val, ...}}

        # Plot state
        self.scatter_state = False
        self.grid_state = False

        # Trace selection panel
        self.trace_selection_panel = None

        # Endpoint detection
        self.detected_endpoints = {}

        # Time window management
        self.original_time_range = None  # Store original time range (min, max)

    def init_ui(self):
        """
        Initialise the UI layout with tabbed interface optimised for small screens.
        Layout features:
        - Top: Graph (constant, always visible, expandable)
        - Middle: Fit and Clear buttons (always visible)
        - Bottom: Tabbed controls for different settings
        """
        logger.info("Initialising Kinetics Processor UI with tabbed layout...")

        # Initialise GUI handler
        self.gui = KineticsGUI(self, self.plot_widget)

        # Set a very high default max time for the time window
        max_time = 999999

        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Build callback dictionary
        callbacks = {
            'on_mode_changed': self.on_mode_changed,
            'on_load_data_clicked': self.on_load_data_clicked,
            'simulate_and_compare': self.simulate_and_compare,
            'clear_fits': self.clear_fits,
            'export_hd_graph': self.export_hd_graph,
            'export_results': self.export_results,
            'export_html_results': self.export_html_results,
            'open_experiment_info_dialog': self.open_experiment_info_dialog,
            'generate_replicate_average_plot': self.generate_replicate_average_plot,
            'update_plot': self.update_plot,
            'on_trace_selection_changed': self.on_trace_selection_changed,
            'manage_replicates': self.manage_replicates,
            'reset_time_window': self.reset_time_window
        }

        # Setup tabbed layout
        widgets = self.gui.setup_tabbed_layout(main_layout, max_time, callbacks)

        # Extract UI element references from returned dictionary
        self.start_time_spinbox = widgets.get('start_time_spinbox')
        self.end_time_spinbox = widgets.get('end_time_spinbox')
        self.time_window_status_label = widgets.get('status_label')
        self.time_window_error_label = widgets.get('error_label')

        self.fit_model_button = widgets.get('fit_model_button')
        self.clear_button = widgets.get('clear_button')

        self.trace_selection_panel = widgets.get('trace_selection_panel')

        self.results_browser = widgets.get('results_browser')
        self.export_html_button = widgets.get('export_html_button')
        self.r2_threshold_spinbox = widgets.get('r2_threshold_spinbox')

        self.reset_time_window_button = widgets.get('reset_time_window_button')

        # Mode selectors
        self.bimolecular_radio = widgets.get('bimolecular_radio')
        self.catalytic_radio = widgets.get('catalytic_radio')
        self.stacked_widget = widgets.get('stacked_widget')

        self.scatter_view_checkbox = widgets.get('scatter_view_checkbox')

        # Endpoint detection controls (integrated into bimolecular widget)
        self.endpoint_window_start = widgets.get('endpoint_window_start')
        self.endpoint_window_end = widgets.get('endpoint_window_end')
        self.detect_endpoints_button = widgets.get('detect_endpoints_button')

        # Bimolecular widget
        self.bimolecular_widget = widgets.get('bimolecular_widget')

        # Get the widgets from the stacked widget children
        # Index 0: Unified Bimolecular Reaction widget
        # Index 1: Catalytic Turnover widget
        self.ode_widget = self.stacked_widget.widget(1)

        # Extract UI element references from the unified bimolecular widget
        if self.bimolecular_widget and hasattr(self.bimolecular_widget, 'widgets'):
            bimol_widgets = self.bimolecular_widget.widgets
            # Bimolecular parameter spinbox references
            self.initial_I_entry = bimol_widgets.get('X0_guess_spinbox')
            self.initial_SN_entry = bimol_widgets.get('manual_Z0_default')
            self.initial_guess_entry = bimol_widgets.get('kf_guess_spinbox')

        if hasattr(self.ode_widget, 'widgets'):
            ode_widgets = self.ode_widget.widgets
            self.catalytic_template_T_spinbox = ode_widgets.get('catalytic_template_T_spinbox')
            self.fluorescence_full_scale_nM_spinbox = ode_widgets.get('fluorescence_full_scale_nM_spinbox')
            self.catalytic_S10_guess_spinbox = ode_widgets.get('catalytic_S10_guess_spinbox')
            self.catalytic_k_guess_spinbox = ode_widgets.get('catalytic_k_guess_spinbox')
            self.catalytic_K_guess_spinbox = ode_widgets.get('catalytic_K_guess_spinbox')
            # Bounds entries
            self.catalytic_k_lower_entry = ode_widgets.get('catalytic_k_lower_entry')
            self.catalytic_k_upper_entry = ode_widgets.get('catalytic_k_upper_entry')
            self.catalytic_K_lower_entry = ode_widgets.get('catalytic_K_lower_entry')
            self.catalytic_K_upper_entry = ode_widgets.get('catalytic_K_upper_entry')

        # Connect endpoint detection
        self.detect_endpoints_button.clicked.connect(self.detect_and_display_endpoints)

        # Connect scatter view checkbox
        self.scatter_view_checkbox.stateChanged.connect(self.toggle_scatter_view)

        # Initialise modules that depend on io_utils
        self.state_manager = StateManager(self, self.io_utils)

        self.setWindowTitle("Kinetics Processor")
        self.setGeometry(100, 100, 1400, 900)
        logger.info("UI initialised successfully with tabbed layout")

        # Note: Endpoint detection UI is now integrated into the bimolecular widget

        # Ensure the plot is consistent at program start
        self.update_plot()

    # -------------------------------------------------------------------------
    # Event Handlers
    # -------------------------------------------------------------------------

    def on_mode_changed(self):
        """
        Update displayed widget based on selected simulation mode and refresh plot.

        Simplified to two top-level modes:
        - Bimolecular Reaction (index 0) - with sub-selector for endpoint/manual
        - Catalytic Turnover (index 1)
        """
        if self.bimolecular_radio and self.bimolecular_radio.isChecked():
            self.stacked_widget.setCurrentIndex(0)
        elif self.catalytic_radio and self.catalytic_radio.isChecked():
            self.stacked_widget.setCurrentIndex(1)

        # Redraw so any fitted vs. raw styling reflects the new mode immediately
        self.update_plot()


    def toggle_scatter_view(self, state):
        """
        Toggle scatter view on/off.

        Args:
            state: Qt.CheckState value from checkbox
        """
        self.scatter_state = (state == Qt.Checked)
        self.update_plot()
        logger.debug("Scatter view toggled to: %s", self.scatter_state)

    def clear_fits(self):
        """Clear all fitted data and refresh the plot."""
        logger.info("Clearing fitted data...")
        self.fitted_df = None
        self.results_browser.setHtml("<h3>Fits cleared. Click 'Simulate' to generate new fits.</h3>")
        self.update_plot()
        logger.info("Fitted data cleared successfully.")

    def reset_time_window(self):
        """
        Reset time window to the full available data range.
        """
        if self.data_df is None:
            QMessageBox.warning(
                self,
                "No Data Loaded",
                "Please load kinetic data first."
            )
            return

        # Get full time range from data
        min_time = float(self.data_df[self.time_col].min())
        max_time = float(self.data_df[self.time_col].max())

        # Reset spinbox values to full range
        self.start_time_spinbox.setValue(min_time)
        self.end_time_spinbox.setValue(max_time)

        # Update plot
        self.update_plot()

        logger.info("Reset time window to full range: %.2f to %.2f min", min_time, max_time)

    def on_load_data_clicked(self):
        """Handle the Load Data File button click."""
        loaded_df = self.io_utils.load_data()
        if loaded_df is None:
            return

        # Check if this is the same dataset as before (via hash comparison)
        new_dataset_hash = self.io_utils.current_dataset_hash
        previous_dataset_hash = getattr(self, '_previous_dataset_hash', None)

        same_dataset = (previous_dataset_hash is not None and
                       new_dataset_hash is not None and
                       previous_dataset_hash == new_dataset_hash)

        # Check if we have stored groups for this dataset hash
        has_stored_groups = new_dataset_hash in self._replicate_groups_by_hash

        if same_dataset:
            logger.info("Same dataset re-loaded - preserving groups and fits")
        else:
            logger.info("New dataset loaded - clearing fits")
            # Clear fits for new dataset (but not groups - they're in the hash map)
            self.fitted_df = None
            self.fitted_parameters = {}
            self.detected_endpoints = {}
            self.replicate_info = {}

            # Check if we have stored groups for this new dataset
            if has_stored_groups and self.data_processor:
                stored_groups = self._replicate_groups_by_hash[new_dataset_hash]
                self.data_processor.user_defined_groups = stored_groups.copy()
                num_groups = len(stored_groups)
                logger.info("Loaded %d stored replicate groups for this dataset (hash: %s...)",
                           num_groups, new_dataset_hash[:16])
            else:
                if self.data_processor:
                    self.data_processor.user_defined_groups = {}

        # Store current hash for next comparison
        self._previous_dataset_hash = new_dataset_hash

        self.data_df = loaded_df
        self.time_col = self.io_utils.time_col
        self.filename = self.io_utils.filename

        # Detect and average replicates
        self.data_df = self.initialise_data_display()

        # FIXED: Update time window based on loaded data with proper range
        if self.time_col in self.data_df.columns:
            max_time = self.data_df[self.time_col].max()
            min_time = self.data_df[self.time_col].min()
            
            logger.info("Data time range: %.2f to %.2f", min_time, max_time)
            
            # Set maximum to allow viewing beyond data range
            self.start_time_spinbox.setMaximum(max_time * 1.5)
            self.start_time_spinbox.setMinimum(0)
            self.start_time_spinbox.setValue(min_time)
            
            self.end_time_spinbox.setMaximum(max_time * 1.5)
            self.end_time_spinbox.setMinimum(0)
            self.end_time_spinbox.setValue(max_time)

            # Auto-set endpoint detection window to last 15 minutes
            self.auto_set_endpoint_window_last_15min()

        # Update y-axis label based on data range
        self._update_y_axis_label()

        # Update the plot to show loaded data
        self.update_plot()

        # Auto-save state after loading data
        self.save_state()

        status_msg = f"Successfully loaded data from {self.filename}\n" \
                    f"{len(self.data_df)} rows, {len(self.data_df.columns)} columns\n" \
                    f"Time range: {min_time:.2f} to {max_time:.2f} min"

        if not same_dataset:
            if has_stored_groups and num_groups > 0:
                status_msg += f"\n\nNote: Fits cleared (new dataset)\nStored replicate groups loaded ({num_groups} groups)"
            else:
                status_msg += "\n\nNote: Fits cleared (new dataset)"
        else:
            status_msg += "\n\nNote: Same dataset - preserving groups and fits"

        QMessageBox.information(self, "Data Loaded", status_msg)

    def open_experiment_info_dialog(self):
        """Open dialog for editing experiment information."""
        self.gui.open_experiment_info_dialog(
            self.experiment_title,
            self.experiment_info,
            self.save_experiment_info
        )

    def save_experiment_info(self, dialog, title, info):
        """Save experiment information."""
        self.experiment_title = title
        self.experiment_info = info
        dialog.accept()
        self.update_plot()

    def manage_replicates(self):
        """Launch interactive replicate management dialogue."""
        if self.data_df is None:
            QMessageBox.warning(
                self,
                "No Data Loaded",
                "Please load kinetic data before managing replicates."
            )
            return

        # Launch replicate manager with colour palette for preview
        changed = self.data_processor.manage_replicates_interactive(
            self.data_df,
            self.time_col,
            self.gui,
            colour_palette=self.default_colours
        )

        if changed:
            # Reprocess replicates with new groups
            self.data_df = self.data_processor.process_replicates(
                self.data_df,
                self.time_col,
                show_dialogue=False,
                gui_handler=self.gui
            )

            # Update replicate info from data processor
            self.replicate_stats_df = self.data_processor.replicate_stats_df
            self.replicate_info = self.data_processor.replicate_info

            # Save the updated groups to hash mapping
            self.save_groups_to_hash_mapping()

            # Update plot to reflect changes
            self.update_plot()

            # Save state
            self.save_state()

            logger.info("Replicate groups updated successfully")

    def load_dataframe_directly(self, df: pd.DataFrame, label: str = "Imported Data", metadata: dict = None):
        """
        Load a DataFrame directly into the kinetics processor.

        This method is used when data is forwarded from other tabs
        (e.g., Convert Data tab) instead of loading from a file.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing time-series data with 'Well' and 'Time [min]' columns
        label : str
            Label describing the dataset source
        metadata : dict, optional
            Metadata from conversion, may contain:
            - per_trace_Z0: dict mapping trace names to initial concentrations (nM)
            - mode: conversion mode (e.g., 'HMSD')
            - species: species name
        """
        if df is None or df.empty:
            QMessageBox.warning(
                self,
                "Invalid Data",
                "Cannot load empty DataFrame."
            )
            return

        # Find time column
        time_col = None
        for col in df.columns:
            if 'time' in col.lower():
                time_col = col
                break

        if time_col is None:
            QMessageBox.warning(
                self,
                "Invalid Data",
                "DataFrame must contain a time column (e.g., 'Time [min]')."
            )
            return

        # Drop 'Well' column if present - it contains species names from converted data
        # and is not needed for kinetics processing
        if 'Well' in df.columns:
            df = df.drop(columns=['Well'])
            logger.info("Dropped 'Well' column (contains species names, not needed for kinetics)")

        # Store the DataFrame and metadata
        self.data_df = df.copy()
        self.time_col = time_col
        self.filename = label

        # Store metadata for later use
        self._forwarded_metadata = metadata or {}

        # Calculate dataset hash so groups are preserved across sessions
        self.io_utils.current_dataset_hash = self.io_utils.calculate_dataset_hash(
            self.data_df, self.filename
        )
        new_dataset_hash = self.io_utils.current_dataset_hash
        logger.info("Calculated dataset hash for forwarded data: %s...", new_dataset_hash[:16])

        # Check for stored groups from a previous session with this dataset
        previous_dataset_hash = getattr(self, '_previous_dataset_hash', None)
        same_dataset = (previous_dataset_hash is not None and
                       new_dataset_hash is not None and
                       previous_dataset_hash == new_dataset_hash)

        if not same_dataset:
            # Clear fits for new dataset
            self.fitted_df = None
            self.fitted_parameters = {}
            self.detected_endpoints = {}
            self.replicate_info = {}

            # Restore stored groups if available for this dataset
            if new_dataset_hash in self._replicate_groups_by_hash and self.data_processor:
                stored_groups = self._replicate_groups_by_hash[new_dataset_hash]
                self.data_processor.user_defined_groups = stored_groups.copy()
                logger.info("Restored %d stored replicate groups for forwarded dataset",
                           len(stored_groups))
            elif self.data_processor:
                self.data_processor.user_defined_groups = {}

        self._previous_dataset_hash = new_dataset_hash

        logger.info("Loaded DataFrame directly: %s", label)
        logger.info("  Shape: %d rows x %d columns", df.shape[0], df.shape[1])
        logger.info("  Time column: %s", time_col)

        # Detect and average replicates
        self.data_df = self.initialise_data_display()

        # Update time window based on loaded data
        if self.time_col in self.data_df.columns:
            max_time = self.data_df[self.time_col].max()
            min_time = self.data_df[self.time_col].min()

            logger.info("Data time range: %.2f to %.2f", min_time, max_time)

            # Set maximum to allow viewing beyond data range
            self.start_time_spinbox.setMaximum(max_time * 1.5)
            self.start_time_spinbox.setMinimum(0)
            self.start_time_spinbox.setValue(min_time)

            self.end_time_spinbox.setMaximum(max_time * 1.5)
            self.end_time_spinbox.setMinimum(0)
            self.end_time_spinbox.setValue(max_time)

            # Auto-set endpoint detection window to last 15 minutes
            self.auto_set_endpoint_window_last_15min()

        # Auto-save state after loading data
        self.save_state()

        # Ensure main plot configuration is correct
        self.reset_main_plot_configuration()

        # Update the plot to show loaded data
        self.update_plot()

        # Handle per-trace Z0 values from HMSD conversion metadata
        per_trace_Z0 = self._forwarded_metadata.get('per_trace_Z0', {})
        if per_trace_Z0:
            self._apply_forwarded_initial_concentrations(per_trace_Z0)

        # Show success message
        QMessageBox.information(
            self,
            "Data Loaded",
            f"Successfully loaded data: {label}\n"
            f"{len(self.data_df)} rows, {len(self.data_df.columns)} columns\n"
            f"Time range: {min_time:.2f} to {max_time:.2f} min"
        )

    def _apply_forwarded_initial_concentrations(self, per_trace_Z0: dict) -> None:
        """
        Apply forwarded initial concentrations to the bimolecular manual per-trace mode.

        This method is called when data is forwarded from the Convert Data tab
        with per-trace initial concentrations (e.g., [S1-T]_0 from HMSD conversion).
        It pre-populates the manual per-trace Z0 table and switches to manual mode.

        Parameters
        ----------
        per_trace_Z0 : dict
            Dictionary mapping trace/well names to initial concentrations in nM.
            Keys are well names (e.g., 'A01', 'B02') or averaged trace names.
        """
        if not per_trace_Z0:
            return

        logger.info(f"Applying {len(per_trace_Z0)} forwarded initial concentrations")

        # Check if bimolecular widget exists
        if not hasattr(self, 'bimolecular_widget') or self.bimolecular_widget is None:
            logger.warning("Bimolecular widget not available, cannot apply initial concentrations")
            return

        try:
            # Get trace names from current data (after replicate averaging)
            trace_names = [col for col in self.data_df.columns if col != self.time_col]

            # Build mapping from forwarded values to current trace names
            # Handle both exact matches and partial matches for averaged traces
            matched_values = {}
            for trace_name in trace_names:
                # Try exact match first
                if trace_name in per_trace_Z0:
                    matched_values[trace_name] = per_trace_Z0[trace_name]
                    continue

                # For averaged traces like "A01-A03 (avg)", try matching component wells
                # Extract well names from the trace name
                for well_name, value in per_trace_Z0.items():
                    if well_name in trace_name:
                        # Use the first matching well's value (they should be similar for replicates)
                        if trace_name not in matched_values:
                            matched_values[trace_name] = value
                        break

            if not matched_values:
                logger.warning("No trace names matched the forwarded initial concentrations")
                return

            logger.info(f"Matched {len(matched_values)} traces to forwarded concentrations")

            # Store the values for when the table is populated
            self.bimolecular_widget.set_last_per_trace_Z0_values(matched_values)

            # Populate the per-trace Z0 table with the matched values
            self.bimolecular_widget.populate_per_trace_Z0_table(
                trace_names,
                default_value=None,
                saved_values=matched_values
            )

            # Switch to manual mode
            if hasattr(self.bimolecular_widget, 'widgets'):
                manual_radio = self.bimolecular_widget.widgets.get('manual_radio')
                if manual_radio:
                    manual_radio.setChecked(True)
                    logger.info("Switched to manual per-trace mode with pre-populated concentrations")

            # Update status text
            if hasattr(self.bimolecular_widget, 'update_status'):
                self.bimolecular_widget.update_status(
                    f"Loaded {len(matched_values)} initial concentrations from conversion"
                )

        except Exception as e:
            logger.error(f"Failed to apply forwarded initial concentrations: {e}", exc_info=True)

    # -------------------------------------------------------------------------
    # Plot Configuration Management
    # -------------------------------------------------------------------------

    def reset_main_plot_configuration(self):
        """
        Reset the main plot widget configuration to ensure it's not affected by other operations.

        This method ensures the main plot widget maintains its proper axis styling
        and configuration, especially after operations that might have modified settings.
        """
        try:
            # Reset axis pens to standard display thickness
            axis_pen = pg.mkPen('k', width=2)
            self.plot_widget.getAxis("bottom").setPen(axis_pen)
            self.plot_widget.getAxis("left").setPen(axis_pen)

            # Ensure SI prefixes are disabled
            self.plot_widget.getAxis("bottom").enableAutoSIPrefix(False)
            self.plot_widget.getAxis("left").enableAutoSIPrefix(False)

            # Reset axis labels
            self.plot_widget.setLabel('bottom', 'Time (min)')
            self.plot_widget.setLabel('left', 'Signal')

            # Reset tick font
            tick_font = QtGui.QFont()
            tick_font.setPointSize(10)
            self.plot_widget.getAxis("bottom").setStyle(tickFont=tick_font)
            self.plot_widget.getAxis("left").setStyle(tickFont=tick_font)

            logger.debug("Main plot configuration reset successfully")

        except Exception as e:
            logger.warning("Failed to reset main plot configuration: %s", e)

    def _update_y_axis_label(self):
        """Update the main plot y-axis label based on the current data range."""
        if self.data_df is None or self.data_df.empty:
            return

        data_cols = [c for c in self.data_df.columns
                     if c != self.time_col
                     and not c.endswith('_fitted')
                     and not c.endswith(' Std')
                     and not c.endswith(' SEM')]

        if not data_cols:
            return

        y_min = self.data_df[data_cols].min().min()
        y_max = self.data_df[data_cols].max().max()
        label = infer_y_axis_label(y_min, y_max)
        self.plot_widget.setLabel('left', label)
        logger.info("Y-axis label set to: %s", label)

    # -------------------------------------------------------------------------
    # Data Processing
    # -------------------------------------------------------------------------

    def initialise_data_display(self):
        """
        Initialise trace UI and detect replicate groups after data is loaded.

        This method handles two separate concerns:
        1. Trace display - populates trace checkboxes, per-trace parameter
           tables, and initialises trace_settings for the main plot.
        2. Replicate groups - delegates auto-detection to data_processor
           and stores the resulting statistics for the replicate averages window.
        """
        if self.data_df is None or self.time_col not in self.data_df.columns:
            logger.warning("Cannot initialise data display without valid data and a time column")
            return self.data_df

        df = self.data_df.copy()
        time_col = self.time_col

        # Get all data columns (exclude time and fitted curves)
        data_cols = [c for c in df.columns if c != time_col and not c.endswith("_fitted")]

        # Initialise trace settings: show all traces by default
        for col in data_cols:
            if col not in self.trace_settings:
                self.trace_settings[col] = {'show_trace': True}

        # Detect replicate groups and calculate statistics (for replicate averages window)
        df = self.data_processor.process_replicates(df, time_col, show_dialogue=False)
        self.replicate_stats_df = self.data_processor.replicate_stats_df
        self.replicate_info = self.data_processor.replicate_info

        # Populate the trace selection list (flat list - no group headers on main plot)
        if self.trace_selection_panel:
            self.gui.populate_trace_list(
                self.trace_selection_panel,
                df,
                time_col,
                {}
            )

        # Populate per-trace parameter tables
        self._populate_per_trace_T_table(data_cols)
        self._populate_per_trace_Z0_table(data_cols)

        # Update endpoint detection time window to last 15 min of dataset
        self._update_endpoint_window_defaults()

        logger.info("Data display initialised with %d traces", len(data_cols))
        if self.replicate_stats_df is not None and not self.replicate_stats_df.empty:
            logger.info("Detected %d replicate groups for replicate averages", len(self.replicate_info))

        return df

    def save_groups_to_hash_mapping(self):
        """
        Save current replicate groups to the hash-based mapping.
        This allows groups to be preserved when switching between datasets.
        """
        if not self.data_processor or not hasattr(self.data_processor, 'user_defined_groups'):
            logger.warning("Cannot save groups - data processor not initialised")
            return

        # Calculate hash if not already set
        if not hasattr(self.io_utils, 'current_dataset_hash') or not self.io_utils.current_dataset_hash:
            if self.data_df is not None:
                self.io_utils.current_dataset_hash = self.io_utils.calculate_dataset_hash(
                    self.data_df, self.filename
                )
                logger.info("Calculated dataset hash for group saving: %s...",
                           self.io_utils.current_dataset_hash[:16])
            else:
                logger.warning("Cannot save groups - no data loaded")
                return

        dataset_hash = self.io_utils.current_dataset_hash

        # Save a copy of the current groups to the hash mapping
        current_groups = self.data_processor.user_defined_groups.copy()
        self._replicate_groups_by_hash[dataset_hash] = current_groups

        num_groups = len(current_groups)
        logger.info("Saved %d replicate groups for dataset (hash: %s...)",
                   num_groups, dataset_hash[:16])

        # Trigger state save
        self.save_state()

    # -------------------------------------------------------------------------
    # Simulation and Fitting
    # -------------------------------------------------------------------------

    def _get_model_configuration(self):
        """
        Determine current model type and extract parameters.

        Returns
        -------
        tuple
            (model_type: str, params: dict) or (None, None) if invalid

        Model types:
            - 'bimolecular': Unified bimolecular reaction (covers TMSD/internal TMSD/HMSD)
            - 'catalytic': Catalytic turnover
        """
        if self.bimolecular_radio and self.bimolecular_radio.isChecked():
            # Unified bimolecular mode
            if self.bimolecular_widget and hasattr(self.bimolecular_widget, 'get_parameters'):
                bimol_params = self.bimolecular_widget.get_parameters()
                input_mode = bimol_params.get('input_mode', 'endpoint')

                return 'bimolecular', {
                    'input_mode': input_mode,
                    'X0_guess': bimol_params.get('X0_guess', 10.0),
                    'kf_guess': bimol_params.get('kf_guess', 1e5),
                    'Z0_default': bimol_params.get('Z0_default', 10.0),
                    'Z0_manual': bimol_params.get('Z0_manual', 10.0),
                }

        elif self.catalytic_radio and self.catalytic_radio.isChecked():
            # Determine sub-model type (full or simple)
            catalytic_sub_model = 'full'
            if hasattr(self.ode_widget, 'get_catalytic_sub_model'):
                catalytic_sub_model = self.ode_widget.get_catalytic_sub_model()

            return 'catalytic', {
                'catalytic_sub_model': catalytic_sub_model,
                'catalytic_template_T': self.catalytic_template_T_spinbox.value() if self.catalytic_template_T_spinbox else 1.0,
                'fluorescence_full_scale_nM': self.fluorescence_full_scale_nM_spinbox.value() if self.fluorescence_full_scale_nM_spinbox else 10.0,
                'catalytic_S10_guess': self.catalytic_S10_guess_spinbox.value() if self.catalytic_S10_guess_spinbox else 10.0,
                'catalytic_k_guess': self.catalytic_k_guess_spinbox.value() if self.catalytic_k_guess_spinbox else 1.0,
                'catalytic_K_guess': self.catalytic_K_guess_spinbox.value() if self.catalytic_K_guess_spinbox else 10.0,
                # Bounds for global fitting
                'catalytic_k_lower': self.catalytic_k_lower_entry.value() if getattr(self, 'catalytic_k_lower_entry', None) else 1e-6,
                'catalytic_k_upper': self.catalytic_k_upper_entry.value() if getattr(self, 'catalytic_k_upper_entry', None) else 100.0,
                'catalytic_K_lower': self.catalytic_K_lower_entry.value() if getattr(self, 'catalytic_K_lower_entry', None) else 0.1,
                'catalytic_K_upper': self.catalytic_K_upper_entry.value() if getattr(self, 'catalytic_K_upper_entry', None) else 1000.0,
            }


        else:
            logger.error("No model type selected")
            return None, None

    def simulate_and_compare(self):
        """
        Execute fitting for kinetic models.

        Routes to bimolecular or catalytic fitting based on selected model.
        """
        logger.info("simulate_and_compare() called")

        # Prechecks
        if self.data_df is None or self.data_df.empty:
            logger.warning("No data available")
            QMessageBox.warning(self, "No Data", "Please load data before fitting.")
            return

        t0 = self.start_time_spinbox.value()
        t1 = self.end_time_spinbox.value()
        logger.debug("Time window: start=%.6g min, end=%.6g min", t0, t1)

        if t0 >= t1:
            logger.error("Start time must be less than end time")
            QMessageBox.warning(
                self,
                "Invalid Time Window",
                "Start time must be less than end time."
            )
            return

        # Copy necessary: thread safety (worker will use this copy)
        df = self.data_df.copy()
        time_col = self.time_col

        # Ensure monotonic time (sort BEFORE filtering for correctness)
        if not df[time_col].is_monotonic_increasing:
            logger.debug("Sorting data by time column before filtering")
            df = df.sort_values(time_col).reset_index(drop=True)

        # Apply time window with consistent tolerance handling
        # CRITICAL FIX: Use filter_time_window() instead of df.between()
        # Consistent boundary tolerance across all code paths
        from .data_processor import filter_time_window
        df = filter_time_window(df, time_col, t0, t1)

        if df.empty:
            logger.warning("No data in time window")
            QMessageBox.warning(
                self,
                "Empty Time Window",
                f"No data exists in the time range {t0:.2f} to {t1:.2f} minutes."
            )
            return

        logger.info("Filtering to time window: %.2f-%.2f min (%d timepoints)",
                   t0, t1, len(df))

        # Determine model type and gather parameters
        model_type, params = self._get_model_configuration()

        if model_type is None:
            logger.error("Could not determine model type")
            return

        # Model-specific setup
        if model_type == 'catalytic':
            self._sync_per_trace_T_to_settings()
        elif model_type == 'bimolecular':
            # Unified bimolecular model - sync per-trace Z0 values based on input mode
            input_mode = params.get('input_mode', 'endpoint')
            if input_mode == 'manual':
                # Manual mode: use per-trace Z0 values from trace settings
                self._sync_per_trace_Z0_to_settings()
            else:
                # Endpoint detection mode: require detected endpoints
                if self.detected_endpoints:
                    # Sync detected endpoints to trace_settings as Z0 values
                    self._sync_endpoints_to_Z0_settings()
                else:
                    # No endpoints detected - prompt user to run detection first
                    QMessageBox.warning(
                        self,
                        "Endpoints Required",
                        "No endpoints have been detected.\n\n"
                        "Please click 'Detect Endpoints' first, or switch to "
                        "'Manual per-trace' mode to specify [Z]_0 values directly."
                    )
                    logger.warning("Fitting aborted: no endpoints detected in endpoint mode")
                    return

        # Copy filtered dataframe - will be modified with fitted results
        # IMPORTANT: Keep ORIGINAL time coordinates (not normalised)
        # This allows plot to handle both data_df and fitted_df consistently
        self.fitted_df = df.copy()

        # CRITICAL: Store ACTUAL first datapoint time for normalisation
        # Use first actual datapoint in filtered data, NOT UI spinbox value
        # This ensures fitted curves align with displayed data
        actual_first_time = df[time_col].min()
        params['time_window_start'] = actual_first_time  # Use actual first datapoint!
        params['time_window_end'] = df[time_col].max()

        # Log the normalisation reference for debugging
        if abs(actual_first_time - t0) > 0.01:
            logger.info("Time window adjustment: UI requested %.2f min, but first actual datapoint is %.2f min (gap of %.2f min)",
                       t0, actual_first_time, actual_first_time - t0)
            # Inform user about the gap
            gap_info = (f"Note: Time window starts at {t0:.2f} min, but first actual datapoint is at {actual_first_time:.2f} min.\n"
                       f"Plots and fits will display with t=0 at {actual_first_time:.2f} min (gap of {actual_first_time - t0:.2f} min).")
            logger.info(gap_info)
        else:
            logger.debug("Time window starts at first datapoint: %.2f min", actual_first_time)

        # Count visible traces
        visible_count = sum(
            1 for c in df.columns
            if c != time_col
            and not c.endswith('_fitted')
            and self.trace_settings.get(c, {}).get('show_trace', True)
        )

        if visible_count == 0:
            logger.warning("No visible traces to fit")
            QMessageBox.warning(
                self,
                "No Traces",
                "No traces are visible for fitting. Please select traces to fit."
            )
            return

        logger.info("Starting %s model fit", model_type)

        # Create and run fitting engine synchronously
        engine = FittingEngine(df, time_col, params, model_type, self.trace_settings)

        try:
            results = engine.run()
            self._on_fitting_complete(results)
        except Exception as e:
            logger.error("Fitting error: %s", e, exc_info=True)
            QMessageBox.critical(
                self,
                "Fitting Error",
                f"An error occurred during fitting:\n\n{e}"
            )

    # -------------------------------------------------------------------------
    # Plotting and Export
    # -------------------------------------------------------------------------

    def generate_replicate_average_plot(self):
        """Delegate replicate average plot generation to plot_manager."""
        if self.data_df is None or self.data_df.empty:
            QMessageBox.warning(self, "No Data", "Please load data first.")
            return

        t_start = self.start_time_spinbox.value() if self.start_time_spinbox else 0
        t_end = self.end_time_spinbox.value() if self.end_time_spinbox else 60

        # Determine if Catalytic model is selected and get the scale factor
        # When Catalytic mode is active, experimental data (0-1 normalised) should be
        # scaled to nM for proper averaging alongside the fitted curve (already in nM)
        catalytic_scale_factor = None
        if self.catalytic_radio and self.catalytic_radio.isChecked():
            if self.fluorescence_full_scale_nM_spinbox:
                catalytic_scale_factor = self.fluorescence_full_scale_nM_spinbox.value()

        r2_threshold = self.r2_threshold_spinbox.value()
        self.plot_manager.generate_replicate_average_plot(
            data_df=self.data_df,
            fitted_df=self.fitted_df,
            time_col=self.time_col,
            replicate_info=self.replicate_info,
            trace_settings=self.trace_settings,
            fitted_parameters=self.fitted_parameters,
            default_colours=self.default_colours,
            report_generator=self.report_generator,
            t_start=t_start,
            t_end=t_end,
            catalytic_scale_factor=catalytic_scale_factor,
            r2_threshold=r2_threshold
        )


    def update_plot(self, *_) -> None:
        """
        Update the plot with current data.

        Accepts and ignores any positional args so it can be connected directly
        to Qt signals like QDoubleSpinBox.valueChanged which pass the new value.
        """
        # Determine if Catalytic model is selected and get the scale factor
        # When Catalytic mode is active, experimental data (0-1 normalised) should be
        # scaled to nM for display alongside the fitted curve (which is already in nM)
        catalytic_scale_factor = None
        if self.catalytic_radio and self.catalytic_radio.isChecked():
            if self.fluorescence_full_scale_nM_spinbox:
                catalytic_scale_factor = self.fluorescence_full_scale_nM_spinbox.value()

        self.gui.update_plot(
            self.data_df,
            self.fitted_df,
            self.time_col,
            self.trace_settings,
            self.start_time_spinbox.value() if self.start_time_spinbox else 0,
            self.end_time_spinbox.value() if self.end_time_spinbox else 60,
            self.replicate_info,
            self.show_legend,
            catalytic_scale_factor=catalytic_scale_factor
        )

    def export_hd_graph(self):
        """Delegate publication graph export to report_generator."""
        if self.data_df is None or self.data_df.empty:
            QMessageBox.warning(self, "No Data", "No data available to export.")
            return

        t_start = self.start_time_spinbox.value() if self.start_time_spinbox else 0
        t_end = self.end_time_spinbox.value() if self.end_time_spinbox else self.data_df[self.time_col].max()

        self.report_generator.export_publication_graph(
            data_df=self.data_df,
            fitted_df=self.fitted_df,
            time_col=self.time_col,
            trace_settings=self.trace_settings,
            replicate_stats_df=self.replicate_stats_df,
            replicate_info=self.replicate_info,
            t_start=t_start,
            t_end=t_end
        )

    def export_results(self):
        """Export results to HTML with embedded graph."""
        from pyqtgraph.exporters import ImageExporter
        import tempfile

        results_html = self.results_browser.toHtml()

        # Build model info HTML based on selected model
        model_info_html = ""

        if self.bimolecular_radio and self.bimolecular_radio.isChecked():
            # Unified Bimolecular Reaction Model Information
            # Get parameters from widget if available
            if self.bimolecular_widget and hasattr(self.bimolecular_widget, 'get_parameters'):
                bimol_params = self.bimolecular_widget.get_parameters()
                input_mode = bimol_params.get('input_mode', 'endpoint')
                X0_guess = bimol_params.get('X0_guess', 10.0)
                kf_guess = bimol_params.get('kf_guess', 1e5)
                if input_mode == 'endpoint':
                    Z0_value = bimol_params.get('Z0_manual', 10.0)
                else:
                    Z0_value = bimol_params.get('Z0_default', 10.0)
            else:
                input_mode = 'endpoint'
                X0_guess = self.initial_I_entry.value() if self.initial_I_entry else 10.0
                kf_guess = self.initial_guess_entry.value() if self.initial_guess_entry else 1e5
                Z0_value = self.initial_SN_entry.value() if self.initial_SN_entry else 10.0

            fixed_conc_source = "endpoint detection" if input_mode == 'endpoint' else "manual input"

            model_info_html = (
                f"<div style='font-family:Arial; font-size:11pt; margin-bottom:20px; "
                f"background:#e8f5e9; padding:15px; border-left:4px solid #4caf50; border-radius:4px;'>"
                f"<h3 style='margin-top:0; color:#2e7d32; font-size:13pt;'>Model Information</h3>"
                f"<p><b>Model Type:</b> Irreversible Bimolecular Reaction</p>"
                f"<p><b>Reaction Scheme:</b> X + Z &rarr; Y + W</p>"
                f"<p><b>Rate Law:</b> dY/dt = k<sub>f</sub> &middot; ([X]<sub>0</sub> &minus; Y) &middot; ([Z]<sub>0</sub> &minus; Y)</p>"
                f"<p><b>Initial Condition:</b> Y(0) = 0</p>"
                f"<p style='margin-top:10px;'><b>Per-Trace Fitted Parameters:</b></p>"
                f"<ul style='margin:5px 0 10px 20px;'>"
                f"<li><b>k<sub>f</sub></b> &ndash; forward rate constant (M<sup>&minus;1</sup> s<sup>&minus;1</sup>)</li>"
                f"<li><b>[X]<sub>0</sub></b> &ndash; initial reactant concentration (nM)</li>"
                f"</ul>"
                f"<p><b>Per-Trace Fixed Parameter:</b></p>"
                f"<ul style='margin:5px 0 10px 20px;'>"
                f"<li><b>[Z]<sub>0</sub></b> &ndash; fixed reactant concentration (nM), obtained via {fixed_conc_source}</li>"
                f"</ul>"
                f"<p><b>Time Units:</b> Seconds</p>"
                f"<p><b>Fitting Procedure:</b></p>"
                f"<p style='font-size:10pt; margin-left:10px;'>"
                f"Each trace is fitted independently using non-linear least squares (<code>scipy.optimize.curve_fit</code>) "
                f"with an analytical solution to the rate equation. The fitting parameters k<sub>f</sub> and [X]<sub>0</sub> "
                f"are constrained to be non-negative. Fit quality is quantified using R<sup>2</sup>.</p>"
                f"<p style='margin-top:10px;'><b>Initial Guesses:</b></p>"
                f"<ul style='margin:5px 0 10px 20px;'>"
                f"<li>k<sub>f</sub>: {kf_guess:.3e} M<sup>&minus;1</sup> s<sup>&minus;1</sup></li>"
                f"<li>[X]<sub>0</sub>: {X0_guess:.2f} nM</li>"
                f"<li>[Z]<sub>0</sub> (fixed): {Z0_value:.2f} nM</li>"
                f"</ul>"
                f"</div><hr>"
            )

        elif self.catalytic_radio and self.catalytic_radio.isChecked():
            # Catalytic Model Information - Michaelis-Menten kinetics (n=1)
            # Get default T value from UI
            default_T = self.catalytic_template_T_spinbox.value() if hasattr(self, 'catalytic_template_T_spinbox') else 0

            model_info_html = (
                f"<div style='font-family:Arial; font-size:11pt; margin-bottom:20px; "
                f"background:#e8f5e9; padding:15px; border-left:4px solid #4caf50; border-radius:4px;'>"
                f"<h3 style='margin-top:0; color:#2e7d32; font-size:13pt;'>Model Information</h3>"
                f"<p><b>Model Type:</b> Catalytic Turnover (Michaelis&ndash;Menten kinetics)</p>"
                f"<p><b>Rate Law:</b> d[R-S-D]/dt = (k &middot; [R-L]) / (K + [R-L]) &middot; T</p>"
                f"<p><b>where:</b> [R-L] = [R-L]<sub>0</sub> &minus; [R-S-D](t) (mass conservation)</p>"
                f"<p style='margin-top:10px;'><b>Global Fitted Parameters (shared across all traces):</b></p>"
                f"<ul style='margin:5px 0 10px 20px;'>"
                f"<li><b>k</b> &ndash; turnover rate constant (min<sup>&minus;1</sup>)</li>"
                f"<li><b>K</b> &ndash; (nM)</li>"
                f"</ul>"
                f"<p><b>Per-Trace Fitted Parameter:</b> [R-L]<sub>0</sub> (initial substrate concentration, nM)</p>"
                f"<p><b>Per-Trace Fixed Parameters:</b></p>"
                f"<ul style='margin:5px 0 10px 20px;'>"
                f"<li><b>[T]</b> &ndash; template concentration (default: {default_T:.2f} nM)</li>"
                f"<li><b>X<sub>0</sub></b> &ndash; initial product concentration (taken from first data point)</li>"
                f"</ul>"
                f"<p><b>Time Units:</b> Minutes</p>"
                f"<p><b>Fitting Procedure:</b></p>"
                f"<p style='font-size:10pt; margin-left:10px;'>"
                f"Global fitting is performed using non-linear least squares (trust region reflective algorithm) "
                f"to minimise residuals across all traces simultaneously. The parameters k and K are shared "
                f"globally, whilst [R-L]<sub>0</sub> is fitted independently for each trace. "
                f"Numerical integration uses LSODA (via <code>scipy.integrate.odeint</code>).</p>"
                f"<p style='font-size:10pt; color:#666; margin-top:10px;'>"
                f"Note: Per-trace [T] values are shown in the results table. "
                f"Different traces may use different template concentrations.</p>"
                f"</div><hr>"
            )
        else:
            # Generic/Unknown model
            model_info_html = (
                f"<div style='font-family:Arial; font-size:11pt; margin-bottom:20px; "
                f"background:#fff3cd; padding:15px; border-left:4px solid #ff9800; border-radius:4px;'>"
                f"<h3 style='margin-top:0; color:#f57c00; font-size:13pt;'>Model Information</h3>"
                f"<p><b>Model Type:</b> Custom or Undefined</p>"
                f"<p>Please select a model type before exporting.</p>"
                f"</div><hr>"
            )

        # Export the plot to a temporary file for embedding
        try:
            exporter = ImageExporter(self.plot_widget.plotItem)
            exporter.parameters()['width'] = 2400  # High resolution for export

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                temp_filename = tmp.name

            exporter.export(temp_filename)
            logger.info("Plot exported to temp file for HTML embedding")

            # Get time window parameters from UI spinboxes
            time_window_start = self.start_time_spinbox.value()
            time_window_end = self.end_time_spinbox.value()

            logger.info("Exporting results with time window: %.4f to %.4f min", time_window_start, time_window_end)

            # Determine if catalytic model is selected
            is_catalytic = self.catalytic_radio and self.catalytic_radio.isChecked()

            # Call the io_utils export with the temp filename, replicate stats, and group info
            self.io_utils.export_results_with_plot_file(
                temp_filename,
                results_html,
                model_info_html,
                self.experiment_info,
                self._last_mean_rate_constant,
                self.fitted_df,
                self.time_col,
                self.replicate_stats_df,  # Pass replicate statistics for separate Excel sheet
                self.replicate_info,  # Pass replicate group metadata
                time_window_start,  # Pass UI time window start for consistent normalisation
                time_window_end,  # Pass UI time window end for validation
                is_catalytic=is_catalytic  # Pass model type for appropriate report formatting
            )
        except Exception as e:
            logger.error("Results export failed: %s", e, exc_info=True)
            QMessageBox.critical(self, "Export Error", f"Results export failed: {e}")

    def export_html_results(self):
        """Export fitting results as standalone HTML file."""
        from PyQt5.QtWidgets import QFileDialog
        from datetime import datetime

        # Get HTML content from results browser
        results_html = self.results_browser.toHtml()

        # Check if there are any results to export
        if not results_html or "Load data and click" in results_html:
            QMessageBox.warning(
                self,
                "No Results",
                "No fitting results available to export. Please fit the model first."
            )
            return

        # Prompt user for save location
        default_filename = f"kinetics_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Results as HTML",
            default_filename,
            "HTML Files (*.html);;All Files (*)"
        )

        if not filename:
            return  # User cancelled

        try:
            # Create complete HTML document with proper styling
            complete_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Kinetics Fitting Results</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            font-size: 12pt;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .header {{
            border-bottom: 3px solid #4caf50;
            padding-bottom: 15px;
            margin-bottom: 25px;
        }}
        .header h1 {{
            margin: 0;
            color: #2e7d32;
            font-size: 20pt;
        }}
        .metadata {{
            color: #666;
            font-size: 11pt;
            margin-top: 10px;
            line-height: 1.6;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            font-size: 11pt;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 10px;
            text-align: left;
        }}
        th {{
            background-color: #4caf50;
            color: white;
            font-weight: bold;
            font-size: 11pt;
        }}
        .summary-box {{
            background: #e8f5e9;
            border-left: 4px solid #4caf50;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
            font-size: 11pt;
        }}
        .summary-box h3 {{
            margin-top: 0;
            color: #2e7d32;
            font-size: 13pt;
        }}
        .summary-box p {{
            font-size: 11pt;
            line-height: 1.6;
        }}
        h2 {{
            color: #2e7d32;
            font-size: 14pt;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Kinetics Fitting Results</h1>
            <div class="metadata">
                <strong>Exported:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
                <strong>Data file:</strong> {self.filename if hasattr(self, 'filename') and self.filename else 'N/A'}<br>
                <strong>Model:</strong> {'Catalytic Turnover' if (self.catalytic_radio and self.catalytic_radio.isChecked()) else 'Bimolecular Reaction'}<br>
                <strong>Time window:</strong> {self.start_time_spinbox.value():.2f} - {self.end_time_spinbox.value():.2f} min
            </div>
        </div>
        <div class="results">
            {results_html}
        </div>
    </div>
</body>
</html>"""

            # Write to file
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(complete_html)

            logger.info("Results exported to HTML: %s", filename)
            QMessageBox.information(
                self,
                "Export Successful",
                f"Results exported successfully to:\n{filename}"
            )

        except Exception as e:
            logger.error("HTML export failed: %s", e, exc_info=True)
            QMessageBox.critical(
                self,
                "Export Error",
                f"Failed to export HTML results:\n{str(e)}"
            )

    # -------------------------------------------------------------------------
    # Endpoint Detection
    # -------------------------------------------------------------------------

    def auto_set_endpoint_window_last_15min(self):
        """
        Automatically set the endpoint detection time window to the last 15 minutes.

        This function is called automatically on data load to provide a sensible
        default for endpoint detection. The window is set to:
        - Start: max_time - 15 minutes (or 0 if data is shorter than 15 min)
        - End: max_time

        Note: These values are NOT saved to the session state, so they will be
        recalculated fresh on every data load.
        """
        if self.data_df is None or self.data_df.empty:
            logger.debug("Cannot auto-set endpoint window: no data loaded")
            return

        if self.time_col not in self.data_df.columns:
            logger.debug("Cannot auto-set endpoint window: time column not found")
            return

        if not self.endpoint_window_start or not self.endpoint_window_end:
            logger.debug("Cannot auto-set endpoint window: spinboxes not initialised")
            return

        # Get time range from data
        time_data = self.data_df[self.time_col]
        max_time = time_data.max()
        min_time = time_data.min()
        total_duration = max_time - min_time

        # Set window length to 15 minutes (or full duration if shorter)
        window_length = min(15.0, total_duration)

        # Calculate window start (15 minutes before end, or data start if shorter)
        # Use floor (int) to ensure the timepoint exists in the dataframe
        endpoint_start = int(max(min_time, max_time - window_length))
        endpoint_end = int(max_time)

        # Update spinbox ranges first (integer spinboxes)
        self.endpoint_window_start.setRange(0, int(max_time * 1.5))
        self.endpoint_window_end.setRange(0, int(max_time * 1.5))

        # Set values
        self.endpoint_window_start.setValue(endpoint_start)
        self.endpoint_window_end.setValue(endpoint_end)

        logger.info(
            "Auto-set endpoint window to last %.1f minutes: %.2f to %.2f min",
            window_length, endpoint_start, endpoint_end
        )

    def detect_and_display_endpoints(self):
        """
        Detect endpoint concentrations from all active traces and display in popup.

        This method:
        1. Validates data availability and time window parameter
        2. Calculates mean signal over specified time window for each trace
        3. Stores results in self.detected_endpoints
        4. Displays results in a popup dialog
        """
        logger.info("Endpoint detection initiated...")

        # Validate data availability
        if self.data_df is None or self.data_df.empty:
            QMessageBox.warning(
                self,
                "No Data",
                "Please load kinetic data before detecting endpoints."
            )
            logger.warning("Endpoint detection failed: no data loaded")
            return

        # Get time window parameters from the new start/end spinboxes
        if not self.endpoint_window_start or not self.endpoint_window_end:
            logger.error("Endpoint window spinboxes not initialised")
            return

        endpoint_start_time = self.endpoint_window_start.value()
        endpoint_end_time = self.endpoint_window_end.value()

        if endpoint_end_time <= endpoint_start_time:
            QMessageBox.warning(
                self,
                "Invalid Window",
                "End time must be greater than start time for endpoint detection."
            )
            logger.warning("Endpoint detection failed: invalid time window (end <= start)")
            return

        # Get full dataset time range
        time_col = self.time_col
        if time_col not in self.data_df.columns:
            QMessageBox.critical(self, "Error", "Time column not found in data.")
            return

        time_data = self.data_df[time_col].values
        max_time = np.max(time_data)
        min_time = np.min(time_data)

        # Calculate window length for logging/display
        window_length_min = endpoint_end_time - endpoint_start_time

        # Validate that window is within data range
        if endpoint_start_time < min_time or endpoint_end_time > max_time:
            QMessageBox.warning(
                self,
                "Window Out of Range",
                f"Specified window ({endpoint_start_time:.2f} - {endpoint_end_time:.2f} min) "
                f"is outside data range ({min_time:.2f} - {max_time:.2f} min).\n\n"
                "Please adjust the time window."
            )
            logger.warning("Endpoint window outside data range")
            return

        # Filter data to endpoint window (between start and end time)
        endpoint_mask = (time_data >= endpoint_start_time) & (time_data <= endpoint_end_time)
        endpoint_df = self.data_df[endpoint_mask]

        if len(endpoint_df) < 2:
            QMessageBox.warning(
                self,
                "Insufficient Data",
                "Not enough data points in the specified endpoint window.\n\n"
                "Try increasing the window length."
            )
            logger.warning("Insufficient data points in endpoint window")
            return

        # Clear previous results
        self.detected_endpoints = {}

        # Get all active data columns (exclude time, fitted, stats)
        data_cols = [
            c for c in self.data_df.columns
            if c != time_col
               and not c.endswith("_fitted")
               and not c.endswith(" Std")
               and not c.endswith(" SEM")
        ]

        # Calculate endpoints for each active trace
        endpoint_results = []
        for col in data_cols:
            # Check if trace is visible
            if not self.trace_settings.get(col, {}).get("show_trace", True):
                continue

            # Extract endpoint window data
            trace_data = endpoint_df[col].astype(float)
            valid_data = trace_data[np.isfinite(trace_data)]

            if len(valid_data) < 2:
                logger.warning("Skipping %s: insufficient valid data in endpoint window", col)
                continue

            # Calculate mean endpoint concentration
            endpoint_conc = np.mean(valid_data)
            endpoint_std = np.std(valid_data, ddof=1) if len(valid_data) > 1 else 0.0

            # Store result
            self.detected_endpoints[col] = {
                'concentration': endpoint_conc,
                'std': endpoint_std,
                'n_points': len(valid_data)
            }

            endpoint_results.append({
                'well': col,
                'concentration': endpoint_conc,
                'std': endpoint_std,
                'n': len(valid_data)
            })

            logger.debug("%s: endpoint = %.3f +/- %.3f nM (n=%d)", col, endpoint_conc, endpoint_std, len(valid_data))

        if not endpoint_results:
            QMessageBox.warning(
                self,
                "No Endpoints Detected",
                "No valid endpoints could be calculated from active traces.\n\n"
                "Ensure traces are visible and contain valid data."
            )
            logger.warning("No endpoints detected from active traces")
            return

        # Display results in popup dialogue
        self._show_endpoint_results_dialog(endpoint_results, window_length_min, endpoint_start_time, endpoint_end_time)

        # Update bimolecular widget status panel with detected endpoints
        self._update_bimolecular_status()

        logger.info("Successfully detected %d endpoints", len(endpoint_results))

    def _update_bimolecular_status(self):
        """Update the bimolecular widget status panel with detected endpoint summary."""
        if not hasattr(self, 'bimolecular_widget') or self.bimolecular_widget is None:
            return

        if not hasattr(self.bimolecular_widget, 'update_status'):
            return

        if not self.detected_endpoints:
            self.bimolecular_widget.update_status("No endpoints detected")
            return

        # Calculate summary statistics
        concentrations = [ep['concentration'] for ep in self.detected_endpoints.values()]
        n_traces = len(concentrations)
        mean_conc = sum(concentrations) / n_traces if n_traces > 0 else 0
        min_conc = min(concentrations) if concentrations else 0
        max_conc = max(concentrations) if concentrations else 0

        # Format status text
        status_text = (
            f"Detected {n_traces} endpoints\n"
            f"Mean [Z]0: {mean_conc:.2f} nM\n"
            f"Range: {min_conc:.2f} - {max_conc:.2f} nM"
        )
        self.bimolecular_widget.update_status(status_text)

    def _show_endpoint_results_dialog(self, results, window_length, start_time, end_time):
        """
        Display endpoint detection results in a popup dialogue.

        Args:
            results: List of dictionaries with well, concentration, std, n
            window_length: Time window length in minutes
            start_time: Start time of endpoint window
            end_time: End time of endpoint window
        """
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem, QPushButton, \
            QLabel
        from PyQt5.QtCore import Qt

        dialogue = QDialog(self)
        dialogue.setWindowTitle("Detected Endpoint Concentrations")
        dialogue.setMinimumWidth(500)
        dialogue.setMinimumHeight(400)

        layout = QVBoxLayout(dialogue)
        layout.setSpacing(10)

        # Header information
        header_label = QLabel(
            f"<b>Endpoint Detection Results</b><br>"
            f"Time window: {start_time:.2f} - {end_time:.2f} min (last {window_length:.2f} min)<br>"
            f"Number of traces analysed: {len(results)}"
        )
        header_label.setStyleSheet("font-size: 10px; padding: 8px; background-color: #E3F2FD; border-radius: 4px;")
        header_label.setWordWrap(True)
        layout.addWidget(header_label)

        # Create table
        table = QTableWidget()
        table.setRowCount(len(results))
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(["Well/Trace", "Endpoint Conc. (nM)", "Std Dev (nM)", "N Points"])

        # Configure table appearance
        table.setStyleSheet("""
            QTableWidget {
                font-size: 9px;
                gridline-color: #CCCCCC;
                background-color: white;
            }
            QTableWidget::item {
                padding: 4px;
            }
            QHeaderView::section {
                background-color: #F5F5F5;
                font-weight: bold;
                font-size: 9px;
                padding: 6px;
                border: 1px solid #CCCCCC;
            }
        """)

        # Populate table
        for i, result in enumerate(results):
            # Well name
            well_item = QTableWidgetItem(result['well'])
            well_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            table.setItem(i, 0, well_item)

            # Concentration
            conc_item = QTableWidgetItem(f"{result['concentration']:.3f}")
            conc_item.setTextAlignment(Qt.AlignCenter)
            table.setItem(i, 1, conc_item)

            # Std dev
            std_item = QTableWidgetItem(f"{result['std']:.3f}")
            std_item.setTextAlignment(Qt.AlignCenter)
            table.setItem(i, 2, std_item)

            # N points
            n_item = QTableWidgetItem(str(result['n']))
            n_item.setTextAlignment(Qt.AlignCenter)
            table.setItem(i, 3, n_item)

        # Adjust column widths
        table.horizontalHeader().setStretchLastSection(False)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)

        layout.addWidget(table)

        # Info label
        info_label = QLabel(
            "ℹ️ These endpoint values represent the mean signal concentration over the final time window.\n"
            "Check 'Use Detected Endpoints' to apply these as [SN]₀ initial values in model fitting."
        )
        info_label.setStyleSheet("font-size: 8px; color: #666; font-style: italic; padding: 4px;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Close button
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        close_button = QPushButton("Close")
        close_button.setStyleSheet(self.gui.button_style_standard)
        close_button.clicked.connect(dialogue.accept)
        close_button.setMinimumWidth(100)
        button_layout.addWidget(close_button)

        layout.addLayout(button_layout)

        # Show dialogue
        dialogue.exec_()

    # -------------------------------------------------------------------------
    # State Saving and Restoration
    # -------------------------------------------------------------------------

    def save_state(self):
        """Delegate state saving to state manager."""
        if self.state_manager:
            self.state_manager.save_state()

    def restore_state(self):
        """Delegate state restoration to state manager."""
        if self.state_manager:
            self.state_manager.restore_state()

    # -------------------------------------------------------------------------
    # Fitting Results Processing
    # -------------------------------------------------------------------------

    def _on_fitting_complete(self, all_results):
        """
        Process fitting results and update UI.

        Parameters
        ----------
        all_results : dict
            Complete fitting results for all traces
        """
        r2_threshold = self.r2_threshold_spinbox.value()
        is_catalytic = self.catalytic_radio and self.catalytic_radio.isChecked()

        # Store fitted curves and parameters
        self._store_fitted_data(all_results, r2_threshold)

        # Render HTML results
        html = format_results_html(all_results, is_catalytic, r2_threshold)
        self.results_browser.setHtml(html)

        # Log global catalytic parameters
        if is_catalytic:
            for result in all_results.values():
                if result.get('success', False) and result.get('model', '').startswith('Catalytic'):
                    global_k = result.get('k_fit_per_min')
                    global_K = result.get('K_fit_nM')
                    if global_K is not None:
                        logger.info("Global catalytic fit: k=%.8f min^-1, K=%.2f nM",
                                   global_k, global_K)
                    else:
                        logger.info("Global catalytic fit (simple): k=%.8f min^-1", global_k)
                    break

        self.update_plot()
        self.save_state()

    def _store_fitted_data(self, all_results, r2_threshold):
        """
        Store fitted curves in fitted_df and fitted parameters for replicate averaging.

        Parameters
        ----------
        all_results : dict
            Complete fitting results for all traces.
        r2_threshold : float
            Minimum R-squared for plotting fitted curves.
        """
        for trace_name, result in all_results.items():
            if not result.get('success', False):
                if 'error' in result:
                    logger.warning("%s: Fit failed - %s", trace_name, result['error'])
                else:
                    logger.warning("%s: Fit unsuccessful", trace_name)
                continue

            model_type = result.get('model', 'Unknown')
            r2 = result.get('r2', 0.0)

            # Store fitted curve if R2 meets threshold
            if r2 >= r2_threshold:
                fitted_values = result.get('y_fit_nM')
                if fitted_values is None:
                    fitted_values = result.get('B1B2_fit_nM')
                if fitted_values is not None:
                    n_rows = len(self.fitted_df)
                    if len(fitted_values) == n_rows:
                        self.fitted_df[trace_name + "_fitted"] = fitted_values
                    else:
                        logger.warning(
                            "%s: Fitted curve length (%d) does not match dataframe length (%d)",
                            trace_name, len(fitted_values), n_rows
                        )
            else:
                logger.info("%s: R2 = %.4f below threshold %.2f, fit curve not plotted",
                           trace_name, r2, r2_threshold)

            # Store fitted parameters for replicate averaging
            if model_type == 'bimolecular':
                kf = result.get('kf_fit', result.get('k_f_fit_M-1_s-1', 0.0))
                self.fitted_parameters[trace_name] = {
                    'k_f': kf,
                    'X0_nM': result.get('fitted_initial', result.get('X0_fit_nM', 0.0)),
                    'Z0_nM': result.get('fixed_initial', result.get('Z0_nM', 0.0)),
                    'r2': r2,
                    'model': model_type
                }
            elif model_type.startswith('Catalytic'):
                self.fitted_parameters[trace_name] = {
                    'k_per_min': result.get('k_fit_per_min', 0.0),
                    'K_nM': result.get('K_fit_nM'),
                    'S10_nM': result.get('S10_fit_nM', 0.0),
                    'T_nM': result.get('T_nM') or result.get('T_fixed_nM') or 0.0,
                    'X0_nM': result.get('X0_nM', 0.0),
                    'r2': r2,
                    'model': model_type
                }

    # -------------------------------------------------------------------------
    # Event Handlers - Widget Lifecycle
    # -------------------------------------------------------------------------

    def closeEvent(self, event):
        """Save state on close so the session can be restored later."""
        try:

            logger.info("Kinetics processor closing - saving state immediately...")
            self.state_manager.save_state_now()
            logger.info("State saved successfully on close")
        except Exception as e:
            logger.error("Failed to save state on close: %s", e, exc_info=True)
        finally:
            # Accept the event to allow the widget to close
            event.accept()

    # -------------------------------------------------------------------------
    # UI Callbacks - Per-Trace Template Concentration
    # -------------------------------------------------------------------------

    def _populate_per_trace_T_table(self, trace_names):
        """
        Populate the per-trace template [T] table in the catalytic widget.

        Parameters
        ----------
        trace_names : list of str
            List of trace column names to display in the table.
        """
        # The catalytic widget is stored as self.ode_widget (stacked widget index 1)
        if not hasattr(self, 'ode_widget') or self.ode_widget is None:
            return

        if not hasattr(self.ode_widget, 'populate_per_trace_T_table'):
            logger.debug("Catalytic widget does not have populate_per_trace_T_table method")
            return

        # Filter to only data traces (exclude means, std, sem columns)
        filtered_traces = [
            t for t in trace_names
            if not t.endswith(' Mean')
            and not t.endswith(' Std')
            and not t.endswith(' SEM')
        ]

        # Collect saved T_nM values from trace_settings
        saved_values = {}
        for trace_name in filtered_traces:
            if trace_name in self.trace_settings:
                t_val = self.trace_settings[trace_name].get('T_nM')
                if t_val is not None:
                    try:
                        saved_values[trace_name] = float(t_val)
                    except (TypeError, ValueError):
                        logger.warning("Invalid T_nM value for trace %s: %s, using default",
                                       trace_name, t_val)
                        saved_values[trace_name] = -1.0
                else:
                    # If T_nM is None or not set, use -1 (default)
                    saved_values[trace_name] = -1.0

        # Pass saved values to the widget so they are restored
        self.ode_widget.populate_per_trace_T_table(filtered_traces, saved_values=saved_values)
        logger.debug("Populated per-trace T table with %d traces, %d with saved values",
                     len(filtered_traces), len([v for v in saved_values.values() if v >= 0]))

    def _sync_per_trace_T_to_settings(self):
        """
        Sync per-trace [T] values from the catalytic widget to trace_settings.

        This should be called before fitting to ensure trace_settings contains
        the correct T_nM values for each trace.
        """
        # The catalytic widget is stored as self.ode_widget (stacked widget index 1)
        if not hasattr(self, 'ode_widget') or self.ode_widget is None:
            return

        if not hasattr(self.ode_widget, 'get_per_trace_T_values'):
            return

        per_trace_T = self.ode_widget.get_per_trace_T_values()

        for trace_name, T_value in per_trace_T.items():
            if trace_name not in self.trace_settings:
                self.trace_settings[trace_name] = {'show_trace': True}

            if T_value is not None:
                # T_value >= 0 means use this specific value (0 = negative control)
                self.trace_settings[trace_name]['T_nM'] = T_value
                logger.debug("Set T_nM=%.3f for trace %s", T_value, trace_name)
            else:
                # T_value is None means use global default (-1 in UI)
                if 'T_nM' in self.trace_settings[trace_name]:
                    del self.trace_settings[trace_name]['T_nM']

        per_trace_count = sum(1 for t in per_trace_T.values() if t is not None)
        logger.info("Synced per-trace T values: %d traces with custom T", per_trace_count)

    def _sync_per_trace_Z0_to_settings(self):
        """
        Sync per-trace [Z]_0 values from the bimolecular widget to trace_settings.

        This should be called before fitting when in manual input mode.
        """
        if not hasattr(self, 'bimolecular_widget') or self.bimolecular_widget is None:
            return

        if not hasattr(self.bimolecular_widget, 'get_per_trace_Z0_values'):
            return

        per_trace_Z0 = self.bimolecular_widget.get_per_trace_Z0_values()

        for trace_name, Z0_value in per_trace_Z0.items():
            if trace_name not in self.trace_settings:
                self.trace_settings[trace_name] = {'show_trace': True}

            if Z0_value is not None:
                # Z0_value >= 0 means use this specific value
                self.trace_settings[trace_name]['Z0_nM'] = Z0_value
                logger.debug("Set Z0_nM=%.3f for trace %s", Z0_value, trace_name)
            else:
                # Z0_value is None means use global default (-1 in UI)
                if 'Z0_nM' in self.trace_settings[trace_name]:
                    del self.trace_settings[trace_name]['Z0_nM']

        per_trace_count = sum(1 for z in per_trace_Z0.values() if z is not None)
        logger.info("Synced per-trace Z0 values: %d traces with custom [Z]_0", per_trace_count)

    def _sync_endpoints_to_Z0_settings(self):
        """
        Sync detected endpoint values to trace_settings as Z0_nM.

        This converts detected endpoints (originally for TMSD [SN]_0) to the
        unified bimolecular model's [Z]_0 parameter.
        """
        if not self.detected_endpoints:
            logger.warning("No detected endpoints to sync")
            return

        for trace_name, endpoint_data in self.detected_endpoints.items():
            if trace_name not in self.trace_settings:
                self.trace_settings[trace_name] = {'show_trace': True}

            # detected_endpoints stores dicts with concentration, std, n_points
            endpoint_value = endpoint_data.get('concentration', 0.0)

            self.trace_settings[trace_name]['Z0_nM'] = endpoint_value
            logger.debug("Set Z0_nM=%.3f (from endpoint) for trace %s", endpoint_value, trace_name)

        logger.info("Synced %d endpoint values to Z0_nM", len(self.detected_endpoints))

    def _populate_per_trace_Z0_table(self, trace_names):
        """
        Populate the per-trace [Z]_0 table in the bimolecular widget.

        Parameters
        ----------
        trace_names : list of str
            List of trace column names to display in the table.
        """
        if not hasattr(self, 'bimolecular_widget') or self.bimolecular_widget is None:
            return

        if not hasattr(self.bimolecular_widget, 'populate_per_trace_Z0_table'):
            logger.debug("Bimolecular widget does not have populate_per_trace_Z0_table method")
            return

        # Filter to only data traces (exclude means, std, sem columns)
        filtered_traces = [
            t for t in trace_names
            if not t.endswith(' Mean')
            and not t.endswith(' Std')
            and not t.endswith(' SEM')
        ]

        # Collect saved Z0_nM values from trace_settings
        saved_values = {}
        for trace_name in filtered_traces:
            if trace_name in self.trace_settings:
                z_val = self.trace_settings[trace_name].get('Z0_nM')
                if z_val is not None:
                    try:
                        saved_values[trace_name] = float(z_val)
                    except (TypeError, ValueError):
                        logger.warning("Invalid Z0_nM value for trace %s: %s, using default",
                                       trace_name, z_val)
                        saved_values[trace_name] = -1.0

        self.bimolecular_widget.populate_per_trace_Z0_table(filtered_traces, saved_values=saved_values)
        logger.debug("Populated per-trace Z0 table with %d traces, %d with saved values",
                     len(filtered_traces), len([v for v in saved_values.values() if v >= 0]))

    def _update_endpoint_window_defaults(self):
        """
        Set endpoint detection time window to the last 15 minutes of the dataset.

        This is called when new data is loaded to provide sensible defaults
        based on the actual data range.
        """
        if self.data_df is None or self.data_df.empty:
            return

        time_col = self.time_col
        if time_col not in self.data_df.columns:
            return

        # Get dataset time range
        time_data = self.data_df[time_col].values
        max_time = float(np.max(time_data))
        min_time = float(np.min(time_data))

        # Set window to last 15 minutes (or full range if shorter)
        window_duration = 15.0
        if max_time - min_time < window_duration:
            window_duration = max_time - min_time

        # Use floor (int) to ensure timepoints exist in data
        endpoint_start = int(max_time - window_duration)
        endpoint_end = int(max_time)

        # Update the spinboxes if they exist (integer spinboxes)
        if self.endpoint_window_start:
            self.endpoint_window_start.setValue(endpoint_start)
        if self.endpoint_window_end:
            self.endpoint_window_end.setValue(endpoint_end)

        logger.debug("Endpoint window set to %d - %d min (last %.1f min of data)",
                     endpoint_start, endpoint_end, window_duration)

    # -------------------------------------------------------------------------
    # UI Callbacks - Trace Selection
    # -------------------------------------------------------------------------

    def on_trace_selection_changed(self):
        """
        Handle trace selection changes from list widget.
        Updates trace_settings based on current selection and refreshes plot.
        """
        if self.trace_selection_panel and 'manager' in self.trace_selection_panel:
            manager = self.trace_selection_panel['manager']
            visible_traces = manager.get_visible_traces()

            # Get excluded traces
            excluded_traces = []
            if hasattr(manager, 'get_excluded_traces'):
                excluded_traces = manager.get_excluded_traces()

            # Ensure ALL columns are in trace_settings
            if self.data_df is not None:
                time_col = self.time_col
                data_cols = [c for c in self.data_df.columns
                             if c != time_col
                             and not c.endswith("_fitted")
                             and not c.endswith(" Std")
                             and not c.endswith(" SEM")]

                for col in data_cols:
                    if col not in self.trace_settings:
                        self.trace_settings[col] = {'show_trace': True, 'exclude_from_fit': False}

            # Update trace visibility and exclusion based on selection
            for col in self.trace_settings.keys():
                self.trace_settings[col]['show_trace'] = col in visible_traces
                self.trace_settings[col]['exclude_from_fit'] = col in excluded_traces

            n_excluded = len(excluded_traces)
            logger.debug("Selection changed: %d traces visible, %d excluded from fitting",
                        len(visible_traces), n_excluded)

        # Update the plot immediately
        self.update_plot()

        # Auto-save state after selection changes
        self.save_state()
