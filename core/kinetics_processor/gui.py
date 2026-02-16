"""
GUI Components for Kinetics Processor
--------------------------------------
Interface construction for kinetic data analysis.

Author: Križan Jurinović
Date: October 2025
"""

import logging
import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QStackedWidget,
    QGroupBox, QDoubleSpinBox, QCheckBox, QPushButton,
    QTextBrowser, QLabel, QRadioButton, QSpinBox, QLineEdit,
    QDialog, QTextEdit, QScrollArea, QGridLayout,
    QTableWidget, QTableWidgetItem, QHeaderView, QListWidget,
    QListWidgetItem, QTabWidget, QSplitter, QMessageBox, QFrame, QFormLayout,
    QSizePolicy, QButtonGroup, QAbstractSpinBox
)
import pyqtgraph as pg
from PyQt5.QtGui import QColor

# Import centralised time processing utilities
from .data_processor import normalise_time_column, filter_time_window

# Import unified UI theme for consistent styling
from core.common.ui_theme import Colours, UITheme

# Import widget factories
from .gui_widget_factory_bimolecular import create_bimolecular_widget, _create_section_box
from .gui_widget_factory import block_wheel_event

logger = logging.getLogger(__name__)


class KineticsGUI:
    """GUI handler for the Kinetics Processor."""

    def __init__(self, parent_widget, plot_widget):
        """
        Initialise GUI handler.

        Args:
            parent_widget: Parent KineticsProcessor instance
            plot_widget: PyQtGraph PlotWidget for displaying results
        """
        self.parent = parent_widget
        self.plot_widget = plot_widget

        # Incremental plot update tracking (P2 #7)
        # Stores plot items by trace name for selective updates
        self._plot_items = {}  # {trace_name: {'data': PlotDataItem, 'fit': PlotDataItem, 'error': FillBetweenItem}}
        self._last_plot_state = {}  # Stores last parameters used for each trace

        # Use unified colour scheme from ui_theme.py
        self.COLOR_PRIMARY = Colours.ACCENT_BLUE
        self.COLOR_SUCCESS = Colours.ACCENT_GREEN
        self.COLOR_NEUTRAL = Colours.TEXT_SECONDARY
        self.COLOR_BORDER = Colours.BORDER_LIGHT
        self.COLOR_BG_LIGHT = Colours.SECTION_BACKGROUND
        self.COLOR_TEXT_PRIMARY = Colours.TEXT_PRIMARY
        self.COLOR_TEXT_SECONDARY = Colours.TEXT_SECONDARY

        # Define consistent styles using unified theme
        self._define_styles()

    def _define_styles(self):
        """Define consistent style sheets using unified UI theme."""
        # Use unified theme styles for consistency across all Clearissa modules
        self.param_group_style = UITheme.get_groupbox_style()
        self.button_style_standard = UITheme.get_button_style_standard()
        self.button_style_primary = UITheme.get_button_style_primary()
        self.button_style_simulate = UITheme.get_button_style_success()
        self.button_style_compact = UITheme.get_button_style_standard()
        self.label_style_bold = UITheme.get_label_style_secondary()
        self.label_style_info = UITheme.get_label_style_tertiary()

    def _create_compact_spinbox(self, min_val, max_val, default_val, step,
                                is_float=False, decimals=2, callback=None, tooltip=None):
        """
        Create a compact spinbox widget with proper callback handling.

        Parameters
        ----------
        min_val : float or int
            Minimum allowed value.
        max_val : float or int
            Maximum allowed value.
        default_val : float or int
            Initial value to set.
        step : float or int
            Single step increment value.
        is_float : bool, optional
            If True, use QDoubleSpinBox, otherwise QSpinBox.
        decimals : int, optional
            Number of decimal places for QDoubleSpinBox.
        callback : callable or None, optional
            Function to call on value change.
        tooltip : str or None, optional
            Tooltip text for the widget.

        Returns
        -------
        QSpinBox or QDoubleSpinBox
            Configured spinbox widget.
        """
        if is_float:
            sb = QDoubleSpinBox()
            sb.setDecimals(decimals)
            sb.setRange(float(min_val), float(max_val))
            sb.setSingleStep(float(step))
            sb.setValue(float(default_val))
        else:
            sb = QSpinBox()
            sb.setRange(int(min_val), int(max_val))
            sb.setSingleStep(int(step))
            sb.setValue(int(default_val))

        sb.setMaximumWidth(120)
        sb.setStyleSheet("font-size: 9px; font-family: inherit;")
        sb.setFocusPolicy(Qt.StrongFocus)
        sb.setButtonSymbols(QAbstractSpinBox.NoButtons)
        block_wheel_event(sb)

        if tooltip:
            sb.setToolTip(tooltip)

        if callback:
            # Handle both parameter signatures gracefully
            sb.valueChanged.connect(lambda value: callback(value) if callback.__code__.co_argcount > 0 else callback())

        return sb

    # =========================================================================
    # LEFT PANEL SETUP METHODS (for 30:70 layout)
    # =========================================================================

    def setup_data_loading_frame(self, parent_layout, callbacks):
        """Create Data Loading frame for left control rail."""
        frame = QGroupBox("Data Loading")
        frame.setObjectName("frame_data_loading")
        frame.setStyleSheet(self.param_group_style)
        layout = QVBoxLayout(frame)
        layout.setSpacing(6)
        layout.setContentsMargins(8, 8, 8, 8)
        parent_layout.addWidget(frame)

        load_button = QPushButton("📁 Load Kinetic Data")
        load_button.setStyleSheet(self.button_style_primary)
        load_button.setToolTip("Load CSV file containing time-series kinetic data")
        load_button.clicked.connect(callbacks['on_load_data_clicked'])
        layout.addWidget(load_button)

        metadata_button = QPushButton("✏️ Edit Metadata")
        metadata_button.setStyleSheet(self.button_style_standard)
        metadata_button.setToolTip("Add experiment title and notes for reports")
        metadata_button.clicked.connect(callbacks['open_experiment_info_dialog'])
        layout.addWidget(metadata_button)

        replicates_button = QPushButton("🔗 Manage Replicates")
        replicates_button.setStyleSheet(self.button_style_standard)
        replicates_button.setToolTip("Define which traces are replicates for averaging and error analysis")
        replicates_button.clicked.connect(callbacks.get('manage_replicates', lambda: None))
        layout.addWidget(replicates_button)

        status_label = QLabel("No data loaded")
        status_label.setStyleSheet(self.label_style_info)
        status_label.setWordWrap(True)
        layout.addWidget(status_label)

        return {'status_label': status_label, 'replicates_button': replicates_button}

    def setup_time_window_frame(self, parent_layout, default_end_time, update_plot_callback, callbacks):
        """Create Time Window frame for left control rail."""
        frame = QGroupBox("Time Window")
        frame.setObjectName("frame_time_window")
        frame.setStyleSheet(self.param_group_style)
        layout = QVBoxLayout(frame)
        layout.setSpacing(6)
        layout.setContentsMargins(8, 8, 8, 8)
        parent_layout.addWidget(frame)

        time_range_layout = QHBoxLayout()
        time_range_layout.setSpacing(8)

        start_label = QLabel("Start:")
        start_label.setStyleSheet("font-size: 9px; font-weight: bold;")
        time_range_layout.addWidget(start_label)

        start_time_spinbox = QDoubleSpinBox()
        start_time_spinbox.setRange(0.0, default_end_time)
        start_time_spinbox.setValue(0.0)
        start_time_spinbox.setSuffix(" min")
        start_time_spinbox.setDecimals(2)
        start_time_spinbox.setMaximumWidth(100)
        start_time_spinbox.setStyleSheet("font-size: 9px;")
        start_time_spinbox.setToolTip("Start time for analysis window (minutes)")
        time_range_layout.addWidget(start_time_spinbox)

        end_label = QLabel("End:")
        end_label.setStyleSheet("font-size: 9px; font-weight: bold; margin-left: 8px;")
        time_range_layout.addWidget(end_label)

        end_time_spinbox = QDoubleSpinBox()
        end_time_spinbox.setRange(0.0, default_end_time)
        end_time_spinbox.setValue(default_end_time)
        end_time_spinbox.setSuffix(" min")
        end_time_spinbox.setDecimals(2)
        end_time_spinbox.setMaximumWidth(100)
        end_time_spinbox.setStyleSheet("font-size: 9px;")
        end_time_spinbox.setToolTip("End time for analysis window (minutes)")
        time_range_layout.addWidget(end_time_spinbox)

        time_range_layout.addStretch()
        layout.addLayout(time_range_layout)

        status_label = QLabel("Window: 0.00 - 0.00 min | 0 points")
        status_label.setStyleSheet(self.label_style_info)
        layout.addWidget(status_label)

        error_label = QLabel("")
        error_label.setStyleSheet("font-size: 8px; color: #D32F2F; font-weight: bold;")
        error_label.setVisible(False)
        layout.addWidget(error_label)

        start_time_spinbox.valueChanged.connect(update_plot_callback)
        end_time_spinbox.valueChanged.connect(update_plot_callback)

        return {
            'start_time_spinbox': start_time_spinbox,
            'end_time_spinbox': end_time_spinbox,
            'status_label': status_label,
            'error_label': error_label
        }

    def setup_fit_controls_frame(self, parent_layout, callbacks):
        """Create Model Fitting Controls frame."""
        frame = QGroupBox("Model Fitting")
        frame.setObjectName("frame_fit_controls")
        frame.setStyleSheet(self.param_group_style)
        layout = QVBoxLayout(frame)
        layout.setSpacing(6)
        layout.setContentsMargins(8, 8, 8, 8)
        parent_layout.addWidget(frame)

        button_layout = QHBoxLayout()
        button_layout.setSpacing(8)

        fit_model_button = QPushButton("⚡ Fit Model")
        fit_model_button.setStyleSheet(self.button_style_simulate)
        fit_model_button.setToolTip("Fit the selected kinetic model")
        fit_model_button.clicked.connect(callbacks['simulate_and_compare'])
        fit_model_button.setMinimumWidth(100)
        button_layout.addWidget(fit_model_button)

        clear_button = QPushButton("🗑 Clear")
        clear_button.setStyleSheet(self.button_style_standard)
        clear_button.setToolTip("Remove fitted curves")
        clear_button.clicked.connect(callbacks['clear_fits'])
        clear_button.setMinimumWidth(80)
        button_layout.addWidget(clear_button)

        layout.addLayout(button_layout)
        return {'fit_model_button': fit_model_button, 'clear_button': clear_button}

    def setup_trace_visibility_frame(self, parent_layout, on_selection_changed_callback):
        """Create Trace Visibility Controls frame with two-column layout for space efficiency."""
        frame = QGroupBox("Trace Visibility Controls")
        frame.setObjectName("frame_trace_visibility")
        frame.setStyleSheet(UITheme.get_groupbox_style())
        layout = QVBoxLayout(frame)
        layout.setSpacing(4)
        layout.setContentsMargins(6, 8, 6, 6)
        parent_layout.addWidget(frame)

        # Summary row at top
        summary_row = QHBoxLayout()
        summary_row.setSpacing(8)
        summary_label = QLabel("0 shown of 0 total")
        summary_label.setStyleSheet(UITheme.get_header_style_section(size=9))
        summary_row.addWidget(summary_label)
        summary_row.addStretch()
        layout.addLayout(summary_row)

        # Main content: two-column horizontal layout
        content_layout = QHBoxLayout()
        content_layout.setSpacing(8)

        # =====================================================================
        # LEFT COLUMN: Show/Hide Traces
        # =====================================================================
        left_column = QFrame()
        left_column.setStyleSheet(f"""
            QFrame {{
                background-color: {Colours.CARD_BACKGROUND};
                border: 1px solid {Colours.CARD_BORDER};
                border-radius: 0px;
            }}
        """)
        left_layout = QVBoxLayout(left_column)
        left_layout.setContentsMargins(6, 6, 6, 6)
        left_layout.setSpacing(4)

        # Header row with title and buttons
        left_header_row = QHBoxLayout()
        left_header_row.setSpacing(4)

        visibility_header = QLabel("Show/Hide")
        visibility_header.setStyleSheet(UITheme.get_header_style_section(size=10))
        left_header_row.addWidget(visibility_header)

        left_header_row.addStretch()

        select_all_btn = QPushButton("All")
        select_all_btn.setStyleSheet(UITheme.get_button_style_standard())
        select_all_btn.setFixedWidth(55)
        select_all_btn.setToolTip("Show all traces")
        left_header_row.addWidget(select_all_btn)

        deselect_all_btn = QPushButton("None")
        deselect_all_btn.setStyleSheet(UITheme.get_button_style_standard())
        deselect_all_btn.setFixedWidth(55)
        deselect_all_btn.setToolTip("Hide all traces")
        left_header_row.addWidget(deselect_all_btn)

        left_layout.addLayout(left_header_row)

        # Hint text
        visibility_hint = QLabel("Click to toggle visibility")
        visibility_hint.setStyleSheet(UITheme.get_label_style_tertiary())
        left_layout.addWidget(visibility_hint)

        # Scroll area for visibility checkboxes
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setStyleSheet(UITheme.get_scrollarea_style())
        left_scroll.setMinimumHeight(120)

        grid_container = QWidget()
        grid_container.setStyleSheet(f"background-color: {Colours.CARD_BACKGROUND};")
        grid_layout = QGridLayout(grid_container)
        grid_layout.setSpacing(4)
        grid_layout.setHorizontalSpacing(8)
        grid_layout.setContentsMargins(2, 2, 2, 2)
        grid_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        left_scroll.setWidget(grid_container)
        left_layout.addWidget(left_scroll, stretch=1)

        content_layout.addWidget(left_column, stretch=1)

        # =====================================================================
        # RIGHT COLUMN: Exclude from Fitting
        # =====================================================================
        right_column = QFrame()
        right_column.setStyleSheet(f"""
            QFrame {{
                background-color: {Colours.CARD_BACKGROUND};
                border: 1px solid {Colours.CARD_BORDER};
                border-radius: 0px;
            }}
        """)
        right_layout = QVBoxLayout(right_column)
        right_layout.setContentsMargins(6, 6, 6, 6)
        right_layout.setSpacing(4)

        # Header row with title and buttons
        right_header_row = QHBoxLayout()
        right_header_row.setSpacing(4)

        exclude_header = QLabel("Exclude from Fit")
        exclude_header.setStyleSheet(UITheme.get_header_style_section(size=10))
        right_header_row.addWidget(exclude_header)

        right_header_row.addStretch()

        exclude_all_btn = QPushButton("All")
        exclude_all_btn.setStyleSheet(UITheme.get_button_style_standard())
        exclude_all_btn.setFixedWidth(55)
        exclude_all_btn.setToolTip("Exclude all traces from fitting")
        right_header_row.addWidget(exclude_all_btn)

        exclude_none_btn = QPushButton("None")
        exclude_none_btn.setStyleSheet(UITheme.get_button_style_standard())
        exclude_none_btn.setFixedWidth(55)
        exclude_none_btn.setToolTip("Clear all exclusions - fit all visible traces")
        right_header_row.addWidget(exclude_none_btn)

        right_layout.addLayout(right_header_row)

        # Hint text
        exclude_hint = QLabel("Checked traces displayed but not fitted")
        exclude_hint.setStyleSheet(UITheme.get_label_style_tertiary())
        right_layout.addWidget(exclude_hint)

        # Scroll area for exclude checkboxes
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setStyleSheet(UITheme.get_scrollarea_style())
        right_scroll.setMinimumHeight(120)

        exclude_container = QWidget()
        exclude_container.setStyleSheet(f"background-color: {Colours.CARD_BACKGROUND};")
        exclude_grid = QGridLayout(exclude_container)
        exclude_grid.setSpacing(4)
        exclude_grid.setHorizontalSpacing(8)
        exclude_grid.setContentsMargins(2, 2, 2, 2)
        exclude_grid.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        right_scroll.setWidget(exclude_container)
        right_layout.addWidget(right_scroll, stretch=1)

        content_layout.addWidget(right_column, stretch=1)

        layout.addLayout(content_layout, stretch=1)

        # Create manager wrapper for compatibility with grid layout
        class GridWidgetManager:
            """Wrapper to provide manager interface for grid-based checkbox layout."""
            def __init__(self, grid_layout, grid_container):
                self.grid_layout = grid_layout
                self.grid_container = grid_container
                self.checkboxes = []  # List of (checkbox, col_name) tuples
                self.headers = []  # List of header labels

            def get_visible_traces(self):
                """Get list of checked trace column names."""
                visible = []
                for checkbox, col_name in self.checkboxes:
                    if checkbox.isChecked():
                        visible.append(col_name)
                return visible

            def clear_all(self):
                """Clear all checkboxes and headers."""
                while self.grid_layout.count():
                    item = self.grid_layout.takeAt(0)
                    if item.widget():
                        item.widget().deleteLater()
                self.checkboxes.clear()
                self.headers.clear()

            def clear_exclude_checkboxes(self):
                """Clear all exclude checkboxes."""
                if hasattr(self, 'exclude_checkboxes'):
                    while self.exclude_grid.count():
                        item = self.exclude_grid.takeAt(0)
                        if item.widget():
                            item.widget().deleteLater()
                    self.exclude_checkboxes.clear()

            def get_excluded_traces(self):
                """Get list of traces excluded from fitting."""
                excluded = []
                if hasattr(self, 'exclude_checkboxes'):
                    for checkbox, col_name in self.exclude_checkboxes:
                        if checkbox.isChecked():
                            excluded.append(col_name)
                return excluded

        manager = GridWidgetManager(grid_layout, grid_container)

        # Store exclude checkboxes in manager
        manager.exclude_checkboxes = []
        manager.exclude_grid = exclude_grid
        manager.exclude_container = exclude_container

        selection_panel = {
            'frame': frame, 'grid_layout': grid_layout, 'grid_container': grid_container,
            'manager': manager, 'callback': on_selection_changed_callback,
            'select_all_btn': select_all_btn, 'deselect_all_btn': deselect_all_btn,
            'summary_label': summary_label,
            'exclude_frame': None, 'exclude_grid': exclude_grid,
            'exclude_container': exclude_container,
            'exclude_all_btn': exclude_all_btn, 'exclude_none_btn': exclude_none_btn,
        }

        # Connect button signals
        select_all_btn.clicked.connect(lambda: self._on_bulk_operation(selection_panel, 'select_all'))
        deselect_all_btn.clicked.connect(lambda: self._on_bulk_operation(selection_panel, 'deselect_all'))
        exclude_all_btn.clicked.connect(lambda: self._on_bulk_operation(selection_panel, 'exclude_all'))
        exclude_none_btn.clicked.connect(lambda: self._on_bulk_operation(selection_panel, 'exclude_none'))

        return selection_panel

    def setup_fitting_results_frame(self, parent_layout, callbacks):
        """Create Fitting Results frame."""
        frame = QGroupBox("Fitting Results")
        frame.setObjectName("frame_fit_results")
        frame.setStyleSheet(self.param_group_style)
        layout = QVBoxLayout(frame)
        layout.setSpacing(6)
        layout.setContentsMargins(8, 8, 8, 8)
        parent_layout.addWidget(frame)

        # R² threshold control
        threshold_layout = QHBoxLayout()
        threshold_layout.setSpacing(8)

        threshold_label = QLabel("R² Threshold for Quality Metrics:")
        threshold_label.setStyleSheet("font-size: 9px; font-weight: bold;")
        threshold_layout.addWidget(threshold_label)

        r2_threshold_spinbox = QDoubleSpinBox()
        r2_threshold_spinbox.setRange(0.0, 1.0)
        r2_threshold_spinbox.setValue(0.10)
        r2_threshold_spinbox.setDecimals(2)
        r2_threshold_spinbox.setSingleStep(0.05)
        r2_threshold_spinbox.setMaximumWidth(80)
        r2_threshold_spinbox.setStyleSheet("font-size: 9px;")
        r2_threshold_spinbox.setFocusPolicy(Qt.StrongFocus)
        r2_threshold_spinbox.setButtonSymbols(QAbstractSpinBox.NoButtons)
        block_wheel_event(r2_threshold_spinbox)
        r2_threshold_spinbox.setToolTip(
            "Minimum R2 value for quality filtering.\n"
            "Fits below this threshold will NOT be plotted.\n"
            "Default: 0.10 (excludes only very poor fits)"
        )
        threshold_layout.addWidget(r2_threshold_spinbox)

        threshold_layout.addStretch()
        layout.addLayout(threshold_layout)

        results_browser = QTextBrowser()
        results_browser.setStyleSheet(
            f"QTextBrowser {{background-color: white; border: 1px solid {self.COLOR_BORDER}; font-size: 11px;}}")
        results_browser.setMinimumHeight(180)
        results_browser.setMaximumHeight(350)
        results_browser.setHtml(
            "<p style='color: #666; font-style: italic;'>Load data and click 'Fit Model' to see results here.</p>")
        layout.addWidget(results_browser)

        export_html_button = QPushButton("💾 Export HTML Results")
        export_html_button.setStyleSheet(self.button_style_compact)
        export_html_button.setToolTip("Export fitting results as standalone HTML file")
        if callbacks.get('export_html_results'):
            export_html_button.clicked.connect(callbacks['export_html_results'])
        layout.addWidget(export_html_button)

        return {
            'results_browser': results_browser,
            'export_html_button': export_html_button,
            'r2_threshold_spinbox': r2_threshold_spinbox
        }

    def setup_export_frame(self, parent_layout, callbacks):
        """Create Export frame."""
        frame = QGroupBox("Export")
        frame.setObjectName("frame_export")
        frame.setStyleSheet(self.param_group_style)
        layout = QVBoxLayout(frame)
        layout.setSpacing(6)
        layout.setContentsMargins(8, 8, 8, 8)
        parent_layout.addWidget(frame)

        # Create horizontal layout for three equal-width buttons
        button_row = QHBoxLayout()
        button_row.setSpacing(8)

        export_graph_button = QPushButton("📊 Export Graph")
        export_graph_button.setStyleSheet(self.button_style_standard)
        export_graph_button.setToolTip("Export graph as PNG/PDF/SVG")
        export_graph_button.clicked.connect(callbacks['export_hd_graph'])
        button_row.addWidget(export_graph_button)

        export_report_button = QPushButton("📄 Export Report")
        export_report_button.setStyleSheet(self.button_style_standard)
        export_report_button.setToolTip("Export analysis report with plots and data")
        export_report_button.clicked.connect(callbacks['export_results'])
        button_row.addWidget(export_report_button)

        layout.addLayout(button_row)
        layout.addStretch()  # Push buttons to top of frame

        return {
            'export_graph_button': export_graph_button,
            'export_report_button': export_report_button
        }

    # =========================================================================
    # TABBED LAYOUT SETUP (Optimised for Small Screens)
    # =========================================================================

    def setup_tabbed_layout(self, parent_layout, max_time, callbacks):
        """
        Create tabbed interface optimised for small screens.

        Layout:
        - Top: Graph (constant, always visible, expandable)
        - Middle: Fit and Clear buttons (always visible)
        - Bottom: Tabs for different parameter sections

        Parameters
        ----------
        parent_layout : QVBoxLayout
            Main layout to add widgets to
        max_time : float
            Maximum time value for spinboxes
        callbacks : dict
            Dictionary of callback functions

        Returns
        -------
        dict
            Dictionary containing all UI element references
        """
        # Main container with vertical layout
        main_container = QWidget()
        main_layout = QVBoxLayout(main_container)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)
        parent_layout.addWidget(main_container)

        # Create vertical splitter for graph and controls
        splitter = QSplitter(Qt.Vertical)
        splitter.setHandleWidth(4)
        splitter.setChildrenCollapsible(False)

        # ============================================================
        # TOP SECTION: Graph (always visible, takes most space)
        # ============================================================
        graph_container = QWidget()
        graph_container.setStyleSheet("background-color: white;")
        graph_layout = QVBoxLayout(graph_container)
        graph_layout.setContentsMargins(2, 2, 2, 2)
        graph_layout.setSpacing(0)
        graph_layout.addWidget(self.plot_widget)

        # ============================================================
        # BUTTON BAR: Fit, Clear, and Time Window (always visible)
        # ============================================================
        button_bar = QWidget()
        # Darker background for better contrast and visibility
        button_bar.setStyleSheet(f"""
            QWidget {{
                background-color: {Colours.SECTION_BACKGROUND};
                border: 1px solid {Colours.CARD_BORDER};
                border-radius: 0px;
                padding: 3px;
            }}
        """)
        button_layout = QHBoxLayout(button_bar)
        button_layout.setContentsMargins(6, 6, 6, 6)
        button_layout.setSpacing(12)

        fit_button = QPushButton("⚡ Fit Model")
        fit_button.setStyleSheet(self.button_style_simulate)
        fit_button.setToolTip("Fit the selected kinetic model to the data")
        fit_button.clicked.connect(callbacks['simulate_and_compare'])
        fit_button.setMinimumHeight(32)
        button_layout.addWidget(fit_button)

        clear_button = QPushButton("🗑 Clear Fits")
        clear_button.setStyleSheet(self.button_style_standard)
        clear_button.setToolTip("Remove all fitted curves from the plot")
        clear_button.clicked.connect(callbacks['clear_fits'])
        clear_button.setMinimumHeight(32)
        button_layout.addWidget(clear_button)

        # Replicate Averages button (using unified theme)
        replicate_avg_button = QPushButton("📊 Replicate Averages")
        replicate_avg_button.setStyleSheet(UITheme.get_button_style_success())
        replicate_avg_button.setToolTip(
            "View averaged data and fitted curves for replicate groups.\n"
            "Shows range (min-max) across replicates."
        )
        replicate_avg_button.clicked.connect(callbacks.get('generate_replicate_average_plot', lambda: None))
        replicate_avg_button.setMinimumHeight(32)
        button_layout.addWidget(replicate_avg_button)

        button_layout.addSpacing(12)

        # Time Window controls (no separators, clean layout)
        # Plain text labels - no borders or styling
        time_label = QLabel("Time Window:")
        time_label.setStyleSheet("font-size: 9pt; font-weight: 600; color: #212121;")
        button_layout.addWidget(time_label)

        start_time_label = QLabel("Start:")
        start_time_label.setStyleSheet("font-size: 9pt; color: #212121;")
        button_layout.addWidget(start_time_label)

        start_time_spinbox = QDoubleSpinBox()
        start_time_spinbox.setRange(0.0, max_time)
        start_time_spinbox.setValue(0.0)
        start_time_spinbox.setSuffix(" min")
        start_time_spinbox.setDecimals(2)
        start_time_spinbox.setMinimumWidth(90)
        start_time_spinbox.setMaximumHeight(21)  # Compact height matching data_frame_processor
        start_time_spinbox.setAlignment(Qt.AlignLeft)
        start_time_spinbox.setStyleSheet(UITheme.get_spinbox_style(decimals=True))
        start_time_spinbox.setToolTip("Start time for analysis window (minutes)")
        if callbacks.get('update_plot'):
            start_time_spinbox.valueChanged.connect(callbacks['update_plot'])
        button_layout.addWidget(start_time_spinbox)

        end_time_label = QLabel("End:")
        end_time_label.setStyleSheet("font-size: 9pt; color: #212121;")
        button_layout.addWidget(end_time_label)

        end_time_spinbox = QDoubleSpinBox()
        end_time_spinbox.setRange(0.0, max_time)
        end_time_spinbox.setValue(max_time)
        end_time_spinbox.setSuffix(" min")
        end_time_spinbox.setDecimals(2)
        end_time_spinbox.setMinimumWidth(90)
        end_time_spinbox.setMaximumHeight(21)  # Compact height matching data_frame_processor
        end_time_spinbox.setAlignment(Qt.AlignLeft)
        end_time_spinbox.setStyleSheet(UITheme.get_spinbox_style(decimals=True))
        end_time_spinbox.setToolTip("End time for analysis window (minutes)")
        if callbacks.get('update_plot'):
            end_time_spinbox.valueChanged.connect(callbacks['update_plot'])
        button_layout.addWidget(end_time_spinbox)

        button_layout.addSpacing(12)

        # Reset Time Window button with proper sizing
        reset_time_window_button = QPushButton("Reset Time Window")
        reset_time_window_button.setStyleSheet(self.button_style_compact)
        reset_time_window_button.setToolTip("Reset time window to full available range")
        reset_time_window_button.setMinimumWidth(140)  # Ensure text is fully visible
        reset_time_window_button.setMaximumHeight(28)  # Match toolbar button height
        if callbacks.get('reset_time_window'):
            reset_time_window_button.clicked.connect(callbacks['reset_time_window'])
        button_layout.addWidget(reset_time_window_button)

        # Scatter mode checkbox - unified styling
        scatter_view_checkbox = QCheckBox("Scatter")
        scatter_view_checkbox.setStyleSheet(UITheme.get_checkbox_style())
        scatter_view_checkbox.setToolTip("Toggle between line and scatter plot display")
        button_layout.addWidget(scatter_view_checkbox)

        button_layout.addStretch()

        graph_layout.addWidget(button_bar)


        # ============================================================
        # BOTTOM SECTION: Tabbed controls
        # ============================================================
        tab_widget = QTabWidget()
        tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #CCC;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #E0E0E0;
                border: 1px solid #CCC;
                border-bottom: none;
                padding: 6px 16px;
                margin-right: 2px;
                font-size: 9px;
                font-weight: bold;
                min-width: 0px;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom: 1px solid white;
            }
            QTabBar::tab:hover {
                background-color: #F0F0F0;
            }
        """)

        # Tab 1: Data & Setup
        tab_data = QWidget()
        tab_data_layout = QVBoxLayout(tab_data)
        tab_data_layout.setContentsMargins(8, 8, 8, 8)
        tab_data_layout.setSpacing(8)

        # Data loading section
        data_widgets = self.setup_data_loading_frame(tab_data_layout, callbacks)

        tab_data_layout.addStretch()
        tab_widget.addTab(tab_data, "📁 Data & Setup")

        # Tab 2: Model Parameters
        tab_model = QWidget()
        tab_model_layout = QVBoxLayout(tab_model)
        tab_model_layout.setContentsMargins(8, 8, 8, 8)
        tab_model_layout.setSpacing(8)

        # Add scroll area for model parameters
        scroll_model = QScrollArea()
        scroll_model.setWidgetResizable(True)
        scroll_model.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_model.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_model.setStyleSheet("QScrollArea { background-color: white; border: none; }")

        model_container = QWidget()
        model_container_layout = QVBoxLayout(model_container)
        model_container_layout.setContentsMargins(0, 0, 0, 0)
        model_container_layout.setSpacing(12)

        params_widgets = self.setup_reaction_model_parameters(model_container_layout, callbacks)

        scroll_model.setWidget(model_container)
        tab_model_layout.addWidget(scroll_model)

        tab_widget.addTab(tab_model, "⚙️ Model Parameters")

        # Tab 3: Trace Visibility
        tab_traces = QWidget()
        tab_traces_layout = QVBoxLayout(tab_traces)
        tab_traces_layout.setContentsMargins(8, 8, 8, 8)
        tab_traces_layout.setSpacing(8)

        trace_selection_panel = self.setup_trace_visibility_frame(
            tab_traces_layout,
            callbacks.get('on_trace_selection_changed')
        )

        tab_widget.addTab(tab_traces, "👁️ Trace Visibility")

        # Tab 4: Results
        tab_results = QWidget()
        tab_results_layout = QVBoxLayout(tab_results)
        tab_results_layout.setContentsMargins(8, 8, 8, 8)
        tab_results_layout.setSpacing(8)

        results_widgets = self.setup_fitting_results_frame(tab_results_layout, callbacks)

        tab_widget.addTab(tab_results, "📊 Results")

        # Tab 5: Export
        tab_export = QWidget()
        tab_export_layout = QVBoxLayout(tab_export)
        tab_export_layout.setContentsMargins(8, 8, 8, 8)
        tab_export_layout.setSpacing(8)

        export_widgets = self.setup_export_frame(tab_export_layout, callbacks)

        tab_widget.addTab(tab_export, "💾 Export")

        # Add to splitter
        splitter.addWidget(graph_container)
        splitter.addWidget(tab_widget)

        # Set initial sizes (graph gets 75% of space, controls get 25%)
        splitter.setStretchFactor(0, 75)
        splitter.setStretchFactor(1, 25)

        main_layout.addWidget(splitter)

        # Return all widget references
        return {
            'splitter': splitter,
            'tab_widget': tab_widget,
            'fit_model_button': fit_button,
            'clear_button': clear_button,
            'start_time_spinbox': start_time_spinbox,
            'end_time_spinbox': end_time_spinbox,
            'reset_time_window_button': reset_time_window_button,
            'scatter_view_checkbox': scatter_view_checkbox,
            'trace_selection_panel': trace_selection_panel,
            **data_widgets,
            **params_widgets,
            **results_widgets
        }

    # =========================================================================
    # RIGHT-TOP PANEL SETUP (Reaction Model Parameters)
    # =========================================================================

    def setup_reaction_model_parameters(self, parent_layout, callbacks):
        """
        Create Reaction Model Parameters section with 3-column bimolecular layout.

        Layout:
        - Top row: Compact model selector (Bimolecular | Catalytic)
        - Below: Full-width stacked widget for model-specific parameters

        The bimolecular widget uses a 3-column horizontal layout:
        - Column 1: Model context (narrow, fixed)
        - Column 2: Z concentration workflow (wide, stretches)
        - Column 3: Global parameters (medium, fixed)
        """
        main_container = QWidget()
        main_layout = QVBoxLayout(main_container)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(8)
        parent_layout.addWidget(main_container)

        # -------------------------------------------------------------------------
        # Top row: Compact model selector
        # -------------------------------------------------------------------------
        selector_row = QHBoxLayout()
        selector_row.setSpacing(16)

        selector_label = QLabel("Model:")
        selector_label.setStyleSheet(UITheme.get_label_style_secondary())
        selector_row.addWidget(selector_label)

        bimolecular_radio = QRadioButton("Bimolecular Reaction")
        bimolecular_radio.setStyleSheet(UITheme.get_radiobutton_style())
        bimolecular_radio.setChecked(True)
        bimolecular_radio.setToolTip(
            "Single-step bimolecular reaction: X + Z \u2192 Y + W\n"
            "Fits: dY/dt = k_f \u00d7 (X\u2080 - Y) \u00d7 (Z\u2080 - Y)\n"
            "Covers TMSD, internal TMSD, and HMSD assays."
        )
        bimolecular_radio.toggled.connect(callbacks['on_mode_changed'])
        selector_row.addWidget(bimolecular_radio)

        catalytic_radio = QRadioButton("Catalytic Turnover")
        catalytic_radio.setStyleSheet(UITheme.get_radiobutton_style())
        catalytic_radio.setToolTip(
            "Catalytic turnover with Michaelis-Menten-like kinetics.\n"
            "Template-mediated reactions with enzyme-like behavior."
        )
        catalytic_radio.toggled.connect(callbacks['on_mode_changed'])
        selector_row.addWidget(catalytic_radio)

        selector_row.addStretch()
        main_layout.addLayout(selector_row)

        # Thin separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet(UITheme.get_separator_style())
        main_layout.addWidget(separator)

        # -------------------------------------------------------------------------
        # Stacked widget for model parameters (full width)
        # -------------------------------------------------------------------------
        stacked_widget = QStackedWidget()

        # Index 0: Bimolecular widget with 3-column layout
        bimolecular_widget = create_bimolecular_widget(
            callbacks={
                'on_detect_endpoints': callbacks.get('on_detect_endpoints')
            }
        )
        stacked_widget.addWidget(bimolecular_widget)

        # Index 1: Catalytic parameters
        stacked_widget.addWidget(self._create_model_params_widget('catalytic', None, compact=True))

        stacked_widget.setCurrentIndex(0)
        main_layout.addWidget(stacked_widget, stretch=1)

        # Extract endpoint widgets for external access
        endpoint_widgets = bimolecular_widget.widgets

        return {
            'bimolecular_radio': bimolecular_radio,
            'catalytic_radio': catalytic_radio,
            'stacked_widget': stacked_widget,
            'bimolecular_widget': bimolecular_widget,
            'endpoint_window_start': endpoint_widgets.get('endpoint_window_start'),
            'endpoint_window_end': endpoint_widgets.get('endpoint_window_end'),
            'detect_endpoints_button': endpoint_widgets.get('detect_endpoints_button'),
        }

    # =========================================================================
    # MODEL-SPECIFIC PARAMETER WIDGETS (Unified Approach)
    # =========================================================================

    def _create_model_params_widget(self, model_type, callback, compact=False):
        """
        Unified parameter widget creator for kinetic models.

        Parameters
        ----------
        model_type : str
            Model type identifier ('catalytic')
        callback : callable
            Function to call when values change
        compact : bool
            If True, creates compact layout for right panel

        Returns
        -------
        QWidget
            Configured parameter widget
        """
        if model_type == 'catalytic':
            return self._create_catalytic_params(callback, compact)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _create_catalytic_params(self, callback, compact=False):
        """
        Create Catalytic Turnover parameter inputs with 3-column boxed layout.

        Uses the same visual factory pattern as the bimolecular widget for
        consistent styling (white boxed sections, blue headers, UITheme spinboxes).

        Layout:
          Column 1: Model identity with full/simple mode selector
          Column 2: Parameters (fixed, initial guesses)
          Column 3: Per-trace template [T] table
        """
        from .gui_widget_factory import PerTraceTable

        widget = QWidget()
        widget.setStyleSheet(f"background-color: {Colours.MAIN_BACKGROUND};")

        main_layout = QHBoxLayout(widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(12)

        widgets = {}

        # =====================================================================
        # COLUMN 1: Model Identity (narrow, fixed width)
        # =====================================================================
        col1 = QWidget()
        col1.setStyleSheet(f"background-color: {Colours.MAIN_BACKGROUND};")
        col1_layout = QVBoxLayout(col1)
        col1_layout.setContentsMargins(0, 0, 0, 0)
        col1_layout.setSpacing(8)

        # Box 1: Model Description with sub-model selector
        box1, box1_layout = _create_section_box("Reaction Model")

        # Equation label (updated dynamically by sub-model switch)
        equation_label = QLabel("dY/dt = k S/(K+S) T")
        equation_label.setStyleSheet(f"""
            QLabel {{
                font-size: 11pt;
                font-weight: bold;
                padding: 6px 8px;
                background-color: {Colours.INFO_PANEL_INTERNAL_BG};
                border-left: 3px solid {Colours.INFO_PANEL_INTERNAL_BORDER};
                border-radius: 0px;
                color: {Colours.INFO_PANEL_INTERNAL_TEXT};
            }}
        """)
        box1_layout.addWidget(equation_label)

        desc_label = QLabel("Michaelis-Menten (n=1)")
        desc_label.setStyleSheet(UITheme.get_label_style_tertiary())
        desc_label.setToolTip(
            "Full catalytic turnover model:\n"
            "  dY/dt = k S(t) / (K + S(t)) T\n\n"
            "where S(t) = [R-L]0 - Y(t) (mass balance)\n\n"
            "Fitted: k, K (global), [R-L]0 (per-trace)\n"
            "Fixed: T (per-trace), Y(t0) from data"
        )
        box1_layout.addWidget(desc_label)

        # Sub-model selector: Full vs Simple
        mode_box, mode_box_layout = _create_section_box("Model Variant")

        full_model_radio = QRadioButton("Full (with saturation)")
        full_model_radio.setStyleSheet(UITheme.get_radiobutton_style())
        full_model_radio.setChecked(True)
        full_model_radio.setToolTip(
            "Full Michaelis-Menten model with saturation:\n"
            "  dY/dt = k S / (K + S) T\n\n"
            "Fits: k, K (global), [R-L]0 (per-trace)"
        )
        mode_box_layout.addWidget(full_model_radio)

        simple_model_radio = QRadioButton("Simple (no saturation)")
        simple_model_radio.setStyleSheet(UITheme.get_radiobutton_style())
        simple_model_radio.setToolTip(
            "Simplified first-order model without saturation:\n"
            "  dY/dt = k S T\n\n"
            "Fits: k (global), [R-L]0 (per-trace)\n"
            "K parameter is removed."
        )
        mode_box_layout.addWidget(simple_model_radio)

        model_mode_group = QButtonGroup(widget)
        model_mode_group.addButton(full_model_radio, 0)
        model_mode_group.addButton(simple_model_radio, 1)

        widgets['full_model_radio'] = full_model_radio
        widgets['simple_model_radio'] = simple_model_radio

        col1_layout.addWidget(box1)
        col1_layout.addWidget(mode_box)

        # Box 3: Fitting summary
        box_fit, box_fit_layout = _create_section_box("Fitting")

        notes_label = QLabel("Global: k, K\nPer-trace: [R-L]0")
        notes_label.setStyleSheet(f"""
            QLabel {{
                font-size: 9pt;
                font-weight: 500;
                padding: 4px 8px;
                background-color: {Colours.SECTION_BACKGROUND};
                border-radius: 0px;
                color: {Colours.TEXT_SECONDARY};
            }}
        """)
        box_fit_layout.addWidget(notes_label)

        col1_layout.addWidget(box_fit)
        col1_layout.addStretch()

        col1.setFixedWidth(180)
        col1.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        main_layout.addWidget(col1)

        # =====================================================================
        # COLUMN 2: Parameters (medium width)
        # =====================================================================
        col2 = QWidget()
        col2.setStyleSheet(f"background-color: {Colours.MAIN_BACKGROUND};")
        col2_layout = QVBoxLayout(col2)
        col2_layout.setContentsMargins(0, 0, 0, 0)
        col2_layout.setSpacing(8)

        # Box 1: Fixed Parameters
        box_fixed, box_fixed_layout = _create_section_box("Fixed Parameters")

        fixed_form = QFormLayout()
        fixed_form.setSpacing(10)
        fixed_form.setContentsMargins(0, 4, 0, 0)
        fixed_form.setLabelAlignment(Qt.AlignRight)

        fluorescence_spinbox = QDoubleSpinBox()
        fluorescence_spinbox.setRange(0.1, 1000.0)
        fluorescence_spinbox.setValue(10.0)
        fluorescence_spinbox.setDecimals(1)
        fluorescence_spinbox.setSingleStep(1.0)
        fluorescence_spinbox.setMinimumWidth(120)
        fluorescence_spinbox.setStyleSheet(UITheme.get_spinbox_style(decimals=True))
        fluorescence_spinbox.setFocusPolicy(Qt.StrongFocus)
        fluorescence_spinbox.setButtonSymbols(QAbstractSpinBox.NoButtons)
        block_wheel_event(fluorescence_spinbox)
        fluorescence_spinbox.setToolTip("Fluorescence normalisation: RF=1.0 corresponds to this concentration in nM")
        widgets['fluorescence_full_scale_nM_spinbox'] = fluorescence_spinbox

        scale_label = QLabel("C<sub>ref</sub> (nM):")
        scale_label.setStyleSheet(UITheme.get_label_style_primary())
        fixed_form.addRow(scale_label, fluorescence_spinbox)

        box_fixed_layout.addLayout(fixed_form)
        col2_layout.addWidget(box_fixed)

        # Box 2: Initial Guesses
        box_guesses, box_guesses_layout = _create_section_box("Initial Guesses")

        guesses_form = QFormLayout()
        guesses_form.setSpacing(10)
        guesses_form.setContentsMargins(0, 4, 0, 0)
        guesses_form.setLabelAlignment(Qt.AlignRight)

        S10_spinbox = QDoubleSpinBox()
        S10_spinbox.setRange(0.1, 1000.0)
        S10_spinbox.setValue(10.0)
        S10_spinbox.setDecimals(2)
        S10_spinbox.setSingleStep(1.0)
        S10_spinbox.setMinimumWidth(120)
        S10_spinbox.setStyleSheet(UITheme.get_spinbox_style(decimals=True))
        S10_spinbox.setFocusPolicy(Qt.StrongFocus)
        S10_spinbox.setButtonSymbols(QAbstractSpinBox.NoButtons)
        block_wheel_event(S10_spinbox)
        S10_spinbox.setToolTip("Initial guess for [R-L]0 (substrate pool) in nM")
        widgets['catalytic_S10_guess_spinbox'] = S10_spinbox

        S10_label = QLabel("[R-L]0 guess:")
        S10_label.setStyleSheet(UITheme.get_label_style_primary())
        guesses_form.addRow(S10_label, S10_spinbox)

        k_spinbox = QDoubleSpinBox()
        k_spinbox.setRange(1e-6, 1e6)
        k_spinbox.setValue(1.0)
        k_spinbox.setDecimals(4)
        k_spinbox.setSingleStep(0.1)
        k_spinbox.setMinimumWidth(120)
        k_spinbox.setStyleSheet(UITheme.get_spinbox_style(decimals=True))
        k_spinbox.setFocusPolicy(Qt.StrongFocus)
        k_spinbox.setButtonSymbols(QAbstractSpinBox.NoButtons)
        block_wheel_event(k_spinbox)
        k_spinbox.setToolTip("Initial guess for rate constant k (min-1)")
        widgets['catalytic_k_guess_spinbox'] = k_spinbox

        k_label = QLabel("k guess (min-1):")
        k_label.setStyleSheet(UITheme.get_label_style_primary())
        guesses_form.addRow(k_label, k_spinbox)

        K_spinbox = QDoubleSpinBox()
        K_spinbox.setRange(1e-6, 1e9)
        K_spinbox.setValue(10.0)
        K_spinbox.setDecimals(2)
        K_spinbox.setSingleStep(1.0)
        K_spinbox.setMinimumWidth(120)
        K_spinbox.setStyleSheet(UITheme.get_spinbox_style(decimals=True))
        K_spinbox.setFocusPolicy(Qt.StrongFocus)
        K_spinbox.setButtonSymbols(QAbstractSpinBox.NoButtons)
        block_wheel_event(K_spinbox)
        K_spinbox.setToolTip("Initial guess for Michaelis constant K (nM)")
        widgets['catalytic_K_guess_spinbox'] = K_spinbox

        K_label = QLabel("K guess (nM):")
        K_label.setStyleSheet(UITheme.get_label_style_primary())
        guesses_form.addRow(K_label, K_spinbox)

        # Store references for toggling visibility
        widgets['K_label'] = K_label
        widgets['K_spinbox'] = K_spinbox

        box_guesses_layout.addLayout(guesses_form)
        col2_layout.addWidget(box_guesses)

        col2_layout.addStretch()
        col2.setFixedWidth(200)
        col2.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        main_layout.addWidget(col2)

        # =====================================================================
        # COLUMN 3: Per-Trace Template [T] (stretches)
        # =====================================================================
        col3 = QWidget()
        col3.setStyleSheet(f"background-color: {Colours.MAIN_BACKGROUND};")
        col3_layout = QVBoxLayout(col3)
        col3_layout.setContentsMargins(0, 0, 0, 0)
        col3_layout.setSpacing(8)

        box_per_trace, box_per_trace_layout = _create_section_box("Per-Trace Template [T]")

        # Instruction text
        per_trace_info = QLabel("Enter the template concentration [T] for each trace:")
        per_trace_info.setStyleSheet(UITheme.get_label_style_tertiary())
        box_per_trace_layout.addWidget(per_trace_info)

        # Control row with [T] default spinbox and Apply button
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        default_T_label = QLabel("[T] default:")
        default_T_label.setStyleSheet(UITheme.get_label_style_primary())
        btn_row.addWidget(default_T_label)

        default_T_spinbox = QDoubleSpinBox()
        default_T_spinbox.setRange(0.001, 1000.0)
        default_T_spinbox.setValue(1.0)
        default_T_spinbox.setDecimals(3)
        default_T_spinbox.setSingleStep(0.1)
        default_T_spinbox.setFixedWidth(80)
        default_T_spinbox.setStyleSheet(UITheme.get_spinbox_style(decimals=True))
        default_T_spinbox.setFocusPolicy(Qt.StrongFocus)
        default_T_spinbox.setButtonSymbols(QAbstractSpinBox.NoButtons)
        block_wheel_event(default_T_spinbox)
        default_T_spinbox.setToolTip("Default template concentration [T] in nM to apply to all traces")
        widgets['catalytic_template_T_spinbox'] = default_T_spinbox
        btn_row.addWidget(default_T_spinbox)

        apply_global_T_btn = QPushButton("Apply to All")
        apply_global_T_btn.setStyleSheet(UITheme.get_button_style_standard())
        apply_global_T_btn.setToolTip("Set all traces to the default [T] value")
        btn_row.addWidget(apply_global_T_btn)
        btn_row.addStretch()
        box_per_trace_layout.addLayout(btn_row)

        # Table for per-trace T values (4-column layout)
        per_trace_T_table = QTableWidget()
        per_trace_T_table.setColumnCount(4)
        per_trace_T_table.setHorizontalHeaderLabels(["Trace", "[T] (nM)", "Trace", "[T] (nM)"])
        per_trace_T_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        per_trace_T_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Fixed)
        per_trace_T_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        per_trace_T_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Fixed)
        per_trace_T_table.setColumnWidth(1, 100)
        per_trace_T_table.setColumnWidth(3, 100)
        per_trace_T_table.horizontalHeader().setStretchLastSection(False)
        per_trace_T_table.setStyleSheet(UITheme.get_table_style())
        per_trace_T_table.setToolTip("Enter the template concentration [T] for each trace")
        per_trace_T_table.verticalHeader().setVisible(False)
        per_trace_T_table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        box_per_trace_layout.addWidget(per_trace_T_table, stretch=1)

        col3_layout.addWidget(box_per_trace, stretch=1)
        col3.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        main_layout.addWidget(col3, stretch=1)

        # Per-trace T table manager
        t_table_manager = PerTraceTable(
            table=per_trace_T_table,
            min_val=0.0, max_val=10000.0, decimals=3, step=0.1,
            tooltip_template="Template [T] for {name} in nM",
            use_default_marker=False,
        )

        widgets['per_trace_T_table'] = per_trace_T_table
        widgets['per_trace_T_spinboxes'] = t_table_manager.spinboxes
        widgets['apply_global_T_btn'] = apply_global_T_btn

        # =====================================================================
        # SUB-MODEL SWITCHING LOGIC
        # =====================================================================
        def on_sub_model_changed(checked):
            """Handle full/simple model radio button toggle."""
            if not checked:
                return
            is_simple = simple_model_radio.isChecked()
            # Toggle K parameter visibility
            K_label.setVisible(not is_simple)
            K_spinbox.setVisible(not is_simple)
            # Update equation display
            if is_simple:
                equation_label.setText("dY/dt = k S T")
                desc_label.setText("First-order (no saturation)")
                notes_label.setText("Global: k\nPer-trace: [R-L]0")
            else:
                equation_label.setText("dY/dt = k S/(K+S) T")
                desc_label.setText("Michaelis-Menten (n=1)")
                notes_label.setText("Global: k, K\nPer-trace: [R-L]0")

        full_model_radio.toggled.connect(on_sub_model_changed)
        simple_model_radio.toggled.connect(on_sub_model_changed)

        def get_catalytic_sub_model():
            """Return current catalytic sub-model type: 'full' or 'simple'."""
            return 'simple' if simple_model_radio.isChecked() else 'full'

        def apply_global_T_to_all():
            """Set all per-trace T values to the default [T] value."""
            t_table_manager.set_all(widgets['catalytic_template_T_spinbox'].value())

        apply_global_T_btn.clicked.connect(apply_global_T_to_all)

        # Store methods on widget (delegate to PerTraceTable)
        widget.populate_per_trace_T_table = t_table_manager.populate
        widget.get_per_trace_T_values = t_table_manager.get_values
        widget.get_all_per_trace_T_values = t_table_manager.get_all_values
        widget.set_last_per_trace_T_values = t_table_manager.set_last_values
        widget.apply_global_T_to_all = apply_global_T_to_all
        widget.get_catalytic_sub_model = get_catalytic_sub_model

        widget.widgets = widgets
        return widget


    # =========================================================================
    # BULK SELECTION OPERATIONS
    # =========================================================================

    def _on_bulk_operation(self, selection_panel, operation):
        """
        Handle bulk selection operations for trace visibility.

        Parameters
        ----------
        selection_panel : dict
            Dictionary containing selection panel components
        operation : str
            One of 'select_all', 'deselect_all', 'exclude_all', 'exclude_none'
        """
        manager = selection_panel['manager']

        # Block signals during bulk update
        for checkbox, col_name in manager.checkboxes:
            checkbox.blockSignals(True)

        if operation == 'select_all':
            for checkbox, col_name in manager.checkboxes:
                checkbox.setChecked(True)

        elif operation == 'deselect_all':
            for checkbox, col_name in manager.checkboxes:
                checkbox.setChecked(False)

        elif operation == 'exclude_all':
            # Check all exclude checkboxes (exclude all traces from fitting)
            if hasattr(manager, 'exclude_checkboxes'):
                for checkbox, col_name in manager.exclude_checkboxes:
                    checkbox.blockSignals(True)
                    checkbox.setChecked(True)
                    checkbox.blockSignals(False)
            # Trigger callback to update trace settings
            selection_panel['callback']()
            logger.debug("Excluded all traces from fitting")
            return

        elif operation == 'exclude_none':
            # Clear all exclude checkboxes (fit all traces)
            if hasattr(manager, 'exclude_checkboxes'):
                for checkbox, col_name in manager.exclude_checkboxes:
                    checkbox.blockSignals(True)
                    checkbox.setChecked(False)
                    checkbox.blockSignals(False)
            # Trigger callback to update trace settings
            selection_panel['callback']()
            logger.debug("Cleared all fit exclusions")
            return
        else:
            logger.warning("Unknown bulk operation: %s", operation)
            for checkbox, col_name in manager.checkboxes:
                checkbox.blockSignals(False)
            return

        # Unblock signals
        for checkbox, col_name in manager.checkboxes:
            checkbox.blockSignals(False)

        # Update summary
        self._update_trace_summary(selection_panel)

        # Trigger callback to update plot
        selection_panel['callback']()

        logger.debug("Bulk operation '%s' completed", operation)

    def _update_trace_summary(self, selection_panel):
        """
        Update the summary label showing how many traces are visible.

        Parameters
        ----------
        selection_panel : dict
            Dictionary containing selection panel components
        """
        if 'summary_label' not in selection_panel:
            return

        manager = selection_panel['manager']
        total = len(manager.checkboxes)
        visible = sum(1 for checkbox, _ in manager.checkboxes if checkbox.isChecked())

        selection_panel['summary_label'].setText(f"{visible} shown of {total} total")

    # =========================================================================
    # PLOTTING AND VISUALISATION
    # =========================================================================

    def _compute_trace_state_hash(self, col, visible, has_fit, scatter_mode, colour,
                                   time_hash, data_hash, fit_hash=None, has_error=False,
                                   scale_factor=None):
        """
        Compute a state hash for a trace to detect changes.

        Returns a tuple representing the trace's current state.
        """
        return (col, visible, has_fit, scatter_mode, colour,
                time_hash, data_hash, fit_hash, has_error, scale_factor)

    def _remove_plot_item(self, trace_name):
        """Remove plot items for a specific trace."""
        if trace_name not in self._plot_items:
            return

        items = self._plot_items[trace_name]
        for item_type in ['data', 'fit', 'error']:
            if item_type in items and items[item_type] is not None:
                try:
                    self.plot_widget.removeItem(items[item_type])
                except Exception as e:
                    logger.debug("Failed to remove plot item %s for %s: %s",
                               item_type, trace_name, e)

        del self._plot_items[trace_name]

    def _clear_all_plot_items(self):
        """Clear all tracked plot items."""
        for trace_name in list(self._plot_items.keys()):
            self._remove_plot_item(trace_name)
        self._plot_items.clear()
        self._last_plot_state.clear()
        try:
            self.plot_widget.clear()
        except Exception as e:
            logger.error("Failed to clear plot widget: %s", e)

    def update_plot(
        self,
        data_df,
        fitted_df,
        time_col,
        trace_settings,
        start_time,
        end_time,
        replicate_info,
        show_legend,
        catalytic_scale_factor=None
    ):
        """
        Update the plot with current data and fitted curves using incremental updates.

        Only redraws traces that have changed, keeping unchanged traces for better performance.

        Parameters
        ----------
        data_df : pd.DataFrame
            Experimental data
        fitted_df : pd.DataFrame
            Fitted curve data
        time_col : str
            Name of time column
        trace_settings : dict
            Visibility settings for each trace
        start_time : float
            Start of time window (minutes)
        end_time : float
            End of time window (minutes)
        replicate_info : dict
            Information about replicate groups
        show_legend : bool
            Whether to show legend
        catalytic_scale_factor : float or None
            If provided (Catalytic model active), scale experimental data from
            normalised units (0-1) to nM by multiplying by this factor.
            This ensures experimental traces align with fitted curves which are in nM.
        """
        # Safety check: ensure plot_widget exists and is valid
        if self.plot_widget is None:
            logger.error("Plot widget is None - cannot update plot")
            return

        if data_df is None or data_df.empty:
            logger.debug("No data to plot")
            # Clear all plot items if no data
            self._clear_all_plot_items()
            return

        # Copy data for plotting
        plot_df = data_df.copy()
        fitted_plot_df = fitted_df.copy() if fitted_df is not None else None

        # Filter to time window using centralised utility
        plot_df = filter_time_window(plot_df, time_col, start_time, end_time)
        if fitted_plot_df is not None:
            fitted_plot_df = filter_time_window(fitted_plot_df, time_col, start_time, end_time)

        if plot_df.empty:
            logger.debug("No data in selected time window")
            self._clear_all_plot_items()
            return

        # Normalise time for plot display using FIRST DATA POINT
        # This ensures plot shows same time scale as exported data (starting at 0)
        plot_df_display = plot_df.copy()
        first_time_point = plot_df[time_col].min()
        normalise_time_column(plot_df_display, time_col, first_time_point)
        time_array = plot_df_display[time_col].values

        # Determine scatter mode state
        scatter_mode = getattr(self.parent, 'scatter_state', False)

        # Colour palette
        colours = self.parent.default_colours if hasattr(self.parent, 'default_colours') else [
            '#E6194B', '#3CB44B', '#0082C8', '#F58231', '#911EB4',
            '#46F0F0', '#F032E6', '#D2F53C', '#FABEBE', '#008080'
        ]

        # Incremental update strategy: compute state fingerprints for each trace to enable
        # selective redrawing. Hash-based change detection avoids unnecessary plot operations
        # by comparing data arrays, visibility settings, and rendering parameters against
        # the previous update cycle. This approach significantly improves performance when
        # toggling trace visibility or adjusting time windows, as unchanged traces remain
        # in the plot buffer without recomputation.
        current_state = {}
        visible_traces = set()
        colour_idx = 0

        # Build state fingerprints for all data columns
        for col in plot_df.columns:
            if col == time_col or col.endswith("_fitted") or col.endswith(" Std") or col.endswith(" SEM"):
                continue

            # Check visibility
            is_visible = trace_settings.get(col, {}).get('show_trace', True)
            if not is_visible:
                # Hidden traces marked as None to trigger removal if currently plotted
                current_state[col] = None
                continue

            visible_traces.add(col)
            y_data = plot_df[col].values
            colour = colours[colour_idx % len(colours)]
            colour_idx += 1

            # Check for fitted data
            fit_col = f"{col}_fitted"
            has_fit = (fitted_plot_df is not None and
                      fit_col in fitted_plot_df.columns and
                      is_visible)

            # Error bands disabled - feature removed
            has_error = False

            # Compute state fingerprint using hash of data arrays
            # Hash comparison is more efficient than element-wise array comparison
            time_hash = hash(time_array.tobytes())
            data_hash = hash(y_data.tobytes())
            fit_hash = None
            if has_fit:
                y_fitted = fitted_plot_df[fit_col].values
                fit_hash = hash(y_fitted.tobytes())

            state = self._compute_trace_state_hash(
                col, True, has_fit, scatter_mode, colour,
                time_hash, data_hash, fit_hash, has_error,
                scale_factor=catalytic_scale_factor
            )
            current_state[col] = state

        # Differential update: identify traces requiring redraw through state comparison
        # Set difference identifies removed traces; hash comparison detects modifications
        traces_to_remove = set(self._last_plot_state.keys()) - set(current_state.keys())
        traces_to_update = set()

        for col, state in current_state.items():
            if state is None:
                # Trace hidden by user
                if col in self._last_plot_state:
                    traces_to_remove.add(col)
            elif col not in self._last_plot_state or self._last_plot_state[col] != state:
                # New trace or modified data/settings
                traces_to_update.add(col)

        # Remove obsolete plot items
        for col in traces_to_remove:
            self._remove_plot_item(col)
            if col in self._last_plot_state:
                del self._last_plot_state[col]

        # Selective redraw of modified traces only
        colour_idx = 0
        for col in plot_df.columns:
            if col == time_col or col.endswith("_fitted") or col.endswith(" Std") or col.endswith(" SEM"):
                continue

            if col not in traces_to_update:
                # Trace unchanged; maintain colour index synchronisation
                if col in visible_traces:
                    colour_idx += 1
                continue

            # Remove existing plot items before redrawing
            if col in self._plot_items:
                self._remove_plot_item(col)

            # Get trace properties
            y_data = plot_df[col].values.copy()

            # Apply catalytic scale factor if provided
            # This converts normalised data (0-1) to nM to match fitted curves
            if catalytic_scale_factor is not None:
                y_data = y_data * catalytic_scale_factor

            colour = colours[colour_idx % len(colours)]
            colour_idx += 1

            # Create plot items dictionary for this trace
            self._plot_items[col] = {}

            # Plot data trace
            if scatter_mode:
                data_item = self.plot_widget.plot(
                    time_array,
                    y_data,
                    pen=None,
                    symbol='o',
                    symbolSize=2,
                    symbolBrush=colour,
                    symbolPen=colour,
                    name=col
                )
            else:
                data_item = self.plot_widget.plot(
                    time_array,
                    y_data,
                    pen=pg.mkPen(colour, width=2),
                    name=col
                )
            self._plot_items[col]['data'] = data_item


            # Plot fitted curve if available
            if fitted_plot_df is not None:
                fit_col = f"{col}_fitted"
                if fit_col in fitted_plot_df.columns:
                    # CRITICAL FIX: Use same normalisation as data (first_time_point)
                    # NOT start_time (UI spinbox) to ensure alignment
                    fitted_time_array = fitted_plot_df[time_col].values - first_time_point
                    y_fitted = fitted_plot_df[fit_col].values

                    fit_item = self.plot_widget.plot(
                        fitted_time_array,
                        y_fitted,
                        pen=pg.mkPen('k', width=2, style=Qt.DashLine),
                        name=None
                    )
                    self._plot_items[col]['fit'] = fit_item

            # Update state tracking
            self._last_plot_state[col] = current_state[col]

        # Configure legend
        if show_legend:
            if hasattr(self.plot_widget, 'legend') and self.plot_widget.legend is not None:
                self.plot_widget.legend.setVisible(True)
        else:
            if hasattr(self.plot_widget, 'legend') and self.plot_widget.legend is not None:
                self.plot_widget.legend.setVisible(False)

        # Auto-range axes to fit the normalised time window
        # This ensures axes update correctly when time window changes
        self.plot_widget.plotItem.autoRange()

        logger.debug("Plot updated incrementally (updated %d traces) - time window: %.2f-%.2f min",
                    len(traces_to_update), start_time, end_time)

    # =========================================================================
    # TRACE LIST POPULATION
    # =========================================================================

    def populate_trace_list(self, selection_panel, data_df, time_col, group_display_info):
        """
        Populate the trace selection grid with available data columns in 3-column layout.

        Parameters
        ----------
        selection_panel : dict
            Dictionary containing grid layout and manager
        data_df : pd.DataFrame
            Data containing traces to display
        time_col : str
            Name of time column (excluded from list)
        group_display_info : dict
            Optional visual grouping info for organising traces under headers.
            Pass {} for a flat list. Structure matches replicate_info format:
            {group_name: {'columns': [...], 'mean_col': '...'}}
        """
        if data_df is None or time_col not in data_df.columns:
            logger.warning("Cannot populate trace list without valid data and time column")
            return

        manager = selection_panel['manager']
        grid_layout = selection_panel['grid_layout']

        # Clear existing items
        manager.clear_all()

        # Get all data columns (exclude time, fitted, Std, SEM)
        data_cols = [
            c for c in data_df.columns
            if c != time_col
            and not c.endswith("_fitted")
            and not c.endswith(" Std")
            and not c.endswith(" SEM")
        ]

        # Track which columns are part of display groups
        grouped_cols = set()
        for group_id, group_data in group_display_info.items():
            grouped_cols.update(group_data.get('columns', []))

        # Use unified checkbox style from UITheme
        checkbox_style = UITheme.get_checkbox_style()

        # Style for header labels (spanning all columns)
        header_style = f"font-size: 10pt; font-weight: bold; color: {Colours.ACCENT_BLUE}; padding: 4px 0px;"

        current_row = 0
        current_col = 0
        num_columns = 3  # Reduced for two-column layout

        def add_checkbox(text, col_name):
            """Add a checkbox to the grid at the current position."""
            nonlocal current_row, current_col
            checkbox = QCheckBox(text)
            checkbox.setStyleSheet(checkbox_style)
            checkbox.setChecked(True)  # Default: checked
            checkbox.stateChanged.connect(lambda: self._on_checkbox_changed(selection_panel))
            grid_layout.addWidget(checkbox, current_row, current_col)
            manager.checkboxes.append((checkbox, col_name))

            # Move to next position (left to right, then top to bottom)
            current_col += 1
            if current_col >= num_columns:
                current_col = 0
                current_row += 1

        def add_header(text):
            """Add a header label spanning all 6 columns."""
            nonlocal current_row, current_col
            # Start new row for header
            if current_col != 0:
                current_row += 1
                current_col = 0
            header_label = QLabel(text)
            header_label.setStyleSheet(header_style)
            grid_layout.addWidget(header_label, current_row, 0, 1, num_columns)  # Span all columns
            manager.headers.append(header_label)
            current_row += 1

        # Add grouped traces first (if any display groups provided)
        for group_id, group_data in sorted(group_display_info.items()):
            mean_col = group_data.get('mean_col')
            group_columns = group_data.get('columns', [])

            # Only add if mean column has valid data
            if mean_col and mean_col in data_df.columns:
                mean_data = data_df[mean_col]
                if mean_data.notna().any() and not mean_data.isna().all():
                    # Add group header
                    add_header(f"=== Group {group_id} ===")

                    # Add mean
                    add_checkbox(f"  {mean_col}", mean_col)

                    # Add individual traces in this group
                    for col in sorted(group_columns):
                        if col in data_df.columns:
                            add_checkbox(f"  {col}", col)

        # Add ungrouped traces
        standalone_cols = [c for c in data_cols if c not in grouped_cols]
        if standalone_cols:
            if len(manager.checkboxes) > 0:  # Add separator if we have groups above
                add_header("=== Individual Traces ===")

            for col in sorted(standalone_cols):
                add_checkbox(col, col)  # No emoji prefix for cleaner appearance

        # Update summary label
        selectable_count = len(manager.checkboxes)
        if 'summary_label' in selection_panel:
            selection_panel['summary_label'].setText(f"{selectable_count} shown of {selectable_count} total")

        # Populate exclude from fitting checkboxes
        self._populate_exclude_checkboxes(selection_panel, data_cols)

        logger.info("Populated trace grid with %d selectable traces in 3-column layout", selectable_count)

    def _populate_exclude_checkboxes(self, selection_panel, data_cols):
        """
        Populate the exclude from fitting checkboxes.

        Parameters
        ----------
        selection_panel : dict
            Dictionary containing selection panel components
        data_cols : list
            List of data column names to create checkboxes for
        """
        manager = selection_panel['manager']

        # Clear existing exclude checkboxes
        if hasattr(manager, 'clear_exclude_checkboxes'):
            manager.clear_exclude_checkboxes()

        if not hasattr(manager, 'exclude_grid') or not hasattr(manager, 'exclude_checkboxes'):
            return

        exclude_grid = manager.exclude_grid

        # Use unified checkbox style with red accent for exclusion
        exclude_checkbox_style = f"""
            QCheckBox {{
                font-size: 9pt;
                color: {Colours.TEXT_PRIMARY};
                spacing: 6px;
            }}
            QCheckBox:disabled {{
                color: {Colours.TEXT_DISABLED};
            }}
            QCheckBox::indicator {{
                width: 16px;
                height: 16px;
                border: 1px solid {Colours.CARD_BORDER};
                border-radius: 2px;
                background-color: {Colours.CARD_BACKGROUND};
            }}
            QCheckBox::indicator:hover {{
                border-color: {Colours.ACCENT_RED};
            }}
            QCheckBox::indicator:checked {{
                background-color: {Colours.ACCENT_RED};
                border-color: {Colours.ACCENT_RED};
            }}
        """

        current_row = 0
        current_col = 0
        num_columns = 3  # Match visibility section columns

        for col_name in sorted(data_cols):
            checkbox = QCheckBox(col_name)
            checkbox.setStyleSheet(exclude_checkbox_style)
            checkbox.setChecked(False)  # Default: not excluded
            checkbox.setToolTip(f"Check to exclude {col_name} from fitting")
            checkbox.stateChanged.connect(lambda state, c=col_name: self._on_exclude_changed(selection_panel, c, state))
            exclude_grid.addWidget(checkbox, current_row, current_col)
            manager.exclude_checkboxes.append((checkbox, col_name))

            current_col += 1
            if current_col >= num_columns:
                current_col = 0
                current_row += 1

    def _on_exclude_changed(self, selection_panel, col_name, state):
        """Handle exclude checkbox state change."""
        is_excluded = state == Qt.Checked
        logger.debug("Trace %s: exclude_from_fit = %s", col_name, is_excluded)
        # Trigger callback to update trace settings
        selection_panel['callback']()

    def _on_checkbox_changed(self, selection_panel):
        """Handle checkbox state change."""
        self._update_trace_summary(selection_panel)
        selection_panel['callback']()

    # =========================================================================
    # DIALOGUE WINDOWS
    # =========================================================================

    def open_experiment_info_dialog(self, current_title, current_info, save_callback):
        """
        Open dialogue for editing experiment metadata.

        Parameters
        ----------
        current_title : str
            Current experiment title
        current_info : str
            Current experiment information
        save_callback : callable
            Callback function to save changes
        """
        dialogue = QDialog(self.parent)
        dialogue.setWindowTitle("Edit Experiment Metadata")
        dialogue.setMinimumWidth(500)
        dialogue.setMinimumHeight(400)

        layout = QVBoxLayout(dialogue)
        layout.setSpacing(10)

        # Title input
        title_label = QLabel("<b>Experiment Title:</b>")
        title_label.setStyleSheet("font-size: 10px;")
        layout.addWidget(title_label)

        title_input = QLineEdit()
        title_input.setText(current_title)
        title_input.setStyleSheet("font-size: 10px; padding: 6px;")
        title_input.setPlaceholderText("Enter experiment title...")
        layout.addWidget(title_input)

        # Info/notes input
        info_label = QLabel("<b>Experiment Notes:</b>")
        info_label.setStyleSheet("font-size: 10px; margin-top: 10px;")
        layout.addWidget(info_label)

        info_input = QTextEdit()
        info_input.setPlainText(current_info)
        info_input.setStyleSheet("font-size: 10px; padding: 6px;")
        info_input.setPlaceholderText("Enter experiment notes, conditions, observations, etc...")
        layout.addWidget(info_input)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        cancel_button = QPushButton("Cancel")
        cancel_button.setStyleSheet(self.button_style_standard)
        cancel_button.clicked.connect(dialogue.reject)
        cancel_button.setMinimumWidth(80)
        button_layout.addWidget(cancel_button)

        save_button = QPushButton("Save")
        save_button.setStyleSheet(self.button_style_primary)
        save_button.clicked.connect(
            lambda: save_callback(dialogue, title_input.text(), info_input.toPlainText())
        )
        save_button.setMinimumWidth(80)
        button_layout.addWidget(save_button)

        layout.addLayout(button_layout)

        # Show dialogue
        dialogue.exec_()

