"""
Endpoint Detection Module for Kinetics Processor
-------------------------------------------------
Handles endpoint concentration detection and display.

Author: Krizan Jurinovic
Date: November 2025
"""

import logging
import numpy as np
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTableWidget,
                             QTableWidgetItem, QPushButton, QLabel, QMessageBox, QHeaderView)
from PyQt5.QtCore import Qt

logger = logging.getLogger(__name__)


class EndpointDetector:
    """
    Handles endpoint concentration detection for kinetics analysis.

    Responsibilities:
    - Detect endpoint concentrations from time-series data
    - Calculate statistics for endpoints
    - Display results in popup dialogues
    """

    def __init__(self, parent_widget):
        """
        Initialise the endpoint detector.

        Parameters
        ----------
        parent_widget : QWidget
            Parent widget for displaying dialogues
        """
        self.parent = parent_widget
        self.detected_endpoints = {}

    def detect_endpoints(self, data_df, time_col, window_length_min, trace_settings):
        """
        Detect endpoint concentrations from all active traces.

        Parameters
        ----------
        data_df : pd.DataFrame
            Data frame containing experimental data
        time_col : str
            Name of the time column
        window_length_min : float
            Time window length in minutes for averaging
        trace_settings : dict
            Dictionary of trace visibility settings

        Returns
        -------
        list or None
            List of endpoint result dictionaries, or None if detection fails
        """
        logger.info("Endpoint detection initiated")

        # Validate data availability
        if data_df is None or data_df.empty:
            QMessageBox.warning(
                self.parent,
                "No Data",
                "Please load kinetic data before detecting endpoints."
            )
            logger.warning("Endpoint detection failed: no data loaded")
            return None

        # Validate window length
        if window_length_min <= 0:
            QMessageBox.warning(
                self.parent,
                "Invalid Window",
                "Please specify a positive time window length for endpoint averaging."
            )
            logger.warning("Endpoint detection failed: invalid window length")
            return None

        # Get full dataset time range
        if time_col not in data_df.columns:
            QMessageBox.critical(self.parent, "Error", "Time column not found in data.")
            return None

        time_data = data_df[time_col].values
        max_time = np.max(time_data)
        min_time = np.min(time_data)
        total_duration = max_time - min_time

        # Validate window length
        if window_length_min > total_duration:
            QMessageBox.warning(
                self.parent,
                "Window Too Large",
                f"Specified window ({window_length_min:.2f} min) exceeds total data duration ({total_duration:.2f} min).\n\n"
                "Please use a shorter time window."
            )
            logger.warning("Window length %.2f exceeds data duration %.2f",
                          window_length_min, total_duration)
            return None

        # Calculate endpoint time threshold
        endpoint_start_time = max_time - window_length_min

        # Filter data to endpoint window
        endpoint_mask = time_data >= endpoint_start_time
        endpoint_df = data_df[endpoint_mask]

        if len(endpoint_df) < 2:
            QMessageBox.warning(
                self.parent,
                "Insufficient Data",
                "Not enough data points in the specified endpoint window.\n\n"
                "Try increasing the window length."
            )
            logger.warning("Insufficient data points in endpoint window")
            return None

        # Clear previous results
        self.detected_endpoints = {}

        # Get all active data columns (exclude time, fitted, stats)
        data_cols = [
            c for c in data_df.columns
            if c != time_col
               and not c.endswith("_fitted")
               and not c.endswith(" Std")
               and not c.endswith(" SEM")
        ]

        # Calculate endpoints for each active trace
        endpoint_results = []
        for col in data_cols:
            # Check if trace is visible
            if not trace_settings.get(col, {}).get("show_trace", True):
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

            logger.debug("%s: endpoint = %.3f +/- %.3f nM (n=%d)",
                        col, endpoint_conc, endpoint_std, len(valid_data))

        if not endpoint_results:
            QMessageBox.warning(
                self.parent,
                "No Endpoints Detected",
                "No valid endpoints could be calculated from active traces.\n\n"
                "Ensure traces are visible and contain valid data."
            )
            logger.warning("No endpoints detected from active traces")
            return None

        logger.info("Successfully detected %d endpoints", len(endpoint_results))

        return endpoint_results

    def show_endpoint_results_dialogue(self, results, window_length, start_time, end_time, gui_handler):
        """
        Display endpoint detection results in a popup dialogue.

        Parameters
        ----------
        results : list
            List of dictionaries with well, concentration, std, n
        window_length : float
            Time window length in minutes
        start_time : float
            Start time of endpoint window
        end_time : float
            End time of endpoint window
        gui_handler : KineticsGUI
            GUI handler for accessing button styles
        """
        dialogue = QDialog(self.parent)
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
        header_label.setStyleSheet(
            "font-size: 10px; padding: 8px; background-color: #E3F2FD; border-radius: 4px;")
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
            "Info: These endpoint values represent the mean signal concentration over the final time window.\n"
            "Check 'Use Detected Endpoints' to apply these as [SN]0 initial values in model fitting."
        )
        info_label.setStyleSheet("font-size: 8px; color: #666; font-style: italic; padding: 4px;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Close button
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        close_button = QPushButton("Close")
        close_button.setStyleSheet(gui_handler.button_style_standard)
        close_button.clicked.connect(dialogue.accept)
        close_button.setMinimumWidth(100)
        button_layout.addWidget(close_button)

        layout.addLayout(button_layout)

        # Show dialogue
        dialogue.exec_()

    def get_endpoint_for_trace(self, trace_name):
        """
        Get the detected endpoint concentration for a specific trace.

        Parameters
        ----------
        trace_name : str
            Name of the trace

        Returns
        -------
        float or None
            Endpoint concentration, or None if not detected
        """
        if trace_name in self.detected_endpoints:
            return float(self.detected_endpoints[trace_name]['concentration'])
        return None
