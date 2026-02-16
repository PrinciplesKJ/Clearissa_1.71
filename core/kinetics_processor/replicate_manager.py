"""
Replicate Management Module for Kinetics Processor
--------------------------------------------------
Provides interactive dialogue for defining and managing replicate groups.
Supports group reordering to control colour assignment in the replicate
averages graph.

Author: Krizan Jurinovic
Date: November 2025
"""

import logging
import re
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QListWidget,
                             QPushButton, QLabel, QFrame, QListWidgetItem,
                             QMessageBox, QInputDialog, QScrollArea, QWidget)
from PyQt5.QtCore import Qt

from core.common.ui_theme import UITheme, Colours

logger = logging.getLogger(__name__)


class ReplicateManagerDialogue(QDialog):
    """
    Interactive dialogue for managing replicate groups.

    Allows users to:
    - View automatically detected replicate groups
    - Create custom replicate groups
    - Reorder groups (order determines colour in replicate averages graph)
    - Modify or delete existing groups
    - Preview group colours before applying
    """

    def __init__(self, available_traces, auto_detected_groups, parent=None,
                 gui_handler=None, colour_palette=None):
        """
        Initialise the replicate manager dialogue.

        Parameters
        ----------
        available_traces : list
            List of all available trace names
        auto_detected_groups : dict
            Dictionary of automatically detected groups {group_name: [traces]}
        parent : QWidget, optional
            Parent widget
        gui_handler : KineticsGUI, optional
            GUI handler (retained for compatibility)
        colour_palette : list, optional
            List of hex colour strings used in the replicate averages graph
        """
        super().__init__(parent)
        self.available_traces = sorted(available_traces)
        self.auto_detected_groups = auto_detected_groups
        self.gui_handler = gui_handler
        self.colour_palette = colour_palette or [
            '#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE',
            '#AA3377', '#BBBBBB', '#EE7733', '#0077BB', '#33BBEE',
        ]
        self.user_groups = {}
        self.group_order = []  # Explicit ordering - determines graph colours

        self.setWindowTitle("Manage Replicate Groups")
        self.setMinimumWidth(700)
        self.setMinimumHeight(500)

        self._init_ui()

        # Populate with auto-detected or existing groups
        if auto_detected_groups:
            self.user_groups = auto_detected_groups.copy()
            self.group_order = list(auto_detected_groups.keys())
            self._refresh_group_display()

    def _init_ui(self):
        """Initialise the user interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(10, 10, 10, 10)

        # Header
        header = QLabel(
            "<b>Replicate Group Manager</b><br>"
            "Define which traces are replicates. Replicates will be averaged "
            "together and displayed with error bands (mean +/- SEM).<br>"
            "<i>Group order determines colour assignment in the replicate "
            "averages graph.</i>"
        )
        header.setStyleSheet(
            f"font-size: 9pt; padding: 8px; "
            f"background-color: {Colours.SECTION_BACKGROUND}; "
            f"border: 1px solid {Colours.CARD_BORDER}; border-radius: 3px; "
            f"color: {Colours.TEXT_PRIMARY};"
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        # Main content area - two columns
        content_layout = QHBoxLayout()
        content_layout.setSpacing(8)

        # --- Left column: Available traces ---
        left_frame = QFrame()
        left_frame.setStyleSheet(
            f"QFrame {{ background-color: {Colours.CARD_BACKGROUND}; "
            f"border: 1px solid {Colours.CARD_BORDER}; border-radius: 3px; }}"
        )
        left_layout = QVBoxLayout(left_frame)
        left_layout.setContentsMargins(8, 8, 8, 8)

        left_title = QLabel("Available Traces")
        left_title.setStyleSheet(UITheme.get_label_style_secondary())
        left_layout.addWidget(left_title)

        self.trace_list = QListWidget()
        self.trace_list.setSelectionMode(QListWidget.MultiSelection)
        self.trace_list.setStyleSheet(f"""
            QListWidget {{
                font-size: 9pt;
                background-color: {Colours.CARD_BACKGROUND};
                border: 1px solid {Colours.CARD_BORDER};
                border-radius: 2px;
                color: {Colours.TEXT_PRIMARY};
            }}
            QListWidget::item {{
                padding: 4px;
            }}
            QListWidget::item:selected {{
                background-color: {Colours.ACCENT_BLUE};
                color: white;
            }}
        """)

        for trace in self.available_traces:
            self.trace_list.addItem(trace)

        left_layout.addWidget(self.trace_list)

        self.create_group_btn = QPushButton("Create Group from Selection")
        self.create_group_btn.setStyleSheet(UITheme.get_button_style_primary())
        self.create_group_btn.clicked.connect(self._create_group_from_selection)
        left_layout.addWidget(self.create_group_btn)

        self.individual_groups_btn = QPushButton("Each Trace as Own Group")
        self.individual_groups_btn.setStyleSheet(UITheme.get_button_style_standard())
        self.individual_groups_btn.setToolTip(
            "Assign every available trace to its own individual group.\n"
            "Useful for viewing all traces in the Replicate Averages window\n"
            "even without true replicates."
        )
        self.individual_groups_btn.clicked.connect(self._create_individual_groups)
        left_layout.addWidget(self.individual_groups_btn)

        info_label = QLabel(
            "Select one or more traces and click 'Create Group'. "
            "Single traces can be added as individual groups."
        )
        info_label.setStyleSheet(UITheme.get_label_style_tertiary())
        info_label.setWordWrap(True)
        left_layout.addWidget(info_label)

        content_layout.addWidget(left_frame)

        # --- Right column: Defined groups ---
        right_frame = QFrame()
        right_frame.setStyleSheet(
            f"QFrame {{ background-color: {Colours.CARD_BACKGROUND}; "
            f"border: 1px solid {Colours.CARD_BORDER}; border-radius: 3px; }}"
        )
        right_layout = QVBoxLayout(right_frame)
        right_layout.setContentsMargins(8, 8, 8, 8)

        right_title = QLabel("Replicate Groups")
        right_title.setStyleSheet(UITheme.get_label_style_secondary())
        right_layout.addWidget(right_title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            f"QScrollArea {{ background-color: {Colours.CARD_BACKGROUND}; "
            f"border: 1px solid {Colours.CARD_BORDER}; border-radius: 2px; }}"
        )

        self.groups_container = QWidget()
        self.groups_layout = QVBoxLayout(self.groups_container)
        self.groups_layout.setAlignment(Qt.AlignTop)
        self.groups_layout.setSpacing(4)
        self.groups_layout.setContentsMargins(4, 4, 4, 4)

        scroll.setWidget(self.groups_container)
        right_layout.addWidget(scroll)

        content_layout.addWidget(right_frame)

        layout.addLayout(content_layout)

        # --- Bottom buttons ---
        bottom_layout = QHBoxLayout()
        bottom_layout.addStretch()

        self.auto_detect_btn = QPushButton("Auto-Detect Groups")
        self.auto_detect_btn.setStyleSheet(UITheme.get_button_style_standard())
        self.auto_detect_btn.clicked.connect(self._run_auto_detection)
        bottom_layout.addWidget(self.auto_detect_btn)

        self.clear_all_btn = QPushButton("Clear All Groups")
        self.clear_all_btn.setStyleSheet(UITheme.get_button_style_standard())
        self.clear_all_btn.clicked.connect(self._clear_all_groups)
        bottom_layout.addWidget(self.clear_all_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setStyleSheet(UITheme.get_button_style_standard())
        self.cancel_btn.clicked.connect(self.reject)
        bottom_layout.addWidget(self.cancel_btn)

        self.apply_btn = QPushButton("Apply Groups")
        self.apply_btn.setStyleSheet(UITheme.get_button_style_success())
        self.apply_btn.clicked.connect(self.accept)
        bottom_layout.addWidget(self.apply_btn)

        layout.addLayout(bottom_layout)

    def _get_group_colour(self, index):
        """Get the colour for a group at the given position in the order."""
        return self.colour_palette[index % len(self.colour_palette)]

    def _create_group_from_selection(self):
        """Create a new replicate group from selected traces."""
        selected_items = self.trace_list.selectedItems()

        if len(selected_items) < 1:
            QMessageBox.warning(
                self,
                "Invalid Selection",
                "Please select at least 1 trace to create a group."
            )
            return

        selected_traces = [item.text() for item in selected_items]

        # Ask for group name
        group_name, ok = QInputDialog.getText(
            self,
            "Group Name",
            "Enter a name for this replicate group:",
            text=f"Group_{len(self.user_groups) + 1}"
        )

        if not ok or not group_name.strip():
            return

        group_name = group_name.strip()

        # Check if name already exists
        if group_name in self.user_groups:
            reply = QMessageBox.question(
                self,
                "Group Exists",
                f"A group named '{group_name}' already exists. Overwrite?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
            # Keep existing position in order
        else:
            self.group_order.append(group_name)

        self.user_groups[group_name] = selected_traces
        logger.info("Created replicate group '%s' with %d traces",
                     group_name, len(selected_traces))

        self._refresh_group_display()
        self.trace_list.clearSelection()

    def _create_individual_groups(self):
        """Assign every available trace to its own individual group."""
        if not self.available_traces:
            QMessageBox.warning(self, "No Traces", "No traces are available.")
            return

        reply = QMessageBox.question(
            self,
            "Create Individual Groups",
            f"This will create {len(self.available_traces)} groups "
            f"(one per trace), replacing any existing groups.\n\nProceed?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        self.user_groups = {}
        self.group_order = []

        for trace in self.available_traces:
            self.user_groups[trace] = [trace]
            self.group_order.append(trace)

        logger.info("Created %d individual groups", len(self.available_traces))
        self._refresh_group_display()

    def _refresh_group_display(self):
        """Refresh the display of defined groups in the current order."""
        # Clear existing widgets
        while self.groups_layout.count():
            item = self.groups_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not self.user_groups:
            no_groups_label = QLabel("No replicate groups defined.")
            no_groups_label.setStyleSheet(UITheme.get_label_style_tertiary())
            no_groups_label.setAlignment(Qt.AlignCenter)
            self.groups_layout.addWidget(no_groups_label)
            return

        # Display each group in the user-defined order
        for idx, group_name in enumerate(self.group_order):
            if group_name not in self.user_groups:
                continue
            colour = self._get_group_colour(idx)
            is_first = (idx == 0)
            is_last = (idx == len(self.group_order) - 1)
            group_widget = self._create_group_widget(
                group_name, self.user_groups[group_name],
                colour, is_first, is_last
            )
            self.groups_layout.addWidget(group_widget)

    def _create_group_widget(self, group_name, traces, colour, is_first, is_last):
        """Create a widget for a single replicate group with colour and reorder controls."""
        frame = QFrame()
        frame.setStyleSheet(
            f"QFrame#groupCard {{ background-color: {Colours.CARD_BACKGROUND}; "
            f"border: 1px solid {Colours.CARD_BORDER}; border-radius: 3px; }}"
        )
        frame.setObjectName("groupCard")

        main_layout = QVBoxLayout(frame)
        main_layout.setContentsMargins(8, 6, 8, 6)
        main_layout.setSpacing(3)

        # Top row: colour swatch + group name + reorder/delete buttons
        top_row = QHBoxLayout()
        top_row.setSpacing(6)

        # Colour swatch showing the graph colour
        swatch = QLabel()
        swatch.setFixedSize(42, 14)
        swatch.setStyleSheet(
            f"background-color: {colour}; "
            f"border: 1px solid {Colours.BORDER_MEDIUM}; "
            f"border-radius: 2px;"
        )
        swatch.setToolTip(f"Graph colour: {colour}")
        top_row.addWidget(swatch)

        # Group name (normal weight — trace names below are the key info)
        name_label = QLabel(group_name)
        name_label.setStyleSheet(
            f"font-size: 9pt; font-weight: normal; "
            f"color: {Colours.TEXT_PRIMARY}; border: none;"
        )
        top_row.addWidget(name_label)
        top_row.addStretch()

        # Compact button style for the row
        small_btn_style = f"""
            QPushButton {{
                font-size: 8pt; padding: 2px 6px;
                background-color: {Colours.CARD_BACKGROUND};
                color: {Colours.TEXT_SECONDARY};
                border: 1px solid {Colours.BORDER_LIGHT};
                border-radius: 2px;
            }}
            QPushButton:hover {{
                background-color: {Colours.SECTION_BACKGROUND};
                border-color: {Colours.BORDER_MEDIUM};
            }}
            QPushButton:disabled {{
                color: {Colours.TEXT_DISABLED};
                border-color: {Colours.CARD_BORDER};
            }}
        """

        btn_size = 24

        up_btn = QPushButton("\u25B2")
        up_btn.setFixedSize(btn_size, btn_size)
        up_btn.setEnabled(not is_first)
        up_btn.setStyleSheet(small_btn_style)
        up_btn.setToolTip("Move group up")
        up_btn.clicked.connect(lambda: self._move_group_up(group_name))
        top_row.addWidget(up_btn)

        down_btn = QPushButton("\u25BC")
        down_btn.setFixedSize(btn_size, btn_size)
        down_btn.setEnabled(not is_last)
        down_btn.setStyleSheet(small_btn_style)
        down_btn.setToolTip("Move group down")
        down_btn.clicked.connect(lambda: self._move_group_down(group_name))
        top_row.addWidget(down_btn)

        del_btn_style = f"""
            QPushButton {{
                font-size: 8pt; font-weight: bold; padding: 2px 6px;
                background-color: {Colours.ACCENT_RED};
                color: white;
                border: none; border-radius: 2px;
            }}
            QPushButton:hover {{
                background-color: {Colours.ACCENT_RED_HOVER};
            }}
        """
        delete_btn = QPushButton("x")
        delete_btn.setFixedSize(btn_size, btn_size)
        delete_btn.setStyleSheet(del_btn_style)
        delete_btn.setToolTip("Delete group")
        delete_btn.clicked.connect(lambda: self._delete_group(group_name))
        top_row.addWidget(delete_btn)

        main_layout.addLayout(top_row)

        # Traces list (bold — the important information)
        traces_label = QLabel(", ".join(traces))
        traces_label.setStyleSheet(
            f"font-size: 8pt; font-weight: bold; "
            f"color: {Colours.TEXT_SECONDARY}; border: none;"
        )
        traces_label.setWordWrap(True)
        main_layout.addWidget(traces_label)

        return frame

    def _move_group_up(self, group_name):
        """Move a group up in the order (lower index = earlier colour)."""
        idx = self.group_order.index(group_name)
        if idx > 0:
            self.group_order[idx], self.group_order[idx - 1] = (
                self.group_order[idx - 1], self.group_order[idx]
            )
            logger.info("Moved group '%s' up to position %d", group_name, idx - 1)
            self._refresh_group_display()

    def _move_group_down(self, group_name):
        """Move a group down in the order (higher index = later colour)."""
        idx = self.group_order.index(group_name)
        if idx < len(self.group_order) - 1:
            self.group_order[idx], self.group_order[idx + 1] = (
                self.group_order[idx + 1], self.group_order[idx]
            )
            logger.info("Moved group '%s' down to position %d", group_name, idx + 1)
            self._refresh_group_display()

    def _delete_group(self, group_name):
        """Delete a replicate group."""
        reply = QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to delete the group '{group_name}'?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            del self.user_groups[group_name]
            self.group_order.remove(group_name)
            logger.info("Deleted replicate group '%s'", group_name)
            self._refresh_group_display()

    def _run_auto_detection(self):
        """Run automatic replicate detection."""
        detected = self._auto_detect_replicates()

        if not detected:
            QMessageBox.information(
                self,
                "Auto-Detection",
                "No replicate groups detected automatically.\n\n"
                "Automatic detection looks for traces with the same well column "
                "(e.g., A01, B01, C01 all end in '01')."
            )
            return

        reply = QMessageBox.question(
            self,
            "Auto-Detection Complete",
            f"Detected {len(detected)} potential replicate groups.\n\n"
            "Do you want to use these groups? This will replace any existing groups.",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.user_groups = detected
            self.group_order = sorted(detected.keys())
            self._refresh_group_display()
            logger.info("Applied %d auto-detected replicate groups", len(detected))

    def _auto_detect_replicates(self):
        """
        Automatically detect replicate groups based on well column patterns.

        Returns
        -------
        dict
            Dictionary of detected groups {group_name: [traces]}
        """
        replicate_groups = {}

        for trace in self.available_traces:
            # Extract numeric suffix (well plate column identifier)
            match = re.search(r'(\d+)$', trace)

            if match:
                well_column = match.group(1)
                group_key = f"Col{well_column}"

                if group_key not in replicate_groups:
                    replicate_groups[group_key] = []
                replicate_groups[group_key].append(trace)

        # Filter to only groups with 2+ members
        replicate_groups = {k: v for k, v in replicate_groups.items() if len(v) >= 2}

        logger.debug("Auto-detected %d replicate groups", len(replicate_groups))
        return replicate_groups

    def _clear_all_groups(self):
        """Clear all defined groups."""
        if not self.user_groups:
            return

        reply = QMessageBox.question(
            self,
            "Confirm Clear All",
            "Are you sure you want to delete all replicate groups?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.user_groups = {}
            self.group_order = []
            self._refresh_group_display()
            logger.info("Cleared all replicate groups")

    def get_user_groups(self):
        """
        Get the user-defined replicate groups in the user-specified order.

        Returns
        -------
        dict
            Dictionary of replicate groups {group_name: [traces]} preserving
            the display order set by the user.
        """
        ordered = {}
        for name in self.group_order:
            if name in self.user_groups:
                ordered[name] = self.user_groups[name]
        return ordered


class ReplicateManager:
    """
    Manages replicate group definition and storage.

    Responsibilities:
    - Launch interactive dialogue for defining groups
    - Store user-defined groups persistently
    - Calculate statistics for replicate groups
    """

    def __init__(self, parent_widget):
        """
        Initialise the replicate manager.

        Parameters
        ----------
        parent_widget : QWidget
            Parent widget for dialogues
        """
        self.parent = parent_widget
        self.current_replicate_groups = {}

    def show_replicate_dialogue(self, available_traces, auto_detected_groups=None,
                                gui_handler=None, colour_palette=None):
        """
        Show the replicate management dialogue.

        Parameters
        ----------
        available_traces : list
            List of all available trace names
        auto_detected_groups : dict, optional
            Dictionary of automatically detected groups
        gui_handler : KineticsGUI, optional
            GUI handler for accessing button styles
        colour_palette : list, optional
            List of hex colour strings for graph colour preview

        Returns
        -------
        dict or None
            Dictionary of user-defined groups in display order, or None if cancelled
        """
        dialogue = ReplicateManagerDialogue(
            available_traces,
            auto_detected_groups or {},
            parent=self.parent,
            gui_handler=gui_handler,
            colour_palette=colour_palette
        )

        result = dialogue.exec_()

        if result == QDialog.Accepted:
            self.current_replicate_groups = dialogue.get_user_groups()
            logger.info("User defined %d replicate groups", len(self.current_replicate_groups))
            return self.current_replicate_groups
        else:
            logger.info("Replicate group dialogue cancelled by user")
            return None

    def calculate_group_statistics(self, df, time_col, user_groups):
        """
        Calculate descriptive statistics for user-defined replicate groups.

        This function computes mean, standard deviation, and standard error of the mean
        (SEM) for each replicate group at every time point, enabling statistical analysis
        of experimental variability across biological or technical replicates.

        STATISTICAL CALCULATIONS
        ------------------------
        For each replicate group containing n traces:

        1. Mean (arithmetic average):
           mean = (1/n) x sum_i(x_i)

           Represents the central tendency of replicate measurements at each time point.

        2. Standard Deviation (sample SD with Bessel's correction):
           std = sqrt( (1/(n-1)) x sum_i(x_i - mean)^2 )

           Uses ddof=1 (degrees of freedom = n-1) for unbiased estimation of population
           standard deviation from a sample. Measures spread of individual measurements.

        3. Standard Error of the Mean (SEM):
           sem = std / sqrt(n)

           Represents the uncertainty in the mean estimate. Smaller than std by factor sqrt(n).
           Used for error bars when comparing group means.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing time-series experimental data with one column per trace
        time_col : str
            Name of the column containing time values
        user_groups : dict
            Dictionary mapping group names to lists of trace column names
            Format: {group_name: [trace1, trace2, ...]}

        Returns
        -------
        tuple (pd.DataFrame, dict) or None
            If successful, returns:
            - stats_df: DataFrame with columns:
                - Time column (unchanged)
                - {group_name}_mean: Mean concentration at each time point (nM)
                - {group_name}_std: Standard deviation (nM)
                - {group_name}_sem: Standard error of the mean (nM)
                - {group_name}_n: Number of valid data points at each time
            - replicate_info: Dictionary with metadata about each group

            Returns None if no valid groups could be processed.

        Notes
        -----
        - Groups with only one trace will have std=0 and sem=0 (no variation calculable)
        - Missing values (NaN) are automatically excluded from calculations
        - Invalid trace names (not found in DataFrame) are silently skipped with warning
        """
        if not user_groups:
            logger.debug("No user groups defined - no statistics calculated")
            return None

        import pandas as pd

        stats_df = df[[time_col]].copy()
        replicate_info = {}

        for group_name, traces in user_groups.items():
            # Validate all traces exist
            valid_traces = [t for t in traces if t in df.columns]

            if len(valid_traces) < 1:
                logger.warning("Group '%s' has no valid traces - skipping", group_name)
                continue

            # Get data for all members
            group_data = df[valid_traces]

            # Calculate statistics
            stats_df[f"{group_name}_mean"] = group_data.mean(axis=1)

            # Standard deviation and SEM only meaningful for 2+ replicates
            if len(valid_traces) >= 2:
                stats_df[f"{group_name}_std"] = group_data.std(axis=1, ddof=1)
                stats_df[f"{group_name}_sem"] = group_data.sem(axis=1)
            else:
                # Single trace - no variation to calculate
                stats_df[f"{group_name}_std"] = 0.0
                stats_df[f"{group_name}_sem"] = 0.0

            stats_df[f"{group_name}_n"] = group_data.count(axis=1)

            # Store mapping
            replicate_info[group_name] = {
                'columns': valid_traces,
                'mean_col': f"{group_name}_mean",
                'std_col': f"{group_name}_std",
                'sem_col': f"{group_name}_sem"
            }

            if len(valid_traces) == 1:
                logger.debug("Added single-trace group '%s': %s (no std/sem)", group_name, valid_traces)
            else:
                logger.debug("Calculated statistics for group '%s': %s (%d replicates)",
                           group_name, valid_traces, len(valid_traces))

        if not replicate_info:
            return None

        return stats_df, replicate_info
