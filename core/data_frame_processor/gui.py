"""
GUI construction for the DataFrameProcessor module.
Contains layout builders for the 'View Data' tab, parameter frame,
well selection frame, status frame, and right-side plot frame.

All event handling and data logic are handled by the main DataFrameProcessor class.

RESPONSIVE DESIGN:
- Adaptive layouts that scale with window size
- Minimum sizes instead of fixed widths
- Scrollable sections for small screens
- Well selection always visible and accessible

Author: Krizan Jurinovic, November 2025
"""

import re

import pandas as pd
from PyQt5.QtWidgets import (
    QWidget, QFrame, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QGridLayout, QToolButton, QDoubleSpinBox, QComboBox, QSizePolicy,
    QScrollArea, QLayout
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QColor, QPalette

from core.common.ui_theme import Colors, UITheme
from .gui_components import ElidedLabel, LongClickButton, WellButton, ModernCard


# Backward compatibility aliases for local colour constants
CARD_BACKGROUND = Colors.CARD_BACKGROUND
CARD_BORDER_COLOUR = Colors.CARD_BORDER
MAIN_BACKGROUND = Colors.MAIN_BACKGROUND
CONTENT_BACKGROUND = Colors.CONTENT_BACKGROUND


class DataFrameProcessorGUI:
    """GUI construction for the DataFrameProcessor."""

    def __init__(self, processor):
        self.p = processor  # reference to DataFrameProcessor

    # -------------------------------------------------------------------------
    # ENTRY POINT
    # -------------------------------------------------------------------------
    def build_main_view(self, parent):
        """
        Build responsive main view with adaptive layout.

        Layout strategy:
        - Left panel: minimum 320px, maximum 400px, adapts to content
        - Right panel: expands to fill remaining space
        - Scroll areas for overflow on small screens
        """
        main_frame = QWidget(parent)

        # Set modern background colour - consistent across the application
        main_frame.setAutoFillBackground(True)
        palette = main_frame.palette()
        palette.setColor(QPalette.Window, QColor(MAIN_BACKGROUND))
        main_frame.setPalette(palette)

        layout = QHBoxLayout(main_frame)
        layout.setSpacing(15)
        layout.setContentsMargins(0, 0, 0, 0)  # Remove outer white border

        # Create scrollable left panel for small screens
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_scroll.setStyleSheet(f"""
                QScrollArea {{ 
                    border: none; 
                    background-color: {MAIN_BACKGROUND}; 
                }}
            """)

        left_frame = self.create_left_frame(main_frame)
        left_scroll.setWidget(left_frame)

        # Responsive sizing: minimum 320px, preferred 360px, maximum 400px
        left_scroll.setMinimumWidth(320)
        left_scroll.setMaximumWidth(400)
        left_frame.setMinimumWidth(300)

        right_frame = self.create_right_frame(main_frame)

        # Add to layout with stretch factors
        layout.addWidget(left_scroll, 0)  # Fixed proportion
        layout.addWidget(right_frame, 1)  # Expands to fill

        return main_frame

    # -------------------------------------------------------------------------
    # HELPER
    # -------------------------------------------------------------------------
    def createLabel(self, text, align=Qt.AlignLeft, bold=False):
        label = QLabel(text)
        label.setAlignment(align)
        if bold:
            label.setStyleSheet("font-size: 9px; color: {Colors.TEXT_SECONDARY}; font-weight: 600;")
        else:
            label.setStyleSheet("font-size: 9px; color: {Colors.TEXT_TERTIARY};")
        return label

    def apply_card_style(self, card):
        """Apply consistent card styling with sharp rectangular design."""
        card.setStyleSheet(UITheme.get_card_style())

    def create_section_header(self, text):
        """Create a section header label with blue accent colour."""
        label = QLabel(text)
        label.setStyleSheet(UITheme.get_header_style_section(size=11))
        return label

    # -------------------------------------------------------------------------
    # LEFT FRAME
    # -------------------------------------------------------------------------
    def create_left_frame(self, parent):
        container = QWidget(parent)
        container.setStyleSheet(f"background-color: {MAIN_BACKGROUND};")
        layout = QVBoxLayout(container)
        layout.setSpacing(12)
        layout.setContentsMargins(15, 15, 0, 15)  # Add margins except right side

        param_frame = self.create_view_data_parameter_frame(container)
        self.apply_card_style(param_frame)
        layout.addWidget(param_frame)

        well_frame = self.create_view_data_well_selection_frame(container)
        self.apply_card_style(well_frame)
        layout.addWidget(well_frame, 1)

        # Give status frame more space by setting a minimum height
        status_frame = self.create_status_frame(container)
        self.apply_card_style(status_frame)
        status_frame.setMinimumHeight(300)  # Much larger default size
        layout.addWidget(status_frame)

        return container

    def create_status_frame(self, parent):
        """
        Create modern status frame with elided labels and dynamic content sections.

        The status frame displays two main sections:
        - File Information: filename, experiment ID, data dimensions, time range, channels
        - Well Assignments: colour-coded well categories (data, controls)

        All text is elided with tooltips on hover to prevent horizontal scrolling.

        Parameters
        ----------
        parent : QWidget
            Parent widget for the status frame.

        Returns
        -------
        QFrame
            Configured status frame with modern card styling.

        Notes
        -----
        This method initialises the frame structure and calls _create_status_content()
        to populate the scrollable content area. Update methods (update_file_info_display,
        update_wells_info_display) should be called separately to populate with data.
        """
        card = ModernCard(parent)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(0)

        # Combined scrollable area for all status information
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setStyleSheet(f"""
            QScrollArea {{
                background-color: {CONTENT_BACKGROUND};
                border: none;
                border-radius: 0px;
            }}
            QScrollBar:vertical {{
                background-color: transparent;
                width: 4px;
                margin: 2px 1px 2px 0px;
            }}
            QScrollBar::handle:vertical {{
                background-color: rgba(157, 163, 174, 0.3);
                border-radius: 0px;
                min-height: 20px;
            }}
            QScrollBar::handle:vertical:hover {{
                background-color: rgba(157, 163, 174, 0.5);
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
                background: none;
            }}
        """)

        # Container for scrollable content
        scroll_content = QWidget()
        scroll_content.setStyleSheet(f"background-color: {CONTENT_BACKGROUND}; border: none;")
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(10)
        scroll_layout.setContentsMargins(8, 8, 8, 8)

        # Create the modern status content sections
        self._create_status_content(scroll_layout)

        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area)

        return card

    def _create_status_content(self, parent_layout):
        """
        Create compact status information display with elided labels.

        This method creates two main sections:
        1. File Information Section - displays file metadata
        2. Well Assignments Section - displays categorised well selections

        The sections are dynamically populated via update methods and use
        ElidedLabel widgets to handle long text gracefully.

        Parameters
        ----------
        parent_layout : QVBoxLayout
            Layout to add status sections to.

        Notes
        -----
        Stores references to sections in self.p for later updates:
        - self.p.file_info_section: File information display area
        - self.p.wells_info_section: Well assignments display area

        The elided text mechanism works as follows:
        - Text exceeding available width is truncated with "..." (ellipsis)
        - Full text appears in a tooltip on hover
        - ElidedLabel automatically updates on resize events
        """
        # File Information Section
        file_section = self._create_info_section("File")
        parent_layout.addWidget(file_section)

        # Store reference for updates
        self.p.file_info_section = file_section

        # Well Assignments Section
        wells_section = self._create_info_section("Wells")
        parent_layout.addWidget(wells_section)

        # Store reference for updates
        self.p.wells_info_section = wells_section

        # Add stretch to push content to top
        parent_layout.addStretch(1)

    def _create_info_section(self, title):
        """
        Create an information section with compact display styling.

        Parameters
        ----------
        title : str
            Section header text (e.g., "File", "Wells").

        Returns
        -------
        QFrame
            Section container with content_layout attribute for dynamic updates.

        Notes
        -----
        The returned frame has a 'content_layout' attribute which is a QVBoxLayout
        that should be used to add information rows dynamically.
        """
        section = QFrame()
        section.setStyleSheet(f"""
            QFrame {{
                background-color: white;
                border-radius: 0px;
                padding: 0px;
                border: 1px solid {Colors.CARD_BORDER};
            }}
        """)

        layout = QVBoxLayout(section)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # Content area - will be populated dynamically (header removed as redundant)
        content_layout = QVBoxLayout()
        content_layout.setSpacing(3)
        content_layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(content_layout)

        # Store reference to content layout for updates
        section.content_layout = content_layout

        return section

    def _add_info_row(self, parent_layout, label_text, value_text):
        """
        Add a compact information row with elided value and hover tooltip.

        Parameters
        ----------
        parent_layout : QVBoxLayout
            Layout to add the row to.
        label_text : str
            Label text (e.g., "File", "Exp ID").
        value_text : str
            Value text to display (will be elided if too long).

        Returns
        -------
        ElidedLabel
            The value label widget for further customisation if needed.

        Notes
        -----
        The elided label automatically shows a tooltip with full text when
        the displayed text is truncated. This provides a clean UI without
        horizontal scrolling whilst maintaining full information accessibility.
        """
        row = QHBoxLayout()
        row.setSpacing(4)
        row.setContentsMargins(0, 0, 0, 0)

        # Label
        label = QLabel(label_text + ":")
        label.setStyleSheet(f"""
            font-size: 9px;
            font-weight: 600;
            color: {Colors.TEXT_SECONDARY};
            min-width: 50px;
            background-color: transparent;
            border: none;
        """)
        label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        row.addWidget(label)

        # Value with elision and tooltip
        value = ElidedLabel(value_text)
        value.setStyleSheet(f"""
            font-size: 9px;
            color: {Colors.TEXT_PRIMARY};
            padding: 2px 4px;
            background-color: {CONTENT_BACKGROUND};
            border-radius: 0px;
            border: none;
        """)
        value.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        value.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        row.addWidget(value, 1)

        parent_layout.addLayout(row)
        return value

    def update_file_info_display(self, file_info_dict):
        """
        Update file information section with compact display.

        Args:
            file_info_dict: Dictionary with keys like 'filename', 'rows', 'columns', etc.

        Elided Text Mechanism
        ----------------------
        The elided text mechanism is used to truncate long text with an ellipsis (...)
        when it exceeds the available width. The full text is preserved and displayed
        in a tooltip when the user hovers over the label.

        Data Flow
        ----------
        1. The method receives a dictionary with file information.
        2. Existing content in the file info section is cleared.
        3. Compact information rows are added based on the available data in the dictionary.
        """
        if not hasattr(self.p, 'file_info_section'):
            return

        # Clear existing content
        layout = self.p.file_info_section.content_layout
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self._clear_layout_recursive(item.layout())

        # Add compact information rows
        if 'filename' in file_info_dict:
            self._add_info_row(layout, "File", file_info_dict['filename'])

        if 'experiment_id' in file_info_dict:
            self._add_info_row(layout, "Exp ID", file_info_dict['experiment_id'])

        if 'rows' in file_info_dict and 'columns' in file_info_dict:
            self._add_info_row(layout, "Size", f"{file_info_dict['rows']} x {file_info_dict['columns']}")

        if 'time_range' in file_info_dict:
            self._add_info_row(layout, "Time", file_info_dict['time_range'])

        if 'channels' in file_info_dict:
            self._add_info_row(layout, "Channels", file_info_dict['channels'])

    def update_wells_info_display(self, wells_dict):
        """
        Update well assignments section with compact display.

        Args:
            wells_dict: Dictionary with keys like 'data', blank, 'pos_ctrl', etc.

        Elided Text Mechanism
        ----------------------
        The elided text mechanism is used to truncate long text with an ellipsis (...)
        when it exceeds the available width. The full text is preserved and displayed
        in a tooltip when the user hovers over the label.

        Data Flow
        ----------
        1. The method receives a dictionary with well assignment information.
        2. Existing content in the wells info section is cleared.
        3. Information rows are added for each category (data, controls) present in the dictionary.
        """
        if not hasattr(self.p, 'wells_info_section'):
            return

        # Clear existing content
        layout = self.p.wells_info_section.content_layout
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self._clear_layout_recursive(item.layout())

        # Category display order and labels
        categories = [
            ('data', 'Data', '#4CAF50'),
            ('blank', 'Blank', '#F44336'),
            ('neg_ctrl', 'Neg Ctrl', '#000000'),
            ('pos_ctrl', 'Pos Ctrl', Colors.ACCENT_BLUE),
            ('donor_ctrl', 'Donor', '#9C27B0'),
            ('acceptor_ctrl', 'Acceptor', '#FF9800'),
            ('blocked_ctrl', 'Blocked', '#795548'),
        ]

        for key, label, color in categories:
            if key in wells_dict and wells_dict[key]:
                wells_text = wells_dict[key]
                self._add_colored_info_row(layout, label, wells_text, color)

    def _add_colored_info_row(self, parent_layout, label_text, value_text, color):
        """Add information row with coloured indicator."""
        row = QHBoxLayout()
        row.setSpacing(4)
        row.setContentsMargins(0, 0, 0, 0)

        # Colour indicator
        indicator = QLabel("*")
        indicator.setStyleSheet(f"""
            font-size: 10px;
            color: {color};
            min-width: 12px;
            max-width: 12px;
            background-color: transparent;
            border: none;
        """)
        indicator.setAlignment(Qt.AlignCenter | Qt.AlignTop)
        row.addWidget(indicator)

        # Label
        label = QLabel(label_text + ":")
        label.setStyleSheet(f"""
            font-size: 9px;
            font-weight: 600;
            color: {Colors.TEXT_SECONDARY};
            min-width: 45px;
            background-color: transparent;
            border: none;
        """)
        label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        row.addWidget(label)

        # Value with elision
        value = ElidedLabel(value_text)
        value.setStyleSheet(f"""
            font-size: 9px;
            color: {Colors.TEXT_PRIMARY};
            padding: 2px 4px;
            background-color: {CONTENT_BACKGROUND};
            border-left: 2px solid {color};
            border-radius: 0px;
        """)
        value.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        value.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        row.addWidget(value, 1)

        parent_layout.addLayout(row)
        return value

    def _clear_layout_recursive(self, layout):
        """Recursively clear a layout."""
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self._clear_layout_recursive(item.layout())

    # -------------------------------------------------------------------------
    # RIGHT FRAME
    # -------------------------------------------------------------------------
    def create_right_frame(self, parent):
        # Use plain QWidget with grey background (no white border/card)
        container = QWidget(parent)
        container.setStyleSheet(f"background-color: {MAIN_BACKGROUND};")
        layout = QVBoxLayout(container)
        layout.setSpacing(8)
        layout.setContentsMargins(0, 15, 15, 15)  # Add margins except left side

        # Floating toolbar - aligned with time range frame top
        layout.addWidget(self.create_toolbar(container))

        self.p.plot_frame = QFrame(container)
        self.p.plot_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {CARD_BACKGROUND};
                border: 1px solid {CARD_BORDER_COLOUR};
                border-radius: 0px;
            }}
        """)
        # Set size policy to ensure plot expands properly
        self.p.plot_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Initialise plot_frame with a layout
        plot_layout = QVBoxLayout(self.p.plot_frame)
        plot_layout.setContentsMargins(5, 5, 5, 5)
        plot_layout.setSpacing(0)  # Remove spacing to allow plot to fill frame

        layout.addWidget(self.p.plot_frame, 1)
        return container

    def create_toolbar(self, parent):
        """
        Create toolbar matching card styling - aligned with time range frame.
        """
        toolbar = QFrame(parent)
        # Match the card background and use minimal border
        toolbar.setStyleSheet(f"""
            QFrame {{
                background-color: {CARD_BACKGROUND};
                border: 1px solid {CARD_BORDER_COLOUR};
                border-radius: 0px;
                padding: 3px;
            }}
        """)

        # Single row layout
        main_layout = QHBoxLayout(toolbar)
        main_layout.setSpacing(4)
        main_layout.setContentsMargins(6, 6, 6, 6)

        # Responsive button sizing
        button_size_standard = QSize(70, 28)
        button_size_analysis = QSize(140, 32)  # Larger for clearer text

        def add_button(text, callback, tooltip, highlighted=False):
            """Add button with tooltip."""
            b = QToolButton()
            b.setText(text)
            b.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            b.setToolTip(tooltip)

            if highlighted:
                b.setMinimumSize(button_size_analysis)
                b.setStyleSheet(UITheme.get_button_style_toolbar_highlighted())
            else:
                b.setMinimumSize(button_size_standard)
                b.setStyleSheet(UITheme.get_button_style_toolbar())

            b.setMaximumHeight(28)
            b.clicked.connect(callback)
            main_layout.addWidget(b)

        # Display Controls
        add_button("Scatter", self.p.toggle_scatter, "Toggle scatter points on/off\nKeyboard: S")
        add_button("Grid", self.p.toggle_grid, "Toggle plot grid on/off\nKeyboard: G")
        add_button("B/W", self.p.toggle_black_white, "Toggle black/white mode (dark background)\nKeyboard: B")
        add_button("Legend", self.p.toggle_legend, "Toggle legend visibility\nKeyboard: L")

        # Separator
        main_layout.addSpacing(12)

        # View and Export
        add_button("Plot Window", self.p.show_plot_in_window, "Open plot in separate window\nKeyboard: W")
        add_button("Export", self.p.export_function_view_data, "Export plot and data to file\nKeyboard: Ctrl+E")

        # Separator
        main_layout.addSpacing(12)

        # Analysis Tools
        add_button("Standard Curve", self.p.create_standard_curve,
                  "Create calibration curve from control wells\nKeyboard: Ctrl+S", highlighted=True)
        add_button("Conversion", self.p.open_conversion_tab,
                  "Convert fluorescence to concentrations\nKeyboard: Ctrl+C", highlighted=True)

        # Push everything to the left
        main_layout.addStretch()

        return toolbar

    # -------------------------------------------------------------------------
    # PARAMETER FRAME
    # -------------------------------------------------------------------------
    def create_view_data_parameter_frame(self, parent):
        card = ModernCard(parent)
        layout = QVBoxLayout(card)
        layout.setSpacing(10)
        layout.setContentsMargins(12, 12, 12, 12)

        # Header with icon
        header = self.create_section_header("Time Range")
        layout.addWidget(header)

        # Time range controls - modernized layout
        time_layout = QGridLayout()
        time_layout.setSpacing(8)
        time_layout.setHorizontalSpacing(10)
        time_layout.setVerticalSpacing(6)

        # Start Time
        start_label = QLabel("Start Time (min)")
        start_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        start_label.setStyleSheet("font-size: 9px; font-weight: 600; color: {Colors.TEXT_SECONDARY}; background-color: transparent; border: none;")
        time_layout.addWidget(start_label, 0, 0)

        self.p.start_time_spinbox = QDoubleSpinBox()
        self.p.start_time_spinbox.setDecimals(2)
        self.p.start_time_spinbox.setRange(0, 1e9)  # Increased range to support very long experiments
        self.p.start_time_spinbox.setSingleStep(1.0)  # Step by 1 minute for reasonable increments
        self.p.start_time_spinbox.setMinimumWidth(90)  # Compact but sufficient for large numbers
        self.p.start_time_spinbox.setMaximumHeight(21)  # Reduced height by ~25%
        self.p.start_time_spinbox.setAlignment(Qt.AlignLeft)
        self.p.start_time_spinbox.setStyleSheet(UITheme.get_spinbox_style(decimals=True))
        self.p.start_time_spinbox.valueChanged.connect(self.p.update_plot)
        time_layout.addWidget(self.p.start_time_spinbox, 1, 0)

        # End Time
        end_label = QLabel("End Time (min)")
        end_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        end_label.setStyleSheet("font-size: 9px; font-weight: 600; color: {Colors.TEXT_SECONDARY}; background-color: transparent; border: none;")
        time_layout.addWidget(end_label, 0, 1)

        self.p.end_time_spinbox = QDoubleSpinBox()
        self.p.end_time_spinbox.setDecimals(2)
        self.p.end_time_spinbox.setRange(0, 1e9)  # Increased range to support very long experiments
        self.p.end_time_spinbox.setSingleStep(1.0)  # Step by 1 minute for reasonable increments
        self.p.end_time_spinbox.setMinimumWidth(90)  # Compact but sufficient for large numbers
        self.p.end_time_spinbox.setMaximumHeight(21)  # Reduced height by ~25%
        self.p.end_time_spinbox.setAlignment(Qt.AlignLeft)
        self.p.end_time_spinbox.setStyleSheet(UITheme.get_spinbox_style(decimals=True))
        self.p.end_time_spinbox.valueChanged.connect(self.p.update_plot)
        time_layout.addWidget(self.p.end_time_spinbox, 1, 1)

        layout.addLayout(time_layout)

        # Time range quick presets
        presets_layout = QHBoxLayout()
        presets_layout.setSpacing(4)

        def create_preset_button(text, callback, tooltip):
            """Helper to create consistent preset buttons."""
            btn = QPushButton(text)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #FFFFFF;
                    border: 1.5px solid {Colors.BORDER_MEDIUM};
                    border-radius: 0px;
                    padding: 3px 6px;
                    font-size: 8px;
                    font-weight: 600;
                    min-height: 20px;
                    max-height: 24px;
                    color: {Colors.TEXT_PRIMARY};
                }
                QPushButton:hover {
                    background-color: {Colors.INFO_PANEL_INTERNAL_BG};
                    border-color: {Colors.ACCENT_BLUE};
                }
                QPushButton:pressed {
                    background-color: {Colors.INFO_PANEL_INTERNAL_BG};
                }
            """)
            btn.setToolTip(tooltip)
            btn.clicked.connect(callback)
            btn.setCursor(Qt.PointingHandCursor)
            return btn

        # Add preset buttons
        presets_layout.addWidget(create_preset_button("Full", self.p.preset_full_range, "Set to full data range"))
        presets_layout.addWidget(create_preset_button("30 min", self.p.preset_first_30min, "Set to first 30 minutes"))
        presets_layout.addWidget(create_preset_button("1 h", self.p.preset_first_hour, "Set to first hour"))
        presets_layout.addWidget(create_preset_button("Last h", self.p.preset_last_hour, "Set to last hour"))
        presets_layout.addStretch()

        layout.addLayout(presets_layout)

        # Spacing between sections
        layout.addSpacing(4)

        # Time unit controls - modernized layout
        unit_layout = QGridLayout()
        unit_layout.setSpacing(8)
        unit_layout.setHorizontalSpacing(10)
        unit_layout.setVerticalSpacing(6)

        # Source Unit
        source_label = QLabel("Source Time Unit")
        source_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        source_label.setStyleSheet("font-size: 9px; font-weight: 600; color: {Colors.TEXT_SECONDARY}; background-color: transparent; border: none;")
        unit_layout.addWidget(source_label, 0, 0)

        self.p.original_unit_combo = QComboBox()
        self.p.original_unit_combo.addItems(["Minutes", "Seconds"])
        self.p.original_unit_combo.setStyleSheet(UITheme.get_combobox_style())
        self.p.original_unit_combo.setMaximumHeight(21)
        unit_layout.addWidget(self.p.original_unit_combo, 1, 0)

        # Display Unit
        display_label = QLabel("Display Time Unit")
        display_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        display_label.setStyleSheet("font-size: 9px; font-weight: 600; color: {Colors.TEXT_SECONDARY}; background-color: transparent; border: none;")
        unit_layout.addWidget(display_label, 0, 1)

        self.p.display_unit_combo = QComboBox()
        self.p.display_unit_combo.addItems(["Seconds", "Minutes", "Hours", "Days", "Weeks"])
        self.p.display_unit_combo.setCurrentText("Minutes")  # Set default to Minutes
        self.p.display_unit_combo.currentIndexChanged.connect(self.p.update_plot)
        self.p.display_unit_combo.setStyleSheet(UITheme.get_combobox_style())
        self.p.display_unit_combo.setMaximumHeight(21)
        unit_layout.addWidget(self.p.display_unit_combo, 1, 1)

        layout.addLayout(unit_layout)

        # Separator
        layout.addSpacing(6)
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet(f"background-color: {CARD_BORDER_COLOUR}; max-height: 1px; margin: 4px 0; border: none;")
        layout.addWidget(separator)
        layout.addSpacing(3)

        # Channel selector - NO LABEL, just the buttons
        layout.addLayout(self.create_channel_checkboxes(card))

        return card

    def create_channel_checkboxes(self, parent):
        layout = QGridLayout()
        layout.setSpacing(6)

        if self.p.merged_dataframe is None or self.p.merged_dataframe.empty:
            return layout

        # Check if 'Well' column exists; if not, add a default one
        if 'Well' not in self.p.merged_dataframe.columns:
            self.p.logger.debug("No 'Well' column found - adding default 'Well' column for minimal format")
            self.p.merged_dataframe.insert(0, 'Well', 'Default')

        # Get unique channels and filter out NaN values
        unique_channels = self.p.merged_dataframe["Well"].unique()

        # Filter out NaN/None values
        unique_channels = [ch for ch in unique_channels if pd.notna(ch)]

        if not unique_channels:
            self.p.logger.warning("No valid channels found after filtering NaN values")
            return layout

        self.p.logger.debug(f"Found {len(unique_channels)} valid channels: {unique_channels}")

        self.p.selected_channels_checkboxes = {}
        self.p.channel_colours = {}

        for i, channel in enumerate(unique_channels):
            # Get display name from channel map
            display = self.p.channel_map.get(channel, channel)
            display = str(display)

            wl = self.p.extract_wavelength_from_channel(display)
            colour = self.p.get_wavelength_colour(wl) if wl else "#8BC34A"
            self.p.channel_colours[channel] = colour

            # Extract wavelength label for display
            matches = re.findall(r'(\d{3})(?:[-/](\d{2,3}))?', display)
            wavelengths = [f"{m[0]}/{m[1]}" if m[1] else m[0] for m in matches]

            if len(wavelengths) == 2:
                label_text = f"{wavelengths[0]} Ex / {wavelengths[1]} Em"
            elif len(wavelengths) == 1:
                label_text = f"{wavelengths[0]} nm"
            else:
                label_text = display

            label = QLabel(label_text)
            label.setAlignment(Qt.AlignCenter)
            label.setFixedHeight(26)
            label._selected = False
            label.setStyleSheet(f"""
                    QLabel {{
                        border: 2px solid {colour};
                        border-radius: 6px;
                        padding: 4px;
                        background-color: white;
                        color: {Colors.TEXT_SECONDARY};
                        font-size: 9px;
                        font-weight: 500;
                    }}
                    QLabel:hover {{
                        background-color: {Colors.SECTION_BACKGROUND};
                    }}
                """)
            label.mousePressEvent = lambda e, ch=channel: self.toggle_channel_selection(ch)
            self.p.selected_channels_checkboxes[channel] = label
            layout.addWidget(label, i // 2, i % 2)

        return layout

    def toggle_channel_selection(self, channel):
        """Toggle channel selection with modern styling."""
        lbl = self.p.selected_channels_checkboxes[channel]
        selected = getattr(lbl, "_selected", False)
        colour = self.p.channel_colours.get(channel, "#8BC34A")

        from core.data_frame_processor.plot_utils import PlotUtils
        text_colour = PlotUtils.get_contrast_text_colour(colour)

        if selected:
            # Deselect
            lbl.setStyleSheet(f"""
                QLabel {{
                    font-size: 9px;
                    font-weight: 500;
                    padding: 4px;
                    border: 2px solid {colour};
                    border-radius: 6px;
                    background-color: white;
                    color: {Colors.TEXT_SECONDARY};
                }}
                QLabel:hover {{
                    background-color: {Colors.SECTION_BACKGROUND};
                }}
            """)
            lbl._selected = False
        else:
            # Select
            lbl.setStyleSheet(f"""
                QLabel {{
                    font-size: 9px;
                    font-weight: 600;
                    padding: 4px;
                    border: 2px solid {colour};
                    border-radius: 6px;
                    background-color: {colour};
                    color: {text_colour};
                }}
                QLabel:hover {{
                    opacity: 0.9;
                }}
            """)
            lbl._selected = True

        self.p.plot_in_viewdataframe_action(channel, lbl)

    # -------------------------------------------------------------------------
    # WELL SELECTION FRAME
    # -------------------------------------------------------------------------
    def create_view_data_well_selection_frame(self, parent):
        """96-well plate visual selection - optimised for accessibility."""
        card = ModernCard(parent)
        layout = QVBoxLayout(card)
        layout.setSpacing(6)
        layout.setContentsMargins(10, 10, 10, 10)

        # Header - matching Parameters frame style exactly
        header = self.create_section_header("Well Selection")
        layout.addWidget(header)

        # Sizing constants - account for 1px border on each side
        WELL_SIZE = 24  # Inner well size
        LABEL_W = 20  # Row label width
        GRID_HSP = 0  # Horizontal spacing
        GRID_VSP = 0  # Vertical spacing

        def normalise_well_id(name):
            """
            Standardise well identifiers to a consistent format.

            Handles variations like:
            - 'A01' -> 'A1' (removes leading zeros)
            - 'A 10' -> 'A10' (removes spaces)
            - 'A1' -> 'A1' (already normalised)

            Args:
                name: Column name that may contain a well identifier

            Returns:
                Standardised well ID (e.g., 'A1', 'B12') or None if not a valid well
            """
            if not isinstance(name, str):
                return None
            # Fixed regex: Try two-digit numbers (10-12) BEFORE single digits (1-9)
            m = re.search(r'([A-H])\s*0*(1[0-2]|[1-9])', name.strip().upper())
            normalised = f"{m.group(1)}{int(m.group(2))}" if m else None
            return normalised

        # Create plate map from column names
        self.p.logger.info("Creating well selection grid from %d columns", len(self.p.columns))
        plate_map = {normalise_well_id(col): col for col in sorted(self.p.columns) if normalise_well_id(col)}
        self.p.logger.info("Created plate map with %d entries", len(plate_map))

        # Create grid with proper alignment
        grid = QGridLayout()
        grid.setHorizontalSpacing(GRID_HSP)
        grid.setVerticalSpacing(GRID_VSP)
        grid.setContentsMargins(0, 0, 0, 0)
        # Note: SetFixedSize removed to prevent rendering issues on initial load

        # Header row - column numbers
        corner = QLabel("")
        corner.setFixedSize(LABEL_W, WELL_SIZE)
        grid.addWidget(corner, 0, 0)

        for c in range(1, 13):
            lbl = QLabel(str(c))
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setFixedSize(WELL_SIZE, WELL_SIZE)
            lbl.setStyleSheet("font-size: 9px; font-weight: 600; color: {Colors.TEXT_SECONDARY}; background-color: transparent; border: none;")
            grid.addWidget(lbl, 0, c)

        # Well grid - 96-well plate (A-H, 1-12)
        for r_i, r_lbl in enumerate("ABCDEFGH", 1):
            lbl = QLabel(r_lbl)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setFixedSize(LABEL_W, WELL_SIZE)
            lbl.setStyleSheet("font-size: 9px; font-weight: 600; color: {Colors.TEXT_SECONDARY}; background-color: transparent; border: none;")
            grid.addWidget(lbl, r_i, 0)

            for c in range(1, 13):
                wid = f"{r_lbl}{c}"
                if wid in plate_map:
                    well_id = plate_map[wid]

                    btn = WellButton(well_id, self.p.well_selection_manager)
                    btn.setFixedSize(WELL_SIZE, WELL_SIZE)
                    btn.setStyleSheet(self._well_button_style())

                    self.p.well_buttons[well_id] = btn
                    self.p.well_selection_manager.well_buttons[well_id] = btn
                    grid.addWidget(btn, r_i, c)
                else:
                    # Use QFrame instead of QLabel for proper border-radius rendering
                    ph = QFrame()
                    ph.setFixedSize(WELL_SIZE, WELL_SIZE)
                    ph.setStyleSheet("""
                        QFrame {
                            background-color: #D0D0D0;
                            border: 1px solid #A0A0A0;
                            border-radius: 4px;
                        }
                    """)
                    grid.addWidget(ph, r_i, c)

        # Wrap grid in a widget to control alignment
        grid_container = QWidget()
        grid_container.setStyleSheet("background-color: transparent; border: none;")
        grid_container.setLayout(grid)

        # Force proper layout before showing to prevent overlapping squares
        grid_container.updateGeometry()
        from PyQt5.QtWidgets import QApplication
        QApplication.processEvents()

        layout.addWidget(grid_container, alignment=Qt.AlignLeft)

        # Category assignment label
        category_label = QLabel("Assign Selected Wells:")
        category_label.setStyleSheet(f"font-size: 9px; font-weight: 600; color: {Colors.TEXT_SECONDARY}; margin-top: 2px; margin-bottom: 4px; background-color: transparent; border: none;")
        layout.addWidget(category_label)

        # First row: Clear (with long-click) and Data (with long-click) - Sharp corners for distinction
        special_buttons_row = QHBoxLayout()
        special_buttons_row.setSpacing(6)

        # Clear button with long-click: hold 3s to clear all assignments
        clear_btn = LongClickButton(
            "Clear",
            short_callback=self.p.add_clear_wells_on_click,
            long_callback=self.p.clear_all_assignments,
            tooltip_text="Hold for 3 seconds to erase all assignments.",
            parent=card
        )
        clear_btn.setStyleSheet(UITheme.get_button_style_category("#4F0142"))  # Violet
        clear_btn.setCursor(Qt.PointingHandCursor)
        special_buttons_row.addWidget(clear_btn)

        # Data button with long-click: hold 3s to assign all wells to Data
        data_btn = LongClickButton(
            "Data",
            short_callback=self.p.add_data_wells_on_click,
            long_callback=lambda: self.p.well_selection_manager.assign_all('data'),
            tooltip_text="Hold for 3 seconds to assign all wells as Data.",
            parent=card
        )
        data_btn.setStyleSheet(UITheme.get_button_style_category("#4CAF50"))
        data_btn.setCursor(Qt.PointingHandCursor)
        special_buttons_row.addWidget(data_btn)

        layout.addLayout(special_buttons_row)

        # Second row: Blank, Neg Ctrl, and Pos Ctrl
        button_row2 = QHBoxLayout()
        button_row2.setSpacing(4)

        # Blank button
        blank_btn = QPushButton("Blank")
        blank_btn.setStyleSheet(UITheme.get_button_style_category("#F44336"))
        blank_btn.setCursor(Qt.PointingHandCursor)
        blank_btn.clicked.connect(self.p.add_blank_wells_on_click)
        button_row2.addWidget(blank_btn, 3)

        # Neg Ctrl button - black colour
        neg_ctrl_btn = QPushButton("Neg Ctrl")
        neg_ctrl_btn.setStyleSheet(UITheme.get_button_style_category("#000000"))
        neg_ctrl_btn.setCursor(Qt.PointingHandCursor)
        neg_ctrl_btn.clicked.connect(self.p.add_negctrl_wells_on_click)
        button_row2.addWidget(neg_ctrl_btn, 3)

        # Pos Ctrl button
        pos_ctrl_btn = QPushButton("Pos Ctrl")
        pos_ctrl_btn.setStyleSheet(UITheme.get_button_style_category("#2196F3"))
        pos_ctrl_btn.setCursor(Qt.PointingHandCursor)
        pos_ctrl_btn.clicked.connect(self.p.add_posctrl_wells_on_click)
        button_row2.addWidget(pos_ctrl_btn, 3)

        layout.addLayout(button_row2)

        # Third row: Donor and Acceptor controls - Full width
        button_row3 = QHBoxLayout()
        button_row3.setSpacing(4)

        for text, color, callback in [
            ("Donor Control", "#9C27B0", self.p.add_donorctrl_wells_on_click),
            ("Acceptor Control", "#FF9800", self.p.add_acceptorctrl_wells_on_click),
            ("Blocked Ctrl", "#795548", self.p.add_blockedctrl_wells_on_click),
        ]:
            btn = QPushButton(text)
            btn.setStyleSheet(UITheme.get_button_style_category(color))
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(callback)
            button_row3.addWidget(btn)

        layout.addLayout(button_row3)

        # Add stretch at the bottom to push all content to the top
        layout.addStretch(1)

        return card

    def _well_button_style(self):
        """Perfect well button styling - larger, more accessible."""
        return f"""
            QPushButton {{
                background-color: white;
                border: 2px solid {Colors.CARD_BORDER};
                border-radius: 4px;
            }}
            QPushButton:hover {{
                background-color: {Colors.INFO_PANEL_INTERNAL_BG};
                border-color: {Colors.ACCENT_BLUE};
            }}
            QPushButton:checked {{
                background-color: #8BC34A;
                border-color: #689F38;
            }}
        """

    # -------------------------------------------------------------------------
    # MISC
    # -------------------------------------------------------------------------
    def clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()