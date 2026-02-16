"""
Unified Bimolecular Widget Factory - 3-Column Layout with Boxed Sections
-------------------------------------------------------------------------
Creates the bimolecular reaction fitting widget with a horizontal 3-column
layout optimised for width utilisation and minimal vertical scrolling.

Visual Design:
  - Light grey background (#E9E9E9) as base canvas
  - White rectangular containers for logical units (sharp corners)
  - Blue-toned section headers with larger font
  - Clear internal padding and consistent spacing
  - Matches the View Data tab visual language

Layout structure:
  Column 1 (narrow, fixed ~200px): Model identity and context
    - Box 1: Reaction model description
    - Box 2: Covered assay types
  Column 2 (wide, stretches): Z concentration workflow with mode switching
    - Box 1: Mode selector (Endpoint vs Manual)
    - Box 2: Mode-specific controls (stacked)
  Column 3 (medium, fixed ~240px): Global model parameters
    - Box 1: Global fitting parameters
    - Box 2: Status display

Author: Krizan Jurinovic
Date: January 2026
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QPushButton,
    QDoubleSpinBox, QSpinBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QRadioButton, QButtonGroup, QStackedWidget, QFormLayout,
    QSizePolicy, QScrollArea, QAbstractSpinBox
)
from PyQt5.QtCore import Qt

from core.common.ui_theme import UITheme, Colours
from .gui_widget_factory import PerTraceTable, block_wheel_event


def _create_section_box(title=None):
    """
    Create a white boxed section container matching View Data tab styling.

    Parameters
    ----------
    title : str, optional
        Section header text. If provided, adds a blue header label.

    Returns
    -------
    tuple
        (QFrame, QVBoxLayout) - The container frame and its layout.
    """
    box = QFrame()
    box.setStyleSheet(f"""
        QFrame {{
            background-color: {Colours.CARD_BACKGROUND};
            border: 1px solid {Colours.CARD_BORDER};
            border-radius: 0px;
        }}
    """)

    layout = QVBoxLayout(box)
    layout.setContentsMargins(10, 10, 10, 10)
    layout.setSpacing(8)

    if title:
        header = QLabel(title)
        header.setStyleSheet(UITheme.get_header_style_section(size=11))
        layout.addWidget(header)

    return box, layout


def create_bimolecular_widget(on_input_mode_changed=None, callbacks=None):
    """
    Create bimolecular reaction widget with 3-column horizontal layout.

    Uses white boxed sections on a grey background to match the View Data
    tab visual language whilst maintaining horizontal orientation.

    Parameters
    ----------
    on_input_mode_changed : callable, optional
        Callback when input mode changes. Signature: f(mode: str)
        where mode is 'endpoint' or 'manual'.
    callbacks : dict, optional
        - 'on_detect_endpoints': callable for endpoint detection button

    Returns
    -------
    QWidget
        Widget with 3-column layout for bimolecular fitting parameters.
    """
    callbacks = callbacks or {}

    # Main widget with grey background
    widget = QWidget()
    widget.setStyleSheet(f"background-color: {Colours.MAIN_BACKGROUND};")

    main_layout = QHBoxLayout(widget)
    main_layout.setContentsMargins(8, 8, 8, 8)
    main_layout.setSpacing(12)

    # =========================================================================
    # COLUMN 1: Model Identity and Context (narrow, fixed width)
    # =========================================================================
    col1 = _create_column1_model_context()
    col1.setFixedWidth(180)
    col1.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
    main_layout.addWidget(col1)

    # =========================================================================
    # COLUMN 2: Global Model Parameters (medium, fixed width)
    # =========================================================================
    col3, col3_widgets = _create_column3_global_params()
    col3.setFixedWidth(200)
    col3.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
    main_layout.addWidget(col3)

    # =========================================================================
    # COLUMN 3: Z Concentration Workflow (wide, stretches to fill space)
    # =========================================================================
    col2, col2_widgets = _create_column2_z_workflow(callbacks)
    col2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
    main_layout.addWidget(col2, stretch=1)

    # =========================================================================
    # STATE MANAGEMENT
    # =========================================================================
    endpoint_radio = col2_widgets['endpoint_radio']
    manual_radio = col2_widgets['manual_radio']
    mode_panels_stacked = col2_widgets['mode_panels_stacked']
    endpoint_panel = col2_widgets['endpoint_panel']
    manual_panel = col2_widgets['manual_panel']

    kf_guess_spinbox = col3_widgets['kf_guess_spinbox']
    X0_guess_spinbox = col3_widgets['X0_guess_spinbox']
    status_text = col3_widgets['status_text']

    # Per-trace Z0 table manager
    z0_table_manager = PerTraceTable(
        table=manual_panel.widgets['per_trace_Z0_table'],
        min_val=-1.0, max_val=10000.0, decimals=2, step=0.5,
        tooltip_template="[Z]_0 for {name}. Set to -1 for default.",
        use_default_marker=True,
    )

    def get_input_mode():
        """Return current input mode: 'endpoint' or 'manual'."""
        return 'endpoint' if endpoint_radio.isChecked() else 'manual'

    def on_input_mode_toggled(checked):
        """Handle input mode radio button toggle."""
        if not checked:
            return
        mode = get_input_mode()
        mode_panels_stacked.setCurrentIndex(0 if mode == 'endpoint' else 1)
        if on_input_mode_changed:
            on_input_mode_changed(mode)

    endpoint_radio.toggled.connect(on_input_mode_toggled)
    manual_radio.toggled.connect(on_input_mode_toggled)

    def get_parameters():
        """Get all bimolecular fitting parameters."""
        mode = get_input_mode()
        if mode == 'endpoint':
            return {
                'input_mode': 'endpoint',
                'X0_guess': X0_guess_spinbox.value(),
                'kf_guess': kf_guess_spinbox.value(),
            }
        else:
            return {
                'input_mode': 'manual',
                'X0_guess': X0_guess_spinbox.value(),
                'kf_guess': kf_guess_spinbox.value(),
                'Z0_default': manual_panel.widgets['default_Z0_spinbox'].value(),
            }

    def update_status(text):
        """Update the status display text."""
        status_text.setText(text)

    # Apply all / Clear buttons
    def apply_default_to_all():
        z0_table_manager.set_all(manual_panel.widgets['default_Z0_spinbox'].value())

    def clear_overrides():
        z0_table_manager.set_all(-1.0)

    manual_panel.widgets['apply_all_button'].clicked.connect(apply_default_to_all)
    manual_panel.widgets['clear_button'].clicked.connect(clear_overrides)

    # =========================================================================
    # STORE WIDGET REFERENCES
    # =========================================================================
    widget.widgets = {
        'endpoint_radio': endpoint_radio,
        'manual_radio': manual_radio,
        'mode_panels_stacked': mode_panels_stacked,
        'endpoint_panel': endpoint_panel,
        'manual_panel': manual_panel,
        # Global parameters
        'kf_guess_spinbox': kf_guess_spinbox,
        'X0_guess_spinbox': X0_guess_spinbox,
        'status_text': status_text,
        # Endpoint mode widgets
        'endpoint_X0_guess': X0_guess_spinbox,
        'endpoint_kf_guess': kf_guess_spinbox,
        # Endpoint detection controls
        'endpoint_window_start': endpoint_panel.widgets['endpoint_window_start'],
        'endpoint_window_end': endpoint_panel.widgets['endpoint_window_end'],
        'detect_endpoints_button': endpoint_panel.widgets['detect_endpoints_button'],
        # Manual mode widgets
        'manual_X0_guess': X0_guess_spinbox,
        'manual_Z0_default': manual_panel.widgets['default_Z0_spinbox'],
        'manual_kf_guess': kf_guess_spinbox,
        'per_trace_Z0_table': manual_panel.widgets['per_trace_Z0_table'],
        'per_trace_Z0_spinboxes': z0_table_manager.spinboxes,
    }

    # Store methods (delegate to PerTraceTable)
    widget.get_input_mode = get_input_mode
    widget.get_parameters = get_parameters
    widget.populate_per_trace_Z0_table = z0_table_manager.populate
    widget.get_per_trace_Z0_values = z0_table_manager.get_values
    widget.get_all_per_trace_Z0_values = z0_table_manager.get_all_values
    widget.set_last_per_trace_Z0_values = z0_table_manager.set_last_values
    widget.update_status = update_status

    return widget


def _create_column1_model_context():
    """
    Create Column 1: Model identity and context.

    Contains two white boxed sections:
    - Box 1: Model name and reaction equation
    - Box 2: Covered assay types
    """
    col = QWidget()
    col.setStyleSheet(f"background-color: {Colours.MAIN_BACKGROUND};")
    layout = QVBoxLayout(col)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(8)

    # -------------------------------------------------------------------------
    # Box 1: Model Description
    # -------------------------------------------------------------------------
    box1, box1_layout = _create_section_box("Reaction Model")

    # Reaction equation (compact, single line)
    equation_label = QLabel("X + Z \u2192 Y + W")
    equation_label.setStyleSheet(f"""
        QLabel {{
            font-size: 11pt;
            font-weight: bold;
            padding: 6px 8px;
            background-color: {Colours.INFO_PANEL_TMSD_BG};
            border-left: 3px solid {Colours.INFO_PANEL_TMSD_BORDER};
            border-radius: 0px;
            color: {Colours.INFO_PANEL_TMSD_TEXT};
        }}
    """)
    box1_layout.addWidget(equation_label)

    # One-line description with tooltip for details
    desc_label = QLabel("Fits k<sub>f</sub> and [X]<sub>0</sub>")
    desc_label.setStyleSheet(UITheme.get_label_style_tertiary())
    desc_label.setToolTip(
        "dY/dt = k_f x (X0 - Y) x (Z0 - Y)\n\n"
        "One concentration is fixed ([Z]0),\n"
        "the other is fitted ([X]0) along with k_f."
    )
    box1_layout.addWidget(desc_label)

    layout.addWidget(box1)

    # -------------------------------------------------------------------------
    # Box 2: Covered Assay Types
    # -------------------------------------------------------------------------
    box2, box2_layout = _create_section_box("Covers")

    assay_items = [
        ("TMSD", Colours.INFO_PANEL_TMSD_BG, Colours.INFO_PANEL_TMSD_TEXT),
        ("Internal TMSD", Colours.INFO_PANEL_INTERNAL_BG, Colours.INFO_PANEL_INTERNAL_TEXT),
        ("HMSD", Colours.INFO_PANEL_HMSD_BG, Colours.INFO_PANEL_HMSD_TEXT),
    ]

    for assay_name, bg_colour, text_colour in assay_items:
        assay_label = QLabel(assay_name)
        assay_label.setStyleSheet(f"""
            QLabel {{
                font-size: 9pt;
                font-weight: 500;
                padding: 4px 8px;
                background-color: {bg_colour};
                border-radius: 0px;
                color: {text_colour};
            }}
        """)
        box2_layout.addWidget(assay_label)

    layout.addWidget(box2)
    layout.addStretch()

    return col


def _create_column2_z_workflow(callbacks):
    """
    Create Column 2: Z concentration workflow.

    Contains two white boxed sections:
    - Box 1: Mode selector (Endpoint vs Manual)
    - Box 2: Mode-specific controls (stacked widget)
    """
    col = QWidget()
    col.setStyleSheet(f"background-color: {Colours.MAIN_BACKGROUND};")
    layout = QVBoxLayout(col)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(8)

    # -------------------------------------------------------------------------
    # Box 1: Mode Selector
    # -------------------------------------------------------------------------
    box1, box1_layout = _create_section_box("Fixed [Z]\u2080 Source")

    mode_row = QHBoxLayout()
    mode_row.setSpacing(16)

    endpoint_radio = QRadioButton("Endpoint Detection")
    endpoint_radio.setStyleSheet(UITheme.get_radiobutton_style())
    endpoint_radio.setChecked(True)
    endpoint_radio.setToolTip(
        "Automatically determine [Z]0 from endpoint plateau.\n"
        "Used for TMSD and Internal TMSD assays."
    )
    mode_row.addWidget(endpoint_radio)

    # Subtle assay hint for endpoint mode
    endpoint_hint = QLabel("(TMSD)")
    endpoint_hint.setStyleSheet(f"font-size: 8pt; color: {Colours.TEXT_TERTIARY}; font-style: italic;")
    mode_row.addWidget(endpoint_hint)

    mode_row.addSpacing(24)

    manual_radio = QRadioButton("Manual Per-Trace")
    manual_radio.setStyleSheet(UITheme.get_radiobutton_style())
    manual_radio.setToolTip(
        "Manually specify [Z]0 for each trace.\n"
        "Used for HMSD assays where donor concentration is known."
    )
    mode_row.addWidget(manual_radio)

    # Subtle assay hint for manual mode
    manual_hint = QLabel("(HMSD)")
    manual_hint.setStyleSheet(f"font-size: 8pt; color: {Colours.TEXT_TERTIARY}; font-style: italic;")
    mode_row.addWidget(manual_hint)

    input_mode_button_group = QButtonGroup(col)
    input_mode_button_group.addButton(endpoint_radio, 0)
    input_mode_button_group.addButton(manual_radio, 1)

    mode_row.addStretch()
    box1_layout.addLayout(mode_row)

    layout.addWidget(box1)

    # -------------------------------------------------------------------------
    # Box 2: Mode-Specific Controls (Stacked)
    # -------------------------------------------------------------------------
    box2 = QFrame()
    box2.setStyleSheet(f"""
        QFrame {{
            background-color: {Colours.CARD_BACKGROUND};
            border: 1px solid {Colours.CARD_BORDER};
            border-radius: 0px;
        }}
    """)
    box2_layout = QVBoxLayout(box2)
    box2_layout.setContentsMargins(10, 10, 10, 10)
    box2_layout.setSpacing(0)

    mode_panels_stacked = QStackedWidget()

    # Page 0: Endpoint detection panel
    endpoint_panel = _create_endpoint_panel(callbacks)
    mode_panels_stacked.addWidget(endpoint_panel)

    # Page 1: Manual per-trace panel
    manual_panel = _create_manual_panel()
    mode_panels_stacked.addWidget(manual_panel)

    box2_layout.addWidget(mode_panels_stacked)
    layout.addWidget(box2, stretch=1)

    widgets = {
        'endpoint_radio': endpoint_radio,
        'manual_radio': manual_radio,
        'mode_panels_stacked': mode_panels_stacked,
        'endpoint_panel': endpoint_panel,
        'manual_panel': manual_panel,
    }

    return col, widgets


def _create_endpoint_panel(callbacks):
    """
    Create endpoint detection panel with time window controls.

    Layout:
    - Header: Endpoint Detection
    - Time window controls: Start and End spinboxes
    - Detect button
    - Explanatory hint text
    """
    panel = QWidget()
    panel.setStyleSheet(f"background-color: {Colours.CARD_BACKGROUND};")
    layout = QVBoxLayout(panel)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(10)

    # Header
    header = QLabel("Endpoint Detection / Plateau Regime")
    header.setStyleSheet(UITheme.get_header_style_section(size=10))
    layout.addWidget(header)

    # Time window controls
    time_row = QHBoxLayout()
    time_row.setSpacing(8)

    time_label = QLabel("Plateau regime:")
    time_label.setStyleSheet(UITheme.get_label_style_primary())
    time_row.addWidget(time_label)

    endpoint_window_start = QSpinBox()
    endpoint_window_start.setRange(0, 10000)
    endpoint_window_start.setValue(50)
    endpoint_window_start.setSuffix(" min")
    endpoint_window_start.setMinimumWidth(90)
    endpoint_window_start.setStyleSheet(UITheme.get_spinbox_style(decimals=False))
    endpoint_window_start.setToolTip("Start of plateau regime (where reaction has reached completion)")
    time_row.addWidget(endpoint_window_start)

    to_label = QLabel("to")
    to_label.setStyleSheet(UITheme.get_label_style_primary())
    time_row.addWidget(to_label)

    endpoint_window_end = QSpinBox()
    endpoint_window_end.setRange(0, 10000)
    endpoint_window_end.setValue(65)
    endpoint_window_end.setSuffix(" min")
    endpoint_window_end.setMinimumWidth(90)
    endpoint_window_end.setStyleSheet(UITheme.get_spinbox_style(decimals=False))
    endpoint_window_end.setToolTip("End of plateau regime")
    time_row.addWidget(endpoint_window_end)

    time_row.addSpacing(16)

    detect_button = QPushButton("Detect Endpoints")
    detect_button.setStyleSheet(UITheme.get_button_style_primary())
    detect_button.setToolTip(
        "Detect endpoint values from plateau region.\n"
        "Click to analyse the time window and determine [Z]0 for each trace."
    )
    if callbacks.get('on_detect_endpoints'):
        detect_button.clicked.connect(callbacks['on_detect_endpoints'])
    time_row.addWidget(detect_button)

    time_row.addStretch()
    layout.addLayout(time_row)

    # Explanatory hint
    hint_box = QLabel(
        "The endpoint detection algorithm calculates the mean concentration over the\n"
        "specified time window for each visible trace. These values are used as [Z]0\n"
        "for the bimolecular fitting model."
    )
    hint_box.setStyleSheet(f"""
        QLabel {{
            font-size: 8pt;
            color: {Colours.TEXT_TERTIARY};
            padding: 8px;
            background-color: {Colours.SECTION_BACKGROUND};
            border-radius: 0px;
        }}
    """)
    hint_box.setWordWrap(True)
    layout.addWidget(hint_box)

    layout.addStretch()

    panel.widgets = {
        'endpoint_window_start': endpoint_window_start,
        'endpoint_window_end': endpoint_window_end,
        'detect_endpoints_button': detect_button,
    }

    return panel


def _create_manual_panel():
    """
    Create manual per-trace input panel.

    Layout:
    - Header: Manual Per-Trace [Z]0
    - Default Z0 control with Apply to All / Clear buttons
    - Scrollable per-trace table
    """
    panel = QWidget()
    panel.setStyleSheet(f"background-color: {Colours.CARD_BACKGROUND};")
    layout = QVBoxLayout(panel)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(8)

    # Header
    header = QLabel("Manual Per-Trace [Z]\u2080")
    header.setStyleSheet(UITheme.get_header_style_section(size=10))
    layout.addWidget(header)

    # Default Z0 and action buttons row
    control_row = QHBoxLayout()
    control_row.setSpacing(8)

    default_label = QLabel("Default [Z]\u2080:")
    default_label.setStyleSheet(UITheme.get_label_style_primary())
    control_row.addWidget(default_label)

    default_Z0_spinbox = QDoubleSpinBox()
    default_Z0_spinbox.setRange(0.0, 10000.0)
    default_Z0_spinbox.setValue(10.0)
    default_Z0_spinbox.setDecimals(2)
    default_Z0_spinbox.setSingleStep(0.5)
    default_Z0_spinbox.setSuffix(" nM")
    default_Z0_spinbox.setMinimumWidth(100)
    default_Z0_spinbox.setStyleSheet(UITheme.get_spinbox_style(decimals=True))
    default_Z0_spinbox.setFocusPolicy(Qt.StrongFocus)
    default_Z0_spinbox.setButtonSymbols(QAbstractSpinBox.NoButtons)
    block_wheel_event(default_Z0_spinbox)
    default_Z0_spinbox.setToolTip("Default value for traces marked as '(default)'")
    control_row.addWidget(default_Z0_spinbox)

    control_row.addSpacing(16)

    apply_all_button = QPushButton("Apply to All")
    apply_all_button.setStyleSheet(UITheme.get_button_style_standard())
    apply_all_button.setToolTip("Set all traces to the default value")
    control_row.addWidget(apply_all_button)

    clear_button = QPushButton("Clear")
    clear_button.setStyleSheet(UITheme.get_button_style_standard())
    clear_button.setToolTip("Reset all traces to use default")
    control_row.addWidget(clear_button)

    control_row.addStretch()
    layout.addLayout(control_row)

    # Per-trace table with internal scrolling (4-column layout)
    per_trace_Z0_table = QTableWidget()
    per_trace_Z0_table.setColumnCount(4)
    per_trace_Z0_table.setHorizontalHeaderLabels(["Trace", "[Z]\u2080 (nM)", "Trace", "[Z]\u2080 (nM)"])
    per_trace_Z0_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
    per_trace_Z0_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Fixed)
    per_trace_Z0_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
    per_trace_Z0_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Fixed)
    per_trace_Z0_table.setColumnWidth(1, 100)
    per_trace_Z0_table.setColumnWidth(3, 100)
    per_trace_Z0_table.horizontalHeader().setStretchLastSection(False)
    per_trace_Z0_table.setStyleSheet(UITheme.get_table_style())
    per_trace_Z0_table.setToolTip("Per-trace [Z]0 values. Set to -1 for default.")
    per_trace_Z0_table.verticalHeader().setVisible(False)
    per_trace_Z0_table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    layout.addWidget(per_trace_Z0_table, stretch=1)

    panel.widgets = {
        'default_Z0_spinbox': default_Z0_spinbox,
        'apply_all_button': apply_all_button,
        'clear_button': clear_button,
        'per_trace_Z0_table': per_trace_Z0_table,
    }

    return panel


def _create_column3_global_params():
    """
    Create Column 3: Global model parameters.

    Contains two white boxed sections:
    - Box 1: Global fitting parameters (kf, X0)
    - Box 2: Status display
    """
    col = QWidget()
    col.setStyleSheet(f"background-color: {Colours.MAIN_BACKGROUND};")
    layout = QVBoxLayout(col)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(8)

    # -------------------------------------------------------------------------
    # Box 1: Global Fitting Parameters
    # -------------------------------------------------------------------------
    box1, box1_layout = _create_section_box("Global Parameters")

    # Form layout for parameters
    form = QFormLayout()
    form.setSpacing(10)
    form.setContentsMargins(0, 4, 0, 0)
    form.setLabelAlignment(Qt.AlignRight)

    # kf initial guess
    kf_guess_spinbox = QDoubleSpinBox()
    kf_guess_spinbox.setRange(1e3, 1e9)
    kf_guess_spinbox.setValue(1e5)
    kf_guess_spinbox.setDecimals(0)
    kf_guess_spinbox.setSingleStep(1e4)
    kf_guess_spinbox.setMinimumWidth(120)
    kf_guess_spinbox.setStyleSheet(UITheme.get_spinbox_style(decimals=True))
    kf_guess_spinbox.setFocusPolicy(Qt.StrongFocus)
    kf_guess_spinbox.setButtonSymbols(QAbstractSpinBox.NoButtons)
    block_wheel_event(kf_guess_spinbox)
    kf_guess_spinbox.setToolTip("Initial guess for forward rate constant k_f (M^-1 s^-1)")

    kf_label = QLabel("k<sub>f</sub> guess:")
    kf_label.setStyleSheet(UITheme.get_label_style_primary())
    form.addRow(kf_label, kf_guess_spinbox)

    # X0 initial guess
    X0_guess_spinbox = QDoubleSpinBox()
    X0_guess_spinbox.setRange(0.0, 10000.0)
    X0_guess_spinbox.setValue(10.0)
    X0_guess_spinbox.setDecimals(2)
    X0_guess_spinbox.setSingleStep(0.5)
    X0_guess_spinbox.setSuffix(" nM")
    X0_guess_spinbox.setMinimumWidth(120)
    X0_guess_spinbox.setStyleSheet(UITheme.get_spinbox_style(decimals=True))
    X0_guess_spinbox.setFocusPolicy(Qt.StrongFocus)
    X0_guess_spinbox.setButtonSymbols(QAbstractSpinBox.NoButtons)
    block_wheel_event(X0_guess_spinbox)
    X0_guess_spinbox.setToolTip("Initial guess for fitted reactant [X]0 (nM)")

    X0_label = QLabel("[X]\u2080 guess:")
    X0_label.setStyleSheet(UITheme.get_label_style_primary())
    form.addRow(X0_label, X0_guess_spinbox)

    box1_layout.addLayout(form)
    layout.addWidget(box1)

    # -------------------------------------------------------------------------
    # Box 2: Status Display
    # -------------------------------------------------------------------------
    box2, box2_layout = _create_section_box("Status")

    status_text = QLabel("No endpoints detected")
    status_text.setStyleSheet(f"""
        QLabel {{
            font-size: 9pt;
            color: {Colours.TEXT_SECONDARY};
            padding: 8px;
            background-color: {Colours.SECTION_BACKGROUND};
            border-radius: 0px;
        }}
    """)
    status_text.setWordWrap(True)
    status_text.setMinimumHeight(60)
    box2_layout.addWidget(status_text)

    layout.addWidget(box2)
    layout.addStretch()

    widgets = {
        'kf_guess_spinbox': kf_guess_spinbox,
        'X0_guess_spinbox': X0_guess_spinbox,
        'status_text': status_text,
    }

    return col, widgets
