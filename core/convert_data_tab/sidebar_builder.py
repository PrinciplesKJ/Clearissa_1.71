"""Sidebar builder for the Convert Data tab.

Consolidates the fixed-width control stack so gui.py can stay focused on
behavioural logic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.common.ui_theme import Colors, UITheme
from .ui_components import (
    ComboBoxFactory,
    create_section_header,
    create_section_separator,
    create_info_button,
)

if TYPE_CHECKING:  # pragma: no cover - type check only
    from .gui import ConvertDataTab


class SidebarBuilder:
    """Assemble the industrial control sidebar with sharp corners."""

    def __init__(self, host: "ConvertDataTab") -> None:
        self.host = host

    def build(self) -> QFrame:
        sidebar = QFrame(self.host)
        sidebar.setObjectName("convertSidebar")
        sidebar.setFixedWidth(470)
        sidebar.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        sidebar.setStyleSheet(
            f"QFrame#convertSidebar {{ background-color: {Colors.CONTENT_BACKGROUND};"
            f" border: 1px solid {Colors.CARD_BORDER}; border-radius: 0px; }}"
        )

        outer = QVBoxLayout(sidebar)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea(sidebar)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet(
            f"QScrollArea {{ background-color: {Colors.MAIN_BACKGROUND}; border: none; }}"
        )

        right_panel = QWidget(scroll)
        right_panel.setStyleSheet(f"background-color: {Colors.MAIN_BACKGROUND};")
        layout = QVBoxLayout(right_panel)
        layout.setContentsMargins(0, 0, 0, 12)
        layout.setSpacing(12)
        layout.setAlignment(Qt.AlignTop)

        # Top Card: Approach and Method Configuration
        top_card = self._create_card_container(right_panel)
        top_card_layout = QVBoxLayout(top_card)
        top_card_layout.setContentsMargins(10, 10, 10, 10)
        top_card_layout.setSpacing(4)
        self._add_approach_section(top_card_layout, top_card)
        self._add_species_section(top_card_layout)
        layout.addWidget(top_card)

        # Middle Card: Parameters and Action Buttons
        middle_card = self._create_card_container(right_panel)
        middle_card_layout = QVBoxLayout(middle_card)
        middle_card_layout.setContentsMargins(10, 10, 10, 10)
        middle_card_layout.setSpacing(4)
        self._add_parameters_section(middle_card_layout)
        self._add_action_section(middle_card_layout)
        layout.addWidget(middle_card)

        # Bottom Card: Coefficients and Forward
        bottom_card = self._create_card_container(right_panel)
        bottom_card_layout = QVBoxLayout(bottom_card)
        bottom_card_layout.setContentsMargins(10, 10, 10, 10)
        bottom_card_layout.setSpacing(4)
        self._add_matrix_section(bottom_card_layout)
        self._add_forward_section(bottom_card_layout)
        layout.addWidget(bottom_card)

        layout.addStretch()

        scroll.setWidget(right_panel)
        outer.addWidget(scroll)
        return sidebar

    def _create_card_container(self, parent: QWidget) -> QFrame:
        """Create a card-like container with distinct visual styling."""
        card = QFrame(parent)
        card.setObjectName("sidebarCard")
        card.setStyleSheet(
            f"QFrame#sidebarCard {{"
            f" background-color: {Colors.CARD_BACKGROUND};"
            f" border: 1px solid {Colors.CARD_BORDER};"
            f" border-radius: 0px;"
            f"}}"
        )
        card.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        return card

    # ------------------------------------------------------------------ helpers
    def _add_approach_section(self, layout: QVBoxLayout, parent: QWidget) -> None:
        block = QVBoxLayout()
        block.setContentsMargins(0, 0, 0, 4)
        block.setSpacing(3)

        mode_label = QLabel("Mode", parent)
        mode_label.setStyleSheet(UITheme.get_label_style_secondary())
        block.addWidget(mode_label)

        # Horizontal layout for combo + info button
        mode_row = QHBoxLayout()
        mode_row.setSpacing(4)

        host = self.host
        host.approach_combo = QComboBox(parent)
        host.approach_combo.setStyleSheet(UITheme.get_combobox_style())
        ComboBoxFactory.create_approach_combo(host.approach_combo)
        host.approach_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        mode_row.addWidget(host.approach_combo)

        # Info button styled like action buttons
        host.method_info_button = QPushButton("Info", parent)
        host.method_info_button.setToolTip("Click to view detailed method information")
        host.method_info_button.setStyleSheet(
            f"QPushButton {{ background-color: {Colors.ACCENT_BLUE}; color: white; border: none;"
            f" padding: 5px 12px; border-radius: 0px; font-weight: 600; font-size: 8pt; min-width: 50px; }}"
            f"QPushButton:hover {{ background-color: #0c5591; }}"
        )
        host.method_info_button.clicked.connect(host._show_mode_info_dialog)
        mode_row.addWidget(host.method_info_button)

        block.addLayout(mode_row)
        layout.addLayout(block)

    def _add_species_section(self, layout: QVBoxLayout) -> None:
        host = self.host
        host.species_box = host._build_species_group()
        layout.addWidget(host.species_box)

    def _add_parameters_section(self, layout: QVBoxLayout) -> None:
        host = self.host
        parent = layout.parentWidget()

        layout.addWidget(create_section_header("Parameters", parent))

        conc_label = QLabel("Control concentrations", parent)
        conc_label.setStyleSheet(
            f"font-size: 8pt; font-weight: 600; color: {Colors.TEXT_SECONDARY}; background-color: transparent;"
        )
        layout.addWidget(conc_label)

        conc_grid = QGridLayout()
        conc_grid.setContentsMargins(0, 1, 0, 4)
        conc_grid.setHorizontalSpacing(6)  # Increased from 2 for better label-input proximity
        conc_grid.setVerticalSpacing(2)
        # Remove column stretching to keep labels close to inputs

        def _add_pair(row_idx: int, col_idx: int, text: str, entry: QLineEdit) -> None:
            label = QLabel(text, parent)
            label.setStyleSheet(f"font-size: 8pt; font-weight: 500; color: {Colors.TEXT_SECONDARY}; background-color: transparent;")
            conc_grid.addWidget(label, row_idx, col_idx * 2)
            entry.setFixedWidth(90)
            conc_grid.addWidget(entry, row_idx, col_idx * 2 + 1)

        row = 0
        col = 0
        host.pos_ctrl_concentration_entries.clear()
        if host.pos_ctrl_data is not None and hasattr(host.pos_ctrl_data, "columns"):
            for i, well in enumerate(host._datawell_columns(host.pos_ctrl_data), start=1):
                entry = QLineEdit(parent)
                entry.setPlaceholderText("nM")
                entry.setStyleSheet(UITheme.get_lineedit_style(max_width=55))
                host.pos_ctrl_concentration_entries.append(entry)
                _add_pair(row, col, f"Pos. Ctrl ({well})", entry)
                col += 1
                if col >= 2:
                    col = 0
                    row += 1

        if host.donor_data is not None and (not hasattr(host.donor_data, "empty") or not host.donor_data.empty):
            host.donor_ctrl_concentration_entry = QLineEdit(parent)
            host.donor_ctrl_concentration_entry.setPlaceholderText("nM")
            host.donor_ctrl_concentration_entry.setStyleSheet(UITheme.get_lineedit_style(max_width=55))
            _add_pair(row, col, "Donor ctrl", host.donor_ctrl_concentration_entry)
            col = (col + 1) % 2
            if col == 0:
                row += 1

        if host.acceptor_data is not None and (not hasattr(host.acceptor_data, "empty") or not host.acceptor_data.empty):
            host.acceptor_ctrl_concentration_entry = QLineEdit(parent)
            host.acceptor_ctrl_concentration_entry.setPlaceholderText("nM")
            host.acceptor_ctrl_concentration_entry.setStyleSheet(UITheme.get_lineedit_style(max_width=55))
            _add_pair(row, col, "Acceptor ctrl", host.acceptor_ctrl_concentration_entry)
            if col == 0:
                col = 1
            else:
                col = 0
                row += 1

        # B1L control concentration entry (for runtime calibration slope estimation)
        # This is separate from B1L_0 (experimental starting concentration) because the
        # control may be at a different concentration than the experimental samples
        if host.blocked_ctrl_data is not None and (not hasattr(host.blocked_ctrl_data, "empty") or not host.blocked_ctrl_data.empty):
            host.b1l_ctrl_concentration_entry = QLineEdit(parent)
            host.b1l_ctrl_concentration_entry.setPlaceholderText("nM")
            host.b1l_ctrl_concentration_entry.setStyleSheet(UITheme.get_lineedit_style(max_width=55))
            host.b1l_ctrl_concentration_entry.setToolTip(
                "Concentration of the B1L control sample [nM].\n"
                "Used for runtime slope calibration.\n"
                "This may differ from the experimental B1L starting concentration."
            )
            _add_pair(row, col, "B1L ctrl", host.b1l_ctrl_concentration_entry)
            if col == 0:
                col = 1
            else:
                col = 0
                row += 1

        # Negative control concentration entries (for HP Quenching mode: HI product controls)
        host.neg_ctrl_concentration_entries.clear()
        host.neg_ctrl_concentration_labels = []
        if host.neg_ctrl_data is not None and hasattr(host.neg_ctrl_data, "columns"):
            # Label explaining what "negative control" means in HP Quenching context
            if col != 0:
                col = 0
                row += 1
            host.neg_ctrl_conc_label = QLabel("Neg. Ctrl = quenched HI product", parent)
            host.neg_ctrl_conc_label.setStyleSheet(
                f"font-size: 7pt; font-style: italic; color: {Colors.TEXT_SECONDARY}; background-color: transparent;"
            )
            conc_grid.addWidget(host.neg_ctrl_conc_label, row, 0, 1, 4)
            row += 1

            for i, well in enumerate(host._datawell_columns(host.neg_ctrl_data), start=1):
                entry = QLineEdit(parent)
                entry.setPlaceholderText("nM")
                entry.setStyleSheet(UITheme.get_lineedit_style(max_width=55))
                host.neg_ctrl_concentration_entries.append(entry)
                label = QLabel(f"Neg. Ctrl ({well})", parent)
                label.setStyleSheet(f"font-size: 8pt; font-weight: 500; color: {Colors.TEXT_SECONDARY}; background-color: transparent;")
                host.neg_ctrl_concentration_labels.append(label)
                conc_grid.addWidget(label, row, col * 2)
                entry.setFixedWidth(90)
                conc_grid.addWidget(entry, row, col * 2 + 1)
                col += 1
                if col >= 2:
                    col = 0
                    row += 1

        layout.addLayout(conc_grid)

        # Pos Ctrl Ref - displayed as label with change button (rarely modified)
        # Placed right after control concentrations
        c_ref_row = QHBoxLayout()
        c_ref_row.setSpacing(6)
        c_ref_row.setContentsMargins(0, 4, 0, 8)

        host.c_ref_display_label = QLabel(parent)
        host.c_ref_display_label.setStyleSheet(f"font-size: 8pt; color: {Colors.TEXT_SECONDARY}; font-weight: 500; background-color: transparent;")
        host.c_ref_display_label.setToolTip("Positive control reference concentration for normalisation")
        c_ref_row.addWidget(host.c_ref_display_label)

        host.c_ref_change_button = QPushButton("Change", parent)
        host.c_ref_change_button.setStyleSheet(
            f"QPushButton {{ background-color: transparent; color: {Colors.ACCENT_BLUE}; border: 1px solid {Colors.ACCENT_BLUE};"
            f" padding: 2px 8px; border-radius: 2px; font-size: 7pt; font-weight: 600; }}"
            f"QPushButton:hover {{ background-color: {Colors.ACCENT_BLUE}; color: white; }}"
        )
        host.c_ref_change_button.setFixedHeight(22)
        host.c_ref_change_button.setToolTip("Change the positive control reference concentration (default: 10 nM)")
        c_ref_row.addWidget(host.c_ref_change_button)
        c_ref_row.addStretch()

        layout.addLayout(c_ref_row)

        time_label = QLabel("Timing windows", parent)
        time_label.setStyleSheet(
            f"font-size: 8pt; font-weight: 600; color: {Colors.TEXT_SECONDARY}; background: transparent;"
        )
        layout.addWidget(time_label)

        time_layout = QVBoxLayout()
        time_layout.setContentsMargins(0, 1, 0, 4)
        time_layout.setSpacing(4)

        # Initialisation Window - label on left, fields on right
        init_row = QHBoxLayout()
        init_row.setSpacing(6)
        init_label = QLabel("Initialisation Window [min]", parent)
        init_label.setStyleSheet(f"font-size: 8pt; color: {Colors.TEXT_SECONDARY}; font-weight: 500; background-color: transparent;")
        init_label.setFixedWidth(175)  # Fixed width for alignment with neg ctrl window
        init_row.addWidget(init_label)

        host.init_start_time_entry.setFixedWidth(65)
        host.init_end_time_entry.setFixedWidth(65)
        host.init_start_time_entry.setStyleSheet(UITheme.get_lineedit_style(max_width=70))
        host.init_end_time_entry.setStyleSheet(UITheme.get_lineedit_style(max_width=70))
        host.init_start_time_entry.setToolTip("Start of initialisation window")
        host.init_end_time_entry.setToolTip("End of initialisation window")
        init_row.addWidget(host.init_start_time_entry)
        separator = QLabel("—", parent)
        separator.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 10pt; background-color: transparent;")
        init_row.addWidget(separator)
        init_row.addWidget(host.init_end_time_entry)
        init_row.addStretch()
        time_layout.addLayout(init_row)

        # Negative Control Window - label on left, fields on right
        neg_ctrl_row = QHBoxLayout()
        neg_ctrl_row.setSpacing(6)
        host.neg_ctrl_window_label = QLabel("Neg Ctrl Baseline [min]", parent)
        host.neg_ctrl_window_label.setStyleSheet(f"font-size: 8pt; color: {Colors.TEXT_SECONDARY}; font-weight: 500; background-color: transparent;")
        host.neg_ctrl_window_label.setFixedWidth(175)  # Fixed width for alignment with init window
        host.neg_ctrl_window_label.setToolTip(
            "Baseline window for negative control (typically 30 min).\n"
            "Select AFTER initial transients have subsided."
        )
        neg_ctrl_row.addWidget(host.neg_ctrl_window_label)

        host.neg_ctrl_start_time_entry.setFixedWidth(65)
        host.neg_ctrl_end_time_entry.setFixedWidth(65)
        host.neg_ctrl_start_time_entry.setStyleSheet(UITheme.get_lineedit_style(max_width=70))
        host.neg_ctrl_end_time_entry.setStyleSheet(UITheme.get_lineedit_style(max_width=70))
        neg_ctrl_tooltip = (
            "Baseline window for negative control (typically 30 min).\n\n"
            "Select a window AFTER initial transients from reaction\n"
            "initiation have subsided. The mean negative control signal\n"
            "in this window defines the baseline (RF = 0).\n\n"
            "Example: If reaction starts at t=40 min and transients\n"
            "settle by t=60 min, use window 60-90 min."
        )
        host.neg_ctrl_start_time_entry.setToolTip(neg_ctrl_tooltip)
        host.neg_ctrl_end_time_entry.setToolTip(neg_ctrl_tooltip)
        neg_ctrl_row.addWidget(host.neg_ctrl_start_time_entry)
        host.neg_ctrl_separator = QLabel("—", parent)
        host.neg_ctrl_separator.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 10pt; background-color: transparent;")
        neg_ctrl_row.addWidget(host.neg_ctrl_separator)
        neg_ctrl_row.addWidget(host.neg_ctrl_end_time_entry)
        neg_ctrl_row.addStretch()
        time_layout.addLayout(neg_ctrl_row)

        # Individual time parameters - use compact rows instead of grid
        host.inj1_label = QLabel("Injection 1", parent)
        host.inj1_label.setStyleSheet(UITheme.get_label_style_secondary())
        host.inj2_label = QLabel("Injection 2", parent)
        host.inj2_label.setStyleSheet(UITheme.get_label_style_secondary())
        host.b1l_label = QLabel("B1-L [nM]", parent)
        host.b1l_label.setStyleSheet(UITheme.get_label_style_secondary())

        # Style all entries consistently
        for entry in [host.reaction_trigger_timepoint, host.injection_for_nuking_entry,
                      host.injection_for_nuking_entry_2, host.b1l_initial_conc_entry]:
            entry.setStyleSheet(UITheme.get_lineedit_style(max_width=70))
            entry.setFixedWidth(90)

        # Create grid layout for individual time parameters - matching control concentration layout
        time_params_grid = QGridLayout()
        time_params_grid.setContentsMargins(0, 4, 0, 4)
        time_params_grid.setHorizontalSpacing(6)
        time_params_grid.setVerticalSpacing(2)

        # Helper to add label-input pairs matching the control concentration style
        def _add_time_pair(row_idx: int, col_idx: int, label_widget: QLabel, entry: QLineEdit) -> None:
            label_widget.setStyleSheet(f"font-size: 8pt; font-weight: 500; color: {Colors.TEXT_SECONDARY}; background-color: transparent;")
            time_params_grid.addWidget(label_widget, row_idx, col_idx * 2)
            time_params_grid.addWidget(entry, row_idx, col_idx * 2 + 1)

        # Row 0, Col 0: Trigger
        trigger_label = QLabel("Trigger", parent)
        _add_time_pair(0, 0, trigger_label, host.reaction_trigger_timepoint)

        # Row 0, Col 1: Injection 1 (Nuking time)
        _add_time_pair(0, 1, host.inj1_label, host.injection_for_nuking_entry)

        # Row 1, Col 0: Injection 2
        _add_time_pair(1, 0, host.inj2_label, host.injection_for_nuking_entry_2)

        # Row 1, Col 1: B1-L init
        _add_time_pair(1, 1, host.b1l_label, host.b1l_initial_conc_entry)

        time_layout.addLayout(time_params_grid)

        # Calibration mode section (for mass-action mode)
        host.calibration_mode_label = QLabel("Calibration Mode", parent)
        host.calibration_mode_label.setStyleSheet(UITheme.get_label_style_secondary())
        host.calibration_mode_label.setVisible(False)

        calib_layout = QGridLayout()
        calib_layout.setContentsMargins(0, 4, 0, 4)
        calib_layout.setHorizontalSpacing(6)
        calib_layout.setVerticalSpacing(2)

        # Calibration mode dropdown (spans full width)
        calib_layout.addWidget(host.calibration_mode_label, 0, 0)
        calib_layout.addWidget(host.calibration_mode_combo, 0, 1, 1, 3)

        time_layout.addLayout(calib_layout)

        layout.addLayout(time_layout)

    def _add_action_section(self, layout: QVBoxLayout) -> None:
        host = self.host
        parent = layout.parentWidget()

        # Create a container frame with border for the buttons
        btn_container = QFrame(parent)
        btn_container.setStyleSheet(
            f"QFrame {{ border: 1px solid {Colors.CARD_BORDER}; border-radius: 0px; padding: 2px; background-color: transparent; }}"
        )

        btn_row = QHBoxLayout(btn_container)
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(3)

        host.plot_mode_button = QPushButton("Scatter", btn_container)
        host.plot_mode_button.setCheckable(True)
        host.plot_mode_button.setToolTip("Toggle between scatter and line plot modes")
        host.plot_mode_button.setStyleSheet(
            f"QPushButton {{ background-color: {Colors.ACCENT_BLUE}; color: white; border: none;"
            f" padding: 5px 10px; border-radius: 0px; min-width: 65px; font-weight: 600; font-size: 8pt; }}"
            f"QPushButton:hover {{ background-color: #0c5591; }}"
            f"QPushButton:checked {{ background-color: {Colors.ACCENT_ORANGE}; }}"
            f"QPushButton:checked:hover {{ background-color: #9a4000; }}"
        )
        btn_row.addWidget(host.plot_mode_button)

        host.convert_button = QPushButton("Convert", btn_container)
        host.convert_button.setStyleSheet(
            f"QPushButton {{ background-color: {Colors.ACCENT_GREEN}; color: white; border: none;"
            f" padding: 5px 12px; border-radius: 0px; font-weight: 600; font-size: 8pt; }}"
            f"QPushButton:hover {{ background-color: #0f5c27; }}"
        )
        btn_row.addWidget(host.convert_button)

        host.export_button = QPushButton("Export", btn_container)
        host.export_button.setEnabled(False)
        host.export_button.setStyleSheet(
            f"QPushButton {{ background-color: {Colors.CARD_BACKGROUND}; color: {Colors.TEXT_SECONDARY};"
            f" border: 1px solid {Colors.CARD_BORDER}; padding: 5px 12px; border-radius: 0px;"
            f" font-weight: 600; font-size: 8pt; }}"
            f"QPushButton:hover {{ background-color: {Colors.CONTENT_BACKGROUND}; border-color: {Colors.ACCENT_BLUE}; }}"
            f"QPushButton:disabled {{ background-color: {Colors.SECTION_BACKGROUND}; color: {Colors.TEXT_TERTIARY}; }}"
        )
        btn_row.addWidget(host.export_button)

        btn_row.addStretch()

        layout.addWidget(btn_container)

    def _add_matrix_section(self, layout: QVBoxLayout) -> None:
        host = self.host
        parent = layout.parentWidget()

        # Store reference to section header for visibility control
        host.matrix_header = create_section_header("Coefficients and Bleed-through", parent)
        layout.addWidget(host.matrix_header)

        host.coeff_table = host._create_matrix_table()
        host.bleed_table = host._create_matrix_table()
        host.coeff_table.setMaximumWidth(440)
        host.bleed_table.setMaximumWidth(440)
        host.coeff_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        host.bleed_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

        # Store reference to labels for visibility control
        host.coeff_label = QLabel("Brightness ratios (beta, alpha, gamma)", parent)
        host.coeff_label.setStyleSheet(
            f"font-size: 8pt; color: {Colors.TEXT_SECONDARY}; background-color: transparent; margin-top: 4px;"
        )
        layout.addWidget(host.coeff_label)
        layout.addWidget(host.coeff_table)

        host.slopes_label = QLabel("Calibration slopes (AFU per nM)", parent)
        host.slopes_label.setStyleSheet(
            f"font-size: 8pt; color: {Colors.TEXT_SECONDARY}; background-color: transparent; margin-top: 6px;"
        )
        layout.addWidget(host.slopes_label)
        layout.addWidget(host.bleed_table)

    def _add_forward_section(self, layout: QVBoxLayout) -> None:
        host = self.host
        parent = layout.parentWidget()
        layout.addWidget(create_section_header("Forward to Kinetics Processor", parent))

        dataset_label = QLabel("Select Dataset:", parent)
        dataset_label.setStyleSheet(UITheme.get_label_style_secondary())
        layout.addWidget(dataset_label)

        host.dataset_combo = QComboBox(parent)
        host.dataset_combo.setEnabled(False)
        host.dataset_combo.addItem("No datasets available")
        host.dataset_combo.setStyleSheet(UITheme.get_combobox_style())
        layout.addWidget(host.dataset_combo)

        host.forward_button = QPushButton("Send to Kinetics Processor", parent)
        host.forward_button.setEnabled(False)
        host.forward_button.setStyleSheet(
            f"QPushButton {{ background-color: {Colors.ACCENT_BLUE}; color: white; border: none;"
            f" border-radius: 0px; padding: 6px 10px; font-weight: 600; font-size: 8pt; }}"
            f"QPushButton:hover {{ background-color: #0c5591; }}"
            f"QPushButton:disabled {{ background-color: #C4C9D4; color: #5C6675; }}"
        )
        layout.addWidget(host.forward_button)

        host.forward_info_label = QLabel("Convert data first to enable forwarding", parent)
        host.forward_info_label.setStyleSheet(
            f"color: {Colors.TEXT_TERTIARY}; font-style: italic; font-size: 8pt; background-color: transparent;"
        )
        host.forward_info_label.setWordWrap(True)
        layout.addWidget(host.forward_info_label)

    # ---------------------------------------------------------------- utilities
    @property
    def INPUT_STYLE_CONC(self) -> str:
        return self.host.INPUT_STYLE_CONC  # type: ignore[attr-defined]
