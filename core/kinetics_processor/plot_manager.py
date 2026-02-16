"""
Plot Management Module for Kinetics Processor
----------------------------------------------
Handles interactive plotting and visualisation for kinetics analysis.

Author: Krizan Jurinovic
Date: November 2025
"""

import logging
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout,
                             QPushButton, QFileDialog, QMessageBox, QGroupBox,
                             QLabel, QSpinBox, QDoubleSpinBox, QCheckBox,
                             QTabWidget, QWidget, QScrollArea, QSizePolicy, QSplitter,
                             QFrame, QFormLayout, QLineEdit, QAbstractSpinBox)
from PyQt5.QtGui import QFont, QPainter, QColor, QPen, QFontMetrics, QImage
from PyQt5.QtCore import Qt, QTimer, QRectF
import pyqtgraph as pg

# Import unified UI theme for consistent styling
from core.common.ui_theme import Colours, UITheme
from core.kinetics_processor.gui_widget_factory import block_wheel_event

logger = logging.getLogger(__name__)


def infer_y_axis_label(y_min, y_max):
    """
    Infer appropriate y-axis label based on data value range.

    Uses simple heuristics to determine whether data represents:
    - Relative Fluorescence (R.F.U.): Values typically in range -0.5 to 1.5
    - Concentration (nM): Values typically larger, representing actual concentrations

    Parameters
    ----------
    y_min : float
        Minimum y value in the data.
    y_max : float
        Maximum y value in the data.

    Returns
    -------
    str
        A single label string for the y-axis, e.g. "Concentration (nM)" or "R.F.U.".
    """
    data_range = y_max - y_min

    # Heuristic: if values are roughly in -0.5 to 1.5 range with small span,
    # it is likely relative fluorescence (normalised 0-1 data)
    if y_min >= -1.0 and y_max <= 2.0 and data_range <= 3.0:
        return 'R.F.U.'
    else:
        return 'Concentration (nM)'


class ReplicateLegendWidget(QWidget):
    """
    Read-only legend widget that maps replicate group colours to their
    constituent well traces. Renders via QPainter for crisp on-screen
    display and clean image/SVG export.

    Entries flow horizontally and wrap to the next row when the
    available width is exhausted.
    """

    # Layout constants
    SWATCH_W = 30
    SWATCH_H = 12
    SWATCH_TEXT_GAP = 6
    ENTRY_H_GAP = 18
    ROW_HEIGHT = 22
    VERTICAL_PAD = 6

    def __init__(self, group_colours, replicate_info, trace_settings,
                 parent=None):
        super().__init__(parent)
        self._entries = []

        logger.info(
            "ReplicateLegendWidget: %d group_colours, %d replicate_info entries",
            len(group_colours), len(replicate_info))

        for group_name, colour in group_colours.items():
            info = replicate_info.get(group_name)
            if info is None:
                logger.warning("Legend: group '%s' not found in replicate_info", group_name)
                continue
            member_cols = info['columns']
            selected = [c for c in member_cols
                        if trace_settings.get(c, {}).get('show_trace', True)]
            if not selected:
                continue
            traces_str = ', '.join(selected)
            label = f"{group_name}: {traces_str}"
            self._entries.append((colour, label))
            logger.info("Legend entry: '%s' (%d members)", label, len(selected))

        self._font = QFont("Arial", 10)
        self._fm = QFontMetrics(self._font)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    # -----------------------------------------------------------------
    # Size computation
    # -----------------------------------------------------------------
    def _compute_row_count(self, width):
        """Return number of rows needed to lay out entries at *width*."""
        if not self._entries:
            return 0
        x = 0
        rows = 1
        for _colour, label in self._entries:
            entry_w = (self.SWATCH_W + self.SWATCH_TEXT_GAP
                       + self._fm.horizontalAdvance(label))
            if x > 0 and x + entry_w > width:
                rows += 1
                x = 0
            x += entry_w + self.ENTRY_H_GAP
        return rows

    def _height_for_width(self, width):
        rows = self._compute_row_count(width)
        return rows * self.ROW_HEIGHT + 2 * self.VERTICAL_PAD if rows else 0

    def sizeHint(self):
        from PyQt5.QtCore import QSize
        w = self.width() if self.width() > 0 else 600
        return QSize(w, self._height_for_width(w))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.setFixedHeight(self._height_for_width(event.size().width()))
        self.update()

    # -----------------------------------------------------------------
    # Painting
    # -----------------------------------------------------------------
    def _paint_legend(self, painter, width):
        """Render all legend entries into *painter* at the given *width*."""
        painter.setFont(self._font)
        border_colour = QColor(Colours.BORDER_MEDIUM)
        x = 0
        y = self.VERTICAL_PAD

        for colour_hex, label in self._entries:
            entry_w = (self.SWATCH_W + self.SWATCH_TEXT_GAP
                       + self._fm.horizontalAdvance(label))
            # Wrap to next row if needed
            if x > 0 and x + entry_w > width:
                x = 0
                y += self.ROW_HEIGHT

            # Draw colour swatch
            swatch_y = y + (self.ROW_HEIGHT - self.SWATCH_H) // 2
            painter.setPen(QPen(border_colour, 1))
            painter.setBrush(QColor(colour_hex))
            painter.drawRect(QRectF(x, swatch_y, self.SWATCH_W, self.SWATCH_H))

            # Draw label text
            text_x = x + self.SWATCH_W + self.SWATCH_TEXT_GAP
            text_y = y + (self.ROW_HEIGHT + self._fm.ascent()
                          - self._fm.descent()) // 2
            painter.setPen(QColor('#333333'))
            painter.drawText(int(text_x), int(text_y), label)

            x += entry_w + self.ENTRY_H_GAP

    def paintEvent(self, event):
        if not self._entries:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        self._paint_legend(painter, self.width())
        painter.end()

    # -----------------------------------------------------------------
    # Export helpers
    # -----------------------------------------------------------------
    def render_to_image(self, width=2000, scale=2):
        """Return a QImage of the legend at the given *width* and *scale*."""
        h = self._height_for_width(width)
        img = QImage(int(width * scale), int(h * scale),
                     QImage.Format_ARGB32_Premultiplied)
        img.fill(QColor('#FFFFFF'))
        painter = QPainter(img)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.scale(scale, scale)
        self._paint_legend(painter, width)
        painter.end()
        return img

    def render_to_svg(self, file_path, width=2000):
        """Write an SVG file of the legend at the given *width*."""
        from PyQt5.QtSvg import QSvgGenerator
        from PyQt5.QtCore import QSize as _QSize
        h = self._height_for_width(width)
        generator = QSvgGenerator()
        generator.setFileName(str(file_path))
        generator.setSize(_QSize(int(width), int(h)))
        generator.setViewBox(QRectF(0, 0, width, h))
        painter = QPainter(generator)
        painter.setRenderHint(QPainter.Antialiasing)
        self._paint_legend(painter, width)
        painter.end()


class ReplicateAveragePlotDialogue(QDialog):
    """
    Interactive dialogue for replicate average plots with customisable settings.

    Allows users to configure:
    - Font size for labels and ticks
    - Y-axis (concentration) range (auto or manual)
    - Tick density for both axes

    Note: Display window can now be adjusted independently of the fitting window.
    """

    def __init__(self, parent, averaged_data, averaged_fits, time_col,
                 quality_report, group_colours, has_fits, time_data,
                 report_html, avg_kf, sem_kf, n_kf, t_start, t_end,
                 full_data_df=None, full_fitted_df=None, replicate_info=None,
                 trace_settings=None, catalytic_scale_factor=None,
                 fit_window_start=None, fit_window_end=None,
                 full_time_min=None, full_time_max=None, fitted_parameters=None):
        super().__init__(parent)

        # Store data for plotting
        self.averaged_data = averaged_data
        self.averaged_fits = averaged_fits
        self.time_col = time_col
        self.quality_report = quality_report
        self.group_colours = group_colours
        self.has_fits = has_fits
        self.time_data = time_data
        self.report_html = report_html
        self.avg_kf = avg_kf
        self.sem_kf = sem_kf
        self.n_kf = n_kf

        # Store time range for current display
        self.t_start = t_start
        self.t_end = t_end

        # Store full data for dynamic display window changes
        self.full_data_df = full_data_df
        self.full_fitted_df = full_fitted_df
        self.replicate_info = replicate_info or {}
        self.trace_settings = trace_settings or {}
        self.catalytic_scale_factor = catalytic_scale_factor
        self.fitted_parameters = fitted_parameters or {}

        # Store fitting window (the range used for actual fitting)
        self.fit_window_start = fit_window_start if fit_window_start is not None else t_start
        self.fit_window_end = fit_window_end if fit_window_end is not None else t_end

        # Store full data time range
        self.full_time_min = full_time_min if full_time_min is not None else 0.0
        self.full_time_max = full_time_max if full_time_max is not None else t_end

        # Calculate default data ranges
        self.data_time_min = float(time_data.min())
        self.data_time_max = float(time_data.max())

        # Calculate full data range INCLUDING error bands (min/max values, not just means)
        all_y_values = []
        for group_name in quality_report:
            group = group_name['group']
            # Include both mean and error band extremes
            data_mean_key = f"{group}_data_mean"
            data_min_key = f"{group}_data_min"
            data_max_key = f"{group}_data_max"

            if data_mean_key in averaged_data:
                all_y_values.extend(averaged_data[data_mean_key])
            if data_min_key in averaged_data:
                all_y_values.extend(averaged_data[data_min_key])
            if data_max_key in averaged_data:
                all_y_values.extend(averaged_data[data_max_key])

            if has_fits:
                fit_mean_key = f"{group}_fit_mean"
                if fit_mean_key in averaged_fits:
                    all_y_values.extend(averaged_fits[fit_mean_key])

        self.data_y_min = float(np.nanmin(all_y_values)) if all_y_values else 0
        self.data_y_max = float(np.nanmax(all_y_values)) if all_y_values else 10.8

        # Set up UI
        self.setWindowTitle("Replicate Average Analysis")

        # Window sizing — base values; adjusted after legend creation to
        # ensure the plot gets the same space as the pre-legend layout.
        self._base_min_h = 450
        self._base_resize_h = 550
        self.setMinimumSize(700, self._base_min_h)
        self.resize(900, self._base_resize_h)

        # Base height reference for UI (kept for reference; actual square size is driven by frame size)
        self.base_ui_height = 320

        # Main layout - vertical with content area and button bar
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # =====================================================================
        # HORIZONTAL SPLIT: Plot (left) | Side Panel (right)
        # =====================================================================
        content_splitter = QSplitter(Qt.Horizontal)
        content_splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #E0E0E0;
                width: 3px;
            }
            QSplitter::handle:hover {
                background-color: #1976D2;
            }
        """)

        # -----------------------------------------------------------------
        # LEFT SIDE: Plot in a frame
        # -----------------------------------------------------------------
        self.plot_frame = QFrame()
        self.plot_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {Colours.CARD_BACKGROUND};
                border: 1px solid {Colours.CARD_BORDER};
                border-radius: 3px;
            }}
        """)
        self.plot_frame_layout = QVBoxLayout(self.plot_frame)
        self.plot_frame_layout.setContentsMargins(8, 8, 8, 8)
        self.plot_frame_layout.setSpacing(4)

        # Plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.plotItem.setContentsMargins(15, 20, 15, 15)
        self.plot_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.plot_widget.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Let the layout and our square-resize logic govern its size
        self.plot_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Centre the square plot within the frame
        self.plot_frame_layout.addWidget(self.plot_widget)

        content_splitter.addWidget(self.plot_frame)

        self.content_splitter = content_splitter

        # -----------------------------------------------------------------
        # RIGHT SIDE: Settings and Report tabs in side panel
        # -----------------------------------------------------------------
        side_panel = QFrame()
        side_panel.setStyleSheet(f"""
            QFrame {{
                background-color: {Colours.SECTION_BACKGROUND};
                border: none;
            }}
        """)
        side_panel.setMinimumWidth(240)
        side_panel.setMaximumWidth(320)
        side_panel_layout = QVBoxLayout(side_panel)
        side_panel_layout.setContentsMargins(4, 4, 4, 4)
        side_panel_layout.setSpacing(4)

        # Tabbed interface for settings and report
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet(f"""
            QTabWidget::pane {{
                border: 1px solid {Colours.CARD_BORDER};
                background-color: {Colours.CARD_BACKGROUND};
                border-radius: 0px;
            }}
            QTabBar::tab {{
                background-color: {Colours.MAIN_BACKGROUND};
                border: 1px solid {Colours.CARD_BORDER};
                border-bottom: none;
                padding: 5px 14px;
                margin-right: 1px;
                font-size: 8pt;
                font-weight: 600;
                color: {Colours.TEXT_SECONDARY};
                min-width: 80px;
            }}
            QTabBar::tab:selected {{
                background-color: {Colours.CARD_BACKGROUND};
                border-bottom: 1px solid {Colours.CARD_BACKGROUND};
                color: {Colours.ACCENT_BLUE};
            }}
            QTabBar::tab:hover {{
                background-color: {Colours.SECTION_BACKGROUND};
            }}
        """)

        # Tab 1: Plot Settings (vertical form layout for side panel)
        settings_tab = QWidget()
        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        settings_scroll.setStyleSheet(UITheme.get_scrollarea_style())

        settings_content = self._create_settings_panel_vertical()
        settings_scroll.setWidget(settings_content)

        settings_tab_layout = QVBoxLayout(settings_tab)
        settings_tab_layout.setContentsMargins(0, 0, 0, 0)
        settings_tab_layout.addWidget(settings_scroll)

        self.tab_widget.addTab(settings_tab, "Settings")

        side_panel_layout.addWidget(self.tab_widget)

        content_splitter.addWidget(side_panel)

        # Set initial splitter proportions (plot gets ~70%, side panel ~30%)
        content_splitter.setStretchFactor(0, 7)
        content_splitter.setStretchFactor(1, 3)
        content_splitter.setSizes([600, 260])

        main_layout.addWidget(content_splitter, 1)

        # =====================================================================
        # FOOTER: Legend + Button bar in a single fixed-height frame
        # The legend is placed above the buttons so that it does NOT reduce
        # the content_splitter's available space relative to the original
        # button-bar-only layout. The dialogue is grown by the legend height.
        # =====================================================================
        footer_frame = QFrame()
        footer_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {Colours.SECTION_BACKGROUND};
                border-top: 1px solid {Colours.CARD_BORDER};
                border-radius: 0px;
            }}
        """)
        footer_layout = QVBoxLayout(footer_frame)
        footer_layout.setContentsMargins(0, 0, 0, 0)
        footer_layout.setSpacing(0)

        # Legend row
        self.legend_widget = ReplicateLegendWidget(
            self.group_colours, self.replicate_info, self.trace_settings)
        self.legend_widget.setContentsMargins(10, 4, 10, 0)
        footer_layout.addWidget(self.legend_widget)

        # Button row
        button_bar = QWidget()
        button_bar.setFixedHeight(48)
        button_layout = QHBoxLayout(button_bar)
        button_layout.setContentsMargins(10, 8, 10, 8)
        button_layout.setSpacing(8)

        # Save/Load settings buttons - standard styling
        save_settings_btn = QPushButton("Save Settings")
        save_settings_btn.setStyleSheet(UITheme.get_button_style_standard())
        save_settings_btn.clicked.connect(self._save_settings)
        button_layout.addWidget(save_settings_btn)

        load_settings_btn = QPushButton("Load Settings")
        load_settings_btn.setStyleSheet(UITheme.get_button_style_standard())
        load_settings_btn.clicked.connect(self._load_settings)
        button_layout.addWidget(load_settings_btn)

        export_report_btn = QPushButton("Export Report as HTML")
        export_report_btn.setStyleSheet(UITheme.get_button_style_standard())
        export_report_btn.clicked.connect(self._export_report)
        button_layout.addWidget(export_report_btn)

        button_layout.addStretch()

        # Export buttons - primary styling (blue)
        export_png_btn = QPushButton("Export PNG")
        export_png_btn.setStyleSheet(UITheme.get_button_style_success())
        export_png_btn.clicked.connect(self._export_png)
        button_layout.addWidget(export_png_btn)

        export_svg_btn = QPushButton("Export SVG")
        export_svg_btn.setStyleSheet(UITheme.get_button_style_success())
        export_svg_btn.clicked.connect(self._export_svg)
        button_layout.addWidget(export_svg_btn)

        # Close button - standard styling
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet(UITheme.get_button_style_standard())
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)

        footer_layout.addWidget(button_bar)

        # Compute legend height and grow the dialogue to compensate, so the
        # content_splitter (and thus the plot) gets the same space as before.
        legend_h = self.legend_widget._height_for_width(600)
        footer_frame.setFixedHeight(48 + legend_h + 4)
        main_layout.addWidget(footer_frame)

        # Grow dialogue by legend height so the plot is not shrunk
        self.setMinimumSize(700, self._base_min_h + legend_h + 4)
        self.resize(900, self._base_resize_h + legend_h + 4)

        # Initial square sizing of plot widget (reinforced after show event)

        self._update_plot_widget_square()

        # Auto-load saved settings (if any exist)
        self._auto_load_settings()

        # Generate initial plot
        self._refresh_plot()

        # Ensure layout and plot sizing are correct once the window is visible
        QTimer.singleShot(0, self._apply_initial_layout)

    # ----------------------------------------------------------------------
    # Initial layout configuration
    # ----------------------------------------------------------------------
    def _apply_initial_layout(self):
        """Ensure splitter and plot sizing are correct once the dialogue is visible."""
        self._update_plot_widget_square()

        if hasattr(self, 'content_splitter'):
            total_width = self.content_splitter.size().width() or self.width()
            side_width = max(240, min(320, int(total_width * 0.32)))
            plot_width = max(total_width - side_width, 320)
            self.content_splitter.setSizes([plot_width, side_width])

    # ----------------------------------------------------------------------
    # Aspect-ratio-aware sizing helper and resize hook
    # ----------------------------------------------------------------------
    def _update_plot_widget_square(self):
        """
        Resize the plot widget to fit within the plot frame whilst respecting
        the user-configured aspect ratio (width:height).

        The widget dimensions are calculated to be as large as possible within
        the available frame area, constrained by the aspect ratio setting.
        A ratio of 1.0 produces a square preview; values > 1.0 produce a wider
        preview; values < 1.0 produce a narrower preview.
        """
        if self.plot_widget is None or self.plot_frame is None:
            return

        # Available size inside the frame (subtract margins)
        frame_rect = self.plot_frame.contentsRect()
        available_width = frame_rect.width() - 16
        available_height = frame_rect.height() - 16

        # Do not try to size to zero during initial layout passes
        if available_width <= 0 or available_height <= 0:
            return

        # Retrieve the user-configured aspect ratio (width / height)
        aspect_ratio = getattr(self, 'aspect_ratio_spin', None)
        if aspect_ratio is not None:
            ratio = aspect_ratio.value()
        else:
            ratio = 1.0  # Default to square if spinner not yet initialised

        # Calculate the largest widget size that fits within the frame
        # whilst maintaining the requested aspect ratio.
        # ratio = width / height  =>  width = height * ratio
        height_if_width_constrained = available_width / ratio
        if height_if_width_constrained <= available_height:
            # Width is the limiting dimension
            plot_width = available_width
            plot_height = int(height_if_width_constrained)
        else:
            # Height is the limiting dimension
            plot_height = available_height
            plot_width = int(available_height * ratio)

        # Ensure minimum sensible size
        plot_width = max(plot_width, 100)
        plot_height = max(plot_height, 100)

        self.plot_widget.setFixedSize(plot_width, plot_height)

    def resizeEvent(self, event):
        """
        On any dialogue resize, update the plot widget to maintain the
        configured aspect ratio within the available frame space.
        """
        super().resizeEvent(event)
        self._update_plot_widget_square()

    # ----------------------------------------------------------------------
    # Settings panel definitions
    # ----------------------------------------------------------------------
    def _create_settings_panel_vertical(self):
        """Create vertical settings panel optimised for side panel layout."""
        panel = QWidget()
        panel.setStyleSheet(f"background-color: {Colours.CARD_BACKGROUND};")
        layout = QVBoxLayout(panel)
        layout.setSpacing(12)
        layout.setContentsMargins(8, 8, 8, 8)

        # =================================================================
        # SECTION 1: Export Info
        # =================================================================
        export_info = QLabel("Export: 486×486px @ 400 DPI")
        export_info.setStyleSheet(f"""
            background-color: {Colours.INFO_PANEL_TMSD_BG};
            border: 1px solid {Colours.INFO_PANEL_TMSD_BORDER};
            border-radius: 3px;
            padding: 6px;
            font-size: 8pt;
            color: {Colours.INFO_PANEL_TMSD_TEXT};
        """)
        export_info.setAlignment(Qt.AlignCenter)
        layout.addWidget(export_info)
        self.export_info_label = export_info

        # =================================================================
        # SECTION 2: Font & Aspect Ratio
        # =================================================================
        font_aspect_group = QGroupBox("Appearance")
        font_aspect_group.setStyleSheet(UITheme.get_groupbox_style())
        font_aspect_layout = QFormLayout(font_aspect_group)
        font_aspect_layout.setSpacing(6)
        font_aspect_layout.setContentsMargins(8, 12, 8, 8)

        # Font size
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(12, 48)
        self.font_size_spin.setValue(24)
        self.font_size_spin.setSuffix(" pt")
        self.font_size_spin.setStyleSheet(UITheme.get_spinbox_style())
        self.font_size_spin.setFocusPolicy(Qt.StrongFocus)
        self.font_size_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        block_wheel_event(self.font_size_spin)
        self.font_size_spin.valueChanged.connect(lambda: self._on_setting_changed())
        self.font_size_spin.editingFinished.connect(lambda: self._on_setting_changed())
        font_aspect_layout.addRow("Font Size:", self.font_size_spin)

        # Aspect ratio (affects both preview and export dimensions)
        self.aspect_ratio_spin = QDoubleSpinBox()
        self.aspect_ratio_spin.setRange(0.3, 3.0)
        self.aspect_ratio_spin.setValue(1.0)
        self.aspect_ratio_spin.setDecimals(2)
        self.aspect_ratio_spin.setSingleStep(0.1)
        self.aspect_ratio_spin.setStyleSheet(UITheme.get_spinbox_style(decimals=True))
        self.aspect_ratio_spin.setFocusPolicy(Qt.StrongFocus)
        self.aspect_ratio_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        block_wheel_event(self.aspect_ratio_spin)
        self.aspect_ratio_spin.setToolTip("Width:Height ratio\n1.0 = square\n>1.0 = wider\n<1.0 = narrower")
        self.aspect_ratio_spin.valueChanged.connect(lambda: self._on_setting_changed())
        self.aspect_ratio_spin.editingFinished.connect(lambda: self._on_setting_changed())
        font_aspect_layout.addRow("Aspect Ratio:", self.aspect_ratio_spin)

        # Axis padding (offset between axes and data)
        self.axis_padding_spin = QDoubleSpinBox()
        self.axis_padding_spin.setRange(0.0, 50.0)
        self.axis_padding_spin.setValue(5.0)  # Default: 5%
        self.axis_padding_spin.setDecimals(1)
        self.axis_padding_spin.setSingleStep(0.5)
        self.axis_padding_spin.setSuffix(" %")
        self.axis_padding_spin.setStyleSheet(UITheme.get_spinbox_style(decimals=True))
        self.axis_padding_spin.setFocusPolicy(Qt.StrongFocus)
        self.axis_padding_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        block_wheel_event(self.axis_padding_spin)
        self.axis_padding_spin.setToolTip(
            "Padding between axes and data (as % of range)\n"
            "0% = data touches axes"
        )
        self.axis_padding_spin.valueChanged.connect(lambda: self._on_setting_changed())
        self.axis_padding_spin.editingFinished.connect(lambda: self._on_setting_changed())
        font_aspect_layout.addRow("Axis Padding:", self.axis_padding_spin)

        # Axis line width
        self.axis_width_spin = QDoubleSpinBox()
        self.axis_width_spin.setRange(0.5, 5.0)
        self.axis_width_spin.setValue(3.0)  # Default: 3.0 pt
        self.axis_width_spin.setDecimals(1)
        self.axis_width_spin.setSingleStep(0.5)
        self.axis_width_spin.setSuffix(" pt")
        self.axis_width_spin.setStyleSheet(UITheme.get_spinbox_style(decimals=True))
        self.axis_width_spin.setFocusPolicy(Qt.StrongFocus)
        self.axis_width_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        block_wheel_event(self.axis_width_spin)
        self.axis_width_spin.setToolTip(
            "Width of axis lines and tick marks"
        )
        self.axis_width_spin.valueChanged.connect(lambda: self._on_setting_changed())
        self.axis_width_spin.editingFinished.connect(lambda: self._on_setting_changed())
        font_aspect_layout.addRow("Axis Width:", self.axis_width_spin)

        # Marker (dot) size for data points
        self.marker_size_spin = QSpinBox()
        self.marker_size_spin.setRange(1, 20)
        self.marker_size_spin.setValue(3)  # Default: 3 pt
        self.marker_size_spin.setSuffix(" pt")
        self.marker_size_spin.setStyleSheet(UITheme.get_spinbox_style())
        self.marker_size_spin.setFocusPolicy(Qt.StrongFocus)
        self.marker_size_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        block_wheel_event(self.marker_size_spin)
        self.marker_size_spin.setToolTip(
            "Size of data point markers (export only)\n"
            "Preview always uses small dots (3 pt)\n"
            "3 pt = default\n"
            "Use 5+ for better visibility"
        )
        self.marker_size_spin.valueChanged.connect(lambda: self._on_setting_changed())
        self.marker_size_spin.editingFinished.connect(lambda: self._on_setting_changed())
        font_aspect_layout.addRow("Marker Size:", self.marker_size_spin)

        # Minor ticks toggle
        self.show_minor_ticks_checkbox = QCheckBox("Show Minor Ticks")
        self.show_minor_ticks_checkbox.setStyleSheet(UITheme.get_checkbox_style())
        self.show_minor_ticks_checkbox.setChecked(False)  # Default: off
        self.show_minor_ticks_checkbox.setToolTip(
            "Show minor tick marks between major ticks"
        )
        self.show_minor_ticks_checkbox.stateChanged.connect(lambda: self._on_setting_changed())

        # Create horizontal layout for checkbox and reset button
        minor_ticks_layout = QHBoxLayout()
        minor_ticks_layout.setContentsMargins(0, 0, 0, 0)
        minor_ticks_layout.addWidget(self.show_minor_ticks_checkbox)

        # Reset to defaults button
        reset_btn = QPushButton("Reset Defaults")
        reset_btn.setStyleSheet(UITheme.get_button_style_standard())
        reset_btn.setToolTip(
            "Reset all appearance settings to defaults:\n"
            "• Axis Padding: 5%\n"
            "• Axis Width: 3.0 pt\n"
            "• Minor Ticks: Off"
        )
        reset_btn.clicked.connect(self._reset_to_defaults)
        minor_ticks_layout.addWidget(reset_btn)

        minor_ticks_widget = QWidget()
        minor_ticks_widget.setLayout(minor_ticks_layout)
        font_aspect_layout.addRow(minor_ticks_widget)

        layout.addWidget(font_aspect_group)

        # =================================================================
        # SECTION 3: Time Axis
        # =================================================================
        time_group = QGroupBox("Time Axis (X)")
        time_group.setStyleSheet(UITheme.get_groupbox_style())
        time_layout = QFormLayout(time_group)
        time_layout.setSpacing(6)
        time_layout.setContentsMargins(8, 12, 8, 8)

        # Fitting window info (read-only)
        fit_window_label = QLabel(f"{self.fit_window_start:.1f} - {self.fit_window_end:.1f} min")
        fit_window_label.setStyleSheet(UITheme.get_label_style_tertiary())
        fit_window_label.setToolTip("The time range used for fitting (set in main UI)")
        time_layout.addRow("Fit Window:", fit_window_label)

        # Display window start
        self.display_start_spin = QDoubleSpinBox()
        # Allow display to start before data begins (can show negative time)
        self.display_start_spin.setRange(self.full_time_min - self.full_time_max, 99999)
        self.display_start_spin.setValue(self.fit_window_start)
        self.display_start_spin.setDecimals(1)
        self.display_start_spin.setSuffix(" min")
        self.display_start_spin.setSingleStep(1.0)
        self.display_start_spin.setStyleSheet(UITheme.get_spinbox_style(decimals=True))
        self.display_start_spin.setToolTip(
            "Start of displayed time range\n"
            "Can extend before the fitting window"
        )
        self.display_start_spin.valueChanged.connect(self._on_display_window_changed)
        self.display_start_spin.editingFinished.connect(self._on_display_window_changed)
        time_layout.addRow("Display Start:", self.display_start_spin)

        # Display window end
        self.display_end_spin = QDoubleSpinBox()
        # Allow display to extend well beyond available data
        self.display_end_spin.setRange(self.full_time_min, 99999)
        self.display_end_spin.setValue(self.fit_window_end)
        self.display_end_spin.setDecimals(1)
        self.display_end_spin.setSuffix(" min")
        self.display_end_spin.setSingleStep(1.0)
        self.display_end_spin.setStyleSheet(UITheme.get_spinbox_style(decimals=True))
        self.display_end_spin.setToolTip(
            "End of displayed time range\n"
            "Can extend beyond available data\n"
            "(missing values shown as blank)"
        )
        self.display_end_spin.valueChanged.connect(self._on_display_window_changed)
        self.display_end_spin.editingFinished.connect(self._on_display_window_changed)
        time_layout.addRow("Display End:", self.display_end_spin)

        # Buttons for quick selection
        display_btn_layout = QHBoxLayout()
        display_btn_layout.setSpacing(4)

        reset_fit_btn = QPushButton("Fit Window")
        reset_fit_btn.setStyleSheet(UITheme.get_button_style_standard())
        reset_fit_btn.setToolTip("Reset display to the fitting window")
        reset_fit_btn.clicked.connect(self._reset_to_fit_window)
        display_btn_layout.addWidget(reset_fit_btn)

        display_btn_widget = QWidget()
        display_btn_widget.setLayout(display_btn_layout)
        time_layout.addRow(display_btn_widget)

        # Time Zero offset - allows user to set which original timepoint displays as 0
        self.time_zero_spin = QDoubleSpinBox()
        self.time_zero_spin.setRange(self.full_time_min, self.full_time_max)
        self.time_zero_spin.setValue(self.fit_window_start)
        self.time_zero_spin.setDecimals(1)
        self.time_zero_spin.setSuffix(" min")
        self.time_zero_spin.setSingleStep(1.0)
        self.time_zero_spin.setStyleSheet(UITheme.get_spinbox_style(decimals=True))
        self.time_zero_spin.setToolTip(
            "Set which original timepoint should display as t=0\n"
            "e.g., if reaction starts at 80 min, enter 80 to show that as 0"
        )
        self.time_zero_spin.valueChanged.connect(self._on_time_zero_changed)
        self.time_zero_spin.editingFinished.connect(self._on_time_zero_changed)
        time_layout.addRow("Time Zero:", self.time_zero_spin)

        # X tick density
        self.x_tick_density_spin = QSpinBox()
        self.x_tick_density_spin.setRange(2, 10)
        self.x_tick_density_spin.setValue(4)
        self.x_tick_density_spin.setStyleSheet(UITheme.get_spinbox_style())
        self.x_tick_density_spin.setFocusPolicy(Qt.StrongFocus)
        self.x_tick_density_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        block_wheel_event(self.x_tick_density_spin)
        self.x_tick_density_spin.valueChanged.connect(lambda: self._on_setting_changed())
        self.x_tick_density_spin.editingFinished.connect(lambda: self._on_setting_changed())
        time_layout.addRow("Tick Count:", self.x_tick_density_spin)

        layout.addWidget(time_group)

        # =================================================================
        # SECTION 4: Y Axis
        # =================================================================
        conc_group = QGroupBox("Y Axis")
        conc_group.setStyleSheet(UITheme.get_groupbox_style())
        conc_layout = QFormLayout(conc_group)
        conc_layout.setSpacing(6)
        conc_layout.setContentsMargins(8, 12, 8, 8)

        # Auto range checkbox
        self.y_auto_checkbox = QCheckBox("Auto Range")
        self.y_auto_checkbox.setStyleSheet(UITheme.get_checkbox_style())
        self.y_auto_checkbox.setChecked(False)
        self.y_auto_checkbox.stateChanged.connect(self._on_y_auto_changed)
        conc_layout.addRow(self.y_auto_checkbox)

        # Normalised values checkbox (divide by 10)
        self.normalise_checkbox = QCheckBox("Normalised Values (\u00f710)")
        self.normalise_checkbox.setStyleSheet(UITheme.get_checkbox_style())
        self.normalise_checkbox.setChecked(False)
        self.normalise_checkbox.setToolTip(
            "Scale Y-axis values by dividing by 10\n"
            "Useful for displaying normalised concentrations"
        )
        self.normalise_checkbox.stateChanged.connect(self._on_normalise_changed)
        conc_layout.addRow(self.normalise_checkbox)

        # Y-axis label (auto-detected or user-customised)
        inferred_label = infer_y_axis_label(self.data_y_min, self.data_y_max)
        self.y_axis_label_edit = QLineEdit()
        self.y_axis_label_edit.setText(inferred_label)
        self.y_axis_label_edit.setStyleSheet(UITheme.get_lineedit_style())
        self.y_axis_label_edit.setToolTip(
            "Y-axis label (auto-detected based on data range)\n"
            "Edit to customise the label for your plot"
        )
        self.y_axis_label_edit.editingFinished.connect(lambda: self._on_setting_changed())
        conc_layout.addRow("Label:", self.y_axis_label_edit)
        # Min value
        self.y_min_spin = QDoubleSpinBox()
        self.y_min_spin.setRange(-100000, 100000)
        self.y_min_spin.setValue(self.data_y_min)
        self.y_min_spin.setDecimals(1)
        self.y_min_spin.setStyleSheet(UITheme.get_spinbox_style(decimals=True))
        self.y_min_spin.setFocusPolicy(Qt.StrongFocus)
        self.y_min_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        block_wheel_event(self.y_min_spin)
        self.y_min_spin.valueChanged.connect(lambda: self._on_setting_changed())
        self.y_min_spin.editingFinished.connect(lambda: self._on_setting_changed())
        conc_layout.addRow("Min:", self.y_min_spin)

        # Max value
        self.y_max_spin = QDoubleSpinBox()
        self.y_max_spin.setRange(-100000, 100000)
        self.y_max_spin.setValue(10.8)
        self.y_max_spin.setDecimals(1)
        self.y_max_spin.setStyleSheet(UITheme.get_spinbox_style(decimals=True))
        self.y_max_spin.setFocusPolicy(Qt.StrongFocus)
        self.y_max_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        block_wheel_event(self.y_max_spin)
        self.y_max_spin.valueChanged.connect(lambda: self._on_setting_changed())
        self.y_max_spin.editingFinished.connect(lambda: self._on_setting_changed())
        conc_layout.addRow("Max:", self.y_max_spin)

        # Y tick density
        self.y_tick_density_spin = QSpinBox()
        self.y_tick_density_spin.setRange(2, 10)
        self.y_tick_density_spin.setValue(5)
        self.y_tick_density_spin.setStyleSheet(UITheme.get_spinbox_style())
        self.y_tick_density_spin.setFocusPolicy(Qt.StrongFocus)
        self.y_tick_density_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        block_wheel_event(self.y_tick_density_spin)
        self.y_tick_density_spin.valueChanged.connect(lambda: self._on_setting_changed())
        self.y_tick_density_spin.editingFinished.connect(lambda: self._on_setting_changed())
        conc_layout.addRow("Tick Count:", self.y_tick_density_spin)

        layout.addWidget(conc_group)

        # Add stretch at bottom
        layout.addStretch()

        return panel

    def _on_y_auto_changed(self, state):
        """Handle Y-axis auto checkbox state change."""
        is_auto = state == Qt.Checked
        self.y_min_spin.setEnabled(not is_auto)
        self.y_max_spin.setEnabled(not is_auto)
        if is_auto:
            scale = 0.1 if self.normalise_checkbox.isChecked() else 1.0
            self.y_min_spin.setValue(self.data_y_min * scale)
            self.y_max_spin.setValue(self.data_y_max * scale)
        self._on_setting_changed()

    def _on_normalise_changed(self, state):
        """Handle normalise checkbox state change — rescale Y-axis limits."""
        if state == Qt.Checked:
            self.y_min_spin.setValue(self.y_min_spin.value() / 10.0)
            self.y_max_spin.setValue(self.y_max_spin.value() / 10.0)
        else:
            self.y_min_spin.setValue(self.y_min_spin.value() * 10.0)
            self.y_max_spin.setValue(self.y_max_spin.value() * 10.0)
        self._on_setting_changed()

    def _on_setting_changed(self):
        """
        Handle any setting change - refresh plot and auto-save settings.

        This method is called whenever a plot setting is changed by the user.
        It updates the plot immediately and saves settings to disk automatically,
        ensuring settings are preserved across sessions without manual save.
        """
        self._refresh_plot()
        self._auto_save_settings()

    def _reset_to_defaults(self):
        """
        Reset appearance settings to default values.

        Default values:
        - Axis Padding: 5%
        - Axis Width: 3.0 pt
        - Minor Ticks: Off
        """
        # Block signals to prevent multiple refreshes during reset
        self.axis_padding_spin.blockSignals(True)
        self.axis_width_spin.blockSignals(True)
        self.show_minor_ticks_checkbox.blockSignals(True)

        # Apply default values
        self.axis_padding_spin.setValue(5.0)
        self.axis_width_spin.setValue(3.0)
        self.show_minor_ticks_checkbox.setChecked(False)

        # Unblock signals
        self.axis_padding_spin.blockSignals(False)
        self.axis_width_spin.blockSignals(False)
        self.show_minor_ticks_checkbox.blockSignals(False)

        # Refresh plot and save settings once
        self._on_setting_changed()

        logger.info("Reset appearance settings to defaults")

    def _on_display_window_changed(self):
        """Handle changes to the display window spinboxes."""
        # Ensure start < end
        start = self.display_start_spin.value()
        end = self.display_end_spin.value()

        if start >= end:
            # Silently adjust - don't refresh yet
            if self.sender() == self.display_start_spin:
                self.display_end_spin.blockSignals(True)
                self.display_end_spin.setValue(start + 1.0)
                self.display_end_spin.blockSignals(False)
            else:
                self.display_start_spin.blockSignals(True)
                self.display_start_spin.setValue(end - 1.0)
                self.display_start_spin.blockSignals(False)

        # Recalculate averaged data for new display window
        self._recalculate_for_display_window()

    def _reset_to_fit_window(self):
        """Reset display window to match the fitting window."""
        self.display_start_spin.blockSignals(True)
        self.display_end_spin.blockSignals(True)
        self.time_zero_spin.blockSignals(True)

        self.display_start_spin.setValue(self.fit_window_start)
        self.display_end_spin.setValue(self.fit_window_end)
        self.time_zero_spin.setValue(self.fit_window_start)

        self.display_start_spin.blockSignals(False)
        self.display_end_spin.blockSignals(False)
        self.time_zero_spin.blockSignals(False)

        self._recalculate_for_display_window()
        logger.info("Reset display window to fit window: %.1f - %.1f min",
                   self.fit_window_start, self.fit_window_end)

    def _on_time_zero_changed(self):
        """Handle changes to the time zero offset spinbox."""
        self._recalculate_for_display_window()
        time_zero = self.time_zero_spin.value()
        logger.info("Time zero set to %.1f min (will display as t=0)", time_zero)

    def _recalculate_for_display_window(self):
        """
        Recalculate averaged data for the current display window.

        This filters the full dataset to the display window and recalculates
        averages, then refreshes the plot. The time zero offset determines
        which original timepoint displays as t=0.
        """
        if self.full_data_df is None:
            # No full data available - just refresh with current data
            self._on_setting_changed()
            return

        display_start = self.display_start_spin.value()
        display_end = self.display_end_spin.value()
        time_zero = self.time_zero_spin.value()

        # Filter to display window
        mask = (self.full_data_df[self.time_col] >= display_start) & \
               (self.full_data_df[self.time_col] <= display_end)
        filtered_df = self.full_data_df[mask].copy()

        if filtered_df.empty:
            logger.warning("No data in selected display window")
            return

        # Normalise time using the time zero offset (not display_start)
        # This allows the user to set which original timepoint displays as t=0
        filtered_df[self.time_col] = filtered_df[self.time_col] - time_zero

        # Update display time range (relative to time zero)
        self.t_start = display_start - time_zero
        self.t_end = display_end - time_zero

        # Recalculate averaged data
        self.averaged_data = {self.time_col: filtered_df[self.time_col].values}
        self.time_data = filtered_df[self.time_col].values

        for group_name in self.group_colours.keys():
            if group_name not in self.replicate_info:
                continue

            member_cols = self.replicate_info[group_name]['columns']
            selected_cols = [col for col in member_cols
                            if self.trace_settings.get(col, {}).get('show_trace', True)]

            if not selected_cols:
                continue

            # Calculate average data signal
            data_values = []
            for col in selected_cols:
                if col in filtered_df.columns:
                    col_data = filtered_df[col].values.copy()
                    if self.catalytic_scale_factor is not None:
                        col_data = col_data * self.catalytic_scale_factor
                    data_values.append(col_data)

            if len(data_values) > 0:
                data_array = np.array(data_values)
                self.averaged_data[f"{group_name}_data_mean"] = np.nanmean(data_array, axis=0)
                self.averaged_data[f"{group_name}_data_min"] = np.nanmin(data_array, axis=0)
                self.averaged_data[f"{group_name}_data_max"] = np.nanmax(data_array, axis=0)

        # Recalculate fitted data if available
        if self.has_fits and self.full_fitted_df is not None:
            mask_fit = (self.full_fitted_df[self.time_col] >= display_start) & \
                       (self.full_fitted_df[self.time_col] <= display_end)
            filtered_fitted = self.full_fitted_df[mask_fit].copy()

            if not filtered_fitted.empty:
                # Use time_zero offset for normalisation (same as data)
                filtered_fitted[self.time_col] = filtered_fitted[self.time_col] - time_zero
                self.averaged_fits = {self.time_col: filtered_fitted[self.time_col].values}

                for group_name in self.group_colours.keys():
                    if group_name not in self.replicate_info:
                        continue

                    member_cols = self.replicate_info[group_name]['columns']
                    selected_cols = [col for col in member_cols
                                    if self.trace_settings.get(col, {}).get('show_trace', True)]

                    fit_values = []
                    for col in selected_cols:
                        fit_col = f"{col}_fitted"
                        if fit_col in filtered_fitted.columns:
                            fit_values.append(filtered_fitted[fit_col].values)

                    if len(fit_values) > 0:
                        fit_array = np.array(fit_values)
                        self.averaged_fits[f"{group_name}_fit_mean"] = np.nanmean(fit_array, axis=0)
                        self.averaged_fits[f"{group_name}_fit_min"] = np.nanmin(fit_array, axis=0)
                        self.averaged_fits[f"{group_name}_fit_max"] = np.nanmax(fit_array, axis=0)

        # Refresh the plot
        self._on_setting_changed()

    def _auto_save_settings(self):
        """
        Automatically save current settings to file without user notification.

        Any change is saved immediately and restored when the dialogue is reopened.
        """
        try:
            settings = self._get_current_settings()
            settings_file = self._get_settings_file_path()

            with open(settings_file, 'w') as f:
                json.dump(settings, f, indent=2)

            logger.debug("Auto-saved plot settings to %s", settings_file)
        except Exception as e:
            # Silently log errors - don't interrupt user workflow for auto-save failures
            logger.warning("Failed to auto-save settings: %s", e)

    def _auto_load_settings(self):
        """
        Automatically load saved settings on dialogue initialisation.

        If no saved settings exist, uses defaults. This method is called
        during dialogue setup to restore the user's previous configuration.
        """
        try:
            settings_file = self._get_settings_file_path()

            if not settings_file.exists():
                logger.debug("No saved settings found, using defaults")
                return

            with open(settings_file, 'r') as f:
                settings = json.load(f)

            # Ensure all expected keys exist with sensible defaults
            settings.setdefault('font_size', 24)
            settings.setdefault('aspect_ratio', 1.0)
            settings.setdefault('y_min', self.data_y_min)
            settings.setdefault('y_max', 10.8)
            settings.setdefault('y_auto', False)
            settings.setdefault('x_tick_density', 4)
            settings.setdefault('y_tick_density', 5)
            settings.setdefault('axis_padding', 5.0)  # Default: 5%
            settings.setdefault('axis_width', 3.0)  # Default: 3.0 pt
            settings.setdefault('show_minor_ticks', False)  # Default: off
            settings.setdefault('marker_size', 3)  # Default: 3 pt
            settings.setdefault('normalise_divide_10', False)  # Default: off

            # Y-axis label (auto-detected based on data range)
            default_label = infer_y_axis_label(self.data_y_min, self.data_y_max)
            settings.setdefault('y_axis_label', default_label)

            # Apply settings without triggering refresh (we'll do that after)
            self._apply_settings(settings, skip_refresh=True)

            logger.info("Auto-loaded plot settings from %s", settings_file)
        except Exception as e:
            logger.warning("Failed to auto-load settings: %s", e)

    def _get_current_settings(self):
        """Get current plot settings."""
        return {
            'font_size': self.font_size_spin.value(),
            'aspect_ratio': self.aspect_ratio_spin.value(),
            'y_min': self.y_min_spin.value(),
            'y_max': self.y_max_spin.value(),
            'y_auto': self.y_auto_checkbox.isChecked(),
            'y_axis_label': self.y_axis_label_edit.text(),
            'x_tick_density': self.x_tick_density_spin.value(),
            'y_tick_density': self.y_tick_density_spin.value(),
            'axis_padding': self.axis_padding_spin.value(),
            'axis_width': self.axis_width_spin.value(),
            'show_minor_ticks': self.show_minor_ticks_checkbox.isChecked(),
            'marker_size': self.marker_size_spin.value(),
            'normalise_divide_10': self.normalise_checkbox.isChecked()
        }

    def _apply_settings(self, settings, skip_refresh=False):
        """
        Apply settings to UI controls.

        Parameters
        ----------
        settings : dict
            Settings dictionary with keys: font_size, aspect_ratio, y_min, y_max, y_auto,
            y_axis_label, x_tick_density, y_tick_density, axis_padding,
            axis_width, show_minor_ticks, marker_size
        skip_refresh : bool
            If True, skip the plot refresh (used during initialisation)
        """
        # Block signals to prevent multiple refreshes
        self.font_size_spin.blockSignals(True)
        self.aspect_ratio_spin.blockSignals(True)
        self.y_min_spin.blockSignals(True)
        self.y_max_spin.blockSignals(True)
        self.y_auto_checkbox.blockSignals(True)
        self.y_axis_label_edit.blockSignals(True)
        self.x_tick_density_spin.blockSignals(True)
        self.y_tick_density_spin.blockSignals(True)
        self.axis_padding_spin.blockSignals(True)
        self.axis_width_spin.blockSignals(True)
        self.show_minor_ticks_checkbox.blockSignals(True)
        self.marker_size_spin.blockSignals(True)
        self.normalise_checkbox.blockSignals(True)

        # Apply settings
        self.font_size_spin.setValue(settings.get('font_size', 24))
        self.aspect_ratio_spin.setValue(settings.get('aspect_ratio', 1.0))
        self.y_min_spin.setValue(settings.get('y_min', self.data_y_min))
        self.y_max_spin.setValue(settings.get('y_max', 10.8))
        self.x_tick_density_spin.setValue(settings.get('x_tick_density', 4))
        self.y_tick_density_spin.setValue(settings.get('y_tick_density', 5))
        self.axis_padding_spin.setValue(settings.get('axis_padding', 5.0))
        self.axis_width_spin.setValue(settings.get('axis_width', 3.0))
        self.show_minor_ticks_checkbox.setChecked(settings.get('show_minor_ticks', False))
        self.marker_size_spin.setValue(settings.get('marker_size', 3))
        self.normalise_checkbox.setChecked(settings.get('normalise_divide_10', False))

        # Apply y-axis label (use auto-detected default if not in settings)
        default_label = infer_y_axis_label(self.data_y_min, self.data_y_max)
        self.y_axis_label_edit.setText(settings.get('y_axis_label', default_label))

        # Apply auto checkbox and enable/disable controls
        y_auto = settings.get('y_auto', False)  # Default to manual mode
        self.y_auto_checkbox.setChecked(y_auto)
        self.y_min_spin.setEnabled(not y_auto)
        self.y_max_spin.setEnabled(not y_auto)

        # Unblock signals
        self.font_size_spin.blockSignals(False)
        self.aspect_ratio_spin.blockSignals(False)
        self.y_min_spin.blockSignals(False)
        self.y_max_spin.blockSignals(False)
        self.y_auto_checkbox.blockSignals(False)
        self.y_axis_label_edit.blockSignals(False)
        self.x_tick_density_spin.blockSignals(False)
        self.y_tick_density_spin.blockSignals(False)
        self.axis_padding_spin.blockSignals(False)
        self.axis_width_spin.blockSignals(False)
        self.show_minor_ticks_checkbox.blockSignals(False)
        self.marker_size_spin.blockSignals(False)
        self.normalise_checkbox.blockSignals(False)

        # Refresh plot once with all settings applied
        if not skip_refresh:
            self._refresh_plot()

    def _get_settings_file_path(self):
        """Get the path to the settings file."""
        # Store in config directory
        config_dir = Path.home() / ".clearissa" / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir / "replicate_plot_settings.json"

    def _save_settings(self):
        """Save current plot settings to file."""
        try:
            settings = self._get_current_settings()
            settings_file = self._get_settings_file_path()

            with open(settings_file, 'w') as f:
                json.dump(settings, f, indent=2)

            logger.info("Plot settings saved to %s", settings_file)
            QMessageBox.information(
                self,
                "Settings Saved",
                f"Plot settings saved successfully!\n\n"
                f"File: {settings_file}\n\n"
                f"Font size: {settings['font_size']}pt\n"
                f"Aspect ratio: {settings['aspect_ratio']:.2f}\n"
                f"Axis padding: {settings['axis_padding']:.1f}%\n"
                f"Axis width: {settings['axis_width']:.1f}pt\n"
                f"Marker size: {settings['marker_size']}pt\n"
                f"Minor ticks: {'On' if settings['show_minor_ticks'] else 'Off'}\n"
                f"Y-axis: {settings.get('y_axis_label', 'Concentration (nM)')}\n"
                f"Y range: {settings['y_min']:.1f}-{settings['y_max']:.1f} (Auto: {settings['y_auto']})\n"
                f"Time axis ticks: {settings['x_tick_density']}\n"
                f"Y-axis ticks: {settings['y_tick_density']}"
            )
        except Exception as e:
            logger.error("Failed to save settings: %s", e, exc_info=True)
            QMessageBox.critical(
                self,
                "Save Error",
                f"Failed to save settings:\n{str(e)}"
            )

    def _load_settings(self):
        """Load plot settings from file."""
        try:
            settings_file = self._get_settings_file_path()

            if not settings_file.exists():
                QMessageBox.information(
                    self,
                    "No Settings Found",
                    f"No saved settings found.\n\n"
                    f"Expected location: {settings_file}\n\n"
                    f"Use 'Save Settings' to create a settings file."
                )
                return

            with open(settings_file, 'r') as f:
                settings = json.load(f)

            # Validate required keys
            required_keys = ['font_size', 'y_min', 'y_max', 'y_auto']
            if not all(key in settings for key in required_keys):
                raise ValueError("Settings file is missing required keys")

            # Ensure all expected keys exist with sensible defaults
            settings.setdefault('aspect_ratio', 1.0)
            settings.setdefault('x_tick_density', 4)
            settings.setdefault('y_tick_density', 5)
            settings.setdefault('axis_padding', 5.0)
            settings.setdefault('axis_width', 3.0)
            settings.setdefault('show_minor_ticks', False)
            settings.setdefault('marker_size', 3)

            # Y-axis label (auto-detected based on data range)
            default_label = infer_y_axis_label(self.data_y_min, self.data_y_max)
            settings.setdefault('y_axis_label', default_label)

            # Apply settings
            self._apply_settings(settings)

            logger.info("Plot settings loaded from %s", settings_file)
            QMessageBox.information(
                self,
                "Settings Loaded",
                f"Plot settings loaded successfully!\n\n"
                f"Font size: {settings['font_size']}pt\n"
                f"Aspect ratio: {settings['aspect_ratio']:.2f}\n"
                f"Axis padding: {settings['axis_padding']:.1f}%\n"
                f"Axis width: {settings['axis_width']:.1f}pt\n"
                f"Marker size: {settings['marker_size']}pt\n"
                f"Minor ticks: {'On' if settings['show_minor_ticks'] else 'Off'}\n"
                f"Y-axis: {settings.get('y_axis_label', 'Concentration (nM)')}\n"
                f"Y range: {settings['y_min']:.1f}-{settings['y_max']:.1f} (Auto: {settings['y_auto']})\n"
                f"Time axis ticks: {settings['x_tick_density']}\n"
                f"Y-axis ticks: {settings['y_tick_density']}"
            )
        except Exception as e:
            logger.error("Failed to load settings: %s", e, exc_info=True)
            QMessageBox.critical(
                self,
                "Load Error",
                f"Failed to load settings:\n{str(e)}"
            )

    def _refresh_plot(self, for_svg_export=False, for_export=False):
        """
        Regenerate the plot with current settings.

        Uses configurable width for axes and ticks. For SVG export, halves pen
        widths to compensate for pyqtgraph's SVGExporter which doubles all pen
        widths internally. All axes and ticks are rendered in black for
        clean appearance.

        The on-screen preview respects the user's aspect ratio setting via
        _update_plot_widget_square(). The same aspect ratio is used for
        PNG and SVG exports.

        Parameters
        ----------
        for_svg_export : bool, optional
            If True, halves pen widths to compensate for SVGExporter doubling.
            Default is False.
        for_export : bool, optional
            If True, uses user-configured marker size for export.
            If False, uses fixed small marker size (3) for preview.
            Default is False.
        """
        settings = self._get_current_settings()

        # Normalisation scale factor (÷10 when checkbox is ticked)
        norm_scale = 0.1 if settings.get('normalise_divide_10', False) else 1.0

        # IMPORTANT: only enforce square sizing for on-screen preview.
        # For exports (PNG/SVG), the exporter uses the underlying scene, and
        # forcing a resize here on the first call could leave the scene at an
        # outdated geometry, causing a different scaling on first export.
        if not for_svg_export:
            self._update_plot_widget_square()

        # Update export info label
        export_height = 2000
        export_width = int(export_height * settings['aspect_ratio'])
        self.export_info_label.setText(f"Export: {export_width}x{export_height}px @ 400 DPI")

        # Clear existing plot
        self.plot_widget.clear()
        plot = self.plot_widget.getPlotItem()

        # Do not lock data aspect ratio; the widget is square so preview looks square anyway
        plot.vb.setAspectLocked(False)

        # CRITICAL: Disable auto-range to prevent PyQtGraph from resetting axis limits
        # when data is added. This ensures custom y-axis ranges (including negative
        # values or non-zero baselines) are preserved in both preview and export.
        plot.vb.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=False)

        # Get user-specified axis width (SVGExporter doubles pen widths, so halve for SVG)
        user_axis_width = settings.get('axis_width', 1.0)
        if for_svg_export:
            axis_width = user_axis_width / 2.0
            tick_width = user_axis_width / 2.0
        else:
            axis_width = user_axis_width
            tick_width = user_axis_width
        tick_length = 4

        # Get axis references
        axis_left = plot.getAxis('left')
        axis_bottom = plot.getAxis('bottom')

        # Configure text properties first (these do not affect pen width)
        axis_left.setTextPen('k')
        axis_bottom.setTextPen('k')
        axis_left.enableAutoSIPrefix(False)
        axis_bottom.enableAutoSIPrefix(False)

        # Set labels with configured font size
        font_size_str = f"{settings['font_size']}pt"
        font = {'color': 'k', 'font-size': font_size_str, 'font-weight': 'bold', 'font-family': 'Arial'}
        y_label = settings.get('y_axis_label', 'Concentration (nM)')
        plot.setLabel('left', y_label, **font)
        plot.setLabel('bottom', 'Time (min)', **font)

        # Configure tick font
        tick_font = QFont("Arial", settings['font_size'])
        tick_font.setBold(True)

        # Calculate tick intervals using user-specified tick density
        x_interval, x_tick_positions = calculate_intelligent_tick_interval(
            self.t_start, self.t_end, max_ticks=settings['x_tick_density']
        )
        y_interval, y_tick_positions = calculate_intelligent_tick_interval(
            settings['y_min'], settings['y_max'], max_ticks=settings['y_tick_density']
        )

        # Generate minor ticks (only if show_minor_ticks is enabled)
        show_minor_ticks = settings.get('show_minor_ticks', True)
        x_minor_ticks = []
        y_minor_ticks = []

        if show_minor_ticks:
            for i in range(len(x_tick_positions) - 1):
                mid_point = x_tick_positions[i] + x_interval / 2
                if mid_point < x_tick_positions[-1]:
                    x_minor_ticks.append(mid_point)

            for i in range(len(y_tick_positions) - 1):
                mid_point = y_tick_positions[i] + y_interval / 2
                if mid_point < y_tick_positions[-1]:
                    y_minor_ticks.append(mid_point)

        # Format tick labels
        def format_tick(value):
            if value == int(value):
                return str(int(value))
            else:
                return f"{value:.1f}"

        # Create pens FIRST - ensures consistent axis appearance regardless of tick configuration
        axis_pen = pg.mkPen(color=(0, 0, 0, 255), width=axis_width)
        tick_pen = pg.mkPen(color=(0, 0, 0, 255), width=tick_width)

        # Apply pens BEFORE setting ticks - this ensures all ticks (major and minor) use the same pen
        axis_bottom.setPen(axis_pen)
        axis_bottom.setTickPen(tick_pen, which='major')
        axis_bottom.setTickPen(tick_pen, which='minor')

        axis_left.setPen(axis_pen)
        axis_left.setTickPen(tick_pen, which='major')
        axis_left.setTickPen(tick_pen, which='minor')

        # Configure bottom axis ticks and style
        if show_minor_ticks:
            axis_bottom.setTicks([
                [(pos, format_tick(pos)) for pos in x_tick_positions],
                [(pos, '') for pos in x_minor_ticks]
            ])
        else:
            # Only major ticks, no minor ticks
            axis_bottom.setTicks([
                [(pos, format_tick(pos)) for pos in x_tick_positions]
            ])
        axis_bottom.setStyle(tickLength=-tick_length, tickFont=tick_font, tickTextOffset=10)

        # Configure left axis ticks and style
        if show_minor_ticks:
            axis_left.setTicks([
                [(pos, format_tick(pos)) for pos in y_tick_positions],
                [(pos, '') for pos in y_minor_ticks]
            ])
        else:
            # Only major ticks, no minor ticks
            axis_left.setTicks([
                [(pos, format_tick(pos)) for pos in y_tick_positions]
            ])
        axis_left.setStyle(tickLength=-tick_length, tickFont=tick_font, tickTextOffset=10)

        # Re-apply pens AFTER style configuration to ensure they aren't overwritten
        axis_bottom.setPen(axis_pen)
        axis_bottom.setTickPen(tick_pen, which='major')
        axis_bottom.setTickPen(tick_pen, which='minor')

        axis_left.setPen(axis_pen)
        axis_left.setTickPen(tick_pen, which='major')
        axis_left.setTickPen(tick_pen, which='minor')

        # Plot data for each group
        def hex_to_rgb(hex_colour):
            """Convert hex colour string to RGB tuple."""
            hex_colour = hex_colour.lstrip('#')
            return tuple(int(hex_colour[i:i+2], 16) for i in (0, 2, 4))

        def find_gap_indices(time_arr, threshold_factor=3.0):
            """
            Find indices where there are large gaps in time data.

            A gap is defined as a time difference greater than threshold_factor
            times the median time step.

            Returns list of indices where gaps START (i.e., the last index before each gap).
            """
            if len(time_arr) < 2:
                return []

            diffs = np.diff(time_arr)
            median_step = np.median(diffs)

            if median_step <= 0:
                return []

            # Find where gaps exceed threshold
            gap_threshold = median_step * threshold_factor
            gap_indices = np.where(diffs > gap_threshold)[0]

            return gap_indices.tolist()

        for group_name in self.quality_report:
            group = group_name['group']
            colour = self.group_colours.get(group, (100, 100, 100))

            # Plot data mean with range band
            data_mean_key = f"{group}_data_mean"
            data_min_key = f"{group}_data_min"
            data_max_key = f"{group}_data_max"

            if data_mean_key in self.averaged_data:
                mean = self.averaged_data[data_mean_key] * norm_scale
                data_min = self.averaged_data.get(data_min_key,
                    self.averaged_data[data_mean_key]) * norm_scale
                data_max = self.averaged_data.get(data_max_key,
                    self.averaged_data[data_mean_key]) * norm_scale

                # Plot mean as dots
                # Use fixed small size (3) for preview, configurable size only for export
                if for_export:
                    marker_size = settings.get('marker_size', 3)
                else:
                    marker_size = 3  # Fixed small size for on-screen preview
                plot.plot(self.time_data, mean, pen=None,
                          symbol='o', symbolSize=marker_size, symbolBrush=colour,
                          symbolPen=None,
                          name=f"{group} (data)")

                # Add shaded range band - split at gaps to avoid connecting across missing data
                if isinstance(colour, str):
                    rgb = hex_to_rgb(colour)
                else:
                    rgb = colour

                # Find gaps in time data
                gap_indices = find_gap_indices(self.time_data)

                if not gap_indices:
                    # No gaps - draw single continuous band
                    error_item = pg.FillBetweenItem(
                        pg.PlotCurveItem(self.time_data, data_max),
                        pg.PlotCurveItem(self.time_data, data_min),
                        brush=pg.mkBrush((*rgb, 100))
                    )
                    plot.addItem(error_item)
                else:
                    # Draw separate bands for each continuous segment
                    segment_starts = [0] + [i + 1 for i in gap_indices]
                    segment_ends = [i + 1 for i in gap_indices] + [len(self.time_data)]

                    for start, end in zip(segment_starts, segment_ends):
                        if end - start < 2:
                            continue  # Need at least 2 points for a band

                        seg_time = self.time_data[start:end]
                        seg_min = data_min[start:end]
                        seg_max = data_max[start:end]

                        error_item = pg.FillBetweenItem(
                            pg.PlotCurveItem(seg_time, seg_max),
                            pg.PlotCurveItem(seg_time, seg_min),
                            brush=pg.mkBrush((*rgb, 100))
                        )
                        plot.addItem(error_item)

            # Plot fit mean as clean solid line (if available)
            if self.has_fits:
                fit_mean_key = f"{group}_fit_mean"

                if fit_mean_key in self.averaged_fits:
                    fit_time = self.averaged_fits[self.time_col]
                    fit_mean = self.averaged_fits[fit_mean_key] * norm_scale

                    # Plot mean as solid black line
                    plot.plot(fit_time, fit_mean,
                              pen=pg.mkPen(color='k', width=1.5),
                              name=f"{group} (fit)")

        # CRITICAL: Apply axis ranges AFTER all data is plotted to ensure custom
        # ranges are preserved. Setting ranges before plotting can be overridden
        # by PyQtGraph's internal auto-range behaviour. This fixes the issue where
        # SVG exports would reset to auto-calculated limits instead of respecting
        # user-specified y-axis ranges (e.g., negative values or non-zero baselines).
        #
        # Axis padding creates visual offset between axes and data (like PyQtGraph's
        # "View All" feature accessed via the "A" button). Convert percentage to decimal.
        axis_padding = settings.get('axis_padding', 2.0) / 100.0

        # Calculate padded ranges for display (extends beyond data range)
        x_range = self.t_end - self.t_start
        y_range = settings['y_max'] - settings['y_min']

        x_pad_amount = x_range * axis_padding
        y_pad_amount = y_range * axis_padding

        padded_x_min = self.t_start - x_pad_amount
        padded_x_max = self.t_end + x_pad_amount
        padded_y_min = settings['y_min'] - y_pad_amount
        padded_y_max = settings['y_max'] + y_pad_amount

        # Apply padded ranges to the plot
        plot.setXRange(padded_x_min, padded_x_max, padding=0)
        plot.setYRange(padded_y_min, padded_y_max, padding=0)

        # Set explicit view limits to include the padded area for consistent export.
        # This ensures the ViewBox respects the padded bounds during export operations.
        plot.vb.setLimits(
            xMin=padded_x_min, xMax=padded_x_max,
            yMin=padded_y_min, yMax=padded_y_max
        )

    def _export_png(self):
        """Export plot as PNG without legend (adjustable aspect ratio, 400 DPI)."""
        from pyqtgraph.exporters import ImageExporter
        from PIL import Image
        import tempfile
        import os

        file_dialog = QFileDialog(self)
        file_path, _ = file_dialog.getSaveFileName(
            self,
            "Export PNG",
            "",
            "PNG Image (*.png)"
        )

        if not file_path:
            return

        try:
            # Refresh plot for export (uses user-configured marker size)
            self._refresh_plot(for_export=True)

            # Ensure correct extension
            if not file_path.lower().endswith('.png'):
                file_path += '.png'

            # Calculate dimensions based on aspect ratio (height = 2000px, width = ratio * height)
            settings = self._get_current_settings()
            aspect_ratio = settings['aspect_ratio']
            target_height = 2000
            target_width = int(target_height * aspect_ratio)

            # Export to temporary file first
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                temp_path = tmp_file.name

            try:
                # Export at very high resolution
                exporter = ImageExporter(self.plot_widget.plotItem)
                export_params = exporter.parameters()
                export_params['width'] = target_width * 2  # Export at 2x target for quality
                export_params['height'] = target_height * 2
                exporter.export(temp_path)

                # Now resize to exact dimensions using PIL
                img = Image.open(temp_path)
                img_resized = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
                img_resized.save(file_path, dpi=(400, 400))

            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

            # Calculate physical dimensions in inches at 400 DPI
            physical_width = target_width / 400
            physical_height = target_height / 400

            logger.info("PNG exported to %s - exact dimensions %dx%d pixels at 400 DPI",
                        file_path, target_width, target_height)

            # Export legend as separate PNG
            legend_path = str(Path(file_path).with_stem(
                Path(file_path).stem + '_legend'))
            legend_img = self.legend_widget.render_to_image(
                width=target_width, scale=2)
            legend_img.save(legend_path, "PNG")
            logger.info("Legend PNG exported to %s", legend_path)

            QMessageBox.information(
                self,
                "PNG Export Successful",
                f"PNG exported to:\n{file_path}\n"
                f"Legend exported to:\n{legend_path}\n"
            )

            # Refresh plot back to preview mode (smaller markers)
            self._refresh_plot(for_export=False)
        except Exception as e:
            logger.error("PNG export failed - %s", e, exc_info=True)
            QMessageBox.critical(self, "Export Error", f"Failed to export PNG:\n{str(e)}")

    def _export_svg(self):
        """Export plot as SVG with legend positioned outside on the right for Illustrator editing."""
        from pyqtgraph.exporters import SVGExporter
        from PyQt5.QtWidgets import QApplication

        file_dialog = QFileDialog(self)
        file_path, _ = file_dialog.getSaveFileName(
            self,
            "Export SVG",
            "",
            "SVG Vector (*.svg)"
        )

        if not file_path:
            return

        try:
            # Ensure the on-screen preview is up-to-date before export.
            # First, refresh for on-screen styling (including square sizing),
            # then reapply styling optimised for SVG (thinner pens) without
            # forcing another geometry change. This two-step refresh prevents
            # the first export from using a stale scene size while keeping
            # consistent visual appearance between preview and export.
            self._refresh_plot(for_svg_export=False, for_export=True)

            # Process all pending Qt events to ensure widget geometry is fully updated
            # before exporting. This prevents the first export from using stale scene dimensions.
            QApplication.processEvents()

            self._refresh_plot(for_svg_export=True, for_export=True)

            # Ensure correct extension
            if not file_path.lower().endswith('.svg'):
                file_path += '.svg'

            # Export as SVG (vector format)

            settings = self._get_current_settings()
            aspect_ratio = settings['aspect_ratio']

            target_height = 486
            target_width = int(target_height * aspect_ratio)

            exporter = SVGExporter(self.plot_widget.plotItem)
            export_params = exporter.parameters()
            export_params['width'] = target_width
            export_params['height'] = target_height
            exporter.export(file_path)

            logger.info("SVG exported to %s (aspect ratio: %.2f)", file_path, aspect_ratio)

            # Export legend as separate SVG
            legend_path = str(Path(file_path).with_stem(
                Path(file_path).stem + '_legend'))
            self.legend_widget.render_to_svg(legend_path, width=target_width)
            logger.info("Legend SVG exported to %s", legend_path)

            QMessageBox.information(
                self,
                "SVG Export Successful",
                f"SVG exported to:\n{file_path}\n"
                f"Legend exported to:\n{legend_path}\n"
            )

            # Refresh plot back to preview mode (smaller markers)
            self._refresh_plot(for_export=False)
        except Exception as e:
            logger.error("SVG export failed - %s", e, exc_info=True)
            QMessageBox.critical(self, "Export Error", f"Failed to export SVG:\n{str(e)}")

    def _export_report(self):
        """Export the quality report as standalone HTML file."""
        file_dialog = QFileDialog(self)
        file_path, _ = file_dialog.getSaveFileName(
            self,
            "Export Report as HTML",
            "",
            "HTML Files (*.html)"
        )

        if not file_path:
            return

        try:
            # Ensure correct extension
            if not file_path.lower().endswith('.html'):
                file_path += '.html'

            # Get the report HTML content
            report_html = self.report_html

            # Write to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(report_html)

            logger.info("Report exported to %s", file_path)

            QMessageBox.information(
                self,
                "Report Exported",
                f"Report exported to:\n{file_path}"
            )
        except Exception as e:
            logger.error("Report export failed - %s", e, exc_info=True)
            QMessageBox.critical(self, "Export Error", f"Failed to export report:\n{str(e)}")


def calculate_intelligent_tick_interval(data_min, data_max, max_ticks=4):
    """
    Calculate a round-number tick interval for the given data range.

    Selects from nice intervals (1, 2, 5, 10, 20, 50, etc.) and keeps the
    total number of ticks at or below *max_ticks*.

    Parameters
    ----------
    data_min : float
        Minimum value in the data range
    data_max : float
        Maximum value in the data range
    max_ticks : int
        Maximum number of major ticks to display (default: 4)

    Returns
    -------
    tuple of (float, list)
        (tick_interval, major_tick_positions)
        - tick_interval: The spacing between major ticks
        - major_tick_positions: List of tick positions
    """
    # Calculate data range
    data_range = data_max - data_min

    # Handle edge case: zero or very small range
    if data_range <= 0:
        return 1.0, [data_min, data_min + 1]

    # Calculate rough tick interval to get approximately max_ticks ticks
    rough_interval = data_range / max_ticks

    # Find the magnitude (power of 10) of the rough interval
    magnitude = 10 ** np.floor(np.log10(rough_interval))

    # Normalise the rough interval to get a value between 1 and 10
    normalised = rough_interval / magnitude

    # Round to nice numbers: 1, 2, 5, or 10
    if normalised <= 1.5:
        nice_interval = 1
    elif normalised <= 3:
        nice_interval = 2
    elif normalised <= 7:
        nice_interval = 5
    else:
        nice_interval = 10

    # Calculate actual tick interval
    tick_interval = nice_interval * magnitude

    # Generate tick positions
    tick_start = np.floor(data_min / tick_interval) * tick_interval
    tick_end = np.ceil(data_max / tick_interval) * tick_interval

    num_ticks = int((tick_end - tick_start) / tick_interval) + 1
    major_tick_positions = [tick_start + i * tick_interval for i in range(num_ticks)]

    # Ensure we do not have too many ticks (safety check)
    while len(major_tick_positions) > max_ticks + 2:
        tick_interval *= 2
        tick_start = np.floor(data_min / tick_interval) * tick_interval
        tick_end = np.ceil(data_max / tick_interval) * tick_interval
        num_ticks = int((tick_end - tick_start) / tick_interval) + 1
        major_tick_positions = [tick_start + i * tick_interval for i in range(num_ticks)]

    return tick_interval, major_tick_positions


class PlotManager:
    """
    Manages plotting operations for kinetics analysis.

    Responsibilities:
    - Generate replicate average plots
    - Handle plot export
    - Create high-resolution visualisations
    - Manage pyqtgraph plot configurations
    """

    def __init__(self, parent_widget):
        """
        Initialise the plot manager.

        Parameters
        ----------
        parent_widget : QWidget
            Parent widget for dialogues
        """
        self.parent = parent_widget

    def generate_replicate_average_plot(self, data_df, fitted_df, time_col,
                                        replicate_info, trace_settings,
                                        fitted_parameters, default_colours,
                                        report_generator,
                                        t_start=0, t_end=60,
                                        catalytic_scale_factor=None,
                                        r2_threshold=0.10):
        """
        Generate a specialised plot showing averaged replicate data and fits.

        This function:
        1. Identifies replicate groups from the data
        2. Only includes SELECTED traces (show_trace=True) in averaging
        3. Calculates average data signal across selected replicates
        4. Calculates average fitted curves across selected replicates (if fits exist)
        5. Creates a plot with black axes and fitting lines
        6. Displays average forward rate constant (if fits exist)
        7. Provides export functionality

        Parameters
        ----------
        catalytic_scale_factor : float or None
            If provided (for Catalytic model), the experimental data is in normalised
            fluorescence (0-1) and must be scaled by this factor to convert to nM.
            The fitted curves are already in nM, so scaling the data ensures both
            are on the same scale for proper averaging and display.
        """

        def hex_to_rgb(hex_colour):
            """Convert hex colour string to RGB tuple."""
            hex_colour = hex_colour.lstrip('#')
            return tuple(int(hex_colour[i:i+2], 16) for i in (0, 2, 4))

        logger.info("Generating replicate average plot")

        # Validation checks
        if data_df is None or data_df.empty:
            QMessageBox.warning(self.parent, "No Data", "Please load data first.")
            return

        if not replicate_info:
            QMessageBox.information(
                self.parent,
                "No Replicates Detected",
                "No replicate groups were found in the data.\n\n"
                "Replicates are detected by grouping columns with the same trailing numbers "
                "(e.g., A01, B01, C01 are replicates in column 01)."
            )
            return

        # Check if fits are available (optional now)
        has_fits = fitted_df is not None and not fitted_df.empty
        if not has_fits:
            logger.info("No fits available, will plot averaged data only")

        # Store full data for dynamic display window changes in dialogue
        full_data_df = data_df.copy()
        full_fitted_df = fitted_df.copy() if has_fits else None

        # Store full time range before filtering
        full_time_min = float(data_df[time_col].min())
        full_time_max = float(data_df[time_col].max())

        # Store the original fitting window
        fit_window_start = t_start
        fit_window_end = t_end

        # Copy data for processing
        data_df = data_df.copy()
        if has_fits:
            fitted_df = fitted_df.copy()

        # Filter to time window
        mask_data = (data_df[time_col] >= t_start) & (data_df[time_col] <= t_end)
        data_df = data_df[mask_data]
        if has_fits:
            mask_fitted = (fitted_df[time_col] >= t_start) & (fitted_df[time_col] <= t_end)
            fitted_df = fitted_df[mask_fitted]

        if data_df.empty:
            QMessageBox.warning(self.parent, "No Data", "No data in selected time window.")
            return

        # Normalise time to start from zero (consistent with fitting and export)
        data_df = data_df.copy()
        data_df[time_col] = data_df[time_col] - t_start
        if has_fits:
            fitted_df = fitted_df.copy()
            fitted_df[time_col] = fitted_df[time_col] - t_start

        # Update time range for plot (now starts at 0)
        t_end = t_end - t_start
        t_start = 0.0

        logger.info("Normalised time for replicate plot: time now starts at 0, ends at %.2f min", t_end)

        # Log if catalytic scaling is active
        if catalytic_scale_factor is not None:
            logger.info("Catalytic mode: scaling experimental data by %.2f to convert to nM",
                        catalytic_scale_factor)

        # Build averaged data - only for selected traces
        averaged_data = {time_col: data_df[time_col].values}
        averaged_fits = {time_col: fitted_df[time_col].values} if has_fits else {}
        quality_report = []
        group_colours = {}  # Store consistent colours for each group
        all_kf_values = []  # Collect all forward rate constants from selected traces

        logger.info("Fitted parameters available: %d traces", len(fitted_parameters))
        if fitted_parameters:
            logger.debug("Fitted parameter keys (sample): %s", list(fitted_parameters.keys())[:10])

        # Use group order as defined by user (dict preserves insertion order)
        # This allows users to control colour assignment via the replicate manager
        ordered_group_names = list(replicate_info.keys())
        group_to_colour_idx = {name: idx for idx, name in enumerate(ordered_group_names)}

        logger.info("Replicate groups received (%d groups):", len(replicate_info))
        for gn, gi in replicate_info.items():
            logger.info("  '%s': %d members -> %s", gn, len(gi['columns']), gi['columns'])

        for group_name, group_info in replicate_info.items():
            member_cols = group_info['columns']

            # Filter to only selected traces (show_trace=True)
            selected_cols = [col for col in member_cols
                             if trace_settings.get(col, {}).get('show_trace', True)]

            if not selected_cols:
                logger.info("Skipping group %s as no traces selected", group_name)
                continue

            # Assign consistent colour based on sorted group position
            group_idx = group_to_colour_idx[group_name]
            group_colours[group_name] = default_colours[group_idx % len(default_colours)]

            # Calculate average data signal (only from selected traces)
            # For Catalytic model, scale normalised fluorescence (0-1) to nM
            data_values = []
            for col in selected_cols:
                if col in data_df.columns:
                    col_data = data_df[col].values.copy()
                    # Apply catalytic scaling if in Catalytic mode
                    if catalytic_scale_factor is not None:
                        col_data = col_data * catalytic_scale_factor
                    data_values.append(col_data)

            if len(data_values) > 0:
                data_array = np.array(data_values)
                averaged_data[f"{group_name}_data_mean"] = np.nanmean(data_array, axis=0)
                averaged_data[f"{group_name}_data_min"] = np.nanmin(data_array, axis=0)
                averaged_data[f"{group_name}_data_max"] = np.nanmax(data_array, axis=0)

            # Calculate parameter-averaged fit curves and collect rate constants (only if fits exist)
            # Instead of averaging fit curves directly, we average the fitted parameters
            # and generate a representative fit using those mean values.
            fit_r2_values = []

            # Bimolecular parameters
            group_kf_values = []
            group_X0_values = []
            group_Z0_values = []

            # Catalytic parameters
            group_k_values = []
            group_K_values = []
            group_S10_values = []
            group_T_values = []
            group_cat_X0_values = []
            is_catalytic_model = False

            if has_fits:
                for col in selected_cols:
                    fit_col = f"{col}_fitted"
                    if fit_col in fitted_df.columns:
                        # Calculate R-squared for this trace
                        if col in data_df.columns:
                            y_data = data_df[col].values.copy()
                            # Apply catalytic scaling to data for proper R-squared calculation
                            if catalytic_scale_factor is not None:
                                y_data = y_data * catalytic_scale_factor
                            y_fit = fitted_df[fit_col].values

                            valid = np.isfinite(y_data) & np.isfinite(y_fit)
                            if valid.sum() > 0:
                                y_data_valid = y_data[valid]
                                y_fit_valid = y_fit[valid]

                                ss_tot = np.sum((y_data_valid - np.mean(y_data_valid)) ** 2)
                                ss_res = np.sum((y_data_valid - y_fit_valid) ** 2)
                                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
                                fit_r2_values.append(r2)

                        # Collect fitted parameters for parameter averaging
                        if col in fitted_parameters:
                            params = fitted_parameters[col]
                            model_type = params.get('model', 'bimolecular')

                            if model_type.startswith('Catalytic'):
                                is_catalytic_model = True
                                k = params.get('k_per_min')
                                K = params.get('K_nM')
                                S10 = params.get('S10_nM')
                                T = params.get('T_nM')
                                X0 = params.get('X0_nM')

                                if k is not None:
                                    group_k_values.append(k)
                                if K is not None:
                                    group_K_values.append(K)
                                if S10 is not None:
                                    group_S10_values.append(S10)
                                if T is not None:
                                    group_T_values.append(T)
                                if X0 is not None:
                                    group_cat_X0_values.append(X0)

                                logger.debug("Retrieved catalytic params for trace %s: k=%.8f, K=%.2f, S10=%.2f",
                                            col, k or 0, K or 0, S10 or 0)
                            else:
                                # Bimolecular model
                                kf = params.get('k_f')
                                X0 = params.get('X0_nM')
                                Z0 = params.get('Z0_nM')

                                if kf is not None:
                                    group_kf_values.append(kf)
                                    all_kf_values.append(kf)
                                    logger.debug("Retrieved k_f=%.3e for trace %s", kf, col)

                                if X0 is not None:
                                    group_X0_values.append(X0)
                                if Z0 is not None:
                                    group_Z0_values.append(Z0)
                        else:
                            logger.debug("Trace %s not found in fitted_parameters (available: %d)",
                                         col, len(fitted_parameters))

                # Generate parameter-averaged fit curve
                if is_catalytic_model and group_k_values and group_S10_values:
                    # Catalytic model: use global k (and K if full model) with average S10, X0
                    from core.kinetics_processor.kinetic_models import CatalyticModel

                    mean_k = group_k_values[0]
                    mean_S10 = np.mean(group_S10_values)
                    mean_X0 = np.mean(group_cat_X0_values) if group_cat_X0_values else 0.0
                    mean_T = group_T_values[0] if group_T_values else 1.0

                    model = CatalyticModel()
                    fit_time = fitted_df[time_col].values
                    t_from_zero = fit_time - fit_time[0]

                    if group_K_values:
                        # Full model with saturation
                        mean_K = group_K_values[0]
                        logger.info(
                            "Group %s catalytic parameter-averaged fit: k=%.8f min-1, K=%.2f nM, "
                            "mean_S10=%.2f nM, T=%.2f nM (from %d traces)",
                            group_name, mean_k, mean_K, mean_S10, mean_T, len(group_S10_values)
                        )
                        param_averaged_fit = model.simulate(t_from_zero, mean_k, mean_K, mean_S10, mean_T, mean_X0)
                    else:
                        # Simple model without saturation
                        logger.info(
                            "Group %s catalytic parameter-averaged fit (simple): k=%.8f nM-1 min-1, "
                            "mean_S10=%.2f nM, T=%.2f nM (from %d traces)",
                            group_name, mean_k, mean_S10, mean_T, len(group_S10_values)
                        )
                        param_averaged_fit = model.simulate_simple(t_from_zero, mean_k, mean_S10, mean_T, mean_X0)

                    averaged_fits[f"{group_name}_fit_mean"] = param_averaged_fit
                    averaged_fits[f"{group_name}_fit_min"] = param_averaged_fit
                    averaged_fits[f"{group_name}_fit_max"] = param_averaged_fit

                elif group_kf_values and group_X0_values and group_Z0_values:
                    # Bimolecular model
                    from core.kinetics_processor.kinetic_models import BimolecularModel

                    mean_kf = np.mean(group_kf_values)
                    mean_X0 = np.mean(group_X0_values)
                    mean_Z0 = np.mean(group_Z0_values)

                    logger.info(
                        "Group %s parameter-averaged fit: mean_kf=%.3e M-1 s-1, "
                        "mean_X0=%.2f nM, mean_Z0=%.2f nM (from %d traces)",
                        group_name, mean_kf, mean_X0, mean_Z0, len(group_kf_values)
                    )

                    # Generate representative fit curve from averaged parameters.
                    # Shift time to start from zero so the model begins at Y(0)=0,
                    # consistent with individual trace fitting.
                    model = BimolecularModel()
                    fit_time = fitted_df[time_col].values
                    t_from_zero = fit_time - fit_time[0]
                    t_sec = t_from_zero * 60.0
                    param_averaged_fit = model.simulate(t_sec, mean_kf, mean_X0, mean_Z0)

                    averaged_fits[f"{group_name}_fit_mean"] = param_averaged_fit
                    averaged_fits[f"{group_name}_fit_min"] = param_averaged_fit
                    averaged_fits[f"{group_name}_fit_max"] = param_averaged_fit
                else:
                    if is_catalytic_model:
                        logger.warning(
                            "Group %s: insufficient catalytic parameters for parameter-averaged fit "
                            "(k: %d, S10: %d)",
                            group_name, len(group_k_values), len(group_S10_values)
                        )
                    else:
                        logger.warning(
                            "Group %s: insufficient parameters for parameter-averaged fit "
                            "(kf: %d, X0: %d, Z0: %d)",
                            group_name, len(group_kf_values), len(group_X0_values), len(group_Z0_values)
                        )

            # Quality metrics (always include group if it has selected data)
            if len(data_values) > 0:
                n_replicates = len(selected_cols)
                n_fitted = len(group_kf_values) if not is_catalytic_model else len(group_S10_values)
                mean_r2 = np.mean(fit_r2_values) if fit_r2_values else None
                mean_kf = np.mean(group_kf_values) if group_kf_values else None
                std_kf = np.std(group_kf_values, ddof=1) if len(group_kf_values) > 1 else 0.0
                sem_kf = std_kf / np.sqrt(len(group_kf_values)) if len(group_kf_values) > 1 else 0.0

                quality_report.append({
                    'group': group_name,
                    'n_replicates': n_replicates,
                    'n_fitted': n_fitted,
                    'mean_r2': mean_r2,
                    'r2_values': fit_r2_values,
                    'mean_kf': mean_kf,
                    'std_kf': std_kf,
                    'sem_kf': sem_kf,
                    'kf_values': group_kf_values,
                    'is_catalytic': is_catalytic_model,
                    'catalytic_k': group_k_values[0] if group_k_values else None,
                    'catalytic_K': group_K_values[0] if group_K_values else None,
                })

        if not quality_report:
            QMessageBox.warning(
                self.parent,
                "No Selected Traces",
                "No selected traces found. Please select traces in the fitting interface."
            )
            return

        # Determine if this is a catalytic model fit
        is_catalytic = any(q.get('is_catalytic', False) for q in quality_report)

        # Calculate overall average forward rate constant (only for bimolecular fits)
        avg_kf = np.mean(all_kf_values) if all_kf_values else None
        std_kf = np.std(all_kf_values, ddof=1) if len(all_kf_values) > 1 else 0.0
        sem_kf = std_kf / np.sqrt(len(all_kf_values)) if len(all_kf_values) > 1 else 0.0

        # Generate quality report HTML
        report_html = report_generator.generate_quality_report_html(
            quality_report,
            has_fits=has_fits,
            r2_threshold=r2_threshold
        )

        # Add overall summary based on model type
        if has_fits:
            if is_catalytic:
                # Get catalytic global parameters from first group with data
                catalytic_k = None
                catalytic_K = None
                for q in quality_report:
                    if q.get('catalytic_k') is not None:
                        catalytic_k = q['catalytic_k']
                        catalytic_K = q.get('catalytic_K')
                        break

                if catalytic_k is not None:
                    cat_info = (
                        '<div class="summary-box">'
                        '<h3>Global Catalytic Parameters</h3>'
                        f'<p><strong>k = {catalytic_k:.4f} min<sup>-1</sup></strong> (rate constant)</p>'
                    )
                    if catalytic_K is not None:
                        cat_info += f'<p><strong>K = {catalytic_K:.2f} nM</strong></p>'
                    cat_info += (
                        '<p style="font-size: 9pt; color: #666;">'
                        '(Shared across all fitted traces)'
                        '</p></div>'
                    )
                    report_html = report_html.replace("</body>", cat_info + "</body>")
            elif avg_kf is not None:
                kf_info = f"""
                <div class="summary-box">
                    <h3>Overall Average Forward Rate Constant</h3>
                    <p><strong>k<sub>f</sub> = {avg_kf:.3e} +/- {sem_kf:.3e} M<sup>-1</sup> s<sup>-1</sup></strong></p>
                    <p style="font-size: 9pt; color: #666;">
                        (Mean +/- SEM from {len(all_kf_values)} fitted traces across all selected replicates)
                    </p>
                </div>
                """
                report_html = report_html.replace("</body>", kf_info + "</body>")

        # Extract time data for dialogue
        time_data = averaged_data[time_col]

        # Create and show customisable dialogue
        dialogue = ReplicateAveragePlotDialogue(
            self.parent,
            averaged_data,
            averaged_fits,
            time_col,
            quality_report,
            group_colours,
            has_fits,
            time_data,
            report_html,
            avg_kf,
            sem_kf,
            len(all_kf_values),
            t_start,
            t_end,
            full_data_df=full_data_df,
            full_fitted_df=full_fitted_df,
            replicate_info=replicate_info,
            trace_settings=trace_settings,
            catalytic_scale_factor=catalytic_scale_factor,
            fit_window_start=fit_window_start,
            fit_window_end=fit_window_end,
            full_time_min=full_time_min,
            full_time_max=full_time_max,
            fitted_parameters=fitted_parameters
        )
        dialogue.exec_()
        logger.info("Replicate average plot dialogue closed")
