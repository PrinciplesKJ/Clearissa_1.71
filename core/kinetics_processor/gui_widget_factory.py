"""
GUI Widget Factory for Kinetics Processor
------------------------------------------
Provides shared UI utilities for the kinetics processor widgets.

Author: Krizan Jurinovic
Date: November 2025
"""

from PyQt5.QtCore import QObject, QEvent


class _WheelBlockFilter(QObject):
    """Event filter that blocks mouse wheel events, passing them to the parent."""

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Wheel:
            event.ignore()
            return True
        return super().eventFilter(obj, event)


def block_wheel_event(widget):
    """
    Install an event filter that prevents mouse wheel from changing widget values.

    The ignored wheel event propagates to the parent (e.g. scroll area) so
    page scrolling still works normally.
    """
    guard = _WheelBlockFilter(widget)
    widget.installEventFilter(guard)


class PerTraceTable:
    """
    Reusable 4-column per-trace value table with QDoubleSpinBox inputs.

    Manages a QTableWidget with a paired layout (Trace, Value, Trace, Value)
    and provides populate/get/set methods for the per-trace values.

    Parameters
    ----------
    table : QTableWidget
        A pre-configured 4-column QTableWidget.
    min_val : float
        Minimum spinbox value. Use -1.0 to enable "(default)" special text.
    max_val : float
        Maximum spinbox value.
    decimals : int
        Number of decimal places.
    step : float
        Spinbox single-step increment.
    tooltip_template : str
        Format string for per-trace tooltips, with {name} placeholder.
    use_default_marker : bool
        If True, -1 triggers special "(default)" text and get_values()
        returns None for those entries. If False, all values returned as-is.
    """

    def __init__(self, table, min_val=-1.0, max_val=10000.0, decimals=2,
                 step=0.5, tooltip_template="{name}", use_default_marker=True):
        self._table = table
        self._min_val = min_val
        self._max_val = max_val
        self._decimals = decimals
        self._step = step
        self._tooltip_template = tooltip_template
        self._use_default_marker = use_default_marker
        self._spinboxes = {}
        self._last_values = {}

    @property
    def spinboxes(self):
        """Dict mapping trace_name -> QDoubleSpinBox."""
        return self._spinboxes

    def populate(self, trace_names, default_value=None, saved_values=None):
        """
        Populate the table with trace names in a 4-column layout.

        Parameters
        ----------
        trace_names : list of str
            Trace names to display.
        default_value : float, optional
            Default spinbox value. If None, uses -1.0 when use_default_marker
            is True, otherwise uses the minimum value.
        saved_values : dict, optional
            Previously saved values to restore (trace_name -> float).
        """
        import math
        from PyQt5.QtWidgets import QDoubleSpinBox, QTableWidgetItem, QAbstractSpinBox
        from PyQt5.QtCore import Qt
        from core.common.ui_theme import UITheme

        self._table.setRowCount(0)
        self._spinboxes.clear()

        if default_value is None:
            default_value = -1.0 if self._use_default_marker else self._min_val

        num_rows = math.ceil(len(trace_names) / 2)
        self._table.setRowCount(num_rows)

        for idx, trace_name in enumerate(trace_names):
            row_idx = idx // 2
            col_offset = (idx % 2) * 2

            # Trace name (read-only)
            name_item = QTableWidgetItem(trace_name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            self._table.setItem(row_idx, col_offset, name_item)

            # Determine initial value: saved > last used > default
            initial_value = default_value
            if saved_values and trace_name in saved_values:
                initial_value = saved_values[trace_name]
            elif trace_name in self._last_values:
                initial_value = self._last_values[trace_name]

            # Value spinbox
            spinbox = QDoubleSpinBox()
            spinbox.setRange(self._min_val, self._max_val)
            spinbox.setValue(initial_value)
            spinbox.setDecimals(self._decimals)
            spinbox.setSingleStep(self._step)
            spinbox.setStyleSheet(UITheme.get_spinbox_style(decimals=True))
            spinbox.setFocusPolicy(Qt.StrongFocus)
            spinbox.setButtonSymbols(QAbstractSpinBox.NoButtons)
            block_wheel_event(spinbox)
            spinbox.setToolTip(self._tooltip_template.format(name=trace_name))

            if self._use_default_marker:
                spinbox.setSpecialValueText("(default)")

            def on_value_changed(val, name=trace_name):
                self._last_values[name] = val
            spinbox.valueChanged.connect(on_value_changed)

            self._table.setCellWidget(row_idx, col_offset + 1, spinbox)
            self._spinboxes[trace_name] = spinbox

    def get_values(self):
        """
        Get per-trace values.

        Returns
        -------
        dict
            Mapping of trace_name -> value. If use_default_marker is True,
            entries with value < 0 are returned as None.
        """
        result = {}
        for trace_name, spinbox in self._spinboxes.items():
            value = spinbox.value()
            if self._use_default_marker and value < 0:
                result[trace_name] = None
            else:
                result[trace_name] = value
        return result

    def get_all_values(self):
        """Get all per-trace values for state saving (including defaults)."""
        return {name: spinbox.value() for name, spinbox in self._spinboxes.items()}

    def set_last_values(self, values):
        """Set stored values for restoration on next populate."""
        self._last_values = dict(values) if values else {}

    def set_all(self, value):
        """Set all spinboxes to the given value."""
        for spinbox in self._spinboxes.values():
            spinbox.setValue(value)
