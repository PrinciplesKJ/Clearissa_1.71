"""
Custom GUI components and widgets for the DataFrameProcessor module.
Contains reusable UI elements like ElidedLabel, ModernCard, WellButton, etc.

Author: Krizan Jurinovic, November 2025
"""

from PyQt5.QtWidgets import (QWidget, QFrame, QLabel, QPushButton,
                            QVBoxLayout, QHBoxLayout, QGridLayout,
                            QToolButton, QDoubleSpinBox, QComboBox,
                            QSizePolicy, QScrollArea, QApplication)
from PyQt5.QtCore import Qt, QSize, QTimer
from PyQt5.QtGui import QColor, QPalette

class ElidedLabel(QLabel):
    """
    QLabel that truncates long text with ellipsis and shows the full text
    as a tooltip on hover.

    Parameters
    ----------
    text : str, optional
        Initial text to display (default: "").
    parent : QWidget, optional
        Parent widget (default: None).
    """

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.full_text = text
        self.setTextFormat(Qt.PlainText)  # Use plain text for better control
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

    def setText(self, text):
        """
        Set the label text and trigger elision if necessary.

        Parameters
        ----------
        text : str or None
            Text to display. None is converted to empty string.
        """
        self.full_text = str(text) if text else ""
        self._update_display()

    def _update_display(self):
        """
        Update the displayed text with elision if needed.

        This method is called automatically when text changes or the widget
        is resized. It uses Qt's font metrics to determine if elision is needed.

        Elision Process
        ---------------
        1. Calculate available width (widget width minus padding)
        2. Use QFontMetrics.elidedText() to truncate with "..."
        3. Update tooltip only if text is actually elided
        """
        if not self.full_text:
            super().setText("")
            self.setToolTip("")
            return

        # Get available width (subtract padding for accurate measurement)
        available_width = self.width() - 10  # 10px padding
        if available_width <= 0:
            available_width = 250  # Default minimum width

        # Use font metrics to elide text at the right edge
        metrics = self.fontMetrics()
        elided = metrics.elidedText(self.full_text, Qt.ElideRight, available_width)

        super().setText(elided)

        # Only show tooltip if text is actually elided (prevents redundant tooltips)
        if elided != self.full_text:
            self.setToolTip(self.full_text)
        else:
            self.setToolTip("")

    def resizeEvent(self, event):
        """
        Handle widget resize events by updating text elision.

        Parameters
        ----------
        event : QResizeEvent
            Resize event containing new widget dimensions.
        """
        super().resizeEvent(event)

        # Only update if widget is fully initialised and visible
        try:
            if self.isVisible() and self.width() > 0:
                self._update_display()
        except RuntimeError:
            # Widget is being deleted, ignore resize event
            pass

class LongClickButton(QPushButton):
    """
    QPushButton with separate short-click and long-click (>= 1.3 s) callbacks.
    Shows a hover tooltip after 2 seconds.
    """

    def __init__(self, text, short_callback, long_callback, tooltip_text, parent=None):
        super().__init__(text, parent)

        self.short_callback = short_callback
        self.long_callback = long_callback
        self.tooltip_text = tooltip_text

        # Long-click mechanics
        self.press_timer = QTimer(self)
        self.press_timer.setSingleShot(True)
        self.press_timer.timeout.connect(self._on_long_press)
        self.long_press_duration = 1300

        self.is_long_press = False
        self.is_pressed = False
        self.original_style = ""

        # Set tooltip immediately - no delay needed
        self.setToolTip(tooltip_text)

        # Disable default click behaviour - we handle it manually
        self.setAutoDefault(False)
        self.setDefault(False)

    def enterEvent(self, event):
        """Called when mouse enters the button."""
        super().enterEvent(event)

    def leaveEvent(self, event):
        """Called when mouse leaves the button."""
        # Don't clear tooltip - keep it persistent
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        """Start long-press timer on mouse press."""
        if event.button() == Qt.LeftButton:
            self.is_long_press = False
            self.is_pressed = True
            self.original_style = self.styleSheet()

            # Start the long-press timer
            self.press_timer.start(self.long_press_duration)

            # Visual feedback: slightly darken during press
            self._apply_press_feedback()

        # Don't call super() to prevent default button behaviour
        event.accept()

    def mouseReleaseEvent(self, event):
        """Handle button release - execute short or long action."""
        if event.button() == Qt.LeftButton and self.is_pressed:
            # Stop the timer
            self.press_timer.stop()

            # Restore original style
            self.setStyleSheet(self.original_style)

            # Execute appropriate callback
            if self.is_long_press:
                # Long press was triggered
                self.long_callback()
            else:
                # Short press
                self.short_callback()

            # Reset flags
            self.is_long_press = False
            self.is_pressed = False

        # Don't call super() to prevent default button behaviour
        event.accept()

    def _on_long_press(self):
        """Called when long-press duration is reached."""
        self.is_long_press = True
        # Enhanced visual feedback when long-press is triggered
        self._apply_long_press_feedback()

    def _apply_press_feedback(self):
        """Apply subtle visual feedback during press."""
        # Darken the button slightly during press
        current_style = self.styleSheet()
        self.setStyleSheet(current_style + " QPushButton { opacity: 0.8; }")

    def _apply_long_press_feedback(self):
        """Apply stronger visual feedback when long-press is triggered."""
        # Add a visible flash effect when long-press threshold is reached
        current_style = self.original_style
        self.setStyleSheet(current_style + " QPushButton { border: 3px solid #FFEB3B !important; }")


class ModernCard(QFrame):
    """Industrial card container without shadow effects."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)

        # No shadow effect - industrial style uses flat design
        # Styling is done by callers via apply_card_style()

class WellButton(QPushButton):
    """
    Custom well button with drag selection support.
    """

    def __init__(self, well_id, selection_manager, parent=None):
        super().__init__("", parent)
        self.well_id = well_id
        self.selection_manager = selection_manager
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)

        # Enable mouse tracking to detect enter events during drag
        self.setMouseTracking(True)

        # Drag tracking
        self.mouse_pressed = False
        self.drag_start_pos = None
        self.drag_threshold = 3  # Reduced threshold for more responsive drag

    def mousePressEvent(self, event):
        """Handle mouse press - start potential drag or click."""
        if event.button() == Qt.LeftButton:
            self.mouse_pressed = True
            self.drag_start_pos = event.pos()
            # Don't do anything here - wait for release or move

        # Don't call super() to prevent default button behavior conflicts
        event.accept()

    def mouseMoveEvent(self, event):
        """Handle mouse move - both for drag detection and drag continuation."""
        if self.mouse_pressed and self.drag_start_pos:
            # Check if we've moved enough to consider it a drag
            if (event.pos() - self.drag_start_pos).manhattanLength() > self.drag_threshold:
                # This is a drag operation - start it if not already active
                if not self.selection_manager.drag_active:
                    self.selection_manager.start_drag_selection(self.well_id)
                else:
                    # Continue dragging on this well
                    self.selection_manager.continue_drag_selection(self.well_id)

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Handle mouse release - finalise selection."""
        if event.button() == Qt.LeftButton and self.mouse_pressed:
            self.mouse_pressed = False

            # Check if Shift key is held for rectangular selection
            if QApplication.keyboardModifiers() == Qt.ShiftModifier:
                if self.selection_manager.drag_start_well:
                    # Complete rectangular selection
                    self.selection_manager.select_rectangular_region(
                        self.selection_manager.drag_start_well,
                        self.well_id
                    )
                else:
                    # First well with Shift - store as drag start and select it
                    self.selection_manager.drag_start_well = self.well_id
                    self.selection_manager.select_well(self.well_id)
            elif self.selection_manager.drag_active:
                # End drag operation
                self.selection_manager.end_drag_selection()
            else:
                # Simple click - just select the well
                self.selection_manager.select_well(self.well_id)

            self.drag_start_pos = None

        # Don't call super() to prevent default button behavior conflicts
        event.accept()

    def enterEvent(self, event):
        """Handle mouse enter - detect and continue drag selection."""
        # Check if left mouse button is currently pressed (indicates drag operation)
        if QApplication.mouseButtons() & Qt.LeftButton:
            if not self.selection_manager.drag_active:
                # Start drag if not already active and mouse is pressed
                self.selection_manager.start_drag_selection(self.well_id)
            else:
                # Continue selecting this well as part of the drag operation
                self.selection_manager.continue_drag_selection(self.well_id)

        super().enterEvent(event)
