# core/convert_data_tab/ui_components.py
"""
UI Component Utilities for Convert Data Tab
============================================
Reusable UI components and helpers for the conversion tab interface:
event filters, species combo boxes, matrix tables, and layout utilities.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING
import logging

from PyQt5.QtCore import Qt, QEvent, QObject
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtWidgets import (
    QComboBox, QTableWidget, QHeaderView, QTableWidgetItem,
    QAbstractScrollArea, QToolTip, QCompleter, QPushButton,
    QFrame, QLabel
)

if TYPE_CHECKING:
    from .gui import ConvertDataTab

logger = logging.getLogger(__name__)


# ==================== STYLING CONSTANTS ====================

from core.common.ui_theme import Colors, UITheme

# Style constants used by this module
SUB_HEADER_STYLE = UITheme.get_header_style_sub(size=10)
COMBOBOX_STYLE = UITheme.get_combobox_style()
INFO_BUTTON_STYLE = UITheme.get_button_style_info(size=20)

# Compact species combo style: no grey drop-down button, minimal padding
SPECIES_COMBO_STYLE = f"""
    QComboBox {{
        background-color: {Colors.CARD_BACKGROUND};
        border: 1px solid {Colors.CARD_BORDER};
        border-radius: 2px;
        padding: 2px 4px;
        font-size: 7pt;
        color: {Colors.TEXT_PRIMARY};
        min-height: 20px;
    }}
    QComboBox:focus {{
        border-color: {Colors.ACCENT_BLUE};
    }}
    QComboBox:hover {{
        border-color: {Colors.BORDER_MEDIUM};
    }}
    QComboBox::drop-down {{
        border: none;
        width: 12px;
        background: transparent;
    }}
    QComboBox::down-arrow {{
        image: none;
        border-left: 3px solid transparent;
        border-right: 3px solid transparent;
        border-top: 4px solid {Colors.TEXT_SECONDARY};
        width: 0px;
        height: 0px;
        margin-right: 2px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {Colors.CARD_BACKGROUND};
        border: 1px solid {Colors.ACCENT_BLUE};
        selection-background-color: {Colors.ACCENT_BLUE};
        selection-color: white;
        padding: 2px;
        font-size: 8pt;
    }}
"""


# ==================== EVENT FILTERS ====================

class ComboTooltipFilter(QObject):
    """Event filter that shows tooltips when hovering over QComboBox dropdown items."""

    def __init__(self, view):
        """
        Initialise the tooltip filter.

        Args:
            view: The QListView or QTableView of the combo box dropdown
        """
        super().__init__(view)
        self._view = view

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        """
        Filter events to show tooltips on hover.

        Args:
            obj: The object that triggered the event
            event: The event to filter

        Returns:
            True if event was handled, False otherwise
        """
        if event.type() == QEvent.ToolTip:
            index = self._view.indexAt(event.pos())
            if index.isValid():
                text = index.data(Qt.ToolTipRole)
                if text:
                    QToolTip.showText(
                        self._view.viewport().mapToGlobal(event.pos()),
                        text,
                        self._view
                    )
                    return True
            QToolTip.hideText()
            event.ignore()
            return True
        return super().eventFilter(obj, event)


# ==================== COMBO BOX BUILDERS ====================

class ComboBoxFactory:
    """
    Factory class for creating specialised combo boxes with calibration data.

    This factory creates QComboBox widgets populated with species from
    calibration records, complete with colour coding, tooltips, and
    autocomplete functionality.
    """

    @staticmethod
    def create_approach_combo(combo: QComboBox) -> None:
        """
        Populate conversion mode combo box with available approaches.

        Args:
            combo: QComboBox to populate with conversion modes
        """
        combo.addItem("Basic TMSD", userData="tmsd")
        combo.addItem("Internal Toehold", userData="internal_tmsd")
        combo.addItem("Normalised SD (Relative Fluorescence)", userData="normalised_sd")
        combo.addItem("One-step HMSD", userData="hmsd")
        combo.addItem("Hairpin Quenching TMSD", userData="hp_quenching")
        combo.addItem("Mass-action catalytic", userData="mass")

    @staticmethod
    def create_species_combo(
            parent: ConvertDataTab,
            *,
            hint: Optional[str] = None,
            prefer: Optional[List[str]] = None
    ) -> QComboBox:
        """
        Create a sorted, colour-coded combo box populated from calibration records.

        Args:
            parent: Parent ConvertDataTab widget (provides calibration_records)
            hint: Optional hint for species categorisation
            prefer: Optional list of preferred categories for sorting

        Returns:
            Configured QComboBox with species selections
        """
        from .species import SpeciesManager

        cb = QComboBox(parent)
        cb.setEditable(True)
        cb.setStyleSheet(SPECIES_COMBO_STYLE)

        # Update tooltip to show the full species name when selection changes
        def _update_species_tooltip(index: int):
            text = cb.itemText(index)
            cb.setToolTip(text if text and text != "NONE" else "")
        cb.currentIndexChanged.connect(_update_species_tooltip)

        # Load calibration records and build index
        try:
            recs = parent.calibration_records
            from .calibration import build_index, list_species
            idx = build_index(recs)
            names = list_species(idx)
        except Exception as e:
            logger.debug(f"Calibration listing failed: {e}")
            names, idx = [], None

        # Add "NONE" option first
        cb.addItem("NONE")

        # Sort names alphabetically
        names.sort(key=lambda n: n.lower())

        # Add each species with formatting
        for name in names:
            cb.addItem(name)
            item = cb.model().item(cb.count() - 1)

            # Categorise and colour-code
            cat = SpeciesManager.categorise_species(name, hint)
            qc = QColor(SpeciesManager.COLORS.get(cat, SpeciesManager.COLORS["default"]))

            # Apply bold font with increased size
            font = item.font()
            font.setBold(True)
            font.setPointSize(font.pointSize() + 1)
            item.setFont(font)

            # Set foreground colour
            item.setForeground(QBrush(qc))

            # Add tooltip with calibration details
            if idx:
                from .calibration import lookup
                rec = lookup(idx, name) or {}
                slope = rec.get("slope", "n/a")
                intercept = rec.get("intercept", "n/a")
                comment = rec.get("comments", "No comment")
                tt = (
                    f"Species: {name}\n"
                    f"Comment: {comment}\n"
                    f"Slope: {slope}\n"
                    f"Intercept: {intercept}"
                )
                item.setData(tt, Qt.ToolTipRole)

        # Setup autocomplete
        completer = QCompleter(cb.model(), cb)
        completer.setFilterMode(Qt.MatchContains)
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        completer.setCompletionMode(QCompleter.PopupCompletion)
        cb.setCompleter(completer)

        # Enable tooltips in dropdown
        view = cb.view()
        view.setMouseTracking(True)
        view.viewport().setMouseTracking(True)
        flt = ComboTooltipFilter(view)
        view.viewport().installEventFilter(flt)
        cb._tooltip_filter = flt  # Keep reference to prevent garbage collection

        # Make the dropdown popup 3x wider than the combo box itself
        # so the full species names are readable
        view.setMinimumWidth(320)

        return cb


# ==================== TABLE WIDGETS ====================

class MatrixTableFactory:
    """
    Factory class for creating formatted table widgets for coefficient display.

    This factory creates consistent table widgets used for displaying
    calibration slopes and brightness ratios in the conversion interface.
    """

    @staticmethod
    def create_matrix_table() -> QTableWidget:
        """
        Create a formatted table widget for coefficient/slope display.

        Returns:
            Styled QTableWidget configured for matrix data display
        """
        table = QTableWidget()

        # Apply table styling
        table.setStyleSheet("""
            QTableWidget {
                font-size: 8pt;
                gridline-color: #DDD;
            }
            QHeaderView::section {
                background-color: #F5F5F5;
                padding: 2px;
                border: 1px solid #DDD;
                font-weight: bold;
                font-size: 8pt;
            }
            QTableWidget::item {
                padding: 2px;
            }
        """)

        # Configure resize behaviour
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        table.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)

        # Set size constraints
        table.setMinimumHeight(100)
        table.setMaximumHeight(200)

        return table

    @staticmethod
    def populate_coefficient_table(
            table: QTableWidget,
            species_map: Dict[tuple[str, str], Dict[str, Any]],
            channels: List[str],
            species: List[str]
    ) -> None:
        """
        Populate table with brightness ratios normalised to B1B2 = 1.0.

        Each cell shows the ratio: slope(species, channel) / slope(B1B2, channel)
        This represents the relative brightness contribution to each channel.

        Args:
            table: QTableWidget to populate
            species_map: Dictionary mapping (channel, species) to calibration data
            channels: List of channel names (rows)
            species: List of species names (columns)
        """
        table.setRowCount(len(channels))
        table.setColumnCount(len(species))
        table.setHorizontalHeaderLabels(species)
        table.setVerticalHeaderLabels(channels)

        for r, channel in enumerate(channels):
            # Get B1B2 slope for this channel (reference = 1.0)
            b1b2_data = species_map.get((channel, "B1B2"), {})
            b1b2_slope = b1b2_data.get("slope")

            for c, sp in enumerate(species):
                species_data = species_map.get((channel, sp), {})
                species_slope = species_data.get("slope")

                item = QTableWidgetItem()

                if b1b2_slope and species_slope and abs(b1b2_slope) > 1e-12:
                    ratio = species_slope / b1b2_slope
                    item.setText(f"{ratio:.4f}")

                    # Colour code based on magnitude
                    if sp == "B1B2":
                        item.setBackground(QBrush(QColor("#E8F5E9")))  # Light green for product
                    elif ratio > 0.5:
                        item.setBackground(QBrush(QColor("#FFEBEE")))  # Light red for high bleed
                    elif ratio > 0.1:
                        item.setBackground(QBrush(QColor("#FFF3E0")))  # Light orange for medium bleed
                    else:
                        item.setBackground(QBrush(QColor("#FFFFFF")))  # White for low bleed
                else:
                    item.setText("—")
                    item.setForeground(QBrush(QColor("#999")))

                item.setTextAlignment(Qt.AlignCenter)
                table.setItem(r, c, item)

    @staticmethod
    def populate_slope_table(
            table: QTableWidget,
            species_map: Dict[tuple[str, str], Dict[str, Any]],
            channels: List[str],
            species: List[str]
    ) -> None:
        """
        Populate table with actual calibration slopes in AFU/nM.

        Each cell shows the measured slope from calibration data.

        Args:
            table: QTableWidget to populate
            species_map: Dictionary mapping (channel, species) to calibration data
            channels: List of channel names (rows)
            species: List of species names (columns)
        """
        table.setRowCount(len(channels))
        table.setColumnCount(len(species))
        table.setHorizontalHeaderLabels(species)
        table.setVerticalHeaderLabels(channels)

        for r, channel in enumerate(channels):
            for c, sp in enumerate(species):
                species_data = species_map.get((channel, sp), {})
                species_slope = species_data.get("slope")
                species_name = species_data.get("species", "")

                item = QTableWidgetItem()

                if species_slope is not None:
                    item.setText(f"{species_slope:.1f}")
                    item.setToolTip(f"{species_name}\n{species_slope:.4f} AFU/nM")

                    # Colour code by species type
                    if sp == "B1B2":
                        item.setBackground(QBrush(QColor("#E8F5E9")))  # Product
                    elif sp == "B2" or sp == "B1L":
                        item.setBackground(QBrush(QColor("#E3F2FD")))  # Donor-like
                    elif sp == "B1AA":
                        item.setBackground(QBrush(QColor("#F3E5F5")))  # Acceptor-like
                    else:
                        item.setBackground(QBrush(QColor("#FFFFFF")))
                else:
                    item.setText("—")
                    item.setForeground(QBrush(QColor("#999")))

                item.setTextAlignment(Qt.AlignCenter)
                table.setItem(r, c, item)


# ==================== UTILITY FUNCTIONS ====================

def set_combo_text_safe(combo: QComboBox, text: str) -> None:
    """
    Helper to safely set combo box text.

    This function attempts to find and set the exact text match first,
    and if not found, sets the current text directly (useful for editable combos).

    Args:
        combo: QComboBox to update
        text: Text value to set
    """
    if not text:
        return

    index = combo.findText(text)
    if index >= 0:
        combo.setCurrentIndex(index)
    else:
        # For editable combo boxes, set the text directly
        combo.setCurrentText(text)


def datawell_columns(df: Any) -> List[str]:
    """
    Return data-well column names, excluding 'Well' and 'Time [min]'.

    Args:
        df: DataFrame to extract column names from

    Returns:
        List of data column names (typically well identifiers like A01, A02, etc.)
    """
    try:
        cols = list(getattr(df, "columns", []))
        return cols[2:] if len(cols) > 2 else []
    except Exception:
        return []


def create_info_button(tooltip_text: str, parent=None) -> QPushButton:
    """
    Create a circular info button that can be clicked to show information.

    Parameters
    ----------
    tooltip_text : str
        The brief tooltip text (full info shown in dialog on click)
    parent : QWidget, optional
        Parent widget

    Returns
    -------
    QPushButton
        Configured info button
    """
    info_btn = QPushButton("i", parent)
    info_btn.setStyleSheet(INFO_BUTTON_STYLE)

    # Simple tooltip for the button itself
    info_btn.setToolTip(tooltip_text)

    info_btn.setCursor(Qt.PointingHandCursor)
    info_btn.setFocusPolicy(Qt.NoFocus)  # Don't steal focus from other widgets
    # Button is clickable - caller should connect the clicked signal
    return info_btn


def create_section_separator(parent=None) -> QWidget:
    """
    Create a thin horizontal line separator for visual section distinction.
    The separator is more subtle (lighter color) and shorter (70% width).

    Parameters
    ----------
    parent : QWidget, optional
        Parent widget

    Returns
    -------
    QWidget
        Container with centered horizontal line separator
    """
    from PyQt5.QtWidgets import QWidget, QHBoxLayout
    from PyQt5.QtCore import Qt

    container = QWidget(parent)
    container.setStyleSheet("background-color: transparent;")
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 4, 0, 4)

    # Add left stretch
    layout.addStretch(15)

    # Create the line
    line = QFrame(container)
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Plain)
    # More subtle color - lighter grey with some transparency
    line.setStyleSheet(f"background-color: rgba(224, 224, 224, 0.4); max-height: 1px;")
    layout.addWidget(line, 70)  # 70% of available space

    # Add right stretch
    layout.addStretch(15)

    return container


def create_section_header(text: str, parent=None) -> QLabel:
    """
    Create a consistent section header label.

    Parameters
    ----------
    text : str
        Header text
    parent : QWidget, optional
        Parent widget

    Returns
    -------
    QLabel
        Styled section header
    """
    header = QLabel(text, parent)
    header.setStyleSheet(SUB_HEADER_STYLE)
    return header


# ==================== MODE INFORMATION PROVIDER ====================

class ModeInfoProvider:
    """
    Provider class for conversion mode information and descriptions.

    This class provides detailed information text for each conversion mode,
    including reaction mechanisms, conservation laws, and required calibration
    species. Extracts docstrings from actual conversion functions.
    """

    @staticmethod
    def get_mode_info_text(mode: str) -> str:
        """
        Return detailed information text for each conversion mode.

        Extracts the actual docstring from the conversion function for maximum
        accuracy and detail.

        Args:
            mode: Conversion mode identifier (e.g., 'tmsd', 'hmsd', 'mass')

        Returns:
            Detailed description of the conversion mode from function docstring
        """
        try:
            # Import conversion functions to access their docstrings
            from .conversion_TMSD import convert_tmsd_to_conc
            from .conversion_Internal_TMSD import convert_internal_tmsd_to_conc
            from .conversion_normalised_sd import convert_normalised_sd
            from .conversion_HMSD_FRET import convert_fret_onestep_to_conc
            from .conversion_catalytic_FRET import convert_catalytic_fret_2x2

            # Map modes to their conversion functions
            function_map = {
                "tmsd": convert_tmsd_to_conc,
                "internal_tmsd": convert_internal_tmsd_to_conc,
                "normalised_sd": convert_normalised_sd,
                "hmsd": convert_fret_onestep_to_conc,
                "mass": convert_catalytic_fret_2x2,
            }

            func = function_map.get(mode)
            if func:
                docstring = func.__doc__
                if docstring:
                    logger.debug(f"Successfully extracted docstring for mode '{mode}' ({len(docstring)} chars)")
                    return docstring.strip()
                else:
                    logger.warning(f"Function for mode '{mode}' has no docstring")
            else:
                logger.warning(f"No function mapping found for mode '{mode}'")

            # Fallback for modes without docstrings
            logger.debug(f"Using fallback info for mode '{mode}'")
            return ModeInfoProvider._get_fallback_info(mode)

        except Exception as e:
            logger.error(f"Error extracting docstring for mode '{mode}': {e}", exc_info=True)
            return ModeInfoProvider._get_fallback_info(mode)

    @staticmethod
    def _get_fallback_info(mode: str) -> str:
        """Fallback info text when docstring extraction fails."""
        mode_info = {
            "tmsd": (
                "One-step Toehold-Mediated Strand Displacement\n\n"
                "Reaction: Ax + B -> AB + x\n"
                "(Ax = blocked substrate, AB = product)\n\n"
                "Uses 'nuking' to determine total [Ax]0 via excess invader.\n"
                "Corrects for residual Ax fluorescence (quenched acceptor).\n\n"
                "No calibration species required - beta is calculated empirically\n"
                "per well from initial and nuking plateaus."
            ),
            "internal_tmsd": (
                "Internal Toehold-Mediated Strand Displacement\n\n"
                "Reaction: ABT + F -> ABF + T\n\n"
                "Mechanism: A non-fluorescent fuel strand (F) invades the duplex-template "
                "complex (ABT) via an internal toehold, releasing the quencher-conjugated "
                "template (T). The template quenches the fluorophore on the AB duplex. "
                "Upon displacement by the fuel strand, the duplex binds to fuel (ABF), "
                "resulting in unquenching and fluorescence emission.\n\n"
                "Uses negative control (ABT without fuel) as baseline reference for initial "
                "fluorescence. No calibration species required - slope calculated from positive "
                "control wells."
            ),
            "normalised_sd": (
                "Normalised SD (Relative Fluorescence)\n\n"
                "Converts raw fluorescence to normalised relative fluorescence units.\n\n"
                "This mode is used when absolute concentration calibration is not available "
                "or when relative changes in fluorescence are sufficient for analysis.\n\n"
                "No calibration species required."
            ),
            "hmsd": (
                "One-step FRET/Handhold Mediated Strand Displacement\n\n"
                "Reaction: B1AA + B2 -> B1B2 + AA\n"
                "(Donor-first initialisation)\n\n"
                "Uses ONLY donor channel: Y_D(t) = [B2](t) + Beta * [B1B2](t)\n"
                "Beta is calculated from donor and product control signals.\n"
                "Conservation: [B2](t) + [B1B2](t) = [B2]0\n\n"
                "No calibration species required - beta is calculated at runtime\n"
                "from donor-only and product control fluorescence signals."
            ),
            "mass": (
                "Catalytic 2-step TMSD with FRET Readout\n\n"
                "Reactions:\n"
                "(1) B1L + AA -> B1AA + L\n"
                "(2) B1AA + B2 -> B1B2 + AA\n\n"
                "Conservation:\n"
                "- B1L + B1AA + B1B2 = B1L0 (user-specified)\n"
                "- B2 + B1B2 = B20 (measured)\n\n"
                "Solves for B1B2 and B1AA using Donor+FRET channels (2x2 solver).\n\n"
                "Required: 4 species x 2 channels = 8 calibration slopes\n"
                "(B1B2, B1L, B1AA, B2 for both Donor and FRET channels)."
            ),
        }
        return mode_info.get(mode, "Select a conversion mode to see method details")


