"""
Clearissa - Unified UI Theme
-----------------------------
Centralised colour and style definitions for consistent visual design across
all Clearissa modules.

Provides:
- Colour palette (backgrounds, borders, text, accents)
- Standard widget styles (buttons, inputs, group boxes, etc.)
- Component-specific styling functions

Usage:
    from core.common.ui_theme import UITheme, Colours
    button_style = UITheme.get_button_style_primary()
    widget.setStyleSheet(button_style)

Author: Krizan Jurinovic
"""

from typing import Optional


# ---------------------------------------------------------------------------
# Colour Palette
# ---------------------------------------------------------------------------

class Colours:
    """Named colour constants for the Clearissa user interface.

    All colours are hex strings for direct use in Qt stylesheets.
    """

    # Base colours - backgrounds and surfaces
    CARD_BACKGROUND = "#FFFFFF"
    CONTENT_BACKGROUND = "#FFFFFF"
    MAIN_BACKGROUND = "#E9E9E9"
    SECTION_BACKGROUND = "#F5F5F5"

    # Border and separator colours
    CARD_BORDER = "#E0E0E0"
    BORDER_LIGHT = "#CCCCCC"
    BORDER_MEDIUM = "#999999"
    SEPARATOR = "#E0E0E0"

    # Text colours
    TEXT_PRIMARY = "#1A1A1A"
    TEXT_SECONDARY = "#4A4A4A"
    TEXT_TERTIARY = "#757575"
    TEXT_DISABLED = "#BDBDBD"

    # Accent colours - actions and states
    ACCENT_BLUE = "#1976D2"
    ACCENT_BLUE_HOVER = "#1565C0"
    ACCENT_BLUE_PRESSED = "#0D47A1"

    ACCENT_GREEN = "#4CAF50"
    ACCENT_GREEN_HOVER = "#388E3C"
    ACCENT_GREEN_PRESSED = "#1B5E20"

    ACCENT_ORANGE = "#F57C00"
    ACCENT_ORANGE_HOVER = "#E65100"

    ACCENT_RED = "#D32F2F"
    ACCENT_RED_HOVER = "#B71C1C"

    # Info panel colours (mode-specific)
    INFO_PANEL_TMSD_BG = "#E4F4EA"
    INFO_PANEL_TMSD_BORDER = "#7EB98A"
    INFO_PANEL_TMSD_TEXT = "#1F4F2C"

    INFO_PANEL_INTERNAL_BG = "#E4EFFA"
    INFO_PANEL_INTERNAL_BORDER = "#6D9CCF"
    INFO_PANEL_INTERNAL_TEXT = "#1D4C7F"

    INFO_PANEL_HMSD_BG = "#FFF1E2"
    INFO_PANEL_HMSD_BORDER = "#E2A768"
    INFO_PANEL_HMSD_TEXT = "#7C3F00"

    INFO_PANEL_MASS_BG = "#E4EFFA"
    INFO_PANEL_MASS_BORDER = "#8AB6E1"
    INFO_PANEL_MASS_TEXT = "#1D4C7F"

    # Well category colours (Data Frame Processor)
    WELL_UNASSIGNED = "#FFFFFF"
    WELL_DATA = "#90CAF9"
    WELL_BLANK = "#FFFFFF"
    WELL_POS_CTRL = "#A5D6A7"
    WELL_NEG_CTRL = "#EF9A9A"
    WELL_DONOR_CTRL = "#FFE082"
    WELL_ACCEPTOR_CTRL = "#FFAB91"

    # Channel wavelength colours (visual indicators)
    WAVELENGTH_BLUE = "#2563EB"
    WAVELENGTH_CYAN = "#06B6D4"
    WAVELENGTH_GREEN = "#16A34A"
    WAVELENGTH_YELLOW = "#CA8A04"
    WAVELENGTH_ORANGE = "#EA580C"
    WAVELENGTH_RED = "#DC2626"
    WAVELENGTH_FAR_RED = "#991B1B"

    # Plot colours
    PLOT_GRID = "#CCCCCC"
    PLOT_BACKGROUND = "#FFFFFF"
    PLOT_FOREGROUND = "#1A1A1A"


# ---------------------------------------------------------------------------
# Widget Style Templates
# ---------------------------------------------------------------------------

class UITheme:
    """Pre-built stylesheet templates for common Qt widgets.

    All methods return CSS stylesheet strings for use with setStyleSheet().
    """

    # -- Internal template helpers --

    @staticmethod
    def _coloured_button(bg: str, hover_bg: str, pressed_bg: str = None,
                         border_radius: int = 3, padding: str = "8px 12px",
                         font_size: str = "9pt") -> str:
        """Generate a coloured button stylesheet from parameters."""
        pressed = pressed_bg or hover_bg
        return f"""
            QPushButton {{
                background-color: {bg};
                color: white;
                border: none;
                border-radius: {border_radius}px;
                padding: {padding};
                font-size: {font_size};
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {hover_bg};
            }}
            QPushButton:pressed {{
                background-color: {pressed};
            }}
            QPushButton:disabled {{
                background-color: {Colours.BORDER_LIGHT};
                color: {Colours.TEXT_DISABLED};
            }}
        """

    # -- Button styles --

    @staticmethod
    def get_button_style_standard() -> str:
        """Neutral push button for secondary actions."""
        return f"""
            QPushButton {{
                background-color: {Colours.CARD_BACKGROUND};
                color: {Colours.TEXT_PRIMARY};
                border: 1px solid {Colours.BORDER_LIGHT};
                border-radius: 3px;
                padding: 6px 12px;
                font-size: 9pt;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: {Colours.SECTION_BACKGROUND};
                border-color: {Colours.BORDER_MEDIUM};
            }}
            QPushButton:pressed {{
                background-color: {Colours.MAIN_BACKGROUND};
            }}
            QPushButton:disabled {{
                background-color: {Colours.SECTION_BACKGROUND};
                color: {Colours.TEXT_DISABLED};
                border-color: {Colours.CARD_BORDER};
            }}
        """

    @staticmethod
    def get_button_style_primary(background_colour: str = None) -> str:
        """Primary action button. Optionally accepts a custom background colour."""
        bg = background_colour or Colours.ACCENT_BLUE
        hover_bg = Colours.ACCENT_BLUE_HOVER if background_colour is None else background_colour
        return UITheme._coloured_button(bg, hover_bg, Colours.ACCENT_BLUE_PRESSED)

    @staticmethod
    def get_button_style_success() -> str:
        """Success/action button (green) for simulation or positive actions."""
        return UITheme._coloured_button(
            Colours.ACCENT_GREEN, Colours.ACCENT_GREEN_HOVER, Colours.ACCENT_GREEN_PRESSED
        )

    @staticmethod
    def get_button_style_warning() -> str:
        """Warning button (orange) for actions requiring attention."""
        return UITheme._coloured_button(
            Colours.ACCENT_ORANGE, Colours.ACCENT_ORANGE_HOVER
        )

    @staticmethod
    def get_button_style_danger() -> str:
        """Destructive action button (red) for delete/remove operations."""
        return UITheme._coloured_button(
            Colours.ACCENT_RED, Colours.ACCENT_RED_HOVER
        )

    @staticmethod
    def get_button_style_info(size: int = 20) -> str:
        """Circular information button (e.g. 'i' icon)."""
        radius = size // 2
        return f"""
            QPushButton {{
                background-color: {Colours.ACCENT_BLUE};
                color: white;
                border: none;
                border-radius: {radius}px;
                font-weight: bold;
                font-size: 9pt;
                min-width: {size}px;
                max-width: {size}px;
                min-height: {size}px;
                max-height: {size}px;
            }}
            QPushButton:hover {{
                background-color: {Colours.ACCENT_BLUE_HOVER};
            }}
            QPushButton:pressed {{
                background-color: {Colours.ACCENT_BLUE_PRESSED};
            }}
        """

    @staticmethod
    def get_button_style_secondary() -> str:
        """Secondary button with border, sharp corners."""
        return f"""
            QPushButton {{
                background-color: {Colours.CARD_BACKGROUND};
                color: {Colours.TEXT_SECONDARY};
                border: 1px solid {Colours.CARD_BORDER};
                border-radius: 0px;
                padding: 5px 12px;
                font-weight: 600;
                font-size: 8pt;
            }}
            QPushButton:hover {{
                background-color: {Colours.CONTENT_BACKGROUND};
                border-color: {Colours.ACCENT_BLUE};
            }}
            QPushButton:disabled {{
                background-color: {Colours.SECTION_BACKGROUND};
                color: {Colours.TEXT_TERTIARY};
            }}
        """

    @staticmethod
    def get_button_style_category(colour: str) -> str:
        """Category assignment button coloured by the given hex value."""
        return f"""
            QPushButton {{
                background-color: {colour};
                color: white;
                border: 2px solid {colour};
                border-radius: 0px;
                padding: 4px 10px;
                font-size: 9px;
                font-weight: 600;
                min-height: 28px;
            }}
            QPushButton:hover {{
                background-color: {colour};
                border: 2px solid white;
            }}
            QPushButton:pressed {{
                background-color: {colour};
                border: 2px solid {Colours.BORDER_LIGHT};
            }}
        """

    @staticmethod
    def get_button_style_toolbar() -> str:
        """Toolbar button with sharp rectangular appearance."""
        return f"""
            QToolButton {{
                font-size: 9px;
                background-color: white;
                border: 1px solid {Colours.BORDER_MEDIUM};
                border-radius: 0px;
                padding: 4px 8px;
                font-weight: 500;
                color: {Colours.TEXT_SECONDARY};
            }}
            QToolButton:hover {{
                background-color: {Colours.INFO_PANEL_INTERNAL_BG};
                border-color: {Colours.ACCENT_BLUE};
                color: {Colours.ACCENT_BLUE};
            }}
            QToolButton:pressed {{
                background-color: {Colours.INFO_PANEL_INTERNAL_BG};
            }}
        """

    @staticmethod
    def get_button_style_toolbar_highlighted() -> str:
        """Highlighted toolbar button for emphasis."""
        return f"""
            QToolButton {{
                font-size: 10px;
                background-color: white;
                border: 2px solid {Colours.ACCENT_BLUE};
                border-radius: 0px;
                padding: 6px 10px;
                font-weight: 600;
                color: {Colours.ACCENT_BLUE};
            }}
            QToolButton:hover {{
                background-color: {Colours.ACCENT_BLUE};
                color: white;
            }}
            QToolButton:pressed {{
                background-color: {Colours.ACCENT_BLUE_HOVER};
                border-color: {Colours.ACCENT_BLUE_HOVER};
            }}
        """

    # -- Input styles --

    @staticmethod
    def get_lineedit_style(max_width: Optional[int] = None) -> str:
        """Standard text input field. Optional max_width in pixels."""
        width_rule = "max-width: %dpx;" % max_width if max_width else ""
        return f"""
            QLineEdit {{
                background-color: {Colours.CARD_BACKGROUND};
                border: 1px solid {Colours.CARD_BORDER};
                border-radius: 2px;
                padding: 3px 6px;
                font-size: 9pt;
                color: {Colours.TEXT_PRIMARY};
                {width_rule}
            }}
            QLineEdit:focus {{
                border-color: {Colours.ACCENT_BLUE};
            }}
            QLineEdit:hover {{
                border-color: {Colours.BORDER_MEDIUM};
            }}
            QLineEdit:disabled {{
                background-color: {Colours.SECTION_BACKGROUND};
                color: {Colours.TEXT_DISABLED};
            }}
        """

    @staticmethod
    def get_combobox_style() -> str:
        """Standard combo box."""
        return f"""
            QComboBox {{
                background-color: {Colours.CARD_BACKGROUND};
                border: 1px solid {Colours.CARD_BORDER};
                border-radius: 2px;
                padding: 4px 6px;
                font-size: 9pt;
                color: {Colours.TEXT_PRIMARY};
                min-height: 22px;
            }}
            QComboBox:focus {{
                border-color: {Colours.ACCENT_BLUE};
            }}
            QComboBox:hover {{
                border-color: {Colours.BORDER_MEDIUM};
            }}
            QComboBox:disabled {{
                background-color: {Colours.SECTION_BACKGROUND};
                color: {Colours.TEXT_DISABLED};
            }}
            QComboBox::drop-down {{
                border: none;
                width: 18px;
            }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid {Colours.TEXT_SECONDARY};
                width: 0px;
                height: 0px;
                margin-right: 4px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {Colours.CARD_BACKGROUND};
                border: 1px solid {Colours.ACCENT_BLUE};
                selection-background-color: {Colours.ACCENT_BLUE};
                selection-color: white;
                padding: 4px;
            }}
        """

    @staticmethod
    def get_spinbox_style(decimals: bool = False) -> str:
        """Numeric spin box. Set decimals=True for QDoubleSpinBox."""
        widget_type = "QDoubleSpinBox" if decimals else "QSpinBox"
        return f"""
            {widget_type} {{
                background-color: {Colours.CARD_BACKGROUND};
                border: 1px solid {Colours.CARD_BORDER};
                border-radius: 2px;
                padding: 3px;
                font-size: 9pt;
                color: {Colours.TEXT_PRIMARY};
            }}
            {widget_type}:focus {{
                border-color: {Colours.ACCENT_BLUE};
            }}
            {widget_type}:hover {{
                border-color: {Colours.BORDER_MEDIUM};
            }}
            {widget_type}:disabled {{
                background-color: {Colours.SECTION_BACKGROUND};
                color: {Colours.TEXT_DISABLED};
            }}
        """

    @staticmethod
    def get_checkbox_style() -> str:
        """Standard check box."""
        return f"""
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
                border-color: {Colours.BORDER_MEDIUM};
            }}
            QCheckBox::indicator:checked {{
                background-color: {Colours.ACCENT_BLUE};
                border-color: {Colours.ACCENT_BLUE};
            }}
        """

    @staticmethod
    def get_radiobutton_style() -> str:
        """Standard radio button."""
        return f"""
            QRadioButton {{
                font-size: 9pt;
                color: {Colours.TEXT_PRIMARY};
                spacing: 6px;
            }}
            QRadioButton:disabled {{
                color: {Colours.TEXT_DISABLED};
            }}
            QRadioButton::indicator {{
                width: 16px;
                height: 16px;
                border: 1px solid {Colours.CARD_BORDER};
                border-radius: 8px;
                background-color: {Colours.CARD_BACKGROUND};
            }}
            QRadioButton::indicator:hover {{
                border-color: {Colours.BORDER_MEDIUM};
            }}
            QRadioButton::indicator:checked {{
                background-color: {Colours.ACCENT_BLUE};
                border-color: {Colours.ACCENT_BLUE};
            }}
        """

    # -- Container styles --

    @staticmethod
    def get_groupbox_style() -> str:
        """Standard group box container."""
        return f"""
            QGroupBox {{
                background-color: {Colours.CARD_BACKGROUND};
                border: 1px solid {Colours.CARD_BORDER};
                border-radius: 3px;
                margin-top: 8px;
                padding: 6px 4px 4px 4px;
                font-weight: 600;
                font-size: 9pt;
                color: {Colours.TEXT_SECONDARY};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 8px;
                padding: 0 4px;
                background-color: {Colours.CARD_BACKGROUND};
            }}
        """

    @staticmethod
    def get_card_style() -> str:
        """Framed card or panel container."""
        return f"""
            QFrame {{
                background-color: {Colours.CARD_BACKGROUND};
                border: 1px solid {Colours.CARD_BORDER};
                border-radius: 3px;
            }}
        """

    @staticmethod
    def get_separator_style() -> str:
        """Horizontal separator line."""
        return f"""
            QFrame {{
                background-color: {Colours.SEPARATOR};
                border: none;
                max-height: 1px;
                min-height: 1px;
            }}
        """

    # -- Label and header styles --

    @staticmethod
    def get_label_style_primary() -> str:
        """Primary text label."""
        return "font-size: 9pt; color: %s; background-color: transparent;" % Colours.TEXT_PRIMARY

    @staticmethod
    def get_label_style_secondary() -> str:
        """Secondary (emphasised) text label."""
        return "font-size: 9pt; font-weight: 600; color: %s; background-color: transparent;" % Colours.TEXT_SECONDARY

    @staticmethod
    def get_label_style_tertiary() -> str:
        """Tertiary text (hints, help text)."""
        return "font-size: 8pt; color: %s; background-color: transparent; font-style: italic;" % Colours.TEXT_TERTIARY

    @staticmethod
    def get_header_style_sub(size: int = 10) -> str:
        """Sub-header label."""
        return "font-size: %dpt; font-weight: 600; color: %s; background-color: transparent;" % (size, Colours.TEXT_SECONDARY)

    @staticmethod
    def get_header_style_section(size: int = 11) -> str:
        """Section header label with blue accent colour."""
        return "font-size: %dpt; font-weight: 700; color: %s; background-color: transparent;" % (size, Colours.ACCENT_BLUE)

    # -- Table and scroll area styles --

    @staticmethod
    def get_table_style() -> str:
        """Standard table widget."""
        return f"""
            QTableWidget {{
                background-color: {Colours.CARD_BACKGROUND};
                border: 1px solid {Colours.CARD_BORDER};
                gridline-color: {Colours.SEPARATOR};
                font-size: 9pt;
            }}
            QTableWidget::item {{
                padding: 4px;
                border: none;
            }}
            QTableWidget::item:selected {{
                background-color: {Colours.ACCENT_BLUE};
                color: white;
            }}
            QHeaderView::section {{
                background-color: {Colours.SECTION_BACKGROUND};
                border: 1px solid {Colours.CARD_BORDER};
                padding: 4px;
                font-weight: bold;
                font-size: 9pt;
            }}
        """

    @staticmethod
    def get_scrollarea_style() -> str:
        """Scroll area with compact scrollbars."""
        return f"""
            QScrollArea {{
                background-color: {Colours.CONTENT_BACKGROUND};
                border: none;
            }}
            QScrollBar:vertical {{
                background-color: transparent;
                width: 8px;
                margin: 2px;
            }}
            QScrollBar::handle:vertical {{
                background-color: {Colours.BORDER_LIGHT};
                border-radius: 4px;
                min-height: 20px;
            }}
            QScrollBar::handle:vertical:hover {{
                background-color: {Colours.BORDER_MEDIUM};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
                background: none;
            }}
            QScrollBar:horizontal {{
                background-color: transparent;
                height: 8px;
                margin: 2px;
            }}
            QScrollBar::handle:horizontal {{
                background-color: {Colours.BORDER_LIGHT};
                border-radius: 4px;
                min-width: 20px;
            }}
            QScrollBar::handle:horizontal:hover {{
                background-color: {Colours.BORDER_MEDIUM};
            }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                width: 0px;
            }}
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
                background: none;
            }}
        """


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def get_wavelength_colour(wavelength_nm: int) -> str:
    """Map an emission wavelength (nm) to a representative UI colour."""
    try:
        wl = int(wavelength_nm)
        if wl < 495:
            return Colours.WAVELENGTH_BLUE
        if wl < 520:
            return Colours.WAVELENGTH_CYAN
        if wl < 565:
            return Colours.WAVELENGTH_GREEN
        if wl < 590:
            return Colours.WAVELENGTH_YELLOW
        if wl < 625:
            return Colours.WAVELENGTH_ORANGE
        if wl < 700:
            return Colours.WAVELENGTH_RED
        return Colours.WAVELENGTH_FAR_RED
    except (ValueError, TypeError):
        return Colours.TEXT_TERTIARY


def darken_colour(hex_colour: str, factor: float = 0.8) -> str:
    """Return a darker variant of a hex colour (factor < 1.0 = darker)."""
    hex_colour = hex_colour.lstrip('#')
    r = int(int(hex_colour[0:2], 16) * factor)
    g = int(int(hex_colour[2:4], 16) * factor)
    b = int(int(hex_colour[4:6], 16) * factor)
    return "#%02x%02x%02x" % (r, g, b)


def lighten_colour(hex_colour: str, factor: float = 1.2) -> str:
    """Return a lighter variant of a hex colour (factor > 1.0 = lighter)."""
    hex_colour = hex_colour.lstrip('#')
    r = min(255, int(int(hex_colour[0:2], 16) * factor))
    g = min(255, int(int(hex_colour[2:4], 16) * factor))
    b = min(255, int(int(hex_colour[4:6], 16) * factor))
    return "#%02x%02x%02x" % (r, g, b)


# ---------------------------------------------------------------------------
# Backward Compatibility Aliases
# ---------------------------------------------------------------------------

Colors = Colours
get_wavelength_color = get_wavelength_colour
darken_color = darken_colour
lighten_color = lighten_colour
