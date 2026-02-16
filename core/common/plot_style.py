"""
Clearissa - Centralised Plotting Style Module
---------------------------------------------
Shared plotting style configuration for matplotlib and PyQtGraph.

Author: Križan Jurinović
Date: October 2025
"""

import logging
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
import numpy as np

logger = logging.getLogger(__name__)

# ==========================================================================
# GLOBAL VISUAL CONSTANTS
# ==========================================================================

# Basic colour scheme
COLOUR_AXIS_BLACK = "black"
COLOUR_GRID_GREY = "#CCCCCC"
COLOUR_BACKGROUND_WHITE = "white"
COLOUR_LINE_NAVY = "#001F3F"
COLOUR_SCATTER_BLACK = "black"

# Colourblind-aware trace colours
# Wong (2011) Nature Methods + Paul Tol qualitative schemes
# Ref: Paul Tol (2012), SRON/EPS/TN/09-002 — https://personal.sron.nl/~pault/
TRACE_COLORS = [
    # Paul Tol "Bright" palette (7 colours)
    '#4477AA',  # Blue
    '#EE6677',  # Red
    '#228833',  # Green
    '#CCBB44',  # Yellow
    '#66CCEE',  # Cyan
    '#AA3377',  # Purple
    '#BBBBBB',  # Grey

    # Paul Tol "Vibrant" extended (13 additional colours)
    '#EE7733',  # Orange
    '#0077BB',  # Strong blue
    '#33BBEE',  # Bright cyan
    '#EE3377',  # Magenta
    '#CC3311',  # Dark red
    '#009988',  # Teal
    '#BBBBBB',  # Grey (repeated)
    '#000000',  # Black
    '#DDCC77',  # Sand
    '#117733',  # Forest green
    '#882255',  # Wine
    '#44AA88',  # Sea green
    '#AA4499'   # Orchid
]

# Matplotlib settings
AXIS_LINE_WIDTH = 1.2
AXIS_LABEL_SIZE = 11
TICK_LABEL_SIZE = 10
LEGEND_FONT_SIZE = 10
TITLE_FONT_SIZE = 12
DEFAULT_DPI = 120
EXPORT_DPI = 300
LINE_WIDTH_DEFAULT = 1.8
LINE_WIDTH_THIN = 1.5
LINE_WIDTH_THICK = 2.0
GRID_LINE_WIDTH = 0.5
SCATTER_SIZE_DEFAULT = 12
SCATTER_ALPHA = 0.5

# PyQtGraph settings
PYQTGRAPH_AXIS_PEN_WIDTH = 1.2
PYQTGRAPH_TICK_FONT_SIZE = 10
PYQTGRAPH_LABEL_FONT_SIZE = 11


# ==========================================================================
# MATPLOTLIB GLOBAL STYLE CONFIGURATION
# ==========================================================================

def apply_clearissa_style():
    """Apply Clearissa plotting defaults globally to matplotlib.

    This function should be called once during application initialisation
    before any plotting occurs. It configures ``matplotlib.rcParams`` so
    that all subsequent plots use a consistent appearance.

    The configuration includes:

    * Black axes, labels, and ticks on a white background
    * Standard font sizes for labels, ticks, legends, and titles
    * Light grey dashed grid lines
    * Line width defaults

    Returns
    -------
    None
    """
    try:
        plt.style.use("default")
        plt.rcParams.update({
            # Axes styling
            "axes.edgecolor": COLOUR_AXIS_BLACK,
            "axes.labelcolor": COLOUR_AXIS_BLACK,
            "axes.linewidth": AXIS_LINE_WIDTH,
            "axes.labelsize": AXIS_LABEL_SIZE,
            "axes.titlesize": TITLE_FONT_SIZE,
            "axes.facecolor": COLOUR_BACKGROUND_WHITE,

            # Tick styling
            "xtick.color": COLOUR_AXIS_BLACK,
            "ytick.color": COLOUR_AXIS_BLACK,
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "xtick.major.width": AXIS_LINE_WIDTH,
            "ytick.major.width": AXIS_LINE_WIDTH,

            # Legend styling
            "legend.fontsize": LEGEND_FONT_SIZE,
            "legend.framealpha": 0.9,
            "legend.edgecolor": COLOUR_AXIS_BLACK,

            # Figure styling
            "figure.dpi": DEFAULT_DPI,
            "figure.facecolor": COLOUR_BACKGROUND_WHITE,

            # Grid styling
            "grid.color": COLOUR_GRID_GREY,
            "grid.linestyle": "--",
            "grid.linewidth": GRID_LINE_WIDTH,
            "grid.alpha": 0.5,

            # Line styling
            "lines.linewidth": LINE_WIDTH_DEFAULT,

            # Font
            "font.size": 10,
        })
        logger.info("Matplotlib plotting style applied globally")
    except Exception as e:  # pragma: no cover - defensive logging only
        logger.error("Failed to apply plotting style: %s", e)


# ==========================================================================
# MATPLOTLIB AXIS CONFIGURATION
# ==========================================================================

def configure_axes(ax: Axes,
                   x_label: str = "Time [min]",
                   y_label: str = "Signal [AFU]",
                   title: str | None = None,
                   enable_grid: bool = True) -> None:
    """Configure a matplotlib axes object with Clearissa defaults.

    This helper applies consistent styling to a given axes object,
    including label text, tick parameters, spine appearance, optional
    grid lines, and enforcement of plain numeric time values on the
    x-axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matplotlib axes object to configure.
    x_label : str, optional
        X-axis label. The default is ``"Time [min]"``.
    y_label : str, optional
        Y-axis label. The default is ``"Signal [AFU]"``.
    title : str, optional
        Plot title. If ``None``, no title is set.
    enable_grid : bool, optional
        If ``True`` (default), draw a light dashed grid. If ``False``,
        the grid is disabled.

    Returns
    -------
    None

    Notes
    -----
    The helper also calls :func:`enforce_time_units` to disable
    automatic SI-prefix scaling on the x-axis, so that time values are
    displayed as plain numbers in minutes.
    """
    # Set labels
    ax.set_xlabel(x_label, fontsize=AXIS_LABEL_SIZE, color=COLOUR_AXIS_BLACK)
    ax.set_ylabel(y_label, fontsize=AXIS_LABEL_SIZE, color=COLOUR_AXIS_BLACK)

    # Set title if provided
    if title:
        ax.set_title(title, fontsize=TITLE_FONT_SIZE, color=COLOUR_AXIS_BLACK, pad=6)

    # Configure tick parameters
    ax.tick_params(axis="both", colors=COLOUR_AXIS_BLACK,
                   labelsize=TICK_LABEL_SIZE, width=AXIS_LINE_WIDTH)

    # Set spine colours (axis lines)
    for spine in ax.spines.values():
        spine.set_color(COLOUR_AXIS_BLACK)
        spine.set_linewidth(AXIS_LINE_WIDTH)

    # Disable automatic SI scaling on x-axis
    enforce_time_units(ax)

    # Configure grid
    if enable_grid:
        ax.grid(True, linestyle="--", linewidth=GRID_LINE_WIDTH,
                color=COLOUR_GRID_GREY, alpha=0.5)
    else:
        ax.grid(False)


def enforce_time_units(ax: Axes) -> None:
    """Enforce plain minute-based display on the x-axis.

    This helper prevents matplotlib from applying automatic SI prefixes
    (for example, ``"1.5k min"``) or scientific notation to the x-axis.
    Tick labels are shown as plain numbers in minutes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matplotlib axes object to configure.

    Returns
    -------
    None
    """
    # Disable offset and scientific notation on x-axis
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)

    # Also configure the formatter directly, where supported
    x_formatter = ax.get_xaxis().get_major_formatter()
    if hasattr(x_formatter, "set_useOffset"):
        x_formatter.set_useOffset(False)
    if hasattr(x_formatter, "set_scientific"):
        x_formatter.set_scientific(False)

    logger.debug("Time axis configured for plain display (no scaling)")


def plot_trace(ax: Axes,
               x: np.ndarray,
               y: np.ndarray,
               label: str | None = None,
               colour: str = COLOUR_LINE_NAVY,
               linewidth: float = LINE_WIDTH_DEFAULT,
               linestyle: str = "-",
               show_scatter: bool = False,
               marker: str = "o",
               markersize: float = 4):
    """Plot a single trace with the standard Clearissa style.

    This helper provides a unified interface for plotting traces and can
    optionally draw both a line and markers on the same axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matplotlib axes to plot on.
    x : numpy.ndarray
        X-axis data (typically time in minutes).
    y : numpy.ndarray
        Y-axis data (for example, signal or concentration).
    label : str, optional
        Legend label for this trace.
    colour : str, optional
        Line and marker colour. The default is navy.
    linewidth : float, optional
        Line width in points. The default is ``1.8``.
    linestyle : str, optional
        Matplotlib line style. The default is ``"-"``.
    show_scatter : bool, optional
        If ``True``, draw markers along the line. If ``False`` (default),
        draw only the line.
    marker : str, optional
        Marker style when ``show_scatter`` is ``True``. The default is
        ``"o"``.
    markersize : float, optional
        Marker size when ``show_scatter`` is ``True``. The default is ``4``.

    Returns
    -------
    matplotlib.lines.Line2D
        The line object created.

    Notes
    -----
    When ``show_scatter`` is ``True``, line and markers are drawn using a
    single :func:`Axes.plot` call with marker properties specified.
    """
    if show_scatter:
        # Combined line and marker path
        line = ax.plot(
            x,
            y,
            color=colour,
            linewidth=linewidth,
            linestyle=linestyle,
            marker=marker,
            markersize=markersize,
            markerfacecolor=colour,
            markeredgecolor="white",
            markeredgewidth=0.5,
            label=label,
        )
        return line[0]

    # Line-only path
    line = ax.plot(x, y, color=colour, linewidth=linewidth,
                   linestyle=linestyle, label=label)
    return line[0]


def add_scatter_overlay(ax: Axes,
                        x: np.ndarray,
                        y: np.ndarray,
                        colour: str = COLOUR_SCATTER_BLACK,
                        size: float = SCATTER_SIZE_DEFAULT,
                        alpha: float = SCATTER_ALPHA,
                        marker: str = "o"):
    """Add a scatter overlay to an existing plot.

    This is a visual-only operation that does not alter the underlying
    data. It is typically used to highlight individual measurements on
    top of a line trace.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matplotlib axes to add scatter to.
    x : numpy.ndarray
        X-axis data.
    y : numpy.ndarray
        Y-axis data.
    colour : str, optional
        Marker colour. The default is black.
    size : float, optional
        Marker size (``s`` argument to :func:`Axes.scatter`).
        The default is ``12``.
    alpha : float, optional
        Marker opacity. The default is ``0.5``.
    marker : str, optional
        Marker style. The default is ``"o"``.

    Returns
    -------
    matplotlib.collections.PathCollection
        The scatter plot object.
    """
    scatter = ax.scatter(x, y, s=size, color=colour, alpha=alpha,
                         marker=marker, zorder=10)
    return scatter


def save_figure(fig: Figure,
                filepath: str,
                dpi: int = EXPORT_DPI,
                format: str | None = None) -> None:
    """Save a figure with consistent export settings.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    filepath : str
        Output file path.
    dpi : int, optional
        Resolution for raster formats. The default is ``300``.
    format : str, optional
        File format. If ``None``, matplotlib infers the format from
        ``filepath``.

    Returns
    -------
    None

    Notes
    -----
    The function always uses ``bbox_inches="tight"`` to minimise
    clipping of labels and legends. Vector formats such as PDF and SVG
    are supported alongside raster formats such as PNG.
    """
    try:
        fig.savefig(
            filepath,
            dpi=dpi,
            bbox_inches="tight",
            format=format,
            facecolor="white",
            edgecolor="none",
        )
        logger.info("Figure saved to %s (DPI: %d)", filepath, dpi)
    except Exception as e:  # pragma: no cover - filesystem error path
        logger.error("Failed to save figure to %s: %s", filepath, e)
        raise


# ==========================================================================
# PYQTGRAPH STYLING
# ==========================================================================

def configure_pyqtgraph_widget(plot_widget: pg.PlotWidget,
                               x_label: str = "Time [min]",
                               y_label: str = "Signal [AFU]",
                               title: str | None = None,
                               enable_grid: bool = True,
                               background: str = "w") -> None:
    """Configure a :class:`pyqtgraph.PlotWidget` for Clearissa.

    This helper applies a consistent style to PyQtGraph widgets and aims
    to approximate the matplotlib appearance. It adapts axis and text
    colours based on the background so that text remains legible on
    light and dark backgrounds.

    Parameters
    ----------
    plot_widget : pyqtgraph.PlotWidget
        Widget to configure.
    x_label : str, optional
        X-axis label. The default is ``"Time [min]"``.
    y_label : str, optional
        Y-axis label. The default is ``"Signal [AFU]"``.
    title : str, optional
        Plot title. If ``None``, no title is set.
    enable_grid : bool, optional
        If ``True`` (default), enable grid lines.
    background : str, optional
        Background colour specification accepted by PyQtGraph, e.g.
        ``"w"`` for white or ``"k"`` for black.

    Returns
    -------
    None
    """
    # Set background
    plot_widget.setBackground(background)

    # Determine axis/text colour based on background
    is_dark_background = background in ["k", "black", "#000000"]
    axis_colour = "w" if is_dark_background else COLOUR_AXIS_BLACK
    text_colour = (255, 255, 255) if is_dark_background else COLOUR_AXIS_BLACK

    # Configure axes appearance with adaptive colour
    axis_pen = pg.mkPen(axis_colour, width=PYQTGRAPH_AXIS_PEN_WIDTH)
    plot_widget.getAxis("bottom").setPen(axis_pen)
    plot_widget.getAxis("left").setPen(axis_pen)

    # Set labels with adaptive colour
    label_style = {"color": text_colour, "font-size": f"{PYQTGRAPH_LABEL_FONT_SIZE}pt"}
    plot_widget.setLabel("bottom", x_label, **label_style)
    plot_widget.setLabel("left", y_label, **label_style)

    # Configure tick font and colour
    tick_font = QtGui.QFont()
    tick_font.setPointSize(PYQTGRAPH_TICK_FONT_SIZE)
    plot_widget.getAxis("bottom").setStyle(tickFont=tick_font, tickTextOffset=10)
    plot_widget.getAxis("left").setStyle(tickFont=tick_font, tickTextOffset=10)
    plot_widget.getAxis("bottom").setTextPen(text_colour)
    plot_widget.getAxis("left").setTextPen(text_colour)

    # Set title if provided
    if title:
        plot_widget.setTitle(title, color=text_colour,
                             size=f"{TITLE_FONT_SIZE}pt")

    # Configure grid with adaptive alpha
    if enable_grid:
        grid_alpha = 0.2 if is_dark_background else 0.3
        plot_widget.showGrid(x=True, y=True, alpha=grid_alpha)

    logger.debug(
        "PyQtGraph widget configured: x_label=%s, y_label=%s, background=%s",
        x_label,
        y_label,
        background,
    )


def get_trace_colour(index: int) -> str:
    """Return a trace colour from the predefined palette.

    The index is taken modulo the palette length so that an arbitrary
    number of traces can be styled in a repeatable way.

    Parameters
    ----------
    index : int
        Zero-based trace index.

    Returns
    -------
    str
        Hex colour code.
    """
    return TRACE_COLORS[index % len(TRACE_COLORS)]


# American spelling alias for compatibility
get_trace_color = get_trace_colour


# ==========================================================================
# UNIT LABEL HELPERS
# ==========================================================================

def get_time_label(unit: str = "min") -> str:
    """Return a standardised time axis label.

    Parameters
    ----------
    unit : str, optional
        Time unit. The default is ``"min"``.

    Returns
    -------
    str
        Formatted label of the form ``"Time [unit]"``.
    """
    return f"Time [{unit}]"


def get_signal_label(unit: str = "AFU") -> str:
    """Return a standardised signal axis label.

    Parameters
    ----------
    unit : str, optional
        Signal unit. The default is ``"AFU"`` (arbitrary fluorescence
        units).

    Returns
    -------
    str
        Formatted label of the form ``"Signal [unit]"``.
    """
    return f"Signal [{unit}]"


def get_concentration_label(unit: str = "nM") -> str:
    """Return a standardised concentration axis label.

    Parameters
    ----------
    unit : str, optional
        Concentration unit. The default is ``"nM"``.

    Returns
    -------
    str
        Formatted label of the form ``"Concentration [unit]"``.
    """
    return f"Concentration [{unit}]"


# ==========================================================================
# VISUAL TOGGLE PATTERN (for GUI implementations)
# ==========================================================================

class PlotRefreshMixin:
    """Mixin providing a simple pattern for refreshing plots.

    The mixin separates visual refresh from data recomputation. GUI
    classes using this mixin are expected to implement two methods:

    * ``_update_data()`` – recompute or fetch data from the current
      application state.
    * ``_redraw_plot_elements()`` – clear and redraw all visual plot
      elements based on the current data attributes.
    """

    def refresh_plot(self, visual_only: bool = False) -> None:
        """Refresh a plot, optionally skipping data recomputation.

        Parameters
        ----------
        visual_only : bool, optional
            If ``True``, call only ``_redraw_plot_elements``. If ``False``
            (default), call ``_update_data`` first and then redraw.
        """
        # Optionally update data
        if not visual_only and hasattr(self, "_update_data"):
            self._update_data()

        # Redraw plot elements if the hook is present
        if hasattr(self, "_redraw_plot_elements"):
            self._redraw_plot_elements()


# ==========================================================================
# INITIALISATION
# ==========================================================================

logger.info("plot_style module loaded; plotting helpers available")
