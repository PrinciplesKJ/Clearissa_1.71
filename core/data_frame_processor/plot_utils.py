"""
Plotting and visual utilities for DataFrameProcessor.

Handles:
- Generating plots from filtered data
- Channel colour and wavelength mapping
- In-tab and popup plot display
"""

import re
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QMainWindow
from core.common.data_processing_utils import create_plot_object

import logging
logger = logging.getLogger(__name__)


class PlotUtils:
    """Provides plotting and visual utilities for DataFrameProcessor."""

    def __init__(self, processor):
        self.p = processor
        self.logger = processor.logger

    # -------------------------------------------------------------------------
    # PLOT CREATION
    # -------------------------------------------------------------------------
    def get_plot_widget(self):
        """
        Generate a plot widget from the current GUI selections.
        Handles channel/well filtering and unit conversion.
        """

        df = getattr(self.p, "merged_dataframe", None)
        if df is None or df.empty:
            self.logger.warning("No data loaded; cannot create plot.")
            return None

        # Get selected time window
        start, end = self.p.get_start_end_times()

        # Determine selected channels (label or checkbox)
        ch_dict = getattr(self.p, "selected_channels_checkboxes", {})
        if not ch_dict:
            self.logger.info("No channel data available for plotting.")
            return None

        if hasattr(next(iter(ch_dict.values())), "_selected"):
            selected_channels = [ch for ch, lbl in ch_dict.items() if getattr(lbl, "_selected", False)]
        else:
            selected_channels = [ch for ch, cb in ch_dict.items() if cb.isChecked()]

        # Determine selected wells - include data, blank, and negative control wells only
        wells_dict = {}

        # Add data wells
        data_wells = getattr(self.p, "select_wells_checkboxes", {})
        wells_dict.update(data_wells)

        # Add blank wells
        blank_wells = getattr(self.p, "select_blank_checkboxes", {})
        wells_dict.update(blank_wells)

        # Add negative control wells
        neg_wells = getattr(self.p, "select_negctrl_checkboxes", {})
        wells_dict.update(neg_wells)

        selected_wells = [w for w in wells_dict.keys()]

        if not selected_channels or not selected_wells:
            self.logger.debug("No channels or wells selected for plotting.")
            return None

        # Filter data
        filtered = self.p.dataops.filter_data(start, end, ch_dict, wells_dict)
        if filtered is None or filtered.empty:
            self.logger.warning("Filtered dataframe is empty; cannot plot.")
            return None

        # Convert display units
        filtered = self._convert_display_units(filtered)

        # Generate plot with channel mapping
        return create_plot_object(
            filtered,
            scatter_state=self.p.scatter_state,
            black_white_state=self.p.black_white_state,
            grid_state=self.p.grid_state,
            show_legend=self.p.legend_state,
            channel_map=self.p.channel_map,
        )


    # -------------------------------------------------------------------------
    # UNIT CONVERSION
    # -------------------------------------------------------------------------
    def _convert_display_units(self, df):
        """Convert 'Time [min]' column according to display unit selection."""
        if "Time [min]" not in df.columns:
            return df

        if hasattr(self.p, "display_unit_combo"):
            unit = self.p.display_unit_combo.currentText()
        else:
            unit = "Minutes"

        factors = {
            "Seconds": 60,
            "Minutes": 1,
            "Hours": 1 / 60,
            "Days": 1 / (60 * 24),
            "Weeks": 1 / (60 * 24 * 7),
        }
        factor = factors.get(unit, 1)
        df["Time [min]"] = df["Time [min]"] * factor
        return df

    # -------------------------------------------------------------------------
    # PLOT DISPLAY
    # -------------------------------------------------------------------------
    def update_plot(self):
        """
        Refresh the main plot in the View Data tab.

        Thread-safe plot update that handles widget cleanup properly.
        This method is safe to call multiple times due to debouncing in main.py.
        """
        # Safety check: ensure plot_frame still exists
        if not hasattr(self.p, 'plot_frame') or self.p.plot_frame is None:
            self.logger.warning("plot_frame not available, skipping plot update")
            return

        plot_widget = self.get_plot_widget()
        if not plot_widget:
            return

        try:
            layout = self.p.plot_frame.layout()
            if layout is None:
                layout = QVBoxLayout(self.p.plot_frame)

            # Remove old widgets safely
            # Use deleteLater() to avoid deleting widgets while they're being processed
            while layout.count():
                item = layout.takeAt(0)
                if item.widget():
                    widget = item.widget()
                    # Disconnect any signals to prevent issues during deletion
                    try:
                        widget.setParent(None)
                    except RuntimeError:
                        pass  # Widget already deleted
                    widget.deleteLater()

            # Add new plot widget with proper sizing
            # Ensure plot expands to fill available space
            from PyQt5.QtWidgets import QSizePolicy
            plot_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            # Set minimum size to ensure plot is visible even in smaller windows
            plot_widget.setMinimumSize(400, 300)
            layout.addWidget(plot_widget)

            # Force layout to update and respect size policies
            layout.update()
            self.p.plot_frame.updateGeometry()

            self.logger.debug("Plot updated successfully")

        except Exception as ex:
            self.logger.error("Error updating plot: {}".format(ex))
            import traceback
            self.logger.error(traceback.format_exc())

    def show_plot_in_window(self):
        """Open the current plot in a new window."""
        plot_widget = self.get_plot_widget()
        if not plot_widget:
            self.logger.warning("No plot available for display.")
            return

        win = QMainWindow()
        central = QWidget(win)
        layout = QVBoxLayout(central)
        layout.addWidget(plot_widget)
        win.setCentralWidget(central)
        win.setWindowTitle("Plot Viewer")
        win.resize(900, 600)
        win.show()

        if not hasattr(self.p, "plot_windows"):
            self.p.plot_windows = []
        self.p.plot_windows.append(win)

    # -------------------------------------------------------------------------
    # COLOUR AND WAVELENGTH HELPERS
    # -------------------------------------------------------------------------
    @staticmethod
    def extract_wavelength(channel_name):
        """Extract numeric wavelength from text like '550-10/590-10'."""
        matches = re.findall(r"(\d{3,4})", str(channel_name))
        return int(matches[0]) if matches else None

    @staticmethod
    def wavelength_to_colour(wavelength):
        """Approximate colour mapping for visible-light wavelengths."""
        try:
            wl = int(wavelength)
        except (TypeError, ValueError):
            return "#8BC34A"

        if wl < 420:
            return "#8B00FF"  # Violet
        elif wl < 450:
            return "#0000FF"  # Blue
        elif wl < 495:
            return "#00BFFF"  # Cyan
        elif wl < 570:
            return "#00FF00"  # Green
        elif wl < 590:
            return "#FFFF00"  # Yellow
        elif wl < 620:
            return "#FFA500"  # Orange
        else:
            return "#FF0000"  # Red

    @staticmethod
    def get_contrast_text_colour(background_hex):
        """Return black or white text depending on background brightness."""
        hex_ = background_hex.lstrip("#")
        try:
            r, g, b = [int(hex_[i:i+2], 16) for i in (0, 2, 4)]
        except ValueError:
            return "black"
        luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
        return "black" if luminance > 0.5 else "white"
