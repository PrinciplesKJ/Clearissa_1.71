"""
Interactive Dashboard replacement for plot_df_in_new_tab()

"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QTreeWidget,
    QTreeWidgetItem, QPushButton, QLabel, QScrollArea, QGridLayout,
    QCheckBox
)
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import pandas as pd
import numpy as np
import logging

from core.common.ui_theme import Colors

logger = logging.getLogger(__name__)


class PlotDashboard(QWidget):
    """
    Interactive dashboard for displaying multiple related plots with selective visibility.
    """

    def __init__(self, dataframes, titles, parent=None, plot_mode='line'):
        """
        Initialise the dashboard.

        Args:
            dataframes: List of pandas DataFrames to plot
            titles: List of titles corresponding to each DataFrame
            parent: Parent widget
            plot_mode: 'line' or 'scatter' - determines the plot style
        """
        super().__init__(parent)
        self.dataframes = dataframes
        self.titles = titles
        self.plot_items = {}  # (title, well) -> plot_widget mapping
        self.tree_items = {}  # title -> QTreeWidgetItem mapping
        self.plot_mode = plot_mode  # 'line' or 'scatter'

        logger.info(f"Creating dashboard with {len(dataframes)} DataFrames in {plot_mode} mode")

        self._setup_ui()
        self._populate_tree()
        self._create_all_plots()
        self._update_grid()

        logger.info(f"Dashboard created: {len(self.plot_items)} plots in {len(self.tree_items)} tree items")

    def _setup_ui(self):
        """Create the main dashboard layout."""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(3, 3, 3, 3)
        main_layout.setSpacing(3)

        # Left panel - tree view with controls
        left_panel = QWidget()
        left_panel.setMaximumWidth(200)  # Reduced from 250 for small screens
        left_panel.setMinimumWidth(150)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(3)

        # Header
        header = QLabel("Plots")
        header.setStyleSheet("""
            QLabel {
                font-size: 10pt;
                font-weight: 600;
                color: #333;
                padding: 4px;
                background-color: {Colors.SECTION_BACKGROUND};
                border-radius: 3px;
            }
        """)
        left_layout.addWidget(header)

        # Tree widget
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setStyleSheet("""
            QTreeWidget {
                border: 1px solid #DDD;
                border-radius: 3px;
                background-color: white;
                font-size: 8pt;
            }
            QTreeWidget::item {
                padding: 2px;
            }
            QTreeWidget::item:hover {
                background-color: {Colors.INFO_PANEL_INTERNAL_BG};
            }
        """)
        self.tree.itemChanged.connect(self._on_tree_item_changed)
        left_layout.addWidget(self.tree)

        # Control buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(3)

        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self._select_all)
        select_all_btn.setStyleSheet("""
            QPushButton {
                padding: 4px 8px;
                background-color: {Colors.ACCENT_BLUE};
                color: white;
                border: none;
                border-radius: 3px;
                font-size: 8pt;
            }
            QPushButton:hover {
                background-color: {Colors.ACCENT_BLUE_HOVER};
            }
        """)

        clear_all_btn = QPushButton("Clear All")
        clear_all_btn.clicked.connect(self._clear_all)
        clear_all_btn.setStyleSheet("""
            QPushButton {
                padding: 4px 8px;
                background-color: {Colors.ACCENT_RED};
                color: white;
                border: none;
                border-radius: 3px;
                font-size: 8pt;
            }
            QPushButton:hover {
                background-color: {Colors.ACCENT_RED_HOVER};
            }
        """)

        btn_layout.addWidget(select_all_btn)
        btn_layout.addWidget(clear_all_btn)
        left_layout.addLayout(btn_layout)

        # Right panel - scrollable grid of plots
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Scroll area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: {Colors.SECTION_BACKGROUND};
            }
        """)

        # Grid container
        self.grid_container = QWidget()
        self.grid_layout = QGridLayout(self.grid_container)
        self.grid_layout.setSpacing(6)  # Reduced from 10
        self.grid_layout.setContentsMargins(6, 6, 6, 6)  # Reduced from 10

        self.scroll_area.setWidget(self.grid_container)
        right_layout.addWidget(self.scroll_area)

        # Add panels to splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)  # Prevent collapse
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)  # Fixed width for left
        splitter.setStretchFactor(1, 1)  # Expandable right

        main_layout.addWidget(splitter)

    def _populate_tree(self):
        """Populate the tree widget with plot items."""
        self.tree.blockSignals(True)

        for title, df in zip(self.titles, self.dataframes):
            # Create parent item for this DataFrame with EXACT title
            parent_item = QTreeWidgetItem(self.tree)
            parent_item.setText(0, title)
            parent_item.setFlags(parent_item.flags() | Qt.ItemIsUserCheckable)
            parent_item.setCheckState(0, Qt.Checked)
            parent_item.setExpanded(True)

            # Style parent
            font = parent_item.font(0)
            font.setBold(True)
            parent_item.setFont(0, font)

            self.tree_items[title] = parent_item

            # If DataFrame has 'Well' column, check if we need child items
            if isinstance(df, pd.DataFrame) and 'Well' in df.columns:
                wells = sorted(df['Well'].unique())

                # Only create children if there are multiple wells
                if len(wells) > 1:
                    for well in wells:
                        child_item = QTreeWidgetItem(parent_item)
                        child_item.setText(0, f"  {well}")
                        child_item.setFlags(child_item.flags() | Qt.ItemIsUserCheckable)
                        child_item.setCheckState(0, Qt.Checked)
                        self.tree_items[f"{title}_{well}"] = child_item
                else:
                    # Single well - parent controls visibility
                    if len(wells) == 1:
                        self.tree_items[f"{title}_{wells[0]}"] = parent_item

        self.tree.blockSignals(False)

    def _create_all_plots(self):
        """Create all plot widgets (hidden initially)."""
        for title, df in zip(self.titles, self.dataframes):
            if not isinstance(df, pd.DataFrame):
                logger.warning(f"Skipping non-DataFrame: '{title}'")
                continue

            if df.empty:
                logger.warning(f"Skipping empty DataFrame: '{title}'")
                continue

            if 'Well' in df.columns:
                wells = df['Well'].unique()

                if len(wells) == 1:
                    # Single well - create one plot with descriptive title
                    plot_widget = self._create_single_plot(df, wells[0], title)
                    if plot_widget:
                        self.plot_items[(title, str(wells[0]))] = plot_widget
                    else:
                        logger.warning(f"Failed to create plot: '{title}'")
                else:
                    # Multiple wells - create separate plots
                    for well in sorted(wells):
                        plot_widget = self._create_single_plot(df, well, title)
                        if plot_widget:
                            self.plot_items[(title, str(well))] = plot_widget
                        else:
                            logger.warning(f"Failed to create plot: '{title}' well {well}")
            else:
                # No Well column - single plot for entire dataframe
                plot_widget = self._create_single_plot(df, None, title)
                if plot_widget:
                    self.plot_items[(title, None)] = plot_widget
                else:
                    logger.warning(f"Failed to create plot: '{title}'")

    def _create_single_plot(self, df, well, title):
        """
        Create a single PyQtGraph plot widget.

        Args:
            df: DataFrame to plot
            well: Well identifier from 'Well' column (or None for single plot)
            title: Descriptive title from the dictionary key

        Returns:
            PlotWidget or None
        """
        try:
            # Filter by well if specified
            if well is not None:
                sub_df = df[df['Well'] == well].copy()
                plot_title = title
            else:
                sub_df = df.copy()
                plot_title = title

            if sub_df.empty:
                return None

            # Find time column
            time_col = None
            for col in ['Time [min]', 'Time', 'time', 't']:
                if col in sub_df.columns:
                    time_col = col
                    break

            if time_col is None:
                logger.warning(f"No time column found for '{plot_title}'")
                return None

            # Create plot widget
            plot_widget = pg.PlotWidget()
            plot_widget.setBackground('w')
            plot_widget.setMinimumHeight(180)  # Reduced for small screen compatibility
            plot_widget.setMinimumWidth(200)   # Reduced for small screen compatibility

            # Set labels and title
            # Determine appropriate y-axis label based on data type
            title_lower = title.lower()
            if 'relative fluorescence' in title_lower or 'rf(t)' in title_lower:
                y_label = 'Relative Fluorescence'
            elif 'concentration' in title_lower or '[nm]' in title_lower:
                y_label = 'Concentration (nM)'
            elif 'afu' in title_lower:
                y_label = 'Fluorescence (AFU)'
            else:
                y_label = 'Signal'
            plot_widget.setLabel('left', y_label, **{'font-size': '8pt'})
            plot_widget.setLabel('bottom', 'Time (min)', **{'font-size': '8pt'})
            plot_widget.setTitle(plot_title, color='#333', size='9pt')  # Reduced from 10pt
            plot_widget.showGrid(x=True, y=True, alpha=0.2)

            # CRITICAL: Disable automatic SI prefix scaling on axes
            # This prevents inconsistent display like "signal (x0.001)" between IDE and frozen exe
            # Y-axis should show actual values (e.g., 0.8) not scaled values (e.g., 800 with x0.001)
            plot_widget.getAxis('left').enableAutoSIPrefix(False)
            plot_widget.getAxis('bottom').enableAutoSIPrefix(False)

            # Get data columns (exclude Well, Time, Content)
            exclude_cols = ['Well', time_col, 'Content']
            y_cols = [c for c in sub_df.columns if c not in exclude_cols]

            if not y_cols:
                logger.warning(f"No data columns to plot for '{plot_title}'")
                return None

            # Plot each data column
            plotted_count = 0
            for i, y_col in enumerate(y_cols):
                try:
                    # Convert to numeric and get valid data
                    xdata = pd.to_numeric(sub_df[time_col], errors='coerce')
                    ydata = pd.to_numeric(sub_df[y_col], errors='coerce')

                    valid = np.isfinite(xdata) & np.isfinite(ydata)
                    if not valid.any():
                        continue

                    x_np = xdata[valid].to_numpy()
                    y_np = ydata[valid].to_numpy()

                    # Create pen with distinct colour
                    color = pg.intColor(i, hues=len(y_cols))
                    pen = pg.mkPen(color=color, width=2)

                    # Plot the data based on mode
                    if self.plot_mode == 'scatter':
                        # Scatter plot mode
                        plot_widget.plot(x_np, y_np, pen=None, symbol='o', symbolPen=color,
                                       symbolBrush=color, symbolSize=3, name=y_col)
                    else:
                        # Line plot mode (default)
                        plot_widget.plot(x_np, y_np, pen=pen, name=y_col)
                    plotted_count += 1

                except Exception:
                    continue

            if plotted_count == 0:
                logger.warning(f"No traces plotted for '{plot_title}'")
                return None

            # Add legend if not too many items
            if len(y_cols) <= 10:
                legend = plot_widget.addLegend(offset=(5, 5))  # Reduced offset
                legend.setLabelTextColor('#333')
                legend.setLabelTextSize('7pt')  # Reduced from 8pt

            return plot_widget

        except Exception as e:
            logger.error(f"Failed to create plot '{title}': {e}")
            return None

    def _update_grid(self):
        """Update the grid layout based on checked items."""
        # Clear existing layout
        for i in reversed(range(self.grid_layout.count())):
            widget = self.grid_layout.itemAt(i).widget()
            if widget:
                self.grid_layout.removeWidget(widget)
                widget.setParent(None)

        # Collect visible plots
        visible_plots = []
        for (title, well), plot_widget in self.plot_items.items():
            # Get the parent item
            parent_item = self.tree_items.get(title)
            if not parent_item or parent_item.checkState(0) != Qt.Checked:
                continue

            # Check if this is a multi-well or single-well plot
            well_str = str(well) if well is not None else None
            child_key = f"{title}_{well_str}" if well_str else None
            child_item = self.tree_items.get(child_key) if child_key else None

            # Include the plot if parent is checked and (no child OR child is checked)
            if child_item is None or child_item == parent_item:
                visible_plots.append(plot_widget)
            elif child_item.checkState(0) == Qt.Checked:
                visible_plots.append(plot_widget)

        if not visible_plots:
            # Show message if nothing selected
            label = QLabel("No plots selected. Check items in the tree to display plots.")
            label.setStyleSheet("""
                QLabel {
                    color: #666;
                    font-size: 10pt;
                    padding: 15px;
                }
            """)
            label.setAlignment(Qt.AlignCenter)
            self.grid_layout.addWidget(label, 0, 0)
            return

        # Determine grid dimensions - optimised for small screens
        # Use conservative column counts to ensure plots are visible on small displays
        n_plots = len(visible_plots)
        if n_plots == 1:
            rows, cols = 1, 1
        elif n_plots == 2:
            rows, cols = 2, 1  # Stack vertically for better visibility
        elif n_plots <= 4:
            rows, cols = 2, 2
        elif n_plots <= 6:
            rows, cols = 3, 2  # Use 2 columns instead of 3 for small screens
        else:
            # For many plots, use 2 columns for better small screen readability
            cols = 2
            rows = (n_plots + cols - 1) // cols

        # Add plots to grid
        for idx, plot_widget in enumerate(visible_plots):
            row = idx // cols
            col = idx % cols
            self.grid_layout.addWidget(plot_widget, row, col)

    def _on_tree_item_changed(self, item, column):
        """Handle tree item check state changes."""
        # If parent item changed, update all children
        if item.childCount() > 0:
            new_state = item.checkState(0)
            self.tree.blockSignals(True)
            for i in range(item.childCount()):
                child = item.child(i)
                child.setCheckState(0, new_state)
            self.tree.blockSignals(False)

        # If child item changed, update parent if needed
        elif item.parent() is not None:
            parent = item.parent()
            all_checked = True
            any_checked = False
            for i in range(parent.childCount()):
                child_state = parent.child(i).checkState(0)
                if child_state != Qt.Checked:
                    all_checked = False
                if child_state == Qt.Checked:
                    any_checked = True

            self.tree.blockSignals(True)
            if all_checked:
                parent.setCheckState(0, Qt.Checked)
            elif any_checked:
                parent.setCheckState(0, Qt.PartiallyChecked)
            else:
                parent.setCheckState(0, Qt.Unchecked)
            self.tree.blockSignals(False)

        # Update the grid
        self._update_grid()

    def _select_all(self):
        """Check all items in the tree."""
        self.tree.blockSignals(True)
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            item.setCheckState(0, Qt.Checked)
            for j in range(item.childCount()):
                item.child(j).setCheckState(0, Qt.Checked)
        self.tree.blockSignals(False)
        self._update_grid()

    def _clear_all(self):
        """Uncheck all items in the tree."""
        self.tree.blockSignals(True)
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            item.setCheckState(0, Qt.Unchecked)
            for j in range(item.childCount()):
                item.child(j).setCheckState(0, Qt.Unchecked)
        self.tree.blockSignals(False)
        self._update_grid()

    def update_plot_mode(self, mode):
        """
        Update the plot mode (line or scatter) and recreate all plots.

        Args:
            mode: 'line' or 'scatter'
        """
        if mode not in ['line', 'scatter']:
            logger.warning(f"Invalid plot mode: {mode}. Using 'line'.")
            mode = 'line'

        if self.plot_mode == mode:
            return  # No change needed

        logger.info(f"Updating plot mode from {self.plot_mode} to {mode}")
        self.plot_mode = mode

        # Recreate all plots with new mode
        self.plot_items.clear()
        self._create_all_plots()
        self._update_grid()

        logger.info(f"Plot mode updated to {mode}")


# Replace the plot_df_in_new_tab method in ConvertDataTab class:
def plot_df_in_new_tab(self, *dataframes, titles=None):
    """
    Create an interactive dashboard tab for the supplied DataFrames.

    Args:
        *dataframes: One or more pandas DataFrames to plot
        titles: str | list[str] | None - Title or list of titles for the dataframes
    """
    if not dataframes:
        return

    # Handle titles argument
    if titles is None:
        titles = [f"DataFrame {i + 1}" for i in range(len(dataframes))]
    elif isinstance(titles, str):
        titles = [titles] * len(dataframes)
    elif isinstance(titles, list):
        if len(titles) != len(dataframes):
            raise ValueError("Number of titles must match number of DataFrames.")
    else:
        raise TypeError("Titles must be a string, a list of strings, or None.")

    # Enhanced title mapping with descriptions
    title_map = {
        'g(t)': ('g(t)', 'Normalized Signal', 'Signal ratio to positive control'),
        'concentration': ('Conc.', 'Concentration', 'Calculated concentration values'),
        'fluorescence': ('Fluor.', 'Fluorescence', 'Raw or processed fluorescence'),
        'normalized': ('Norm.', 'Normalized Data', 'Normalized fluorescence values'),
        'background_subtracted': ('Bg Sub.', 'Background Subtracted', 'Background-subtracted fluorescence'),
        'ratio': ('Ratio', 'Signal Ratio', 'Ratio of signals')
    }

    # Map titles to display names
    display_titles = []
    for title in titles:
        title_lower = title.lower()
        found = False
        for key, (short, long, desc) in title_map.items():
            if key in title_lower:
                display_titles.append(long)
                found = True
                break
        if not found:
            display_titles.append(title)

    # Create dashboard widget
    try:
        dashboard = PlotDashboard(list(dataframes), display_titles, self)

        # Determine tab name (use first title, shortened)
        if display_titles:
            first_title = display_titles[0]
            tab_name = first_title[:12] + "..." if len(first_title) > 12 else first_title
        else:
            tab_name = "Results"

        # Add to tab widget
        self.left_tab_widget.addTab(dashboard, tab_name)

        # Switch to the new tab
        self.left_tab_widget.setCurrentWidget(dashboard)

        logger.info(f"Created dashboard tab '{tab_name}' with {len(dataframes)} datasets")

    except Exception as e:
        logger.error(f"Failed to create plot dashboard: {e}", exc_info=True)
        # Fallback to error message
        error_tab = QWidget()
        error_layout = QVBoxLayout(error_tab)
        error_label = QLabel(f"Error creating dashboard: {e}")
        error_label.setWordWrap(True)
        error_label.setStyleSheet("color: {Colors.ACCENT_RED}; padding: 20px;")
        error_layout.addWidget(error_label)
        self.left_tab_widget.addTab(error_tab, "Error")