"""
Clearissa - data_frame_processor/main.py
-----------------------------------------
Main controller for data viewing, transformation, and analysis operations.

Author: Križan Jurinović
Date: October 2025
"""

import logging
from tokenize import blank_re

from PyQt5.QtWidgets import QWidget, QVBoxLayout

# Internal imports
from core.data_frame_processor.gui import DataFrameProcessorGUI
from core.data_frame_processor.data_ops import DataOps
from core.data_frame_processor.io_utils import IOUtils
from core.data_frame_processor.plot_utils import PlotUtils
from core.data_frame_processor.well_selection import WellSelectionManager
from core.common import data_processing_utils as dpu
from core.common.ui_theme import Colors


class DataFrameProcessor:
    """
    Main coordination class for the data viewing and analysis interface.

    This class manages event handling, data state, and communication between
    GUI components and back-end processing modules.

    Parameters
    ----------
    master : QWidget
        Parent widget for Qt hierarchy.
    processdatainstance : object
        Reference to CSV data loader instance.
    options_processor : OptionsPanel
        Reference to application options manager.

    Attributes
    ----------
    merged_dataframe : pandas.DataFrame or None
        Combined experimental data from all loaded files.
    csvdict : dict or None
        Dictionary of individual file data structures.
    selected_wells : list
        Currently selected wells for analysis.
    channel_map : dict
        Mapping of channel identifiers to display names.

    Notes
    -----
    The processor maintains state between sessions via pickle serialisation.
    All GUI interactions are routed through this central coordinator.
    """

    def __init__(self, master, processdatainstance, options_processor):
        self.master = master
        self.processdatainstance = processdatainstance
        self.options_processor = options_processor

        # Logging
        self.logger = logging.getLogger(__name__)

        # Channel name mapping
        self.channel_map = {
            "405-460": "CFP",
            "405-530": "FRET",
            "485-530": "YFP",
            "485-460": "Donor Leakage",
            "535-580": "RFP",
        }

        # Submodules
        self.well_selection_manager = WellSelectionManager(self)
        self.gui = DataFrameProcessorGUI(self)
        self.dataops = DataOps(self)
        self.io = IOUtils(self)
        self.plotutils = PlotUtils(self)

        # Connect well selection signals
        self.well_selection_manager.selection_changed.connect(self._on_selection_changed)
        self.well_selection_manager.assignments_changed.connect(self._on_assignments_changed)

        # Runtime data
        self.merged_dataframe = None
        self.csvdict = None
        self.columns = []
        self.detected_inj_timepoints = None
        self.selected_wells = []

        # Data loading state flag - tracks whether new data was just loaded
        self._fresh_data_loaded = False

        # GUI state references
        self.plot_frame = None

        # Well tracking (legacy compatibility - now mapped to well_selection_manager)
        self.well_buttons = {}
        self.select_wells_checkboxes = {}
        self.select_blank_checkboxes = {}
        self.select_posctrl_checkboxes = {}
        self.select_negctrl_checkboxes = {}
        self.select_donorctrl_checkboxes = {}
        self.select_acceptorctrl_checkboxes = {}

        # Plot toggles
        self.scatter_state = False
        self.black_white_state = True
        self.grid_state = False
        self.legend_state = True

        # Cache for CSV info to avoid repeated computation
        self._cached_csv_info = None
        self._cached_exp_id = None

        # Plot update debouncing to prevent crashes from rapid selections
        # Create the timer ONCE here, reuse it later
        from PyQt5.QtCore import QTimer
        self._plot_update_timer = QTimer()
        self._plot_update_timer.setSingleShot(True)
        self._plot_update_timer.timeout.connect(self._execute_plot_update)
        self._plot_update_pending = False
        self._inside_assignments_changed = False  # Prevent recursion
        self._inside_status_update = False  # Prevent status frame recursion

        # Channel colour map
        self.channel_colours = {
            "405": "#4FC3F7",
            "480": "#81C784",
            "535": "#FFB74D",
            "560": "#BA68C8",
            "590": "#E57373",
            "620": "#64B5F6",
            "650": "#AED581",
            "680": "#FFD54F",
            "750": "#90A4AE",
        }

        # Styling
        self.frame_style = """
            QFrame {
                background-color: {Colors.SECTION_BACKGROUND};
                border: 1px solid {Colors.BORDER_LIGHT};
                border-radius: 4px;
            }
        """

    # -------------------------------------------------------------------------
    # WELL SELECTION SIGNAL HANDLERS
    # -------------------------------------------------------------------------
    def _on_selection_changed(self):
        """Handle well selection changes (temporary selections)."""
        self._refresh_well_display()
        self.update_status_frame()

    def _on_assignments_changed(self):
        """Handle changes to permanent well assignments."""
        # Prevent recursive calls
        if self._inside_assignments_changed:
            self.logger.warning("Preventing recursive call to _on_assignments_changed")
            return

        try:
            self._inside_assignments_changed = True

            # Update legacy dictionaries for backward compatibility
            self.select_wells_checkboxes = {well: True for well in self.well_selection_manager.get_wells_by_category('data')}
            self.select_blank_checkboxes = {well: True for well in self.well_selection_manager.get_wells_by_category('blank')}
            self.select_posctrl_checkboxes = {well: True for well in self.well_selection_manager.get_wells_by_category('pos_ctrl')}
            self.select_negctrl_checkboxes = {well: True for well in self.well_selection_manager.get_wells_by_category('neg_ctrl')}
            self.select_donorctrl_checkboxes = {well: True for well in self.well_selection_manager.get_wells_by_category('donor_ctrl')}
            self.select_acceptorctrl_checkboxes = {well: True for well in self.well_selection_manager.get_wells_by_category('acceptor_ctrl')}
            self.select_blockedctrl_checkboxes = {well: True for well in self.well_selection_manager.get_wells_by_category('blocked_ctrl')}

            # Update visual appearance of all well buttons with signal blocking
            for well_id, button in self.well_buttons.items():
                if not self._is_valid_widget(button):
                    continue

                # Block signals to prevent cascading updates
                button.blockSignals(True)

                colour = self.well_selection_manager.get_well_colour(well_id)
                is_assigned = self.well_selection_manager.is_well_assigned(well_id)

                if is_assigned:
                    # Solid colour for assigned wells
                    darker = self._darken_colour(colour)
                    button.setStyleSheet(f"""
                        QPushButton {{
                            background-color: {colour};
                            border: 2px solid {darker};
                            border-radius: 4px;
                        }}
                        QPushButton:hover {{
                            background-color: {darker};
                        }}
                    """)
                else:
                    # Default style for unassigned wells
                    button.setStyleSheet(self.gui._well_button_style())

                # Restore signal handling
                button.blockSignals(False)

            # Update status display
            self.update_status_frame()

            # Use debounced timer to defer plot update
            self._schedule_plot_update()

        finally:
            self._inside_assignments_changed = False

    def _schedule_plot_update(self):
        """Schedule a debounced plot update (resets on repeated calls)."""
        # Cancel any pending update by stopping the timer
        if self._plot_update_timer.isActive():
            self._plot_update_timer.stop()

        # Restart the timer with 250ms delay (increased for stability)
        # The timer was created once in __init__, we just restart it here
        self._plot_update_timer.start(250)

    def _execute_plot_update(self):
        """Execute the plot update once the debounce timer fires."""
        if self._plot_update_pending:
            self.logger.warning("Plot update already in progress, skipping")
            return  # Already updating, skip

        try:
            self._plot_update_pending = True
            self.logger.debug("Executing debounced plot update")
            self.update_plot()
            self.logger.debug("Plot update completed successfully")
        except Exception as e:
            self.logger.error(f"Error during plot update: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
        finally:
            self._plot_update_pending = False

    def _refresh_well_display(self):
        """Update visual display of all well buttons."""
        for well_id, button in self.well_selection_manager.well_buttons.items():
            if not self._is_valid_widget(button):
                continue

            state = self.well_selection_manager.well_states.get(well_id, {})
            is_selected = state.get("selected", False)
            category = state.get("category", "unassigned")

            # Determine visual style based on state
            if category != "unassigned":
                # Well is assigned to a category
                colour = WellSelectionManager.COLORS[category]
                border_colour = self._darken_colour(colour)
                text_colour = PlotUtils.get_contrast_text_colour(colour)

                if is_selected:
                    # Assigned AND selected - show thick blue selection outline over assignment color
                    button.setStyleSheet(f"""
                        QPushButton {{
                            background-color: {colour};
                            color: {text_colour};
                            border: 3px solid {Colors.ACCENT_BLUE};
                            border-radius: 4px;
                        }}
                        QPushButton:hover {{
                            background-color: {border_colour};
                        }}
                    """)
                else:
                    # Assigned but not selected - normal assignment color
                    button.setStyleSheet(f"""
                        QPushButton {{
                            background-color: {colour};
                            color: {text_colour};
                            border: 2px solid {border_colour};
                            border-radius: 4px;
                        }}
                        QPushButton:hover {{
                            background-color: {border_colour};
                        }}
                    """)
                button.setChecked(True)
            elif is_selected:
                # Unassigned but selected - light blue outline
                button.setStyleSheet(f"""
                    QPushButton {{
                        background-color: white;
                        border: 3px solid {Colors.ACCENT_BLUE};
                        border-radius: 4px;
                    }}
                    QPushButton:hover {{
                        background-color: {Colors.INFO_PANEL_INTERNAL_BG};
                    }}
                """)
                button.setChecked(True)
            else:
                # Unassigned and not selected - default style
                button.setStyleSheet(self.gui._well_button_style())
                button.setChecked(False)


    # -------------------------------------------------------------------------
    # ENTRY POINT
    # -------------------------------------------------------------------------
    def set_merged_data(self, merged_dataframe, csvdict):
        """
        Directly set the merged dataframe and csvdict from the CSV loader.
        This avoids unnecessary disk I/O and ensures data consistency.
        """
        self.logger.info("RECEIVING MERGED DATA DIRECTLY FROM CSV_DATA_LOADER")

        self.merged_dataframe = merged_dataframe
        self.csvdict = csvdict

        # Mark that fresh data has been loaded - will trigger time range reset
        self._fresh_data_loaded = True

        # Clear cached values when new data is loaded
        self._cached_csv_info = None
        self._cached_exp_id = None
        self.detected_inj_timepoints = None

        # Clear legacy well selection dictionaries
        self.select_wells_checkboxes = {}
        self.select_blank_checkboxes = {}
        self.select_posctrl_checkboxes = {}
        self.select_negctrl_checkboxes = {}
        self.select_donorctrl_checkboxes = {}
        self.select_acceptorctrl_checkboxes = {}

        self.logger.info("Cleared cached experiment ID, CSV info, and well selections for new dataset")

        if self.merged_dataframe is not None:
            self.logger.info(f"Received merged dataframe: shape={self.merged_dataframe.shape}")
            self.logger.info(f"Columns: {list(self.merged_dataframe.columns)}")
            self.logger.info(
                f"Data wells: {self.merged_dataframe['Well'].unique().tolist() if 'Well' in self.merged_dataframe.columns else 'N/A'}")
        else:
            self.logger.warning("Received None as merged_dataframe")


    def view_data_window(self):
        """Entry point for creating and displaying the 'View Data' tab."""
        self.logger.info("Initialising View Data tab.")

        # If we don't have data already set directly, load from disk as fallback
        if self.merged_dataframe is None or self.csvdict is None:
            self.logger.info("No data set directly - loading from last run file...")
            self.merged_dataframe, self.csvdict = self.io.load_last_run()
            if self.merged_dataframe is None:
                self.logger.error("Failed to load data; view not built.")
                return
            # Mark as fresh data since we just loaded from disk at program start
            self._fresh_data_loaded = True
        else:
            self.logger.info("Using directly passed merged data (bypassing disk I/O)")

        self.columns = list(self.merged_dataframe.columns[2:])

        # Initialise well selection manager with available wells
        self.well_selection_manager.initialise_wells(self.columns)

        # Check if injection markers have already been removed during merge
        if self.detected_inj_timepoints is None:
            self.logger.info("Checking for injection markers in merged data...")
            self.merged_dataframe, self.detected_inj_timepoints = dpu.remove_injection_marker(self.merged_dataframe)
            if self.detected_inj_timepoints:
                self.logger.info(f"Removed injection markers at timepoints: {self.detected_inj_timepoints}")
        else:
            self.logger.info("Injection markers already processed - skipping")

        # Ensure the view_data_layout exists on master
        if not hasattr(self.master, 'view_data_layout'):
            self.logger.warning("view_data_layout doesn't exist - creating it now")
            from PyQt5.QtWidgets import QVBoxLayout
            # Create the layout on the master's view_data_tab
            if hasattr(self.master, 'view_data_tab'):
                self.master.view_data_layout = QVBoxLayout(self.master.view_data_tab)
                self.master.view_data_layout.setContentsMargins(0, 0, 0, 0)
            else:
                self.logger.error("Neither view_data_layout nor view_data_tab exists on master!")
                return

        # Clear existing layout before adding new content
        if self.master.view_data_layout.count() > 0:
            self.logger.info("Clearing existing View Data tab content.")
            self.clear_layout(self.master.view_data_layout)

        layout_widget = self.gui.build_main_view(self.master)
        self.master.view_data_layout.addWidget(layout_widget)

        # Load settings first (if any exist)
        self.io.load_settings()

        # Initialise time spinboxes with data range (block signals to avoid triggering plot updates)
        if self.merged_dataframe is not None and not self.merged_dataframe.empty:
            min_time_value = self.merged_dataframe['Time [min]'].min()
            max_time_value = self.merged_dataframe['Time [min]'].max()

            # Block signals with try-finally to ensure they're always unblocked
            try:
                self.start_time_spinbox.blockSignals(True)
                self.end_time_spinbox.blockSignals(True)

                # Set minimum based on data, but keep maximum very high to allow manual entry
                # This allows users to type any value without being constrained to data range
                self.start_time_spinbox.setRange(min_time_value, 1e9)
                self.end_time_spinbox.setRange(min_time_value, 1e9)

                # Reset to full extent if:
                # 1. Fresh data was just loaded (new dataset or program start)
                # 2. Current values are out of bounds (legacy behaviour for safety)
                if (self._fresh_data_loaded or
                    self.start_time_spinbox.value() < min_time_value or
                    self.end_time_spinbox.value() > max_time_value):

                    self.start_time_spinbox.setValue(min_time_value)
                    self.end_time_spinbox.setValue(max_time_value)
                    self.logger.info(f"Reset time range to full data extent: {min_time_value:.2f} - {max_time_value:.2f}")

                    # Clear the flag after resetting
                    self._fresh_data_loaded = False
                else:
                    # Preserve existing time range when switching tabs
                    self.logger.info(f"Preserved user time range: {self.start_time_spinbox.value():.2f} - {self.end_time_spinbox.value():.2f}")

            finally:
                self.start_time_spinbox.blockSignals(False)
                self.end_time_spinbox.blockSignals(False)

        self.update_plot()
        self.update_status_frame()

    # -------------------------------------------------------------------------
    # STATUS AND STATE
    # -------------------------------------------------------------------------
    def update_status_frame(self):
        """Update status display with file information and well assignments."""
        # Prevent recursive calls from ElidedLabel resize events
        if self._inside_status_update:
            self.logger.debug("Preventing recursive call to update_status_frame")
            return

        try:
            self._inside_status_update = True

            # Use cached CSV info and experiment ID to avoid repeated computation
            if self._cached_csv_info is None:
                self._cached_csv_info = self.get_csv_info()
            if self._cached_exp_id is None:
                self._cached_exp_id = self.io._get_experiment_identifier()

            # Prepare file information dictionary
            file_info = {
                'filename': self._get_filename_from_csvdict(),
                'experiment_id': self._cached_exp_id,
            }

            # Add data dimensions if available
            if self.merged_dataframe is not None and not self.merged_dataframe.empty:
                file_info['rows'] = len(self.merged_dataframe)
                file_info['columns'] = len(self.merged_dataframe.columns) - 2  # Exclude Time and Well

                # Add time range
                min_time = self.merged_dataframe['Time [min]'].min()
                max_time = self.merged_dataframe['Time [min]'].max()
                display_unit = self.display_unit_combo.currentText() if hasattr(self, 'display_unit_combo') else 'Minutes'
                file_info['time_range'] = f"{min_time:.2f} - {max_time:.2f} {display_unit.lower()}"

                # Add channel information
                if hasattr(self, 'selected_channels_checkboxes'):
                    active_channels = [ch for ch, lbl in self.selected_channels_checkboxes.items()
                                       if getattr(lbl, '_selected', False)]
                    if active_channels:
                        channel_names = [self.channel_map.get(ch, ch) for ch in active_channels]
                        file_info['channels'] = ', '.join(channel_names[:3])
                        if len(channel_names) > 3:
                            file_info['channels'] += f' +{len(channel_names) - 3} more'

            # Update file information display via GUI
            self.gui.update_file_info_display(file_info)

            # Prepare well assignments dictionary
            wells_info = {}

            # Get wells from each category via well selection manager
            data_wells = self.well_selection_manager.get_wells_by_category('data')
            if data_wells:
                wells_info['data'] = ', '.join(sorted(data_wells))

            blank_wells = self.well_selection_manager.get_wells_by_category('blank')
            if blank_wells:
                wells_info['blank'] = ', '.join(sorted(blank_wells))

            pos_ctrl_wells = self.well_selection_manager.get_wells_by_category('pos_ctrl')
            if pos_ctrl_wells:
                wells_info['pos_ctrl'] = ', '.join(sorted(pos_ctrl_wells))

            donor_ctrl_wells = self.well_selection_manager.get_wells_by_category('donor_ctrl')
            if donor_ctrl_wells:
                wells_info['donor_ctrl'] = ', '.join(sorted(donor_ctrl_wells))

            acceptor_ctrl_wells = self.well_selection_manager.get_wells_by_category('acceptor_ctrl')
            if acceptor_ctrl_wells:
                wells_info['acceptor_ctrl'] = ', '.join(sorted(acceptor_ctrl_wells))

            blocked_ctrl_wells = self.well_selection_manager.get_wells_by_category('blocked_ctrl')
            if blocked_ctrl_wells:
                wells_info['blocked_ctrl'] = ', '.join(sorted(blocked_ctrl_wells))

            # Update well assignments display via GUI
            self.gui.update_wells_info_display(wells_info)

        finally:
            self._inside_status_update = False


    def _get_filename_from_csvdict(self):
        """Extract filename from csvdict."""
        if not getattr(self, "csvdict", None):
            return "No file loaded"

        if self.csvdict:
            first_file = next(iter(self.csvdict.keys()))
            return first_file

        return "Unknown file"


    def get_start_end_times(self):
        """
        Retrieve the start and end times from the user interface.
        The times are returned in minutes, with conversion from seconds if necessary.
        """
        if not hasattr(self, 'start_time_spinbox') or not hasattr(self, 'end_time_spinbox'):
            self.logger.warning("Time spinboxes not initialised, returning default values.")
            return 0, 100

        start = self.start_time_spinbox.value()
        end = self.end_time_spinbox.value()

        if hasattr(self, 'original_unit_combo'):
            original_unit = self.original_unit_combo.currentText()
            if original_unit == "Seconds":
                start /= 60
                end /= 60

        return start, end

    def update_well_button_colours(self):
        """Update the colour of well buttons based on their assigned category."""
        # Clean up any deleted buttons first
        self._cleanup_deleted_buttons()

        for well_name, button in list(self.well_buttons.items()):
            # Safety check - skip if button was deleted
            if not self._is_valid_widget(button):
                continue

            try:
                if well_name in self.select_blank_checkboxes:
                    button.setStyleSheet("background-color: red; color: white;")
                    button.setChecked(True)
                elif well_name in self.select_negctrl_checkboxes:
                    button.setStyleSheet("background-color: black; color: white;")
                    button.setChecked(True)
                elif well_name in self.select_posctrl_checkboxes:
                    button.setStyleSheet("background-color: blue; color: white;")
                    button.setChecked(True)
                elif well_name in self.select_donorctrl_checkboxes:
                    button.setStyleSheet("background-color: purple; color: white;")
                    button.setChecked(True)
                elif well_name in self.select_acceptorctrl_checkboxes:
                    button.setStyleSheet("background-color: orange; color: black;")
                    button.setChecked(True)
                elif well_name in self.select_wells_checkboxes:
                    button.setStyleSheet("background-color: #8BC34A; color: black;")
                    button.setChecked(True)
                else:
                    button.setStyleSheet(self.gui._well_button_style())
                    button.setChecked(False)
            except RuntimeError:
                # Button was deleted during iteration
                self.logger.warning(f"Button for well {well_name} was deleted during colour update")
                continue

    # -------------------------------------------------------------------------
    # WELL SELECTION LOGIC
    # -------------------------------------------------------------------------
    def _get_well_category_colour(self, well):
        """Determine the colour for a well based on its category assignment."""
        if well in self.select_blank_checkboxes:
            return "#F44336"  # Red
        elif well in self.select_negctrl_checkboxes:
            return "#000000"  # Black
        elif well in self.select_posctrl_checkboxes:
            return "#2196F3"  # Blue
        elif well in self.select_donorctrl_checkboxes:
            return "#9C27B0"  # Purple
        elif well in self.select_acceptorctrl_checkboxes:
            return "#FF9800"  # Orange
        elif well in self.select_wells_checkboxes:
            return "#4CAF50"  # Green (data wells)
        else:
            return "#8BC34A"  # Default light green (selected but not categorized)

    def _darken_colour(self, hex_colour):
        """Darken a hex colour by 20% for border styling."""
        hex_colour = hex_colour.lstrip('#')
        r, g, b = tuple(int(hex_colour[i:i+2], 16) for i in (0, 2, 4))
        r, g, b = int(r * 0.7), int(g * 0.7), int(b * 0.7)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _is_valid_widget(self, widget):
        """Check if a Qt widget is still valid (not deleted)."""
        try:
            widget.isVisible()
            return True
        except RuntimeError:
            return False

    # -------------------------------------------------------------------------
    # WELL SELECTION INTERFACE (simplified - delegates to manager)
    # -------------------------------------------------------------------------
    def toggle_well_selection(self, well_id):
        """Toggle selection of a single well."""
        self.well_selection_manager.select_well(well_id)

    def toggle_all_wells(self):
        """Toggle selection of all unassigned wells."""
        self.well_selection_manager.toggle_all_selection()

    def select_all_wells(self):
        """Select all wells."""
        self.well_selection_manager.select_all()

    def deselect_all_wells(self):
        """Deselect all wells (keeps assignments)."""
        self.well_selection_manager.deselect_all()

    def clear_all_assignments(self):
        """Clear all well assignments."""
        self.well_selection_manager.clear_all_assignments()

    def assign_wells_to_category(self, category):
        """Assign selected wells to a category."""
        self.well_selection_manager.assign_selected(category)

    # Legacy method names for backward compatibility
    def add_data_wells_on_click(self):
        """Assign selected wells as data wells."""
        self.assign_wells_to_category('data')

    def add_blank_wells_on_click(self):
        """Assign selected wells as blanks (buffer only)."""
        self.assign_wells_to_category('blank')

    def add_posctrl_wells_on_click(self):
        """Assign selected wells as positive controls."""
        self.assign_wells_to_category('pos_ctrl')

    def add_negctrl_wells_on_click(self):
        """Assign selected wells as negative controls."""
        self.assign_wells_to_category('neg_ctrl')

    def add_clear_wells_on_click(self):
        """Clear assignment of selected wells (unassign them)."""
        self.well_selection_manager.clear_selected_assignments()

    def add_donorctrl_wells_on_click(self):
        """Assign selected wells as donor controls."""
        self.assign_wells_to_category('donor_ctrl')

    def add_acceptorctrl_wells_on_click(self):
        """Assign selected wells as acceptor controls."""
        self.assign_wells_to_category('acceptor_ctrl')

    def add_blockedctrl_wells_on_click(self):
        """Assign selected wells as blocked controls (B1-L quenched acceptor)."""
        self.assign_wells_to_category('blocked_ctrl')

    # -------------------------------------------------------------------------
    # PLOTTING AND DISPLAY
    # -------------------------------------------------------------------------
    def update_plot(self):
        """Refresh the main plot based on current selections."""
        self.plotutils.update_plot()

    def plot_in_viewdataframe_action(self, channel, label):
        """Called when a channel label is clicked."""
        self.logger.info(f"Channel {channel} toggled in view.")
        self.update_plot()
        self.save_settings()  # Save after channel toggle

    # -------------------------------------------------------------------------
    # TOGGLES
    # -------------------------------------------------------------------------
    def toggle_scatter(self):
        self.scatter_state = not self.scatter_state
        self.logger.info(f"Scatter mode: {self.scatter_state}")
        self.update_plot()

    def toggle_black_white(self):
        self.black_white_state = not self.black_white_state
        self.logger.info(f"Black/White mode: {self.black_white_state}")
        self.update_plot()

    def toggle_grid(self):
        self.grid_state = not self.grid_state
        self.logger.info(f"Grid mode: {self.grid_state}")
        self.update_plot()

    def toggle_legend(self):
        self.legend_state = not self.legend_state
        self.update_plot()
        self.save_settings()

    # -------------------------------------------------------------------------
    # TIME RANGE PRESETS
    # -------------------------------------------------------------------------
    def preset_full_range(self):
        """Set time range to full data extent."""
        if self.merged_dataframe is not None and not self.merged_dataframe.empty:
            min_time = self.merged_dataframe['Time [min]'].min()
            max_time = self.merged_dataframe['Time [min]'].max()
            self.start_time_spinbox.setValue(min_time)
            self.end_time_spinbox.setValue(max_time)
            self.logger.info(f"Set time range to full extent: {min_time:.2f} - {max_time:.2f} min")

    def preset_first_30min(self):
        """Set time range to first 30 minutes."""
        if self.merged_dataframe is not None and not self.merged_dataframe.empty:
            min_time = self.merged_dataframe['Time [min]'].min()
            self.start_time_spinbox.setValue(min_time)
            self.end_time_spinbox.setValue(min_time + 30.0)
            self.logger.info(f"Set time range to first 30 minutes: {min_time:.2f} - {min_time + 30.0:.2f} min")

    def preset_first_hour(self):
        """Set time range to first hour."""
        if self.merged_dataframe is not None and not self.merged_dataframe.empty:
            min_time = self.merged_dataframe['Time [min]'].min()
            self.start_time_spinbox.setValue(min_time)
            self.end_time_spinbox.setValue(min_time + 60.0)
            self.logger.info(f"Set time range to first hour: {min_time:.2f} - {min_time + 60.0:.2f} min")

    def preset_last_hour(self):
        """Set time range to last hour."""
        if self.merged_dataframe is not None and not self.merged_dataframe.empty:
            max_time = self.merged_dataframe['Time [min]'].max()
            self.start_time_spinbox.setValue(max(0, max_time - 60.0))
            self.end_time_spinbox.setValue(max_time)
            self.logger.info(f"Set time range to last hour: {max(0, max_time - 60.0):.2f} - {max_time:.2f} min")

    # -------------------------------------------------------------------------
    # IO SHORTCUTS
    # -------------------------------------------------------------------------
    def export_function_view_data(self):
        self.io.export_filtered_data()

    def save_settings(self):
        self.io.save_settings()

    def load_settings(self):
        self.io.load_settings()

    def save_state(self):
        """Save current state including well selections and settings."""
        self.io.save_state()

    # -------------------------------------------------------------------------
    # WRAPPERS FOR PLOT UTILS
    # -------------------------------------------------------------------------
    def extract_wavelength_from_channel(self, channel_name: str):
        return PlotUtils.extract_wavelength(channel_name)

    def get_wavelength_colour(self, wavelength: str):
        return PlotUtils.wavelength_to_colour(wavelength)

    def show_plot_in_window(self):
        """Delegate showing the current plot in a standalone window."""
        self.plotutils.show_plot_in_window()

    def get_csv_info(self):
        """
        Retrieve information about the loaded CSV file.
        Extracts filename, experiment name, and info from csvdict structure.
        """
        self.logger.info("get_csv_info() called")

        if not getattr(self, "csvdict", None):
            self.logger.warning("csvdict is None or empty")
            return "No experiment metadata available."

        self.logger.info(f"csvdict keys: {list(self.csvdict.keys())}")

        # Initialise result parts
        result_lines = []

        # Get first file entry from csvdict (usually there's only one)
        if self.csvdict:
            first_file = next(iter(self.csvdict.keys()))
            result_lines.append(f"<b>File:</b> {first_file}")

            file_data = self.csvdict[first_file]
            if isinstance(file_data, dict):
                # Extract experiment name
                experiment = file_data.get('experiment')
                if experiment and experiment != "N/A":
                    result_lines.append(f"<b>Experiment:</b> {experiment}")

                # Extract info/description
                info = file_data.get('info')
                if info and info != "N/A":
                    # Truncate if too long
                    info_text = str(info)
                    if len(info_text) > 50:
                        info_text = info_text[:47] + "..."
                    result_lines.append(f"<b>Info:</b> {info_text}")

                # Extract timestamp if available
                timestamp = file_data.get('timestamp')
                if timestamp:
                    result_lines.append(f"<b>Date:</b> {timestamp}")

                # Show format type
                minimal = file_data.get('minimal', False)
                format_type = "Minimal" if minimal else "Standard"
                result_lines.append(f"<b>Format:</b> {format_type}")

                # Show number of channels if available
                channels = file_data.get('channels', [])
                if channels:
                    result_lines.append(f"<b>Channels:</b> {len(channels)}")

        if not result_lines:
            return "No experiment metadata available."

        return "<br>".join(result_lines)

    # -------------------------------------------------------------------------
    # STANDARD CURVE AND CONVERSION TABS
    # -------------------------------------------------------------------------
    def create_standard_curve(self):
        """
        Create a new Standard Curve tab from the currently selected wells.
        This removes any existing StandardCurveTab to avoid duplicates.
        """
        from core.data_frame_processor.standard_curve_tab import StandardCurveTab

        self.logger.info("Standard Curve request received.")

        try:
            start_time, end_time = self.get_start_end_times()

            channel_checkboxes = self.selected_channels_checkboxes
            well_checkboxes = self.select_wells_checkboxes
            blank_well = self.select_blank_checkboxes

            # Validate required wells - with user-facing error dialogs
            if not well_checkboxes:
                self.logger.warning("No data wells selected for standard curve.")
                self.show_error_dialog("No data wells are selected.\n\nPlease select at least one data well for the standard curve.")
                return

            if not blank_well:
                self.logger.warning("WARNING: Standard Curve aborted - Reason: Missing Blank wells")
                self.show_error_dialog("No Blank wells are assigned.\n\nPlease assign at least one before creating a standard curve.")
                return

            # Prepare data
            selected_data, blank_data, _, _, _, _ = self.dataops.prepare_dataframes(
                start_time, end_time, channel_checkboxes, well_checkboxes, blank_well, blank_well
            )

            # Validate that data preparation succeeded
            if selected_data is None or selected_data.empty:
                self.logger.error("Selected data is empty after preparation.")
                self.show_error_dialog("Failed to prepare data wells.\n\nPlease check your well selections and time range.")
                return

            if blank_data is None or blank_data.empty:
                self.logger.error("Blank data is empty after preparation.")
                self.show_error_dialog("Failed to prepare blank data.\n\nPlease check your control well selections and time range.")
                return

            # Remove existing tab if present
            existing_tab = None
            for i in range(self.master.count()):
                w = self.master.widget(i)
                if w.__class__.__name__ == "StandardCurveTab":
                    existing_tab = w
                    break
            if existing_tab is not None:
                self.logger.info("Removing existing StandardCurveTab instance.")
                self.master.removeTab(self.master.indexOf(existing_tab))
                existing_tab.deleteLater()

            # Create new
            self.logger.info("Creating new StandardCurveTab instance.")
            standard_curve_tab = StandardCurveTab(self.master, selected_data, blank_data)
            self.master.addTab(standard_curve_tab, "Standard Curve")
            self.master.setCurrentWidget(standard_curve_tab)

        except Exception as e:
            self.logger.exception("Unexpected error during standard curve creation: %s", str(e))
            self.show_error_dialog(f"An unexpected error occurred:\n\n{str(e)}")

    def show_error_dialog(self, message):
        """
        Display a user-facing error dialog with a warning icon.

        Args:
            message (str): The error message to display to the user.
        """
        from PyQt5.QtWidgets import QMessageBox

        dlg = QMessageBox(self.master)
        dlg.setWindowTitle("Conversion Error")
        dlg.setText(message)
        dlg.setIcon(QMessageBox.Warning)
        dlg.exec_()

    def _check_well_assignment_conflicts(self, sample_wells, neg_wells, pos_wells, donor_wells, acceptor_wells):
        """
        Check for wells assigned to multiple categories.

        Args:
            sample_wells: Dict of sample well checkboxes
            neg_wells: Dict of blank well checkboxes
            pos_wells: Dict of positive control well checkboxes
            donor_wells: Dict of donor control well checkboxes
            acceptor_wells: Dict of acceptor control well checkboxes

        Returns:
            List of tuples (well_name, [categories]) for conflicting wells
        """
        from collections import defaultdict

        well_categories = defaultdict(list)

        # Track which category each well is assigned to
        for well_id in sample_wells.keys():
            well_categories[well_id].append("Sample")

        for well_id in neg_wells.keys():
            well_categories[well_id].append("Blank")

        for well_id in pos_wells.keys():
            well_categories[well_id].append("Positive Control")

        for well_id in donor_wells.keys():
            well_categories[well_id].append("Donor Control")

        for well_id in acceptor_wells.keys():
            well_categories[well_id].append("Acceptor Control")

        # Find wells with multiple assignments
        conflicts = []
        for well_id, categories in well_categories.items():
            if len(categories) > 1:
                conflicts.append((well_id, categories))

        # Sort by well ID for consistent display
        conflicts.sort(key=lambda x: x[0])

        return conflicts

    def _format_well_conflicts(self, conflicts):
        """
        Format well conflict information for display.

        Args:
            conflicts: List of tuples (well_name, [categories])

        Returns:
            Formatted string describing the conflicts
        """
        if not conflicts:
            return ""

        lines = ["Wells assigned to multiple categories:"]
        for well_id, categories in conflicts:
            cat_str = " and ".join(f"'{c}'" for c in categories)
            lines.append(f"  • {well_id}: {cat_str}")

        return "\n".join(lines)

    def open_conversion_tab(self):
        """
        Open the data conversion tab using currently selected wells and channels.
        """
        from core.convert_data_tab.gui import ConvertDataTab

        self.logger.info("Conversion tab request received.")
        try:
            start_time, end_time = self.get_start_end_times()

            channel_checkboxes = self.selected_channels_checkboxes
            well_checkboxes = self.select_wells_checkboxes
            neg_well = self.select_blank_checkboxes
            pos_well = self.select_posctrl_checkboxes
            donor_well = getattr(self, "select_donorctrl_checkboxes", {})
            acceptor_well = getattr(self, "select_acceptorctrl_checkboxes", {})
            negctrl_well = getattr(self, "select_negctrl_checkboxes", {})
            blocked_well = getattr(self, "select_blockedctrl_checkboxes", {})

            # Validate required wells - with user-facing error dialogs
            missing_controls = []

            if not neg_well:
                missing_controls.append("Blank")
                self.logger.warning("WARNING: Conversion aborted - Reason: Missing Blank wells")

            if not pos_well:
                missing_controls.append("Positive Control")
                self.logger.warning("WARNING: Conversion aborted - Reason: Missing Positive Control wells")

            # Show combined error message if both are missing
            if len(missing_controls) == 2:
                error_msg = ("Conversion cannot proceed.\n\n"
                           "Missing: Positive Control and Blank wells.\n\n"
                           "Please assign both before conversion.")
                self.show_error_dialog(error_msg)
                return
            elif len(missing_controls) == 1:
                control_type = missing_controls[0]
                error_msg = (f"No {control_type} wells are assigned.\n\n"
                           f"Please assign at least one before conversion.")
                self.show_error_dialog(error_msg)
                return

            if not well_checkboxes:
                self.logger.warning("No data wells selected.")
                self.show_error_dialog("No data wells are selected.\n\nPlease select at least one data well before conversion.")
                return

            # Check for duplicate well assignments across categories
            conflicts = self._check_well_assignment_conflicts(
                well_checkboxes, neg_well, pos_well, donor_well, acceptor_well
            )

            if conflicts:
                conflict_msg = self._format_well_conflicts(conflicts)
                error_msg = (
                    "⚠ Well Assignment Conflicts Detected!\n\n"
                    f"{conflict_msg}\n\n"
                    "Each well should only be assigned to one category.\n"
                    "Please review and fix these conflicts before proceeding."
                )
                self.show_error_dialog(error_msg)
                return

            # Prepare data
            result = self.dataops.prepare_dataframes(
                start_time, end_time, channel_checkboxes, well_checkboxes,
                neg_well, pos_well, donor_well, acceptor_well, negctrl_well, blocked_well
            )

            if not isinstance(result, tuple) or len(result) != 7:
                raise ValueError("Unexpected result structure from prepare_dataframes.")

            selected_data, blank_data, pos_ctrl_data, donor_data, acceptor_data, neg_ctrl_data, blocked_ctrl_data = result
            if any(df is None for df in [selected_data, blank_data, pos_ctrl_data]):
                raise ValueError("Missing one or more required datasets.")

            # Remove existing conversion tab if present
            for i in range(self.master.count()):
                if self.master.tabText(i) == "Convert Data":
                    widget = self.master.widget(i)
                    self.master.removeTab(i)
                    widget.deleteLater()
                    break

            channels = getattr(self.options_processor, "active_channels", {})
            csv_info = self.get_csv_info()
            experiment_id = self.io._get_experiment_identifier()

            self.convert_data_tab = ConvertDataTab(
                parent=self.master,
                selected_data=selected_data,
                blank_ctrl_data=blank_data,
                pos_ctrl_data=pos_ctrl_data,
                donor_data=donor_data,
                acceptor_data=acceptor_data,
                neg_ctrl_data=neg_ctrl_data,
                blocked_ctrl_data=blocked_ctrl_data,
                detected_inj_timepoints=self.detected_inj_timepoints,
                csv_info=csv_info,
                experiment_id=experiment_id,
                donor1_channel=channels.get("Donor 1"),
                acceptor1_channel=channels.get("Acceptor 1"),
                fret1_channel=channels.get("FRET 1"),
                donor2_channel=channels.get("Donor 2"),
                acceptor2_channel=channels.get("Acceptor 2"),
                fret2_channel=channels.get("FRET 2"),
            )

            # Connect the forward to kinetics signal
            self.convert_data_tab.forwarded_to_kinetics.connect(self._on_data_forwarded_to_kinetics)

            # Re-detect channels when the user changes Options → Save and Apply
            self.options_processor.channelsUpdated.connect(
                self.convert_data_tab.refresh_channels
            )

            self.master.addTab(self.convert_data_tab, "Convert Data")
            self.master.setCurrentWidget(self.convert_data_tab)
            self.logger.info("Conversion tab successfully added and displayed.")

        except ValueError as ve:
            self.logger.error("Validation error during conversion: %s", str(ve))
            self.show_error_dialog(f"Validation error:\n\n{str(ve)}")
        except Exception as e:
            self.logger.exception("Unexpected error during conversion tab setup: %s", str(e))
            self.show_error_dialog(f"An unexpected error occurred:\n\n{str(e)}")

    def _on_data_forwarded_to_kinetics(self, df, label, metadata=None):
        """
        Handle data forwarded from Convert Data tab to Kinetics Processor.

        This method is called when the user clicks "Send to Kinetics Processor" in the Convert Data tab.
        It creates or switches to the Kinetics tab and loads the data.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing concentration time-series data
        label : str
            Label describing the dataset (e.g., "B1B2 Concentration")
        metadata : dict, optional
            Metadata from conversion, may contain:
            - per_trace_Z0: dict mapping trace names to initial concentrations (nM)
            - mode: conversion mode (e.g., 'HMSD')
            - species: species name
        """
        self.logger.info(f"Received data forward request: {label}")
        if metadata:
            self.logger.info(f"Metadata included: {list(metadata.keys())}")

        try:
            # Check if we have access to the main application window
            if not hasattr(self, 'master') or self.master is None:
                self.logger.error("Cannot access main application window")
                return

            # Find or create Kinetics tab
            kinetics_index = -1
            kinetics_widget = None

            # Search for existing Kinetics tab
            for i in range(self.master.count()):
                if "Kinetics" in self.master.tabText(i):
                    kinetics_index = i
                    kinetics_widget = self.master.widget(i)
                    self.logger.info(f"Found existing Kinetics tab at index {i}")
                    break

            # If no Kinetics tab exists, we need to trigger creation via the main app
            if kinetics_index == -1:
                self.logger.info("No Kinetics tab found, attempting to create one...")

                # Try to access the main application and trigger kinetics tab creation
                # The master is the QTabWidget, so we need to go up to MainApplication
                main_app = self.master
                if hasattr(main_app, 'switch_to_kinetics_tab'):
                    main_app.switch_to_kinetics_tab()

                    # After creation, find it again
                    for i in range(self.master.count()):
                        if "Kinetics" in self.master.tabText(i):
                            kinetics_index = i
                            kinetics_widget = self.master.widget(i)
                            break

            # Get the KineticsProcessor instance
            kinetics_processor = None
            if kinetics_widget is not None:
                # The widget might be the processor itself or contain it
                if hasattr(kinetics_widget, 'kineticproc'):
                    kinetics_processor = kinetics_widget.kineticproc
                elif hasattr(kinetics_widget, 'layout'):
                    # Try to find KineticsProcessor in the layout
                    layout = kinetics_widget.layout()
                    if layout and layout.count() > 0:
                        potential_proc = layout.itemAt(0).widget()
                        if hasattr(potential_proc, 'load_dataframe_directly'):
                            kinetics_processor = potential_proc

            # Also check if master has direct reference
            if kinetics_processor is None and hasattr(self.master, 'kineticproc'):
                kinetics_processor = self.master.kineticproc

            if kinetics_processor is None:
                self.logger.error("Could not find KineticsProcessor instance")
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.warning(
                    self.master,
                    "Error",
                    "Could not access Kinetics Processor. Please try opening the Kinetics tab manually first."
                )
                return

            # Load the data into the kinetics processor
            if hasattr(kinetics_processor, 'load_dataframe_directly'):
                kinetics_processor.load_dataframe_directly(df, label, metadata=metadata)
                self.logger.info(f"Successfully loaded data into Kinetics Processor: {label}")

                # Switch to the kinetics tab
                if kinetics_index >= 0:
                    self.master.setCurrentIndex(kinetics_index)
            else:
                self.logger.error("KineticsProcessor does not have load_dataframe_directly method")
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.warning(
                    self.master,
                    "Error",
                    "Kinetics Processor does not support direct data loading. Please save the data and load it manually."
                )

        except Exception as e:
            self.logger.exception(f"Error forwarding data to Kinetics: {e}")
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(
                self.master,
                "Error",
                f"Failed to forward data to Kinetics Processor:\n\n{str(e)}"
            )

    # -------------------------------------------------------------------------
    # REINITIALISATION AND LAYOUT MANAGEMENT
    # -------------------------------------------------------------------------
    def clear_layout(self, layout):
        """
        Remove all widgets from a given layout.
        This is useful for refreshing parts of the user interface.
        """
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

    def reinitialise(self):
        """
        Reset the internal state of the processor and reload the 'View Data' tab.
        This is useful when new data is loaded or when the user wants to start over.
        """
        self.logger.info("Reinitialising internal state.")
        self.merged_dataframe = None
        self.csvdict = None
        self.columns = []
        self.selected_wells = []
        self.select_wells_checkboxes = {}
        self.select_blank_checkboxes = {}
        self.select_posctrl_checkboxes = {}
        self.select_donorctrl_checkboxes = {}
        self.select_acceptorctrl_checkboxes = {}
        self.selected_channels_checkboxes = {}
        self.scatter_state = False
        self.detected_inj_timepoints = None

        self.logger.info("Reinitialising 'View Data' tab.")
        if hasattr(self.master, 'view_data_layout') and self.master.view_data_layout.count() > 0:
            self.clear_layout(self.master.view_data_layout)
            self.view_data_window()
        else:
            self.logger.info("'View Data' tab is not yet populated; skipping reinitialisation.")

    def select_all(self):
        """Select or deselect all well buttons."""
        # Clean up any deleted buttons first
        self._cleanup_deleted_buttons()

        if not self.well_buttons:
            self.logger.warning("No well buttons available.")
            return

        # Safely check if any buttons are selected
        any_selected = False
        try:
            any_selected = any(btn.isChecked() for btn in self.well_buttons.values() if self._is_valid_widget(btn))
        except RuntimeError:
            self.logger.warning("Buttons were deleted, cleaning up.")
            self._cleanup_deleted_buttons()
            return

        if any_selected:
            # Deselect all
            for well_name, btn in list(self.well_buttons.items()):
                if self._is_valid_widget(btn):
                    btn.setChecked(False)
                    btn.setStyleSheet(self.gui._well_button_style())
            self.selected_wells = []
        else:
            # Select all
            for well_name, btn in list(self.well_buttons.items()):
                if self._is_valid_widget(btn):
                    btn.setChecked(True)
                    if well_name not in self.selected_wells:
                        self.selected_wells.append(well_name)
                    # Apply appropriate colour based on category
                    category_colour = self._get_well_category_colour(well_name)
                    btn.setStyleSheet(f"background-color: {category_colour}; border: 2px solid {self._darken_colour(category_colour)}; border-radius: 4px;")
        self.logger.info("'Select All' triggered.")
        self.update_status_frame()


    def _cleanup_deleted_buttons(self):
        """Remove references to deleted buttons from well_buttons dictionary."""
        deleted_wells = []
        for well_name, btn in list(self.well_buttons.items()):
            if not self._is_valid_widget(btn):
                deleted_wells.append(well_name)

        for well_name in deleted_wells:
            self.well_buttons.pop(well_name, None)
            if well_name in self.selected_wells:
                self.selected_wells.remove(well_name)
    def _assign_to_category(self, wells, category_attr, colour):
        """Generic internal function to assign selected wells to a category."""
        if not wells:
            self.logger.warning(f"No wells selected to assign to {category_attr}")
            return

        # Colour mapping for consistent styling
        colour_map = {
            "green": "#4CAF50",
            "red": "#F44336",
            "blue": "#2196F3",
            "purple": "#9C27B0",
            "orange": "#FF9800"
        }

        hex_colour = colour_map.get(colour, colour)

        # Remove wells from other categories first (except for data wells)
        for well in wells:
            if category_attr != "select_wells_checkboxes":
                self.select_wells_checkboxes.pop(well, None)

        # Assign to new category
        setattr(self, category_attr, {well: True for well in wells})

        # Update button colours immediately
        for well in wells:
            btn = self.well_buttons.get(well)
            if btn:
                btn.setChecked(True)
                btn.setStyleSheet(f"background-color: {hex_colour}; border: 2px solid {self._darken_colour(hex_colour)}; border-radius: 4px;")
                if well not in self.selected_wells:
                    self.selected_wells.append(well)

        self.update_status_frame()
        self.update_plot()
