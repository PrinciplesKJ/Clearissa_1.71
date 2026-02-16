"""
State Management Module for Kinetics Processor
-----------------------------------------------
Handles saving and restoring processor state for session persistence.

Author: Krizan Jurinovic
Date: November 2025
"""

import logging
from PyQt5.QtCore import Qt, QTimer

logger = logging.getLogger(__name__)


class StateManager:
    """
    Manages state persistence for the kinetics processor.

    Uses debounced auto-save to reduce disk I/O whilst maintaining
    reliable session recovery.
    """

    # Auto-save delay in milliseconds
    # State is saved 2 seconds after last change (debounced)
    AUTOSAVE_DELAY_MS = 2000

    def __init__(self, parent_widget, io_utils):
        """
        Initialise the state manager.

        Parameters
        ----------
        parent_widget : KineticsProcessor
            Parent processor widget
        io_utils : KineticsIOUtils
            I/O utilities handler
        """
        self.parent = parent_widget
        self.io_utils = io_utils

        # Debounced auto-save timer
        self._save_timer = QTimer()
        self._save_timer.setSingleShot(True)
        self._save_timer.timeout.connect(self._perform_save)

        # Track if state needs saving
        self._state_dirty = False

    def _get_per_trace_T_values(self):
        """Get per-trace T values from the ODE widget if available."""
        if hasattr(self.parent, 'ode_widget') and self.parent.ode_widget:
            if hasattr(self.parent.ode_widget, 'get_all_per_trace_T_values'):
                return self.parent.ode_widget.get_all_per_trace_T_values()
        return {}

    def _set_per_trace_T_values(self, values):
        """Set per-trace T values on the ODE widget if available."""
        if hasattr(self.parent, 'ode_widget') and self.parent.ode_widget:
            if hasattr(self.parent.ode_widget, 'set_last_per_trace_T_values'):
                self.parent.ode_widget.set_last_per_trace_T_values(values)

    def _get_per_trace_Z0_values(self):
        """Get per-trace Z0 values from the bimolecular widget if available."""
        if hasattr(self.parent, 'bimolecular_widget') and self.parent.bimolecular_widget:
            if hasattr(self.parent.bimolecular_widget, 'get_all_per_trace_Z0_values'):
                return self.parent.bimolecular_widget.get_all_per_trace_Z0_values()
        return {}

    def _set_per_trace_Z0_values(self, values):
        """Set per-trace Z0 values on the bimolecular widget if available."""
        if hasattr(self.parent, 'bimolecular_widget') and self.parent.bimolecular_widget:
            if hasattr(self.parent.bimolecular_widget, 'set_last_per_trace_Z0_values'):
                self.parent.bimolecular_widget.set_last_per_trace_Z0_values(values)

    def _restore_exclude_checkbox_states(self):
        """
        Restore exclude-from-fit checkbox states from trace_settings.

        This method reads the exclude_from_fit flag from trace_settings
        and updates the checkbox state accordingly.
        """
        if not self.parent.trace_selection_panel:
            return

        manager = self.parent.trace_selection_panel.get('manager')
        if not manager or not hasattr(manager, 'exclude_checkboxes'):
            return

        trace_settings = getattr(self.parent, 'trace_settings', {})
        if not trace_settings:
            return

        restored_count = 0
        for checkbox, col_name in manager.exclude_checkboxes:
            # Get the exclude_from_fit setting for this trace
            is_excluded = trace_settings.get(col_name, {}).get('exclude_from_fit', False)
            checkbox.blockSignals(True)
            checkbox.setChecked(is_excluded)
            checkbox.blockSignals(False)
            if is_excluded:
                restored_count += 1

        if restored_count > 0:
            logger.info("Restored exclude-from-fit state for %d traces", restored_count)

    def save_state(self):
        """
        Schedule a debounced state save (2 s after last call).

        For immediate save (e.g. on application close), use save_state_now().
        """
        self._state_dirty = True

        # Restart timer (extends delay if called repeatedly)
        if self._save_timer.isActive():
            self._save_timer.stop()

        self._save_timer.start(self.AUTOSAVE_DELAY_MS)
        logger.debug("State save scheduled in %d ms", self.AUTOSAVE_DELAY_MS)

    def save_state_now(self):
        """
        Save state immediately (bypass debouncing).

        Use this when:
        - Application is closing
        - User explicitly saves
        - Critical state change (e.g., data loaded)
        """
        logger.debug("Immediate state save requested")

        # Cancel any pending debounced save
        if self._save_timer.isActive():
            self._save_timer.stop()

        self._perform_save()

    def _perform_save(self):
        """
        Perform actual state save to disk.

        This is the internal implementation called by the debounce timer
        or immediate save method. Do not call directly; use save_state()
        or save_state_now() instead.
        """
        if not self._state_dirty:
            logger.debug("State not dirty, skipping save")
            return

        try:
            logger.info("Performing state save to disk")

            # Capture currently visible traces from the trace selection panel
            visible_traces = []
            if self.parent.trace_selection_panel and 'manager' in self.parent.trace_selection_panel:
                manager = self.parent.trace_selection_panel['manager']
                visible_traces = manager.get_visible_traces()

            state_dict = {
                # Data references
                'data_df': self.parent.data_df,
                'fitted_df': self.parent.fitted_df,
                'time_col': self.parent.time_col,
                'filename': self.parent.filename,

                # UI State - Mode selection
                'tmsd_mode': self.parent.bimolecular_radio.isChecked() if self.parent.bimolecular_radio else True,
                'catalytic_mode': self.parent.catalytic_radio.isChecked() if self.parent.catalytic_radio else False,

                # UI State - TMSD parameters
                'initial_I': self.parent.initial_I_entry.value() if self.parent.initial_I_entry else 0.0,
                'initial_SN': self.parent.initial_SN_entry.value() if self.parent.initial_SN_entry else 0.0,
                'initial_guess': self.parent.initial_guess_entry.value() if self.parent.initial_guess_entry else 0.0,

                # UI State - Catalytic turnover parameters
                'catalytic_template_T': self.parent.catalytic_template_T_spinbox.value() if self.parent.catalytic_template_T_spinbox else 1.0,
                'fluorescence_full_scale_nM': self.parent.fluorescence_full_scale_nM_spinbox.value() if self.parent.fluorescence_full_scale_nM_spinbox else 10.0,
                'catalytic_S10_guess': self.parent.catalytic_S10_guess_spinbox.value() if self.parent.catalytic_S10_guess_spinbox else 10.0,
                'catalytic_k_guess': self.parent.catalytic_k_guess_spinbox.value() if self.parent.catalytic_k_guess_spinbox else 1.0,
                'catalytic_K_guess': self.parent.catalytic_K_guess_spinbox.value() if self.parent.catalytic_K_guess_spinbox else 10.0,
                # UI State - Catalytic bounds
                'catalytic_k_lower': self.parent.catalytic_k_lower_entry.value() if getattr(self.parent, 'catalytic_k_lower_entry', None) else 1e-6,
                'catalytic_k_upper': self.parent.catalytic_k_upper_entry.value() if getattr(self.parent, 'catalytic_k_upper_entry', None) else 100.0,
                'catalytic_K_lower': self.parent.catalytic_K_lower_entry.value() if getattr(self.parent, 'catalytic_K_lower_entry', None) else 0.1,
                'catalytic_K_upper': self.parent.catalytic_K_upper_entry.value() if getattr(self.parent, 'catalytic_K_upper_entry', None) else 1000.0,
                # UI State - Catalytic sub-model type
                'catalytic_sub_model': self.parent.ode_widget.get_catalytic_sub_model() if hasattr(getattr(self.parent, 'ode_widget', None), 'get_catalytic_sub_model') else 'full',

                # UI State - Per-trace template concentrations
                'per_trace_T_values': self._get_per_trace_T_values(),

                # UI State - Per-trace Z0 values (bimolecular fitting)
                'per_trace_Z0_values': self._get_per_trace_Z0_values(),


                # UI State - Time window
                'start_time': self.parent.start_time_spinbox.value() if self.parent.start_time_spinbox else 0.0,
                'end_time': self.parent.end_time_spinbox.value() if self.parent.end_time_spinbox else 60.0,

                # UI State - Results settings
                'r2_threshold': self.parent.r2_threshold_spinbox.value() if self.parent.r2_threshold_spinbox else 0.90,

                # Experiment metadata
                'experiment_title': getattr(self.parent, 'experiment_title', ''),
                'experiment_info': getattr(self.parent, 'experiment_info', ''),
                'last_mean_rate_constant': getattr(self.parent, '_last_mean_rate_constant', (0, 0)),

                # Trace settings and visibility
                'trace_settings': getattr(self.parent, 'trace_settings', {}),
                'visible_traces': visible_traces,
                'show_legend': getattr(self.parent, 'show_legend', True),
                'replicate_info': getattr(self.parent, 'replicate_info', {}),

                # Fitted parameters storage
                'fitted_parameters': getattr(self.parent, 'fitted_parameters', {}),

                # Plot state
                'scatter_state': getattr(self.parent, 'scatter_state', False),
                'grid_state': getattr(self.parent, 'grid_state', False),

                # Detected endpoints
                'detected_endpoints': getattr(self.parent, 'detected_endpoints', {}),

                # Dataset hash for current session
                'dataset_hash': getattr(self.parent.io_utils, 'current_dataset_hash', None),

                # Hash-to-groups mapping for preserving groups across datasets
                # Format: {dataset_hash: {user_defined_groups_dict}}
                'replicate_groups_by_hash': getattr(self.parent, '_replicate_groups_by_hash', {}),
            }

            self.io_utils.save_state(state_dict)

            self._state_dirty = False
            logger.info("State saved successfully")

        except Exception as e:
            logger.error("Failed to save kinetics processor state: %s", e, exc_info=True)

    def restore_state(self):
        """Restore the last saved session (data, parameters, traces, fits)."""
        try:
            logger.info("Attempting to auto-restore kinetics processor state")

            state_dict = self.io_utils.load_state()
            if state_dict is None:
                logger.info("No saved state found - starting with defaults")
                return

            # Check if we have the same dataset as before
            saved_dataset_hash = state_dict.get('dataset_hash', None)
            current_dataset_hash = getattr(self.parent.io_utils, 'current_dataset_hash', None)

            # CRITICAL FIX: If io_utils doesn't have a hash yet (restored from saved state,
            # not loaded from file), use the saved hash and set it on io_utils.
            # This ensures group lookup works correctly when restoring a session.
            if current_dataset_hash is None and saved_dataset_hash is not None:
                current_dataset_hash = saved_dataset_hash
                self.parent.io_utils.current_dataset_hash = saved_dataset_hash
                logger.info("Restored dataset hash from saved state: %s...", saved_dataset_hash[:16])

            # Load the hash-to-groups mapping
            self.parent._replicate_groups_by_hash = state_dict.get('replicate_groups_by_hash', {})

            # Check if we have stored groups for the current dataset
            groups_available_for_current = (current_dataset_hash and
                                           current_dataset_hash in self.parent._replicate_groups_by_hash)

            # Determine if this is the same dataset
            same_dataset = (saved_dataset_hash is not None and
                           current_dataset_hash is not None and
                           saved_dataset_hash == current_dataset_hash)

            if saved_dataset_hash and current_dataset_hash:
                if same_dataset:
                    logger.info("Same dataset detected (hash: %s...) - restoring fits and groups",
                               current_dataset_hash[:16])
                else:
                    logger.info("Different dataset detected - saved hash: %s..., current hash: %s... - clearing fits",
                               saved_dataset_hash[:16], current_dataset_hash[:16])
                    if groups_available_for_current:
                        logger.info("Stored replicate groups found for this dataset (hash: %s...)",
                                   current_dataset_hash[:16])
            else:
                logger.info("No dataset hash available for comparison - treating as new session")

            # Restore data (always restore data_df as it may not be loaded yet)
            self.parent.data_df = state_dict.get('data_df')

            # Only restore fitted_df if it's the same dataset
            if same_dataset:
                self.parent.fitted_df = state_dict.get('fitted_df')
                logger.info("Restored fitted data from previous session")
            else:
                self.parent.fitted_df = None
                logger.info("Cleared fitted data (new dataset loaded)")

            self.parent.time_col = state_dict.get('time_col', 'Time')
            self.parent.filename = state_dict.get('filename', '')

            # Only restore replicate_info if it's the same dataset
            if same_dataset:
                self.parent.replicate_info = state_dict.get('replicate_info', {})
                if self.parent.replicate_info:
                    logger.info("Restored replicate info from previous session")
            else:
                self.parent.replicate_info = {}
                logger.info("Cleared replicate info (new dataset loaded)")

            # Restore mode selection
            tmsd_mode = state_dict.get('tmsd_mode', True)
            catalytic_mode = state_dict.get('catalytic_mode', False)

            if tmsd_mode and self.parent.bimolecular_radio:
                self.parent.bimolecular_radio.setChecked(True)
                self.parent.stacked_widget.setCurrentIndex(0)
            elif catalytic_mode and self.parent.catalytic_radio:
                self.parent.catalytic_radio.setChecked(True)
                self.parent.stacked_widget.setCurrentIndex(1)

            # Restore TMSD parameters
            if self.parent.initial_I_entry and 'initial_I' in state_dict:
                self.parent.initial_I_entry.setValue(state_dict['initial_I'])
            if self.parent.initial_SN_entry and 'initial_SN' in state_dict:
                self.parent.initial_SN_entry.setValue(state_dict['initial_SN'])
            if self.parent.initial_guess_entry and 'initial_guess' in state_dict:
                self.parent.initial_guess_entry.setValue(state_dict['initial_guess'])

            # Restore ODE parameters (Catalytic turnover)
            if self.parent.catalytic_template_T_spinbox and 'catalytic_template_T' in state_dict:
                self.parent.catalytic_template_T_spinbox.setValue(state_dict['catalytic_template_T'])
            if self.parent.fluorescence_full_scale_nM_spinbox and 'fluorescence_full_scale_nM' in state_dict:
                self.parent.fluorescence_full_scale_nM_spinbox.setValue(state_dict['fluorescence_full_scale_nM'])
            if self.parent.catalytic_S10_guess_spinbox and 'catalytic_S10_guess' in state_dict:
                self.parent.catalytic_S10_guess_spinbox.setValue(state_dict['catalytic_S10_guess'])
            if self.parent.catalytic_k_guess_spinbox and 'catalytic_k_guess' in state_dict:
                self.parent.catalytic_k_guess_spinbox.setValue(state_dict['catalytic_k_guess'])
            if self.parent.catalytic_K_guess_spinbox and 'catalytic_K_guess' in state_dict:
                self.parent.catalytic_K_guess_spinbox.setValue(state_dict['catalytic_K_guess'])

            # Restore catalytic bounds
            if getattr(self.parent, 'catalytic_k_lower_entry', None) and 'catalytic_k_lower' in state_dict:
                self.parent.catalytic_k_lower_entry.setValue(state_dict['catalytic_k_lower'])
            if getattr(self.parent, 'catalytic_k_upper_entry', None) and 'catalytic_k_upper' in state_dict:
                self.parent.catalytic_k_upper_entry.setValue(state_dict['catalytic_k_upper'])
            if getattr(self.parent, 'catalytic_K_lower_entry', None) and 'catalytic_K_lower' in state_dict:
                self.parent.catalytic_K_lower_entry.setValue(state_dict['catalytic_K_lower'])
            if getattr(self.parent, 'catalytic_K_upper_entry', None) and 'catalytic_K_upper' in state_dict:
                self.parent.catalytic_K_upper_entry.setValue(state_dict['catalytic_K_upper'])

            # Restore catalytic sub-model type
            if 'catalytic_sub_model' in state_dict:
                ode_widget = getattr(self.parent, 'ode_widget', None)
                if ode_widget and hasattr(ode_widget, 'widgets'):
                    sub_model = state_dict['catalytic_sub_model']
                    simple_radio = ode_widget.widgets.get('simple_model_radio')
                    full_radio = ode_widget.widgets.get('full_model_radio')
                    if sub_model == 'simple' and simple_radio:
                        simple_radio.setChecked(True)
                    elif full_radio:
                        full_radio.setChecked(True)

            # Restore per-trace T values
            if 'per_trace_T_values' in state_dict:
                self._set_per_trace_T_values(state_dict['per_trace_T_values'])

            # Restore per-trace Z0 values (bimolecular fitting)
            if 'per_trace_Z0_values' in state_dict:
                self._set_per_trace_Z0_values(state_dict['per_trace_Z0_values'])


            # ALWAYS reset time window to full data range on startup
            # (Don't restore saved time window values - user wants fresh start)
            if self.parent.data_df is not None and self.parent.time_col in self.parent.data_df.columns:
                max_time = self.parent.data_df[self.parent.time_col].max()
                min_time = self.parent.data_df[self.parent.time_col].min()

                # Set proper maximum values first
                self.parent.start_time_spinbox.setMaximum(max_time * 1.5)
                self.parent.end_time_spinbox.setMaximum(max_time * 1.5)
                self.parent.start_time_spinbox.setMinimum(0)
                self.parent.end_time_spinbox.setMinimum(0)

                # Reset to full data range (not saved values)
                self.parent.start_time_spinbox.setValue(min_time)
                self.parent.end_time_spinbox.setValue(max_time)

                logger.info("Reset time window to full data range: %.2f to %.2f min (saved values ignored)",
                           min_time, max_time)

            # Restore results settings
            if self.parent.r2_threshold_spinbox and 'r2_threshold' in state_dict:
                self.parent.r2_threshold_spinbox.setValue(state_dict['r2_threshold'])
                logger.info("Restored R^2 threshold: %.2f", state_dict['r2_threshold'])

            # Restore experiment metadata
            self.parent.experiment_title = state_dict.get('experiment_title', '')
            self.parent.experiment_info = state_dict.get('experiment_info', '')
            self.parent._last_mean_rate_constant = state_dict.get('last_mean_rate_constant', (0, 0))

            # Restore trace settings
            self.parent.trace_settings = state_dict.get('trace_settings', {})
            self.parent.show_legend = state_dict.get('show_legend', True)

            # Restore plot state
            self.parent.scatter_state = state_dict.get('scatter_state', False)
            self.parent.grid_state = state_dict.get('grid_state', False)

            # Restore scatter checkbox state
            if self.parent.scatter_view_checkbox:
                self.parent.scatter_view_checkbox.setChecked(self.parent.scatter_state)

            # Only restore fitted parameters if it's the same dataset
            if same_dataset and 'fitted_parameters' in state_dict:
                self.parent.fitted_parameters = state_dict['fitted_parameters']
                logger.info("Restored fitted parameters from previous session")
            else:
                self.parent.fitted_parameters = {}
                if not same_dataset:
                    logger.info("Cleared fitted parameters (new dataset loaded)")

            # Only restore detected endpoints if it's the same dataset
            if same_dataset and 'detected_endpoints' in state_dict:
                self.parent.detected_endpoints = state_dict['detected_endpoints']
                logger.info("Restored detected endpoints from previous session")
                # Update bimolecular status panel
                if hasattr(self.parent, '_update_bimolecular_status'):
                    self.parent._update_bimolecular_status()
            else:
                self.parent.detected_endpoints = {}
                if not same_dataset:
                    logger.info("Cleared detected endpoints (new dataset loaded)")

            # Restore user-defined replicate groups from hash-based mapping
            # This allows groups to be preserved even when switching between datasets
            if current_dataset_hash and self.parent.data_processor:
                if current_dataset_hash in self.parent._replicate_groups_by_hash:
                    user_groups = self.parent._replicate_groups_by_hash[current_dataset_hash]
                    if user_groups:
                        self.parent.data_processor.user_defined_groups = user_groups
                        num_groups = len(user_groups)
                        logger.info("Restored %d user-defined replicate groups for this dataset (hash: %s...)",
                                   num_groups, current_dataset_hash[:16])

                        # CRITICAL FIX: Recalculate replicate statistics from restored groups
                        # This ensures the "Replicate Averages" button works after session restore
                        if self.parent.data_df is not None and self.parent.time_col:
                            result = self.parent.data_processor.replicate_manager.calculate_group_statistics(
                                self.parent.data_df, self.parent.time_col, user_groups
                            )
                            if result:
                                self.parent.data_processor.replicate_stats_df = result[0]
                                self.parent.data_processor.replicate_info = result[1]
                                self.parent.replicate_info = result[1]
                                logger.info("Recalculated replicate statistics for %d groups", num_groups)
                            else:
                                logger.warning("Failed to recalculate replicate statistics")
                    else:
                        self.parent.data_processor.user_defined_groups = {}
                else:
                    self.parent.data_processor.user_defined_groups = {}
                    logger.debug("No stored groups found for this dataset")
            else:
                if self.parent.data_processor:
                    self.parent.data_processor.user_defined_groups = {}

            # Restore trace visibility (must be done after data is loaded)
            if self.parent.data_df is not None and self.parent.trace_selection_panel:
                visible_traces = state_dict.get('visible_traces', [])

                # Populate the trace list - pass empty dict for replicate_info
                # because the main trace list shows all traces flat (replicate
                # grouping is only used in the replicate averages window)
                self.parent.gui.populate_trace_list(
                    self.parent.trace_selection_panel,
                    self.parent.data_df,
                    self.parent.time_col,
                    {}
                )

                # Restore visibility state via the grid checkbox manager
                if visible_traces and 'manager' in self.parent.trace_selection_panel:
                    manager = self.parent.trace_selection_panel['manager']
                    restored_count = 0

                    for checkbox, col_name in manager.checkboxes:
                        checkbox.blockSignals(True)
                        should_show = col_name in visible_traces
                        checkbox.setChecked(should_show)
                        checkbox.blockSignals(False)
                        if should_show:
                            restored_count += 1

                    # Update the summary label to reflect restored state
                    self.parent.gui._update_trace_summary(self.parent.trace_selection_panel)
                    logger.info("Restored visibility for %d of %d traces",
                               restored_count, len(manager.checkboxes))

                # Populate per-trace tables for both models
                data_cols = [c for c in self.parent.data_df.columns
                             if c != self.parent.time_col and not c.endswith("_fitted")]

                # Populate per-trace T table for catalytic model
                if hasattr(self.parent, '_populate_per_trace_T_table'):
                    self.parent._populate_per_trace_T_table(data_cols)
                    logger.info("Populated per-trace T table with %d traces", len(data_cols))

                # Populate per-trace Z0 table for bimolecular model
                if hasattr(self.parent, '_populate_per_trace_Z0_table'):
                    self.parent._populate_per_trace_Z0_table(data_cols)
                    logger.info("Populated per-trace Z0 table with %d traces", len(data_cols))

                # Restore exclude_from_fit checkbox states
                self._restore_exclude_checkbox_states()

            # Sync the previous dataset hash to current for future comparisons
            if current_dataset_hash:
                self.parent._previous_dataset_hash = current_dataset_hash
                logger.debug("Synced _previous_dataset_hash for future comparisons")

            # Update plot if we have data
            if self.parent.data_df is not None:
                self.parent.update_plot()
                logger.info("Kinetics processor state restored successfully with data from '%s'",
                           self.parent.filename)
            else:
                logger.info("Kinetics processor state restored (no data loaded)")

        except Exception as e:
            logger.error("Failed to restore kinetics processor state: %s", e, exc_info=True)
