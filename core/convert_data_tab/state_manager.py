"""
State Manager - Persistent state management for Convert Data Tab
================================================================
Saves and restores UI state via QSettings so users can resume with
previously entered parameters.

Author: Križan Jurinović
Date: October 19, 2025
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, TYPE_CHECKING
from dataclasses import dataclass

from PyQt5.QtCore import QSettings
from PyQt5.QtWidgets import QComboBox

if TYPE_CHECKING:
    from .gui import ConvertDataTab

logger = logging.getLogger(__name__)


@dataclass
class StateSnapshot:
    """Snapshot of the Convert Data Tab's UI state."""
    # Mode selection
    mode: str = "tmsd"

    # Timepoint parameters
    init_start: str = ""
    init_end: str = ""
    trigger: str = ""
    injection_1: str = ""
    injection_2: str = ""
    neg_ctrl_start: str = ""
    neg_ctrl_end: str = ""

    # Concentration parameters
    pos_ctrl_concentrations: List[str] = None
    neg_ctrl_concentrations: List[str] = None
    donor_ctrl_concentration: str = ""
    acceptor_ctrl_concentration: str = ""
    b1l_initial_conc: str = ""
    b1l_ctrl_concentration: str = ""
    c_ref: str = "10"

    # Additional timepoint parameters
    volume_200ul_timepoint: str = ""

    # Plot settings
    plot_mode: str = "line"

    # Mass-action species selections (3×4 grid)
    mass_selections: Dict[str, str] = None


    # Dataset selection for forwarding
    dataset_selection: str = ""

    def __post_init__(self):
        """Initialise mutable default values."""
        if self.pos_ctrl_concentrations is None:
            self.pos_ctrl_concentrations = []
        if self.neg_ctrl_concentrations is None:
            self.neg_ctrl_concentrations = []
        if self.mass_selections is None:
            self.mass_selections = {}


class StateManager:
    """
    Manages persistent state for the Convert Data Tab via QSettings.

    Supports experiment-specific state storage when experiment_id is available.
    """

    SETTINGS_GROUP = "ConvertDataTab"
    SETTINGS_ORG = "MyCompany"
    SETTINGS_APP = "MyApp"

    def __init__(self, parent: ConvertDataTab):
        """
        Initialise the state manager.

        Args:
            parent: The ConvertDataTab widget to manage state for
        """
        self.parent = parent
        self.settings = QSettings(self.SETTINGS_ORG, self.SETTINGS_APP)

    def _get_settings_group_name(self) -> str:
        """
        Get the settings group name, experiment-specific if experiment_id is available.

        Returns:
            Settings group name string
        """
        experiment_id = getattr(self.parent, 'experiment_id', None)
        if experiment_id:
            # Use experiment-specific group name
            return f"{self.SETTINGS_GROUP}/{experiment_id}"
        else:
            # Fall back to default group
            return self.SETTINGS_GROUP

    def capture_current_state(self) -> StateSnapshot:
        """
        Capture the current state of all UI elements.

        Returns:
            StateSnapshot containing all current values
        """
        snapshot = StateSnapshot()

        # Capture mode
        snapshot.mode = self.parent.approach_combo.currentData() or "tmsd"

        # Capture timepoints
        snapshot.init_start = self.parent.init_start_time_entry.text()
        snapshot.init_end = self.parent.init_end_time_entry.text()
        snapshot.trigger = self.parent.reaction_trigger_timepoint.text()
        snapshot.injection_1 = self.parent.injection_for_nuking_entry.text()
        snapshot.injection_2 = self.parent.injection_for_nuking_entry_2.text()

        # Capture negative control timepoints
        if hasattr(self.parent, "neg_ctrl_start_time_entry"):
            snapshot.neg_ctrl_start = self.parent.neg_ctrl_start_time_entry.text()
        if hasattr(self.parent, "neg_ctrl_end_time_entry"):
            snapshot.neg_ctrl_end = self.parent.neg_ctrl_end_time_entry.text()

        # Capture control concentrations
        snapshot.pos_ctrl_concentrations = [
            entry.text() for entry in self.parent.pos_ctrl_concentration_entries
        ]
        snapshot.neg_ctrl_concentrations = [
            entry.text() for entry in self.parent.neg_ctrl_concentration_entries
        ]

        if self.parent.donor_ctrl_concentration_entry is not None:
            snapshot.donor_ctrl_concentration = (
                self.parent.donor_ctrl_concentration_entry.text()
            )

        if self.parent.acceptor_ctrl_concentration_entry is not None:
            snapshot.acceptor_ctrl_concentration = (
                self.parent.acceptor_ctrl_concentration_entry.text()
            )

        # Capture b1l initial concentration
        if hasattr(self.parent, "b1l_initial_conc_entry"):
            snapshot.b1l_initial_conc = self.parent.b1l_initial_conc_entry.text()

        # Capture b1l control concentration (for runtime calibration)
        if hasattr(self.parent, "b1l_ctrl_concentration_entry") and self.parent.b1l_ctrl_concentration_entry is not None:
            snapshot.b1l_ctrl_concentration = self.parent.b1l_ctrl_concentration_entry.text()

        # Capture c_ref value
        if hasattr(self.parent, "c_ref_entry"):
            snapshot.c_ref = self.parent.c_ref_entry.text()

        # Capture additional timepoint
        if hasattr(self.parent, "volume_200ul_timepoint"):
            snapshot.volume_200ul_timepoint = self.parent.volume_200ul_timepoint.text()

        # Capture plot mode
        if hasattr(self.parent, "plot_mode"):
            snapshot.plot_mode = self.parent.plot_mode


        # Capture mass-action selections (3×4 grid)
        snapshot.mass_selections = {}
        for (row, col), combo in self.parent.mass_combos.items():
            key = f"{row}:{col}"
            snapshot.mass_selections[key] = combo.currentText()


        # Capture dataset selection for forwarding
        if hasattr(self.parent, "dataset_combo"):
            snapshot.dataset_selection = self.parent.dataset_combo.currentText()

        return snapshot

    def save_state(self) -> None:
        """Save the current UI state to QSettings."""
        try:
            snapshot = self.capture_current_state()

            settings_group = self._get_settings_group_name()
            self.settings.beginGroup(settings_group)

            # Save mode
            self.settings.setValue("mode", snapshot.mode)

            # Save timepoints
            self.settings.setValue("init_start", snapshot.init_start)
            self.settings.setValue("init_end", snapshot.init_end)
            self.settings.setValue("trigger", snapshot.trigger)
            self.settings.setValue("inj1", snapshot.injection_1)
            self.settings.setValue("inj2", snapshot.injection_2)
            self.settings.setValue("neg_ctrl_start", snapshot.neg_ctrl_start)
            self.settings.setValue("neg_ctrl_end", snapshot.neg_ctrl_end)

            # Save control concentrations
            self.settings.setValue(
                "pos_ctrl_concentrations",
                snapshot.pos_ctrl_concentrations
            )
            self.settings.setValue(
                "donor_ctrl_concentration",
                snapshot.donor_ctrl_concentration
            )
            self.settings.setValue(
                "acceptor_ctrl_concentration",
                snapshot.acceptor_ctrl_concentration
            )
            self.settings.setValue(
                "b1l_initial_conc",
                snapshot.b1l_initial_conc
            )
            self.settings.setValue(
                "b1l_ctrl_concentration",
                snapshot.b1l_ctrl_concentration
            )
            self.settings.setValue(
                "c_ref",
                snapshot.c_ref
            )

            # Save additional timepoint
            self.settings.setValue(
                "volume_200ul_timepoint",
                snapshot.volume_200ul_timepoint
            )

            # Save plot settings
            self.settings.setValue("plot_mode", snapshot.plot_mode)


            # Save mass-action selections
            self.settings.setValue("mass_selections", snapshot.mass_selections)


            # Save dataset selection
            self.settings.setValue("dataset_selection", snapshot.dataset_selection)

            self.settings.endGroup()

            experiment_id = getattr(self.parent, 'experiment_id', None)
            if experiment_id:
                logger.info(f"State saved successfully for experiment: {experiment_id}")
            else:
                logger.info("State saved successfully (default)")

        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def load_state(self) -> Optional[StateSnapshot]:
        """
        Load previously saved state from persistent storage.

        If experiment_id is available, tries to load experiment-specific state first.
        If no experiment-specific state exists, falls back to default state.

        Returns:
            StateSnapshot if state exists, None otherwise
        """
        try:
            experiment_id = getattr(self.parent, 'experiment_id', None)
            settings_group = self._get_settings_group_name()

            # Check if experiment-specific settings exist
            if experiment_id:
                self.settings.beginGroup(settings_group)
                has_experiment_settings = self.settings.contains("mode")
                self.settings.endGroup()

                if not has_experiment_settings:
                    # Fall back to default settings
                    logger.info(f"No saved state found for experiment {experiment_id}, loading default settings")
                    settings_group = self.SETTINGS_GROUP
                else:
                    logger.info(f"Loading state for experiment: {experiment_id}")

            self.settings.beginGroup(settings_group)

            snapshot = StateSnapshot()

            # Load mode
            snapshot.mode = self.settings.value("mode", "tmsd")

            # Load timepoints
            snapshot.init_start = self.settings.value("init_start", "")
            snapshot.init_end = self.settings.value("init_end", "")
            snapshot.trigger = self.settings.value("trigger", "")
            snapshot.injection_1 = self.settings.value("inj1", "")
            snapshot.injection_2 = self.settings.value("inj2", "")
            snapshot.neg_ctrl_start = self.settings.value("neg_ctrl_start", "")
            snapshot.neg_ctrl_end = self.settings.value("neg_ctrl_end", "")

            # Load control concentrations
            snapshot.pos_ctrl_concentrations = self.settings.value(
                "pos_ctrl_concentrations", []
            )
            if not isinstance(snapshot.pos_ctrl_concentrations, list):
                snapshot.pos_ctrl_concentrations = []

            snapshot.donor_ctrl_concentration = self.settings.value(
                "donor_ctrl_concentration", ""
            )
            snapshot.acceptor_ctrl_concentration = self.settings.value(
                "acceptor_ctrl_concentration", ""
            )

            snapshot.b1l_initial_conc = self.settings.value("b1l_initial_conc", "")
            snapshot.b1l_ctrl_concentration = self.settings.value("b1l_ctrl_concentration", "")
            snapshot.c_ref = self.settings.value("c_ref", "10")

            # Load additional timepoint
            snapshot.volume_200ul_timepoint = self.settings.value("volume_200ul_timepoint", "")

            # Load plot settings
            snapshot.plot_mode = self.settings.value("plot_mode", "line")


            # Load mass-action selections
            snapshot.mass_selections = self.settings.value("mass_selections", {})
            if not isinstance(snapshot.mass_selections, dict):
                snapshot.mass_selections = {}


            # Load dataset selection
            snapshot.dataset_selection = self.settings.value("dataset_selection", "")

            self.settings.endGroup()

            if experiment_id and settings_group != self.SETTINGS_GROUP:
                logger.info(f"State loaded successfully for experiment: {experiment_id}")
            else:
                logger.info("State loaded successfully (default)")
            return snapshot

        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return None

    def restore_state(self, snapshot: StateSnapshot) -> None:
        """
        Restore UI state from a snapshot.

        Args:
            snapshot: StateSnapshot to restore from
        """
        try:
            # Restore mode
            for i in range(self.parent.approach_combo.count()):
                if self.parent.approach_combo.itemData(i) == snapshot.mode:
                    self.parent.approach_combo.setCurrentIndex(i)
                    break

            # Restore timepoints
            self.parent.init_start_time_entry.setText(snapshot.init_start)
            self.parent.init_end_time_entry.setText(snapshot.init_end)
            self.parent.reaction_trigger_timepoint.setText(snapshot.trigger)
            self.parent.injection_for_nuking_entry.setText(snapshot.injection_1)
            self.parent.injection_for_nuking_entry_2.setText(snapshot.injection_2)

            # Restore negative control timepoints
            if hasattr(self.parent, "neg_ctrl_start_time_entry"):
                self.parent.neg_ctrl_start_time_entry.setText(snapshot.neg_ctrl_start)
            if hasattr(self.parent, "neg_ctrl_end_time_entry"):
                self.parent.neg_ctrl_end_time_entry.setText(snapshot.neg_ctrl_end)

            # Restore control concentrations
            for i, entry in enumerate(self.parent.pos_ctrl_concentration_entries):
                if i < len(snapshot.pos_ctrl_concentrations):
                    entry.setText(snapshot.pos_ctrl_concentrations[i] or "")

            for i, entry in enumerate(self.parent.neg_ctrl_concentration_entries):
                if i < len(snapshot.neg_ctrl_concentrations):
                    entry.setText(snapshot.neg_ctrl_concentrations[i] or "")

            if self.parent.donor_ctrl_concentration_entry is not None:
                self.parent.donor_ctrl_concentration_entry.setText(
                    snapshot.donor_ctrl_concentration
                )

            if self.parent.acceptor_ctrl_concentration_entry is not None:
                self.parent.acceptor_ctrl_concentration_entry.setText(
                    snapshot.acceptor_ctrl_concentration
                )

            # Restore b1l initial concentration
            if hasattr(self.parent, "b1l_initial_conc_entry"):
                self.parent.b1l_initial_conc_entry.setText(snapshot.b1l_initial_conc)

            # Restore b1l control concentration
            if hasattr(self.parent, "b1l_ctrl_concentration_entry") and self.parent.b1l_ctrl_concentration_entry is not None:
                self.parent.b1l_ctrl_concentration_entry.setText(snapshot.b1l_ctrl_concentration)

            # Restore c_ref value
            if hasattr(self.parent, "c_ref_entry"):
                self.parent.c_ref_entry.setText(snapshot.c_ref)
                # Update the display label
                if hasattr(self.parent, "_update_c_ref_display"):
                    self.parent._update_c_ref_display()

            # Restore additional timepoint
            if hasattr(self.parent, "volume_200ul_timepoint"):
                self.parent.volume_200ul_timepoint.setText(snapshot.volume_200ul_timepoint)

            # Restore plot mode
            if hasattr(self.parent, "plot_mode"):
                self.parent.plot_mode = snapshot.plot_mode


            # Restore mass-action selections
            for (row, col), combo in self.parent.mass_combos.items():
                key = f"{row}:{col}"
                if key in snapshot.mass_selections:
                    self._set_combo_text(combo, snapshot.mass_selections[key])


            # Restore dataset selection
            if hasattr(self.parent, "dataset_combo") and snapshot.dataset_selection:
                self._set_combo_text(self.parent.dataset_combo, snapshot.dataset_selection)

            logger.info("State restored successfully")

        except Exception as e:
            logger.error(f"Failed to restore state: {e}")

    def _set_combo_text(self, combo: QComboBox, text: str) -> None:
        """
        Safely set combo box text, handling both indexed and editable combos.

        Args:
            combo: QComboBox to update
            text: Text value to set
        """
        if not text:
            return

        # Try to find exact match in combo items
        index = combo.findText(text)
        if index >= 0:
            combo.setCurrentIndex(index)
        else:
            # For editable combos, set the text directly
            combo.setCurrentText(text)

    def clear_state(self) -> None:
        """
        Clear all saved state from persistent storage.

        This is useful for debugging or resetting to default state.
        """
        try:
            self.settings.beginGroup(self.SETTINGS_GROUP)
            self.settings.remove("")  # Remove all keys in the group
            self.settings.endGroup()

            logger.info("Saved state cleared")

        except Exception as e:
            logger.error(f"Failed to clear state: {e}")

    def has_saved_state(self) -> bool:
        """
        Check if saved state exists.

        Returns:
            True if saved state exists, False otherwise
        """
        try:
            self.settings.beginGroup(self.SETTINGS_GROUP)
            has_state = bool(self.settings.allKeys())
            self.settings.endGroup()
            return has_state

        except Exception as e:
            logger.debug(f"Error checking for saved state: {e}")
            return False

