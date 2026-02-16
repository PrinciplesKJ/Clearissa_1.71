"""
Well Selection Manager for DataFrameProcessor
----------------------------------------------
Provides clean, single-layer well selection logic with:
- Temporary selection state (highlighted before assignment)
- Permanent category assignments (data, pos_ctrl, blank, donor_ctrl, acceptor_ctrl, blocked_ctrl)
- Consistent colour mapping and visual feedback
- Persistent state saving and loading

Colour Scheme:
- data: #4CAF50 (Green)
- pos_ctrl: #2196F3 (Blue)
- blank: #F44336 (Red)
- neg_ctrl: #000000 (Black)
- donor_ctrl: #9C27B0 (Purple)
- acceptor_ctrl: #FF9800 (Orange)
- blocked_ctrl: #795548 (Brown) - B1-L quenched acceptor control
- unassigned: #FFFFFF (White)
- temp_selected: #E3F2FD (Light Blue Outline)

Workflow:
1. User clicks individual wells -> temporary selection (light outline)
2. User clicks category button -> wells permanently assigned with solid colour
3. User clicks "Deselect All" -> clears temporary selections, keeps assignments
4. User clicks "Clear All Assignments" -> removes all assignments, resets to unassigned
"""

import logging
import json
import os
from PyQt5.QtCore import QObject, pyqtSignal


class WellSelectionManager(QObject):
    """
    Manages well selection state and category assignments.

    Signals:
        selection_changed: Emitted when temporary selection changes
        assignments_changed: Emitted when permanent assignments change
    """

    selection_changed = pyqtSignal()
    assignments_changed = pyqtSignal()

    # Colour mapping for well categories
    COLORS = {
        'data': '#4CAF50',  # Green
        'pos_ctrl': '#2196F3',  # Blue
        'blank': '#F44336',  # Red
        'neg_ctrl': '#000000',  # Black - distinct from all other colours
        'donor_ctrl': '#9C27B0',  # Purple
        'acceptor_ctrl': '#FF9800',  # Orange
        'blocked_ctrl': '#795548',  # Brown - B1-L quenched acceptor control
        'unassigned': '#FFFFFF',  # White
        'temp_selected': '#E3F2FD'  # Light blue
    }

    def __init__(self, processor):
        """
        Initialise the well selection manager.

        Parameters
        ----------
        processor : DataFrameProcessor
            Parent processor instance
        """
        super().__init__()
        self.processor = processor
        self.logger = logging.getLogger(__name__)

        # Well state storage: {well_id: {"selected": bool, "category": str}}
        self.well_states = {}

        # Reference to GUI well buttons (populated by GUI)
        self.well_buttons = {}

        # Drag selection state
        self.drag_active = False
        self.drag_start_well = None
        self.drag_wells = set()  # Wells touched during current drag operation

    def initialise_wells(self, well_ids):
        """
        Initialise well states for a list of well identifiers.
        Clears any previous well states to ensure clean state for new data.

        Parameters
        ----------
        well_ids : list of str
            List of well identifiers (e.g., ['A01', 'A02', ...])
        """
        self.logger.info(f"Initialising {len(well_ids)} wells")

        # Clear old wells from previous datasets
        old_count = len(self.well_states)
        self.well_states.clear()
        if old_count > 0:
            self.logger.info(f"Cleared {old_count} wells from previous dataset")

        # Initialise new wells
        for well_id in well_ids:
            self.well_states[well_id] = {
                "selected": False,
                "category": "unassigned"
            }

        self.logger.info(f"Initialised {len(self.well_states)} wells for current dataset")

    def select_well(self, well_id):
        """
        Toggle temporary selection state of a single well.

        INTUITIVE BEHAVIOR:
        - Well is assigned: toggle selection (keep assignment, allow reassignment or clearing)
        - Well is unassigned: toggle selection
        - Selection is shown as outline over the assignment color

        Parameters
        ----------
        well_id : str
            Well identifier
        """
        if well_id not in self.well_states:
            self.logger.warning(f"Attempted to select unknown well: {well_id}")
            return

        # Simply toggle selection state, regardless of assignment status
        # This allows users to select assigned wells for reassignment or clearing
        current_selected = self.well_states[well_id]["selected"]
        self.well_states[well_id]["selected"] = not current_selected

        category = self.well_states[well_id]["category"]
        if category != "unassigned":
            action = "selected" if not current_selected else "deselected"
            self.logger.info(f"Well {well_id}: {action} (assigned as '{category}')")
        else:
            self.logger.debug(f"Well {well_id} selection toggled: {not current_selected}")

        self.selection_changed.emit()

    def toggle_all_selection(self):
        """
        Toggle selection of all unassigned wells.
        If any unassigned wells are not selected, select all unassigned wells.
        If all unassigned wells are selected, deselect all.

        Note: This only affects unassigned wells. Assigned wells are not affected.
        """
        unassigned_wells = [well_id for well_id, state in self.well_states.items()
                           if state["category"] == "unassigned"]

        if not unassigned_wells:
            self.logger.warning("No unassigned wells to toggle")
            return

        # Check if all unassigned wells are selected
        all_selected = all(self.well_states[well_id]["selected"] for well_id in unassigned_wells)

        # Toggle: if all selected, deselect all; otherwise select all
        new_state = not all_selected

        for well_id in unassigned_wells:
            self.well_states[well_id]["selected"] = new_state

        action = "Selected" if new_state else "Deselected"
        self.logger.info(f"{action} all {len(unassigned_wells)} unassigned wells")
        self.selection_changed.emit()

    def select_all(self):
        """Select all unassigned wells (temporary selection)."""
        unassigned_wells = [well_id for well_id, state in self.well_states.items()
                           if state["category"] == "unassigned"]

        for well_id in unassigned_wells:
            self.well_states[well_id]["selected"] = True

        self.logger.info(f"Selected all {len(unassigned_wells)} unassigned wells")
        self.selection_changed.emit()

    def deselect_all(self):
        """Deselect all wells (clears temporary selection, keeps assignments)."""
        for well_id in self.well_states:
            self.well_states[well_id]["selected"] = False

        self.logger.info("Deselected all wells (assignments preserved)")
        self.selection_changed.emit()

    def clear_all_assignments(self):
        """
        Clear all category assignments, reset all wells to unassigned.
        This is a destructive operation that removes all well categorizations.
        """
        count = sum(1 for state in self.well_states.values() if state["category"] != "unassigned")

        for well_id in self.well_states:
            self.well_states[well_id]["category"] = "unassigned"
            self.well_states[well_id]["selected"] = False

        self.logger.info("EVENT: Clear All Assignments - Reset %d wells to unassigned state", count)

        self.assignments_changed.emit()
        self.selection_changed.emit()

    def clear_selected_assignments(self):
        """
        Clear assignments for currently selected wells only (not all wells).
        This is used by the "Clear" button (short-click).

        Selected wells are unassigned and deselected.
        """
        selected_wells = [well_id for well_id, state in self.well_states.items()
                         if state["selected"]]

        if not selected_wells:
            self.logger.warning("No wells selected for clearing")
            return

        # Count how many were actually assigned (for logging)
        count = sum(1 for well_id in selected_wells
                   if self.well_states[well_id]["category"] != "unassigned")

        # Unassign and deselect the selected wells
        for well_id in selected_wells:
            self.well_states[well_id]["category"] = "unassigned"
            self.well_states[well_id]["selected"] = False

        wells_str = ", ".join(sorted(selected_wells))
        self.logger.info(f"EVENT: Cleared {count} assigned wells from selection - Wells: {wells_str}")

        self.assignments_changed.emit()
        self.selection_changed.emit()

    def assign_selected(self, category):
        """
        Assign currently selected wells to a category.

        Special behaviour for 'clear': unassigns wells instead of assigning them.

        Parameters
        ----------
        category : str
            Category name ('data', 'pos_ctrl', 'blank', 'neg_ctrl', 'donor_ctrl', 'acceptor_ctrl', 'blocked_ctrl', 'clear')
        """
        selected_wells = [well_id for well_id, state in self.well_states.items()
                         if state["selected"]]

        if not selected_wells:
            self.logger.warning(f"No wells selected for assignment to {category}")
            return

        # Special handling for 'clear' category - unassign wells
        if category == 'clear':
            for well_id in selected_wells:
                self.well_states[well_id]["category"] = "unassigned"
                self.well_states[well_id]["selected"] = False

            wells_str = ", ".join(sorted(selected_wells))
            self.logger.info("EVENT: Wells cleared or unassigned - Wells: %s", wells_str)

            self.assignments_changed.emit()
            self.selection_changed.emit()
            return

        # Validate category
        if category not in self.COLORS or category in ['unassigned', 'temp_selected']:
            self.logger.error(f"Invalid category: {category}")
            return

        # Assign category to selected wells
        for well_id in selected_wells:
            self.well_states[well_id]["category"] = category
            self.well_states[well_id]["selected"] = False  # Clear selection after assignment

        wells_str = ", ".join(sorted(selected_wells))
        self.logger.info("EVENT: Well selection update - Category: %s, Wells: %s", category, wells_str)

        self.assignments_changed.emit()
        self.selection_changed.emit()

    def assign_all(self, category):
        """
        Assign ALL wells to a specific category (long-click action).

        This is used for the long-click "assign all wells" functionality.

        Parameters
        ----------
        category : str
            Category name ('data', 'pos_ctrl', 'blank', 'neg_ctrl', 'donor_ctrl', 'acceptor_ctrl', 'blocked_ctrl')
        """
        # Validate category
        if category not in self.COLORS or category in ['unassigned', 'temp_selected', 'clear']:
            self.logger.error(f"Invalid category for assign_all: {category}")
            return

        # Assign all wells to the category
        for well_id in self.well_states:
            self.well_states[well_id]["category"] = category
            self.well_states[well_id]["selected"] = False

        self.logger.info("EVENT: Long-click operation - Action: Assign All Wells, Category: %s, Total wells: %d", category, len(self.well_states))

        # Only emit assignments_changed, not selection_changed
        # This prevents duplicate UI updates and plot refreshes that cause stack overflow
        self.assignments_changed.emit()

    def get_wells_by_category(self, category):
        """
        Get all wells assigned to a specific category.

        Parameters
        ----------
        category : str
            Category name

        Returns
        -------
        list of str
            List of well identifiers in the category
        """
        return [well_id for well_id, state in self.well_states.items()
                if state["category"] == category]

    def get_selected_wells(self):
        """
        Get all temporarily selected wells.

        Returns
        -------
        list of str
            List of well identifiers currently selected
        """
        return [well_id for well_id, state in self.well_states.items()
                if state["selected"]]

    def get_well_colour(self, well_id):
        """
        Get the display colour for a well based on its state.

        Parameters
        ----------
        well_id : str
            Well identifier

        Returns
        -------
        str
            Hex colour code
        """
        if well_id not in self.well_states:
            return self.COLORS['unassigned']

        state = self.well_states[well_id]

        # If well has a category assignment, return that colour
        if state["category"] != "unassigned":
            return self.COLORS[state["category"]]

        # If temporarily selected, return selection colour
        if state["selected"]:
            return self.COLORS['temp_selected']

        # Otherwise, unassigned
        return self.COLORS['unassigned']

    def is_well_assigned(self, well_id):
        """
        Check if a well has a permanent category assignment.

        Parameters
        ----------
        well_id : str
            Well identifier

        Returns
        -------
        bool
            True if well is assigned to a category
        """
        if well_id not in self.well_states:
            return False
        return self.well_states[well_id]["category"] != "unassigned"

    def is_well_selected(self, well_id):
        """
        Check if a well is currently selected.

        Parameters
        ----------
        well_id : str
            Well identifier

        Returns
        -------
        bool
            True if well is selected
        """
        if well_id not in self.well_states:
            return False
        return self.well_states[well_id]["selected"]

    def start_drag_selection(self, well_id):
        """
        Start a drag selection operation.

        Parameters
        ----------
        well_id : str
            Well identifier where drag started
        """
        self.drag_active = True
        self.drag_start_well = well_id
        self.drag_wells = {well_id}
        self.logger.debug(f"Started drag selection at {well_id}")

        # Select the starting well
        if well_id in self.well_states:
            self.well_states[well_id]["selected"] = True
            self.selection_changed.emit()

    def continue_drag_selection(self, well_id):
        """
        Continue drag selection over a well.

        Parameters
        ----------
        well_id : str
            Well identifier being dragged over
        """
        if not self.drag_active:
            return

        if well_id not in self.drag_wells and well_id in self.well_states:
            # Select the well regardless of assignment status
            self.drag_wells.add(well_id)
            self.well_states[well_id]["selected"] = True
            self.logger.debug(f"Drag selecting well {well_id}")
            self.selection_changed.emit()

    def select_rectangular_region(self, start_well_id, end_well_id):
        """
        Select all wells within a rectangular region.
        Selects both assigned and unassigned wells.
        Clears drag_start_well after completion for clean state.
        """
        import re

        def parse_well(well_id):
            """Parse well ID into row and column."""
            match = re.match(r'([A-H])(\d+)', well_id)
            if match:
                row = ord(match.group(1)) - ord('A')  # 0-7
                col = int(match.group(2)) - 1  # 0-11
                return row, col
            return None, None

        start_row, start_col = parse_well(start_well_id)
        end_row, end_col = parse_well(end_well_id)

        if None in [start_row, start_col, end_row, end_col]:
            self.logger.warning(f"Invalid well IDs for rectangular selection: {start_well_id}, {end_well_id}")
            return

        # Determine bounds
        min_row = min(start_row, end_row)
        max_row = max(start_row, end_row)
        min_col = min(start_col, end_col)
        max_col = max(start_col, end_col)

        # Select ALL wells in rectangle (both assigned and unassigned)
        selected_count = 0
        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                well_id = f"{chr(ord('A') + row)}{col + 1}"
                if well_id in self.well_states:
                    self.well_states[well_id]["selected"] = True
                    selected_count += 1

        self.logger.info(
            f"Selected {selected_count} wells in rectangular region {start_well_id}:{end_well_id}")

        # Clear drag_start_well after completing rectangular selection
        # This ensures next Shift+click starts fresh instead of reusing old start point
        self.drag_start_well = None

        self.selection_changed.emit()



    def end_drag_selection(self):
        """
        End the current drag selection operation.
        """
        if self.drag_active:
            self.logger.info(f"Completed drag selection of {len(self.drag_wells)} wells")
            self.drag_active = False
            self.drag_start_well = None
            self.drag_wells.clear()


    def save_state(self, filepath):
        """
        Save well selection state to JSON file.

        Parameters
        ----------
        filepath : str
            Path to save file
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            with open(filepath, 'w') as f:
                json.dump(self.well_states, f, indent=2)

            self.logger.info(f"Saved well selection state to {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to save well selection state: {e}")

    def load_state(self, filepath):
        """
        Load well selection state from JSON file.

        Parameters
        ----------
        filepath : str
            Path to load file

        Returns
        -------
        bool
            True if loaded successfully
        """
        try:
            if not os.path.exists(filepath):
                self.logger.debug(f"No saved state found at {filepath}")
                return False

            with open(filepath, 'r') as f:
                loaded_states = json.load(f)

            # Validate format: all well states must have 'selected' and 'category' keys
            # Old save files may lack 'category' and are incompatible with the current version
            required_keys = {'selected', 'category'}
            for well_id, state in loaded_states.items():
                if not isinstance(state, dict) or not required_keys.issubset(state.keys()):
                    self.logger.warning(
                        f"Incompatible well selection state format in {filepath} "
                        f"(well '{well_id}' missing required keys). Deleting outdated file."
                    )
                    try:
                        os.remove(filepath)
                        self.logger.info(f"Deleted incompatible state file: {filepath}")
                    except OSError as del_err:
                        self.logger.error(f"Failed to delete incompatible state file: {del_err}")
                    return False

            # Validate and merge with current state
            for well_id, state in loaded_states.items():
                if well_id in self.well_states:
                    self.well_states[well_id] = state

            self.logger.info(f"Loaded well selection state from {filepath}")
            self.assignments_changed.emit()
            self.selection_changed.emit()
            return True

        except Exception as e:
            self.logger.error(f"Failed to load well selection state: {e}")
            return False

