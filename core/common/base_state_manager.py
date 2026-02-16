"""
Base State Manager - Common Infrastructure for State Persistence
================================================================

Provides shared functionality for state management across different modules,
including debounced auto-save, error handling, and logging patterns.

All module-specific state managers should inherit from BaseStateManager and
implement the abstract methods for their specific storage mechanisms.

Author: Krizan Jurinovic
Date: November 2025
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional
from PyQt5.QtCore import QTimer

logger = logging.getLogger(__name__)


class BaseStateManager(ABC):
    """
    Abstract base class for state management with common functionality.

    Provides:
    - Debounced auto-save infrastructure (configurable delay)
    - Consistent error handling and logging
    - Common state management patterns
    - Template methods for save/restore operations

    Subclasses must implement:
    - _capture_state(): Capture current state as a dictionary
    - _persist_state(): Write state to storage
    - _load_persisted_state(): Read state from storage
    - _apply_state(): Apply loaded state to UI
    """

    # Default auto-save delay in milliseconds (can be overridden by subclasses)
    DEFAULT_AUTOSAVE_DELAY_MS = 2000

    def __init__(self, parent_widget, autosave_delay_ms: Optional[int] = None):
        """
        Initialise base state manager.

        Parameters
        ----------
        parent_widget : QWidget
            Parent widget to manage state for
        autosave_delay_ms : int, optional
            Auto-save delay in milliseconds (default: 2000)
        """
        self.parent = parent_widget
        self.autosave_delay_ms = autosave_delay_ms or self.DEFAULT_AUTOSAVE_DELAY_MS

        # Debounced auto-save timer
        self._save_timer = QTimer()
        self._save_timer.setSingleShot(True)
        self._save_timer.timeout.connect(self._perform_save)

        # Track if state needs saving
        self._state_dirty = False

        logger.debug("Initialised %s with %dms autosave delay",
                    self.__class__.__name__, self.autosave_delay_ms)

    def save_state(self):
        """
        Schedule state save (debounced).

        State is saved after the configured delay from the last call to this method.
        This reduces disk I/O when multiple parameters change rapidly
        (e.g., user dragging spinbox with mouse).

        For immediate save (e.g., application closing), use save_state_now().
        """
        self._state_dirty = True

        # Restart timer (extends delay if called repeatedly)
        if self._save_timer.isActive():
            self._save_timer.stop()

        self._save_timer.start(self.autosave_delay_ms)
        logger.debug("%s: State save scheduled in %d ms",
                    self.__class__.__name__, self.autosave_delay_ms)

    def save_state_now(self):
        """
        Save state immediately (bypass debouncing).

        Use this when:
        - Application is closing
        - User explicitly saves
        - Critical state change
        """
        logger.debug("%s: Immediate state save requested", self.__class__.__name__)

        # Cancel any pending debounced save
        if self._save_timer.isActive():
            self._save_timer.stop()

        self._perform_save()

    def _perform_save(self):
        """
        Perform actual state save to storage.

        This template method orchestrates the save process:
        1. Check if save is needed
        2. Capture current state (via abstract method)
        3. Persist state to storage (via abstract method)
        4. Handle errors gracefully
        """
        if not self._state_dirty:
            logger.debug("%s: State not dirty, skipping save", self.__class__.__name__)
            return

        try:
            logger.info("%s: Performing state save to disk", self.__class__.__name__)

            # Capture current state (implemented by subclass)
            state_data = self._capture_state()

            # Persist to storage (implemented by subclass)
            self._persist_state(state_data)

            self._state_dirty = False
            logger.info("%s: State saved successfully", self.__class__.__name__)

        except Exception as e:
            logger.error("%s: Failed to save state: %s",
                        self.__class__.__name__, e, exc_info=True)

    def restore_state(self):
        """
        Restore state from storage.

        This template method orchestrates the restore process:
        1. Load state from storage (via abstract method)
        2. Check if state exists
        3. Apply state to UI (via abstract method)
        4. Handle errors gracefully
        """
        try:
            logger.info("%s: Attempting to restore state", self.__class__.__name__)

            # Load state from storage (implemented by subclass)
            state_data = self._load_persisted_state()

            if state_data is None:
                logger.info("%s: No saved state found - using defaults",
                           self.__class__.__name__)
                return

            # Apply state to UI (implemented by subclass)
            self._apply_state(state_data)

            logger.info("%s: State restored successfully", self.__class__.__name__)

        except Exception as e:
            logger.error("%s: Failed to restore state: %s",
                        self.__class__.__name__, e, exc_info=True)

    # =========================================================================
    # Abstract methods to be implemented by subclasses
    # =========================================================================

    @abstractmethod
    def _capture_state(self) -> Any:
        """
        Capture current state from UI.

        This method should read all relevant state from the parent widget
        and return it in a format suitable for persistence.

        Returns
        -------
        Any
            State data (dict, dataclass, or other serializable format)
        """
        pass

    @abstractmethod
    def _persist_state(self, state_data: Any) -> None:
        """
        Persist state data to storage.

        This method should write the captured state to persistent storage
        (e.g., pickle file, QSettings, JSON file, database).

        Parameters
        ----------
        state_data : Any
            State data captured by _capture_state()
        """
        pass

    @abstractmethod
    def _load_persisted_state(self) -> Optional[Any]:
        """
        Load state data from storage.

        This method should read state from persistent storage and return
        it in the same format as _capture_state().

        Returns
        -------
        Any or None
            Loaded state data, or None if no saved state exists
        """
        pass

    @abstractmethod
    def _apply_state(self, state_data: Any) -> None:
        """
        Apply loaded state to UI.

        This method should update all relevant widgets in the parent
        widget with the values from the loaded state.

        Parameters
        ----------
        state_data : Any
            State data loaded by _load_persisted_state()
        """
        pass

    # =========================================================================
    # Optional helper methods (can be overridden by subclasses)
    # =========================================================================

    def mark_dirty(self):
        """
        Mark state as dirty without scheduling a save.

        Useful for batch operations where you want to mark state as changed
        but defer the save scheduling to a later point.
        """
        self._state_dirty = True

    def is_dirty(self) -> bool:
        """
        Check if state has unsaved changes.

        Returns
        -------
        bool
            True if state has been modified since last save
        """
        return self._state_dirty

    def cancel_pending_save(self):
        """
        Cancel any pending debounced save.

        Useful when you want to prevent a scheduled save from executing.
        """
        if self._save_timer.isActive():
            self._save_timer.stop()
            logger.debug("%s: Cancelled pending save", self.__class__.__name__)
