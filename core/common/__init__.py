"""
Clearissa - Common Utilities Module
------------------------------------
Shared utilities used across all processors and modules.

This package contains:
- data_processing_utils: Core data loading and processing functions
- format_detector: Unified format detection for CSV/Excel files
- plot_style: Centralized plotting style and visual standards
- csv_data_loader: CSV loading coordinator
- options: FRET channel settings panel
- settings_manager: Persistent settings across sessions

Author: Križan Jurinović
Date: November 2025
"""

from .data_processing_utils import *
from .format_detector import *
from .plot_style import *
from .csv_data_loader import CSVDataLoader
from .options import OptionsPanel
from .settings_manager import SettingsManager

__all__ = [
    'CSVDataLoader',
    'OptionsPanel',
    'SettingsManager',
    'detect_data_format',
    'DataFormat',
    'apply_clearissa_style',
    'get_trace_color',
    'configure_pyqtgraph_widget',
]

