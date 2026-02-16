# -*- coding: utf-8 -*-
"""
Resource Utilities for Clearissa
---------------------------------
Handles resource path resolution for both development and frozen (PyInstaller) environments.

This module provides functions to locate data files, configuration files, and logs
in a way that works both during development and when the application is compiled
into a standalone executable.
"""

import os
import sys
from pathlib import Path


def get_resource_path(relative_path):
    """
    Get absolute path to resource, works for dev and for PyInstaller.

    Parameters
    ----------
    relative_path : str
        Relative path to the resource file (e.g., 'manual.html', 'images/logo.png')

    Returns
    -------
    str
        Absolute path to the resource file

    Notes
    -----
    When running as a PyInstaller executable, resources are extracted to sys._MEIPASS.
    In development mode, paths are resolved relative to the project root.
    """
    if getattr(sys, 'frozen', False):
        # Running as compiled executable - use PyInstaller's temp folder
        base_path = sys._MEIPASS
    else:
        # Running in development mode - use project root
        base_path = os.path.dirname(os.path.abspath(__file__))

    return os.path.join(base_path, relative_path)


def get_data_path(relative_path=None):
    """
    Get the user data directory for Clearissa, optionally with a relative path appended.

    Parameters
    ----------
    relative_path : str, optional
        Relative path to append to the data directory (e.g., 'config/calibration_data.json')

    Returns
    -------
    Path
        Path to the user data directory or specific file within it

    Notes
    -----
    Creates the directory if it doesn't exist.
    - Windows: C:\\Users\\<username>\\AppData\\Roaming\\Clearissa
    - Linux/macOS: ~/.clearissa

    Examples
    --------
    >>> get_data_path()  # Returns base directory
    WindowsPath('C:/Users/username/AppData/Roaming/Clearissa')

    >>> get_data_path('config/settings.json')  # Returns path to specific file
    WindowsPath('C:/Users/username/AppData/Roaming/Clearissa/config/settings.json')
    """
    if sys.platform == 'win32':
        base_dir = Path(os.environ.get('APPDATA', os.path.expanduser('~')))
    else:
        base_dir = Path.home()

    if sys.platform == 'win32':
        data_dir = base_dir / 'Clearissa'
    else:
        data_dir = base_dir / '.clearissa'

    data_dir.mkdir(parents=True, exist_ok=True)

    if relative_path:
        full_path = data_dir / relative_path
        # Create parent directories if needed
        full_path.parent.mkdir(parents=True, exist_ok=True)
        return full_path

    return data_dir


def get_config_dir():
    """
    Get the configuration directory for Clearissa.

    Returns
    -------
    Path
        Path to the config directory

    Notes
    -----
    Creates the directory if it doesn't exist.
    - Windows: C:\\Users\\<username>\\AppData\\Roaming\\Clearissa\\config
    - Linux/macOS: ~/.clearissa/config
    """
    config_dir = get_data_path() / 'config'
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_log_dir():
    """
    Get the log directory for Clearissa.

    Returns
    -------
    Path
        Path to the log directory

    Notes
    -----
    Creates the directory if it doesn't exist.
    """
    log_dir = get_data_path() / 'log'
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir
