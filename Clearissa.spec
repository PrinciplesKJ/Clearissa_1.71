# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from pathlib import Path

# Ensure SPECPATH is defined for local editing/linting (PyInstaller injects it at runtime)
try:
    SPECPATH
except NameError:
    SPECPATH = os.path.dirname(__file__)

# Get the project root directory
project_root = os.path.abspath(SPECPATH)

block_cipher = None

# Determine icon file based on platform
if sys.platform == 'darwin':
    icon_file = 'clearissa_icon.icns' if os.path.exists('clearissa_icon.icns') else None
else:
    icon_file = 'clearissa_icon.ico'

a = Analysis(
    ['core/Clearissa_main.py'],
    pathex=[project_root],  # Add project root to Python path
    binaries=[],
    datas=[
        # Application resources
        ('manual.html', '.'),
        ('clearissa_logo.png', '.'),
        ('clearissa_icon.ico', '.'),

        # Configuration files and directories
        ('config', 'config'),

        # Resource utilities module (critical for frozen environment path resolution)
        ('resource_utils.py', '.'),
    ],
    hiddenimports=[
        # Core Python and Qt libraries
        'PyQt5',
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'PyQt5.QtWidgets',
        'pyqtgraph',
        'pyqtgraph.graphicsItems',
        'numpy',
        'pandas',
        'matplotlib',
        'matplotlib.backends.backend_qt5agg',
        'scipy',
        'scipy.optimize',
        'scipy.interpolate',

        # Application utilities
        'resource_utils',

        # Core welcome panel
        'core.welcome_panel',

        # Common utilities (shared across modules)
        'core.common',
        'core.common.base_state_manager',
        'core.common.csv_data_loader',
        'core.common.data_processing_utils',
        'core.common.format_detector',
        'core.common.options',
        'core.common.plot_style',
        'core.common.settings_manager',
        'core.common.ui_theme',

        # Convert data tab module
        'core.convert_data_tab',
        'core.convert_data_tab.bleed',
        'core.convert_data_tab.calibration',
        'core.convert_data_tab.channel_detector',
        'core.convert_data_tab.conversion_catalytic_FRET',
        'core.convert_data_tab.conversion_helpers',
        'core.convert_data_tab.conversion_HMSD_FRET',
        'core.convert_data_tab.conversion_Internal_TMSD',
        'core.convert_data_tab.conversion_normalised_sd',
        'core.convert_data_tab.conversion_TMSD',
        'core.convert_data_tab.gui',
        'core.convert_data_tab.io_utils',
        'core.convert_data_tab.matrix_renderer',
        'core.convert_data_tab.params',
        'core.convert_data_tab.plothelper',
        'core.convert_data_tab.sidebar_builder',
        'core.convert_data_tab.species',
        'core.convert_data_tab.state_manager',
        'core.convert_data_tab.ui_components',
        'core.convert_data_tab.widget_factory',

        # Data frame processor module
        'core.data_frame_processor',
        'core.data_frame_processor.main',
        'core.data_frame_processor.data_ops',
        'core.data_frame_processor.gui',
        'core.data_frame_processor.gui_components',
        'core.data_frame_processor.io_utils',
        'core.data_frame_processor.plot_utils',
        'core.data_frame_processor.standard_curve_tab',
        'core.data_frame_processor.well_selection',

        # Kinetics processor module
        'core.kinetics_processor',
        'core.kinetics_processor.main',
        'core.kinetics_processor.gui',
        'core.kinetics_processor.gui_widget_factory',
        'core.kinetics_processor.gui_widget_factory_bimolecular',
        'core.kinetics_processor.io_utils',
        'core.kinetics_processor.data_processor',
        'core.kinetics_processor.endpoint_detector',
        'core.kinetics_processor.fitting_engine',
        'core.kinetics_processor.kinetic_models',
        'core.kinetics_processor.kinetic_models.base',
        'core.kinetics_processor.kinetic_models.bimolecular',
        'core.kinetics_processor.kinetic_models.catalytic',
        'core.kinetics_processor.plot_manager',
        'core.kinetics_processor.replicate_manager',
        'core.kinetics_processor.report_generator',
        'core.kinetics_processor.results_formatter',
        'core.kinetics_processor.state_manager',

        # Plotting/export libraries required by kinetics export
        'plotnine',
        'plotnine.ggplot',
        'mizani',
        'mizani.breaks',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=True,  # Extract all files instead of using base_library.zip
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Clearissa',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,  # Disable strip (avoids warnings if strip utility not found)
    upx=False,  # Disable UPX compression (faster builds, slightly larger exe)
    console=False,  # No console window for GUI app
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_file,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,  # Disable strip (avoids warnings if strip utility not found)
    upx=False,  # Disable UPX compression
    upx_exclude=[],
    name='Clearissa',
)

# macOS-specific: Create .app bundle
if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name='Clearissa.app',
        icon=icon_file,
        bundle_identifier='com.clearissa.app',
        info_plist={
            'NSPrincipalClass': 'NSApplication',
            'NSAppleScriptEnabled': False,
            'CFBundleName': 'Clearissa',
            'CFBundleDisplayName': 'Clearissa',
            'CFBundleGetInfoString': 'Clearissa - Data Analysis Platform',
            'CFBundleVersion': '1.49',
            'CFBundleShortVersionString': '1.49',
            'NSHumanReadableCopyright': 'Copyright © 2025 Križan Jurinović',
            'NSHighResolutionCapable': 'True',
        },
    )
