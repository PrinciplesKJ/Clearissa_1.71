"""
Clearissa - csv_data_loader.py
-------------------------------
CSV data import coordinator and loader interface.

This module provides:
- Folder selection and batch CSV processing
- Data loading coordination with processing pipeline
- Recent data persistence for session recovery
- Integration with DataFrameProcessor for visualisation

Author: Križan Jurinović
Date: October 2025
"""

import logging
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QProgressDialog
from PyQt5.QtCore import Qt
from core.common.data_processing_utils import load_data, merge_data, save_recent_data

logger = logging.getLogger(__name__)


class CSVDataLoader:
    """
    Coordinate CSV file loading and processing operations.

    This class manages the user interface for selecting data sources,
    triggers the data loading pipeline, and passes results to the
    data processor for visualisation and analysis.

    Parameters
    ----------
    parent : QWidget, optional
        Parent widget for dialog display.

    Attributes
    ----------
    data_subfolder : str or None
        Path to selected data folder.
    csvdict : dict or None
        Dictionary of loaded CSV data structures.
    merged_dataframe : pandas.DataFrame or None
        Merged experimental data from all loaded files.
    dataproc : DataFrameProcessor or None
        Reference to data processor instance.

    Notes
    -----
    This class acts as a bridge between file selection UI and the
    core data processing utilities defined in data_processing_utils.
    """

    def __init__(self, parent=None):
        """
        Initialise CSV data loader.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget for Qt hierarchy.
        """
        self.parent = parent
        self.data_subfolder = None
        self.csvdict = None
        self.merged_dataframe = None
        self.dataproc = None

    def load_csv_folder(self):
        """
        Prompt user to select data folder and initiate processing.
        """
        logger.info("==================================================")
        logger.info("DATA IMPORT STARTED")
        logger.info("==================================================")
        logger.debug("[LOADER] Folder selection dialog opened")

        self.data_subfolder = QFileDialog.getExistingDirectory(
            None,
            "Select Data Folder",
            "",
            QFileDialog.ShowDirsOnly
        )

        if self.data_subfolder:
            logger.info("[LOADER] Selected folder: %s", self.data_subfolder)
            self.process_data()

            if self.dataproc is not None:
                # Pass the freshly merged data directly to the processor
                self.dataproc.set_merged_data(self.merged_dataframe, self.csvdict)


                self.dataproc.view_data_window()

                logger.info("[LOADER] Data passed to processor successfully")
            else:
                logger.warning("[LOADER] DataProcessor instance not assigned")
        else:
            logger.info("[LOADER] No folder selected; import cancelled")


    def process_data(self):
        """
        Execute data loading and merging pipeline with progress indication.

        Loads all supported files from the selected folder, merges them
        into a unified DataFrame, and persists results to disk for
        session recovery.

        Shows a progress dialogue to provide user feedback during
        potentially lengthy operations.

        Returns
        -------
        None

        Raises
        ------
        Displays error messages via QMessageBox if processing fails.

        Notes
        -----
        Processing steps:
        1. Load individual files via load_data()
        2. Merge datasets via merge_data()
        3. Save results via save_recent_data()

        All operations are logged for debugging and audit purposes.
        """
        logger.info("***** START: DATA_PROCESSING *****")

        # Create progress dialogue
        progress = QProgressDialog(
            "Loading data files...",
            None,  # No cancel button during loading
            0, 100,
            self.parent
        )
        progress.setWindowTitle("Loading Data")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(500)  # Show after 500ms
        progress.setValue(10)

        try:
            logger.debug("[PROCESS] Loading data from folder...")
            progress.setLabelText("Reading CSV files from folder...")
            progress.setValue(20)

            self.csvdict = load_data(self.data_subfolder)

            if not self.csvdict:
                logger.error("[PROCESS] CSV dictionary empty; cannot continue")
                progress.close()
                QMessageBox.critical(
                    None,
                    "Data Load Error",
                    "The CSV Dictionary is empty. Cannot continue."
                )
                return

            logger.info("==================================================")
            logger.info("MERGING DATA")
            logger.info("==================================================")

            progress.setLabelText(f"Merging {len(self.csvdict)} data files...")
            progress.setValue(60)

            self.merged_dataframe = merge_data(self.csvdict)

            if self.merged_dataframe is None:
                logger.error("[PROCESS] Data merge failed; cannot continue")
                progress.close()
                QMessageBox.critical(
                    None,
                    "Data Merge Error",
                    "Could not merge data. Cannot continue."
                )
                return

            logger.info("==================================================")
            logger.info("MERGE COMPLETE - DATAFRAME READY")
            logger.info("==================================================")
            logger.info("[PROCESS] Final merged shape: %s", self.merged_dataframe.shape)
            logger.info("[PROCESS] Columns: %s", list(self.merged_dataframe.columns))

            # Save the data after processing (backup to disk)
            progress.setLabelText("Saving processed data...")
            progress.setValue(90)

            save_recent_data(self.merged_dataframe, self.csvdict)
            logger.info("[PROCESS] Merged dataframe saved successfully")

            progress.setValue(100)
            progress.close()

            logger.info("***** END: DATA_PROCESSING *****")

        except Exception as e:
            logger.exception("[PROCESS] Error during data processing: %s", e)
            progress.close()
            QMessageBox.critical(
                None,
                "Processing Error",
                f"An error occurred during data processing:\n\n{e}"
            )
