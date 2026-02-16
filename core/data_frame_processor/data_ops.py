"""
Data processing utilities for DataFrameProcessor.

This module isolates all pandas-related logic from the GUI layer.
It provides methods for filtering data, normalising time, and preparing
datasets for conversion or export.
"""


import pandas as pd
import logging



class DataOps:
    """Data manipulation utilities for the DataFrameProcessor."""

    def __init__(self, processor):
        self.p = processor
        self.logger = logging.getLogger(__name__)

    # -------------------------------------------------------------------------
    # CORE FILTERING
    # -------------------------------------------------------------------------
    def filter_data(self, min_time, max_time, channel_checkboxes, well_checkboxes):
        """
        Filter the main dataframe based on a time range, selected wells, and channels.

        Args:
            min_time (float): Minimum time (inclusive)
            max_time (float): Maximum time (inclusive)
            channel_checkboxes (dict): {channel_name: QLabel or QCheckBox}
            well_checkboxes (dict): {well_name: bool}

        Returns:
            pd.DataFrame | None
        """
        if self.p.merged_dataframe is None or self.p.merged_dataframe.empty:
            self.logger.warning("No data available for filtering.")
            return None

        # Handle both QLabel and QCheckBox formats
        try:
            from PyQt5.QtWidgets import QLabel
            if channel_checkboxes and isinstance(list(channel_checkboxes.values())[0], QLabel):
                selected_channels = [
                    ch for ch, label in channel_checkboxes.items()
                    if getattr(label, "_selected", False)
                ]
            else:
                selected_channels = [ch for ch, chk in channel_checkboxes.items() if chk.isChecked()]
        except Exception:
            selected_channels = list(channel_checkboxes.keys())

        selected_wells = list(well_checkboxes.keys())

        if not selected_channels:
            self.logger.error("No channel selected for filtering.")
            return None
        if not selected_wells:
            self.logger.error("No well selected for filtering.")
            return None

        df = self.p.merged_dataframe.copy()
        if "Time [min]" not in df.columns:
            self.logger.warning("'Time [min]' column missing; attempting to normalise.")
            df, _ = self._normalize_time_to_minutes(df)

        df["Time [min]"] = df["Time [min]"].apply(lambda x: round(x, 2))
        filtered = df.loc[
            (df["Well"].isin(selected_channels))
            & (df["Time [min]"] >= min_time)
            & (df["Time [min]"] <= max_time),
            ["Well", "Time [min]"] + selected_wells
        ]
        return filtered

    # -------------------------------------------------------------------------
    # TIME NORMALISATION
    # -------------------------------------------------------------------------
    def _normalize_time_to_minutes(self, df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
        """
        Ensure the dataframe has a numeric 'Time [min]' column in minutes.

        Returns:
            (pd.DataFrame, str): Tuple of modified dataframe and name of the time column.
        """
        if df is None or df.empty:
            return df, "Time [min]"

        # Already minutes
        if "Time [min]" in df.columns:
            df["Time [min]"] = pd.to_numeric(df["Time [min]"], errors="coerce")
            return df, "Time [min]"

        # Seconds -> minutes
        if "Time [sec]" in df.columns:
            df["Time [min]"] = pd.to_numeric(df["Time [sec]"], errors="coerce") / 60.0
            df.drop(columns=["Time [sec]"], inplace=True)
            return df, "Time [min]"

        # Fallback: treat 'Time' as minutes
        if "Time" in df.columns:
            df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
            df.rename(columns={"Time": "Time [min]"}, inplace=True)
            return df, "Time [min]"

        # Last resort: create empty minutes column
        df["Time [min]"] = pd.NA
        return df, "Time [min]"

    # -------------------------------------------------------------------------
    # DATA PREPARATION
    # -------------------------------------------------------------------------
    def prepare_dataframes(
        self,
        min_time,
        max_time,
        channel_checkboxes,
        well_checkboxes,
        blank_well,
        pos_well,
        donor_well=None,
        acceptor_well=None,
        neg_ctrl_well=None,
        blocked_ctrl_well=None,
    ):
        """
        Prepare and convert selected, blank, positive control,
        donor, acceptor, negative control, and blocked control dataframes.

        Returns:
            Tuple of (selected, blank, pos_ctrl, donor, acceptor, neg_ctrl, blocked_ctrl)
        """
        self.logger.info("Preparing dataframes for conversion.")

        try:
            filter_data = self.filter_data
            selected_data = filter_data(min_time, max_time, channel_checkboxes, well_checkboxes)
            blank_data = filter_data(min_time, max_time, channel_checkboxes, blank_well)
            pos_data = filter_data(min_time, max_time, channel_checkboxes, pos_well)
            donor_data = filter_data(min_time, max_time, channel_checkboxes, donor_well) if donor_well else None
            acceptor_data = filter_data(min_time, max_time, channel_checkboxes, acceptor_well) if acceptor_well else None
            neg_ctrl_data = filter_data(min_time, max_time, channel_checkboxes, neg_ctrl_well) if neg_ctrl_well else None
            blocked_ctrl_data = filter_data(min_time, max_time, channel_checkboxes, blocked_ctrl_well) if blocked_ctrl_well else None

            # Validate existence
            required = {"Selected Data": selected_data, "Blank": blank_data, "Pos Ctrl": pos_data}
            for name, df in required.items():
                if df is None or df.empty:
                    self.logger.error(f"{name} is empty or missing.")
                    return None, None, None, None, None, None, None

            # Convert numeric columns (beyond first two)
            for df_name, df in required.items():
                df[df.columns[2:]] = df[df.columns[2:]].apply(pd.to_numeric, errors="coerce")

            if donor_data is not None and not donor_data.empty:
                donor_data[donor_data.columns[2:]] = donor_data[donor_data.columns[2:]].apply(pd.to_numeric, errors="coerce")
            if acceptor_data is not None and not acceptor_data.empty:
                acceptor_data[acceptor_data.columns[2:]] = acceptor_data[acceptor_data.columns[2:]].apply(pd.to_numeric, errors="coerce")
            if neg_ctrl_data is not None and not neg_ctrl_data.empty:
                neg_ctrl_data[neg_ctrl_data.columns[2:]] = neg_ctrl_data[neg_ctrl_data.columns[2:]].apply(pd.to_numeric, errors="coerce")
            if blocked_ctrl_data is not None and not blocked_ctrl_data.empty:
                blocked_ctrl_data[blocked_ctrl_data.columns[2:]] = blocked_ctrl_data[blocked_ctrl_data.columns[2:]].apply(pd.to_numeric, errors="coerce")

            self.logger.info("Dataframes prepared successfully.")
            return selected_data, blank_data, pos_data, donor_data, acceptor_data, neg_ctrl_data, blocked_ctrl_data

        except Exception as e:
            self.logger.exception("Failed to prepare dataframes: %s", str(e))
            return None, None, None, None, None, None, None

    # -------------------------------------------------------------------------
    # BLANK SUBTRACTION
    # -------------------------------------------------------------------------
    def subtract_blank(self, selected_data, blank_data, time_col="Time [min]"):
        """
        Subtract the mean of the blank data from selected wells.

        Both dataframes must have aligned time columns.
        Returns a new dataframe with subtracted values.
        """
        if selected_data is None or selected_data.empty:
            self.logger.warning("Selected data empty; skipping subtraction.")
            return selected_data
        if blank_data is None or blank_data.empty:
            self.logger.warning("Blank data empty; skipping subtraction.")
            return selected_data

        try:
            selected_data[time_col] = pd.to_numeric(selected_data[time_col], errors="coerce")
            blank_data[time_col] = pd.to_numeric(blank_data[time_col], errors="coerce")

            selected_data.sort_values(by=[time_col], inplace=True)
            blank_data.sort_values(by=[time_col], inplace=True)

            # Compute mean per timepoint
            blank_means = blank_data.groupby(time_col).mean(numeric_only=True)
            blank_means = blank_means.mean(axis=1)

            # Apply subtraction
            result = selected_data.copy()
            result.set_index(time_col, inplace=True)
            result.iloc[:, :] = result.iloc[:, :].sub(blank_means, axis=0)
            result.reset_index(inplace=True)

            self.logger.info("Blank subtraction applied.")
            return result

        except Exception as exc:
            self.logger.error("Error during blank subtraction: %s", str(exc))
            return selected_data

