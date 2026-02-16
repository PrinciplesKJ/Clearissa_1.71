"""
Matrix Renderer - Coefficient and slope matrix display for Convert Data Tab
===========================================================================

This module handles the calculation, formatting, and rendering of calibration
matrices showing brightness ratios and calibration slopes for FRET experiments.

The matrices provide visual feedback on:
- Brightness ratios (β, α, γ) normalized to product species
- Calibration slopes in AFU/nM for each species-channel pair
- Bleed-through contributions via color-coded cells

Key features:
- Automatic matrix population from species selections
- Color-coded cells for easy interpretation
- Tooltips with detailed calibration information
- Support for HMSD (3×3) and mass-action (3×4) grids

Author: Križan Jurinović
Date: October 19, 2025
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple, Any, Optional, TYPE_CHECKING

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem

from core.common.ui_theme import Colors

if TYPE_CHECKING:
    from .gui import ConvertDataTab

logger = logging.getLogger(__name__)


class MatrixRenderer:
    """
    Renders calibration coefficient and slope matrices for FRET conversions.

    This class handles the visualisation of calibration data in matrix form,
    showing both normalised brightness ratios and absolute calibration slopes.
    """

    # Colour scheme for matrix cells
    COLORS = {
        'product': "#E8F5E9",      # Light green for B1B2 (product)
        'high_bleed': "#FFEBEE",    # Light red for high bleed-through (>0.5)
        'medium_bleed': "#FFF3E0",  # Light orange for medium bleed (>0.1)
        'low_bleed': Colors.CARD_BACKGROUND,  # White for low bleed
        'donor_like': "#E3F2FD",    # Light blue for donor-like species
        'acceptor_like': "#F3E5F5", # Light purple for acceptor-like species
        'missing': "#999999",       # Grey for missing data
    }

    def __init__(self, parent: ConvertDataTab):
        """
        Initialise the matrix renderer.

        Args:
            parent: Parent ConvertDataTab widget
        """
        self.parent = parent

    def refresh_matrices_for_mode(self, mode: str) -> None:
        """
        Refresh both coefficient and slope matrices for the given mode.

        Args:
            mode: Conversion mode ("mass", etc.)
        """
        try:
            if mode == "mass":
                self._render_mass_action_matrices()
            else:
                # Clear tables for modes that don't use calibration species
                # (tmsd, internal_tmsd, normalised_sd, hmsd use empirical beta)
                self._clear_matrices()

        except Exception as e:
            logger.debug(f"Matrix refresh failed for mode '{mode}': {e}")

    def _render_mass_action_matrices(self) -> None:
        """Render matrices for mass-action catalytic mode."""
        channels = ["Donor", "Acceptor", "FRET"]
        species = ["B1B2", "B1L", "B1AA", "B2"]

        # Collect species calibration data
        species_map = self._collect_species_data_mass(channels, species)

        # Populate both tables
        self._populate_coefficient_table(
            self.parent.coeff_table,
            species_map,
            channels,
            species
        )
        self._populate_slope_table(
            self.parent.bleed_table,
            species_map,
            channels,
            species
        )


    def _collect_species_data_mass(
        self,
        channels: List[str],
        species: List[str]
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        Collect calibration data for mass-action mode from GUI selections.

        Args:
            channels: List of channel names
            species: List of species names

        Returns:
            Dictionary mapping (channel, species) tuples to calibration data
        """
        species_map = {}

        for ch in channels:
            for sp in species:
                combo = self.parent.mass_combos.get((ch, sp))
                if combo:
                    selected = combo.currentText().strip()
                    if selected and selected != "NONE":
                        cal_data = self.parent.get_species_calibration(selected)
                        if cal_data:
                            species_map[(ch, sp)] = cal_data

        return species_map

    def _populate_coefficient_table(
        self,
        table: QTableWidget,
        species_map: Dict[Tuple[str, str], Dict[str, Any]],
        channels: List[str],
        species: List[str],
        species_labels: Optional[List[str]] = None
    ) -> None:
        """
        Populate table with brightness ratios normalized to B1B2 = 1.0.

        Each cell shows the ratio: slope(species, channel) / slope(B1B2, channel)
        This represents the relative brightness contribution to each channel.

        Args:
            table: QTableWidget to populate
            species_map: Calibration data for all species-channel pairs
            channels: List of channel names (rows)
            species: List of species names (columns)
            species_labels: Optional display labels for species (defaults to species)
        """
        table.setRowCount(len(channels))
        table.setColumnCount(len(species))
        table.setHorizontalHeaderLabels(species_labels if species_labels else species)
        table.setVerticalHeaderLabels(channels)

        for r, channel in enumerate(channels):
            # Get B1B2 slope for this channel (reference = 1.0)
            b1b2_data = species_map.get((channel, "B1B2"), {})
            b1b2_slope = b1b2_data.get("slope")

            for c, sp in enumerate(species):
                species_data = species_map.get((channel, sp), {})
                species_slope = species_data.get("slope")

                item = QTableWidgetItem()

                if b1b2_slope and species_slope and abs(b1b2_slope) > 1e-12:
                    ratio = species_slope / b1b2_slope
                    item.setText(f"{ratio:.4f}")

                    # Colour code based on magnitude
                    color = self._get_coefficient_color(sp, ratio)
                    item.setBackground(QBrush(QColor(color)))

                    # Add tooltip
                    species_name = species_data.get("species", sp)
                    item.setToolTip(
                        f"{species_name} in {channel} channel\n"
                        f"Brightness ratio: {ratio:.4f}\n"
                        f"(normalized to B1B2 = 1.0)"
                    )
                else:
                    item.setText("—")
                    item.setForeground(QBrush(QColor(self.COLORS['missing'])))
                    item.setToolTip("No calibration data available")

                item.setTextAlignment(Qt.AlignCenter)
                table.setItem(r, c, item)

    def _populate_slope_table(
        self,
        table: QTableWidget,
        species_map: Dict[Tuple[str, str], Dict[str, Any]],
        channels: List[str],
        species: List[str],
        species_labels: Optional[List[str]] = None
    ) -> None:
        """
        Populate table with actual calibration slopes in AFU/nM.

        Each cell shows the measured slope from calibration data.

        Args:
            table: QTableWidget to populate
            species_map: Calibration data for all species-channel pairs
            channels: List of channel names (rows)
            species: List of species names (columns)
            species_labels: Optional display labels for species (defaults to species)
        """
        table.setRowCount(len(channels))
        table.setColumnCount(len(species))
        table.setHorizontalHeaderLabels(species_labels if species_labels else species)
        table.setVerticalHeaderLabels(channels)

        for r, channel in enumerate(channels):
            for c, sp in enumerate(species):
                species_data = species_map.get((channel, sp), {})
                species_slope = species_data.get("slope")
                species_name = species_data.get("species", "")

                item = QTableWidgetItem()

                if species_slope is not None:
                    item.setText(f"{species_slope:.1f}")

                    # Colour code by species type
                    color = self._get_slope_color(sp)
                    item.setBackground(QBrush(QColor(color)))

                    # Add detailed tooltip
                    item.setToolTip(
                        f"Species: {species_name}\n"
                        f"Channel: {channel}\n"
                        f"Slope: {species_slope:.4f} AFU/nM"
                    )
                else:
                    item.setText("—")
                    item.setForeground(QBrush(QColor(self.COLORS['missing'])))
                    item.setToolTip("No calibration data available")

                item.setTextAlignment(Qt.AlignCenter)
                table.setItem(r, c, item)

    def _get_coefficient_color(self, species: str, ratio: float) -> str:
        """
        Determine cell color based on brightness ratio.

        Args:
            species: Species name
            ratio: Brightness ratio value

        Returns:
            Hex color code
        """
        if species == "B1B2":
            # Product species - always green
            return self.COLORS['product']
        elif ratio > 0.5:
            # High bleed-through - red warning
            return self.COLORS['high_bleed']
        elif ratio > 0.1:
            # Medium bleed-through - orange caution
            return self.COLORS['medium_bleed']
        else:
            # Low bleed-through - white (good)
            return self.COLORS['low_bleed']

    def _get_slope_color(self, species: str) -> str:
        """
        Determine cell color based on species type.

        Args:
            species: Species name

        Returns:
            Hex color code
        """
        if species == "B1B2":
            return self.COLORS['product']
        elif species in ("B2", "B1L"):
            return self.COLORS['donor_like']
        elif species == "B1AA":
            return self.COLORS['acceptor_like']
        else:
            return self.COLORS['low_bleed']

    def _clear_matrices(self) -> None:
        """Clear both coefficient and slope tables."""
        for table in [self.parent.coeff_table, self.parent.bleed_table]:
            table.clear()
            table.setRowCount(0)
            table.setColumnCount(0)


class MatrixFormatter:
    """
    Helper class for formatting matrix cell values and colors.

    Provides static methods for consistent formatting across all matrices.
    """

    @staticmethod
    def format_ratio(ratio: float, precision: int = 4) -> str:
        """
        Format a brightness ratio for display.

        Args:
            ratio: Ratio value
            precision: Number of decimal places

        Returns:
            Formatted string
        """
        return f"{ratio:.{precision}f}"

    @staticmethod
    def format_slope(slope: float, precision: int = 1) -> str:
        """
        Format a calibration slope for display.

        Args:
            slope: Slope value in AFU/nM
            precision: Number of decimal places

        Returns:
            Formatted string
        """
        return f"{slope:.{precision}f}"

    @staticmethod
    def assess_bleed_through(ratio: float) -> str:
        """
        Assess bleed-through severity from brightness ratio.

        Args:
            ratio: Brightness ratio

        Returns:
            Assessment string ("low", "medium", "high")
        """
        if ratio > 0.5:
            return "high"
        elif ratio > 0.1:
            return "medium"
        else:
            return "low"

    @staticmethod
    def get_bleed_description(species: str, channel: str, ratio: float) -> str:
        """
        Generate human-readable bleed-through description.

        Args:
            species: Species name
            channel: Channel name
            ratio: Brightness ratio

        Returns:
            Description string
        """
        severity = MatrixFormatter.assess_bleed_through(ratio)
        percentage = ratio * 100

        if severity == "high":
            return (
                f"⚠️ High bleed: {species} contributes {percentage:.1f}% "
                f"to {channel} channel (reference: B1B2 = 100%)"
            )
        elif severity == "medium":
            return (
                f"⚡ Medium bleed: {species} contributes {percentage:.1f}% "
                f"to {channel} channel"
            )
        else:
            return (
                f"✓ Low bleed: {species} contributes {percentage:.1f}% "
                f"to {channel} channel"
            )

