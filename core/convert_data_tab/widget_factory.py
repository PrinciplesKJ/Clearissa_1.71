"""
Widget Factory - Centralized UI widget creation for Convert Data Tab
====================================================================

This module provides factory functions and classes for creating complex
UI widgets used in the Convert Data Tab. It separates widget construction
logic from business logic and event handling.

Key components:
- Information panels with mode-specific instructions
- Matrix tables for coefficient and slope display

Author: Križan Jurinović
Date: October 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import logging

from PyQt5.QtWidgets import (
    QLabel, QTableWidget, QHeaderView, QAbstractScrollArea
)

if TYPE_CHECKING:
    from .gui import ConvertDataTab

logger = logging.getLogger(__name__)


class WidgetFactory:
    """
    Factory class for creating complex UI widgets for the Convert Data Tab.

    This class provides static and instance methods for building UI components
    that are reused across different conversion modes.
    """

    @staticmethod
    def create_matrix_table() -> QTableWidget:
        """
        Create a formatted table widget for coefficient/slope display.

        Returns:
            Configured QTableWidget with proper styling and sizing
        """
        table = QTableWidget()
        table.setStyleSheet("""
            QTableWidget {
                font-size: 8pt;
                gridline-color: #DDD;
            }
            QHeaderView::section {
                background-color: #F5F5F5;
                padding: 2px;
                border: 1px solid #DDD;
                font-weight: bold;
                font-size: 8pt;
            }
            QTableWidget::item {
                padding: 2px;
            }
        """)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        table.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        table.setMinimumHeight(100)
        table.setMaximumHeight(200)
        return table

    @staticmethod
    def create_info_panel(
        text: str,
        background: str = "#E8F5E9",
        border: str = "#81C784",
        text_color: str = "#2E7D32"
    ) -> QLabel:
        """
        Create an informational panel with styled text.

        Args:
            text: Information text to display
            background: Background colour (hex)
            border: Border colour (hex)
            text_color: Text colour (hex)

        Returns:
            Styled QLabel with word wrapping enabled
        """
        label = QLabel(text)
        label.setStyleSheet(f"""
            QLabel {{
                background-color: {background};
                border: 1px solid {border};
                border-radius: 3px;
                padding: 5px;
                font-size: 8pt;
                color: {text_color};
            }}
        """)
        label.setWordWrap(True)
        return label

    @staticmethod
    def create_tmsd_info_panel() -> QLabel:
        """Create information panel for TMSD mode."""
        return WidgetFactory.create_info_panel(
            "One-step Toehold-Mediated Strand Displacement\n"
            "Reaction: Ax+B→AB+x (Ax=blocked substrate, AB=product)\n"
            "Uses 'nuking' to determine total [Ax]₀ via excess invader\n"
            "Corrects for residual Ax fluorescence (quenched acceptor)",
            background="#E8F5E9",
            border="#81C784",
            text_color="#2E7D32"
        )

    @staticmethod
    def create_hmsd_info_panel() -> QLabel:
        """Create information panel for HMSD mode."""
        return WidgetFactory.create_info_panel(
            "One-step FRET/Handhold Mediated Strand Displacement\n"
            "Reaction: B1AA+B2→B1B2+AA (donor-first initialisation)\n"
            "Uses ONLY donor channel: Y_D(t) = [B2](t) + β·[B1B2](t)\n"
            "β = slope(B1B2)/slope(B2) determines FRET efficiency\n"
            "Conservation: [B2](t) + [B1B2](t) = [B2]₀",
            background="#FFF3E0",
            border="#FFB74D",
            text_color="#E65100"
        )

    @staticmethod
    def create_mass_action_info_panel() -> QLabel:
        """Create information panel for mass-action catalytic mode."""
        return WidgetFactory.create_info_panel(
            "Catalytic 2-step TMSD with FRET readout\n"
            "Reactions: (1) B1L+AA→B1AA+L  (2) B1AA+B2→B1B2+AA\n"
            "Conservation: B1L+B1AA+B1B2=B1L₀ (fixed), B2+B1B2=B2₀\n"
            "Solves for B1B2 & B1AA using Donor+FRET channels\n"
            "⚠️  Acceptor calibrations required but not used in solver",
            background="#E3F2FD",
            border="#90CAF9",
            text_color="#1565C0"
        )
