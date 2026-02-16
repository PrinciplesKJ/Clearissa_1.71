# -*- coding: utf-8 -*-
"""
Clearissa - Modern Welcome Panel
---------------------------------
Welcome interface with responsive design.
Automatically adapts to available screen space with uniform styling.

Author: Križan Jurinović
Date: November 2025
"""

from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QFont, QColor, QLinearGradient, QPainter, QBrush
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QGraphicsDropShadowEffect, QSizePolicy, QScrollArea
)
import os
import sys

# Import from resource_utils at project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from resource_utils import get_resource_path


class FeatureCard(QFrame):
    """
    Clean, modern card widget for feature display.
    """
    clicked = pyqtSignal()

    def __init__(self, icon, title, description, parent=None):
        super().__init__(parent)
        self._icon = icon
        self._title = title
        self._description = description

        self.setMinimumHeight(110)
        self.setMinimumWidth(150)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.setCursor(Qt.PointingHandCursor)

        # Clean card styling
        self.setStyleSheet("""
            QFrame {
                background: white;
                border: 2px solid #E0E0E0;
                border-radius: 12px;
            }
            QFrame:hover {
                background: #F8F9FA;
                border: 2px solid #2196F3;
            }
        """)

        # Subtle shadow
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(10)
        shadow.setXOffset(0)
        shadow.setYOffset(2)
        shadow.setColor(QColor(0, 0, 0, 20))
        self.setGraphicsEffect(shadow)

        # Create layout
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setAlignment(Qt.AlignCenter)

        # Icon
        icon_label = QLabel(self._icon)
        icon_label.setFont(QFont("Segoe UI", 32))
        icon_label.setAlignment(Qt.AlignCenter)
        icon_label.setStyleSheet("background: transparent; color: #2196F3; border: none;")
        layout.addWidget(icon_label)

        # Title
        title_label = QLabel(self._title)
        title_label.setFont(QFont("Segoe UI", 11, QFont.DemiBold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("background: transparent; color: #212121; border: none;")
        layout.addWidget(title_label)

        # Description
        desc_label = QLabel(self._description)
        desc_label.setFont(QFont("Segoe UI", 9))
        desc_label.setAlignment(Qt.AlignCenter)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("background: transparent; color: #757575; border: none;")
        layout.addWidget(desc_label)

    def mousePressEvent(self, event):
        """Emit clicked signal on mouse press."""
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class ModernButton(QPushButton):
    """
    Styled button widget for the welcome panel.
    """

    def __init__(self, text, icon="", primary=False, parent=None):
        super().__init__(parent)
        self.primary = primary
        self.icon = icon

        # Set text with icon if provided
        if icon:
            self.setText(f"{icon}  {text}")
        else:
            self.setText(text)

        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(50)
        self.setFont(QFont("Segoe UI", 10, QFont.Medium))
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        self._apply_style()

    def _apply_style(self):
        """Apply clean modern styling."""
        if self.primary:
            self.setStyleSheet("""
                QPushButton {
                    background: #2196F3;
                    color: white;
                    border: none;
                    border-radius: 8px;
                    padding: 14px 28px;
                    font-weight: 600;
                }
                QPushButton:hover {
                    background: #1976D2;
                }
                QPushButton:pressed {
                    background: #1565C0;
                }
            """)
            shadow = QGraphicsDropShadowEffect(self)
            shadow.setBlurRadius(12)
            shadow.setXOffset(0)
            shadow.setYOffset(3)
            shadow.setColor(QColor(33, 150, 243, 100))
            self.setGraphicsEffect(shadow)
        else:
            self.setStyleSheet("""
                QPushButton {
                    background: white;
                    color: #424242;
                    border: 2px solid #E0E0E0;
                    border-radius: 8px;
                    padding: 12px 24px;
                    font-weight: 500;
                }
                QPushButton:hover {
                    background: #F5F5F5;
                    border: 2px solid #2196F3;
                }
                QPushButton:pressed {
                    background: #EEEEEE;
                }
            """)
            shadow = QGraphicsDropShadowEffect(self)
            shadow.setBlurRadius(8)
            shadow.setXOffset(0)
            shadow.setYOffset(2)
            shadow.setColor(QColor(0, 0, 0, 20))
            self.setGraphicsEffect(shadow)



class WelcomePanel(QWidget):
    """
    Welcome panel with responsive single-column layout.
    Automatically adapts to available screen space.
    """

    # Define signals
    load_folder_clicked = pyqtSignal()
    view_data_clicked = pyqtSignal()
    kinetics_clicked = pyqtSignal()
    manual_clicked = pyqtSignal()
    options_clicked = pyqtSignal()

    def __init__(self, version="1.23", parent=None):
        super().__init__(parent)
        self.version = version
        self._init_ui()

    def paintEvent(self, event):
        """Paint soft gradient background."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Soft gradient background
        gradient = QLinearGradient(0, 0, self.width(), self.height())
        gradient.setColorAt(0.0, QColor(245, 247, 250))
        gradient.setColorAt(1.0, QColor(235, 240, 248))
        painter.fillRect(self.rect(), QBrush(gradient))

    def _init_ui(self):
        """Initialise clean, responsive UI."""
        # Create scroll area for small screens
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")

        # Content widget
        content = QWidget()
        content.setStyleSheet("background: transparent;")

        # Main layout
        main_layout = QVBoxLayout(content)
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(20)
        main_layout.setAlignment(Qt.AlignTop | Qt.AlignHCenter)

        # Header section - Logo on left, Title/Subtitle/Button on right
        header_container = QWidget()
        header_container.setStyleSheet("background: transparent;")
        header_container.setMaximumWidth(750)
        header_layout = QHBoxLayout(header_container)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(40)
        header_layout.setAlignment(Qt.AlignCenter)

        # Logo on the left
        logo_label = QLabel()
        logo_label.setAlignment(Qt.AlignCenter)
        logo_label.setStyleSheet("background: transparent;")
        logo_label.setMinimumSize(200, 200)
        logo_label.setMaximumSize(200, 200)
        try:
            logo_path = get_resource_path("clearissa_logo.png")
            if os.path.exists(logo_path):
                pixmap = QPixmap(logo_path)
                scaled_pixmap = pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                logo_label.setPixmap(scaled_pixmap)
            else:
                logo_label.setText("🔬")
                logo_label.setFont(QFont("Segoe UI", 60))
                logo_label.setStyleSheet("color: #2196F3; background: transparent;")
        except:
            logo_label.setText("🔬")
            logo_label.setFont(QFont("Segoe UI", 60))
            logo_label.setStyleSheet("color: #2196F3; background: transparent;")

        header_layout.addWidget(logo_label, alignment=Qt.AlignVCenter)

        # Title, subtitle, and button on the right - vertically centred
        right_container = QWidget()
        right_container.setStyleSheet("background: transparent;")
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)
        right_layout.setAlignment(Qt.AlignVCenter | Qt.AlignCenter)

        # Title
        title_label = QLabel("Welcome to Clearissa")
        title_label.setFont(QFont("Segoe UI", 26, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #212121; background: transparent;")
        right_layout.addWidget(title_label)

        # Subtitle
        subtitle_label = QLabel("Data Analysis Tool")
        subtitle_label.setFont(QFont("Segoe UI", 14))
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setStyleSheet("color: #616161; background: transparent;")
        right_layout.addWidget(subtitle_label)

        right_layout.addSpacing(8)

        # Load CSV button
        self.load_folder_btn = ModernButton("Load CSV Folder", "📁", primary=True)
        self.load_folder_btn.setMinimumWidth(280)
        self.load_folder_btn.setMaximumWidth(350)
        self.load_folder_btn.clicked.connect(self.load_folder_clicked.emit)
        right_layout.addWidget(self.load_folder_btn, alignment=Qt.AlignCenter)

        header_layout.addWidget(right_container, alignment=Qt.AlignVCenter)

        main_layout.addWidget(header_container, alignment=Qt.AlignCenter)
        main_layout.addSpacing(20)

        # Feature cards - responsive grid
        cards_container = QWidget()
        cards_container.setStyleSheet("background: transparent;")
        cards_container.setMaximumWidth(800)
        cards_layout = QHBoxLayout(cards_container)
        cards_layout.setSpacing(15)
        cards_layout.setContentsMargins(0, 0, 0, 0)

        self.view_data_card = FeatureCard("📊", "Data Viewer", "Explore and Analyse")
        self.view_data_card.clicked.connect(self.view_data_clicked.emit)
        cards_layout.addWidget(self.view_data_card)

        self.kinetics_card = FeatureCard("⚡", "Kinetics", "Process Kinetic Data")
        self.kinetics_card.clicked.connect(self.kinetics_clicked.emit)
        cards_layout.addWidget(self.kinetics_card)

        main_layout.addWidget(cards_container, alignment=Qt.AlignCenter)

        main_layout.addSpacing(30)

        # Utility buttons
        utility_container = QWidget()
        utility_container.setStyleSheet("background: transparent;")
        utility_layout = QHBoxLayout(utility_container)
        utility_layout.setSpacing(12)
        utility_layout.setContentsMargins(0, 0, 0, 0)

        self.manual_btn = ModernButton("User Manual", "📖")
        self.manual_btn.setMinimumWidth(140)
        self.manual_btn.clicked.connect(self.manual_clicked.emit)
        utility_layout.addWidget(self.manual_btn)

        self.options_btn = ModernButton("Options", "⚙️")
        self.options_btn.setMinimumWidth(140)
        self.options_btn.clicked.connect(self.options_clicked.emit)
        utility_layout.addWidget(self.options_btn)

        main_layout.addWidget(utility_container, alignment=Qt.AlignCenter)

        main_layout.addSpacing(20)

        # Footer with credits
        footer = self._create_footer()
        main_layout.addWidget(footer)

        main_layout.addStretch()

        # Set scroll content
        scroll.setWidget(content)

        # Main panel layout
        panel_layout = QVBoxLayout(self)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.addWidget(scroll)


    def _create_footer(self):
        """Create clean footer with version info."""
        container = QWidget()
        container.setStyleSheet("background: transparent;")

        layout = QVBoxLayout(container)
        layout.setSpacing(8)
        layout.setContentsMargins(0, 0, 0, 0)

        # Divider
        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setFixedHeight(1)
        divider.setStyleSheet("background: #E0E0E0;")
        layout.addWidget(divider)

        layout.addSpacing(12)

        # Credits
        credits_label = QLabel(
            f"Clearissa v{self.version}<br>"
            "Developed by <b>Križan Jurinović</b><br>"
            "Tom Ouldridge Research Group · Imperial College London"
        )
        credits_label.setFont(QFont("Segoe UI", 9))
        credits_label.setAlignment(Qt.AlignCenter)
        credits_label.setStyleSheet("color: #757575; background: transparent;")
        credits_label.setTextFormat(Qt.RichText)
        credits_label.setWordWrap(True)
        layout.addWidget(credits_label)

        # Contact
        contact_label = QLabel(
            '<a href="mailto:k.jurinovic22@imperial.ac.uk" '
            'style="color: #2196F3; text-decoration: none;">k.jurinovic22@imperial.ac.uk</a>'
        )
        contact_label.setFont(QFont("Segoe UI", 9))
        contact_label.setAlignment(Qt.AlignCenter)
        contact_label.setOpenExternalLinks(True)
        contact_label.setTextFormat(Qt.RichText)
        contact_label.setStyleSheet("background: transparent;")
        layout.addWidget(contact_label)

        return container


