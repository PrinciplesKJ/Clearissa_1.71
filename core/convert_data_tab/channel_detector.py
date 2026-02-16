"""
Channel Detector - Automatic channel detection for FRET experiments
===================================================================

This module handles automatic detection of donor, acceptor, and FRET channels
from experimental data based on well labels and channel configuration settings.

The detector:
- Loads channel configuration from settings file or uses defaults
- Extracts well labels from all available datasets
- Matches labels against known channel patterns
- Scores and ranks candidate channel triads
- Selects the best-matching configuration

Author: Križan Jurinović
Date: October 2025
"""

from __future__ import annotations

import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ChannelTriad:
    """
    Represents a detected channel triad (Donor, Acceptor, FRET).
    
    Attributes:
        set_name: Name of the channel set (e.g., "Set 1")
        which: Channel set number (1 or 2)
        donor: Donor channel identifier
        acceptor: Acceptor channel identifier
        fret: FRET channel identifier
        score: Match score based on well label matches
    """
    set_name: str
    which: int
    donor: str
    acceptor: str
    fret: str
    score: int


@dataclass
class ChannelDetectionResult:
    """
    Result of channel detection with detailed match information.
    
    Attributes:
        selected_donor: Selected donor channel
        selected_acceptor: Selected acceptor channel
        selected_fret: Selected FRET channel
        resolved_set: Channel set number (1 or 2)
        resolved_set_name: Channel set name
        donor_info: Dict with donor matches and examples
        acceptor_info: Dict with acceptor matches and examples
        fret_info: Dict with FRET matches and examples
    """
    selected_donor: str
    selected_acceptor: str
    selected_fret: str
    resolved_set: int
    resolved_set_name: str
    donor_info: Dict[str, Any]
    acceptor_info: Dict[str, Any]
    fret_info: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format for compatibility."""
        return {
            "resolved_set": self.resolved_set,
            "resolved_set_name": self.resolved_set_name,
            "Donor": self.donor_info,
            "Acceptor": self.acceptor_info,
            "FRET": self.fret_info,
        }


class ChannelDetector:
    """
    Detects and configures channel assignments for FRET experiments.
    
    This class analyses well labels from experimental datasets and matches them
    against known channel configurations to automatically detect the correct
    donor, acceptor, and FRET channel assignments.
    """
    
    # Default channel configurations for common FRET setups
    DEFAULT_CHANNEL_SETS = {
        "Set 1": {
            "Donor 1": "488-14/535-30",
            "Acceptor 1": "540-20/590-30",
            "FRET 1": "488-14/583-30",
            "Donor 2": "540-20/590-30",
            "Acceptor 2": "625-30/680-30",
            "FRET 2": "546-40/690-40",
        },
        "Set 2": {
            "Donor 1": "500-20/550-30",
            "Acceptor 1": "560-20/600-30",
            "FRET 1": "500-20/600-30",
            "Donor 2": "610-30/700-40",
            "Acceptor 2": "720-40/800-50",
            "FRET 2": "610-30/800-50",
        },
    }
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialise the channel detector.
        
        Args:
            config_path: Optional path to channel settings JSON file.
                        If None, uses default configuration.
        """
        self.config_path = config_path
        self.channel_sets = self._load_channel_configuration()
        
    def _load_channel_configuration(self) -> Dict[str, Dict[str, str]]:
        """
        Load channel configuration from file or use defaults.
        
        Returns:
            Dictionary mapping channel set names to channel definitions
        """
        if not self.config_path or not self.config_path.exists():
            logger.debug("Using default channel configuration")
            return self.DEFAULT_CHANNEL_SETS.copy()
        
        try:
            with self.config_path.open("r") as f:
                data = json.load(f)
            
            if isinstance(data, dict) and isinstance(data.get("channels"), dict):
                logger.info(f"Loaded channel configuration from {self.config_path}")
                return data["channels"]
            else:
                logger.warning(f"Invalid channel configuration format in {self.config_path}")
                return self.DEFAULT_CHANNEL_SETS.copy()
                
        except Exception as e:
            logger.warning(f"Failed to load channel configuration: {e}. Using defaults.")
            return self.DEFAULT_CHANNEL_SETS.copy()
    
    @staticmethod
    def extract_well_labels_from_dataframe(df: Optional[pd.DataFrame]) -> List[str]:
        """
        Extract unique well labels from a DataFrame's 'Well' column.
        
        Args:
            df: DataFrame with 'Well' column containing channel identifiers
            
        Returns:
            List of unique well labels in order of first appearance
        """
        if df is None:
            return []
        
        try:
            if "Well" not in getattr(df, "columns", []):
                return []
            
            well_column = df["Well"].astype(str)
            seen = set()
            labels = []
            
            for label in well_column:
                if label not in seen:
                    seen.add(label)
                    labels.append(label)
            
            return labels
            
        except Exception as e:
            logger.debug(f"Failed to extract well labels: {e}")
            return []
    
    def extract_all_well_labels(
        self,
        *dataframes: Optional[pd.DataFrame]
    ) -> List[str]:
        """
        Extract all unique well labels from multiple DataFrames.
        
        Args:
            *dataframes: Variable number of DataFrames to extract labels from
            
        Returns:
            List of all unique well labels across all DataFrames
        """
        all_labels = []
        for df in dataframes:
            all_labels.extend(self.extract_well_labels_from_dataframe(df))
        
        # Deduplicate while preserving order
        seen = set()
        unique_labels = []
        for label in all_labels:
            if label not in seen:
                seen.add(label)
                unique_labels.append(label)
        
        return unique_labels
    
    def count_matches(self, channel_tag: str, well_labels: List[str]) -> int:
        """
        Count how many well labels match a given channel tag.
        
        Args:
            channel_tag: Channel identifier to search for (e.g., "488-14/535-30")
            well_labels: List of well labels to search in
            
        Returns:
            Number of well labels containing the channel tag
        """
        if not channel_tag:
            return 0
        
        try:
            pattern = re.compile(re.escape(str(channel_tag)), flags=re.IGNORECASE)
            return sum(1 for label in well_labels if pattern.search(label))
        except Exception as e:
            logger.debug(f"Pattern matching failed for '{channel_tag}': {e}")
            return 0
    
    def get_matching_examples(
        self,
        channel_tag: str,
        well_labels: List[str],
        max_examples: int = 5
    ) -> List[str]:
        """
        Get example well labels that match a channel tag.
        
        Args:
            channel_tag: Channel identifier to search for
            well_labels: List of well labels to search in
            max_examples: Maximum number of examples to return
            
        Returns:
            List of matching well label examples
        """
        if not channel_tag:
            return []
        
        try:
            pattern = re.compile(re.escape(str(channel_tag)), flags=re.IGNORECASE)
            examples = []
            
            for label in well_labels:
                if pattern.search(label):
                    examples.append(label)
                    if len(examples) >= max_examples:
                        break
            
            return examples
            
        except Exception as e:
            logger.debug(f"Example matching failed for '{channel_tag}': {e}")
            return []
    
    def build_channel_triads(self, well_labels: List[str]) -> List[ChannelTriad]:
        """
        Build and score all candidate channel triads from configuration.
        
        Args:
            well_labels: List of well labels from experimental data
            
        Returns:
            List of ChannelTriad objects sorted by score (highest first)
        """
        triads = []
        
        for set_name, mapping in self.channel_sets.items():
            # Extract channel identifiers
            d1 = mapping.get("Donor 1")
            a1 = mapping.get("Acceptor 1")
            f1 = mapping.get("FRET 1")
            d2 = mapping.get("Donor 2")
            a2 = mapping.get("Acceptor 2")
            f2 = mapping.get("FRET 2")
            
            # Build triad for first channel set
            if d1 and a1 and f1:
                score = (
                    self.count_matches(d1, well_labels) +
                    self.count_matches(a1, well_labels) +
                    self.count_matches(f1, well_labels)
                )
                triads.append(ChannelTriad(
                    set_name=set_name,
                    which=1,
                    donor=d1,
                    acceptor=a1,
                    fret=f1,
                    score=score
                ))
            
            # Build triad for second channel set
            if d2 and a2 and f2:
                score = (
                    self.count_matches(d2, well_labels) +
                    self.count_matches(a2, well_labels) +
                    self.count_matches(f2, well_labels)
                )
                triads.append(ChannelTriad(
                    set_name=set_name,
                    which=2,
                    donor=d2,
                    acceptor=a2,
                    fret=f2,
                    score=score
                ))
        
        # Sort by score (highest first)
        triads.sort(key=lambda t: t.score, reverse=True)
        return triads
    
    def select_best_triad(self, triads: List[ChannelTriad]) -> Optional[ChannelTriad]:
        """
        Select the best channel triad from candidates.
        
        If no matches are found (all scores are 0), falls back to the first
        available channel set.
        
        Args:
            triads: List of scored channel triads
            
        Returns:
            Best matching ChannelTriad or None if no configuration available
        """
        if not triads:
            logger.warning("No channel triads available")
            return None
        
        best = triads[0]
        
        # If no matches at all, use default (first set, channel 1)
        if best.score == 0:
            logger.info("No channel matches found, using default configuration")
            first_set_name = next(iter(self.channel_sets.keys()))
            mapping = self.channel_sets[first_set_name]
            
            return ChannelTriad(
                set_name=first_set_name,
                which=1,
                donor=mapping.get("Donor 1", ""),
                acceptor=mapping.get("Acceptor 1", ""),
                fret=mapping.get("FRET 1", ""),
                score=0
            )
        
        logger.info(
            f"Selected channel set: {best.set_name} (Set {best.which}) "
            f"with score {best.score}"
        )
        return best
    
    def detect_channels(
        self,
        *dataframes: Optional[pd.DataFrame]
    ) -> ChannelDetectionResult:
        """
        Detect optimal channel configuration from experimental data.
        
        This is the main entry point for channel detection. It:
        1. Extracts well labels from all provided DataFrames
        2. Builds candidate channel triads from configuration
        3. Scores each triad based on label matches
        4. Selects the best-matching configuration
        5. Returns detailed detection results
        
        Args:
            *dataframes: Variable number of DataFrames containing experimental data
            
        Returns:
            ChannelDetectionResult with selected channels and match details
            
        Raises:
            ValueError: If no valid channel configuration can be determined
        """
        # Extract all well labels
        well_labels = self.extract_all_well_labels(*dataframes)
        
        if not well_labels:
            logger.warning("No well labels found in provided data")
        
        # Build and score triads
        triads = self.build_channel_triads(well_labels)
        
        # Select best triad
        selected = self.select_best_triad(triads)
        
        if not selected:
            raise ValueError("Unable to determine channel configuration")
        
        # Build detailed result
        result = ChannelDetectionResult(
            selected_donor=selected.donor,
            selected_acceptor=selected.acceptor,
            selected_fret=selected.fret,
            resolved_set=selected.which,
            resolved_set_name=selected.set_name,
            donor_info={
                "selected": selected.donor,
                "matches": self.count_matches(selected.donor, well_labels),
                "examples": self.get_matching_examples(selected.donor, well_labels),
            },
            acceptor_info={
                "selected": selected.acceptor,
                "matches": self.count_matches(selected.acceptor, well_labels),
                "examples": self.get_matching_examples(selected.acceptor, well_labels),
            },
            fret_info={
                "selected": selected.fret,
                "matches": self.count_matches(selected.fret, well_labels),
                "examples": self.get_matching_examples(selected.fret, well_labels),
            },
        )
        
        logger.info(
            f"Channel detection complete: "
            f"Donor={selected.donor} ({result.donor_info['matches']} matches), "
            f"Acceptor={selected.acceptor} ({result.acceptor_info['matches']} matches), "
            f"FRET={selected.fret} ({result.fret_info['matches']} matches)"
        )
        
        return result

