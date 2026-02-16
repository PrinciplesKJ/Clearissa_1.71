# core/convert_data_tab/species.py

from typing import Dict

class SpeciesManager:
    """Manages species categorisation, colours, and sorting for UI components."""

    COLORS: Dict[str, str] = {
        "donor": "#2E8B57",
        "acceptor": "#FF6347",
        "fret": "#4169E1",
        "blocked": "#8B4513",
        "reference": "#9400D3",
        "default": "#2C3E50",
    }

    KEYWORDS = {
        "donor": ["donor"],
        "acceptor": ["acceptor"],
        "fret": ["fret"],
        "blocked": ["blocker", "blocked"],
    }

    @classmethod
    def categorise_species(cls, name: str, hint: str | None = None) -> str:
        name_lower = (name or "").lower()
        for category, keys in cls.KEYWORDS.items():
            if any(k in name_lower for k in keys):
                return category
        if hint:
            hint_lower = hint.lower()
            for category, keys in cls.KEYWORDS.items():
                if any(k in hint_lower for k in keys):
                    return category
        return "default"

    @classmethod
    def get_priority_score(cls, name: str, hint: str | None = None,
                           preferred: list[str] | None = None) -> tuple[int, str]:
        category = cls.categorise_species(name, hint)
        if preferred and category in preferred:
            tier = preferred.index(category)
        else:
            order = ["reference", "fret", "donor", "acceptor", "blocked"]
            tier = order.index(category) if category in order else len(order)
        return (tier, name.lower())
