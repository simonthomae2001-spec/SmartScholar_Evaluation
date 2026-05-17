"""
graph_state.py — Central state schema for the SmartScholar LangGraph pipeline.

Defines:
    ResearchConfig  – profile-dependent parameters (Fast / Medium / Pro).
    GraphState      – the TypedDict that flows through every LangGraph node.

Other developers: import GraphState wherever you need to type-hint the
state dictionary that travels between nodes.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import TypedDict


# ------------------------------------------------------------------ #
#  Research configuration profiles
# ------------------------------------------------------------------ #

@dataclass
class ResearchConfig:
    """
    Profile-dependent tunables that control how aggressively the
    ResearcherAgent searches and filters.

    Profiles
    --------
    fast   – quick scan, few queries, small active-paper set.
    medium – balanced defaults (good for most tasks).
    pro    – exhaustive search, larger result pool, more papers kept.
    """

    max_queries: int = 4
    results_per_query: int = 5
    active_paper_count: int = 5

    # ---- Factory -------------------------------------------------- #

    _PROFILES: dict[str, dict] = None  # populated in __init_subclass__

    @classmethod
    def from_profile(cls, name: str) -> "ResearchConfig":
        """
        Return a ResearchConfig pre-filled for the given profile name.

        Parameters
        ----------
        name : str
            One of ``"fast"``, ``"medium"``, ``"pro"`` (case-insensitive).

        Raises
        ------
        ValueError
            If *name* is not a recognised profile.
        """
        profiles = {
            "fast":   cls(max_queries=3, results_per_query=3, active_paper_count=3),
            "medium": cls(max_queries=4, results_per_query=5, active_paper_count=5),
            "pro":    cls(max_queries=5, results_per_query=10, active_paper_count=8),
        }
        key = name.strip().lower()
        if key not in profiles:
            raise ValueError(
                f"Unknown profile '{name}'. Choose from: {list(profiles.keys())}"
            )
        return profiles[key]

    def to_dict(self) -> dict:
        """Serialize to a plain dict (safe for JSON / LangGraph state)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ResearchConfig":
        """Re-hydrate from a plain dict."""
        return cls(**data)


# ------------------------------------------------------------------ #
#  LangGraph state schema
# ------------------------------------------------------------------ #

class GraphState(TypedDict, total=False):
    """
    The shared state dictionary that travels through every LangGraph
    node in the SmartScholar pipeline.

    Attributes
    ----------
    user_query : str
        The raw research topic entered by the user.
    config_profile : str
        One of ``"fast"``, ``"medium"``, ``"pro"``.
    research_config : dict
        Serialized :class:`ResearchConfig` (created from *config_profile*).
    is_valid : bool
        Set by the GatekeeperAgent.  ``True`` means the query is
        acceptable for research.  Defaults to ``True`` (placeholder).
    search_queries : list[str]
        Refined academic search terms produced by the ResearcherAgent.
    active_papers : list[dict]
        Papers the user has accepted for the final review.
    discarded_papers : list[dict]
        Papers that scored below the cut-off (available for swap-in).
    final_review : str
        The literature review / synthesis produced by the AnalystAgent.
    """

    user_query: str
    config_profile: str
    research_config: dict
    is_valid: bool
    search_queries: list[str]
    active_papers: list[dict]
    discarded_papers: list[dict]
    final_review: str
