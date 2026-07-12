"""
config.py — Centralized configuration provider for SmartScholar.

Loads the central config.yaml file to enforce a strict Single Source of Truth
for all profile settings, numeric limits, and system parameters.
"""

from __future__ import annotations

import os
import yaml
from functools import lru_cache

# Resolve the absolute path to config.yaml at the project root
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config.yaml")

@lru_cache(maxsize=1)
def _load_config() -> dict:
    """Load and parse the config.yaml file once and cache it in memory."""
    if not os.path.exists(_CONFIG_PATH):
        raise FileNotFoundError(f"Configuration file missing: {_CONFIG_PATH}")
        
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_config(profile_name: str) -> dict:
    """
    Return the configuration dict for the requested profile.

    Parameters
    ----------
    profile_name : str
        One of ``"fast"``, ``"medium"``, ``"pro"`` (case-insensitive).

    Returns
    -------
    dict
        The configuration dictionary for the profile, containing limits
        like ``max_queries``, ``top_n_papers``, ``read_depth``, etc.

    Raises
    ------
    ValueError
        If *profile_name* is not recognised.
    """
    config_data = _load_config()
    profiles = config_data.get("profiles", {})
    
    key = profile_name.strip().lower()
    if key not in profiles:
        raise ValueError(
            f"Unknown profile '{profile_name}'. "
            f"Available profiles: {list(profiles.keys())}"
        )
        
    return profiles[key]

def list_profiles() -> list[str]:
    """Return the names of all available profiles."""
    config_data = _load_config()
    return list(config_data.get("profiles", {}).keys())

def get_system_config() -> dict:
    """
    Return the global system configuration dict.

    Returns
    -------
    dict
        The system configuration dictionary, containing global
        settings like network timeouts and UI truncation limits.
    """
    config_data = _load_config()
    return config_data.get("system", {})
