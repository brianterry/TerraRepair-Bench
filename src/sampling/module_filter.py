"""
Filter criteria for TerraDS modules: AWS provider, complexity, stars.

Used to validate candidates before or after scanning.
"""
from __future__ import annotations


def passes_provider_filter(resource_types: list[str], provider: str = "aws") -> bool:
    """Check that module has at least one resource of the given provider."""
    prefix = f"{provider}_"
    return any(rt.startswith(prefix) for rt in resource_types)


def passes_complexity_filter(
    resource_types: list[str], min_distinct_types: int = 4
) -> bool:
    """Check that module has >= min_distinct_types distinct resource types."""
    return len(set(resource_types)) >= min_distinct_types


def passes_stars_filter(stars: int, min_stars: int = 5) -> bool:
    """Check that repository has >= min_stars."""
    return stars >= min_stars


def filter_candidates(
    candidates: list[dict],
    min_resource_types: int = 4,
    min_stars: int = 5,
    provider: str = "aws",
) -> list[dict]:
    """
    Apply all filter criteria to a list of candidate modules.

    Returns candidates that pass provider, complexity, and stars filters.
    """
    result = []
    for c in candidates:
        if not passes_provider_filter(c.get("resource_types", []), provider):
            continue
        if not passes_complexity_filter(c.get("resource_types", []), min_resource_types):
            continue
        if not passes_stars_filter(c.get("stars", 0), min_stars):
            continue
        result.append(c)
    return result
