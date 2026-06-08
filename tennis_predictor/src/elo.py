"""Surface-specific Elo rating system for tennis players."""
from __future__ import annotations

from typing import Iterable

import pandas as pd

SURFACES = ("hard", "clay", "grass", "indoor")
DEFAULT_ELO = 1500.0
ELO_SCALE = 400.0

# K-factors by tournament level
TOURNAMENT_K: dict[str, float] = {
    "Grand Slam":  40.0,
    "Masters":     30.0,
    "ATP 500":     25.0,
    "ATP 250":     20.0,
    "Challenger":  15.0,
    "ITF":         10.0,
}
DEFAULT_K = 15.0

# Below this number of surface matches, blend toward overall average
MIN_SURFACE_MATCHES = 5
SURFACE_BLEND_WEIGHT = 0.30  # weight on overall average when sparse


def initialize_elo(
    players: Iterable[str],
    surfaces: tuple[str, ...] = SURFACES,
    base_rating: float = DEFAULT_ELO,
) -> dict[str, dict[str, float]]:
    """Returns {player: {surface: rating}} initialised to base_rating."""
    return {p: {s: base_rating for s in surfaces} for p in players}


def expected_score(rating_a: float, rating_b: float) -> float:
    """Standard Elo expected score: 1 / (1 + 10^((B-A)/400))."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / ELO_SCALE))


def tournament_k_factor(tournament_level: str) -> float:
    """Returns K-factor for the given tournament level."""
    return TOURNAMENT_K.get(tournament_level, DEFAULT_K)


def update_elo_after_match(
    elo_a: float,
    elo_b: float,
    winner: str,
    tournament_level: str = "ATP 250",
) -> tuple[float, float]:
    """
    Updates Elo ratings after a binary win/loss (no goal-difference multiplier).
    winner: "a" → A won; "b" → B won.
    Returns (new_elo_a, new_elo_b).
    """
    k = tournament_k_factor(tournament_level)
    exp_a = expected_score(elo_a, elo_b)
    score_a = 1.0 if winner == "a" else 0.0
    delta = k * (score_a - exp_a)
    return elo_a + delta, elo_b - delta


def build_elo_history(
    matches_df: pd.DataFrame,
    base_rating: float = DEFAULT_ELO,
) -> dict[str, dict[str, float]]:
    """
    Walk-forward Elo construction from match history.

    Expected columns: date, player_a, player_b, winner ("a"|"b"),
                      surface, tournament_level.

    Cross-surface blending: if a player has fewer than MIN_SURFACE_MATCHES on a
    surface, their rating on that surface blends toward their overall average.

    Returns {player: {surface: final_elo}}.
    """
    df = matches_df.sort_values("date").reset_index(drop=True)

    ratings: dict[str, dict[str, float]] = {}
    # track how many matches per player per surface for blending
    match_counts: dict[str, dict[str, int]] = {}

    def _get(player: str, surface: str) -> float:
        r = ratings.setdefault(player, {s: base_rating for s in SURFACES})
        return r.get(surface, base_rating)

    def _set(player: str, surface: str, value: float) -> None:
        ratings.setdefault(player, {s: base_rating for s in SURFACES})[surface] = value

    def _inc_count(player: str, surface: str) -> None:
        match_counts.setdefault(player, {s: 0 for s in SURFACES})[surface] = (
            match_counts.get(player, {}).get(surface, 0) + 1
        )

    for row in df.itertuples(index=False):
        pa: str = row.player_a
        pb: str = row.player_b
        surface: str = row.surface
        winner: str = row.winner
        level: str = getattr(row, "tournament_level", "ATP 250")

        elo_a = _get(pa, surface)
        elo_b = _get(pb, surface)
        new_a, new_b = update_elo_after_match(elo_a, elo_b, winner, level)

        _set(pa, surface, new_a)
        _set(pb, surface, new_b)
        _inc_count(pa, surface)
        _inc_count(pb, surface)

    # Apply cross-surface blending for sparse surface histories
    for player, surf_ratings in ratings.items():
        overall_avg = sum(surf_ratings.values()) / len(surf_ratings)
        counts = match_counts.get(player, {s: 0 for s in SURFACES})
        for surface in SURFACES:
            n = counts.get(surface, 0)
            if n < MIN_SURFACE_MATCHES:
                blend = SURFACE_BLEND_WEIGHT * (1.0 - n / MIN_SURFACE_MATCHES)
                surf_ratings[surface] = (
                    (1.0 - blend) * surf_ratings[surface] + blend * overall_avg
                )

    return ratings


def get_surface_elo(
    player: str,
    surface: str,
    ratings: dict[str, dict[str, float]],
) -> float:
    """Returns player's Elo on a surface; DEFAULT_ELO if unknown."""
    return ratings.get(player, {}).get(surface, DEFAULT_ELO)


def calculate_elo_difference(
    player_a: str,
    player_b: str,
    surface: str,
    ratings: dict[str, dict[str, float]],
) -> float:
    """Elo_A[surface] - Elo_B[surface]."""
    return get_surface_elo(player_a, surface, ratings) - get_surface_elo(player_b, surface, ratings)
