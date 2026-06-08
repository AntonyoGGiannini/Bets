"""Player service/return statistics, mock data generation and CSV loaders."""
from __future__ import annotations

import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .model import estimate_p_srv, p_win_match

# ── surface baselines (ATP averages) ──────────────────────────────────────────
SURFACE_BASELINES: dict[str, dict[str, float]] = {
    "hard":   {"fsp": 0.62, "fsw": 0.74, "ssw": 0.53, "rpw": 0.39},
    "clay":   {"fsp": 0.64, "fsw": 0.71, "ssw": 0.54, "rpw": 0.41},
    "grass":  {"fsp": 0.63, "fsw": 0.77, "ssw": 0.52, "rpw": 0.37},
    "indoor": {"fsp": 0.61, "fsw": 0.75, "ssw": 0.54, "rpw": 0.40},
}

SURFACES = ("hard", "clay", "grass", "indoor")

# ── mock players with surface-specific stats ──────────────────────────────────
MOCK_PLAYERS: dict[str, dict[str, dict[str, float]]] = {
    "Alcaraz": {
        "hard":   {"fsp": 0.64, "fsw": 0.78, "ssw": 0.54, "rpw": 0.42},
        "clay":   {"fsp": 0.67, "fsw": 0.76, "ssw": 0.55, "rpw": 0.45},
        "grass":  {"fsp": 0.65, "fsw": 0.80, "ssw": 0.52, "rpw": 0.40},
        "indoor": {"fsp": 0.63, "fsw": 0.79, "ssw": 0.53, "rpw": 0.41},
    },
    "Djokovic": {
        "hard":   {"fsp": 0.62, "fsw": 0.76, "ssw": 0.56, "rpw": 0.46},
        "clay":   {"fsp": 0.64, "fsw": 0.74, "ssw": 0.57, "rpw": 0.47},
        "grass":  {"fsp": 0.63, "fsw": 0.77, "ssw": 0.54, "rpw": 0.44},
        "indoor": {"fsp": 0.61, "fsw": 0.77, "ssw": 0.56, "rpw": 0.46},
    },
    "Sinner": {
        "hard":   {"fsp": 0.63, "fsw": 0.77, "ssw": 0.55, "rpw": 0.43},
        "clay":   {"fsp": 0.65, "fsw": 0.75, "ssw": 0.54, "rpw": 0.44},
        "grass":  {"fsp": 0.64, "fsw": 0.78, "ssw": 0.52, "rpw": 0.41},
        "indoor": {"fsp": 0.62, "fsw": 0.78, "ssw": 0.54, "rpw": 0.43},
    },
    "Medvedev": {
        "hard":   {"fsp": 0.60, "fsw": 0.76, "ssw": 0.58, "rpw": 0.44},
        "clay":   {"fsp": 0.58, "fsw": 0.70, "ssw": 0.53, "rpw": 0.40},
        "grass":  {"fsp": 0.61, "fsw": 0.75, "ssw": 0.55, "rpw": 0.38},
        "indoor": {"fsp": 0.59, "fsw": 0.77, "ssw": 0.57, "rpw": 0.44},
    },
    "Zverev": {
        "hard":   {"fsp": 0.61, "fsw": 0.77, "ssw": 0.55, "rpw": 0.41},
        "clay":   {"fsp": 0.63, "fsw": 0.75, "ssw": 0.54, "rpw": 0.43},
        "grass":  {"fsp": 0.62, "fsw": 0.78, "ssw": 0.52, "rpw": 0.39},
        "indoor": {"fsp": 0.60, "fsw": 0.78, "ssw": 0.55, "rpw": 0.41},
    },
    "Rune": {
        "hard":   {"fsp": 0.60, "fsw": 0.74, "ssw": 0.53, "rpw": 0.41},
        "clay":   {"fsp": 0.63, "fsw": 0.73, "ssw": 0.54, "rpw": 0.43},
        "grass":  {"fsp": 0.61, "fsw": 0.75, "ssw": 0.51, "rpw": 0.38},
        "indoor": {"fsp": 0.59, "fsw": 0.75, "ssw": 0.53, "rpw": 0.41},
    },
    # big-server archetype
    "ServerBot": {
        "hard":   {"fsp": 0.66, "fsw": 0.84, "ssw": 0.55, "rpw": 0.36},
        "clay":   {"fsp": 0.62, "fsw": 0.78, "ssw": 0.52, "rpw": 0.34},
        "grass":  {"fsp": 0.68, "fsw": 0.87, "ssw": 0.56, "rpw": 0.35},
        "indoor": {"fsp": 0.65, "fsw": 0.85, "ssw": 0.54, "rpw": 0.36},
    },
    # clay-specialist archetype
    "ClayKing": {
        "hard":   {"fsp": 0.63, "fsw": 0.73, "ssw": 0.55, "rpw": 0.44},
        "clay":   {"fsp": 0.68, "fsw": 0.79, "ssw": 0.60, "rpw": 0.50},
        "grass":  {"fsp": 0.61, "fsw": 0.72, "ssw": 0.52, "rpw": 0.40},
        "indoor": {"fsp": 0.62, "fsw": 0.74, "ssw": 0.54, "rpw": 0.43},
    },
}

TOURNAMENT_LEVELS = ["Grand Slam", "Masters", "ATP 500", "ATP 250", "Challenger", "ITF"]
TOURNAMENT_WEIGHTS = [0.08, 0.12, 0.15, 0.30, 0.25, 0.10]

SURFACE_WEIGHTS = {"hard": 0.50, "clay": 0.30, "grass": 0.12, "indoor": 0.08}

# ── functions ─────────────────────────────────────────────────────────────────

def get_player_stats(
    player: str,
    surface: str,
    stats_db: dict | None = None,
) -> dict[str, float]:
    """
    Returns {"fsp", "fsw", "ssw", "rpw", "p_srv"} for player on surface.
    Falls back to SURFACE_BASELINES if player is unknown.
    """
    db = stats_db if stats_db is not None else MOCK_PLAYERS
    player_data = db.get(player, {})
    raw = player_data.get(surface, SURFACE_BASELINES.get(surface, SURFACE_BASELINES["hard"]))
    result = dict(raw)
    result["p_srv"] = estimate_p_srv(result["fsp"], result["fsw"], result["ssw"])
    return result


def adjust_stats_for_opponent(
    stats_a: dict[str, float],
    stats_b: dict[str, float],
    surface: str = "hard",
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Adjusts each player's effective serve stats based on the opponent's return ability.

    A strong returner (high rpw) reduces the opponent's effective fsw/ssw.
    Correction clipped to ±0.05 to avoid extreme adjustments.
    """
    baseline_rpw = SURFACE_BASELINES.get(surface, SURFACE_BASELINES["hard"])["rpw"]
    weight = 0.5

    def _adjust(server: dict, returner: dict) -> dict:
        delta = (returner["rpw"] - baseline_rpw) * weight
        delta = max(-0.05, min(0.05, delta))
        adj = dict(server)
        adj["fsw"] = max(0.40, min(0.95, server["fsw"] - delta))
        adj["ssw"] = max(0.30, min(0.90, server["ssw"] - delta))
        adj["p_srv"] = estimate_p_srv(adj["fsp"], adj["fsw"], adj["ssw"])
        return adj

    return _adjust(stats_a, stats_b), _adjust(stats_b, stats_a)


def generate_mock_matches(
    n_matches: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Generates a synthetic match history using MOCK_PLAYERS stats."""
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    players = list(MOCK_PLAYERS.keys())
    surfaces = list(SURFACE_WEIGHTS.keys())
    surface_probs = list(SURFACE_WEIGHTS.values())

    start = datetime(2018, 1, 1)
    end = datetime(2025, 12, 1)
    total_days = (end - start).days

    records = []
    for _ in range(n_matches):
        pa, pb = rng.sample(players, 2)
        surface = rng.choices(surfaces, weights=surface_probs, k=1)[0]
        level = rng.choices(TOURNAMENT_LEVELS, weights=TOURNAMENT_WEIGHTS, k=1)[0]
        best_of = 5 if level == "Grand Slam" else 3

        stats_a = get_player_stats(pa, surface)
        stats_b = get_player_stats(pb, surface)
        stats_a, stats_b = adjust_stats_for_opponent(stats_a, stats_b, surface)

        prob_a = p_win_match(stats_a["p_srv"], stats_b["p_srv"], best_of=best_of)
        winner = "a" if np_rng.random() < prob_a else "b"

        date = start + timedelta(days=int(np_rng.integers(0, total_days)))
        records.append({
            "date": date,
            "player_a": pa,
            "player_b": pb,
            "winner": winner,
            "surface": surface,
            "tournament_level": level,
            "tournament": f"{level} {date.year}",
            "best_of": best_of,
        })

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    return df


def generate_mock_fixtures(
    fixtures: list[tuple[str, str, str, int]] | None = None,
) -> pd.DataFrame:
    """
    Returns a DataFrame of upcoming matches to predict.
    Each tuple: (player_a, player_b, surface, best_of).
    """
    if fixtures is None:
        fixtures = [
            ("Alcaraz",   "Sinner",    "hard",   3),
            ("Djokovic",  "Zverev",    "clay",   3),
            ("Alcaraz",   "Djokovic",  "grass",  5),
            ("Medvedev",  "Rune",      "indoor", 3),
            ("Sinner",    "ClayKing",  "clay",   3),
            ("ServerBot", "Medvedev",  "grass",  3),
            ("Zverev",    "Alcaraz",   "clay",   5),
        ]
    rows = [
        {"player_a": a, "player_b": b, "surface": s, "best_of": bo}
        for a, b, s, bo in fixtures
    ]
    return pd.DataFrame(rows)


def generate_mock_odds(
    fixtures: pd.DataFrame,
    predictions: pd.DataFrame | None = None,
    seed: int = 7,
    overround: float = 1.04,
) -> pd.DataFrame:
    """
    Generates decimal odds with ~4% overround for match winner market.
    If predictions are provided, odds are based on model probabilities with noise.
    Otherwise uses equal probability as baseline.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i, row in fixtures.iterrows():
        if predictions is not None and i < len(predictions):
            base_p_a = float(predictions.iloc[i].get("prob_final_a", 0.50))
        else:
            base_p_a = 0.50

        # Add bookmaker noise (±3–8%)
        noise = rng.uniform(-0.06, 0.06)
        p_a_market = max(0.05, min(0.95, base_p_a + noise))
        p_b_market = 1.0 - p_a_market

        odd_a = round(1.0 / (p_a_market * overround), 2)
        odd_b = round(1.0 / (p_b_market * overround), 2)

        rows.append({
            "player_a": row["player_a"],
            "player_b": row["player_b"],
            "surface": row["surface"],
            "best_of": row["best_of"],
            "odd_player_a": odd_a,
            "odd_player_b": odd_b,
        })
    return pd.DataFrame(rows)


def load_stats_from_csv(path: str) -> dict[str, dict[str, dict[str, float]]]:
    """
    Parses a CSV with columns: player, surface, fsp, fsw, ssw, rpw.
    Returns nested dict {player: {surface: {fsp, fsw, ssw, rpw}}}.
    """
    df = pd.read_csv(path)
    result: dict[str, dict[str, dict[str, float]]] = {}
    for _, row in df.iterrows():
        player = str(row["player"])
        surface = str(row["surface"]).lower()
        result.setdefault(player, {})[surface] = {
            "fsp": float(row["fsp"]),
            "fsw": float(row["fsw"]),
            "ssw": float(row["ssw"]),
            "rpw": float(row["rpw"]),
        }
    return result
