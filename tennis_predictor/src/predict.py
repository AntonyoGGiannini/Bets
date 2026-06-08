"""Single match prediction interface for tennis."""
from __future__ import annotations

import argparse
import sys
from functools import lru_cache

import pandas as pd

from . import elo as _elo
from . import stats as _stats
from .markets import build_full_market_sheet


@lru_cache(maxsize=4)
def _load_data(
    data_path: str | None,
    n_recent_matches: int,
) -> tuple[dict, pd.DataFrame]:
    """Load Elo ratings from match history (or generate mock data)."""
    if data_path:
        try:
            df = pd.read_csv(data_path, parse_dates=["date"])
        except Exception:
            df = _stats.generate_mock_matches(n_matches=n_recent_matches * 50)
    else:
        df = _stats.generate_mock_matches(n_matches=max(500, n_recent_matches * 50))

    ratings = _elo.build_elo_history(df)
    return ratings, df


def predict_match(
    player_a: str,
    player_b: str,
    surface: str = "hard",
    best_of: int = 3,
    data_path: str | None = None,
    stats_path: str | None = None,
    n_recent_matches: int = 20,
    elo_blend_weight: float = 0.30,
) -> dict:
    """
    Full match prediction returning all market probabilities.

    Returns a dict with Elo ratings, service stats, analytical probability,
    Elo-blended probability, and all betting markets.
    """
    ratings, _ = _load_data(data_path, n_recent_matches)

    stats_db = _stats.load_stats_from_csv(stats_path) if stats_path else None

    raw_a = _stats.get_player_stats(player_a, surface, stats_db)
    raw_b = _stats.get_player_stats(player_b, surface, stats_db)
    stats_a, stats_b = _stats.adjust_stats_for_opponent(raw_a, raw_b, surface)

    p_srv_a = stats_a["p_srv"]
    p_srv_b = stats_b["p_srv"]

    # Analytical probability
    from .model import p_win_match as _pwm
    prob_analytical_a = _pwm(p_srv_a, p_srv_b, best_of=best_of)

    # Elo-derived probability
    elo_a = _elo.get_surface_elo(player_a, surface, ratings)
    elo_b = _elo.get_surface_elo(player_b, surface, ratings)
    prob_elo_a = _elo.expected_score(elo_a, elo_b)

    # Blended probability
    prob_final_a = (1.0 - elo_blend_weight) * prob_analytical_a + elo_blend_weight * prob_elo_a
    prob_final_b = 1.0 - prob_final_a

    markets = build_full_market_sheet(
        p_srv_a, p_srv_b,
        best_of=best_of,
        elo_win_prob_a=prob_elo_a,
        elo_blend_weight=elo_blend_weight,
    )

    result = {
        "player_a": player_a,
        "player_b": player_b,
        "surface": surface,
        "best_of": best_of,
        "elo_a": round(elo_a, 1),
        "elo_b": round(elo_b, 1),
        "p_srv_a": round(p_srv_a, 4),
        "p_srv_b": round(p_srv_b, 4),
        "prob_analytical_a": round(prob_analytical_a, 4),
        "prob_elo_a": round(prob_elo_a, 4),
        "prob_final_a": round(prob_final_a, 4),
        "prob_final_b": round(prob_final_b, 4),
    }
    result.update(markets)
    return result


def _print(r: dict) -> None:
    """Pretty-print prediction result."""
    sep = "─" * 52
    print(f"\n{sep}")
    print(f"  {r['player_a']} vs {r['player_b']}")
    print(f"  Superfície: {r['surface'].upper()}  |  Melhor de {r['best_of']}")
    print(sep)
    print(f"  Elo  → {r['player_a']}: {r['elo_a']:.0f}  |  {r['player_b']}: {r['elo_b']:.0f}")
    print(f"  Saque win% → {r['player_a']}: {r['p_srv_a']:.1%}  |  {r['player_b']}: {r['p_srv_b']:.1%}")
    print(sep)
    print("  PROBABILIDADES — Vencedor da Partida")
    print(f"    Modelo analítico : {r['player_a']} {r['prob_analytical_a']:.1%}")
    print(f"    Elo              : {r['player_a']} {r['prob_elo_a']:.1%}")
    print(f"    Final (blend)    : {r['player_a']} {r['prob_final_a']:.1%}  |  {r['player_b']} {r['prob_final_b']:.1%}")
    print(sep)
    print("  PLACAR DE SETS")
    for key in sorted(r):
        if key.startswith("prob_set_score_"):
            label = key.replace("prob_set_score_", "").replace("_", "-")
            print(f"    {label:>5}  {r[key]:.1%}")
    print(sep)
    print("  HANDICAP DE SETS")
    print(f"    {r['player_a']} -1.5 sets  {r['prob_handicap_a_minus_1_5']:.1%}")
    print(f"    {r['player_a']} +1.5 sets  {r['prob_handicap_a_plus_1_5']:.1%}")
    print(sep)
    print("  TOTAL DE GAMES")
    for line in ("20_5", "21_5", "22_5", "23_5"):
        label = line.replace("_", ".")
        print(
            f"    Over  {label}  {r.get(f'prob_over_{line}_games', 0):.1%}  |  "
            f"Under {label}  {r.get(f'prob_under_{line}_games', 0):.1%}"
        )
    print(f"    Esperado: {r['expected_total_games']:.1f} games")
    print(sep)
    print("  PRIMEIRO SET")
    print(f"    {r['player_a']} {r['prob_first_set_a']:.1%}  |  {r['player_b']} {r['prob_first_set_b']:.1%}")
    print(sep)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predição de partida de tênis")
    parser.add_argument("player_a", type=str)
    parser.add_argument("player_b", type=str)

    surface_group = parser.add_mutually_exclusive_group()
    surface_group.add_argument("--hard",   action="store_true", help="Hard court (default)")
    surface_group.add_argument("--clay",   action="store_true", help="Saibro")
    surface_group.add_argument("--grass",  action="store_true", help="Grama")
    surface_group.add_argument("--indoor", action="store_true", help="Indoor")
    surface_group.add_argument("--surface", type=str, default=None,
                               choices=["hard", "clay", "grass", "indoor"])

    bo_group = parser.add_mutually_exclusive_group()
    bo_group.add_argument("--bo3", action="store_true", help="Melhor de 3 (default)")
    bo_group.add_argument("--bo5", action="store_true", help="Melhor de 5 (Grand Slams)")

    parser.add_argument("--dados", type=str, default=None, dest="data_path",
                        help="Caminho para CSV de histórico de partidas")
    parser.add_argument("--stats", type=str, default=None, dest="stats_path",
                        help="Caminho para CSV de estatísticas de saque")

    args = parser.parse_args()

    if args.clay:
        surface = "clay"
    elif args.grass:
        surface = "grass"
    elif args.indoor:
        surface = "indoor"
    elif args.surface:
        surface = args.surface
    else:
        surface = "hard"

    best_of = 5 if args.bo5 else 3

    result = predict_match(
        args.player_a,
        args.player_b,
        surface=surface,
        best_of=best_of,
        data_path=args.data_path,
        stats_path=args.stats_path,
    )
    _print(result)
