"""Tennis betting markets: probability computation and edge detection vs bookmaker odds."""
from __future__ import annotations

from . import model as _model
from .model import (
    p_win_match,
    p_win_set,
    p_correct_set_score,
    _games_distribution_in_set,
    TIGHT_SERVE_THRESHOLD,
    TIGHT_MATCH_CORRECTION,
)

# ── odds utilities ────────────────────────────────────────────────────────────

def convert_odds_to_implied_probability(odds: float) -> float:
    """1 / odds. Returns nan for invalid odds."""
    if odds <= 1.0:
        return float("nan")
    return 1.0 / odds


def remove_bookmaker_margin_2way(
    odds_a: float,
    odds_b: float,
) -> dict[str, float]:
    """
    Normalises a 2-way market (no draw) by removing the overround.
    Returns {"prob_a", "prob_b", "overround"}.
    """
    raw_a = convert_odds_to_implied_probability(odds_a)
    raw_b = convert_odds_to_implied_probability(odds_b)
    total = raw_a + raw_b
    return {
        "prob_a": raw_a / total,
        "prob_b": raw_b / total,
        "overround": round(total - 1.0, 4),
    }


def calculate_edge(model_prob: float, market_prob: float) -> float:
    """model_prob − market_prob (in percentage points when multiplied by 100)."""
    return model_prob - market_prob


def classify_edge(edge: float) -> str:
    """Classify edge strength using absolute value."""
    abs_edge = abs(edge)
    if abs_edge < 0.02:
        return "sem sinal"
    if abs_edge < 0.05:
        return "sinal fraco"
    if abs_edge < 0.08:
        return "sinal moderado"
    return "sinal forte"


# ── market calculators ────────────────────────────────────────────────────────

def prob_match_winner(
    p_srv_a: float,
    p_srv_b: float,
    best_of: int = 3,
) -> dict[str, float]:
    """{"prob_a": ..., "prob_b": ...} from the analytical match model."""
    pa = p_win_match(p_srv_a, p_srv_b, best_of=best_of)
    return {"prob_a": pa, "prob_b": 1.0 - pa}


def prob_correct_set_score(
    p_srv_a: float,
    p_srv_b: float,
    best_of: int = 3,
) -> dict[str, float]:
    """
    Probabilities for each possible set score.
    Applies tight-match correction when both players serve dominantly.
    """
    scores = p_correct_set_score(p_srv_a, p_srv_b, best_of=best_of)

    if min(p_srv_a, p_srv_b) > TIGHT_SERVE_THRESHOLD:
        # Inflate 2-1 / 1-2 (or 3-2/2-3) and deflate 2-0 / 0-2 (or 3-0/0-3)
        sets_needed = best_of // 2 + 1
        straight_a = f"{sets_needed}-0"
        straight_b = f"0-{sets_needed}"
        dec_a = str(sets_needed - 1)
        dec_b = str(sets_needed - 1)
        dropped_a = f"{sets_needed}-{dec_a[-1]}"
        dropped_b = f"{dec_b[-1]}-{sets_needed}"

        correction = TIGHT_MATCH_CORRECTION * min(
            scores.get(straight_a, 0.0),
            scores.get(straight_b, 0.0),
        )
        if correction > 0:
            scores[straight_a] = max(0.0, scores[straight_a] - correction)
            scores[straight_b] = max(0.0, scores[straight_b] - correction)
            scores[dropped_a] = scores.get(dropped_a, 0.0) + correction
            scores[dropped_b] = scores.get(dropped_b, 0.0) + correction
            total = sum(scores.values())
            scores = {k: v / total for k, v in scores.items()}

    return scores


def prob_set_handicap(
    p_srv_a: float,
    p_srv_b: float,
    handicap: float,
    best_of: int = 3,
) -> dict[str, float]:
    """
    Asian handicap on sets for player A.
    handicap=-1.5: A covers if A wins straight sets (2-0 in BO3, 3-0 or 3-1 in BO5).
    handicap=+1.5: A covers if A wins OR loses by only 1 set (A wins, or B wins 2-1/3-2).
    Returns {"prob_cover", "prob_not_cover"}.
    """
    scores = prob_correct_set_score(p_srv_a, p_srv_b, best_of=best_of)
    sets_needed = best_of // 2 + 1

    if handicap <= -1.5:
        # A must win straight sets
        covering = scores.get(f"{sets_needed}-0", 0.0)
        if best_of == 5:
            covering += scores.get(f"{sets_needed}-1", 0.0)
    elif handicap >= 1.5:
        # A covers if A wins at all, plus A loses by 1 set
        covering = sum(v for k, v in scores.items() if k.startswith(str(sets_needed)))
        # also covers if B wins but only 2-1 (BO3) or 3-1/3-2 (BO5)
        if best_of == 3:
            covering += scores.get(f"1-{sets_needed}", 0.0)
        else:
            covering += scores.get(f"1-{sets_needed}", 0.0)
            covering += scores.get(f"2-{sets_needed}", 0.0)
    else:
        covering = 0.5  # undefined handicap → 50/50

    covering = max(0.0, min(1.0, covering))
    return {"prob_cover": covering, "prob_not_cover": 1.0 - covering}


def prob_total_games(
    p_srv_a: float,
    p_srv_b: float,
    line: float,
    best_of: int = 3,
) -> dict[str, float]:
    """
    Over/under on total games played in the match.
    Returns {"prob_over", "prob_under", "expected_total_games"}.
    """
    set_dist = _games_distribution_in_set(p_srv_a, p_srv_b)
    set_scores = prob_correct_set_score(p_srv_a, p_srv_b, best_of=best_of)
    sets_needed = best_of // 2 + 1

    # Expected games per set (average across all possible set scores)
    expected_games_per_set = sum(
        (ga + gb) * prob for (ga, gb), prob in set_dist.items()
    )

    # Expected total games via set-score-weighted sum
    expected_total = 0.0
    prob_over = 0.0
    prob_under = 0.0

    # Build a distribution of total games by enumerating set paths
    total_games_dist: dict[int, float] = {}

    for set_score_str, set_score_prob in set_scores.items():
        parts = set_score_str.split("-")
        sa, sb = int(parts[0]), int(parts[1])
        n_sets = sa + sb

        # For each set in this match path, sample games from set_dist
        # Approximation: for simplicity use the same game distribution for each set
        # (same server stats throughout). We convolve n_sets times.
        conv: dict[int, float] = {0: 1.0}
        for _ in range(n_sets):
            new_conv: dict[int, float] = {}
            for prev_games, prev_prob in conv.items():
                for (ga, gb), gp in set_dist.items():
                    g = prev_games + ga + gb
                    new_conv[g] = new_conv.get(g, 0.0) + prev_prob * gp
            conv = new_conv

        for total, p in conv.items():
            weighted = p * set_score_prob
            total_games_dist[total] = total_games_dist.get(total, 0.0) + weighted
            expected_total += total * weighted

    for total, p in total_games_dist.items():
        if total > line:
            prob_over += p
        else:
            prob_under += p

    return {
        "prob_over": round(prob_over, 6),
        "prob_under": round(prob_under, 6),
        "expected_total_games": round(expected_total, 2),
    }


def prob_first_set_winner(
    p_srv_a: float,
    p_srv_b: float,
) -> dict[str, float]:
    """{"prob_a_wins_first_set", "prob_b_wins_first_set"}."""
    pa = p_win_set(p_srv_a, p_srv_b)
    return {"prob_a_wins_first_set": pa, "prob_b_wins_first_set": 1.0 - pa}


def build_full_market_sheet(
    p_srv_a: float,
    p_srv_b: float,
    best_of: int = 3,
    elo_win_prob_a: float | None = None,
    elo_blend_weight: float = 0.30,
) -> dict[str, float]:
    """
    Compute all standard tennis betting markets in one call.
    Optionally blends match-winner probability with Elo-derived probability.
    """
    winner = prob_match_winner(p_srv_a, p_srv_b, best_of)
    pa = winner["prob_a"]

    if elo_win_prob_a is not None:
        pa = (1.0 - elo_blend_weight) * pa + elo_blend_weight * elo_win_prob_a

    result: dict[str, float] = {
        "prob_match_winner_a": round(pa, 6),
        "prob_match_winner_b": round(1.0 - pa, 6),
    }

    set_scores = prob_correct_set_score(p_srv_a, p_srv_b, best_of=best_of)
    for k, v in set_scores.items():
        result[f"prob_set_score_{k.replace('-', '_')}"] = round(v, 6)

    h_a = prob_set_handicap(p_srv_a, p_srv_b, -1.5, best_of)
    result["prob_handicap_a_minus_1_5"] = round(h_a["prob_cover"], 6)
    result["prob_handicap_a_plus_1_5"] = round(
        prob_set_handicap(p_srv_a, p_srv_b, +1.5, best_of)["prob_cover"], 6
    )

    for line in (20.5, 21.5, 22.5, 23.5):
        tg = prob_total_games(p_srv_a, p_srv_b, line, best_of)
        label = str(line).replace(".", "_")
        result[f"prob_over_{label}_games"] = round(tg["prob_over"], 6)
        result[f"prob_under_{label}_games"] = round(tg["prob_under"], 6)

    result["expected_total_games"] = prob_total_games(p_srv_a, p_srv_b, 999.0, best_of)[
        "expected_total_games"
    ]

    first_set = prob_first_set_winner(p_srv_a, p_srv_b)
    result["prob_first_set_a"] = round(first_set["prob_a_wins_first_set"], 6)
    result["prob_first_set_b"] = round(first_set["prob_b_wins_first_set"], 6)

    return result


def analyze_match_odds(
    predictions: dict,
    market_odds: dict,
) -> dict:
    """
    Full edge analysis: model probabilities vs bookmaker odds.

    predictions: output of predict.predict_match()
    market_odds: {"odd_player_a": float, "odd_player_b": float}

    Returns dict with prob_modelo_*, prob_mercado_*, edge_*, sinal_* for each market.
    """
    result: dict = {
        "player_a": predictions.get("player_a"),
        "player_b": predictions.get("player_b"),
        "surface": predictions.get("surface"),
    }

    # Match winner market
    odd_a = market_odds.get("odd_player_a")
    odd_b = market_odds.get("odd_player_b")

    if odd_a and odd_b:
        market = remove_bookmaker_margin_2way(odd_a, odd_b)
        model_a = predictions.get("prob_final_a", predictions.get("prob_match_winner_a", 0.5))
        model_b = 1.0 - model_a

        result.update({
            "prob_modelo_a": round(model_a, 4),
            "prob_mercado_a": round(market["prob_a"], 4),
            "edge_a": round(calculate_edge(model_a, market["prob_a"]), 4),
            "sinal_a": classify_edge(calculate_edge(model_a, market["prob_a"])),
            "prob_modelo_b": round(model_b, 4),
            "prob_mercado_b": round(market["prob_b"], 4),
            "edge_b": round(calculate_edge(model_b, market["prob_b"]), 4),
            "sinal_b": classify_edge(calculate_edge(model_b, market["prob_b"])),
            "margem_casa": round(market["overround"] * 100, 2),
        })

    return result
