"""Analytical point → game → set → match probability model for tennis."""
from __future__ import annotations

import math
from functools import lru_cache

# When both players have high serve-win prob (grass-like), slightly inflate
# probability of a 3rd set (tighter match).
TIGHT_SERVE_THRESHOLD = 0.68
TIGHT_MATCH_CORRECTION = 0.05


def estimate_p_srv(fsp: float, fsw: float, ssw: float) -> float:
    """P(server wins a point): fsp*fsw + (1-fsp)*ssw."""
    return fsp * fsw + (1.0 - fsp) * ssw


def p_win_game(p: float) -> float:
    """
    Analytical P(server wins service game) given point win probability p.

    Deuce rule: first to 4 points with a 2-point lead.
    """
    p = max(0.01, min(0.99, p))
    q = 1.0 - p

    # Win without reaching deuce: 4-0, 4-1, 4-2 (4-3 goes to deuce instead)
    # P(win k+4 total points with k opponent points, k=0,1,2) = C(k+3,k) * p^4 * q^k
    no_deuce = p ** 4 * (1.0 + 4.0 * q + 10.0 * q ** 2)

    # Win from deuce (3-3): P(reach deuce) * P(win from deuce)
    p_deuce = 20.0 * (p ** 3) * (q ** 3)
    p_win_deuce = (p ** 2) / (p ** 2 + q ** 2)

    return no_deuce + p_deuce * p_win_deuce


def p_win_tiebreak(p_a: float, p_b: float) -> float:
    """
    P(player A wins tiebreak).
    p_a = P(A wins a point on A's serve).
    p_b = P(B wins a point on B's serve).

    Server alternates every 2 points (A serves first point, then B 2, A 2, ...),
    modelled via Markov chain with memoised recursion.
    """
    p_a = max(0.01, min(0.99, p_a))
    p_b = max(0.01, min(0.99, p_b))

    @lru_cache(maxsize=None)
    def _tb(i: int, j: int, a_serves: bool) -> float:
        # terminal states
        if i >= 7 and i - j >= 2:
            return 1.0
        if j >= 7 and j - i >= 2:
            return 0.0
        # mini-deuce at 6-6
        if i == 6 and j == 6:
            pa2 = p_a ** 2
            pb2 = (1.0 - p_b) ** 2
            return pa2 / (pa2 + pb2)
        p_point = p_a if a_serves else (1.0 - p_b)
        # server switches every 2 points (total points = i + j)
        next_serves = a_serves if (i + j + 1) % 2 != 0 else (not a_serves)
        return p_point * _tb(i + 1, j, next_serves) + (1.0 - p_point) * _tb(i, j + 1, next_serves)

    result = _tb(0, 0, True)
    _tb.cache_clear()
    return result


def p_win_set(
    p_srv_a: float,
    p_srv_b: float,
    a_serves_first: bool = True,
) -> float:
    """
    P(player A wins a set) given service point win probabilities.
    Standard rules: first to 6 games with 2-game lead; tiebreak at 6-6.
    """
    p_srv_a = max(0.01, min(0.99, p_srv_a))
    p_srv_b = max(0.01, min(0.99, p_srv_b))

    g_a = p_win_game(p_srv_a)   # P(A wins game when A serves)
    g_b = p_win_game(p_srv_b)   # P(B wins game when B serves)
    tb = p_win_tiebreak(p_srv_a, p_srv_b)

    @lru_cache(maxsize=None)
    def _set(ga: int, gb: int, a_serves: bool) -> float:
        # A wins set
        if ga == 6 and gb <= 4:
            return 1.0
        if ga == 7 and gb == 5:
            return 1.0
        if ga == 7 and gb == 6:
            return tb
        # B wins set
        if gb == 6 and ga <= 4:
            return 0.0
        if gb == 7 and ga == 5:
            return 0.0
        if gb == 7 and ga == 6:
            return 1.0 - tb
        # 6-6: always tiebreak
        if ga == 6 and gb == 6:
            return tb

        p_game = g_a if a_serves else (1.0 - g_b)
        return p_game * _set(ga + 1, gb, not a_serves) + (1.0 - p_game) * _set(ga, gb + 1, not a_serves)

    result = _set(0, 0, a_serves_first)
    _set.cache_clear()
    return result


def p_win_match(
    p_srv_a: float,
    p_srv_b: float,
    best_of: int = 3,
    a_serves_first: bool = True,
) -> float:
    """P(player A wins match). best_of: 3 or 5."""
    if best_of not in (3, 5):
        raise ValueError("best_of must be 3 or 5")

    sets_needed = best_of // 2 + 1
    p_set = p_win_set(p_srv_a, p_srv_b, a_serves_first)

    @lru_cache(maxsize=None)
    def _match(sa: int, sb: int) -> float:
        if sa == sets_needed:
            return 1.0
        if sb == sets_needed:
            return 0.0
        return p_set * _match(sa + 1, sb) + (1.0 - p_set) * _match(sa, sb + 1)

    result = _match(0, 0)
    _match.cache_clear()
    return result


def p_correct_set_score(
    p_srv_a: float,
    p_srv_b: float,
    best_of: int = 3,
) -> dict[str, float]:
    """
    Probabilities of each possible set score in the match.
    best_of=3: keys "2-0", "2-1", "0-2", "1-2"
    best_of=5: keys "3-0", "3-1", "3-2", "0-3", "1-3", "2-3"

    Negative binomial formula: to win N sets while losing k, the last set is
    always the winner's, so: P = C(N-1+k, k) * p^N * q^k.
    """
    if best_of not in (3, 5):
        raise ValueError("best_of must be 3 or 5")

    sets_needed = best_of // 2 + 1
    p_set = p_win_set(p_srv_a, p_srv_b)
    q_set = 1.0 - p_set

    scores: dict[str, float] = {}

    for k in range(sets_needed):  # k = sets dropped by the winner (0..N-1)
        coeff = math.comb(sets_needed - 1 + k, k)
        scores[f"{sets_needed}-{k}"] = coeff * (p_set ** sets_needed) * (q_set ** k)
        scores[f"{k}-{sets_needed}"] = coeff * (q_set ** sets_needed) * (p_set ** k)

    # Normalise to absorb floating-point drift
    total = sum(scores.values())
    return {k: v / total for k, v in scores.items()}


def _games_distribution_in_set(
    p_srv_a: float,
    p_srv_b: float,
    a_serves_first: bool = True,
) -> dict[tuple[int, int], float]:
    """
    Returns {(games_a, games_b): probability} for all valid final set scores.
    Used internally for total-games market calculation.
    """
    p_srv_a = max(0.01, min(0.99, p_srv_a))
    p_srv_b = max(0.01, min(0.99, p_srv_b))

    g_a = p_win_game(p_srv_a)
    g_b = p_win_game(p_srv_b)
    tb = p_win_tiebreak(p_srv_a, p_srv_b)

    distribution: dict[tuple[int, int], float] = {}

    @lru_cache(maxsize=None)
    def _walk(ga: int, gb: int, a_serves: bool) -> dict[tuple[int, int], float]:
        # terminal
        if (ga == 6 and gb <= 4) or (ga == 7 and gb == 5):
            return {(ga, gb): 1.0}
        if (gb == 6 and ga <= 4) or (gb == 7 and ga == 5):
            return {(ga, gb): 1.0}
        if ga == 6 and gb == 6:
            return {(7, 6): tb, (6, 7): 1.0 - tb}

        p_game = g_a if a_serves else (1.0 - g_b)
        win_branch = _walk(ga + 1, gb, not a_serves)
        lose_branch = _walk(ga, gb + 1, not a_serves)

        merged: dict[tuple[int, int], float] = {}
        for score, prob in win_branch.items():
            merged[score] = merged.get(score, 0.0) + p_game * prob
        for score, prob in lose_branch.items():
            merged[score] = merged.get(score, 0.0) + (1.0 - p_game) * prob
        return merged

    result = _walk(0, 0, a_serves_first)
    _walk.cache_clear()
    return result
