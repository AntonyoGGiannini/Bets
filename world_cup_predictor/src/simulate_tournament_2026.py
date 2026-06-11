"""
simulate_tournament_2026.py
===========================

Simulação Monte Carlo da Copa do Mundo 2026 inteira, do grupo ao campeão,
usando o modelo (Elo + forma recente -> lambdas -> Poisson) sobre o histórico
REAL (data/raw/results.csv).

Formato 2026: 12 grupos de 4. Avançam os 2 primeiros de cada grupo + os 8
melhores terceiros (32 times) para o mata-mata. Mata-mata simples por bracket
seedado pela força (Elo) dos classificados.

Uso:
    python src/simulate_tournament_2026.py [n_simulacoes]
"""

from __future__ import annotations

import sys
from collections import defaultdict

import numpy as np
import pandas as pd

from data_fetcher import load_kaggle_results, filter_copa_teams, COPA_2026_GROUPS
from elo_model import build_elo_history
from feature_engineering import build_team_strengths, create_elo_diff
from goal_model import estimate_lambdas_for_fixture

DATA = "../data/raw/results.csv"


def train():
    matches = filter_copa_teams(load_kaggle_results(DATA, min_date="2000-01-01"))
    ratings = build_elo_history(matches)
    matches = create_elo_diff(matches, ratings)
    strengths = build_team_strengths(matches, ratings, n_games=10)
    return ratings, strengths


def make_lambda_cache(teams, ratings, strengths):
    """Pré-computa (lambda_a, lambda_b) para cada par ordenado (jogo neutro)."""
    cache = {}
    for a in teams:
        for b in teams:
            if a == b:
                continue
            de = ratings.get(a) - ratings.get(b)
            cache[(a, b)] = estimate_lambdas_for_fixture(
                a, b, strengths, diferenca_elo=de,
                jogo_eliminatorio=0, mando_neutro=1,
            )
    # versão mata-mata
    cache_ko = {}
    for a in teams:
        for b in teams:
            if a == b:
                continue
            de = ratings.get(a) - ratings.get(b)
            cache_ko[(a, b)] = estimate_lambdas_for_fixture(
                a, b, strengths, diferenca_elo=de,
                jogo_eliminatorio=1, mando_neutro=1,
            )
    return cache, cache_ko


def play(rng, la, lb):
    return rng.poisson(la), rng.poisson(lb)


# Nome da fase pelo nº de times ainda no bracket (formato 2026: KO de 32).
ROUND_BY_SIZE = {32: "16-avos", 16: "Oitavas", 8: "Quartas", 4: "Semifinal", 2: "Final"}


def knockout_match(rng, a, b, cache_ko):
    """Joga um mata-mata e devolve (vencedor, gols_a, gols_b)."""
    la, lb = cache_ko[(a, b)]
    ga, gb = play(rng, la, lb)
    if ga > gb:
        return a, ga, gb
    if gb > ga:
        return b, ga, gb
    # empate -> pênaltis (50/50 levemente ponderado pelo lambda)
    p = la / (la + lb) if (la + lb) > 0 else 0.5
    return (a if rng.random() < p else b), ga, gb


def knockout_winner(rng, a, b, cache_ko):
    return knockout_match(rng, a, b, cache_ko)[0]


def simulate_once_detailed(rng, ratings, cache, cache_ko):
    """Simula um torneio completo e devolve os detalhes da edição.

    Returns
    -------
    dict
        ``champion``, ``pos1``/``pos2`` (grupo -> time), ``third_qualified``
        (8 melhores terceiros), ``rounds`` (fase -> lista de
        ``(time_a, time_b, vencedor)``) e ``goals`` (gols marcados por time
        na edição, grupos + mata-mata).
    """
    third_place = []  # (group, team, pts, gd, gf)
    qualified_1_2 = {}  # group -> [winner, runner_up]
    goals = defaultdict(int)
    pos1, pos2 = {}, {}

    for group, teams in COPA_2026_GROUPS.items():
        pts = defaultdict(int)
        gf = defaultdict(int)
        ga = defaultdict(int)
        # round robin
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                a, b = teams[i], teams[j]
                la, lb = cache[(a, b)]
                xa, xb = play(rng, la, lb)
                gf[a] += xa; ga[a] += xb
                gf[b] += xb; ga[b] += xa
                goals[a] += xa; goals[b] += xb
                if xa > xb:
                    pts[a] += 3
                elif xb > xa:
                    pts[b] += 3
                else:
                    pts[a] += 1; pts[b] += 1
        # ranking dentro do grupo: pts, saldo, gols feitos, depois Elo (desempate)
        table = sorted(
            teams,
            key=lambda t: (pts[t], gf[t] - ga[t], gf[t], ratings.get(t)),
            reverse=True,
        )
        qualified_1_2[group] = [table[0], table[1]]
        pos1[group], pos2[group] = table[0], table[1]
        t3 = table[2]
        third_place.append((group, t3, pts[t3], gf[t3] - ga[t3], gf[t3]))

    # 8 melhores terceiros
    third_sorted = sorted(
        third_place,
        key=lambda x: (x[2], x[3], x[4], ratings.get(x[1])),
        reverse=True,
    )
    best_thirds = [t[1] for t in third_sorted[:8]]

    # 32 classificados -> seed por Elo e bracket simples
    qualifiers = []
    for g in COPA_2026_GROUPS:
        qualifiers.extend(qualified_1_2[g])
    qualifiers.extend(best_thirds)

    seeded = sorted(qualifiers, key=lambda t: ratings.get(t), reverse=True)
    # bracket 1v32, 2v31, ... mantendo favoritos separados
    rounds = {}
    bracket = seeded[:]
    while len(bracket) > 1:
        nxt = []
        n = len(bracket)
        matches = []
        for i in range(n // 2):
            a = bracket[i]
            b = bracket[n - 1 - i]
            w, xa, xb = knockout_match(rng, a, b, cache_ko)
            goals[a] += xa; goals[b] += xb
            matches.append((a, b, w))
            nxt.append(w)
        rounds[ROUND_BY_SIZE[n]] = matches
        bracket = nxt

    return {
        "champion": bracket[0],
        "pos1": pos1,
        "pos2": pos2,
        "third_qualified": best_thirds,
        "rounds": rounds,
        "goals": dict(goals),
    }


def simulate_once(rng, ratings, cache, cache_ko):
    """Compatível com a V1: devolve apenas o campeão da edição simulada."""
    return simulate_once_detailed(rng, ratings, cache, cache_ko)["champion"]


def main(n=2000):
    ratings, strengths = train()
    teams = sorted({t for ts in COPA_2026_GROUPS.values() for t in ts})
    cache, cache_ko = make_lambda_cache(teams, ratings, strengths)

    rng = np.random.default_rng(42)
    champs = defaultdict(int)
    for _ in range(n):
        champs[simulate_once(rng, ratings, cache, cache_ko)] += 1

    s = pd.Series(champs).sort_values(ascending=False) / n * 100
    print(f"\n=== PROBABILIDADE DE TÍTULO — Copa 2026 ({n} simulações) ===\n")
    for team, p in s.head(15).items():
        print(f"  {team:24s} {p:5.1f}%   (Elo {ratings.get(team):.0f})")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    main(n)
