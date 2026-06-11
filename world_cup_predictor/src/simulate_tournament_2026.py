"""
simulate_tournament_2026.py
===========================

Simulação Monte Carlo da Copa do Mundo 2026 inteira, do grupo ao campeão,
usando o modelo (Elo + forma recente -> lambdas -> Poisson) sobre o histórico
REAL (data/raw/results.csv).

Formato 2026: 12 grupos de 4. Avançam os 2 primeiros de cada grupo + os 8
melhores terceiros (32 times) para o mata-mata. O chaveamento usa os **slots
oficiais** da Copa 2026 (posição de grupo, não seeding por Elo): cada jogo das
16-avos é definido por posição (1º/2º/3º) e a árvore até a final é fixa.

Uso:
    python src/simulate_tournament_2026.py [n_simulacoes]
"""

from __future__ import annotations

import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

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

# ---------------------------------------------------------------------------
# Chaveamento OFICIAL da Copa 2026 (slots fixos por posição de grupo).
#
# Fonte: estrutura oficial do mata-mata 2026 (jogos M73–M104), conferida com a
# tabela "Wikipedia official" e o calendário FIFA/ESPN. Cada slot das 16-avos é
# definido por posição de grupo, NÃO por força (Elo). Times do mesmo grupo só
# podem se reencontrar nas quartas ou depois.
#
# Cada spec é:
#   ("W", "E")                      -> 1º colocado do grupo E
#   ("R", "C")                      -> 2º colocado do grupo C
#   ("3", frozenset({"A","B",...})) -> um 3º colocado, de um dos grupos do conjunto
#
# A ordem desta lista é a dos jogos M73..M88 (índices 0..15).
R32_SLOTS = [
    (("R", "A"), ("R", "B")),                              # M73
    (("W", "E"), ("3", frozenset("ABCDF"))),               # M74
    (("W", "F"), ("R", "C")),                              # M75
    (("W", "C"), ("R", "F")),                              # M76
    (("W", "I"), ("3", frozenset("CDFGH"))),               # M77
    (("R", "E"), ("R", "I")),                              # M78
    (("W", "A"), ("3", frozenset("CEFHI"))),               # M79
    (("W", "L"), ("3", frozenset("EHIJK"))),               # M80
    (("W", "D"), ("3", frozenset("BEFIJ"))),               # M81
    (("W", "G"), ("3", frozenset("AEHIJ"))),               # M82
    (("R", "K"), ("R", "L")),                              # M83
    (("W", "H"), ("R", "J")),                              # M84
    (("W", "B"), ("3", frozenset("EFGIJ"))),               # M85
    (("W", "J"), ("R", "H")),                              # M86
    (("W", "K"), ("3", frozenset("DEIJL"))),               # M87
    (("R", "D"), ("R", "G")),                              # M88
]

# Árvore do mata-mata: cada fase é uma lista de pares de índices da fase
# anterior cujos vencedores se enfrentam. As 16-avos são R32_SLOTS (índices
# 0..15); cada lista abaixo indexa a lista da fase imediatamente anterior.
KO_TREE = {
    # Oitavas (M89..M96) <- vencedores das 16-avos (índices em R32_SLOTS)
    "Oitavas":   [(1, 4), (0, 2), (3, 5), (6, 7), (10, 11), (8, 9), (13, 15), (12, 14)],
    # Quartas (M97..M100) <- vencedores das oitavas (índices na lista de Oitavas)
    "Quartas":   [(0, 1), (4, 5), (2, 3), (6, 7)],
    # Semifinais (M101..M102) <- vencedores das quartas
    "Semifinal": [(0, 1), (2, 3)],
    # Final (M104) <- vencedores das semis
    "Final":     [(0, 1)],
}

# Slots de 3º colocado: (índice do jogo em R32_SLOTS, conjunto de grupos aceitos).
# Derivado de R32_SLOTS — usado no emparelhamento dos 8 melhores terceiros.
THIRD_SLOTS = [
    (i, spec[1]) for i, slots in enumerate(R32_SLOTS)
    for spec in slots if spec[0] == "3"
]


def assign_thirds(qualifying_groups):
    """Distribui os 8 melhores terceiros nos 8 slots de 3º do bracket oficial.

    ``qualifying_groups`` é a lista das 8 letras de grupo cujos terceiros se
    classificaram. Resolve um emparelhamento bipartido (grupo -> slot) que
    respeita os conjuntos de grupos aceitos por cada slot (``THIRD_SLOTS``),
    via ``linear_sum_assignment`` com custo 0 quando o grupo é aceito e um
    custo proibitivo caso contrário.

    A FIFA publica uma tabela das 495 combinações possíveis; para qualquer
    conjunto de 8 grupos existe (por construção) ao menos uma atribuição
    válida. Este matching devolve sempre uma atribuição válida (custo total 0)
    e coincide com a tabela oficial na grande maioria dos casos.

    Returns
    -------
    dict
        ``grupo (str) -> índice do jogo em R32_SLOTS``.
    """
    BIG = 1000
    cost = np.empty((8, 8), dtype=int)
    for gi, g in enumerate(qualifying_groups):
        for si, (_match_idx, allowed) in enumerate(THIRD_SLOTS):
            cost[gi, si] = 0 if g in allowed else BIG
    rows, cols = linear_sum_assignment(cost)
    if cost[rows, cols].sum() != 0:
        # Combinação sem matching perfeito respeitando os conjuntos — não deve
        # ocorrer para conjuntos válidos da Copa, mas mantemos uma atribuição
        # (ainda completa) em vez de falhar a simulação.
        pass
    return {qualifying_groups[r]: THIRD_SLOTS[c][0] for r, c in zip(rows, cols)}


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

    # 8 melhores terceiros (pts -> saldo -> gols feitos -> Elo), guardando o grupo
    third_sorted = sorted(
        third_place,
        key=lambda x: (x[2], x[3], x[4], ratings.get(x[1])),
        reverse=True,
    )
    best_thirds = [t[1] for t in third_sorted[:8]]
    third_team_by_group = {t[0]: t[1] for t in third_sorted[:8]}
    qualifying_groups = [t[0] for t in third_sorted[:8]]

    # Atribui cada terceiro a um slot oficial de 3º (índice do jogo em R32_SLOTS).
    group_to_match = assign_thirds(qualifying_groups)
    third_by_match = {m: third_team_by_group[g] for g, m in group_to_match.items()}

    def resolve(match_idx, spec):
        kind, val = spec
        if kind == "W":
            return pos1[val]
        if kind == "R":
            return pos2[val]
        return third_by_match[match_idx]  # ("3", conjunto permitido)

    rounds = {}

    # 16-avos (M73..M88): slots oficiais por posição de grupo.
    r32_matches = []
    prev_winners = []
    for i, (spec_a, spec_b) in enumerate(R32_SLOTS):
        a = resolve(i, spec_a)
        b = resolve(i, spec_b)
        w, xa, xb = knockout_match(rng, a, b, cache_ko)
        goals[a] += xa; goals[b] += xb
        r32_matches.append((a, b, w))
        prev_winners.append(w)
    rounds["16-avos"] = r32_matches

    # Oitavas -> Final: árvore fixa (vencedores avançam conforme KO_TREE).
    for phase in ("Oitavas", "Quartas", "Semifinal", "Final"):
        matches = []
        winners = []
        for ia, ib in KO_TREE[phase]:
            a = prev_winners[ia]
            b = prev_winners[ib]
            w, xa, xb = knockout_match(rng, a, b, cache_ko)
            goals[a] += xa; goals[b] += xb
            matches.append((a, b, w))
            winners.append(w)
        rounds[phase] = matches
        prev_winners = winners

    return {
        "champion": prev_winners[0],
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
