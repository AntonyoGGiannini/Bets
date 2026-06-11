"""
top_scorer.py
=============

Estimativa do artilheiro da Copa 2026.

A base de placares (``results.csv``) não tem jogadores; usamos o
``goalscorers.csv`` do mesmo dataset (martj42/international_results), que
lista cada gol internacional com autor, seleção e flags de gol contra/pênalti.

Modelo simples e interpretável, em duas etapas:

1. **Gols esperados do time no torneio** — média de gols que cada seleção
   marca por edição simulada (saída do Monte Carlo de
   ``simulate_tournament_2026``). Times que vão mais longe jogam mais jogos
   e, portanto, têm mais gols esperados.
2. **Fatia do jogador** — participação do jogador nos gols da própria
   seleção numa janela recente (padrão: desde 2023, capturando o ciclo
   atual da Copa). ``gols_esperados_jogador = fatia × gols_esperados_time``.

Limitações honestas: não sabemos escalações, minutos, lesões nem
aposentadorias; a fatia histórica é um proxy do papel ofensivo do jogador
no ciclo atual.

Uso:
    python src/top_scorer.py [n_simulacoes]
"""

from __future__ import annotations

import os
import sys

import pandas as pd

from data_fetcher import COPA_2026_TEAMS

GOALSCORERS_CSV = os.path.join(
    os.path.dirname(__file__), "..", "data", "raw", "goalscorers.csv"
)

# Janela da "fatia" do jogador: o ciclo atual da Copa. Curta demais = ruído;
# longa demais = jogadores fora do ciclo (aposentados, fora da seleção).
DEFAULT_MIN_DATE = "2023-01-01"

# Gols mínimos na janela para entrar no ranking (evita fatias infladas por
# amostra minúscula, ex.: 1 gol num time que marcou 3).
MIN_GOALS = 4


def load_goalscorers(path: str = GOALSCORERS_CSV, min_date: str = DEFAULT_MIN_DATE) -> pd.DataFrame:
    """Carrega o goalscorers.csv filtrando janela, gols contra e times da Copa."""
    df = pd.read_csv(path, parse_dates=["date"])
    df = df[df["date"] >= pd.Timestamp(min_date)]
    df = df[df["own_goal"].astype(str).str.upper() != "TRUE"]
    df = df.dropna(subset=["scorer"])
    df = df[df["team"].isin(COPA_2026_TEAMS)]
    return df


def player_shares(goalscorers: pd.DataFrame) -> pd.DataFrame:
    """Fatia de cada jogador nos gols da própria seleção na janela.

    Returns
    -------
    DataFrame com colunas: time, jogador, gols_janela, gols_time_janela, fatia.
    """
    by_player = (
        goalscorers.groupby(["team", "scorer"]).size().rename("gols_janela").reset_index()
    )
    by_team = goalscorers.groupby("team").size().rename("gols_time_janela")
    out = by_player.join(by_team, on="team")
    out["fatia"] = out["gols_janela"] / out["gols_time_janela"]
    return out.rename(columns={"team": "time", "scorer": "jogador"})


def expected_top_scorers(
    team_expected_goals: dict[str, float],
    goalscorers_path: str = GOALSCORERS_CSV,
    min_date: str = DEFAULT_MIN_DATE,
    min_goals: int = MIN_GOALS,
    top: int = 15,
) -> pd.DataFrame:
    """Ranqueia os candidatos a artilheiro da Copa 2026.

    Parameters
    ----------
    team_expected_goals:
        Gols esperados de cada seleção por edição do torneio (média das
        simulações Monte Carlo). Times ausentes do dict são ignorados.
    """
    shares = player_shares(load_goalscorers(goalscorers_path, min_date))
    shares = shares[shares["gols_janela"] >= min_goals]
    shares = shares[shares["time"].isin(team_expected_goals)]

    shares["gols_esperados_time"] = shares["time"].map(team_expected_goals)
    shares["gols_esperados_copa"] = shares["fatia"] * shares["gols_esperados_time"]

    cols = ["jogador", "time", "gols_janela", "fatia", "gols_esperados_time", "gols_esperados_copa"]
    return (
        shares.sort_values("gols_esperados_copa", ascending=False)
        .head(top)[cols]
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    from collections import defaultdict

    import numpy as np

    import simulate_tournament_2026 as sim

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000

    ratings, strengths = sim.train()
    teams = sorted({t for ts in sim.COPA_2026_GROUPS.values() for t in ts})
    cache, cache_ko = sim.make_lambda_cache(teams, ratings, strengths)

    rng = np.random.default_rng(42)
    goals_sum: dict[str, float] = defaultdict(float)
    for _ in range(n):
        d = sim.simulate_once_detailed(rng, ratings, cache, cache_ko)
        for t, g in d["goals"].items():
            goals_sum[t] += g
    team_xg = {t: g / n for t, g in goals_sum.items()}

    table = expected_top_scorers(team_xg)
    print(f"\n=== CANDIDATOS A ARTILHEIRO — Copa 2026 ({n} simulações) ===\n")
    for _, r in table.iterrows():
        print(
            f"  {r.jogador:28s} {r.time:15s} "
            f"{r.gols_esperados_copa:4.2f} gols esp.  "
            f"(fatia {r.fatia*100:4.1f}% × {r.gols_esperados_time:.1f} gols do time)"
        )
