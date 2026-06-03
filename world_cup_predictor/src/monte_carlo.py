"""
monte_carlo.py
==============

Simulação Monte Carlo de partidas a partir dos lambdas estimados.

A matriz de Poisson (em ``goal_model``) já dá probabilidades exatas, mas a
simulação Monte Carlo:

- serve de verificação cruzada (os dois métodos devem concordar);
- escala naturalmente para simular torneios inteiros, onde a propagação de
  incerteza entre fases não tem forma fechada simples;
- permite, no futuro, injetar ruído extra (lesões, cartões) por simulação.

Para cada partida sorteamos ``gols_a ~ Poisson(lambda_a)`` e
``gols_b ~ Poisson(lambda_b)`` e contabilizamos os mercados.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def simulate_match(
    lambda_a: float,
    lambda_b: float,
    n_simulations: int = 10_000,
    seed: int | None = None,
) -> Dict[str, float]:
    """Simula ``n_simulations`` partidas e resume os mercados.

    Returns
    -------
    dict
        Probabilidades de 1X2, over/under, ambos marcam e o placar mais
        frequente observado na simulação.
    """
    rng = np.random.default_rng(seed)
    goals_a = rng.poisson(lambda_a, n_simulations)
    goals_b = rng.poisson(lambda_b, n_simulations)

    home = goals_a > goals_b
    draw = goals_a == goals_b
    away = goals_a < goals_b
    total = goals_a + goals_b

    # Placar mais frequente.
    pairs = np.stack([goals_a, goals_b], axis=1)
    uniq, counts = np.unique(pairs, axis=0, return_counts=True)
    best = uniq[np.argmax(counts)]
    placar = f"{int(best[0])}x{int(best[1])}"

    n = float(n_simulations)
    return {
        "prob_vitoria_time_a": home.sum() / n,
        "prob_empate": draw.sum() / n,
        "prob_vitoria_time_b": away.sum() / n,
        "prob_over_0_5": (total >= 1).sum() / n,
        "prob_over_1_5": (total >= 2).sum() / n,
        "prob_over_2_5": (total >= 3).sum() / n,
        "prob_under_2_5": (total <= 2).sum() / n,
        "prob_ambos_marcam": ((goals_a >= 1) & (goals_b >= 1)).sum() / n,
        "placar_mais_provavel": placar,
        "media_gols_time_a": float(goals_a.mean()),
        "media_gols_time_b": float(goals_b.mean()),
    }


def simulate_tournament(matches: pd.DataFrame, n_simulations: int = 10_000) -> pd.DataFrame:
    """Simula uma lista de partidas independentes.

    ``matches`` deve conter as colunas ``time_a``, ``time_b``, ``lambda_time_a``
    e ``lambda_time_b``.

    Returns
    -------
    pandas.DataFrame
        Uma linha por partida com os resultados resumidos da simulação.
    """
    rows: List[dict] = []
    for i, row in enumerate(matches.itertuples(index=False)):
        res = simulate_match(
            float(row.lambda_time_a),
            float(row.lambda_time_b),
            n_simulations=n_simulations,
            seed=1000 + i,
        )
        res["time_a"] = row.time_a
        res["time_b"] = row.time_b
        rows.append(res)
    return pd.DataFrame(rows)


def summarize_simulation_results(simulations: pd.DataFrame) -> pd.DataFrame:
    """Formata/arredonda os resultados de simulação para exibição."""
    out = simulations.copy()
    prob_cols = [c for c in out.columns if c.startswith("prob_") or c.startswith("media_")]
    out[prob_cols] = out[prob_cols].round(4)
    return out
