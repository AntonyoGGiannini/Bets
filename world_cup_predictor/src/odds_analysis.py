"""
odds_analysis.py
================

Comparação entre as probabilidades do modelo e as odds de mercado (V2).

Fluxo:
1. Converter odds decimais em probabilidade implícita (1 / odd).
2. Remover a margem da casa (overround) normalizando as três probabilidades.
3. Calcular o ``edge`` = prob_modelo - prob_mercado.
4. Classificar o sinal (sem sinal / fraco / moderado / forte).

ALERTA IMPORTANTE (seção 11): nenhuma decisão deve ser tomada apenas pelo
edge. É preciso avaliar liquidez, margem da casa, qualidade dos dados e a
incerteza do próprio modelo.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def convert_odds_to_implied_probability(odds: float) -> float:
    """Probabilidade implícita (com margem) de uma odd decimal: ``1 / odd``."""
    if odds is None or odds <= 1.0:
        return float("nan")
    return 1.0 / float(odds)


def remove_bookmaker_margin(
    odds_home: float,
    odds_draw: float,
    odds_away: float,
) -> Dict[str, float]:
    """Remove a margem da casa de um mercado 1X2.

    Soma as probabilidades implícitas (que dá > 1 por causa do overround) e
    renormaliza para somar exatamente 1.

    Returns
    -------
    dict
        ``prob_time_a``, ``prob_empate``, ``prob_time_b`` e ``overround``.
    """
    p_home = convert_odds_to_implied_probability(odds_home)
    p_draw = convert_odds_to_implied_probability(odds_draw)
    p_away = convert_odds_to_implied_probability(odds_away)

    overround = p_home + p_draw + p_away
    if overround <= 0:
        return {"prob_time_a": np.nan, "prob_empate": np.nan, "prob_time_b": np.nan, "overround": np.nan}

    return {
        "prob_time_a": p_home / overround,
        "prob_empate": p_draw / overround,
        "prob_time_b": p_away / overround,
        "overround": overround - 1.0,  # margem da casa em fração
    }


def calculate_edge(model_probability: float, market_probability: float) -> float:
    """Edge em pontos percentuais (fração): ``modelo - mercado``."""
    return float(model_probability) - float(market_probability)


def classify_edge(edge: float) -> str:
    """Classifica o edge segundo a régua da seção 11.

    Entrada em fração (ex.: 0.06 = 6 p.p.).
    """
    pp = abs(edge) * 100.0
    if pp < 2.0:
        return "sem sinal"
    if pp < 5.0:
        return "sinal fraco"
    if pp < 8.0:
        return "sinal moderado"
    return "sinal forte"


def analyze_fixture_odds(
    predictions: pd.DataFrame,
    odds: pd.DataFrame,
) -> pd.DataFrame:
    """Junta previsões do modelo com odds de mercado e calcula edges 1X2.

    Parameters
    ----------
    predictions:
        DataFrame com ``time_a``, ``time_b`` e as probabilidades do modelo
        (``prob_vitoria_time_a``, ``prob_empate``, ``prob_vitoria_time_b``).
    odds:
        DataFrame com ``time_a``, ``time_b`` e ``odd_time_a``, ``odd_empate``,
        ``odd_time_b``.

    Returns
    -------
    pandas.DataFrame
        Uma linha por confronto com probabilidades de mercado (sem margem),
        edges e a classificação para cada resultado (A / X / B).
    """
    merged = predictions.merge(odds, on=["time_a", "time_b"], how="inner")
    rows = []
    for row in merged.itertuples(index=False):
        market = remove_bookmaker_margin(row.odd_time_a, row.odd_empate, row.odd_time_b)

        edge_a = calculate_edge(row.prob_vitoria_time_a, market["prob_time_a"])
        edge_x = calculate_edge(row.prob_empate, market["prob_empate"])
        edge_b = calculate_edge(row.prob_vitoria_time_b, market["prob_time_b"])

        rows.append({
            "time_a": row.time_a,
            "time_b": row.time_b,
            "prob_modelo_a": round(row.prob_vitoria_time_a, 4),
            "prob_mercado_a": round(market["prob_time_a"], 4),
            "edge_a": round(edge_a, 4),
            "sinal_a": classify_edge(edge_a),
            "prob_modelo_x": round(row.prob_empate, 4),
            "prob_mercado_x": round(market["prob_empate"], 4),
            "edge_x": round(edge_x, 4),
            "sinal_x": classify_edge(edge_x),
            "prob_modelo_b": round(row.prob_vitoria_time_b, 4),
            "prob_mercado_b": round(market["prob_time_b"], 4),
            "edge_b": round(edge_b, 4),
            "sinal_b": classify_edge(edge_b),
            "margem_casa": round(market["overround"], 4),
        })
    return pd.DataFrame(rows)
