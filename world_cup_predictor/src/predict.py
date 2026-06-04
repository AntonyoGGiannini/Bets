"""
predict.py
==========

Interface de consumo do modelo para UM confronto.

Carrega o histórico (real, se o CSV existir; senão mock), constrói Elo +
forças de forma recente e devolve as probabilidades de um jogo.

Uso programático:

    from predict import predict_match
    r = predict_match("Mexico", "South Africa", mando_neutro=0)
    print(r["prob_vitoria_time_a"], r["placar_mais_provavel"])

Uso por linha de comando:

    # campo neutro (padrão da Copa)
    python src/predict.py Mexico "South Africa"

    # México mandante (anfitrião 2026)
    python src/predict.py Mexico "South Africa" --casa

    # com histórico real
    python src/predict.py Mexico "South Africa" --casa --dados data/raw/results.csv
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

import pandas as pd

import data_loader
from elo_model import build_elo_history, DEFAULT_ELO
from feature_engineering import build_team_strengths, create_elo_diff
from goal_model import (
    estimate_lambdas_for_fixture,
    calculate_score_matrix,
    probabilities_from_matrix,
)
from monte_carlo import simulate_match


@lru_cache(maxsize=4)
def _train(data_path: Optional[str], n_recent_games: int):
    """Carrega o histórico e devolve (ratings, strengths). Cacheado."""
    if data_path and os.path.exists(data_path):
        from data_fetcher import load_kaggle_results, filter_copa_teams
        matches = filter_copa_teams(load_kaggle_results(data_path, min_date="2000-01-01"))
    else:
        matches = data_loader.generate_mock_matches()
    ratings = build_elo_history(matches)
    matches = create_elo_diff(matches, ratings)
    strengths = build_team_strengths(matches, ratings, n_games=n_recent_games)
    return ratings, strengths


def predict_match(
    time_a: str,
    time_b: str,
    mando_neutro: int = 1,
    jogo_eliminatorio: int = 0,
    data_path: Optional[str] = None,
    n_recent_games: int = 10,
    n_simulations: int = 10_000,
    max_goals: int = 6,
) -> dict:
    """Prevê um confronto e devolve as probabilidades de mercado.

    Parameters
    ----------
    time_a, time_b:
        Nomes das seleções. Devem bater com os nomes no histórico
        (inglês para dados reais, português para o mock).
    mando_neutro:
        1 = campo neutro (padrão; quase toda a Copa). 0 = ``time_a`` é
        mandante (ex.: anfitriões EUA/México/Canadá na fase de grupos).
    jogo_eliminatorio:
        1 em jogos de mata-mata (reduz levemente os gols esperados).
    data_path:
        Caminho para ``results.csv`` (dados reais). Se None ou inexistente,
        usa o histórico mock.

    Returns
    -------
    dict
        Probabilidades (Poisson) + verificação Monte Carlo + λ estimados.
    """
    ratings, strengths = _train(data_path, n_recent_games)

    for t in (time_a, time_b):
        if t not in strengths.index:
            disp = ", ".join(sorted(strengths.index)[:20])
            raise KeyError(
                f"Seleção sem histórico suficiente: {t!r}.\n"
                f"Exemplos disponíveis: {disp} ..."
            )

    diferenca_elo = ratings.get(time_a, DEFAULT_ELO) - ratings.get(time_b, DEFAULT_ELO)
    lambda_a, lambda_b = estimate_lambdas_for_fixture(
        time_a, time_b, strengths,
        diferenca_elo=diferenca_elo,
        jogo_eliminatorio=jogo_eliminatorio,
        mando_neutro=mando_neutro,
    )
    probs = probabilities_from_matrix(calculate_score_matrix(lambda_a, lambda_b, max_goals=max_goals))
    mc = simulate_match(lambda_a, lambda_b, n_simulations=n_simulations, seed=42)

    return {
        "time_a": time_a,
        "time_b": time_b,
        "mando_neutro": mando_neutro,
        "elo_time_a": round(float(ratings.get(time_a, DEFAULT_ELO)), 1),
        "elo_time_b": round(float(ratings.get(time_b, DEFAULT_ELO)), 1),
        "lambda_time_a": round(lambda_a, 3),
        "lambda_time_b": round(lambda_b, 3),
        **{k: round(v, 4) if isinstance(v, float) else v for k, v in probs.items()},
        "mc_prob_vitoria_time_a": round(mc["prob_vitoria_time_a"], 4),
        "mc_prob_empate": round(mc["prob_empate"], 4),
        "mc_prob_vitoria_time_b": round(mc["prob_vitoria_time_b"], 4),
    }


def _print(r: dict) -> None:
    a, b = r["time_a"], r["time_b"]
    casa = "" if r["mando_neutro"] else f"  (mando do {a})"
    print(f"\n=== {a} x {b}{casa} ===")
    print(f"  Elo: {a} {r['elo_time_a']:.0f}  |  {b} {r['elo_time_b']:.0f}")
    print(f"  Gols esperados (λ): {a} {r['lambda_time_a']}  |  {b} {r['lambda_time_b']}")
    print(f"  Vitória {a:14s}: {r['prob_vitoria_time_a']*100:4.1f}%   (MC {r['mc_prob_vitoria_time_a']*100:.1f}%)")
    print(f"  Empate                : {r['prob_empate']*100:4.1f}%   (MC {r['mc_prob_empate']*100:.1f}%)")
    print(f"  Vitória {b:14s}: {r['prob_vitoria_time_b']*100:4.1f}%   (MC {r['mc_prob_vitoria_time_b']*100:.1f}%)")
    print(f"  Over 2.5              : {r['prob_over_2_5']*100:4.1f}%")
    print(f"  Ambos marcam          : {r['prob_ambos_marcam']*100:4.1f}%")
    print(f"  Placar mais provável  : {r['placar_mais_provavel']}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Prevê um confronto com o World Cup Predictor.")
    ap.add_argument("time_a", help="Seleção mandante / time A")
    ap.add_argument("time_b", help="Seleção visitante / time B")
    ap.add_argument("--casa", action="store_true", help="time_a joga em casa (não neutro)")
    ap.add_argument("--mata-mata", action="store_true", help="jogo eliminatório")
    ap.add_argument("--dados", default="data/raw/results.csv",
                    help="CSV de histórico real (cai no mock se não existir)")
    args = ap.parse_args()

    r = predict_match(
        args.time_a, args.time_b,
        mando_neutro=0 if args.casa else 1,
        jogo_eliminatorio=1 if args.mata_mata else 0,
        data_path=args.dados,
    )
    _print(r)
