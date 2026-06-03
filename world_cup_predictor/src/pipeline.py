"""
pipeline.py
===========

Orquestra a V1 de ponta a ponta usando dados mockados:

    histórico -> Elo -> forma recente -> lambdas -> Poisson -> Monte Carlo
    -> tabela de probabilidades (+ comparação opcional com odds).

Rode com:

    python src/pipeline.py

Saídas geradas em ``outputs/``:
    - predictions.csv   (tabela final por partida)
    - simulations.csv   (resultados da simulação Monte Carlo)
    - odds_edges.csv    (comparação modelo vs. mercado, V2)
"""

from __future__ import annotations

import os

import pandas as pd

import data_loader
from elo_model import build_elo_history
from feature_engineering import build_team_strengths, create_elo_diff, create_context_features
from goal_model import (
    estimate_lambdas_for_fixture,
    calculate_score_matrix,
    probabilities_from_matrix,
)
from monte_carlo import simulate_match
from odds_analysis import analyze_fixture_odds

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def run_pipeline(
    n_history: int = 600,
    n_recent_games: int = 10,
    n_simulations: int = 10_000,
    max_goals: int = 6,
    write_outputs: bool = True,
    real_data_path: str | None = None,
) -> dict:
    """Executa a V1 completa e devolve os DataFrames produzidos.

    Parameters
    ----------
    real_data_path:
        Caminho para o CSV real do Kaggle (``results.csv``). Se fornecido,
        usa dados reais (seleções em inglês) em vez do histórico mock; as
        fixtures também passam a usar nomes em inglês.
    """
    # 1. Dados históricos + confrontos futuros + odds.
    if real_data_path:
        from data_fetcher import load_kaggle_results, filter_copa_teams, generate_real_fixtures
        matches = filter_copa_teams(
            load_kaggle_results(real_data_path, min_date="2000-01-01")
        )
        fixtures = generate_real_fixtures()
    else:
        matches = data_loader.generate_mock_matches(n_matches=n_history)
        fixtures = data_loader.generate_mock_fixtures()
    odds = data_loader.generate_mock_odds(fixtures)

    # 2. Elo a partir do histórico cronológico.
    ratings = build_elo_history(matches)

    # 3. Força recente (ataque/defesa) por seleção.
    matches_with_elo = create_elo_diff(matches, ratings)
    strengths = build_team_strengths(matches_with_elo, ratings, n_games=n_recent_games)

    # 4. Para cada confronto futuro: lambdas + Poisson + Monte Carlo.
    fixtures = create_context_features(create_elo_diff(fixtures, ratings))

    pred_rows = []
    sim_rows = []
    for i, row in enumerate(fixtures.itertuples(index=False)):
        lambda_a, lambda_b = estimate_lambdas_for_fixture(
            row.time_a,
            row.time_b,
            strengths,
            diferenca_elo=float(row.diferenca_elo),
            jogo_eliminatorio=int(row.jogo_eliminatorio),
        )

        # Probabilidades exatas via matriz de Poisson.
        matrix = calculate_score_matrix(lambda_a, lambda_b, max_goals=max_goals)
        poisson_probs = probabilities_from_matrix(matrix)

        # Verificação cruzada via Monte Carlo. Seed determinístico pelo índice
        # do confronto — reprodutível entre processos (hash() de strings é
        # salgado por processo e não serve como seed estável).
        mc = simulate_match(lambda_a, lambda_b, n_simulations=n_simulations, seed=2000 + i)

        pred_rows.append({
            "data_jogo": row.data_jogo,
            "time_a": row.time_a,
            "time_b": row.time_b,
            "fase": row.fase,
            "lambda_time_a": round(lambda_a, 3),
            "lambda_time_b": round(lambda_b, 3),
            "prob_vitoria_time_a": round(poisson_probs["prob_vitoria_time_a"], 4),
            "prob_empate": round(poisson_probs["prob_empate"], 4),
            "prob_vitoria_time_b": round(poisson_probs["prob_vitoria_time_b"], 4),
            "prob_over_2_5": round(poisson_probs["prob_over_2_5"], 4),
            "prob_under_2_5": round(poisson_probs["prob_under_2_5"], 4),
            "prob_ambos_marcam": round(poisson_probs["prob_ambos_marcam"], 4),
            "placar_mais_provavel": poisson_probs["placar_mais_provavel"],
        })

        sim_rows.append({
            "time_a": row.time_a,
            "time_b": row.time_b,
            "lambda_time_a": round(lambda_a, 3),
            "lambda_time_b": round(lambda_b, 3),
            **{k: round(v, 4) for k, v in mc.items() if isinstance(v, float)},
            "placar_mc": mc["placar_mais_provavel"],
        })

    predictions = pd.DataFrame(pred_rows)
    simulations = pd.DataFrame(sim_rows)

    # 5. Comparação com odds de mercado (V2).
    edges = analyze_fixture_odds(predictions, odds)

    if write_outputs:
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        predictions.to_csv(os.path.join(OUTPUTS_DIR, "predictions.csv"), index=False)
        simulations.to_csv(os.path.join(OUTPUTS_DIR, "simulations.csv"), index=False)
        edges.to_csv(os.path.join(OUTPUTS_DIR, "odds_edges.csv"), index=False)

    return {
        "matches": matches,
        "ratings": ratings,
        "strengths": strengths,
        "predictions": predictions,
        "simulations": simulations,
        "edges": edges,
    }


def _print_summary(result: dict) -> None:
    predictions = result["predictions"]
    edges = result["edges"]

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)

    print("\n=== TABELA DE PROBABILIDADES (Poisson) ===")
    print(predictions.to_string(index=False))

    print("\n=== EXEMPLO DETALHADO ===")
    r = predictions.iloc[0]
    print(f"{r['time_a']} x {r['time_b']}  ({r['fase']})")
    print(f"  lambda {r['time_a']}: {r['lambda_time_a']}")
    print(f"  lambda {r['time_b']}: {r['lambda_time_b']}")
    print(f"  Vitória {r['time_a']}: {r['prob_vitoria_time_a']*100:.0f}%")
    print(f"  Empate:            {r['prob_empate']*100:.0f}%")
    print(f"  Vitória {r['time_b']}: {r['prob_vitoria_time_b']*100:.0f}%")
    print(f"  Over 2.5:          {r['prob_over_2_5']*100:.0f}%")
    print(f"  Under 2.5:         {r['prob_under_2_5']*100:.0f}%")
    print(f"  Ambos marcam:      {r['prob_ambos_marcam']*100:.0f}%")
    print(f"  Placar provável:   {r['placar_mais_provavel']}")

    print("\n=== EDGE vs. MERCADO (V2) ===")
    cols = ["time_a", "time_b", "prob_modelo_a", "prob_mercado_a", "edge_a", "sinal_a"]
    print(edges[cols].to_string(index=False))

    print("\nArquivos salvos em outputs/: predictions.csv, simulations.csv, odds_edges.csv")
    print("\nLembrete: edge != recomendação. Avalie liquidez, margem e incerteza do modelo.")


if __name__ == "__main__":
    import sys

    # Uso: python src/pipeline.py [caminho_para_results.csv]
    # Com argumento → roda com dados reais; sem argumento → dados mock.
    real_path = sys.argv[1] if len(sys.argv) > 1 else None
    if real_path:
        print(f"Rodando com DADOS REAIS: {real_path}")
    else:
        print("Rodando com DADOS MOCK (passe o caminho de results.csv para dados reais)")
    _print_summary(run_pipeline(real_data_path=real_path))
