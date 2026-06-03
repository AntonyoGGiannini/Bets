"""
backtest.py
===========

Backtest walk-forward simples (semente da V3) sobre o histórico mockado.

Percorre as partidas em ordem cronológica. Para cada jogo, usa apenas a
informação disponível ANTES dele (Elo corrente + forma recente acumulada)
para prever 1X2, registra a previsão e só então atualiza o Elo com o
resultado real. Ao final, avalia a calibração com Brier Score, Log Loss e
acurácia direcional.

Objetivo (seção 15): provar que o modelo está razoavelmente calibrado antes
de buscar edge — "quando diz 60%, acontece perto de 60% das vezes".

Rode com:

    python src/backtest.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

import data_loader
from elo_model import initialize_elo, update_elo_after_match, expected_score, DEFAULT_ELO
from feature_engineering import build_team_strengths, create_elo_diff
from goal_model import estimate_lambdas, calculate_score_matrix, probabilities_from_matrix

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def _result_class(goals_a: int, goals_b: int) -> int:
    """0 = vitória A, 1 = empate, 2 = vitória B."""
    if goals_a > goals_b:
        return 0
    if goals_a == goals_b:
        return 1
    return 2


def run_backtest(
    n_history: int = 800,
    warmup: int = 200,
    n_recent_games: int = 10,
) -> dict:
    """Executa o backtest walk-forward e devolve métricas + DataFrame de previsões.

    Parameters
    ----------
    n_history:
        Número de partidas mock a gerar.
    warmup:
        Partidas iniciais usadas só para aquecer o Elo (não avaliadas).
    n_recent_games:
        Janela de forma recente.
    """
    matches = data_loader.generate_mock_matches(n_matches=n_history)
    teams = set(matches["time_a"]).union(matches["time_b"])
    ratings = initialize_elo(teams)

    records = []
    history_so_far = []  # acumula partidas já vistas para a forma recente

    for i, row in enumerate(matches.itertuples(index=False)):
        ga, gb = int(row.gols_time_a), int(row.gols_time_b)

        if i >= warmup and len(history_so_far) >= n_recent_games:
            hist_df = pd.DataFrame(history_so_far)
            hist_df = create_elo_diff(hist_df, ratings)
            strengths = build_team_strengths(hist_df, ratings, n_games=n_recent_games)

            if row.time_a in strengths.index and row.time_b in strengths.index:
                sa, sb = strengths.loc[row.time_a], strengths.loc[row.time_b]
                features = {
                    "forca_ofensiva_a": float(sa["forca_ofensiva"]),
                    "fragilidade_defensiva_a": float(sa["fragilidade_defensiva"]),
                    "forca_ofensiva_b": float(sb["forca_ofensiva"]),
                    "fragilidade_defensiva_b": float(sb["fragilidade_defensiva"]),
                    "diferenca_elo": ratings[row.time_a] - ratings[row.time_b],
                    "jogo_eliminatorio": 0,
                }
                la, lb = estimate_lambdas(features)
                probs = probabilities_from_matrix(calculate_score_matrix(la, lb))

                records.append({
                    "p_a": probs["prob_vitoria_time_a"],
                    "p_draw": probs["prob_empate"],
                    "p_b": probs["prob_vitoria_time_b"],
                    "lambda_a": la,
                    "lambda_b": lb,
                    "gols_a": ga,
                    "gols_b": gb,
                    "resultado": _result_class(ga, gb),
                })

        # Atualiza o Elo DEPOIS de prever (sem vazamento de informação).
        ratings[row.time_a], ratings[row.time_b] = update_elo_after_match(
            ratings[row.time_a], ratings[row.time_b], ga, gb, row.competicao
        )
        history_so_far.append({
            "data_jogo": row.data_jogo,
            "time_a": row.time_a,
            "time_b": row.time_b,
            "gols_time_a": ga,
            "gols_time_b": gb,
            "competicao": row.competicao,
        })

    preds = pd.DataFrame(records)
    metrics = _evaluate(preds)
    return {"predictions": preds, "metrics": metrics, "ratings": ratings}


def _evaluate(preds: pd.DataFrame) -> dict:
    from evaluation import (
        calculate_brier_score,
        calculate_log_loss,
        directional_accuracy,
        evaluate_goal_prediction,
        calibration_curve_data,
    )

    y_true = preds["resultado"].to_numpy()
    y_prob = preds[["p_a", "p_draw", "p_b"]].to_numpy()

    # Baseline ingênuo: probabilidades fixas médias (33/33/33).
    naive = np.full_like(y_prob, 1.0 / 3.0)

    brier = calculate_brier_score(y_true, y_prob)
    logloss = calculate_log_loss(y_true, y_prob)
    acc = directional_accuracy(y_true, y_prob)
    brier_naive = calculate_brier_score(y_true, naive)
    logloss_naive = calculate_log_loss(y_true, naive)

    # Calibração do evento "vitória do time A".
    win_a = (y_true == 0).astype(int)
    mean_pred, obs_freq, counts = calibration_curve_data(win_a, preds["p_a"].to_numpy(), n_bins=10)

    goal_err = evaluate_goal_prediction(
        np.concatenate([preds["gols_a"], preds["gols_b"]]),
        np.concatenate([preds["lambda_a"], preds["lambda_b"]]),
    )

    return {
        "n_jogos_avaliados": int(len(preds)),
        "brier_modelo": round(brier, 4),
        "brier_baseline_33": round(brier_naive, 4),
        "logloss_modelo": round(logloss, 4),
        "logloss_baseline_33": round(logloss_naive, 4),
        "acuracia_direcional": round(acc, 4),
        "gols_mae": round(goal_err["mae"], 4),
        "gols_rmse": round(goal_err["rmse"], 4),
        "gols_bias": round(goal_err["bias"], 4),
        "calibracao": list(zip(np.round(mean_pred, 3), np.round(obs_freq, 3), counts)),
    }


def _print_report(result: dict) -> None:
    m = result["metrics"]
    print("\n=== BACKTEST WALK-FORWARD (dados mock) ===")
    print(f"Jogos avaliados:        {m['n_jogos_avaliados']}")
    print(f"Brier  modelo:          {m['brier_modelo']}   (baseline 33/33/33: {m['brier_baseline_33']})")
    print(f"LogLoss modelo:         {m['logloss_modelo']}   (baseline 33/33/33: {m['logloss_baseline_33']})")
    print(f"Acurácia direcional:    {m['acuracia_direcional']}")
    print(f"Gols  MAE / RMSE / bias: {m['gols_mae']} / {m['gols_rmse']} / {m['gols_bias']}")
    print("\nCalibração 'vitória do time A' (prob_média, freq_observada, n):")
    for mp, of, c in m["calibracao"]:
        if c > 0:
            bar = "#" * int(of * 30) if of == of else ""
            print(f"  pred={mp:>5}  obs={of:>5}  n={c:>4}  {bar}")
    print(
        "\nLeitura: Brier e LogLoss menores que o baseline indicam que o modelo "
        "agrega informação. A coluna de calibração deve ter pred ≈ obs."
    )


if __name__ == "__main__":
    res = run_backtest()
    _print_report(res)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    res["predictions"].to_csv(os.path.join(OUTPUTS_DIR, "backtest_predictions.csv"), index=False)
    print("\nPrevisões do backtest salvas em outputs/backtest_predictions.csv")
