"""Batch prediction pipeline for tennis matches."""
from __future__ import annotations

import os
import sys

import pandas as pd

from . import elo as _elo
from . import stats as _stats
from .markets import analyze_match_odds
from .predict import predict_match


def run_pipeline(
    n_history: int = 500,
    n_recent_matches: int = 20,
    write_outputs: bool = True,
    real_data_path: str | None = None,
    real_stats_path: str | None = None,
    output_dir: str = "outputs",
) -> dict:
    """
    Full pipeline: match history → Elo → predictions → edge analysis.

    Returns {"matches", "ratings", "predictions", "edges"}.
    """
    print("=" * 56)
    print("  Tennis Predictor — Pipeline")
    print("=" * 56)

    # 1. Load / generate match history
    if real_data_path:
        print(f"  Carregando histórico real: {real_data_path}")
        matches = pd.read_csv(real_data_path, parse_dates=["date"])
    else:
        print(f"  Gerando {n_history} partidas mock...")
        matches = _stats.generate_mock_matches(n_matches=n_history)

    print(f"  Partidas no histórico: {len(matches)}")
    print(f"  Jogadores únicos     : {pd.concat([matches['player_a'], matches['player_b']]).nunique()}")

    # 2. Build Elo ratings
    print("  Calculando Elo por superfície...")
    ratings = _elo.build_elo_history(matches)

    # 3. Load / generate serve stats
    stats_db = _stats.load_stats_from_csv(real_stats_path) if real_stats_path else None

    # 4. Load / generate fixtures
    fixtures = _stats.generate_mock_fixtures()
    print(f"  Fixtures para prever : {len(fixtures)}")

    # 5. Predict each fixture
    print("  Calculando probabilidades...")
    prediction_rows = []
    for _, row in fixtures.iterrows():
        r = predict_match(
            row["player_a"],
            row["player_b"],
            surface=row["surface"],
            best_of=row["best_of"],
            data_path=real_data_path,
            stats_path=real_stats_path,
            n_recent_matches=n_recent_matches,
        )
        prediction_rows.append(r)

    predictions_df = pd.DataFrame(prediction_rows)

    # 6. Generate mock odds and run edge analysis
    print("  Gerando odds de mercado e analisando edges...")
    odds_df = _stats.generate_mock_odds(fixtures, predictions_df)

    edge_rows = []
    for i, row in odds_df.iterrows():
        pred = prediction_rows[i]
        market = {
            "odd_player_a": row["odd_player_a"],
            "odd_player_b": row["odd_player_b"],
        }
        edge_rows.append(analyze_match_odds(pred, market))

    edges_df = pd.DataFrame(edge_rows)

    # 7. Optionally write outputs
    if write_outputs:
        os.makedirs(output_dir, exist_ok=True)
        pred_path = os.path.join(output_dir, "predictions.csv")
        edges_path = os.path.join(output_dir, "edges.csv")
        predictions_df.to_csv(pred_path, index=False)
        edges_df.to_csv(edges_path, index=False)
        print(f"  Salvo: {pred_path}")
        print(f"  Salvo: {edges_path}")

    _print_summary({"predictions": predictions_df, "edges": edges_df})

    return {
        "matches": matches,
        "ratings": ratings,
        "predictions": predictions_df,
        "edges": edges_df,
    }


def _print_summary(result: dict) -> None:
    """Print prediction and edge summary."""
    preds = result["predictions"]
    edges = result["edges"]

    print("\n" + "=" * 56)
    print("  PREVISÕES")
    print("=" * 56)
    cols = ["player_a", "player_b", "surface", "best_of", "prob_final_a", "prob_final_b"]
    available_cols = [c for c in cols if c in preds.columns]
    print(preds[available_cols].to_string(index=False))

    print("\n" + "=" * 56)
    print("  EDGES vs MERCADO")
    print("=" * 56)
    if "sinal_a" in edges.columns:
        strong = edges[edges["sinal_a"].isin(["sinal forte", "sinal moderado"])][
            ["player_a", "player_b", "surface", "edge_a", "sinal_a", "edge_b", "sinal_b", "margem_casa"]
        ]
        if not strong.empty:
            print(strong.to_string(index=False))
        else:
            print("  Nenhum sinal moderado ou forte encontrado.")
    print("=" * 56)


if __name__ == "__main__":
    import os as _os
    real_path = sys.argv[1] if len(sys.argv) > 1 else None
    # Run from the tennis_predictor directory
    base_dir = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    out_dir = _os.path.join(base_dir, "outputs")

    run_pipeline(
        real_data_path=real_path,
        output_dir=out_dir,
    )
