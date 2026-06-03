"""
streamlit_app.py
================

App de demonstração da V1. Permite escolher dois times e ver as
probabilidades estimadas (1X2, over/under, ambos marcam, placar provável),
além de inspecionar a matriz de placares.

Rode com:

    streamlit run app/streamlit_app.py

Requer ``streamlit`` instalado (ver requirements.txt).
"""

from __future__ import annotations

import os
import sys

# Permite importar os módulos de src/ ao rodar a partir da raiz do projeto.
SRC = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(SRC))

import numpy as np
import pandas as pd
import streamlit as st

import data_loader
from elo_model import build_elo_history
from feature_engineering import build_team_strengths, create_elo_diff
from goal_model import (
    estimate_lambdas_for_fixture,
    calculate_score_matrix,
    probabilities_from_matrix,
)


@st.cache_data
def load_model(n_history: int = 600, n_recent: int = 10):
    """Treina o modelo (Elo + forças) uma única vez e mantém em cache."""
    matches = data_loader.generate_mock_matches(n_matches=n_history)
    ratings = build_elo_history(matches)
    matches = create_elo_diff(matches, ratings)
    strengths = build_team_strengths(matches, ratings, n_games=n_recent)
    return ratings, strengths


def main() -> None:
    st.set_page_config(page_title="World Cup Predictor 2026", page_icon="⚽", layout="wide")
    st.title("⚽ World Cup Predictor 2026 — V1 (dados mock)")
    st.caption(
        "Modelo híbrido: Elo + forma recente + gols esperados + Poisson + Monte Carlo. "
        "Dados fictícios para demonstração — não usar para apostas reais."
    )

    ratings, strengths = load_model()
    teams = sorted(strengths.index.tolist())

    col1, col2, col3 = st.columns(3)
    with col1:
        time_a = st.selectbox("Time A", teams, index=teams.index("Brasil") if "Brasil" in teams else 0)
    with col2:
        time_b = st.selectbox("Time B", teams, index=teams.index("Alemanha") if "Alemanha" in teams else 1)
    with col3:
        eliminatorio = st.checkbox("Jogo eliminatório (mata-mata)", value=False)

    if time_a == time_b:
        st.warning("Escolha dois times diferentes.")
        return

    diff_elo = ratings.get(time_a, 1500) - ratings.get(time_b, 1500)
    lambda_a, lambda_b = estimate_lambdas_for_fixture(
        time_a, time_b, strengths, diferenca_elo=diff_elo,
        jogo_eliminatorio=int(eliminatorio),
    )
    matrix = calculate_score_matrix(lambda_a, lambda_b)
    probs = probabilities_from_matrix(matrix)

    st.subheader(f"{time_a} x {time_b}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"Vitória {time_a}", f"{probs['prob_vitoria_time_a']*100:.0f}%")
    m2.metric("Empate", f"{probs['prob_empate']*100:.0f}%")
    m3.metric(f"Vitória {time_b}", f"{probs['prob_vitoria_time_b']*100:.0f}%")
    m4.metric("Placar provável", probs["placar_mais_provavel"])

    n1, n2, n3, n4 = st.columns(4)
    n1.metric("λ Time A", f"{lambda_a:.2f}")
    n2.metric("λ Time B", f"{lambda_b:.2f}")
    n3.metric("Over 2.5", f"{probs['prob_over_2_5']*100:.0f}%")
    n4.metric("Ambos marcam", f"{probs['prob_ambos_marcam']*100:.0f}%")

    st.subheader("Matriz de placares (P de gols A × gols B)")
    n = matrix.shape[0]
    matrix_df = pd.DataFrame(
        np.round(matrix * 100, 1),
        index=[f"A={i}" for i in range(n)],
        columns=[f"B={j}" for j in range(n)],
    )
    st.dataframe(matrix_df.style.background_gradient(cmap="Blues"), use_container_width=True)

    st.info(
        "Lembrete: probabilidades são estimativas com incerteza. O objetivo é "
        "calibração, não 'adivinhar' o resultado."
    )


if __name__ == "__main__":
    main()
