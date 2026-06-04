"""
streamlit_app.py
================

App interativo do World Cup Predictor 2026.

Rode com:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os
import sys

SRC = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(SRC))

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Times confirmados / prováveis da Copa 2026 (48 seleções)
# Ordem: UEFA | CONMEBOL | CONCACAF | CAF | AFC | OFC
# ---------------------------------------------------------------------------
COPA_2026_TEAMS = [
    # UEFA (16)
    "France", "Spain", "England", "Germany", "Portugal", "Netherlands",
    "Italy", "Belgium", "Croatia", "Denmark", "Austria", "Switzerland",
    "Poland", "Serbia", "Turkey", "Scotland",
    # CONMEBOL (6)
    "Argentina", "Brazil", "Uruguay", "Colombia", "Ecuador", "Venezuela",
    # CONCACAF (6)
    "United States", "Mexico", "Canada", "Panama", "Honduras", "Jamaica",
    # CAF (9)
    "Morocco", "Senegal", "Egypt", "Nigeria", "Cameroon", "Ghana",
    "Algeria", "South Africa", "Mali",
    # AFC (8)
    "Japan", "South Korea", "Australia", "Saudi Arabia", "Iran",
    "Qatar", "Indonesia", "Uzbekistan",
    # OFC (1)
    "New Zealand",
    # Inter-confederações (2) — vagas a definir
    "Costa Rica", "Ukraine",
]

HOSTS = {"United States", "Mexico", "Canada"}

REAL_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "results.csv")


@st.cache_data(show_spinner="Carregando modelo...")
def load_model():
    """Treina Elo + forças. Usa dados reais se o CSV existir, senão mock."""
    from elo_model import build_elo_history
    from feature_engineering import build_team_strengths, create_elo_diff

    if os.path.exists(REAL_CSV):
        from data_fetcher import load_kaggle_results, filter_copa_teams
        matches = filter_copa_teams(
            load_kaggle_results(REAL_CSV, min_date="2000-01-01"),
            teams=COPA_2026_TEAMS,
        )
        fonte = "real"
    else:
        import data_loader
        matches = data_loader.generate_mock_matches(n_matches=600)
        fonte = "mock"

    ratings = build_elo_history(matches)
    matches = create_elo_diff(matches, ratings)
    strengths = build_team_strengths(matches, ratings, n_games=10)
    return ratings, strengths, fonte


def available_teams(strengths, ratings) -> list[str]:
    """Retorna os times da Copa 2026 que têm histórico no modelo, por Elo desc."""
    disponíveis = [t for t in COPA_2026_TEAMS if t in strengths.index]
    return sorted(disponíveis, key=lambda t: -ratings.get(t, 1500))


def main() -> None:
    st.set_page_config(page_title="World Cup 2026 Predictor", page_icon="⚽", layout="wide")

    st.title("⚽ World Cup 2026 Predictor")
    st.caption("Elo + forma recente + Poisson + Monte Carlo")

    ratings, strengths, fonte = load_model()
    teams = available_teams(strengths, ratings)

    if fonte == "mock":
        st.warning(
            "Rodando com **dados mock** (nomes em inglês podem não estar disponíveis). "
            "Para dados reais baixe `results.csv` — veja o README."
        )

    # ------------------------------------------------------------------
    # Controles
    # ------------------------------------------------------------------
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        default_a = teams.index("Mexico") if "Mexico" in teams else 0
        time_a = st.selectbox("Time A", teams, index=default_a,
                              format_func=lambda t: f"{t}  (Elo {ratings.get(t, 1500):.0f})")
    with col2:
        default_b = teams.index("South Africa") if "South Africa" in teams else min(1, len(teams)-1)
        time_b = st.selectbox("Time B", teams, index=default_b,
                              format_func=lambda t: f"{t}  (Elo {ratings.get(t, 1500):.0f})")
    with col3:
        eliminatorio = st.checkbox("Mata-mata", value=False)

    if time_a == time_b:
        st.warning("Escolha dois times diferentes.")
        return

    # Vantagem de campo — só anfitriões em jogos de grupo
    is_host_a = time_a in HOSTS
    is_host_b = time_b in HOSTS
    if is_host_a or is_host_b:
        mandante_label = (
            f"⚑ {time_a} joga em casa (anfitrião 2026)"
            if is_host_a else
            f"⚑ {time_b} joga em casa (anfitrião 2026)"
        )
        mando_home = st.toggle(mandante_label, value=True)
        # mando_neutro=0 quando o anfitrião for time_a; senão inverte se for time_b
        if is_host_a:
            mando_neutro = 0 if mando_home else 1
        else:
            # anfitrião é time_b — troca a ordem temporariamente não é viável;
            # indicamos apenas que não há bônus de campo para time_a nesse caso
            mando_neutro = 1
    else:
        mando_neutro = 1  # Copa = campo neutro

    # ------------------------------------------------------------------
    # Previsão
    # ------------------------------------------------------------------
    from goal_model import estimate_lambdas_for_fixture, calculate_score_matrix, probabilities_from_matrix

    diff_elo = ratings.get(time_a, 1500) - ratings.get(time_b, 1500)
    lambda_a, lambda_b = estimate_lambdas_for_fixture(
        time_a, time_b, strengths,
        diferenca_elo=diff_elo,
        jogo_eliminatorio=int(eliminatorio),
        mando_neutro=mando_neutro,
    )
    matrix = calculate_score_matrix(lambda_a, lambda_b)
    probs = probabilities_from_matrix(matrix)

    # ------------------------------------------------------------------
    # Resultados
    # ------------------------------------------------------------------
    st.divider()
    context = "mata-mata" if eliminatorio else ("em casa" if mando_neutro == 0 else "campo neutro")
    st.subheader(f"{time_a}  ×  {time_b}   _({context})_")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(f"Vitória {time_a}", f"{probs['prob_vitoria_time_a']*100:.0f}%")
    c2.metric("Empate", f"{probs['prob_empate']*100:.0f}%")
    c3.metric(f"Vitória {time_b}", f"{probs['prob_vitoria_time_b']*100:.0f}%")
    c4.metric("Over 2.5", f"{probs['prob_over_2_5']*100:.0f}%")
    c5.metric("Ambos marcam", f"{probs['prob_ambos_marcam']*100:.0f}%")

    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("λ " + time_a, f"{lambda_a:.2f}")
    d2.metric("λ " + time_b, f"{lambda_b:.2f}")
    d3.metric("Placar provável", probs["placar_mais_provavel"])
    d4.metric("Elo " + time_a, f"{ratings.get(time_a, 1500):.0f}")
    d5.metric("Elo " + time_b, f"{ratings.get(time_b, 1500):.0f}")

    # ------------------------------------------------------------------
    # Matriz de placares
    # ------------------------------------------------------------------
    st.subheader("Matriz de placares (%)")
    n = matrix.shape[0]
    matrix_df = pd.DataFrame(
        np.round(matrix * 100, 1),
        index=[f"{time_a} {i}" for i in range(n)],
        columns=[f"{time_b} {j}" for j in range(n)],
    )
    st.dataframe(matrix_df.style.background_gradient(cmap="Blues"), use_container_width=True)

    # ------------------------------------------------------------------
    # Ranking Elo dos times da Copa 2026
    # ------------------------------------------------------------------
    with st.expander("📊 Ranking Elo — times da Copa 2026"):
        ranking = pd.DataFrame([
            {"Seleção": t, "Elo": round(ratings.get(t, 1500)),
             "Jogos (forma)": int(strengths.loc[t, "jogos_considerados"]) if t in strengths.index else 0,
             "λ ataque": round(float(strengths.loc[t, "forca_ofensiva"]), 2) if t in strengths.index else "-",
             "λ defesa": round(float(strengths.loc[t, "fragilidade_defensiva"]), 2) if t in strengths.index else "-"}
            for t in COPA_2026_TEAMS if t in ratings
        ]).sort_values("Elo", ascending=False).reset_index(drop=True)
        ranking.index += 1
        st.dataframe(ranking, use_container_width=True)

    st.caption(
        "As probabilidades são estimativas com incerteza — "
        "campo neutro para a maioria dos jogos da Copa, exceto anfitriões (EUA/México/Canadá). "
        "`edge` ≠ recomendação de aposta."
    )


if __name__ == "__main__":
    main()
