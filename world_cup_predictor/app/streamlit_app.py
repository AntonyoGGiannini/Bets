"""
streamlit_app.py — World Cup 2026 Predictor
============================================

Rode com:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os
import sys

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, SRC)

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Constantes
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
    # Inter-confederações
    "Costa Rica", "Ukraine",
]

HOSTS = {"United States", "Mexico", "Canada"}
REAL_CSV = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "results.csv"))
HAS_REAL = os.path.exists(REAL_CSV)


# ---------------------------------------------------------------------------
# Cache: carrega e treina modelo UMA vez
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Carregando histórico e treinando modelo...")
def load_model():
    from elo_model import build_elo_history
    from feature_engineering import build_team_strengths, create_elo_diff

    if HAS_REAL:
        from data_fetcher import load_kaggle_results, filter_copa_teams
        matches_all = load_kaggle_results(REAL_CSV, min_date="2000-01-01")
        matches = filter_copa_teams(matches_all, teams=COPA_2026_TEAMS)
        fonte = "real"
    else:
        import data_loader
        matches = data_loader.generate_mock_matches(n_matches=600)
        matches_all = matches
        fonte = "mock"

    ratings = build_elo_history(matches)
    m_elo  = create_elo_diff(matches, ratings)
    strengths = build_team_strengths(m_elo, ratings, n_games=10)
    return ratings, strengths, matches, matches_all, fonte


def _available(strengths, ratings):
    avail = [t for t in COPA_2026_TEAMS if t in strengths.index]
    return sorted(avail, key=lambda t: -ratings.get(t, 1500))


# ---------------------------------------------------------------------------
# Layout principal
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="World Cup 2026 Predictor", page_icon="⚽", layout="wide")
    st.title("⚽ World Cup 2026 Predictor")

    ratings, strengths, matches, matches_all, fonte = load_model()
    teams = _available(strengths, ratings)

    # Barra de status do dataset
    last = matches.iloc[-1]
    last_date = pd.Timestamp(last.data_jogo).date()
    st.caption(
        f"**Fonte:** {'dados reais — ' + f'{len(matches):,}' + ' partidas (Copa 2026 teams)' if fonte == 'real' else 'dados mock'}  ·  "
        f"**Último jogo no dataset:** {last.time_a} {int(last.gols_time_a)}×{int(last.gols_time_b)} {last.time_b} "
        f"em {last_date} ({last.competicao})"
    )
    if fonte == "mock":
        st.warning("CSV real não encontrado. Baixe `results.csv` — veja o README.")

    # Abas
    tab_pred, tab_odds, tab_rank, tab_dados = st.tabs(
        ["🎯 Prever Jogo", "📈 Análise de Odds", "🏆 Ranking", "🗃️ Dados"]
    )

    # ===================================================================
    # TAB 1 — Previsão
    # ===================================================================
    with tab_pred:
        _tab_prediction(teams, ratings, strengths)

    # ===================================================================
    # TAB 2 — Análise de Odds
    # ===================================================================
    with tab_odds:
        _tab_odds(teams, ratings, strengths)

    # ===================================================================
    # TAB 3 — Ranking
    # ===================================================================
    with tab_rank:
        _tab_ranking(teams, ratings, strengths)

    # ===================================================================
    # TAB 4 — Dados e calibração
    # ===================================================================
    with tab_dados:
        _tab_dados(matches, matches_all, fonte)


# ---------------------------------------------------------------------------
# TAB 1 — Previsão
# ---------------------------------------------------------------------------
def _tab_prediction(teams, ratings, strengths):
    from goal_model import estimate_lambdas_for_fixture, calculate_score_matrix, probabilities_from_matrix
    from monte_carlo import simulate_match

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        idx_a = teams.index("Mexico") if "Mexico" in teams else 0
        time_a = st.selectbox("Time A", teams, index=idx_a, key="pred_a",
                              format_func=lambda t: f"{t}  (Elo {ratings.get(t,1500):.0f})")
    with col2:
        idx_b = teams.index("South Africa") if "South Africa" in teams else min(1, len(teams)-1)
        time_b = st.selectbox("Time B", teams, index=idx_b, key="pred_b",
                              format_func=lambda t: f"{t}  (Elo {ratings.get(t,1500):.0f})")
    with col3:
        eliminatorio = st.checkbox("Mata-mata", value=False, key="pred_elim")

    if time_a == time_b:
        st.warning("Escolha dois times diferentes.")
        return

    # Vantagem de campo — anfitriões 2026
    mando_neutro = 1
    is_host_a, is_host_b = time_a in HOSTS, time_b in HOSTS
    if is_host_a:
        if st.toggle(f"⚑ {time_a} joga em casa (anfitrião)", value=True, key="tog_a"):
            mando_neutro = 0
    elif is_host_b:
        st.info(f"ℹ️ {time_b} é anfitrião mas está como visitante — sem bônus de campo.")

    # Lambdas + Poisson
    diff_elo = ratings.get(time_a, 1500) - ratings.get(time_b, 1500)
    la, lb = estimate_lambdas_for_fixture(
        time_a, time_b, strengths,
        diferenca_elo=diff_elo,
        jogo_eliminatorio=int(eliminatorio),
        mando_neutro=mando_neutro,
    )
    matrix = calculate_score_matrix(la, lb)
    probs  = probabilities_from_matrix(matrix)

    # Monte Carlo
    mc = simulate_match(la, lb, n_simulations=10_000, seed=42)

    ctx = "mata-mata" if eliminatorio else ("em casa" if mando_neutro == 0 else "campo neutro")
    st.divider()
    st.subheader(f"{time_a}  ×  {time_b}  —  _{ctx}_")

    # Métricas principais (Poisson)
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric(f"Vitória {time_a}", f"{probs['prob_vitoria_time_a']*100:.0f}%",
              help="Probabilidade Poisson")
    c2.metric("Empate",            f"{probs['prob_empate']*100:.0f}%")
    c3.metric(f"Vitória {time_b}", f"{probs['prob_vitoria_time_b']*100:.0f}%")
    c4.metric("Over 2.5",          f"{probs['prob_over_2_5']*100:.0f}%")
    c5.metric("Ambos marcam",      f"{probs['prob_ambos_marcam']*100:.0f}%")

    # Verificação Monte Carlo (delta vs Poisson)
    d1,d2,d3,d4,d5 = st.columns(5)
    d1.metric("MC Vitória " + time_a, f"{mc['prob_vitoria_time_a']*100:.0f}%",
              delta=f"{(mc['prob_vitoria_time_a']-probs['prob_vitoria_time_a'])*100:+.1f}pp",
              help="Monte Carlo 10.000 simulações — delta vs Poisson")
    d2.metric("MC Empate",             f"{mc['prob_empate']*100:.0f}%",
              delta=f"{(mc['prob_empate']-probs['prob_empate'])*100:+.1f}pp")
    d3.metric("MC Vitória " + time_b,  f"{mc['prob_vitoria_time_b']*100:.0f}%",
              delta=f"{(mc['prob_vitoria_time_b']-probs['prob_vitoria_time_b'])*100:+.1f}pp")
    d4.metric("λ " + time_a, f"{la:.2f}", help="Gols esperados pelo modelo")
    d5.metric("λ " + time_b, f"{lb:.2f}")

    # Placar e Elo
    e1,e2,e3,e4,e5 = st.columns(5)
    e1.metric("Placar provável (Poisson)", probs["placar_mais_provavel"])
    e2.metric("Placar provável (MC)",      mc["placar_mais_provavel"])
    e3.metric("Elo " + time_a, f"{ratings.get(time_a,1500):.0f}")
    e4.metric("Elo " + time_b, f"{ratings.get(time_b,1500):.0f}")
    e5.metric("Diferença Elo", f"{diff_elo:+.0f}")

    # Mercados adicionais
    with st.expander("Mercados adicionais (Poisson)"):
        mkt = {
            "Over 0.5": probs["prob_over_0_5"],
            "Over 1.5": probs["prob_over_1_5"],
            "Over 2.5": probs["prob_over_2_5"],
            "Under 2.5": probs["prob_under_2_5"],
            "Ambos marcam": probs["prob_ambos_marcam"],
        }
        mkt_df = pd.DataFrame({"Mercado": mkt.keys(), "Probabilidade": [f"{v*100:.1f}%" for v in mkt.values()]})
        st.dataframe(mkt_df, hide_index=True, use_container_width=True)

    # Matriz de placares
    st.subheader("Matriz de placares (%)")
    n = matrix.shape[0]
    mdf = pd.DataFrame(
        np.round(matrix * 100, 1),
        index=[f"{time_a} {i}g" for i in range(n)],
        columns=[f"{time_b} {j}g" for j in range(n)],
    )
    st.dataframe(mdf.style.background_gradient(cmap="Blues"), use_container_width=True)

    st.caption("Delta MC vs Poisson < 1pp confirma consistência do modelo. "
               "Campo neutro para quase todos os jogos da Copa — exceto anfitriões.")


# ---------------------------------------------------------------------------
# TAB 2 — Análise de Odds
# ---------------------------------------------------------------------------
def _tab_odds(teams, ratings, strengths):
    from goal_model import estimate_lambdas_for_fixture, calculate_score_matrix, probabilities_from_matrix
    from odds_analysis import remove_bookmaker_margin, calculate_edge, classify_edge

    st.subheader("Comparação modelo × mercado")
    st.caption("Insira as odds decimais de uma casa para calcular o *edge* do modelo.")

    col1, col2 = st.columns(2)
    with col1:
        idx_a = teams.index("Mexico") if "Mexico" in teams else 0
        time_a = st.selectbox("Time A", teams, index=idx_a, key="odds_a",
                              format_func=lambda t: f"{t}  (Elo {ratings.get(t,1500):.0f})")
    with col2:
        idx_b = teams.index("South Africa") if "South Africa" in teams else min(1, len(teams)-1)
        time_b = st.selectbox("Time B", teams, index=idx_b, key="odds_b",
                              format_func=lambda t: f"{t}  (Elo {ratings.get(t,1500):.0f})")

    if time_a == time_b:
        st.warning("Escolha dois times diferentes.")
        return

    st.markdown("**Odds decimais de mercado:**")
    oc1, oc2, oc3 = st.columns(3)
    with oc1:
        odd_a    = st.number_input(f"Odd {time_a}", min_value=1.01, value=2.10, step=0.05, format="%.2f")
    with oc2:
        odd_draw = st.number_input("Odd Empate",   min_value=1.01, value=3.30, step=0.05, format="%.2f")
    with oc3:
        odd_b    = st.number_input(f"Odd {time_b}", min_value=1.01, value=3.50, step=0.05, format="%.2f")

    # Modelo
    diff_elo = ratings.get(time_a, 1500) - ratings.get(time_b, 1500)
    la, lb = estimate_lambdas_for_fixture(time_a, time_b, strengths,
                                          diferenca_elo=diff_elo, mando_neutro=1)
    probs = probabilities_from_matrix(calculate_score_matrix(la, lb))
    p_model = {
        "vitoria_a": probs["prob_vitoria_time_a"],
        "empate":    probs["prob_empate"],
        "vitoria_b": probs["prob_vitoria_time_b"],
    }

    # Mercado (remove margem)
    market = remove_bookmaker_margin(odd_a, odd_draw, odd_b)
    margem = market["overround"]

    st.divider()
    m1,m2,m3,m4 = st.columns(4)
    m4.metric("Margem da casa", f"{margem*100:.1f}%",
              help="Overround: quanto acima de 100% somam as probs implícitas")

    for label, mk, pm, col in [
        (f"Vitória {time_a}", "vitoria_a", market["prob_time_a"], m1),
        ("Empate",            "empate",    market["prob_empate"],  m2),
        (f"Vitória {time_b}", "vitoria_b", market["prob_time_b"], m3),
    ]:
        edge = calculate_edge(p_model[mk], pm)
        sinal = classify_edge(edge)
        col.metric(label,
                   f"Modelo {p_model[mk]*100:.1f}%  /  Mercado {pm*100:.1f}%",
                   delta=f"Edge {edge*100:+.1f}pp — {sinal}")

    st.caption("`edge` = prob. modelo − prob. implícita (sem margem). "
               "Positivo = modelo mais otimista que o mercado. "
               "Não é recomendação de aposta.")


# ---------------------------------------------------------------------------
# TAB 3 — Ranking
# ---------------------------------------------------------------------------
def _tab_ranking(teams, ratings, strengths):
    st.subheader("Ranking Elo — Copa 2026 (48 seleções)")

    conf_map = {}
    for t in ["France","Spain","England","Germany","Portugal","Netherlands",
              "Italy","Belgium","Croatia","Denmark","Austria","Switzerland",
              "Poland","Serbia","Turkey","Scotland"]:
        conf_map[t] = "UEFA"
    for t in ["Argentina","Brazil","Uruguay","Colombia","Ecuador","Venezuela"]:
        conf_map[t] = "CONMEBOL"
    for t in ["United States","Mexico","Canada","Panama","Honduras","Jamaica"]:
        conf_map[t] = "CONCACAF"
    for t in ["Morocco","Senegal","Egypt","Nigeria","Cameroon","Ghana",
              "Algeria","South Africa","Mali"]:
        conf_map[t] = "CAF"
    for t in ["Japan","South Korea","Australia","Saudi Arabia","Iran",
              "Qatar","Indonesia","Uzbekistan"]:
        conf_map[t] = "AFC"
    for t in ["New Zealand","Costa Rica","Ukraine"]:
        conf_map[t] = "OFC/Inter"

    rows = []
    for t in teams:
        s = strengths.loc[t] if t in strengths.index else None
        rows.append({
            "Seleção": t,
            "Confederação": conf_map.get(t, ""),
            "Elo": round(ratings.get(t, 1500)),
            "λ Ataque": round(float(s["forca_ofensiva"]), 2) if s is not None else "-",
            "λ Defesa": round(float(s["fragilidade_defensiva"]), 2) if s is not None else "-",
            "Gols marcados (média)": round(float(s["media_gols_marcados"]), 2) if s is not None else "-",
            "Gols sofridos (média)": round(float(s["media_gols_sofridos"]), 2) if s is not None else "-",
            "Saldo médio": round(float(s["saldo_medio_gols"]), 2) if s is not None else "-",
            "Anfitrião": "✓" if t in HOSTS else "",
        })

    df = pd.DataFrame(rows).reset_index(drop=True)
    df.index += 1

    conf_filter = st.multiselect("Filtrar confederação",
                                 sorted(df["Confederação"].unique()),
                                 default=sorted(df["Confederação"].unique()))
    df_show = df[df["Confederação"].isin(conf_filter)]
    st.dataframe(df_show.style.background_gradient(subset=["Elo"], cmap="Greens"),
                 use_container_width=True)


# ---------------------------------------------------------------------------
# TAB 4 — Dados e calibração
# ---------------------------------------------------------------------------
def _tab_dados(matches, matches_all, fonte):
    st.subheader("Informações do dataset")

    last     = matches.iloc[-1]
    last_all = matches_all.iloc[-1]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total de partidas (dataset)",         f"{len(matches_all):,}")
    col2.metric("Partidas (times da Copa 2026)",        f"{len(matches):,}")
    col3.metric("Período coberto",
                f"{matches_all['data_jogo'].min().year} – {matches_all['data_jogo'].max().year}")
    col4.metric("Gols/time (média real)",
                f"{matches_all[['gols_time_a','gols_time_b']].values.mean():.2f}")

    st.divider()
    st.markdown("**Último jogo no dataset (geral):**")
    st.info(
        f"**{last_all.time_a}** {int(last_all.gols_time_a)} × {int(last_all.gols_time_b)} "
        f"**{last_all.time_b}** — "
        f"{pd.Timestamp(last_all.data_jogo).strftime('%d/%m/%Y')} "
        f"({last_all.competicao})"
    )
    st.markdown("**Último jogo entre times da Copa 2026:**")
    st.success(
        f"**{last.time_a}** {int(last.gols_time_a)} × {int(last.gols_time_b)} "
        f"**{last.time_b}** — "
        f"{pd.Timestamp(last.data_jogo).strftime('%d/%m/%Y')} "
        f"({last.competicao})"
    )

    st.divider()
    st.subheader("Distribuição por competição")
    dist = matches["competicao"].value_counts().reset_index()
    dist.columns = ["Competição", "Partidas"]
    st.bar_chart(dist.set_index("Competição"))

    st.subheader("Distribuição de gols por time por jogo")
    gols = pd.concat([matches["gols_time_a"], matches["gols_time_b"]])
    gols_dist = gols.value_counts().sort_index().reset_index()
    gols_dist.columns = ["Gols", "Frequência"]
    st.bar_chart(gols_dist.set_index("Gols"))

    st.divider()
    st.subheader("Calibração do modelo (backtest walk-forward)")
    st.caption("O backtest testa cada jogo usando só o histórico anterior (sem vazamento). "
               "Pode demorar ~30s.")
    if st.button("▶ Rodar backtest agora"):
        _run_backtest_ui(matches)

    if fonte == "real":
        st.divider()
        with st.expander("⬇️ Como atualizar os dados"):
            st.code(
                "# PowerShell / Windows\n"
                "Invoke-WebRequest -Uri "
                "\"https://raw.githubusercontent.com/martj42/international_results/master/results.csv\" "
                "-OutFile \"data\\raw\\results.csv\"\n\n"
                "# macOS / Linux\n"
                "curl -o data/raw/results.csv "
                "https://raw.githubusercontent.com/martj42/international_results/master/results.csv",
                language="bash",
            )
            st.info("Após baixar, reinicie o app (⋮ → Rerun) para usar os dados mais recentes.")


def _run_backtest_ui(matches):
    from backtest import run_backtest, _print_report
    from evaluation import calibration_curve_data

    with st.spinner("Rodando backtest walk-forward..."):
        result = run_backtest(matches=matches, warmup=200)

    m = result["metrics"]
    st.success("Backtest concluído.")

    b1,b2,b3,b4 = st.columns(4)
    b1.metric("Brier (modelo)",    m["brier_modelo"],
              delta=f"{m['brier_modelo']-m['brier_baseline_33']:.3f} vs baseline",
              delta_color="inverse")
    b2.metric("Brier (baseline 33/33/33)", m["brier_baseline_33"])
    b3.metric("LogLoss (modelo)",   m["logloss_modelo"],
              delta=f"{m['logloss_modelo']-m['logloss_baseline_33']:.3f} vs baseline",
              delta_color="inverse")
    b4.metric("Acurácia direcional", f"{m['acuracia_direcional']*100:.1f}%")

    g1,g2,g3 = st.columns(3)
    g1.metric("Gols MAE",  m["gols_mae"])
    g2.metric("Gols RMSE", m["gols_rmse"])
    g3.metric("Gols bias", m["gols_bias"],
              help="Positivo = subestima gols em média")

    preds = result["predictions"]
    y_true = preds["resultado"].to_numpy()
    y_prob  = preds["p_a"].to_numpy()
    win_a   = (y_true == 0).astype(int)
    mean_p, obs_f, counts = calibration_curve_data(win_a, y_prob, n_bins=10)

    cal_df = pd.DataFrame({
        "Prob prevista (bin)": np.round(mean_p, 2),
        "Freq observada":      np.round(obs_f, 2),
        "N jogos":             counts,
    }).dropna()
    st.subheader("Curva de calibração (vitória time A)")
    st.caption("Uma calibração perfeita teria `Prob prevista ≈ Freq observada`.")
    st.dataframe(cal_df, hide_index=True, use_container_width=True)
    st.line_chart(cal_df.set_index("Prob prevista (bin)")["Freq observada"])


if __name__ == "__main__":
    main()
