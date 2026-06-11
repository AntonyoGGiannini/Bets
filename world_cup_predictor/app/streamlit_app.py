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
# Constantes — 48 times reais da Copa 2026, organizados por grupo
# ---------------------------------------------------------------------------
COPA_2026_GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

# Lista plana ordenada por grupo (usada nos dropdowns)
COPA_2026_TEAMS = [t for teams in COPA_2026_GROUPS.values() for t in teams]

# Mapa inverso time → grupo
TEAM_GROUP = {t: g for g, teams in COPA_2026_GROUPS.items() for t in teams}

HOSTS = {"United States", "Mexico", "Canada"}
REAL_CSV = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "results.csv"))
RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"


def top3_from_matrix(matrix) -> list[tuple[str, float]]:
    """Top-3 placares mais prováveis (label, prob) a partir da matriz de placares.

    Calculado aqui no app — e não lido de ``probabilities_from_matrix`` — para
    não depender de uma chave nova no dict retornado. No Streamlit Cloud o
    re-deploy reexecuta o script mas pode manter ``goal_model`` em cache no
    ``sys.modules``; derivar o top-3 da própria matriz evita o ``KeyError`` se
    o módulo importado estiver desatualizado.
    """
    flat = np.argsort(matrix, axis=None)[::-1][:3]
    out = []
    for f in flat:
        i, j = np.unravel_index(f, matrix.shape)
        out.append((f"{i}x{j}", float(matrix[i, j])))
    return out


@st.cache_data(show_spinner="Baixando dados reais (results.csv)...")
def ensure_real_csv() -> bool:
    """Garante que o results.csv exista localmente, baixando se necessário.

    Retorna True se o CSV real está disponível (já existia ou foi baixado).
    Em ambientes como o Streamlit Cloud o arquivo não está versionado, então
    é baixado da fonte pública (mesmo dataset do Kaggle, espelhado no GitHub).
    """
    if os.path.exists(REAL_CSV):
        return True
    try:
        import urllib.request

        os.makedirs(os.path.dirname(REAL_CSV), exist_ok=True)
        # Baixa para um arquivo temporário e renomeia (download atômico).
        tmp = REAL_CSV + ".part"
        urllib.request.urlretrieve(RESULTS_URL, tmp)
        os.replace(tmp, REAL_CSV)
        return True
    except Exception as exc:  # rede indisponível, etc. → cai no mock
        st.session_state["_download_error"] = str(exc)
        return False


# ---------------------------------------------------------------------------
# Cache: carrega e treina modelo UMA vez
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Carregando histórico e treinando modelo...")
def load_model():
    from elo_model import build_elo_history
    from feature_engineering import build_team_strengths, create_elo_diff

    has_real = ensure_real_csv()
    if has_real:
        from data_fetcher import load_kaggle_results, filter_copa_teams, COPA_2026_TEAMS as REAL_TEAMS
        matches_all = load_kaggle_results(REAL_CSV, min_date="2000-01-01")
        matches = filter_copa_teams(matches_all, teams=list(REAL_TEAMS))
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
    """Times da Copa 2026 com histórico, ordenados por grupo e depois por Elo."""
    avail = [t for t in COPA_2026_TEAMS if t in strengths.index]
    return sorted(avail, key=lambda t: (TEAM_GROUP.get(t, "Z"), -ratings.get(t, 1500)))


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
        err = st.session_state.get("_download_error")
        if err:
            st.warning(
                "Não foi possível baixar o `results.csv` automaticamente "
                f"(usando dados mock). Detalhe: {err}"
            )
        else:
            st.warning("CSV real não encontrado. Baixe `results.csv` — veja o README.")

    # Abas
    tab_pred, tab_all, tab_sim, tab_odds, tab_rank, tab_dados = st.tabs(
        ["🎯 Prever Jogo", "🗓️ Toda a Copa", "🏆 Simular Torneio",
         "📈 Análise de Odds", "🏅 Ranking", "🗃️ Dados"]
    )

    # ===================================================================
    # TAB 1 — Previsão
    # ===================================================================
    with tab_pred:
        _tab_prediction(teams, ratings, strengths)

    # ===================================================================
    # TAB 2 — Todos os jogos da fase de grupos
    # ===================================================================
    with tab_all:
        _tab_all_games(ratings, strengths)

    # ===================================================================
    # TAB 3 — Simulação do torneio inteiro (até o campeão)
    # ===================================================================
    with tab_sim:
        _tab_simulacao(ratings, strengths, fonte)

    # ===================================================================
    # TAB 4 — Análise de Odds
    # ===================================================================
    with tab_odds:
        _tab_odds(teams, ratings, strengths)

    # ===================================================================
    # TAB 4 — Ranking
    # ===================================================================
    with tab_rank:
        _tab_ranking(teams, ratings, strengths)

    # ===================================================================
    # TAB 5 — Dados e calibração
    # ===================================================================
    with tab_dados:
        _tab_dados(matches, matches_all, fonte)


# ---------------------------------------------------------------------------
# TAB 1 — Previsão
# ---------------------------------------------------------------------------
def _tab_prediction(teams, ratings, strengths):
    from goal_model import estimate_lambdas_for_fixture, calculate_score_matrix, probabilities_from_matrix
    from monte_carlo import simulate_match
    from corner_model import estimate_corner_mus_for_fixture, corner_total_probabilities

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        idx_a = teams.index("Mexico") if "Mexico" in teams else 0
        time_a = st.selectbox("Time A", teams, index=idx_a, key="pred_a",
                              format_func=lambda t: f"[{TEAM_GROUP.get(t,'?')}] {t}  (Elo {ratings.get(t,1500):.0f})")
    with col2:
        idx_b = teams.index("South Africa") if "South Africa" in teams else min(1, len(teams)-1)
        time_b = st.selectbox("Time B", teams, index=idx_b, key="pred_b",
                              format_func=lambda t: f"[{TEAM_GROUP.get(t,'?')}] {t}  (Elo {ratings.get(t,1500):.0f})")
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

    # --- Resultado (1X2) — uma única probabilidade por saída (Poisson) -----
    st.markdown("##### 🏁 Resultado (1X2)")
    c1, c2, c3 = st.columns(3)
    c1.metric(f"1 · Vitória {time_a}", f"{probs['prob_vitoria_time_a']*100:.0f}%")
    c2.metric("X · Empate",            f"{probs['prob_empate']*100:.0f}%")
    c3.metric(f"2 · Vitória {time_b}", f"{probs['prob_vitoria_time_b']*100:.0f}%")

    # --- Gols esperados (feitos e sofridos por seleção) --------------------
    st.markdown("##### ⚽ Gols esperados")
    gcol1, gcol2 = st.columns([3, 1])
    with gcol1:
        gols_df = pd.DataFrame({
            "Seleção": [time_a, time_b],
            "Gols feitos (esperado)":   [round(la, 2), round(lb, 2)],
            "Gols sofridos (esperado)": [round(lb, 2), round(la, 2)],
        })
        st.dataframe(gols_df, hide_index=True, use_container_width=True)
    with gcol2:
        st.metric("Total de gols", f"{la + lb:.2f}",
                  help="Soma dos gols esperados das duas seleções")

    # --- Mercados de gols (Over/Under) -------------------------------------
    st.markdown("##### 📊 Mercados de gols (Over/Under)")
    o1, o2, o3, o4, o5 = st.columns(5)
    o1.metric("Over 0.5",     f"{probs['prob_over_0_5']*100:.0f}%")
    o2.metric("Over 1.5",     f"{probs['prob_over_1_5']*100:.0f}%")
    o3.metric("Over 2.5",     f"{probs['prob_over_2_5']*100:.0f}%")
    o4.metric("Under 2.5",    f"{probs['prob_under_2_5']*100:.0f}%")
    o5.metric("Ambos marcam", f"{probs['prob_ambos_marcam']*100:.0f}%")

    # --- Top-3 placares (em vez de um único "placar provável", que dá a
    # falsa impressão de certeza — o modal costuma ter só ~12%) -------------
    st.markdown("##### 🔢 Placares mais prováveis")
    top3 = top3_from_matrix(matrix)
    p1, p2, p3 = st.columns(3)
    for col, (plc, p) in zip([p1, p2, p3], top3):
        col.metric(plc, f"{p*100:.1f}%")
    st.caption(f"Os 3 placares mais prováveis somam {sum(p for _,p in top3)*100:.0f}% — "
               "nenhum placar isolado domina; futebol tem alta variância.")

    # --- Escanteios previstos ----------------------------------------------
    st.markdown("##### ⛳ Escanteios previstos")
    try:
        mu_c_a, mu_c_b = estimate_corner_mus_for_fixture(
            time_a, time_b, strengths,
            diferenca_elo=diff_elo,
            jogo_eliminatorio=int(eliminatorio),
            mando_neutro=mando_neutro,
        )
        cp = corner_total_probabilities(mu_c_a, mu_c_b)

        s1, s2, s3, s4 = st.columns(4)
        s1.metric(f"Escanteios {time_a}", f"{cp['mu_time_a']:.1f}")
        s2.metric(f"Escanteios {time_b}", f"{cp['mu_time_b']:.1f}")
        s3.metric("Total esperado",       f"{cp['mu_total']:.1f}")
        s4.metric("Over 9.5",             f"{cp['prob_over_9_5']*100:.0f}%")

        with st.expander("⛳ Mercados de escanteios detalhados"):
            st.caption(
                "Modelo Poisson calibrado sobre médias internacionais (~10.3 escanteios/jogo). "
                "Quando dados históricos de escanteios não estiverem disponíveis, "
                "o modelo usa apenas o ajuste de Elo como diferencial entre os times."
            )
            ou_data = {
                "Linha": ["8.5", "9.5", "10.5", "11.5", "12.5"],
                "Over %": [
                    f"{cp['prob_over_8_5']*100:.1f}%",
                    f"{cp['prob_over_9_5']*100:.1f}%",
                    f"{cp['prob_over_10_5']*100:.1f}%",
                    f"{cp['prob_over_11_5']*100:.1f}%",
                    f"{cp['prob_over_12_5']*100:.1f}%",
                ],
                "Under %": [
                    f"{cp['prob_under_8_5']*100:.1f}%",
                    f"{cp['prob_under_9_5']*100:.1f}%",
                    f"{cp['prob_under_10_5']*100:.1f}%",
                    f"{cp['prob_under_11_5']*100:.1f}%",
                    f"{cp['prob_under_12_5']*100:.1f}%",
                ],
            }
            st.dataframe(pd.DataFrame(ou_data), hide_index=True, use_container_width=True)

            sc1, sc2, sc3 = st.columns(3)
            sc1.metric(f"{time_a} Over 4.5 corners",
                       f"{cp['prob_escanteios_time_a_over_4_5']*100:.1f}%")
            sc2.metric(f"{time_b} Over 4.5 corners",
                       f"{cp['prob_escanteios_time_b_over_4_5']*100:.1f}%")
            sc3.metric("Linha Asiática", f"{cp['linha_asiatica_escanteios']:+.1f}",
                       help="Handicap de escanteios do Time A vs Time B")

            sd1, sd2, sd3 = st.columns(3)
            sd1.metric(f"Mais corners: {time_a}",
                       f"{cp['spread_a_vence_escanteios']*100:.1f}%")
            sd2.metric("Igual corners",
                       f"{cp['spread_empate_escanteios']*100:.1f}%")
            sd3.metric(f"Mais corners: {time_b}",
                       f"{cp['spread_b_vence_escanteios']*100:.1f}%")
    except Exception as exc:
        st.warning(f"Previsão de escanteios indisponível: {exc}")

    # --- Bastidores do modelo (Elo + verificação Monte Carlo) --------------
    with st.expander("🔍 Detalhes do modelo (Elo e verificação Monte Carlo)"):
        x1, x2, x3 = st.columns(3)
        x1.metric("Elo " + time_a,  f"{ratings.get(time_a,1500):.0f}")
        x2.metric("Elo " + time_b,  f"{ratings.get(time_b,1500):.0f}")
        x3.metric("Diferença Elo",  f"{diff_elo:+.0f}")

        st.markdown("**Verificação cruzada — Monte Carlo (10.000 simulações):**")
        d1, d2, d3 = st.columns(3)
        d1.metric("MC Vitória " + time_a, f"{mc['prob_vitoria_time_a']*100:.0f}%",
                  delta=f"{(mc['prob_vitoria_time_a']-probs['prob_vitoria_time_a'])*100:+.1f}pp vs Poisson")
        d2.metric("MC Empate",            f"{mc['prob_empate']*100:.0f}%",
                  delta=f"{(mc['prob_empate']-probs['prob_empate'])*100:+.1f}pp vs Poisson")
        d3.metric("MC Vitória " + time_b, f"{mc['prob_vitoria_time_b']*100:.0f}%",
                  delta=f"{(mc['prob_vitoria_time_b']-probs['prob_vitoria_time_b'])*100:+.1f}pp vs Poisson")
        st.caption(
            "O Monte Carlo sorteia gols independentes (sem Dixon-Coles), então uma "
            "pequena diferença no empate vs. Poisson é esperada — é o efeito da "
            "correção de empates. Os dois métodos devem concordar nas demais saídas."
        )

    # Matriz de placares
    st.subheader("Matriz de placares (%)")
    with st.expander("❓ Como interpretar a matriz de placares"):
        st.markdown(
            "**O que é:** cada célula mostra a probabilidade (%) de um placar exato. "
            "A **linha** representa gols do Time A e a **coluna** gols do Time B.\n\n"
            "**Como ler:** a célula `Time A 1g / Time B 0g` = probabilidade do placar 1×0. "
            "Cores mais escuras indicam maior probabilidade.\n\n"
            "**Soma das regiões:**\n"
            "- Células onde linha > coluna → prob. de vitória do Time A\n"
            "- Diagonal principal → prob. de empate (0×0, 1×1, 2×2 …)\n"
            "- Células onde coluna > linha → prob. de vitória do Time B\n\n"
            "**Como usar:** mercados de placar exato em casas de apostas costumam pagar "
            "odds altas — a matriz ajuda a identificar quais placares têm probabilidade "
            "acima da implícita na odd oferecida.\n\n"
            "**Totais (linha/coluna):** a coluna/linha **Total** é a probabilidade marginal — "
            "ex.: `Total` na linha de `Time A 2g` = P(Time A fazer exatamente 2 gols), somando "
            "todos os placares dessa linha.\n\n"
            "**Acumulado (Acum. ≤):** probabilidade acumulada — ex.: `Acum. ≤` em `Time A 2g` = "
            "P(Time A fazer **até** 2 gols). Útil para mercados de over/under por time."
        )
    n = matrix.shape[0]
    core = np.round(matrix * 100, 1)
    row_tot = matrix.sum(axis=1) * 100          # P(Time A = i gols)
    col_tot = matrix.sum(axis=0) * 100          # P(Time B = j gols)
    row_cum = np.cumsum(matrix.sum(axis=1)) * 100   # P(Time A ≤ i gols)
    col_cum = np.cumsum(matrix.sum(axis=0)) * 100   # P(Time B ≤ j gols)

    row_labels = [f"{time_a} {i}g" for i in range(n)]
    col_labels = [f"{time_b} {j}g" for j in range(n)]

    mdf = pd.DataFrame(core, index=row_labels, columns=col_labels)
    # Coluna de totais (sobre os gols do Time A) + acumulado
    mdf["Total"]   = np.round(row_tot, 1)
    mdf["Acum. ≤"] = np.round(row_cum, 1)
    # Linha de totais (sobre os gols do Time B) + acumulado
    mdf.loc["Total"]   = list(np.round(col_tot, 1)) + [100.0, np.nan]
    mdf.loc["Acum. ≤"] = list(np.round(col_cum, 1)) + [np.nan, np.nan]

    styler = (
        mdf.style
        .format("{:.1f}", na_rep="")
        # gradiente só no miolo (placares), para os totais não lavarem a escala
        .background_gradient(cmap="Blues", subset=(row_labels, col_labels))
        .set_properties(subset=(["Total", "Acum. ≤"], slice(None)), **{"font-weight": "bold"})
        .set_properties(subset=(slice(None), ["Total", "Acum. ≤"]), **{"font-weight": "bold"})
    )
    st.dataframe(styler, use_container_width=True)

    st.caption("Campo neutro para quase todos os jogos da Copa — exceto anfitriões "
               "(EUA, México e Canadá) na fase de grupos.")


# ---------------------------------------------------------------------------
# TAB 2 — Todos os jogos da fase de grupos
# ---------------------------------------------------------------------------
def _tab_all_games(ratings, strengths):
    import itertools
    from goal_model import (
        estimate_lambdas_for_fixture, calculate_score_matrix, probabilities_from_matrix,
    )

    st.subheader("Previsão — todos os jogos da fase de grupos")
    with st.expander("❓ Como funciona"):
        st.markdown(
            "Gera os **72 jogos** da fase de grupos (12 grupos × 6 confrontos) e prevê cada "
            "um com Poisson + Dixon-Coles. Os **anfitriões** (EUA, México, Canadá) jogam "
            "em casa nos jogos do seu grupo.\n\n"
            "A **classificação projetada** usa **pontos esperados** "
            "(`3 × P(vitória) + 1 × P(empate)`) somados nos 3 jogos de cada seleção. É uma "
            "estimativa de quem avança — **não** uma simulação completa do torneio "
            "(que exigiria sortear resultados e montar o mata-mata)."
        )

    avail = set(strengths.index)
    match_rows = []
    pts = {t: 0.0 for t in COPA_2026_TEAMS}
    n_jogos = {t: 0 for t in COPA_2026_TEAMS}

    for group, gteams in COPA_2026_GROUPS.items():
        for x, y in itertools.combinations(gteams, 2):
            if x not in avail or y not in avail:
                continue
            # Anfitrião (se houver) joga em casa e entra como Time A.
            if x in HOSTS:
                ta, tb, mando = x, y, 0
            elif y in HOSTS:
                ta, tb, mando = y, x, 0
            else:
                ta, tb, mando = x, y, 1

            diff = ratings.get(ta, 1500) - ratings.get(tb, 1500)
            la, lb = estimate_lambdas_for_fixture(
                ta, tb, strengths, diferenca_elo=diff, mando_neutro=mando,
            )
            matrix = calculate_score_matrix(la, lb)
            pr = probabilities_from_matrix(matrix)
            pa, pe, pb = pr["prob_vitoria_time_a"], pr["prob_empate"], pr["prob_vitoria_time_b"]

            pts[ta] += 3 * pa + pe
            pts[tb] += 3 * pb + pe
            n_jogos[ta] += 1
            n_jogos[tb] += 1

            plc, pplc = top3_from_matrix(matrix)[0]
            match_rows.append({
                "Grupo": group,
                "Mando": "🏠 " + ta if mando == 0 else "neutro",
                "Time A": ta, "Time B": tb,
                "Vit A %": round(pa * 100, 1),
                "Empate %": round(pe * 100, 1),
                "Vit B %": round(pb * 100, 1),
                "Placar": f"{plc} ({pplc*100:.0f}%)",
            })

    if not match_rows:
        st.warning("Sem dados suficientes para gerar os jogos. Verifique o dataset.")
        return

    all_df = pd.DataFrame(match_rows)

    sel = st.selectbox("Filtrar grupo", ["Todos os grupos"] + list(COPA_2026_GROUPS.keys()))
    groups_to_show = list(COPA_2026_GROUPS.keys()) if sel == "Todos os grupos" else [sel]
    show = all_df if sel == "Todos os grupos" else all_df[all_df["Grupo"] == sel]

    # Classificação projetada (pontos esperados)
    st.markdown("**Classificação projetada (pontos esperados)**")
    stand_rows = []
    for g in groups_to_show:
        gt = [t for t in COPA_2026_GROUPS[g] if t in avail]
        ranked = sorted(gt, key=lambda t: -pts[t])
        for pos, t in enumerate(ranked, 1):
            stand_rows.append({
                "Grupo": g, "Pos": pos, "Seleção": t,
                "Anfitrião": "🏠" if t in HOSTS else "",
                "Pts esperados": round(pts[t], 2),
                "Jogos": n_jogos[t],
                "Avança": "✅" if pos <= 2 else ("🟡" if pos == 3 else ""),
            })
    stand_df = pd.DataFrame(stand_rows)
    st.dataframe(
        stand_df.style.background_gradient(subset=["Pts esperados"], cmap="Greens"),
        hide_index=True, use_container_width=True,
    )
    st.caption("✅ top 2 (classificados direto) · 🟡 3º colocado (pode avançar como um dos "
               "8 melhores terceiros). Pontos esperados = soma de `3×P(vit) + 1×P(empate)`.")

    # Jogos
    st.markdown("**Jogos previstos**")
    st.dataframe(
        show.style.background_gradient(subset=["Vit A %", "Empate %", "Vit B %"], cmap="Blues"),
        hide_index=True, use_container_width=True,
    )

    csv = all_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Baixar todos os jogos (CSV)", csv,
        file_name="copa2026_fase_grupos.csv", mime="text/csv",
    )


# ---------------------------------------------------------------------------
# TAB 3 — Simulação Monte Carlo do torneio inteiro (até o campeão)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Simulando a Copa 2026 (grupos → mata-mata)...")
def _run_tournament(_ratings, _strengths, n_sims: int, seed: int, cache_key: str):
    """Roda ``n_sims`` torneios completos e devolve a contagem de títulos.

    ``_ratings``/``_strengths`` têm o prefixo ``_`` para o Streamlit não tentar
    fazer hash deles; o ``cache_key`` (fonte dos dados) entra na chave do cache
    para invalidar quando o dataset muda. Reutiliza as funções de
    ``simulate_tournament_2026`` para não duplicar a lógica do bracket.
    """
    from collections import defaultdict
    import simulate_tournament_2026 as sim

    teams = [t for t in COPA_2026_TEAMS if t in _strengths.index]
    cache, cache_ko = sim.make_lambda_cache(teams, _ratings, _strengths)

    rng = np.random.default_rng(seed)
    champs = defaultdict(int)
    for _ in range(n_sims):
        champs[sim.simulate_once(rng, _ratings, cache, cache_ko)] += 1

    return pd.Series(champs, dtype="int64").sort_values(ascending=False)


def _tab_simulacao(ratings, strengths, fonte):
    st.subheader("Simulação do torneio inteiro — do grupo ao campeão")
    with st.expander("❓ Como funciona"):
        st.markdown(
            "Diferente da aba **Toda a Copa** (que estima pontos esperados na fase de "
            "grupos), aqui o torneio é **simulado por inteiro** via Monte Carlo:\n\n"
            "1. **Fase de grupos** — cada um dos 72 jogos tem o placar sorteado de uma "
            "Poisson com os λ do modelo; monta-se a tabela de cada grupo (pontos, saldo, "
            "gols, Elo no desempate).\n"
            "2. **Classificação** — avançam os **2 primeiros** de cada grupo + os **8 "
            "melhores terceiros** (formato 2026 → 32 seleções).\n"
            "3. **Mata-mata** — bracket seedado por Elo; cada confronto é sorteado "
            "(empate → pênaltis) até sobrar o campeão.\n\n"
            "Repetindo isso milhares de vezes, a **frequência de títulos** de cada "
            "seleção estima sua probabilidade de ser campeã.\n\n"
            "⚠️ O chaveamento por Elo é uma **aproximação** (o bracket oficial 2026 "
            "depende do sorteio real). Jogos tratados como campo neutro."
        )

    teams = [t for t in COPA_2026_TEAMS if t in strengths.index]
    faltando = [t for t in COPA_2026_TEAMS if t not in strengths.index]
    if faltando:
        # A simulação do torneio inteiro precisa dos 12 grupos completos (48
        # seleções). Com algum time sem histórico (ex.: base mock) o bracket
        # ficaria distorcido e ``simulate_once`` quebraria ao indexar um
        # confronto ausente — então bloqueamos com uma mensagem clara em vez
        # de avisar e estourar um KeyError ao clicar em "Simular torneio".
        st.error(
            f"⚠️ Simulação indisponível: {len(faltando)} seleção(ões) sem "
            f"histórico suficiente ({', '.join(faltando)}).\n\n"
            "A simulação completa do torneio exige as **48 seleções** com "
            "histórico (todos os 12 grupos cheios), o que acontece com os "
            "**dados reais**. Verifique se o `results.csv` foi carregado "
            "(veja a aba **🗃️ Dados**)."
        )
        return

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        n_sims = st.select_slider(
            "Número de simulações", options=[500, 1000, 2000, 3000, 5000, 10000],
            value=2000,
            help="Mais simulações = estimativa mais estável, porém mais lenta.",
        )
    with col2:
        seed = st.number_input("Seed", min_value=0, value=42, step=1,
                               help="Fixa o sorteio para resultados reprodutíveis.")
    with col3:
        st.write("")
        st.write("")
        rodar = st.button("▶ Simular torneio", type="primary", use_container_width=True)

    if not rodar:
        st.info("Ajuste os parâmetros e clique em **Simular torneio**.")
        return

    champs = _run_tournament(ratings, strengths, int(n_sims), int(seed), fonte)
    total = int(champs.sum())
    prob = (champs / total * 100)

    df = pd.DataFrame({
        "Seleção": prob.index,
        "Grupo": [TEAM_GROUP.get(t, "?") for t in prob.index],
        "Elo": [round(ratings.get(t, 1500)) for t in prob.index],
        "Títulos": [int(champs[t]) for t in prob.index],
        "Prob. título %": prob.round(1).values,
    }).reset_index(drop=True)
    df.index += 1

    campea = df.iloc[0]
    st.success(
        f"🏆 Campeã mais provável: **{campea['Seleção']}** "
        f"({campea['Prob. título %']:.1f}% dos {total:,} torneios simulados)"
    )

    ctop = st.columns(min(5, len(df)))
    for col, (_, r) in zip(ctop, df.head(5).iterrows()):
        col.metric(r["Seleção"], f"{r['Prob. título %']:.1f}%", help=f"Elo {r['Elo']}")

    st.markdown("**Probabilidade de título por seleção**")
    st.dataframe(
        df.head(20).style.background_gradient(subset=["Prob. título %"], cmap="Greens"),
        use_container_width=True,
    )

    chart_df = df.head(12).set_index("Seleção")["Prob. título %"]
    st.bar_chart(chart_df)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Baixar probabilidades de título (CSV)", csv,
        file_name="copa2026_prob_titulo.csv", mime="text/csv",
    )
    st.caption(
        "Nenhum favorito costuma passar de ~25% — futebol de seleção é de alta "
        "variância. 'Campeã provável' = a aposta menos arriscada, não uma certeza."
    )


# ---------------------------------------------------------------------------
# TAB 4 — Análise de Odds
# ---------------------------------------------------------------------------
def _tab_odds(teams, ratings, strengths):
    from goal_model import estimate_lambdas_for_fixture, calculate_score_matrix, probabilities_from_matrix
    from odds_analysis import remove_bookmaker_margin, calculate_edge, classify_edge

    st.subheader("Comparação modelo × mercado")
    st.caption("Insira as odds decimais de uma casa para calcular o *edge* do modelo.")
    with st.expander("❓ O que é a comparação modelo × mercado?"):
        st.markdown(
            "**O que é:** compara as probabilidades geradas pelo modelo com as probabilidades implícitas "
            "nas odds da casa de apostas — após remover a margem da casa.\n\n"
            "**Edge (vantagem):** `edge = prob_modelo − prob_implícita_mercado`\n"
            "- **Positivo** → o modelo acredita que o resultado é *mais provável* do que a casa sugere "
            "→ a odd está 'grande' em relação à nossa estimativa.\n"
            "- **Negativo** → modelo mais pessimista que o mercado.\n"
            "- **Próximo de zero** → modelo e mercado concordam.\n\n"
            "**Margem da casa (overround):** toda casa cobra uma comissão inflando as probabilidades "
            "implícitas acima de 100%. Removemos essa margem antes de comparar, para a comparação "
            "ser justa.\n\n"
            "**Atenção — edge ≠ recomendação de aposta.** Considere:\n"
            "- A incerteza do modelo (poucos jogos internacionais por seleção)\n"
            "- Se o mercado tem informações que o modelo não tem (lesões, escalações, clima)\n"
            "- Liquidez e limites de aposta da casa"
        )

    col1, col2 = st.columns(2)
    with col1:
        idx_a = teams.index("Mexico") if "Mexico" in teams else 0
        time_a = st.selectbox("Time A", teams, index=idx_a, key="odds_a",
                              format_func=lambda t: f"[{TEAM_GROUP.get(t,'?')}] {t}  (Elo {ratings.get(t,1500):.0f})")
    with col2:
        idx_b = teams.index("South Africa") if "South Africa" in teams else min(1, len(teams)-1)
        time_b = st.selectbox("Time B", teams, index=idx_b, key="odds_b",
                              format_func=lambda t: f"[{TEAM_GROUP.get(t,'?')}] {t}  (Elo {ratings.get(t,1500):.0f})")

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

    def _leitura_edge(edge: float) -> str:
        """Tradução amigável do edge (usa a régua de classify_edge)."""
        sinal = classify_edge(edge)
        if sinal == "sem sinal":
            return "⚖️ Modelo e mercado concordam"
        if edge > 0:
            return {
                "sinal fraco":    "🙂 Odd levemente atrativa",
                "sinal moderado": "✅ Possível valor na odd",
                "sinal forte":    "🔥 Forte valor potencial",
            }[sinal]
        return {
            "sinal fraco":    "🤏 Odd um pouco baixa",
            "sinal moderado": "⚠️ Odd abaixo do justo",
            "sinal forte":    "🚫 Odd muito baixa — evitar",
        }[sinal]

    outcomes = [
        (f"1 · Vitória {time_a}", "vitoria_a", odd_a,    market["prob_time_a"]),
        ("X · Empate",            "empate",    odd_draw, market["prob_empate"]),
        (f"2 · Vitória {time_b}", "vitoria_b", odd_b,    market["prob_time_b"]),
    ]

    st.divider()

    # Métricas resumidas: probabilidade do modelo + delta vs mercado
    m1, m2, m3, m4 = st.columns(4)
    for col, (label, mk, _odd, pm) in zip([m1, m2, m3], outcomes):
        edge = calculate_edge(p_model[mk], pm)
        col.metric(label, f"{p_model[mk]*100:.0f}%",
                   delta=f"{edge*100:+.1f}pp vs mercado",
                   help=f"Modelo {p_model[mk]*100:.1f}% · Mercado (sem margem) {pm*100:.1f}%")
    m4.metric("Margem da casa", f"{margem*100:.1f}%",
              help="Overround: quanto acima de 100% somam as probs implícitas")

    # Tabela detalhada com leitura amigável
    rows = []
    for label, mk, odd, pm in outcomes:
        p = p_model[mk]
        edge = calculate_edge(p, pm)
        rows.append({
            "Resultado": label,
            "Odd da casa": round(odd, 2),
            "Odd justa (modelo)": round(1.0 / p, 2) if p > 0 else None,
            "Prob. modelo": f"{p*100:.1f}%",
            "Prob. mercado": f"{pm*100:.1f}%",
            "Edge (pp)": round(edge * 100, 1),
            "Leitura": _leitura_edge(edge),
        })
    edge_df = pd.DataFrame(rows)

    best = max(rows, key=lambda r: r["Edge (pp)"])
    if best["Edge (pp)"] >= 2.0:
        st.success(
            f"💡 **Melhor oportunidade: {best['Resultado']}** — o modelo estima "
            f"{best['Prob. modelo']} contra {best['Prob. mercado']} do mercado "
            f"({best['Edge (pp)']:+.1f}pp). Odd justa pelo modelo: "
            f"{best['Odd justa (modelo)']} vs {best['Odd da casa']} oferecida."
        )
    else:
        st.info("⚖️ Nenhuma oportunidade clara neste jogo — modelo e mercado "
                "estão de acordo (todos os edges abaixo de 2pp).")

    st.dataframe(
        edge_df.style.background_gradient(
            subset=["Edge (pp)"], cmap="RdYlGn", vmin=-10, vmax=10,
        ),
        hide_index=True, use_container_width=True,
    )

    st.caption(
        "**Como ler:** *Prob. mercado* já vem sem a margem da casa. "
        "*Odd justa* = 1 ÷ probabilidade do modelo — se a odd da casa for **maior** "
        "que a justa, o modelo enxerga valor (edge positivo). "
        "**Edge não é recomendação de aposta** — considere lesões, escalações e a "
        "incerteza do próprio modelo."
    )


# ---------------------------------------------------------------------------
# TAB 3 — Ranking
# ---------------------------------------------------------------------------
def _tab_ranking(teams, ratings, strengths):
    st.subheader("Ranking Elo — Copa 2026 (48 seleções)")
    with st.expander("❓ O que é o rating Elo?"):
        st.markdown(
            "**O que é:** Elo é um sistema de pontuação que mede a força relativa de cada seleção "
            "com base no histórico completo de resultados (vitórias, empates, derrotas e gols).\n\n"
            "**Como funciona:**\n"
            "- Toda seleção começa com **1500 pontos**.\n"
            "- Ao vencer, rouba pontos do adversário; ao perder, cede pontos.\n"
            "- A quantidade transferida depende do resultado esperado: uma vitória surpresa "
            "contra um time muito mais forte vale mais pontos do que uma vitória previsível.\n"
            "- Jogos mais importantes (Copa do Mundo, Eliminatórias) têm fator K maior — "
            "o Elo muda mais rápido após partidas decisivas do que após amistosos.\n\n"
            "**Como usar neste app:**\n"
            "- Elo mais alto → seleção historicamente mais forte.\n"
            "- A **diferença de Elo** ajusta os gols esperados (λ) no modelo: times melhores "
            "têm λ levemente maior.\n"
            "- Uma diferença de ~100 pts ≈ 65% de chance de vitória para o favorito "
            "(em campo neutro, sem outros ajustes)."
        )

    rows = []
    for group, group_teams in COPA_2026_GROUPS.items():
        for t in group_teams:
            s = strengths.loc[t] if t in strengths.index else None
            rows.append({
                "Grupo": group,
                "Seleção": t,
                "Anfitrião": "✓" if t in HOSTS else "",
                "Elo": round(ratings.get(t, 1500)),
                "λ Ataque": round(float(s["forca_ofensiva"]), 2) if s is not None else None,
                "λ Defesa": round(float(s["fragilidade_defensiva"]), 2) if s is not None else None,
                "Gols marcados": round(float(s["media_gols_marcados"]), 2) if s is not None else None,
                "Gols sofridos": round(float(s["media_gols_sofridos"]), 2) if s is not None else None,
                "Saldo": round(float(s["saldo_medio_gols"]), 2) if s is not None else None,
            })

    df = pd.DataFrame(rows)

    view = st.radio("Visualização", ["Por grupo", "Por Elo (ranking geral)"],
                    horizontal=True)

    group_filter = st.multiselect(
        "Filtrar grupo", list(COPA_2026_GROUPS.keys()),
        default=list(COPA_2026_GROUPS.keys()),
    )
    df_show = df[df["Grupo"].isin(group_filter)]

    if view == "Por Elo (ranking geral)":
        df_show = df_show.sort_values("Elo", ascending=False).reset_index(drop=True)
        df_show.index += 1

    st.dataframe(
        df_show.style.background_gradient(subset=["Elo"], cmap="Greens"),
        use_container_width=True,
        hide_index=(view == "Por grupo"),
    )


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
