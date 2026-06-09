"""
feature_engineering.py
=======================

Criação das variáveis preditivas que alimentam o modelo de gols.

Três blocos, seguindo a seção 5 do projeto:

1. Força estrutural  -> diferença de Elo.
2. Forma recente     -> médias de gols marcados/sofridos nos últimos N jogos,
                        ajustadas pela força dos adversários.
3. Contexto          -> fase do torneio, jogo eliminatório, descanso.

A função pública principal é ``build_team_strengths`` (resumo por seleção,
usado para prever jogos futuros) e ``create_match_features`` (linha a linha,
útil para backtest).
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from elo_model import build_elo_history, DEFAULT_ELO

# Fases consideradas "mata-mata" (jogo eliminatório).
KNOCKOUT_PHASES = {"Oitavas", "Quartas", "Semifinal", "Final", "Mata-mata"}

# Pesos por competição para a forma recente (seção 17.4).
COMPETITION_FORM_WEIGHT = {
    "World Cup": 1.5,
    "Continental": 1.2,
    "Qualifiers": 1.2,
    "Eliminatorias": 1.2,
    "Nations League": 1.0,
    "Friendly": 0.5,
    "Amistoso": 0.5,
}


def create_elo_diff(matches: pd.DataFrame, ratings: Dict[str, float]) -> pd.DataFrame:
    """Adiciona ``elo_time_a``, ``elo_time_b`` e ``diferenca_elo`` ao DataFrame."""
    out = matches.copy()
    out["elo_time_a"] = out["time_a"].map(ratings).fillna(DEFAULT_ELO)
    out["elo_time_b"] = out["time_b"].map(ratings).fillna(DEFAULT_ELO)
    out["diferenca_elo"] = out["elo_time_a"] - out["elo_time_b"]
    return out


def create_context_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Cria variáveis de contexto: jogo eliminatório, descanso e diferença."""
    out = matches.copy()
    out["jogo_eliminatorio"] = out.get("fase", pd.Series(index=out.index, dtype=object)) \
        .isin(KNOCKOUT_PHASES).astype(int)
    out["jogo_grupo"] = (out["jogo_eliminatorio"] == 0).astype(int)

    # Dias de descanso: se não vierem nos dados, assume valores neutros.
    if "dias_descanso_time_a" not in out.columns:
        out["dias_descanso_time_a"] = np.nan
    if "dias_descanso_time_b" not in out.columns:
        out["dias_descanso_time_b"] = np.nan
    out["diferenca_dias_descanso"] = (
        out["dias_descanso_time_a"].fillna(0) - out["dias_descanso_time_b"].fillna(0)
    )
    return out


def _team_long_format(matches: pd.DataFrame) -> pd.DataFrame:
    """Reorganiza as partidas em formato "uma linha por (time, jogo)".

    Facilita o cálculo de forma recente por seleção. Inclui o Elo do adversário
    para permitir o ajuste pela força do oponente.
    """
    base_cols = ["data_jogo", "competicao"]
    has_elo = "elo_time_a" in matches.columns and "elo_time_b" in matches.columns

    has_corners = (
        "escanteios_time_a" in matches.columns
        and "escanteios_time_b" in matches.columns
    )

    a = pd.DataFrame({
        "data_jogo": matches["data_jogo"],
        "competicao": matches.get("competicao", "Friendly"),
        "time": matches["time_a"],
        "adversario": matches["time_b"],
        "gols_marcados": matches["gols_time_a"],
        "gols_sofridos": matches["gols_time_b"],
        "elo_adversario": matches["elo_time_b"] if has_elo else DEFAULT_ELO,
        "escanteios_marcados": matches["escanteios_time_a"].values if has_corners else np.nan,
        "escanteios_sofridos": matches["escanteios_time_b"].values if has_corners else np.nan,
    })
    b = pd.DataFrame({
        "data_jogo": matches["data_jogo"],
        "competicao": matches.get("competicao", "Friendly"),
        "time": matches["time_b"],
        "adversario": matches["time_a"],
        "gols_marcados": matches["gols_time_b"],
        "gols_sofridos": matches["gols_time_a"],
        "elo_adversario": matches["elo_time_a"] if has_elo else DEFAULT_ELO,
        "escanteios_marcados": matches["escanteios_time_b"].values if has_corners else np.nan,
        "escanteios_sofridos": matches["escanteios_time_a"].values if has_corners else np.nan,
    })
    long = pd.concat([a, b], ignore_index=True)
    return long.sort_values("data_jogo").reset_index(drop=True)


def build_team_strengths(
    matches: pd.DataFrame,
    ratings: Dict[str, float],
    n_games: int = 10,
    league_avg_goals: float = 1.35,
) -> pd.DataFrame:
    """Resume a forma recente de cada seleção em índices de ataque e defesa.

    Para cada time calcula, sobre os ``n_games`` jogos mais recentes:

    - ``media_gols_marcados`` / ``media_gols_sofridos`` (ponderada por
      recência e pela importância da competição);
    - ``forca_ofensiva`` = média de gols marcados / média da liga;
    - ``fragilidade_defensiva`` = média de gols sofridos / média da liga;
    - um ajuste pela força dos adversários enfrentados (Elo médio dos
      oponentes vs. Elo médio geral).

    Returns
    -------
    pandas.DataFrame
        Uma linha por seleção com as colunas de força.
    """
    long = _team_long_format(matches)
    avg_elo = float(np.mean(list(ratings.values()))) if ratings else DEFAULT_ELO

    records = []
    for team, grp in long.groupby("time"):
        recent = grp.tail(n_games)
        if recent.empty:
            continue

        # Peso = recência (linear) * peso de competição.
        n = len(recent)
        recency = np.linspace(0.5, 1.0, n)
        comp_w = recent["competicao"].map(COMPETITION_FORM_WEIGHT).fillna(1.0).to_numpy()
        weights = recency * comp_w
        weights = weights / weights.sum()

        gm = float(np.dot(recent["gols_marcados"].to_numpy(), weights))
        gs = float(np.dot(recent["gols_sofridos"].to_numpy(), weights))

        # Ajuste pela força dos adversários: enfrentar Elo alto valoriza o ataque
        # e "perdoa" um pouco a defesa.
        mean_opp_elo = float(recent["elo_adversario"].mean())
        opp_factor = 1.0 + (mean_opp_elo - avg_elo) / 400.0
        opp_factor = float(np.clip(opp_factor, 0.7, 1.3))

        forca_ofensiva = (gm / league_avg_goals) * opp_factor
        fragilidade_defensiva = (gs / league_avg_goals) / opp_factor

        records.append({
            "time": team,
            "elo": ratings.get(team, DEFAULT_ELO),
            "jogos_considerados": n,
            "media_gols_marcados": gm,
            "media_gols_sofridos": gs,
            "saldo_medio_gols": gm - gs,
            "elo_medio_adversarios": mean_opp_elo,
            "forca_ofensiva": max(0.2, forca_ofensiva),
            "fragilidade_defensiva": max(0.2, fragilidade_defensiva),
        })

    df_strengths = pd.DataFrame(records).set_index("time")

    # Merge corner strengths: taxa_escanteio_ataque/defesa + médias.
    corner_cols = [
        "taxa_escanteio_ataque",
        "taxa_escanteio_defesa",
        "media_escanteios_marcados",
        "media_escanteios_sofridos",
    ]
    corner_strengths = build_corner_strengths(matches, ratings, n_games)
    df_strengths = df_strengths.join(corner_strengths[corner_cols], how="left")
    df_strengths["taxa_escanteio_ataque"] = df_strengths["taxa_escanteio_ataque"].fillna(1.0)
    df_strengths["taxa_escanteio_defesa"] = df_strengths["taxa_escanteio_defesa"].fillna(1.0)
    df_strengths["media_escanteios_marcados"] = df_strengths["media_escanteios_marcados"].fillna(4.61)
    df_strengths["media_escanteios_sofridos"] = df_strengths["media_escanteios_sofridos"].fillna(4.61)
    return df_strengths


def build_corner_strengths(
    matches: pd.DataFrame,
    ratings: Dict[str, float],
    n_games: int = 10,
    league_avg_corners: float = 4.61,
) -> pd.DataFrame:
    """Resume a taxa de escanteios de cada seleção nos últimos n_games jogos.

    Quando escanteios_time_a/b estão ausentes do DataFrame (ex.: dados Kaggle
    reais), retorna taxas = 1.0 para todos os times como fallback — as
    previsões usarão apenas o ajuste de Elo.

    Returns
    -------
    pandas.DataFrame indexado por time, colunas:
        taxa_escanteio_ataque   (escanteios gerados / media_liga)
        taxa_escanteio_defesa   (escanteios concedidos / media_liga)
        media_escanteios_marcados
        media_escanteios_sofridos
    """
    has_corners = (
        "escanteios_time_a" in matches.columns
        and "escanteios_time_b" in matches.columns
    )

    if not has_corners:
        teams = sorted(set(matches["time_a"]).union(set(matches["time_b"])))
        records = [
            {
                "time": t,
                "taxa_escanteio_ataque": 1.0,
                "taxa_escanteio_defesa": 1.0,
                "media_escanteios_marcados": league_avg_corners,
                "media_escanteios_sofridos": league_avg_corners,
            }
            for t in teams
        ]
        return pd.DataFrame(records).set_index("time")

    long = _team_long_format(matches)
    avg_elo = float(np.mean(list(ratings.values()))) if ratings else DEFAULT_ELO

    records = []
    for team, grp in long.groupby("time"):
        recent = grp.tail(n_games)
        if recent.empty:
            continue

        n = len(recent)
        recency = np.linspace(0.5, 1.0, n)
        comp_w = recent["competicao"].map(COMPETITION_FORM_WEIGHT).fillna(1.0).to_numpy()
        weights = recency * comp_w
        weights = weights / weights.sum()

        cm = float(np.dot(
            recent["escanteios_marcados"].fillna(league_avg_corners).to_numpy(),
            weights,
        ))
        cs = float(np.dot(
            recent["escanteios_sofridos"].fillna(league_avg_corners).to_numpy(),
            weights,
        ))

        mean_opp_elo = float(recent["elo_adversario"].mean())
        opp_factor = 1.0 + (mean_opp_elo - avg_elo) / 400.0
        opp_factor = float(np.clip(opp_factor, 0.7, 1.3))

        taxa_ataque = (cm / league_avg_corners) * opp_factor
        taxa_defesa = (cs / league_avg_corners) / opp_factor

        records.append({
            "time": team,
            "taxa_escanteio_ataque": max(0.3, taxa_ataque),
            "taxa_escanteio_defesa": max(0.3, taxa_defesa),
            "media_escanteios_marcados": cm,
            "media_escanteios_sofridos": cs,
        })

    return pd.DataFrame(records).set_index("time")


def create_recent_form_features(df: pd.DataFrame, n_games: int = 10) -> pd.DataFrame:
    """Wrapper compatível com a assinatura pedida no projeto.

    Calcula o Elo a partir do histórico e devolve o resumo de forma por time.
    """
    ratings = build_elo_history(df)
    enriched = create_elo_diff(df, ratings)
    return build_team_strengths(enriched, ratings, n_games=n_games)


def create_goal_strength_features(df: pd.DataFrame, n_games: int = 10) -> pd.DataFrame:
    """Alias semântico: força ofensiva/defensiva por seleção."""
    return create_recent_form_features(df, n_games=n_games)


def create_match_features(
    matches: pd.DataFrame,
    ratings: Dict[str, float],
    n_games: int = 10,
) -> pd.DataFrame:
    """Pipeline completo de features para um conjunto de partidas.

    Aplica, em sequência: diferença de Elo e variáveis de contexto.
    Usado tanto no backtest quanto na preparação de jogos futuros.
    """
    out = create_elo_diff(matches, ratings)
    out = create_context_features(out)
    return out
