"""
data_fetcher.py
===============

Carregamento e normalização de dados reais de partidas internacionais.

Fonte primária recomendada:
    Kaggle dataset — "International football results from 1872 to 2023"
    Autor: martj42
    URL: https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017
    Download: kaggle datasets download martj42/international-football-results-from-1872-to-2017

O arquivo results.csv do dataset tem as colunas:
    date, home_team, away_team, home_score, away_score, tournament, city, country, neutral

Depois de salvo em ``data/raw/results.csv``, rode:

    from data_fetcher import load_kaggle_results
    matches = load_kaggle_results("data/raw/results.csv")

Fonte secundária (odds históricas):
    football-data.co.uk — CSVs gratuitos por temporada/torneio.
    Colunas relevantes: B365H, B365D, B365A (odds bet365 decimais).
"""

from __future__ import annotations

import re
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Mapeamento de torneio (Kaggle) → categoria do pipeline
# ---------------------------------------------------------------------------
TOURNAMENT_MAP: dict[str, str] = {
    # Copas do Mundo
    "FIFA World Cup": "World Cup",
    "FIFA World Cup qualification": "Qualifiers",
    "CONMEBOL World Cup qualification": "Qualifiers",
    "CONCACAF World Cup qualification": "Qualifiers",
    "UEFA World Cup qualification": "Qualifiers",
    "CAF World Cup qualification": "Qualifiers",
    "AFC World Cup qualification": "Qualifiers",
    "OFC World Cup qualification": "Qualifiers",
    # Continentais
    "UEFA Euro": "Continental",
    "UEFA European Championship": "Continental",
    "Copa América": "Continental",
    "Africa Cup of Nations": "Continental",
    "AFC Asian Cup": "Continental",
    "CONCACAF Gold Cup": "Continental",
    "OFC Nations Cup": "Continental",
    # Eliminatórias continentais
    "UEFA Euro qualification": "Qualifiers",
    "Copa América qualification": "Qualifiers",
    "Africa Cup of Nations qualification": "Qualifiers",
    "AFC Asian Cup qualification": "Qualifiers",
    "CONCACAF Gold Cup qualification": "Qualifiers",
    # Nations League
    "UEFA Nations League": "Nations League",
    "CONMEBOL-UEFA Cup of Champions": "Nations League",
    # Amistosos
    "Friendly": "Friendly",
    "Kirin Cup": "Friendly",
}

# Padrões regex para capturar variantes não listadas explicitamente
_QUALIFIERS_PATTERNS = re.compile(
    r"(World Cup|Championship|Nations Cup|Gold Cup)\s*qualification",
    re.IGNORECASE,
)
_CONTINENTAL_PATTERNS = re.compile(
    r"(UEFA Euro|Copa Am[eé]rica|Africa Cup|Asian Cup|Gold Cup|Nations Cup)",
    re.IGNORECASE,
)
_NATIONS_LEAGUE_PATTERNS = re.compile(r"Nations League", re.IGNORECASE)

# Seleções confirmadas da Copa 2026 — 48 times, 12 grupos (A-L)
# Nomes no formato do dataset Kaggle (inglês).
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

# Set plano para filtros (filter_copa_teams, etc.)
COPA_2026_TEAMS = {t for teams in COPA_2026_GROUPS.values() for t in teams}


def normalize_competition(tournament: str) -> str:
    """Mapeia um nome de torneio do Kaggle para a categoria do pipeline.

    Fallback: qualquer torneio não reconhecido → "Friendly".
    """
    direct = TOURNAMENT_MAP.get(tournament)
    if direct:
        return direct
    if _QUALIFIERS_PATTERNS.search(tournament):
        return "Qualifiers"
    if _NATIONS_LEAGUE_PATTERNS.search(tournament):
        return "Nations League"
    if _CONTINENTAL_PATTERNS.search(tournament):
        return "Continental"
    return "Friendly"


def load_kaggle_results(
    path: str,
    min_date: Optional[str] = None,
    max_date: Optional[str] = None,
    copa_teams_only: bool = False,
) -> pd.DataFrame:
    """Lê o CSV do Kaggle e normaliza para o formato do pipeline.

    Parameters
    ----------
    path:
        Caminho para ``results.csv`` do dataset Kaggle.
    min_date:
        Filtro de data mínima (inclusive), ex.: "2000-01-01".
        None = sem filtro.
    max_date:
        Filtro de data máxima (inclusive), ex.: "2025-12-31".
    copa_teams_only:
        Se True, mantém apenas partidas onde ambas as seleções estão na
        lista ``COPA_2026_TEAMS``.

    Returns
    -------
    pandas.DataFrame
        Colunas no formato esperado pelo pipeline:
        data_jogo, time_a, time_b, gols_time_a, gols_time_b,
        competicao, fase, mando_neutro.
    """
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.rename(
        columns={
            "date": "data_jogo",
            "home_team": "time_a",
            "away_team": "time_b",
            "home_score": "gols_time_a",
            "away_score": "gols_time_b",
        }
    )

    # Remove partidas sem placar (dados incompletos)
    df = df.dropna(subset=["gols_time_a", "gols_time_b"])
    df["gols_time_a"] = df["gols_time_a"].astype(int)
    df["gols_time_b"] = df["gols_time_b"].astype(int)

    # Normaliza torneio → categoria + fase
    df["competicao"] = df["tournament"].apply(normalize_competition)
    df["fase"] = df["tournament"].apply(_infer_phase)

    # Campo de mando: Kaggle usa "neutral" (True/False)
    if "neutral" in df.columns:
        df["mando_neutro"] = df["neutral"].astype(int)
    else:
        df["mando_neutro"] = 0

    # Filtros opcionais
    if min_date:
        df = df[df["data_jogo"] >= pd.Timestamp(min_date)]
    if max_date:
        df = df[df["data_jogo"] <= pd.Timestamp(max_date)]
    if copa_teams_only:
        mask = df["time_a"].isin(COPA_2026_TEAMS) & df["time_b"].isin(COPA_2026_TEAMS)
        df = df[mask]

    # Colunas finais na ordem do pipeline
    cols = ["data_jogo", "time_a", "time_b", "gols_time_a", "gols_time_b",
            "competicao", "fase", "mando_neutro"]
    return df[cols].sort_values("data_jogo").reset_index(drop=True)


def load_odds_footballdata(path: str) -> pd.DataFrame:
    """Lê um CSV do football-data.co.uk e normaliza para o formato do pipeline.

    O football-data.co.uk publica arquivos por temporada com colunas como:
        HomeTeam, AwayTeam, B365H, B365D, B365A (odds bet365 decimais).

    Returns
    -------
    pandas.DataFrame
        Colunas: time_a, time_b, odd_time_a, odd_empate, odd_time_b.
    """
    df = pd.read_csv(path)
    # Tenta bet365; fallback para Pinnacle ou Média mercado
    for home_col, draw_col, away_col in [
        ("B365H", "B365D", "B365A"),
        ("PSH", "PSD", "PSA"),
        ("AvgH", "AvgD", "AvgA"),
        ("MaxH", "MaxD", "MaxA"),
    ]:
        if all(c in df.columns for c in [home_col, draw_col, away_col]):
            break
    else:
        raise ValueError(
            "Nenhuma coluna de odds reconhecida encontrada. "
            "Esperado: B365H/D/A, PSH/D/A, AvgH/D/A ou MaxH/D/A."
        )

    team_col_a = "HomeTeam" if "HomeTeam" in df.columns else "home_team"
    team_col_b = "AwayTeam" if "AwayTeam" in df.columns else "away_team"

    result = pd.DataFrame({
        "time_a": df[team_col_a],
        "time_b": df[team_col_b],
        "odd_time_a": pd.to_numeric(df[home_col], errors="coerce"),
        "odd_empate": pd.to_numeric(df[draw_col], errors="coerce"),
        "odd_time_b": pd.to_numeric(df[away_col], errors="coerce"),
    })
    return result.dropna().reset_index(drop=True)


# Anfitriões da Copa 2026 — jogam em casa na fase de grupos (campo NÃO neutro).
HOSTS_2026 = {"United States", "Mexico", "Canada"}


def generate_real_fixtures(phase: str = "Grupo") -> pd.DataFrame:
    """Gera um confronto de abertura por grupo (12 jogos) com nomes em inglês.

    Usa os grupos reais da Copa 2026 (COPA_2026_GROUPS): cada fixture é o
    primeiro confronto do grupo (posição 0 vs 1). Anfitriões mandam em casa.
    """
    base_date = pd.Timestamp("2026-06-11")
    rows = []
    for i, (group, teams) in enumerate(COPA_2026_GROUPS.items()):
        a, b = teams[0], teams[1]
        neutro = 0 if (a in HOSTS_2026 and phase == "Grupo") else 1
        rows.append({
            "data_jogo": base_date + pd.Timedelta(days=i),
            "time_a": a,
            "time_b": b,
            "competicao": "World Cup",
            "fase": phase,
            "grupo": group,
            "mando_neutro": neutro,
        })
    return pd.DataFrame(rows)


def filter_copa_teams(
    df: pd.DataFrame,
    teams: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Filtra apenas partidas onde ambas as seleções estão no conjunto alvo.

    Parameters
    ----------
    df:
        DataFrame com colunas ``time_a`` e ``time_b``.
    teams:
        Lista de seleções a manter. Padrão: ``COPA_2026_TEAMS``.
    """
    pool = set(teams) if teams is not None else COPA_2026_TEAMS
    mask = df["time_a"].isin(pool) & df["time_b"].isin(pool)
    return df[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------
def _infer_phase(tournament: str) -> str:
    """Infere a fase (Grupo/Oitavas/…) a partir do nome do torneio.

    O Kaggle não separa fases dentro do torneio; usamos o tipo de competição
    como proxy. Para o pipeline V1 isso é suficiente — na V3 pode ser
    enriquecido com dados de fase separados.
    """
    comp = normalize_competition(tournament)
    if comp == "World Cup":
        return "Grupo"  # fase real desconhecida no CSV; refinável na V3
    if comp == "Continental":
        return "Grupo"
    if comp == "Qualifiers":
        return "Eliminatorias"
    if comp == "Nations League":
        return "Grupo"
    return "Amistoso"


# ---------------------------------------------------------------------------
# Execução direta: diagnóstico do CSV
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/results.csv"
    try:
        df = load_kaggle_results(path)
    except FileNotFoundError:
        print(f"Arquivo não encontrado: {path}")
        print()
        print("Para obter os dados:")
        print("  pip install kaggle")
        print("  kaggle datasets download martj42/international-football-results-from-1872-to-2017")
        print("  unzip international-football-results-from-1872-to-2017.zip -d data/raw/")
        raise SystemExit(1)

    print(f"Partidas carregadas    : {len(df):,}")
    print(f"Período                : {df['data_jogo'].min().date()} → {df['data_jogo'].max().date()}")
    print(f"Gols/time (média)      : {df[['gols_time_a', 'gols_time_b']].values.mean():.3f}")
    print(f"Distribuição competição:\n{df['competicao'].value_counts().to_string()}")
    print()

    df_copa = filter_copa_teams(df)
    print(f"Partidas (só Copa 2026 teams): {len(df_copa):,}")
