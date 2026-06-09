"""
data_loader.py
==============

Carregamento de dados e geração de uma base fictícia (mock) para a V1.

A V1 NÃO depende de nenhuma API externa. Quando não houver arquivos reais,
``generate_mock_matches`` cria um histórico sintético, porém realista, de
partidas internacionais. Cada seleção tem uma "força latente" e os gols são
sorteados de uma Poisson condicionada a essa força — exatamente o tipo de
estrutura que o modelo tenta recuperar depois.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Catálogo de seleções com uma "força verdadeira" (oculta) usada apenas para
# gerar os dados mock. O modelo não enxerga esses números; ele os reestima
# via Elo + forma recente.
# ---------------------------------------------------------------------------
MOCK_TEAMS: Dict[str, float] = {
    "Brasil": 2.00,
    "Argentina": 1.95,
    "França": 1.95,
    "Inglaterra": 1.85,
    "Espanha": 1.80,
    "Alemanha": 1.75,
    "Portugal": 1.78,
    "Holanda": 1.70,
    "Itália": 1.65,
    "Bélgica": 1.62,
    "Croácia": 1.55,
    "Uruguai": 1.55,
    "Marrocos": 1.45,
    "México": 1.40,
    "Estados Unidos": 1.38,
    "Japão": 1.40,
    "Coreia do Sul": 1.30,
    "Senegal": 1.35,
    "Equador": 1.25,
    "Austrália": 1.20,
    "Polônia": 1.30,
    "Sérvia": 1.28,
    "Suíça": 1.45,
    "Dinamarca": 1.48,
    "Canadá": 1.18,
    "Gana": 1.15,
    "Camarões": 1.20,
    "Arábia Saudita": 1.05,
    "Catar": 0.95,
    "Costa Rica": 1.00,
    "Tunísia": 1.10,
    "Nigéria": 1.30,
}

COMPETITIONS = [
    ("World Cup", 0.10),
    ("Continental", 0.15),
    ("Qualifiers", 0.30),
    ("Nations League", 0.15),
    ("Friendly", 0.30),
]

PHASES_BY_COMPETITION = {
    "World Cup": ["Grupo", "Oitavas", "Quartas", "Semifinal", "Final"],
    "Continental": ["Grupo", "Quartas", "Semifinal", "Final"],
    "Qualifiers": ["Eliminatorias"],
    "Nations League": ["Grupo", "Final"],
    "Friendly": ["Amistoso"],
}


def generate_mock_matches(
    n_matches: int = 600,
    start_date: str = "2022-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    """Gera um histórico fictício de partidas internacionais.

    O placar de cada jogo é sorteado de uma Poisson cuja média depende da
    força latente do time e da força defensiva do adversário, mais um pequeno
    ajuste de mando de campo.

    Returns
    -------
    pandas.DataFrame
        Colunas no formato esperado pelo restante do pipeline.
    """
    rng = np.random.default_rng(seed)
    teams = list(MOCK_TEAMS.keys())
    base_date = pd.Timestamp(start_date)

    comp_names = [c[0] for c in COMPETITIONS]
    comp_probs = np.array([c[1] for c in COMPETITIONS])
    comp_probs = comp_probs / comp_probs.sum()

    rows: List[dict] = []
    for i in range(n_matches):
        team_a, team_b = rng.choice(teams, size=2, replace=False)

        competition = rng.choice(comp_names, p=comp_probs)
        phase = rng.choice(PHASES_BY_COMPETITION[competition])

        # Força latente -> média de gols. Defesa do adversário reduz a média.
        strength_a = MOCK_TEAMS[team_a]
        strength_b = MOCK_TEAMS[team_b]

        # Mando neutro em torneios; amistosos/eliminatórias podem ter mando.
        neutral = competition in ("World Cup", "Continental")
        home_adv = 0.0 if neutral else rng.choice([0.0, 0.3])

        # BASE calibrado para que duas seleções medianas marquem ~1.35 gols,
        # próximo da média real de partidas internacionais.
        base = 2.25
        lambda_a = max(0.15, strength_a * (base / (1.0 + strength_b)) + home_adv)
        lambda_b = max(0.15, strength_b * (base / (1.0 + strength_a)))

        goals_a = int(rng.poisson(lambda_a))
        goals_b = int(rng.poisson(lambda_b))

        # Escanteios mock: times mais ofensivos geram mais corners; defesas
        # fortes concedem menos. Baseline ~5.15 por time (~10.3 total).
        corner_base = 5.15
        c_att_a = 1.0 + 0.15 * (strength_a - 1.3)
        c_def_b = max(0.5, 1.0 - 0.10 * (strength_b - 1.3))
        c_att_b = 1.0 + 0.15 * (strength_b - 1.3)
        c_def_a = max(0.5, 1.0 - 0.10 * (strength_a - 1.3))
        corners_a = int(rng.poisson(max(1.5, corner_base * c_att_a * c_def_b)))
        corners_b = int(rng.poisson(max(1.5, corner_base * c_att_b * c_def_a)))

        match_date = base_date + pd.Timedelta(days=int(i * 2 + rng.integers(0, 2)))

        rows.append(
            {
                "data_jogo": match_date,
                "time_a": team_a,
                "time_b": team_b,
                "gols_time_a": goals_a,
                "gols_time_b": goals_b,
                "escanteios_time_a": corners_a,
                "escanteios_time_b": corners_b,
                "competicao": competition,
                "fase": phase,
                "mando_neutro": int(neutral),
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("data_jogo").reset_index(drop=True)
    return df


def generate_mock_fixtures(
    teams: Optional[List[str]] = None,
    seed: int = 7,
) -> pd.DataFrame:
    """Gera uma lista de confrontos futuros (a prever) para a Copa de 2026.

    A Copa é em campo neutro (``mando_neutro=1``), exceto os anfitriões
    (EUA, México, Canadá), que mandam os jogos de grupo em casa.

    Returns
    -------
    pandas.DataFrame
        Colunas: ``data_jogo``, ``time_a``, ``time_b``, ``competicao``,
        ``fase``, ``mando_neutro``.
    """
    rng = np.random.default_rng(seed)
    pool = teams or list(MOCK_TEAMS.keys())
    hosts = {"Estados Unidos", "México", "Canadá"}

    fixtures = [
        ("Brasil", "Alemanha", "Grupo"),
        ("Argentina", "França", "Grupo"),
        ("Inglaterra", "Espanha", "Quartas"),
        ("Portugal", "Holanda", "Oitavas"),
        ("Marrocos", "Bélgica", "Grupo"),
        ("México", "Japão", "Grupo"),
        ("Croácia", "Uruguai", "Oitavas"),
        ("Estados Unidos", "Senegal", "Grupo"),
    ]

    base_date = pd.Timestamp("2026-06-11")
    rows = []
    for i, (a, b, phase) in enumerate(fixtures):
        if a not in pool or b not in pool:
            a, b = rng.choice(pool, size=2, replace=False)
        neutro = 0 if (a in hosts and phase == "Grupo") else 1
        rows.append(
            {
                "data_jogo": base_date + pd.Timedelta(days=i),
                "time_a": a,
                "time_b": b,
                "competicao": "World Cup",
                "fase": phase,
                "mando_neutro": neutro,
            }
        )
    return pd.DataFrame(rows)


def generate_mock_odds(fixtures: pd.DataFrame, seed: int = 11) -> pd.DataFrame:
    """Gera odds decimais fictícias de mercado (1X2) para os confrontos.

    As odds incluem uma margem da casa (overround) de ~5%.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for row in fixtures.itertuples(index=False):
        # Probabilidades "verdadeiras" aproximadas pela força latente.
        sa = MOCK_TEAMS.get(row.time_a, 1.3)
        sb = MOCK_TEAMS.get(row.time_b, 1.3)
        p_a = sa / (sa + sb)
        p_a = 0.5 + (p_a - 0.5) * 0.8  # achata um pouco
        p_draw = 0.27
        p_b = max(0.05, 1.0 - p_a - p_draw)

        # Normaliza e aplica margem. odd = 1 / (p * margin) faz as
        # probabilidades implícitas (1/odd) somarem ~margin (overround > 1),
        # reproduzindo a margem da casa de ~5%.
        total = p_a + p_draw + p_b
        p_a, p_draw, p_b = p_a / total, p_draw / total, p_b / total
        margin = 1.05
        odd_a = round(1.0 / (max(p_a, 1e-6) * margin), 2)
        odd_draw = round(1.0 / (max(p_draw, 1e-6) * margin), 2)
        odd_b = round(1.0 / (max(p_b, 1e-6) * margin), 2)

        rows.append(
            {
                "time_a": row.time_a,
                "time_b": row.time_b,
                "odd_time_a": odd_a,
                "odd_empate": odd_draw,
                "odd_time_b": odd_b,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Loaders de arquivos reais. Caem para os geradores mock quando o arquivo
# não existe, de modo que o pipeline V1 roda "out of the box".
# ---------------------------------------------------------------------------
def load_matches(path: Optional[str] = None) -> pd.DataFrame:
    """Carrega o histórico de partidas de um CSV ou gera dados mock."""
    if path and os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["data_jogo"])
        return df.sort_values("data_jogo").reset_index(drop=True)
    return generate_mock_matches()


def load_odds(path: Optional[str] = None, fixtures: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Carrega odds de mercado de um CSV ou gera odds mock para os fixtures."""
    if path and os.path.exists(path):
        return pd.read_csv(path)
    if fixtures is None:
        fixtures = generate_mock_fixtures()
    return generate_mock_odds(fixtures)


def load_teams(path: Optional[str] = None) -> pd.DataFrame:
    """Carrega a lista de seleções (nome + rating Elo inicial opcional)."""
    if path and os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame({"time": list(MOCK_TEAMS.keys())})


# ---------------------------------------------------------------------------
# Loaders para fontes de dados reais (wrappers sobre data_fetcher).
# ---------------------------------------------------------------------------
def load_matches_kaggle(
    path: str,
    min_date: Optional[str] = "2000-01-01",
    copa_teams_only: bool = False,
) -> pd.DataFrame:
    """Lê o CSV do Kaggle e normaliza para o formato do pipeline.

    Atalho para ``data_fetcher.load_kaggle_results``.  Para mais opções
    (filtro de data máxima, lista customizada de times) importe
    ``data_fetcher`` diretamente.

    Parameters
    ----------
    path:
        Caminho para ``results.csv`` do dataset Kaggle.
    min_date:
        Filtra partidas a partir desta data (padrão: 2000-01-01 para evitar
        dados muito antigos com qualidade variável).
    copa_teams_only:
        Se True, mantém só partidas entre seleções da Copa 2026.
    """
    from data_fetcher import load_kaggle_results
    return load_kaggle_results(path, min_date=min_date, copa_teams_only=copa_teams_only)


def load_odds_footballdata(path: str) -> pd.DataFrame:
    """Lê um CSV do football-data.co.uk e normaliza para o formato do pipeline.

    Atalho para ``data_fetcher.load_odds_footballdata``.
    Colunas suportadas: B365H/D/A, PSH/D/A, AvgH/D/A ou MaxH/D/A.
    """
    from data_fetcher import load_odds_footballdata as _load
    return _load(path)
