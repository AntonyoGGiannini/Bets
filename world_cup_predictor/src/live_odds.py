"""
live_odds.py
============

Busca e agregação de odds ao vivo das casas de aposta via The Odds API
(https://the-odds-api.com).

Fluxo:
1. ``fetch_world_cup_odds`` consulta o endpoint de odds da Copa do Mundo
   (mercados ``h2h`` = 1X2 e ``totals`` = Over/Under) em formato decimal.
2. ``aggregate_event_odds`` normaliza os nomes dos times para os usados pelo
   modelo, calcula a **média das odds decimais por resultado entre as casas**
   (além do nº de casas e a melhor/pior odd) e devolve um DataFrame resumo
   por jogo + um detalhe por (jogo, casa).
3. O app compara as médias com as probabilidades do modelo usando os helpers
   de ``odds_analysis`` (1X2) e ``remove_margin_two_way`` daqui (O/U 2.5).

Escolha de agregação: média simples das odds decimais (e não das
probabilidades implícitas) — suficiente para sinalizar consenso de mercado;
a melhor odd (máxima) também é reportada por ser a relevante para o apostador.
"""

from __future__ import annotations

import difflib
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

API_BASE = "https://api.the-odds-api.com/v4"
SPORT_KEY = "soccer_fifa_world_cup"

# Fim da fase de grupos da Copa 2026 (último jogo em 27/06/2026). Jogos com
# início a partir desta data são tratados como mata-mata. Premissa
# simplificadora — ajustar se o calendário oficial mudar.
KNOCKOUT_START = date(2026, 6, 28)

# Nomes usados pela The Odds API → nomes usados pelo modelo (COPA_2026_TEAMS).
# Alimentado conforme aparecem divergências (ver expander de "não mapeados").
API_TO_MODEL_TEAM: Dict[str, str] = {
    "USA": "United States",
    "United States of America": "United States",
    "Korea Republic": "South Korea",
    "Republic of Korea": "South Korea",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Czechia": "Czech Republic",
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "Cabo Verde": "Cape Verde",
    "Cape Verde Islands": "Cape Verde",
    "Democratic Republic of the Congo": "DR Congo",
    "Congo DR": "DR Congo",
    "DR Congo": "DR Congo",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "IR Iran": "Iran",
    "Iran Islamic Republic of": "Iran",
    "Curacao": "Curaçao",
    "Korea DPR": "North Korea",
    "China PR": "China",
}


def match_team(api_name: str, valid_teams: List[str]) -> Optional[str]:
    """Resolve o nome de um time vindo da API para o nome usado pelo modelo.

    Ordem: (1) match exato; (2) dicionário de apelidos; (3) fuzzy match
    conservador (``difflib``, cutoff 0.85). Retorna ``None`` se não resolver.
    """
    if api_name in valid_teams:
        return api_name
    mapped = API_TO_MODEL_TEAM.get(api_name)
    if mapped and mapped in valid_teams:
        return mapped
    close = difflib.get_close_matches(api_name, valid_teams, n=1, cutoff=0.85)
    return close[0] if close else None


def fetch_world_cup_odds(
    api_key: str,
    regions: str = "eu,uk,us",
    markets: str = "h2h,totals",
) -> Tuple[List[dict], Dict[str, str]]:
    """Busca as odds da Copa na The Odds API.

    Returns
    -------
    (events, quota)
        ``events``: lista de eventos no formato da API (cada um com
        ``home_team``, ``away_team``, ``commence_time`` e ``bookmakers``).
        ``quota``: headers de cota (``x-requests-remaining`` etc.).

    Levanta ``requests.HTTPError``/``RequestException`` — o tratamento
    amigável fica no app.
    """
    resp = requests.get(
        f"{API_BASE}/sports/{SPORT_KEY}/odds",
        params={
            "apiKey": api_key,
            "regions": regions,
            "markets": markets,
            "oddsFormat": "decimal",
        },
        timeout=15,
    )
    resp.raise_for_status()
    quota = {
        "remaining": resp.headers.get("x-requests-remaining", "?"),
        "used": resp.headers.get("x-requests-used", "?"),
    }
    return resp.json(), quota


def fetch_available_sports(api_key: str) -> List[dict]:
    """Lista os sport keys disponíveis (custa 0 créditos) — diagnóstico."""
    resp = requests.get(
        f"{API_BASE}/sports", params={"apiKey": api_key, "all": "true"}, timeout=15
    )
    resp.raise_for_status()
    return resp.json()


def remove_margin_two_way(odd_over: float, odd_under: float) -> Dict[str, float]:
    """Remove a margem da casa de um mercado de 2 resultados (Over/Under).

    Espelha ``odds_analysis.remove_bookmaker_margin``, que é de 3 vias.
    """
    if not odd_over or not odd_under or odd_over <= 1.0 or odd_under <= 1.0:
        return {"prob_over": np.nan, "prob_under": np.nan, "overround": np.nan}
    p_over, p_under = 1.0 / odd_over, 1.0 / odd_under
    total = p_over + p_under
    return {
        "prob_over": p_over / total,
        "prob_under": p_under / total,
        "overround": total - 1.0,
    }


def aggregate_event_odds(
    events: List[dict],
    valid_teams: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[dict]]:
    """Agrega as odds de cada evento sobre todas as casas.

    Parameters
    ----------
    events:
        Resposta JSON da The Odds API (lista de eventos).
    valid_teams:
        Times reconhecidos pelo modelo (para mapear os nomes da API).

    Returns
    -------
    (resumo, detalhe, unmatched)
        ``resumo``: 1 linha por jogo — médias/melhor/pior odd 1X2 e O/U 2.5,
        nº de casas por mercado e ``commence_time`` (UTC).
        ``detalhe``: 1 linha por (jogo, casa) com as odds individuais.
        ``unmatched``: eventos descartados por time não mapeado (com os
        nomes brutos da API, para alimentar ``API_TO_MODEL_TEAM``).
    """
    resumo_rows: List[dict] = []
    detalhe_rows: List[dict] = []
    unmatched: List[dict] = []

    for ev in events:
        home = match_team(ev.get("home_team", ""), valid_teams)
        away = match_team(ev.get("away_team", ""), valid_teams)
        if home is None or away is None:
            unmatched.append({
                "home_team": ev.get("home_team"),
                "away_team": ev.get("away_team"),
                "commence_time": ev.get("commence_time"),
            })
            continue

        commence = pd.to_datetime(ev.get("commence_time"), utc=True)
        jogo = f"{home} × {away}"

        # Odds por casa: {casa: {"1":, "X":, "2":, "over25":, "under25":}}
        for bk in ev.get("bookmakers", []):
            row = {
                "jogo": jogo,
                "time_a": home,
                "time_b": away,
                "commence_time": commence,
                "casa": bk.get("title", bk.get("key", "?")),
                "odd_1": np.nan, "odd_x": np.nan, "odd_2": np.nan,
                "odd_over25": np.nan, "odd_under25": np.nan,
            }
            for mkt in bk.get("markets", []):
                if mkt.get("key") == "h2h":
                    for out in mkt.get("outcomes", []):
                        nome = match_team(out.get("name", ""), valid_teams)
                        if out.get("name") == "Draw":
                            row["odd_x"] = out.get("price")
                        elif nome == home:
                            row["odd_1"] = out.get("price")
                        elif nome == away:
                            row["odd_2"] = out.get("price")
                elif mkt.get("key") == "totals":
                    for out in mkt.get("outcomes", []):
                        point = out.get("point")
                        if point is None or abs(float(point) - 2.5) > 1e-9:
                            continue  # só a linha 2.5
                        if out.get("name") == "Over":
                            row["odd_over25"] = out.get("price")
                        elif out.get("name") == "Under":
                            row["odd_under25"] = out.get("price")
            detalhe_rows.append(row)

        ev_det = pd.DataFrame([r for r in detalhe_rows if r["jogo"] == jogo])
        h2h = ev_det.dropna(subset=["odd_1", "odd_x", "odd_2"])
        ou = ev_det.dropna(subset=["odd_over25", "odd_under25"])

        resumo_rows.append({
            "time_a": home,
            "time_b": away,
            "jogo": jogo,
            "commence_time": commence,
            "n_casas_1x2": len(h2h),
            "odd_1_media": h2h["odd_1"].mean() if len(h2h) else np.nan,
            "odd_x_media": h2h["odd_x"].mean() if len(h2h) else np.nan,
            "odd_2_media": h2h["odd_2"].mean() if len(h2h) else np.nan,
            "odd_1_max": h2h["odd_1"].max() if len(h2h) else np.nan,
            "odd_x_max": h2h["odd_x"].max() if len(h2h) else np.nan,
            "odd_2_max": h2h["odd_2"].max() if len(h2h) else np.nan,
            "odd_1_min": h2h["odd_1"].min() if len(h2h) else np.nan,
            "odd_x_min": h2h["odd_x"].min() if len(h2h) else np.nan,
            "odd_2_min": h2h["odd_2"].min() if len(h2h) else np.nan,
            "n_casas_ou": len(ou),
            "odd_over25_media": ou["odd_over25"].mean() if len(ou) else np.nan,
            "odd_under25_media": ou["odd_under25"].mean() if len(ou) else np.nan,
            "odd_over25_max": ou["odd_over25"].max() if len(ou) else np.nan,
            "odd_under25_max": ou["odd_under25"].max() if len(ou) else np.nan,
        })

    resumo = pd.DataFrame(resumo_rows)
    detalhe = pd.DataFrame(detalhe_rows)
    if not resumo.empty:
        resumo = resumo.sort_values("commence_time").reset_index(drop=True)
    return resumo, detalhe, unmatched


# ---------------------------------------------------------------------------
# Resposta de exemplo (formato real da API) — modo demonstração e testes.
# ---------------------------------------------------------------------------
SAMPLE_RESPONSE: List[dict] = [
    {
        "id": "demo-mex-rsa",
        "sport_key": SPORT_KEY,
        "commence_time": "2026-06-11T19:00:00Z",
        "home_team": "Mexico",
        "away_team": "South Africa",
        "bookmakers": [
            {
                "key": "bet365", "title": "Bet365",
                "markets": [
                    {"key": "h2h", "outcomes": [
                        {"name": "Mexico", "price": 1.65},
                        {"name": "Draw", "price": 3.80},
                        {"name": "South Africa", "price": 5.50},
                    ]},
                    {"key": "totals", "outcomes": [
                        {"name": "Over", "point": 2.5, "price": 2.10},
                        {"name": "Under", "point": 2.5, "price": 1.75},
                    ]},
                ],
            },
            {
                "key": "pinnacle", "title": "Pinnacle",
                "markets": [
                    {"key": "h2h", "outcomes": [
                        {"name": "Mexico", "price": 1.70},
                        {"name": "Draw", "price": 3.90},
                        {"name": "South Africa", "price": 5.80},
                    ]},
                    {"key": "totals", "outcomes": [
                        {"name": "Over", "point": 2.5, "price": 2.14},
                        {"name": "Under", "point": 2.5, "price": 1.79},
                    ]},
                ],
            },
            {
                "key": "betfair", "title": "Betfair",
                "markets": [
                    {"key": "h2h", "outcomes": [
                        {"name": "Mexico", "price": 1.68},
                        {"name": "Draw", "price": 3.75},
                        {"name": "South Africa", "price": 5.60},
                    ]},
                ],
            },
        ],
    },
    {
        "id": "demo-usa-par",
        "sport_key": SPORT_KEY,
        "commence_time": "2026-06-12T22:00:00Z",
        "home_team": "USA",
        "away_team": "Paraguay",
        "bookmakers": [
            {
                "key": "bet365", "title": "Bet365",
                "markets": [
                    {"key": "h2h", "outcomes": [
                        {"name": "USA", "price": 1.80},
                        {"name": "Draw", "price": 3.50},
                        {"name": "Paraguay", "price": 4.60},
                    ]},
                    {"key": "totals", "outcomes": [
                        {"name": "Over", "point": 2.5, "price": 2.25},
                        {"name": "Under", "point": 2.5, "price": 1.65},
                    ]},
                ],
            },
            {
                "key": "pinnacle", "title": "Pinnacle",
                "markets": [
                    {"key": "h2h", "outcomes": [
                        {"name": "USA", "price": 1.85},
                        {"name": "Draw", "price": 3.60},
                        {"name": "Paraguay", "price": 4.80},
                    ]},
                    {"key": "totals", "outcomes": [
                        {"name": "Over", "point": 2.5, "price": 2.30},
                        {"name": "Under", "point": 2.5, "price": 1.68},
                    ]},
                ],
            },
        ],
    },
    {
        "id": "demo-bra-mar",
        "sport_key": SPORT_KEY,
        "commence_time": "2026-06-13T16:00:00Z",
        "home_team": "Brazil",
        "away_team": "Morocco",
        "bookmakers": [
            {
                "key": "bet365", "title": "Bet365",
                "markets": [
                    {"key": "h2h", "outcomes": [
                        {"name": "Brazil", "price": 1.95},
                        {"name": "Draw", "price": 3.40},
                        {"name": "Morocco", "price": 4.00},
                    ]},
                    {"key": "totals", "outcomes": [
                        {"name": "Over", "point": 2.5, "price": 2.40},
                        {"name": "Under", "point": 2.5, "price": 1.57},
                    ]},
                ],
            },
            {
                "key": "betano", "title": "Betano",
                "markets": [
                    {"key": "h2h", "outcomes": [
                        {"name": "Brazil", "price": 2.00},
                        {"name": "Draw", "price": 3.45},
                        {"name": "Morocco", "price": 3.90},
                    ]},
                    {"key": "totals", "outcomes": [
                        {"name": "Over", "point": 2.5, "price": 2.35},
                        {"name": "Under", "point": 2.5, "price": 1.60},
                    ]},
                ],
            },
            {
                "key": "pinnacle", "title": "Pinnacle",
                "markets": [
                    {"key": "h2h", "outcomes": [
                        {"name": "Brazil", "price": 1.98},
                        {"name": "Draw", "price": 3.50},
                        {"name": "Morocco", "price": 4.10},
                    ]},
                    {"key": "totals", "outcomes": [
                        {"name": "Over", "point": 2.5, "price": 2.38},
                        {"name": "Under", "point": 2.5, "price": 1.59},
                    ]},
                ],
            },
        ],
    },
]
