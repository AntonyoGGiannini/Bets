"""
corner_model.py
===============

Modelo de escanteios esperados (mu) por time + mercados de totais.

Segue a mesma arquitetura do goal_model.py: funções puras, constantes
calibradas, distribuição Poisson para gerar probabilidades de mercado
(totais O/U, spread por time, linha asiática de escanteios).

Fórmula conceitual:
    mu_a = taxa_escanteio_ataque_a * taxa_escanteio_defesa_b
           * ajuste_elo * ajuste_contexto * media_liga_escanteios
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import poisson

# Média de escanteios por time em uma partida internacional (~10.3 total / 2).
LEAGUE_AVG_CORNERS = 5.15

# Escala Elo para escanteios — ~55% do efeito sobre gols (ELO_SCALE=0.0011 em
# goal_model). Domínio territorial correlaciona com Elo, mas menos do que gols.
# e^(0.0006 × 400) ≈ 1.27 de diferença por 400 pts de Elo.
CORNER_ELO_SCALE = 0.0006

# Shrinkage da forma recente em direção a 1.0 (vs 0.35 para gols).
# Escanteios têm variância maior que gols (~20-30% CV vs ~15%), então
# a forma recente tem menos sinal e puxamos mais para a média.
CORNER_FORM_SHRINKAGE = 0.25

# Mata-mata: ~9% menos escanteios — times adotam postura mais defensiva e
# buscam menos o ataque, gerando menos corners.
KNOCKOUT_CORNER_DAMPING = 0.91

# Vantagem de campo em escanteios: mandante gera ~6% mais corners por
# territorialidade e pressão no estádio próprio.
HOME_ADV_CORNERS_ATTACK = 1.06
HOME_ADV_CORNERS_DEFENSE = 0.97

# Limites de sanidade para mu (escanteios esperados por time por jogo).
CORNER_MU_MIN = 1.5
CORNER_MU_MAX = 12.0

# Máximo de escanteios por time na matriz de probabilidades.
# P(X >= 18 | mu=6) ≈ 0.001 — cobre >99.9% da massa de probabilidade.
MAX_CORNERS = 18


def _corner_shrink(ratio: float, amount: float = CORNER_FORM_SHRINKAGE) -> float:
    """Puxa uma razão de taxa de escanteios (centrada em 1.0) em direção à média."""
    return 1.0 + amount * (ratio - 1.0)


def _corner_elo_adjustment(diferenca_elo: float) -> float:
    """Fator multiplicativo derivado da diferença de Elo para escanteios."""
    return float(np.exp(CORNER_ELO_SCALE * diferenca_elo))


def estimate_corner_mus(
    match_features: Dict[str, float],
    league_avg_corners: float = LEAGUE_AVG_CORNERS,
) -> tuple[float, float]:
    """Estima (mu_a, mu_b): escanteios esperados por time na partida.

    Parameters
    ----------
    match_features:
        Dicionário com:
            taxa_escanteio_ataque_a, taxa_escanteio_defesa_a,
            taxa_escanteio_ataque_b, taxa_escanteio_defesa_b
            (todos opcionais — default 1.0, média da liga);
            diferenca_elo (opcional, default 0);
            jogo_eliminatorio (opcional, default 0);
            mando_neutro (opcional, default 1).
    league_avg_corners:
        Baseline de escanteios por time por jogo.

    Returns
    -------
    (mu_a, mu_b) clipados em [CORNER_MU_MIN, CORNER_MU_MAX].
    """
    att_a = _corner_shrink(float(match_features.get("taxa_escanteio_ataque_a", 1.0)))
    def_b = _corner_shrink(float(match_features.get("taxa_escanteio_defesa_b", 1.0)))
    att_b = _corner_shrink(float(match_features.get("taxa_escanteio_ataque_b", 1.0)))
    def_a = _corner_shrink(float(match_features.get("taxa_escanteio_defesa_a", 1.0)))

    elo_diff = float(match_features.get("diferenca_elo", 0.0))
    elo_adj = _corner_elo_adjustment(elo_diff)

    # mu_a: ataque de A vs. permissividade defensiva de B (corners concedidos)
    mu_a = league_avg_corners * att_a * def_b * elo_adj
    mu_b = league_avg_corners * att_b * def_a / elo_adj

    if not match_features.get("mando_neutro", 1):
        mu_a *= HOME_ADV_CORNERS_ATTACK
        mu_b *= HOME_ADV_CORNERS_DEFENSE

    if match_features.get("jogo_eliminatorio", 0):
        mu_a *= KNOCKOUT_CORNER_DAMPING
        mu_b *= KNOCKOUT_CORNER_DAMPING

    mu_a = float(np.clip(mu_a, CORNER_MU_MIN, CORNER_MU_MAX))
    mu_b = float(np.clip(mu_b, CORNER_MU_MIN, CORNER_MU_MAX))
    return mu_a, mu_b


def estimate_corner_mus_for_fixture(
    time_a: str,
    time_b: str,
    strengths: pd.DataFrame,
    diferenca_elo: Optional[float] = None,
    jogo_eliminatorio: int = 0,
    mando_neutro: int = 1,
    league_avg_corners: float = LEAGUE_AVG_CORNERS,
) -> tuple[float, float]:
    """Conveniência: monta features a partir do DataFrame de forças.

    Retrocompatível: se strengths não tiver colunas de escanteio, usa
    taxa=1.0 (média da liga) e aplica apenas o ajuste de Elo.
    """
    from elo_model import DEFAULT_ELO

    if time_a not in strengths.index or time_b not in strengths.index:
        raise KeyError(f"Time sem histórico nos dados: '{time_a}' ou '{time_b}'")

    sa = strengths.loc[time_a]
    sb = strengths.loc[time_b]

    if diferenca_elo is None:
        diferenca_elo = float(sa.get("elo", DEFAULT_ELO)) - float(sb.get("elo", DEFAULT_ELO))

    features: Dict[str, float] = {
        "taxa_escanteio_ataque_a": float(sa.get("taxa_escanteio_ataque", 1.0)),
        "taxa_escanteio_defesa_a": float(sa.get("taxa_escanteio_defesa", 1.0)),
        "taxa_escanteio_ataque_b": float(sb.get("taxa_escanteio_ataque", 1.0)),
        "taxa_escanteio_defesa_b": float(sb.get("taxa_escanteio_defesa", 1.0)),
        "diferenca_elo": diferenca_elo,
        "jogo_eliminatorio": jogo_eliminatorio,
        "mando_neutro": mando_neutro,
    }
    return estimate_corner_mus(features, league_avg_corners)


def corner_total_probabilities(
    mu_a: float,
    mu_b: float,
    max_corners: int = MAX_CORNERS,
) -> Dict[str, float]:
    """Gera mercados de totais de escanteios via Poisson.

    Assume independência entre escanteios dos dois times — sem correção
    Dixon-Coles (não há incentivo análogo a "travar o jogo" para escanteios).
    A soma total segue Poisson(mu_a + mu_b).

    Returns
    -------
    dict com:
        mu_time_a, mu_time_b, mu_total;
        prob_over/under_{8,9,10,11,12}_5 (totais);
        prob_escanteios_time_a/b_over_{4,5}_5 (por time);
        spread_a_vence/empate/b_vence_escanteios;
        linha_asiatica_escanteios (handicap arredondado ao 0.5).
    """
    mu_total = mu_a + mu_b
    result: Dict[str, float] = {
        "mu_time_a": round(mu_a, 3),
        "mu_time_b": round(mu_b, 3),
        "mu_total": round(mu_total, 3),
    }

    # Over/under totais via Poisson(mu_a + mu_b)
    for line in [8.5, 9.5, 10.5, 11.5, 12.5]:
        threshold = int(line + 0.5)  # 8.5 → threshold=9: P(X≥9) = 1 − P(X≤8)
        p_over = float(1.0 - poisson.cdf(threshold - 1, mu_total))
        key = f"{line:.1f}".replace(".", "_")
        result[f"prob_over_{key}"] = round(p_over, 4)
        result[f"prob_under_{key}"] = round(1.0 - p_over, 4)

    # Mercados por time: over 4.5 e over 5.5
    for team_label, mu in [("time_a", mu_a), ("time_b", mu_b)]:
        for line in [4.5, 5.5]:
            threshold = int(line + 0.5)
            p_over = float(1.0 - poisson.cdf(threshold - 1, mu))
            key = f"{line:.1f}".replace(".", "_")
            result[f"prob_escanteios_{team_label}_over_{key}"] = round(p_over, 4)

    # Spread de escanteios: quem tira mais corners (np.outer sobre as PMFs)
    corners_range = np.arange(0, max_corners + 1)
    p_a = poisson.pmf(corners_range, mu_a)
    p_b = poisson.pmf(corners_range, mu_b)
    matrix = np.outer(p_a, p_b)  # P[i, j] = P(A=i) * P(B=j)

    idx_a = np.arange(max_corners + 1)[:, None]
    idx_b = np.arange(max_corners + 1)[None, :]
    result["spread_a_vence_escanteios"] = round(float(matrix[idx_a > idx_b].sum()), 4)
    result["spread_empate_escanteios"] = round(float(matrix[idx_a == idx_b].sum()), 4)
    result["spread_b_vence_escanteios"] = round(float(matrix[idx_a < idx_b].sum()), 4)

    # Linha asiática: diferença esperada arredondada ao 0.5 mais próximo
    result["linha_asiatica_escanteios"] = round((mu_a - mu_b) * 2) / 2.0

    return result
