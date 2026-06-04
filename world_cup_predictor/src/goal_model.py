"""
goal_model.py
=============

Modelo de gols esperados (lambdas) + matriz de placares via Poisson.

Em vez de prever o vencedor diretamente, estimamos a média de gols esperada
de cada time (``lambda_a`` e ``lambda_b``) e deixamos a Poisson cuidar das
probabilidades de cada placar. Esse desenho gera, de graça, probabilidades
para vários mercados (1X2, over/under, ambos marcam, placar exato).

Fórmula conceitual (seção 6.1):

    lambda_a = forca_ofensiva_a * fragilidade_defensiva_b
               * ajuste_elo * ajuste_contexto * media_liga

    lambda_b = forca_ofensiva_b * fragilidade_defensiva_a
               * (1/ajuste_elo) * ajuste_contexto * media_liga
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import poisson

# Média de gols por time em uma partida internacional típica (baseline).
LEAGUE_AVG_GOALS = 1.35

# Quanto a diferença de Elo "puxa" os lambdas. Calibrado por backtest
# walk-forward (ver src/backtest.py): valores maiores deixam o modelo
# overconfident. 0.0005 ~ fator e^0.20 ≈ 1.22 no ataque por 400 pts de Elo.
ELO_SCALE = 0.0005

# Encolhimento (shrinkage) das razões de forma recente em direção a 1.0.
# Com apenas ~10 jogos a forma é ruidosa; puxá-la para a média evita
# previsões extremas e melhora a calibração (Brier ~0.649 < baseline 0.667).
# 0 = ignora a forma; 1 = usa a forma crua.
FORM_SHRINKAGE = 0.35

# Em jogos de mata-mata o futebol tende a ser mais conservador (seção 17.3).
KNOCKOUT_GOAL_DAMPING = 0.92

# Vantagem de campo: aplicada APENAS quando o time_a joga em casa
# (``mando_neutro == 0``). Em campo neutro — como a maioria dos jogos da Copa
# do Mundo — nenhum ajuste é feito. Calibrado pelo backtest walk-forward em
# dados reais (o modelo sem mando subestimava o mandante). O mandante marca
# ~30% mais; o visitante marca ~7% menos.
HOME_ADV_ATTACK = 1.30
HOME_ADV_DEFENSE = 0.93


def _shrink(ratio: float, amount: float = FORM_SHRINKAGE) -> float:
    """Puxa uma razão de forma (centrada em 1.0) em direção à média."""
    return 1.0 + amount * (ratio - 1.0)


def _elo_adjustment(diferenca_elo: float) -> float:
    """Fator multiplicativo (>1 favorece o time A) derivado da diferença de Elo."""
    return float(np.exp(ELO_SCALE * diferenca_elo))


def estimate_lambdas(
    match_features: Dict[str, float],
    league_avg_goals: float = LEAGUE_AVG_GOALS,
) -> tuple[float, float]:
    """Estima ``(lambda_a, lambda_b)`` para uma partida.

    Parameters
    ----------
    match_features:
        Dicionário com, no mínimo:
            ``forca_ofensiva_a``, ``fragilidade_defensiva_a``,
            ``forca_ofensiva_b``, ``fragilidade_defensiva_b``,
            ``diferenca_elo`` (opcional), ``jogo_eliminatorio`` (opcional).
    league_avg_goals:
        Baseline de gols por time.

    Returns
    -------
    (lambda_a, lambda_b)
    """
    off_a = _shrink(match_features.get("forca_ofensiva_a", 1.0))
    def_a = _shrink(match_features.get("fragilidade_defensiva_a", 1.0))
    off_b = _shrink(match_features.get("forca_ofensiva_b", 1.0))
    def_b = _shrink(match_features.get("fragilidade_defensiva_b", 1.0))

    elo_diff = match_features.get("diferenca_elo", 0.0)
    elo_adj = _elo_adjustment(elo_diff)

    # Ataque do A vs. defesa do B, e vice-versa.
    lambda_a = league_avg_goals * off_a * def_b * elo_adj
    lambda_b = league_avg_goals * off_b * def_a / elo_adj

    # Vantagem de campo: só quando o time_a é mandante (mando_neutro == 0).
    # Em campo neutro (padrão; quase toda a Copa do Mundo) não há ajuste.
    if not match_features.get("mando_neutro", 1):
        lambda_a *= HOME_ADV_ATTACK
        lambda_b *= HOME_ADV_DEFENSE

    # Ajuste de contexto: mata-mata reduz levemente o número de gols.
    if match_features.get("jogo_eliminatorio", 0):
        lambda_a *= KNOCKOUT_GOAL_DAMPING
        lambda_b *= KNOCKOUT_GOAL_DAMPING

    # Limites de sanidade.
    lambda_a = float(np.clip(lambda_a, 0.15, 5.0))
    lambda_b = float(np.clip(lambda_b, 0.15, 5.0))
    return lambda_a, lambda_b


def estimate_lambdas_for_fixture(
    time_a: str,
    time_b: str,
    strengths: pd.DataFrame,
    diferenca_elo: Optional[float] = None,
    jogo_eliminatorio: int = 0,
    mando_neutro: int = 1,
    league_avg_goals: float = LEAGUE_AVG_GOALS,
) -> tuple[float, float]:
    """Conveniência: monta o dicionário de features a partir do resumo de força.

    ``strengths`` é o DataFrame indexado por time produzido em
    ``feature_engineering.build_team_strengths``.

    ``mando_neutro`` controla a vantagem de campo: 1 (padrão) = campo neutro,
    como na maioria dos jogos da Copa; 0 = ``time_a`` é mandante.
    """
    from elo_model import DEFAULT_ELO

    if time_a not in strengths.index or time_b not in strengths.index:
        raise KeyError(f"Time sem histórico de força: {time_a} ou {time_b}")

    sa = strengths.loc[time_a]
    sb = strengths.loc[time_b]

    if diferenca_elo is None:
        diferenca_elo = float(sa.get("elo", DEFAULT_ELO) - sb.get("elo", DEFAULT_ELO))

    features = {
        "forca_ofensiva_a": float(sa["forca_ofensiva"]),
        "fragilidade_defensiva_a": float(sa["fragilidade_defensiva"]),
        "forca_ofensiva_b": float(sb["forca_ofensiva"]),
        "fragilidade_defensiva_b": float(sb["fragilidade_defensiva"]),
        "diferenca_elo": diferenca_elo,
        "jogo_eliminatorio": jogo_eliminatorio,
        "mando_neutro": mando_neutro,
    }
    return estimate_lambdas(features, league_avg_goals)


def calculate_score_matrix(lambda_a: float, lambda_b: float, max_goals: int = 6) -> np.ndarray:
    """Matriz de probabilidades de placar via produto de Poissons independentes.

    ``M[i, j]`` = P(time A faz i gols E time B faz j gols).

    Os gols são tratados como independentes (Poisson simples). O modelo
    Dixon-Coles, que corrige a correlação em placares baixos, fica para a V4.

    Returns
    -------
    numpy.ndarray
        Matriz ``(max_goals+1) x (max_goals+1)`` normalizada para somar 1.
    """
    goals = np.arange(0, max_goals + 1)
    p_a = poisson.pmf(goals, lambda_a)
    p_b = poisson.pmf(goals, lambda_b)
    matrix = np.outer(p_a, p_b)
    # Renormaliza para compensar a cauda truncada em max_goals.
    total = matrix.sum()
    if total > 0:
        matrix = matrix / total
    return matrix


def probabilities_from_matrix(matrix: np.ndarray) -> Dict[str, float]:
    """Deriva probabilidades de mercado a partir da matriz de placares.

    Returns
    -------
    dict
        ``prob_vitoria_time_a``, ``prob_empate``, ``prob_vitoria_time_b``,
        ``prob_over_2_5``, ``prob_under_2_5``, ``prob_ambos_marcam`` e
        ``placar_mais_provavel``.
    """
    n = matrix.shape[0]
    idx = np.arange(n)
    goals_a = idx[:, None]
    goals_b = idx[None, :]

    p_home = matrix[goals_a > goals_b].sum()
    p_draw = matrix[goals_a == goals_b].sum()
    p_away = matrix[goals_a < goals_b].sum()

    total_goals = goals_a + goals_b
    p_over_05 = matrix[total_goals >= 1].sum()
    p_over_15 = matrix[total_goals >= 2].sum()
    p_over_25 = matrix[total_goals >= 3].sum()
    p_under_25 = matrix[total_goals <= 2].sum()

    p_btts = matrix[(goals_a >= 1) & (goals_b >= 1)].sum()

    best = np.unravel_index(np.argmax(matrix), matrix.shape)
    placar = f"{best[0]}x{best[1]}"

    return {
        "prob_vitoria_time_a": float(p_home),
        "prob_empate": float(p_draw),
        "prob_vitoria_time_b": float(p_away),
        "prob_over_0_5": float(p_over_05),
        "prob_over_1_5": float(p_over_15),
        "prob_over_2_5": float(p_over_25),
        "prob_under_2_5": float(p_under_25),
        "prob_ambos_marcam": float(p_btts),
        "placar_mais_provavel": placar,
    }
