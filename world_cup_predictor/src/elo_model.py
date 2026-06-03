"""
elo_model.py
============

Cálculo e atualização de ratings Elo para seleções.

O Elo é a base da "força estrutural" do modelo. A ideia é simples:

- Toda seleção começa com um rating base (default 1500).
- Após cada partida, o vencedor "rouba" pontos do perdedor.
- A quantidade de pontos trocados depende de:
    * o resultado esperado (dado pela diferença de Elo);
    * o resultado real;
    * o fator K (quanto o rating se move por jogo);
    * a importância do jogo (Copa do Mundo move mais que amistoso);
    * a margem de gols (goleadas movem um pouco mais).

Referência conceitual: World Football Elo Ratings.
"""

from __future__ import annotations

from typing import Dict, Iterable

# Rating inicial padrão para uma seleção sem histórico.
DEFAULT_ELO = 1500.0

# Peso por tipo de competição (ver seção 17.4 do projeto).
# Jogos oficiais importantes movem o rating mais do que amistosos.
COMPETITION_WEIGHT = {
    "World Cup": 60.0,
    "Copa do Mundo": 60.0,
    "Continental": 50.0,        # Euro, Copa América
    "Qualifiers": 40.0,         # Eliminatórias
    "Eliminatorias": 40.0,
    "Nations League": 30.0,
    "Friendly": 20.0,
    "Amistoso": 20.0,
}

# Fator K padrão quando a competição não é reconhecida.
DEFAULT_K = 30.0


def initialize_elo(teams: Iterable[str], base_rating: float = DEFAULT_ELO) -> Dict[str, float]:
    """Cria um dicionário ``{time: rating}`` com o rating base para cada time.

    Parameters
    ----------
    teams:
        Iterável com os nomes das seleções.
    base_rating:
        Rating inicial atribuído a todas as seleções.

    Returns
    -------
    dict
        Mapa ``nome_do_time -> rating``.
    """
    return {team: float(base_rating) for team in dict.fromkeys(teams)}


def expected_score(rating_a: float, rating_b: float) -> float:
    """Probabilidade esperada (Elo) de o time A vencer o time B.

    Retorna um valor entre 0 e 1. Empates são absorvidos pela escala
    contínua, então este número representa "expectativa de pontos" e não
    estritamente "probabilidade de vitória".
    """
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def _result_score(goals_a: int, goals_b: int) -> float:
    """Converte o placar no "score" do Elo: 1.0 vitória, 0.5 empate, 0.0 derrota."""
    if goals_a > goals_b:
        return 1.0
    if goals_a < goals_b:
        return 0.0
    return 0.5


def _goal_difference_multiplier(goals_a: int, goals_b: int) -> float:
    """Multiplicador que dá um pouco mais de peso a vitórias por margem larga.

    Segue a heurística do World Football Elo:
        margem 1 -> 1.00
        margem 2 -> 1.50
        margem 3+ -> 1.75, 1.875, ...
    """
    margin = abs(goals_a - goals_b)
    if margin <= 1:
        return 1.0
    if margin == 2:
        return 1.5
    return (11.0 + margin) / 8.0


def competition_k_factor(competition: str) -> float:
    """Retorna o fator K associado a uma competição."""
    return COMPETITION_WEIGHT.get(competition, DEFAULT_K)


def update_elo_after_match(
    elo_a: float,
    elo_b: float,
    goals_a: int,
    goals_b: int,
    competition: str = "Friendly",
) -> tuple[float, float]:
    """Atualiza o Elo de dois times após uma partida.

    Parameters
    ----------
    elo_a, elo_b:
        Ratings atuais dos times A e B.
    goals_a, goals_b:
        Gols marcados por cada time.
    competition:
        Nome da competição, usado para definir o fator K.

    Returns
    -------
    (novo_elo_a, novo_elo_b)
    """
    k = competition_k_factor(competition)
    gd_mult = _goal_difference_multiplier(goals_a, goals_b)

    exp_a = expected_score(elo_a, elo_b)
    score_a = _result_score(goals_a, goals_b)

    delta = k * gd_mult * (score_a - exp_a)

    # O que A ganha, B perde (jogo de soma zero).
    return elo_a + delta, elo_b - delta


def calculate_elo_difference(team_a: str, team_b: str, ratings: Dict[str, float]) -> float:
    """Diferença de Elo (A - B) usando o mapa de ratings.

    Times ausentes do mapa recebem o rating base.
    """
    return ratings.get(team_a, DEFAULT_ELO) - ratings.get(team_b, DEFAULT_ELO)


def build_elo_history(matches, base_rating: float = DEFAULT_ELO) -> Dict[str, float]:
    """Percorre um DataFrame de partidas em ordem cronológica e devolve os
    ratings finais de todas as seleções.

    Espera as colunas:
        ``time_a``, ``time_b``, ``gols_time_a``, ``gols_time_b``,
        ``competicao`` e (opcionalmente) ``data_jogo`` para ordenar.

    Returns
    -------
    dict
        Ratings Elo finais por seleção.
    """
    teams = set(matches["time_a"]).union(set(matches["time_b"]))
    ratings = initialize_elo(teams, base_rating)

    ordered = matches.sort_values("data_jogo") if "data_jogo" in matches.columns else matches

    for row in ordered.itertuples(index=False):
        a, b = row.time_a, row.time_b
        new_a, new_b = update_elo_after_match(
            ratings[a],
            ratings[b],
            int(row.gols_time_a),
            int(row.gols_time_b),
            getattr(row, "competicao", "Friendly"),
        )
        ratings[a], ratings[b] = new_a, new_b

    return ratings
