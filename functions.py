import pandas as pd
import requests
from datetime import datetime, timedelta
import json
import sqlite3
import time

conn = sqlite3.connect("db_sports.sqlite")
cursor = conn.cursor()

def Insert_Team(id_league, season):
    # BUSCA DE TIMES DA LIGA {id_league} e SEASON {season}
    string_sql_name = f"SELECT LEAGUE_NAME, COUNTRY FROM CAD_LEAGUE " \
                      f"WHERE LEAGUE_ID = {id_league}"
    busca = pd.read_sql_query(string_sql_name, conn)
    nome = busca.iloc[0, 0]
    pais = busca.iloc[0, 1]

    agora = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    info = f'[{agora}][INFO] Buscando times da {nome} {pais} {season}...'
    print(info)

    url = "https://api-football-v1.p.rapidapi.com/v3/teams"

    querystring = {"league": f"{id_league}",
                   "season": f"{season}"}

    headers = {
        "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com",
        "X-RapidAPI-Key": "856f12b018mshe79a583bf00cd5bp1cd869jsnf7faf4fb1444"
    }

    response = requests.request("GET", url, headers=headers, params=querystring)
    response = response.text
    response = json.loads(response)['response']

    for item in response:
        id = item['team']['id']
        team = item['team']['name']
        code = item['team']['code']
        country = item['team']['country']
        string = f"INSERT INTO CAD_TEAM (ID_TEAM, TEAM_NAME, TEAM_CODE, TEAM_COUNTRY) VALUES (" \
                 f"{id}, " \
                 f"'{team}', " \
                 f"'{code}', '{country}')"

        try:
            agora = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            info = f'[{agora}][INFO] Inserindo time {team}...'
            print(info)

            cursor.execute(string)
            conn.commit()
        except:
            pass

    agora = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    info = f'[{agora}][INFO] Atualização de times concluída...'
    print(info)
    return ()

def Get_LeagueName(id_league):
    string_sql_name = f"SELECT * FROM CAD_LEAGUE WHERE LEAGUE_ID = {id_league}"
    dados = pd.read_sql_query(string_sql_name, conn)
    nome = dados.iloc[0, 2]
    pais = dados.iloc[0, 3]
    return(nome, pais)

def Get_TeamName(id_team):
    string_sql_name = f"SELECT * FROM CAD_TEAM WHERE ID_TEAM = {id_team}"
    dados = pd.read_sql_query(string_sql_name, conn)
    nome = dados.iloc[0, 2]
    code = dados.iloc[0, 3]
    return(nome, code)

def Update_Match(id_league, season, data_final = ''):
    # BUSCA PARTIDAS DA LIGA {id_league} e SEASON {season}
    string_sql = f"SELECT * FROM CAD_FIXTURE " \
                 f"WHERE ID_LEAGUE = {id_league} AND SEASON = {season} ORDER BY DATE DESC"
    busca_data_inicial = pd.read_sql_query(string_sql, conn)

    # determina a DATA SEGUINTE do último jogo inserido
    if len(busca_data_inicial) == 0:
        data_inicial = '1900-01-01'
    else:
        data_inicial = busca_data_inicial['DATE'][0]
        data_inicial = dt.datetime.strptime(data_inicial, '%Y-%m-%d') + dt.timedelta(days=1)
        data_inicial = dt.datetime.strftime(data_inicial, '%Y-%m-%d')

    if data_final == '':
        data_final = dt.datetime.strftime(dt.datetime.today(), '%Y-%m-%d')

    # ----------------------------------------------------

    # BUSCA NOME DA LIGA {nome} e {pais} da LIGA ESCOLHIDA {id_league}
    string_sql_name = f"SELECT LEAGUE_NAME, COUNTRY FROM CAD_LEAGUE " \
                      f"WHERE LEAGUE_ID = {id_league}"
    busca = pd.read_sql_query(string_sql_name, conn)

    nome = busca.iloc[0, 0]
    pais = busca.iloc[0, 1]
    # ----------------------------------------------------------------

    # APENAS INSERE JOGOS REALIZADOS A PARTIR DO DIA SEGUINTE DO ÚLTIMO DADO DA BASE, ATÉ O DIA DE HOJE
    agora = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    info = f'[{agora}][INFO] Buscando jogos da {nome} {pais} {season} entre {data_inicial} e {data_final}'
    print(info)

    url = "https://api-football-v1.p.rapidapi.com/v3/fixtures"

    querystring = {"league": f"{id_league}",
                   "season": f"{season}",
                   "from": f"{data_inicial}",
                   "to": f"{data_final}"}

    headers = {
        "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com",
        "X-RapidAPI-Key": "856f12b018mshe79a583bf00cd5bp1cd869jsnf7faf4fb1444"
    }

    response = requests.request("GET", url, headers=headers, params=querystring)
    response = response.text
    response = json.loads(response)

    dados = response['response']

    for item in dados:
        status = item['fixture']['status']['short']
        if status == 'FT':
            id_fixture = item['fixture']['id']
            referee = item['fixture']['referee']
            timezone = item['fixture']['timezone']
            datetime = item['fixture']['date']
            date = datetime.split('T')[0]
            time = datetime.split('T')[1].split('+')[0]
            id_venue = item['fixture']['venue']['id']
            name_venue = item['fixture']['venue']['name']
            city_venue = item['fixture']['venue']['city']

            id_home_team = item['teams']['home']['id']
            name_home_team = item['teams']['home']['name']
            id_away_team = item['teams']['away']['id']
            name_away_team = item['teams']['away']['name']

            home_goals = item['goals']['home']
            away_goals = item['goals']['away']

            home_goals_ht = item['score']['halftime']['home']
            away_goals_ht = item['score']['halftime']['away']
            home_goals_ft = item['score']['fulltime']['home']
            away_goals_ft = item['score']['fulltime']['away']

            string = f"INSERT INTO CAD_FIXTURE (ID_FIXTURE, ID_LEAGUE, SEASON, TIMEZONE, DATE, TIME, NAME_VENUE, " \
                     f"CITY_VENUE, ID_HOME_TEAM, ID_AWAY_TEAM, HOME_GOALS, AWAY_GOALS, HOME_GOALS_HALFTIME, AWAY_GOALS_HALFTIME, " \
                     f"HOME_GOALS_FULLTIME, AWAY_GOALS_FULLTIME, REFEREE) VALUES (" \
                     f"{id_fixture}, {id_league}, {season}, '{timezone}', '{date}', '{time}', '{name_venue}', " \
                     f"'{city_venue}', {id_home_team}, {id_away_team}, {home_goals}, {away_goals}, " \
                     f"{home_goals_ht}, {away_goals_ht}, {home_goals_ft}, {away_goals_ft}, " \
                     f"'{referee}')"

            try:
                agora = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                info = f'[{agora}][INFO] Inserindo jogo {id_fixture} | {date} | {name_home_team} {home_goals} X {away_goals} {name_away_team}'
                print(info)

                cursor.execute(string)
                conn.commit()
            except:
                pass

    agora = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    info = f'[{agora}][INFO] Atualização de jogos concluída...'
    print(info)

def Insert_Statistics(id_fixture):
    string_sql_name = f"SELECT * FROM CAD_FIXTURE WHERE ID_FIXTURE = {id_fixture}"
    dados = pd.read_sql_query(string_sql_name, conn).reset_index().drop('index', axis=1)
    nome, pais = Get_LeagueName(dados.iloc[0, 2])
    season = dados.iloc[0, 3]
    home_team, home_code = Get_TeamName(dados.iloc[0, 9])
    away_team, away_code = Get_TeamName(dados.iloc[0, 10])
    data = dados.iloc[0, 5]

    url = "https://api-football-v1.p.rapidapi.com/v3/fixtures/statistics"
    querystring = {"fixture": f"{id_fixture}"}

    headers = {
        "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com",
        "X-RapidAPI-Key": "856f12b018mshe79a583bf00cd5bp1cd869jsnf7faf4fb1444"
    }

    try:
        response = requests.request("GET", url, headers=headers, params=querystring)
        response = response.text
        dados = json.loads(response)
        dados = dados['response']

        # Home Team
        home = dados[0]['statistics']
        if home[0]['type'] == 'Shots on Goal':
            Shots_On_Goal = home[0]['value']

        if home[1]['type'] == 'Shots off Goal':
            Shots_Off_Goal = home[1]['value']

        if home[2]['type'] == 'Total Shots':
            Total_Shots = home[2]['value']

        if home[3]['type'] == 'Blocked Shots':
            Blocked_Shots = home[3]['value']

        if home[4]['type'] == 'Shots insidebox':
            Shots_Insidebox = home[4]['value']

        if home[5]['type'] == 'Shots outsidebox':
            Shots_Outsidebox = home[5]['value']

        if home[6]['type'] == 'Fouls':
            Fouls = home[6]['value']

        if home[7]['type'] == 'Corner Kicks':
            Corner_Kicks = home[7]['value']

        if home[8]['type'] == 'Offsides':
            Offsides = home[8]['value']

        if home[9]['type'] == 'Ball Possession':
            Ball_Possession = home[9]['value']

        if home[10]['type'] == 'Yellow Cards':
            Yellow_Cards = home[10]['value']

        if home[11]['type'] == 'Red Cards':
            Red_Cards = home[11]['value']

        if home[12]['type'] == 'Goalkeeper Saves':
            Goalkeeper_Saves = home[12]['value']

        if home[13]['type'] == 'Total passes':
            Total_passes = home[13]['value']

        if home[14]['type'] == 'Passes accurate':
            Passes_accurate = home[14]['value']

        if home[15]['type'] == 'Passes %':
            Passes_per = home[15]['value']

        string_home = f"INSERT INTO CAD_STATISTICS (ID_FIXTURE, HOME, SHOTS_ON_GOAL, SHOTS_OFF_GOAL, TOTAL_SHOTS, " \
                      f"BLOCKED_SHOTS, SHOTS_INSIDEBOX, SHOTS_OUTSIDEBOX, FOULS, CORNER_KICKS, OFFSIDES, BALL_POSSESION, " \
                      f"YELLOW_CARDS, RED_CARDS, GOALKEEPER_SAVES, TOTAL_PASSES, PASSES_ACCURATE, PASSES_PER) " \
                      f"VALUES ({id_fixture}, {1}, '{Shots_On_Goal}', '{Shots_Off_Goal}', '{Total_Shots}', '{Blocked_Shots}', " \
                      f"'{Shots_Insidebox}', '{Shots_Outsidebox}', '{Fouls}', '{Corner_Kicks}', '{Offsides}', " \
                      f"'{Ball_Possession}', '{Yellow_Cards}', '{Red_Cards}', '{Goalkeeper_Saves}', '{Total_passes}', " \
                      f"'{Passes_accurate}', '{Passes_per}')"

        try:
            cursor.execute(string_home)
            conn.commit()
        except:
            pass

        # Away Team
        away = dados[1]['statistics']
        if away[0]['type'] == 'Shots on Goal':
            Shots_On_Goal_AW = away[0]['value']

        if away[1]['type'] == 'Shots off Goal':
            Shots_Off_Goal_AW = away[1]['value']

        if away[2]['type'] == 'Total Shots':
            Total_Shots_AW = away[2]['value']

        if away[3]['type'] == 'Blocked Shots':
            Blocked_Shots_AW = away[3]['value']

        if away[4]['type'] == 'Shots insidebox':
            Shots_Insidebox_AW = away[4]['value']

        if away[5]['type'] == 'Shots outsidebox':
            Shots_Outsidebox_AW = away[5]['value']

        if away[6]['type'] == 'Fouls':
            Fouls_AW = away[6]['value']

        if away[7]['type'] == 'Corner Kicks':
            Corner_Kicks_AW = away[7]['value']

        if away[8]['type'] == 'Offsides':
            Offsides_AW = away[8]['value']

        if away[9]['type'] == 'Ball Possession':
            Ball_Possession_AW = away[9]['value']

        if away[10]['type'] == 'Yellow Cards':
            Yellow_Cards_AW = away[10]['value']

        if away[11]['type'] == 'Red Cards':
            Red_Cards_AW = away[11]['value']

        if away[12]['type'] == 'Goalkeeper Saves':
            Goalkeeper_Saves_AW = away[12]['value']

        if away[13]['type'] == 'Total passes':
            Total_passes_AW = away[13]['value']

        if away[14]['type'] == 'Passes accurate':
            Passes_accurate_AW = away[14]['value']

        if away[15]['type'] == 'Passes %':
            Passes_per_AW = away[15]['value']

        string_away = f"INSERT INTO CAD_STATISTICS (ID_FIXTURE, HOME, SHOTS_ON_GOAL, SHOTS_OFF_GOAL, TOTAL_SHOTS, " \
                      f"BLOCKED_SHOTS, SHOTS_INSIDEBOX, SHOTS_OUTSIDEBOX, FOULS, CORNER_KICKS, OFFSIDES, BALL_POSSESION, " \
                      f"YELLOW_CARDS, RED_CARDS, GOALKEEPER_SAVES, TOTAL_PASSES, PASSES_ACCURATE, PASSES_PER) " \
                      f"VALUES ({id_fixture}, {0}, '{Shots_On_Goal_AW}', '{Shots_Off_Goal_AW}', '{Total_Shots_AW}', " \
                      f"'{Blocked_Shots_AW}', '{Shots_Insidebox_AW}', '{Shots_Outsidebox_AW}', '{Fouls_AW}', " \
                      f"'{Corner_Kicks_AW}', '{Offsides_AW}', '{Ball_Possession_AW}', '{Yellow_Cards_AW}', '{Red_Cards_AW}', " \
                      f"'{Goalkeeper_Saves_AW}', '{Total_passes_AW}', '{Passes_accurate_AW}', '{Passes_per_AW}')"

        try:
            cursor.execute(string_away)
            conn.commit()
        except:
            agora = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            info = f'[{agora}][ERRO] ERRO ao inserir os dados do jogo {id_fixture} | {data} | {home_team} x {away_team} | {nome} {int(season)} | {pais}...'
            print(info)

        agora = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        info = f'[{agora}][INFO] Inserindo dados do jogo {id_fixture} | {data} | {home_team} x {away_team} | {nome} {int(season)} | {pais}...'
        print(info)

    except:
        if dados['message'] == 'You have exceeded the rate limit per minute for your plan, BASIC, by the API provider':
            print('----------------------------------------------------------')
            print('---------- LIMITE DE BUSCAS POR MINUTO EXCEDIDO ----------')
            print('----------------------------------------------------------')
            time.sleep(10)
            Insert_Statistics(id_fixture)
        else:
            agora = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            info = f'[{agora}][ERRO] ERRO ao inserir os dados do jogo {id_fixture} | {data} | {home_team} x {away_team} | {nome} {int(season)} | {pais}...'
            print(info)

def Update_Statistics(id_league, season):
    # BUSCA NOME DA LIGA {nome} e {pais} da LIGA ESCOLHIDA {id_league}
    string_sql_name = f"SELECT LEAGUE_NAME, COUNTRY FROM CAD_LEAGUE " \
                      f"WHERE LEAGUE_ID = {id_league}"
    busca = pd.read_sql_query(string_sql_name, conn)

    nome = busca.iloc[0, 0]
    pais = busca.iloc[0, 1]
    # ----------------------------------------------------------------

    agora = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    info = f'[{agora}][INFO] Buscando estatísticas dos jogos da {nome} {pais} {season}...'
    print(info)

    # Verificar partidas já inseridas e que não possuem estatísticas
    string_sql = "SELECT DISTINCT ID_FIXTURE FROM CAD_STATISTICS"
    cad_statistics = list(pd.read_sql_query(string_sql, conn)['ID_FIXTURE'])

    string_sql = f"SELECT ID_FIXTURE FROM CAD_FIXTURE WHERE ID_LEAGUE = {id_league} AND SEASON = {season}"
    cad_fixtures = list(pd.read_sql_query(string_sql, conn)['ID_FIXTURE'])

    for id in cad_fixtures:
        if id not in cad_statistics:
            Insert_Statistics(id)

    agora = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    info = f'[{agora}][INFO] Atualização de estatísticas concluída...'
    print(info)

def Insert_Events(id_fixture):
    string_sql_name = f"SELECT * FROM CAD_FIXTURE WHERE ID_FIXTURE = {id_fixture}"
    dados = pd.read_sql_query(string_sql_name, conn).reset_index().drop('index', axis=1)
    nome, pais = Get_LeagueName(dados.iloc[0, 2])
    season = dados.iloc[0, 3]
    home_team, home_code = Get_TeamName(dados.iloc[0, 9])
    away_team, away_code = Get_TeamName(dados.iloc[0, 10])
    data = dados.iloc[0, 5]

    url = "https://api-football-v1.p.rapidapi.com/v3/fixtures/events"
    querystring = {"fixture": f"{id_fixture}"}

    headers = {
        "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com",
        "X-RapidAPI-Key": "856f12b018mshe79a583bf00cd5bp1cd869jsnf7faf4fb1444"
    }

    try:
        response = requests.request("GET", url, headers=headers, params=querystring)
        response = response.text
        dados = json.loads(response)
        dados = dados['response']

        for item in dados:
            id_team = item['team']['id']
            time_str = item['time']['elapsed']
            time_extra = item['time']['extra']
            id_player1 = item['player']['id']
            name_player1 = item['player']['name']
            id_player2 = item['assist']['id']
            name_player2 = item['assist']['name']
            event_type = item['type']
            detail = item['detail']
            comments = item['comments']

            string = f"INSERT INTO CAD_EVENTS (ID_FIXTURE, ID_TEAM, TIME, TIME_EXTRA, ID_PLAYER1, NAME_PLAYER1, ID_PLAYER2, NAME_PLAYER2, EVENT_TYPE, DETAILS, COMMENTS) " \
                      f"VALUES ({id_fixture}, {id_team}, '{time_str}', '{time_extra}', '{id_player1}', '{name_player1}', " \
                      f"'{id_player2}', '{name_player2}', '{event_type}', '{detail}', '{comments}')"

            try:
                cursor.execute(string)
                conn.commit()
            except:
                agora = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                info = f'[{agora}][ERRO] ERRO ao inserir os EVENTOS do jogo {id_fixture} | {data} | {home_team} x {away_team} | {nome} {int(season)} | {pais}...'
                print(info)

        agora = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        info = f'[{agora}][INFO] Inserindo eventos do jogo {id_fixture} | {data} | {home_team} x {away_team} | {nome} {int(season)} | {pais}...'
        print(info)

    except:
        if dados['message'] == 'You have exceeded the rate limit per minute for your plan, BASIC, by the API provider':
            print('----------------------------------------------------------')
            print('---------- LIMITE DE BUSCAS POR MINUTO EXCEDIDO ----------')
            print('----------------------------------------------------------')
            time.sleep(10)
            Insert_Events(id_fixture)
        else:
            agora = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            info = f'[{agora}][ERRO] ERRO ao inserir os EVENTOS do jogo {id_fixture} | {data} | {home_team} x {away_team} | {nome} {int(season)} | {pais}...'
            print(info)

def Update_Events(id_league, season):
    # BUSCA NOME DA LIGA {nome} e {pais} da LIGA ESCOLHIDA {id_league}
    string_sql_name = f"SELECT LEAGUE_NAME, COUNTRY FROM CAD_LEAGUE " \
                      f"WHERE LEAGUE_ID = {id_league}"
    busca = pd.read_sql_query(string_sql_name, conn)

    nome = busca.iloc[0, 0]
    pais = busca.iloc[0, 1]
    # ----------------------------------------------------------------

    agora = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    info = f'[{agora}][INFO] Buscando eventos dos jogos da {nome} {pais} {season}...'
    print(info)

    # Verificar partidas já inseridas e que não possuem eventos
    string_sql = "SELECT DISTINCT ID_FIXTURE FROM CAD_EVENTS"
    cad_events = list(pd.read_sql_query(string_sql, conn)['ID_FIXTURE'])

    string_sql = f"SELECT ID_FIXTURE FROM CAD_FIXTURE WHERE ID_LEAGUE = {id_league} AND SEASON = {season}"
    cad_fixtures = list(pd.read_sql_query(string_sql, conn)['ID_FIXTURE'])

    for id in cad_fixtures:
        if id not in cad_events:
            Insert_Events(id)

    agora = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    info = f'[{agora}][INFO] Atualização de eventos concluída...'
    print(info)

def Insert_PlayerStatistics(id_fixture):
    string_sql_name = f"SELECT * FROM CAD_FIXTURE WHERE ID_FIXTURE = {id_fixture}"
    dados = pd.read_sql_query(string_sql_name, conn).reset_index().drop('index', axis=1)
    nome, pais = Get_LeagueName(dados.iloc[0, 2])
    season = dados.iloc[0, 3]
    home_team, home_code = Get_TeamName(dados.iloc[0, 9])
    away_team, away_code = Get_TeamName(dados.iloc[0, 10])
    data = dados.iloc[0, 5]

    url = "https://api-football-v1.p.rapidapi.com/v3/fixtures/players"
    querystring = {"fixture": f"{id_fixture}"}

    headers = {
        "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com",
        "X-RapidAPI-Key": "856f12b018mshe79a583bf00cd5bp1cd869jsnf7faf4fb1444"
    }

    try:
        response = requests.request("GET", url, headers=headers, params=querystring)
        response = response.text
        dados = json.loads(response)
        dados = dados['response']

        for item in dados:
            id_team = item['team']['id']

            for player in item['players']:
                id_jogador = player['player']['id']
                nome_jogador = player['player']['name'].replace("'", "")

                for statistic in player['statistics']:
                    games_minutes = statistic['games']['minutes']
                    games_number = statistic['games']['number']
                    games_position = statistic['games']['position']
                    games_rating = statistic['games']['rating']
                    games_captain = statistic['games']['captain']
                    games_substitute = statistic['games']['substitute']
                    offsides = statistic['offsides']
                    shots_total = statistic['shots']['total']
                    shots_on_goal = statistic['shots']['on']
                    goals_total = statistic['goals']['total']
                    goals_conceded = statistic['goals']['conceded']
                    goals_assists = statistic['goals']['assists']
                    goals_saves = statistic['goals']['saves']
                    passes_total = statistic['passes']['total']
                    passes_key = statistic['passes']['key']
                    passes_accuracy = statistic['passes']['accuracy']
                    tackles_total = statistic['tackles']['total']
                    tackles_blocks = statistic['tackles']['blocks']
                    tackles_interceptions = statistic['tackles']['interceptions']
                    duels_total = statistic['duels']['total']
                    duels_won = statistic['duels']['won']
                    dribbles_attempts = statistic['dribbles']['attempts']
                    dribbles_success = statistic['dribbles']['success']
                    dribles_past = statistic['dribbles']['past']
                    fouls_drawn = statistic['fouls']['drawn']
                    fouls_committed = statistic['fouls']['committed']
                    cards_yellow = statistic['cards']['yellow']
                    cards_red = statistic['cards']['red']
                    penalty_won = statistic['penalty']['won']
                    penalty_commited = statistic['penalty']['commited']
                    penalty_scored = statistic['penalty']['scored']
                    penalty_missed = statistic['penalty']['missed']
                    penalty_saved = statistic['penalty']['saved']

                    string = f"INSERT INTO CAD_PLAYER_STATISTICS (ID_FIXTURE, ID_TEAM, ID_JOGADOR, NOME_JOGADOR, " \
                             f"GAMES_MINUTES, GAMES_NUMBER, GAMES_POSITION, GAMES_RATING, GAMES_CAPTAIN, GAMES_SUBSTITUTE," \
                             f"OFFSIDES, SHOTS_TOTAL, SHOTS_ON, GOALS_TOTAL, GOALS_CONCEDED, GOALS_ASSISTS, GOALS_SAVES, " \
                             f"PASSES_TOTAL, PASSES_KEY, PASSES_ACCURACY, TACKLES_TOTAL, TACKLES_BLOCKS, TACKLES_INTERCEPTIONS," \
                             f"DUELS_TOTAL, DUELS_WON, DRIBBLES_ATTEMPTS, DRIBBLES_SUCCESS, DRIBBLES_PAST," \
                             f"FOULS_DRAWN, FOULS_COMMITTED, CARDS_YELLOW, CARDS_RED, PENALTY_WON, PENALTY_COMMITED, " \
                             f"PENALTY_SCORED, PENALTY_MISSED, PENALTY_SAVED) " \
                              f"VALUES ({id_fixture}, {id_team}, {id_jogador}, '{nome_jogador}', '{games_minutes}', " \
                             f"'{games_number}', '{games_position}', '{games_rating}', '{games_captain}', '{games_substitute}', " \
                             f"'{offsides}', '{shots_total}', '{shots_on_goal}', '{goals_total}', '{goals_conceded}', " \
                             f"'{goals_assists}', '{goals_saves}', '{passes_total}', '{passes_key}', '{passes_accuracy}', " \
                             f"'{tackles_total}', '{tackles_blocks}', '{tackles_interceptions}', '{duels_total}', '{duels_won}', " \
                             f"'{dribbles_attempts}', '{dribbles_success}', '{dribles_past}', '{fouls_drawn}', '{fouls_committed}', " \
                             f"'{cards_yellow}', '{cards_red}', '{penalty_won}', '{penalty_commited}', '{penalty_scored}', " \
                             f"'{penalty_missed}', '{penalty_saved}')"


                    #try:
                    cursor.execute(string)
                    conn.commit()
                    #except:
                    #    agora = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    #    info = f'[{agora}][ERRO] ERRO ao inserir as ESTATÍSTICAS DOS JOGADORES do jogo {id_fixture} | {data} | {home_team} x {away_team} | {nome} {int(season)} | {pais}...'
                    #    print(info)

        agora = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        info = f'[{agora}][INFO] Inserindo estatísticas dos jogadores do jogo {id_fixture} | {data} | {home_team} x {away_team} | {nome} {int(season)} | {pais}...'
        print(info)

    except:
        if dados['message'] == 'You have exceeded the rate limit per minute for your plan, BASIC, by the API provider':
            print('----------------------------------------------------------')
            print('---------- LIMITE DE BUSCAS POR MINUTO EXCEDIDO ----------')
            print('----------------------------------------------------------')
            time.sleep(10)
            Insert_PlayerStatistics(id_fixture)
        else:
            agora = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            info = f'[{agora}][ERRO] ERRO ao inserir as ESTATÍSTICAS DOS JOGADORES do jogo {id_fixture} | {data} | {home_team} x {away_team} | {nome} {int(season)} | {pais}...'
            print(info)

def Update_PlayerStatistics(id_league, season):
    # BUSCA NOME DA LIGA {nome} e {pais} da LIGA ESCOLHIDA {id_league}
    string_sql_name = f"SELECT LEAGUE_NAME, COUNTRY FROM CAD_LEAGUE " \
                      f"WHERE LEAGUE_ID = {id_league}"
    busca = pd.read_sql_query(string_sql_name, conn)

    nome = busca.iloc[0, 0]
    pais = busca.iloc[0, 1]
    # ----------------------------------------------------------------

    agora = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    info = f'[{agora}][INFO] Buscando estatísticas dos jogadores dos jogos da {nome} {pais} {season}...'
    print(info)

    # Verificar partidas já inseridas e que não possuem eventos
    string_sql = "SELECT DISTINCT ID_FIXTURE FROM CAD_PLAYER_STATISTICS"
    cad_statistics = list(pd.read_sql_query(string_sql, conn)['ID_FIXTURE'])

    string_sql = f"SELECT ID_FIXTURE FROM CAD_FIXTURE WHERE ID_LEAGUE = {id_league} AND SEASON = {season}"
    cad_fixtures = list(pd.read_sql_query(string_sql, conn)['ID_FIXTURE'])

    for id in cad_fixtures:
        if id not in cad_statistics:
            Insert_PlayerStatistics(id)

    agora = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    info = f'[{agora}][INFO] Atualização de estatísticas de jogadores concluída...'
    print(info)

def Update_All(id_league, season):
    data = datetime.strftime(datetime.today() - timedelta(1), '%Y-%m-%d')
    # ROTINA DE ATUALIZACAO
    Update_Match(id_league, season, data)
    print("-----------------------------------------------------------------------")
    Update_Statistics(id_league, season)
    print("-----------------------------------------------------------------------")
    Update_Events(id_league, season)
    print("-----------------------------------------------------------------------")
    Update_PlayerStatistics(id_league, season)
