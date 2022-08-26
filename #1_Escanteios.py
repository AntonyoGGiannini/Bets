import pandas as pd
import requests
from datetime import datetime, timedelta
import json
import sqlite3
import sqlite3
import functions as ft
import numpy as np

desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 10)

conn = sqlite3.connect("db_sports.sqlite")
cursor = conn.cursor()

# JOGO EXEMPLO: FLUMINENSE x PALMEIRAS
mandante = 'Fluminense'
visitante = 'Palmeiras'

print('--------------------------------------------------------------------------------------------------------------------------------------')
# CHUTES E ESCANTEIOS A FAVOR DO MANDANTE
sql_query = f"""SELECT f.ID_FIXTURE, f.DATE, t.TEAM_NAME, t2.TEAM_NAME, s.TOTAL_SHOTS, s.CORNER_KICKS, s.BALL_POSSESION, s.TOTAL_PASSES, s.PASSES_ACCURATE FROM CAD_FIXTURE f
                    JOIN CAD_STATISTICS s ON s.ID_FIXTURE = f.ID_FIXTURE AND s.HOME = 1
                    JOIN CAD_TEAM t ON t.ID_TEAM = f.ID_HOME_TEAM
                    JOIN CAD_TEAM t2 ON t2.ID_TEAM = f.ID_AWAY_TEAM
                    WHERE t.TEAM_NAME = '{mandante}'"""

dados = pd.read_sql_query(sql_query, conn)
print(dados)
print('--------------------------------------------------------------------------------------------------------------------------------------')

sql_query = f"""SELECT f.ID_FIXTURE, f.DATE, t.TEAM_NAME, t2.TEAM_NAME, s.TOTAL_SHOTS, s.CORNER_KICKS, s.BALL_POSSESION, s.TOTAL_PASSES, s.PASSES_ACCURATE FROM CAD_FIXTURE f
                    JOIN CAD_STATISTICS s ON s.ID_FIXTURE = f.ID_FIXTURE AND s.HOME = 0
                    JOIN CAD_TEAM t ON t.ID_TEAM = f.ID_AWAY_TEAM
                    JOIN CAD_TEAM t2 ON t2.ID_TEAM = f.ID_HOME_TEAM
                    WHERE t.TEAM_NAME = '{visitante}'"""

dados = pd.read_sql_query(sql_query, conn)
print(dados)

# VARIAVEIS QUE PODEM INFLUENCIAR NUMERO DE ESCANTEIOS
# - Posse de bola (buscar relação)
# - Passes Totais (buscar relação)
# - Chutes (totais, bloqueados, defendidos, dentro e fora da área)

# POSSIVEL ANALISE APROFUNDADA
# - Shots insidebox e outside box influenciam na chance de ser escanteio?
