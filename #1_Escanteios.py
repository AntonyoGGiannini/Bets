import pandas as pd
import requests
from datetime import datetime, timedelta
import json
import sqlite3
import functions as ft
import numpy as np
from sklearn import datasets, linear_model, metrics
from sklearn.model_selection import train_test_split

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 10)

conn = sqlite3.connect("db_sports.sqlite")
cursor = conn.cursor()

query_mandante = """SELECT (s.CORNER_KICKS + s2.CORNER_KICKS) as ESCANTEIOS, 
                           (s.TOTAL_SHOTS + s2.TOTAL_SHOTS) as CHUTES_TOTAL, 
                           ROUND(REPLACE(CAST(s.BALL_POSSESION as STRING), '%', '')*0.01, 2) as POSSE_MANDANTE, 
                           (s.TOTAL_PASSES + s2.TOTAL_PASSES) as PASSES_TOTAL FROM CAD_FIXTURE f
                    JOIN CAD_STATISTICS s ON s.ID_FIXTURE = f.ID_FIXTURE AND s.HOME = 1
                    JOIN CAD_STATISTICS s2 ON s2.ID_FIXTURE = f.ID_FIXTURE AND s2.HOME = 0"""

df = pd.read_sql_query(query_mandante, conn)
df = df.replace('None', 0)

n_escanteios = 9.5
df['MAIS'] = df['ESCANTEIOS'] > n_escanteios

X = df[['CHUTES_TOTAL', 'POSSE_MANDANTE', 'PASSES_TOTAL']]
y = df['MAIS']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,random_state=1)
reg = linear_model.LogisticRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

print('---------------------------------------------------------')
print("---- Logistic Regression model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100, " ----")
print('---------------------------------------------------------')
print('     ')

# BACKTEST
#   IR JOGO A JOGO em 2022 e VER QUAL SERIA O ACERTO
query = f"""SELECT f.ID_FIXTURE, f.DATE, f.ID_HOME_TEAM, f.ID_AWAY_TEAM, (s.CORNER_KICKS + s2.CORNER_KICKS) as ESCANTEIOS
                FROM CAD_FIXTURE f
                JOIN CAD_STATISTICS s ON s.ID_FIXTURE = f.ID_FIXTURE AND s.HOME = 1
                JOIN CAD_STATISTICS s2 ON s2.ID_FIXTURE = f.ID_FIXTURE AND s2.HOME = 0
            WHERE f.SEASON = 2022 AND f.ID_LEAGUE = 71"""

dados = pd.read_sql_query(query, conn)

g = 0
p = 0
for i in range(21, len(dados['ID_FIXTURE'])):
    data = dados['DATE'][i]
    id_mandante = dados['ID_HOME_TEAM'][i]
    id_visitante = dados['ID_AWAY_TEAM'][i]

    query_mandante = f"""SELECT (s.TOTAL_SHOTS + s2.TOTAL_SHOTS) as CHUTES_TOTAL,
                              ROUND(REPLACE(CAST(s.BALL_POSSESION as STRING), '%', '')*0.01, 2) as POSSE_MANDANTE, 
                              (s.TOTAL_PASSES + s2.TOTAL_PASSES) as PASSES_TOTAL 
                            FROM CAD_FIXTURE f
                            JOIN CAD_STATISTICS s ON s.ID_FIXTURE = f.ID_FIXTURE AND s.HOME = 1
                            JOIN CAD_STATISTICS s2 ON s2.ID_FIXTURE = f.ID_FIXTURE AND s2.HOME = 0
                            WHERE ID_HOME_TEAM = {id_mandante} AND SEASON = 2022 AND ID_LEAGUE = 71 AND DATE < '{data}'"""

    query_visitante = f"""SELECT (s.TOTAL_SHOTS + s2.TOTAL_SHOTS) as CHUTES_TOTAL,
                              ROUND(REPLACE(CAST(s.BALL_POSSESION as STRING), '%', '')*0.01, 2) as POSSE_MANDANTE, 
                              (s.TOTAL_PASSES + s2.TOTAL_PASSES) as PASSES_TOTAL 
                            FROM CAD_FIXTURE f
                            JOIN CAD_STATISTICS s ON s.ID_FIXTURE = f.ID_FIXTURE AND s.HOME = 1
                            JOIN CAD_STATISTICS s2 ON s2.ID_FIXTURE = f.ID_FIXTURE AND s2.HOME = 0
                            WHERE ID_AWAY_TEAM = {id_visitante} AND SEASON = 2022 AND ID_LEAGUE = 71 AND DATE < '{data}'"""

    df_mandante = pd.read_sql_query(query_mandante, conn).mean()
    df_visitante = pd.read_sql_query(query_visitante, conn).mean()
    df = list((df_mandante + df_visitante)/2)

    y_pred = reg.predict([df])
    if dados['ESCANTEIOS'][i] > n_escanteios:
        resultado = 'GANHOU'
        g += 1
    else:
        resultado = 'PERDEU'
        p += 1
    print(f"""Previsão para MAIS DE {n_escanteios} para o jogo {dados['ID_FIXTURE'][i]}: {y_pred} | Realizado: {dados['ESCANTEIOS'][i]}""")

print('------------------------------------------------')
total = len(dados['ID_FIXTURE'])
print(f"Total: {total} | Ganhou: {g} | Perdeu: {p}")
print(f"Assertividade: {round(100*g/total,2)}%")
