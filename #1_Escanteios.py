import pandas as pd
import requests
from datetime import datetime, timedelta
import json
import sqlite3
import sqlite3
import functions as ft
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sn

desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 10)

conn = sqlite3.connect("db_sports.sqlite")
cursor = conn.cursor()

query_mandante = """SELECT  s.CORNER_KICKS, s.SHOTS_ON_GOAL, s.SHOTS_OFF_GOAL,
                           s.TOTAL_SHOTS, REPLACE(CAST(s.BALL_POSSESION as STRING), '%', '')*0.01 as BALL_POSSESION, 
                           s.TOTAL_PASSES, s2.TOTAL_PASSES as PASSES_AWAY FROM CAD_FIXTURE f
                    JOIN CAD_STATISTICS s ON s.ID_FIXTURE = f.ID_FIXTURE AND s.HOME = 1
                    JOIN CAD_STATISTICS s2 ON s2.ID_FIXTURE = f.ID_FIXTURE AND s2.HOME = 0"""

df1 = pd.read_sql_query(query_mandante, conn)
df1 = df1.replace('None', 0)

x = df1[['TOTAL_SHOTS', 'BALL_POSSESION', 'TOTAL_PASSES', 'PASSES_AWAY']]
y = df1['CORNER_KICKS']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

reg = LinearRegression()
reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)

print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
print('R2 Score: %.2f' % r2_score(y_test, y_pred))


# VARIAVEIS QUE PODEM INFLUENCIAR NUMERO DE ESCANTEIOS
# - Posse de bola (buscar relação)
# - Passes Totais (buscar relação)
# - Chutes (totais, bloqueados, defendidos, dentro e fora da área)

# POSSIVEL ANALISE APROFUNDADA
# - Shots insidebox e outside box influenciam na chance de ser escanteio?
