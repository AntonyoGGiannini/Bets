import pandas as pd
import requests
from datetime import datetime, timedelta
import json
import sqlite3
import numpy as np
import sqlite3

desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 10)

conn = sqlite3.connect("db_sports.sqlite")
cursor = conn.cursor()

mandante = 'Palmeiras'
sql_query = f"""SELECT f.ID_FIXTURE, f.DATE, t.TEAM_NAME, f.HOME_GOALS, f.AWAY_GOALS, f.REFEREE 
                FROM CAD_FIXTURE f
                JOIN CAD_STATISTICS s ON s.ID_FIXTURE = f.ID_FIXTURE AND s.HOME = 1
                JOIN CAD_TEAM t ON t.ID_TEAM = f.ID_HOME_TEAM AND s.HOME = 1
                WHERE t.TEAM_NAME = '{mandante}'"""
dados = pd.read_sql_query(sql_query, conn)
print(dados)