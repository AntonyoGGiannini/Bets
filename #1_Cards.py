import pandas as pd
import requests
from datetime import datetime, timedelta
import json
import sqlite3
import sqlite3
pd.set_option('display.max_columns', 50)

conn = sqlite3.connect("db_sports.sqlite")
cursor = conn.cursor()

sql_query = f"""SELECT REPLACE(f.REFEREE, ', Brazil', '') as JUIZ, COUNT(f.REFEREE)/2 as NUM, AVG(s.YELLOW_CARDS) as AMARELO, AVG(s.RED_CARDS) as VERMELHO, AVG(s.FOULS) as FALTAS 
                FROM CAD_FIXTURE f 
                JOIN CAD_STATISTICS s ON s.ID_FIXTURE = f.ID_FIXTURE 
                GROUP BY JUIZ"""
dados = pd.read_sql_query(sql_query, conn)
dados = dados.sort_values('NUM', ascending=False)
print(dados)
