# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Use the kagglehub client library to attach Kaggle resources like competitions, datasets, and models to your session
# Learn more about kagglehub: https://github.com/Kaggle/kagglehub/blob/main/README.md

import kagglehub
# kagglehub.dataset_download('<owner>/<dataset-slug>')


import os
import gc
import xarray as xr
import pandas as pd
import lightgbm as lgb

base_path = '/kaggle/input/competitions/previsao-climatica-de-precipitacao-sobre-a-america-do-sul/'

print("A carregar matrizes base...")
ds_tp = xr.open_dataset(os.path.join(base_path, "treino_tp.nc"))
ds_alvo = xr.open_dataset(os.path.join(base_path, "treino_tp_alvo.nc"))
ds_teste = xr.open_dataset(os.path.join(base_path, "teste_features.nc"))

print("A estruturar dados espaciais e temporais...")
df_treino = ds_tp.to_dataframe().reset_index()
df_treino['tp_alvo'] = ds_alvo.to_dataframe().reset_index()['tp_alvo']

# 1. O Pulo do Gato: Extrair o mês para dar contexto sazonal ao modelo
df_treino['mes'] = df_treino['time'].dt.month
df_treino = df_treino.dropna(subset=['tp_alvo', 'tp'])

print("A calcular a Climatologia Histórica (Média Sazonal)...")
# 2. Criar a "memória climática": média de chuva por pixel e por mês
climatologia = df_treino.groupby(['lat', 'lon', 'mes'])['tp_alvo'].mean().reset_index()
climatologia = climatologia.rename(columns={'tp_alvo': 'tp_clima'})

# Fundir a climatologia de volta aos dados de treino
df_treino = df_treino.merge(climatologia, on=['lat', 'lon', 'mes'], how='left')

# Libertar memória RAM agressivamente antes do treino
del ds_tp, ds_alvo
gc.collect()

# Novas features com contexto sazonal e histórico
features = ['tp', 'lat', 'lon', 'mes', 'tp_clima']
target = 'tp_alvo'

print("A treinar o modelo focado em anomalias...")
modelo = lgb.LGBMRegressor(
    n_estimators=400,       # Mais árvores para aprender os resíduos finos
    learning_rate=0.05, 
    num_leaves=127,         # Maior profundidade para captar padrões regionais
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
modelo.fit(df_treino[features], df_treino[target])

print("A preparar inferência e a alinhar climatologia nos dados de teste...")
df_teste = ds_teste.to_dataframe().reset_index()
df_teste['tp'] = df_teste['tp_ultima_obs'] 
df_teste['mes'] = df_teste['time'].dt.month

# Alinhar a memória histórica com os dados de teste
df_teste = df_teste.merge(climatologia, on=['lat', 'lon', 'mes'], how='left')

# Segurança: preencher valores sem histórico com 0 (evita quebra do preditor)
df_teste['tp_clima'] = df_teste['tp_clima'].fillna(0)

previsoes = modelo.predict(df_teste[features])

print("A guardar submissão final...")
df_submissao = pd.read_csv(os.path.join(base_path, "sample_submission.csv"))
df_submissao['tp_mm_day'] = previsoes
caminho_saida = "/kaggle/working/submissao_climatologia_final_worcap.csv"
df_submissao.to_csv(caminho_saida, index=False)
print(f"Ficheiro de elite gerado com sucesso em: {caminho_saida}")
