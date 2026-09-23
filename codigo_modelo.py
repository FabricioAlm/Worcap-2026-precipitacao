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
import xarray as xr
import pandas as pd
import lightgbm as lgb

# 1. Carregamento direto usando o caminho impresso no log
base_path = '/kaggle/input/competitions/previsao-climatica-de-precipitacao-sobre-a-america-do-sul/'

print(f"Tentando abrir arquivos em: {base_path}")
ds_tp = xr.open_dataset(os.path.join(base_path, "treino_tp.nc"))
ds_alvo = xr.open_dataset(os.path.join(base_path, "treino_tp_alvo.nc"))
ds_teste = xr.open_dataset(os.path.join(base_path, "teste_features.nc"))
df_submissao = pd.read_csv(os.path.join(base_path, "sample_submission.csv"))
print("Arquivos carregados com sucesso!")

# 2. Conversão para formato tabular estruturado
print("Processando dados históricos...")
df_treino = ds_tp.to_dataframe().reset_index()
df_alvo = ds_alvo.to_dataframe().reset_index()

df_treino['tp_alvo'] = df_alvo['tp_alvo']
df_treino = df_treino.dropna(subset=['tp_alvo', 'tp'])

# 3. Definição de variáveis base
features = ['tp', 'lat', 'lon']
target = 'tp_alvo'

# 4. Treinamento do estimador
print("Treinando o modelo (isso pode levar alguns minutos)...")
modelo = lgb.LGBMRegressor(
    n_estimators=100, 
    learning_rate=0.1, 
    random_state=42,
    n_jobs=-1
)
modelo.fit(df_treino[features], df_treino[target])

# 5. Inferência nos dados de teste
print("Gerando previsões para a submissão...")
df_teste = ds_teste.to_dataframe().reset_index()
df_teste['tp'] = df_teste['tp_ultima_obs'] 
previsoes = modelo.predict(df_teste[features])

# 6. Preenchimento e salvamento do arquivo
df_submissao['tp_mm_day'] = previsoes
caminho_saida = "/kaggle/working/submissao_baseline_worcap.csv"
df_submissao.to_csv(caminho_saida, index=False)
print(f"Arquivo gerado com sucesso em: {caminho_saida}")
