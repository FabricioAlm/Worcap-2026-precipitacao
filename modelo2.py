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

print("A carregar matrizes climáticas adicionais...")
ds_tp = xr.open_dataset(os.path.join(base_path, "treino_tp.nc"))
ds_alvo = xr.open_dataset(os.path.join(base_path, "treino_tp_alvo.nc"))
ds_t2 = xr.open_dataset(os.path.join(base_path, "treino_t2.nc"))
ds_u850 = xr.open_dataset(os.path.join(base_path, "treino_u_850.nc"))
ds_teste = xr.open_dataset(os.path.join(base_path, "teste_features.nc"))
df_submissao = pd.read_csv(os.path.join(base_path, "sample_submission.csv"))

print("A planificar o histórico de treino e a fundir variáveis de forma segura...")
# Conversão base principal
df_treino = ds_tp.to_dataframe().reset_index()

# Conversão e alinhamento de cada variável adicional para estrutura 1D
df_treino['tp_alvo'] = ds_alvo.to_dataframe().reset_index()['tp_alvo']
df_treino['t2'] = ds_t2.to_dataframe().reset_index()['t2']
df_treino['u_850'] = ds_u850.to_dataframe().reset_index()['u_850']

df_treino = df_treino.dropna(subset=['tp_alvo', 'tp'])

# Limpeza agressiva de memória RAM antes do cálculo intensivo
del ds_tp, ds_alvo, ds_t2, ds_u850
gc.collect()

features = ['tp', 't2', 'u_850', 'lat', 'lon']
target = 'tp_alvo'

print("A treinar o modelo avançado...")
modelo = lgb.LGBMRegressor(
    n_estimators=300, 
    learning_rate=0.05, 
    num_leaves=63,
    max_depth=8,
    random_state=42,
    n_jobs=-1
)
modelo.fit(df_treino[features], df_treino[target])

print("A gerar inferências com múltiplas variáveis...")
df_teste = ds_teste.to_dataframe().reset_index()
df_teste['tp'] = df_teste['tp_ultima_obs'] 

previsoes = modelo.predict(df_teste[features])

caminho_saida = "/kaggle/working/submissao_avancada_worcap.csv"
df_submissao['tp_mm_day'] = previsoes
df_submissao.to_csv(caminho_saida, index=False)
print(f"Ficheiro otimizado gerado com sucesso em: {caminho_saida}")
