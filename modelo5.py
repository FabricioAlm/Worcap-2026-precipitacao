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
import numpy as np
import xarray as xr
import pandas as pd
import lightgbm as lgb
import xgboost as xgb

base_path = '/kaggle/input/competitions/previsao-climatica-de-precipitacao-sobre-a-america-do-sul/'

print("1. A carregar bases e aplicar o truque do Logaritmo...")
ds_tp = xr.open_dataset(os.path.join(base_path, "treino_tp.nc"))
ds_alvo = xr.open_dataset(os.path.join(base_path, "treino_tp_alvo.nc"))
ds_teste = xr.open_dataset(os.path.join(base_path, "teste_features.nc"))

df_treino = ds_tp.to_dataframe().reset_index()
df_treino['tp_alvo'] = ds_alvo.to_dataframe().reset_index()['tp_alvo']
df_treino['mes'] = df_treino['time'].dt.month
df_treino = df_treino.dropna(subset=['tp_alvo', 'tp'])

# O Truque de Mestre: Comprimir a variância das tempestades
df_treino['tp_alvo_log'] = np.log1p(df_treino['tp_alvo'])
df_treino['tp_log'] = np.log1p(df_treino['tp']) # Comprimir a feature também

print("2. A calcular Climatologia Histórica Logarítmica...")
climatologia = df_treino.groupby(['lat', 'lon', 'mes'])['tp_alvo_log'].mean().reset_index()
climatologia = climatologia.rename(columns={'tp_alvo_log': 'tp_clima_log'})
df_treino = df_treino.merge(climatologia, on=['lat', 'lon', 'mes'], how='left')

del ds_tp, ds_alvo
gc.collect()

features = ['tp_log', 'lat', 'lon', 'mes', 'tp_clima_log']
target = 'tp_alvo_log' # O modelo agora prevê o Logaritmo

print("3. A treinar Ensemble em escala Logarítmica...")
modelo_lgb = lgb.LGBMRegressor(
    n_estimators=600, 
    learning_rate=0.03, 
    num_leaves=127,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)
modelo_lgb.fit(df_treino[features], df_treino[target])

modelo_xgb = xgb.XGBRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=9,
    tree_method='hist',
    random_state=42,
    n_jobs=-1
)
modelo_xgb.fit(df_treino[features], df_treino[target])

print("4. A preparar inferência...")
df_teste = ds_teste.to_dataframe().reset_index()
df_teste['mes'] = df_teste['time'].dt.month
df_teste['tp_log'] = np.log1p(df_teste['tp_ultima_obs'])
df_teste = df_teste.merge(climatologia, on=['lat', 'lon', 'mes'], how='left')
df_teste['tp_clima_log'] = df_teste['tp_clima_log'].fillna(0)

print("5. Inferência, Reversão do Logaritmo e Clipping...")
prev_lgb_log = modelo_lgb.predict(df_teste[features])
prev_xgb_log = modelo_xgb.predict(df_teste[features])

prev_final_log = (prev_lgb_log + prev_xgb_log) / 2.0

# Desfazer a transformação logarítmica para devolver mm/dia reais
previsoes_finais = np.expm1(prev_final_log)
previsoes_finais = np.clip(previsoes_finais, a_min=0, a_max=None)

df_submissao = pd.read_csv(os.path.join(base_path, "sample_submission.csv"))
df_submissao['tp_mm_day'] = previsoes_finais
caminho_saida = "/kaggle/working/submissao_mestre_log.csv"
df_submissao.to_csv(caminho_saida, index=False)
print(f"Ficheiro de elite gerado com sucesso em: {caminho_saida}")
