# WorCAP 2026: Previsão Mensal de Precipitação na América do Sul

Repositório oficial da solução desenvolvida para o hackathon do WorCAP 2026, com foco na previsão da média mensal de precipitação através de modelos de aprendizagem automática sobre dados de reanálise ERA5.

## Estrutura do Projeto
* `codigo_modelo.py`: Script principal contendo o pipeline de pré-processamento tabular, o treino do estimador e a geração das previsões.
* `submissao_baseline_worcap.csv`: Ficheiro de submissão gerado para avaliação na plataforma.

## Requisitos do Ambiente
O modelo foi executado num ambiente Python com as seguintes bibliotecas principais instaladas:
* `xarray`: Manipulação de datasets multidimensionais em formato NetCDF.
* `pandas`: Processamento de estruturas tabulares.
* `lightgbm`: Algoritmo de reforço de gradiente baseado em árvores de decisão para regressão.

## Instruções de Execução
1. Certifique-se de que os ficheiros de dados fornecidos pela organização estão acessíveis no diretório de entrada.
2. Execute o script de treino e inferência para reprocessar os dados históricos e gerar o ficheiro CSV final.
3. Submeta o resultado na plataforma Kaggle conforme as regras estabelecidas.
