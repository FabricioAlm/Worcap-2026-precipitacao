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

## Documentação do Modelo de Linha de Base (Score: 3.17086)

Raciocínio Principal: O foco desta etapa não foi a máxima precisão meteorológica, mas sim a validação do funil de dados ponta a ponta (Produto Mínimo Viável - MVP). O objetivo era garantir o carregamento correto das matrizes espaciais, o treino sem estourar o limite de RAM da máquina virtual do Kaggle e a formatação exata do ficheiro CSV de saída exigido pela organização.

Técnica Aplicada e Justificativa:

Algoritmo (LightGBM): Utilizou-se regressão baseada em árvores de decisão com gradiente reforçado (LGBMRegressor). A escolha deve-se à sua elevada velocidade de convergência e eficiência de memória no processamento de dezenas de milhões de linhas, superando métodos tradicionais.

Seleção de Variáveis: O modelo foi alimentado com um espaço de características ultrarreduzido: apenas lat (latitude), lon (longitude) e tp (precipitação do mês anterior). A justificativa foi manter a estabilidade do hardware.

Processamento de Dados: As matrizes multidimensionais NetCDF foram planificadas diretamente para tabelas Pandas (to_dataframe().reset_index()), tratando cada pixel no espaço e no tempo como uma linha independente, alinhando o alvo (tp_alvo) diretamente pela ordem estrutural dos índices.

Avaliação do Resultado: O erro quadrático médio de 3.17086 foi inteiramente dentro do esperado. A pontuação prova que a coordenada espacial aliada à chuva recente tem poder preditivo funcional. O erro manteve-se na faixa dos 3.0 porque o algoritmo trabalhou sem contexto sazonal (ignorando a época do ano) e sem variáveis atmosféricas de transporte de humidade.

## Documentação do Modelo Avançado (Score: 3.07595)

Raciocínio Principal: Após a validação do pipeline de dados com o modelo de base, o objetivo desta iteração foi incorporar o contexto físico e atmosférico ao algoritmo para reduzir o erro de previsão. A precipitação não é um fenómeno isolado; ela é impulsionada pelo calor (que gera evaporação e convecção) e pelos ventos (que transportam a humidade). O maior desafio técnico foi contornar o limite de memória RAM do Kaggle ao fundir múltiplas matrizes tridimensionais gigantescas num único formato tabular.

Técnica Aplicada e Justificativa:

Engenharia de Variáveis (Física): Adicionaram-se as variáveis t2 (Temperatura a 2 metros) e u_850 (Componente zonal do vento a 850 hPa). A temperatura ajuda o modelo a mapear zonas de alta evaporação, enquanto o vento indica o deslocamento de massas de ar.

Gestão Extrema de Memória: Em vez de usar fusões nativas (merge ou join) que duplicam dados em memória e causam falhas no Kaggle, o código planificou cada NetCDF separadamente e extraiu apenas a coluna estritamente necessária (em formato 1D). Em seguida, utilizou-se del e o coletor de lixo do Python (gc.collect()) para libertar agressivamente a memória antes do treino.

Otimização de Hiperparâmetros (LightGBM): A arquitetura da floresta de decisão foi aprofundada. O número de árvores (n_estimators) subiu para 300, e a taxa de aprendizagem (learning_rate) foi reduzida para 0.05 para garantir uma convergência mais suave. Os limites de complexidade (num_leaves=63 e max_depth=8) foram ajustados para permitir que o modelo compreendesse as interações não lineares entre vento, temperatura e chuva, sem sobreajustar (overfitting).

Avaliação do Resultado: O resultado foi exatamente o esperado. A pontuação melhorou de 3.17086 para 3.07595. A injeção de termodinâmica e cinemática deu ao algoritmo um poder de diferenciação real. No entanto, a melhoria foi linear e não exponencial, confirmando que o modelo ainda sofre de "amnésia sazonal" — ele tenta adivinhar a chuva apenas com a foto do momento, sem saber em que mês do ano está ou qual é a média histórica exata para aquela coordenada geográfica.

## 
