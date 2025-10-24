# Documentação Completa: Pipeline de Análise e Modelagem Preditiva

## 📋 Índice
- [Objetivo do Projeto](#objetivo-do-projeto)
- [Fase 1: Análise Exploratória e Pré-processamento Inicial](#fase-1-análise-exploratória-e-pré-processamento-inicial)
- [Fase 2: Pré-Processamento e Engenharia de Features](#fase-2-pré-processamento-e-engenharia-de-features)
- [Fase 3: Engenharia e Seleção de Features](#fase-3-engenharia-e-seleção-de-features)
- [Fase 4: Treinamento e Otimização de Modelos](#fase-4-treinamento-e-otimização-de-modelos)
- [Fase 5: Implantação e Inferência](#fase-5-implantação-e-inferência)
- [Referências Bibliográficas](#referências-bibliográficas)

## 🎯 Objetivo do Projeto

Este documento detalha um pipeline de Machine Learning de ponta a ponta, projetado para processar, analisar e construir modelos preditivos a partir de um conjunto de dados. O objetivo final é prever as variáveis **Target1**, **Target2**, e **Target3** com a maior acurácia possível, comparando a performance de diferentes algoritmos de regressão.

---

## 🧩 Fase 1: Análise Exploratória e Pré-processamento Inicial

### Propósito Geral

Esta fase inicial tem como objetivo transformar o conjunto de dados brutos em uma base de dados estruturada, limpa e bem compreendida. As atividades aqui realizadas são cruciais para diagnosticar a qualidade dos dados, identificar desafios e informar as estratégias de limpeza, engenharia de features e modelagem que serão aplicadas posteriormente.

Esta fase compreende duas atividades principais:
- **Análise Exploratória de Dados (EDA - Exploratory Data Analysis)**: Um processo investigativo para resumir as principais características dos dados, frequentemente com métodos visuais.
- **Limpeza de Dados (Data Cleaning)**: O processo de detectar e corrigir (ou remover) registros e colunas corrompidos, imprecisos ou irrelevantes do conjunto de dados.

### 📦 BLOCO 1: Importação de Bibliotecas

**Objetivo**: Carregar as ferramentas (bibliotecas) necessárias para manipulação, análise e visualização de dados.

**Descrição Detalhada**:
- `pandas`: A principal biblioteca para manipulação de dados em Python. É utilizada para criar e gerenciar DataFrames, que são estruturas de dados tabulares (semelhantes a planilhas) onde os dados do projeto residem.
- `numpy`: Fornece suporte para operações matemáticas e arrays multidimensionais, sendo a base para muitas funcionalidades do pandas.
- `matplotlib` e `seaborn`: Bibliotecas de visualização de dados usadas para criar gráficos e plots que ajudam a entender os padrões e as distribuições nos dados.
- `warnings.filterwarnings('ignore')`: Suprime mensagens de aviso que podem poluir a saída do notebook, útil para manter o foco nos resultados.
- `pd.set_option(...)`: Configurações do pandas para melhorar a exibição de DataFrames no ambiente de desenvolvimento, garantindo que mais colunas e linhas sejam visíveis.

**Justificativa Técnica**: A importação de bibliotecas é o ponto de partida padrão para qualquer script de análise de dados. A utilização deste conjunto de ferramentas, conhecido como a "Pilha Científica do Python" (Scientific Python Stack), é o padrão da indústria para projetos de Data Science.

### 📂 BLOCO 2: Carregamento de Dados

**Objetivo**: Ingerir o conjunto de dados brutos do arquivo fonte para a memória, dentro de um DataFrame do pandas.

**Descrição Detalhada**: A função `pd.read_excel('JogadoresV1.xlsx')` é chamada para ler o arquivo Excel especificado. O conteúdo é carregado em uma variável chamada `df`. Imediatamente após o carregamento, o atributo `.shape` do DataFrame é verificado para confirmar as dimensões (número de linhas e colunas), garantindo que a importação ocorreu como esperado.

**Justificativa Técnica**: Este é o primeiro passo prático do pipeline. A verificação das dimensões é uma validação inicial crucial para detectar problemas de leitura ou corrupção de arquivos.

### 🔍 BLOCO 3: Exploração Inicial (EDA)

**Objetivo**: Obter uma primeira visão geral da estrutura e do conteúdo do conjunto de dados.

**Descrição Detalhada**:
- `df.head()`: Exibe as primeiras 5 linhas do DataFrame. É usado para uma inspeção visual rápida do formato dos dados e dos nomes das colunas.
- `df.info()`: Fornece um resumo técnico conciso. Mostra o nome de cada coluna, o número de valores não nulos e o tipo de dado (dtype) inferido para cada uma.
- `df.describe()`: Gera estatísticas descritivas para todas as colunas numéricas, incluindo contagem, média, desvio padrão, valor mínimo, máximo e os quartis.

**Justificativa Técnica**: Este bloco é a essência da Análise Exploratória de Dados. `info()` é vital para identificar colunas com tipos de dados incorretos ou com valores ausentes. `describe()` é fundamental para detectar possíveis outliers (ex: valores mínimos ou máximos absurdos) e para entender a escala e a dispersão de cada variável.

### 🧹 BLOCO 3.5: Tratamento Inicial de Tipos e Colunas

**Objetivo**: Realizar a primeira rodada de limpeza, focada em remover dados irrelevantes e corrigir os tipos de dados.

**Descrição Detalhada**:
- **Remoção de Coluna**: A coluna 'Código de Acesso' é removida usando `df.drop()`.
- **Conversão de Tipo**: Uma lista de colunas que deveriam ser numéricas, mas podem conter texto, é processada. A função `pd.to_numeric` com o argumento `errors='coerce'` é aplicada. Este argumento é crucial: ele tenta converter cada valor para número; se falhar (ex: encontrar um texto), ele substitui o valor problemático por NaN (Not a Number), marcando-o para tratamento posterior.

**Justificativa Técnica**:
- Colunas de identificação, como códigos de acesso, não possuem valor preditivo e devem ser removidas para evitar que o modelo as interprete erroneamente como features.
- Garantir os tipos de dados corretos é fundamental. Algoritmos de Machine Learning exigem entradas numéricas. A estratégia `errors='coerce'` é uma abordagem segura e robusta para lidar com dados "sujos" sem interromper o fluxo de execução.

### 🔤 BLOCO 3.6: Identificação de Colunas Não-Numéricas

**Objetivo**: Programaticamente listar todas as colunas que não são numéricas após a limpeza inicial.

**Descrição Detalhada**: O método `df.select_dtypes(exclude=np.number)` filtra o DataFrame, selecionando apenas as colunas cujo tipo de dado não é numérico (como object, que geralmente armazena texto).

**Justificativa Técnica**: Este passo serve como uma verificação de sanidade para as etapas anteriores e cria uma lista de colunas que precisarão de tratamento especial mais adiante, como a aplicação de técnicas de encoding (BLOCO 16).

### 🎯 BLOCO 4: Análise das Variáveis-Alvo (Targets)

**Objetivo**: Investigar em profundidade as variáveis que o modelo de Machine Learning tentará prever.

**Descrição Detalhada**: Para cada uma das três colunas target, o código calcula estatísticas descritivas detalhadas (contagem de ausentes, mínimo, máximo, média, mediana, desvio padrão). Em seguida, um histograma é plotado para cada target, mostrando visualmente a distribuição de seus valores.

**Justificativa Técnica**: A análise do target é a etapa mais importante da EDA. A distribuição dos targets informa sobre:
- **Skewness (Assimetria)**: Se a distribuição for muito assimétrica, pode ser necessário aplicar transformações (como logarítmica) para ajudar o modelo.
- **Outliers**: A presença de valores extremos pode impactar negativamente o treinamento de alguns modelos.
- **Escala**: Entender a faixa de valores (mínimo e máximo) é crucial para interpretar as métricas de erro do modelo (como o RMSE).

### 📊 BLOCO 6 e 7: Diagnóstico de Qualidade de Dados

**Objetivo**: Quantificar sistematicamente dois dos problemas de qualidade de dados mais comuns: valores ausentes e valores negativos inválidos.

**Descrição Detalhada**:
- **Valores Ausentes**: O código calcula a contagem e o percentual de valores NaN para cada coluna e exibe as 20 colunas mais afetadas em uma tabela e em um gráfico de barras.
- **Valores Negativos**: O código itera sobre as colunas numéricas e conta quantos valores são menores que zero, novamente apresentando um resumo tabular e gráfico.

**Justificativa Técnica**: Esta análise é um diagnóstico aprofundado. Os resultados guiarão a estratégia de imputação (BLOCO 10.2), que é a técnica de preenchimento de valores ausentes. Identificar colunas com um percentual muito alto de ausência pode levar à decisão de removê-las completamente. A análise de negativos é crucial para colunas onde eles são semanticamente inválidos (ex: tempo, quantidade).

### 📈 BLOCO 8: Análise de Correlação

**Objetivo**: Identificar quais features possuem a mais forte relação linear com as variáveis-alvo.

**Descrição Detalhada**: Para cada target, o código calcula o Coeficiente de Correlação de Pearson entre ele e todas as outras features numéricas. Os resultados são ordenados pelo valor absoluto da correlação, e as 10 features mais correlacionadas são visualizadas em um gráfico de barras.

**Justificativa Técnica**: A correlação é uma das técnicas mais simples e eficazes para a seleção inicial de features (Feature Selection). Features com alta correlação (próxima de 1 ou -1) com o target são fortes candidatas a serem preditores importantes. Esta análise ajuda a focar os esforços de engenharia de features nas variáveis que já demonstram ter algum poder preditivo.

### 📋 BLOCO 9: Resumo da Análise e Conclusão da Fase

**Objetivo**: Consolidar as descobertas da fase de análise e formalizar a transição para a fase de tratamento e limpeza profunda.

**Descrição Detalhada**: O código cria um dicionário de resumo com as principais métricas do estado atual do dataset (número de colunas com problemas, etc.) e define as próximas etapas do pipeline.

**Justificativa Técnica**: Este bloco funciona como um "checkpoint" de qualidade e planejamento. Ele documenta o que foi aprendido durante a EDA e estabelece um plano de ação claro para as próximas fases, garantindo um fluxo de trabalho organizado e metodológico.

---

## 🛠️ Fase 2: Pré-Processamento e Engenharia de Features

### Propósito Geral

Com base no diagnóstico da Fase 1, esta fase executa a limpeza profunda e, mais importante, a criação de novas variáveis (features) para enriquecer o dataset. O objetivo é transformar os dados em um formato otimizado para o treinamento de modelos de Machine Learning.

### 🧩 BLOCO 10, 10.1 e 10.2: Imputação Robusta e Preservação de Informação

**Objetivo**: Tratar sistematicamente todos os valores ausentes, negativos e inválidos, utilizando estratégias que minimizem a perda de informação.

**Descrição Detalhada**:
- **Dados Categóricos (Bloco 10)**: Valores ausentes ou inválidos em colunas categóricas são substituídos pela moda (o valor mais frequente da coluna).
- **Dados Numéricos Negativos (Bloco 10.1)**: Valores negativos são substituídos pela mediana dos valores positivos. Crucialmente, uma nova coluna "flag" (`_nao_respondeu`) é criada para preservar a informação de que aquele dado era originalmente negativo.
- **Dados Numéricos Ausentes (Bloco 10.2)**: Colunas 100% vazias são removidas. Nos demais casos, valores NaN são preenchidos com a mediana da coluna, e uma "flag" (`_tinha_missing`) é criada.

**Justificativa Técnica**: O uso da mediana para imputação numérica é uma escolha robusta, pois não é afetada por outliers, ao contrário da média. A criação de colunas "flag" é uma técnica avançada que permite ao modelo aprender se a própria ausência ou negação de uma resposta é um padrão de comportamento preditivo, evitando a perda total dessa informação.

### 🎨 BLOCO 12 e 13: Processamento de Tipos de Dados Complexos

**Objetivo**: Converter tipos de dados não-numéricos, como cores e datas, em múltiplas features numéricas que um modelo possa interpretar.

**Descrição Detalhada**:
- **Cores (Bloco 12)**: Códigos hexadecimais são convertidos para o sistema RGB. A partir daí, novas features como brilho, saturação e flags binárias (`_eh_cinza`) são calculadas.
- **Datas (Bloco 13)**: A coluna de data/hora é usada para extrair componentes como `dia_semana`, `hora_dia`, `mes`, `periodo_dia` e `eh_fim_semana`.

**Justificativa Técnica**: Modelos de Machine Learning não conseguem interpretar um código de cor ou uma data diretamente. Este processo "descompacta" a informação contida nesses formatos, transformando-a em variáveis numéricas que podem revelar padrões, como a influência do período do dia no desempenho do jogador.

### ✅ BLOCO 14 e 15: Verificação de Qualidade e Correções Finais

**Objetivo**: Realizar um controle de qualidade final após as transformações e remover features que não agregam valor preditivo.

**Descrição Detalhada**: O código verifica sistematicamente se ainda restam valores ausentes, negativos ou infinitos. Em seguida, identifica e remove colunas com variância zero.

**Justificativa Técnica**: Uma coluna com variância zero contém o mesmo valor para todos os jogadores. Ela é, por definição, inútil para um modelo preditivo, pois não ajuda a diferenciar um resultado do outro. Removê-las é uma otimização que simplifica o modelo sem perda de informação.

### 🔢 BLOCO 16: Encoding de Features Categóricas

**Objetivo**: Converter todas as colunas de texto restantes para um formato numérico.

**Descrição Detalhada**: A técnica de Frequency Encoding é aplicada. Cada categoria de texto (ex: "Solteiro") é substituída por sua frequência de aparição no dataset (ex: 0.45, se 45% dos jogadores forem solteiros).

**Justificativa Técnica**: Esta é a etapa final para garantir que 100% das features de entrada do modelo sejam numéricas. O Frequency Encoding é uma técnica eficiente que captura a "popularidade" de uma categoria em uma única variável numérica, evitando a criação de múltiplas colunas como faria o One-Hot Encoding.

### 🧠 BLOCO 17, 18 e 19: Engenharia de Features Avançada

**Objetivo**: Criar novas features de alto valor a partir de combinações das variáveis existentes, baseando-se no conhecimento do problema.

**Descrição Detalhada**:
- **Desempenho (Bloco 17)**: Criação de métricas como `taxa_acerto`, `tempo_por_questao` e `evolucao_desempenho` (comparando rodadas).
- **Contexto (Bloco 18)**: Criação de scores para `qualidade_sono`, `nivel_social` (estímulo do ambiente) e `media_emocional` (a partir de escalas Likert).
- **Interação (Bloco 19)**: Combinação de features de diferentes domínios, como `desempenho_sono` (taxa de acerto * qualidade do sono).

**Justificativa Técnica**: Esta é a etapa mais crucial para aumentar o poder preditivo de um modelo. Enquanto as features originais descrevem o que aconteceu, as features de engenharia descrevem como e por que aconteceu. Features de interação, em particular, permitem que o modelo capture relações não-lineares complexas que não seriam visíveis de outra forma.

### 💾 BLOCO 20: Conclusão da Fase e Geração de Artefato de Dados

**Objetivo**: Finalizar o pré-processamento, realizar uma última verificação e salvar o dataset final e enriquecido.

**Descrição Detalhada**: O código executa um resumo final do estado do dataset (dimensões, tipos de features criadas, verificação de qualidade). Em seguida, o DataFrame processado é salvo em um arquivo `dados_processados.csv`.

**Justificativa Técnica**: Salvar o resultado em um arquivo CSV cria um artefato de dados. Isso é fundamental para a reprodutibilidade e eficiência. A fase de modelagem pode agora começar diretamente a partir deste arquivo limpo, sem precisar re-executar todo o pipeline de limpeza e engenharia de features a cada vez, garantindo consistência entre os experimentos.

### 🔍 BLOCO 21: Análise de Clusterização e Descoberta de Perfis

#### Propósito Geral

Esta fase representa uma transição crucial da limpeza de dados para a engenharia de features inteligente. Utilizando técnicas de aprendizado não supervisionado, o objetivo é investigar se os jogadores podem ser agrupados em perfis ou "clusters" com base em suas características e comportamentos. A descoberta de tais grupos permite criar uma nova e poderosa feature categórica (o ID do Cluster) que informa aos modelos preditivos a que "tribo" de comportamento um jogador pertence. Em vez de modelar uma população homogênea, passamos a modelar subgrupos distintos, o que pode aumentar drasticamente a acurácia.

#### 1. Preparação e Normalização dos Dados

**Objetivo**: Preparar o dataset limpo para ser analisado por algoritmos de clusterização baseados em distância.

**Descrição Detalhada**: O pipeline inicia com o `dados_processados.csv`. As colunas Target são removidas, pois a clusterização deve descobrir padrões inerentes aos dados de entrada, sem o viés da variável de saída. Em seguida, todas as features numéricas são selecionadas e normalizadas utilizando o `RobustScaler`.

**Justificativa Técnica**: A normalização é um passo mandatório para a clusterização. O RobustScaler foi escolhido especificamente por sua resistência a outliers. Ao centralizar os dados com a mediana e escalonar pelo intervalo interquartil, ele garante que features com valores extremos não dominem o cálculo de distância, resultando em clusters mais estáveis e significativos.

#### 2. Determinação da Estrutura Ótima (Número de Clusters e Algoritmo)

**Objetivo**: Encontrar o número mais natural de grupos (K) nos dados e o algoritmo que melhor os separa.

**Descrição Detalhada**: Uma abordagem multifacetada foi utilizada:
- **Análise de Métricas**: O pipeline testou um intervalo de K de 2 a 10. Para cada K, foram calculadas quatro métricas de qualidade: Inércia (Método do Cotovelo), Score de Silhueta, Davies-Bouldin e Calinski-Harabasz.
- **Votação**: Os resultados das métricas foram unânimes, com o Score de Silhueta (máximo), Davies-Bouldin (mínimo) e Calinski-Harabasz (máximo) todos apontando para K=2 como a solução ótima.
- **Comparação de Algoritmos**: Com K=2, três algoritmos foram comparados: K-Means, Hierarchical Clustering e Gaussian Mixture Models (GMM). O K-Means alcançou o maior Score de Silhueta (0.735), indicando a melhor qualidade de clusterização (clusters densos e bem separados).

**Justificativa Técnica**: Confiar em um único método para determinar K é arriscado. A utilização de um sistema de "votação" com múltiplas métricas que avaliam diferentes aspectos da qualidade do cluster (coesão, separação) torna a escolha de K=2 extremamente robusta. O valor de silhueta de 0.735 é considerado excelente, sugerindo que a divisão em dois grupos é uma estrutura muito forte e natural nos dados. A comparação de algoritmos valida que o K-Means, com suas premissas de clusters esféricos, é o mais adequado para a geometria destes dados.

#### 3. Caracterização e Interpretação dos Perfis

**Objetivo**: Traduzir os clusters matemáticos em perfis de jogadores compreensíveis e acionáveis.

**Descrição Detalhada**:
- **Distribuição**: Foram identificados dois grupos: Cluster 0 (majoritário), com 134 jogadores (77.5%), e Cluster 1 (minoritário), com 39 jogadores (22.5%).
- **Análise de Diferenças**: A análise das médias das features revelou um padrão claro e consistente. O Cluster 1 é fortemente caracterizado por valores significativamente mais altos em features relacionadas à percepção e escolha de cores, como saturação (`Cor0206_saturacao`), brilho (`F0207_brilho`) e o uso de cores primárias ou preto/branco (`F0207_eh_branco`, `Cor0208_eh_preto`). O Cluster 0, em contrapartida, apresenta valores baixos ou nulos nessas mesmas dimensões.
- **Visualização**: A redução de dimensionalidade com PCA confirmou visualmente a existência de dois grupos distintos e bem separados no espaço de features.

**Justificativa Técnica**: Um cluster só tem valor se for interpretável. A análise de caracterização demonstrou que a principal linha divisória entre os jogadores não é o tempo de resposta ou a taxa de acerto, mas sim uma dimensão de comportamento mais sutil ligada às interações com cores. Esta descoberta é uma forma de engenharia de features automática, onde o algoritmo revelou um padrão latente que seria difícil de hipotetizar manualmente.

#### 4. Validação de Estabilidade e Geração de Artefatos

**Objetivo**: Garantir que os clusters são robustos e salvar os resultados de forma que possam ser usados nas próximas fases do pipeline.

**Descrição Detalhada**:
- **Teste de Estabilidade (Bootstrap)**: O processo de clusterização foi repetido 30 vezes em amostras aleatórias dos dados. A estabilidade média de 83.3% (a frequência com que um ponto de dados permaneceu em seu cluster original) confirma que os perfis encontrados são altamente estáveis e não um resultado do acaso.
- **Geração de Artefatos**: O pipeline salvou três saídas essenciais:
  - `jogadores_clusterizados_v2.csv`: O dataset enriquecido com a nova coluna Cluster.
  - `clusters_metadados.csv`: Um resumo da composição dos clusters.
  - `clustering_artifacts_v2.pkl`: Um arquivo contendo o RobustScaler ajustado e o modelo KMeans treinado.

**Justificativa Técnica**: A validação de estabilidade confere um alto grau de confiança científica aos resultados. A geração do arquivo de artefatos `.pkl` é o passo mais crítico para a operacionalização. Ele permite que, na fase de inferência, um novo jogador passe pelo mesmo scaler e seja classificado pelo mesmo modelo K-Means, garantindo que o perfil atribuído seja consistente com a análise original. Isso transforma a clusterização de um exercício analítico em uma ferramenta preditiva reutilizável.

---

## ⚙️ Fase 3: Engenharia e Seleção de Features

### Propósito Geral

Esta fase transforma um dataset já limpo e clusterizado em múltiplos conjuntos de dados altamente otimizados, cada um especificamente ajustado para prever uma das variáveis-alvo. As atividades são divididas em duas grandes etapas:
- **Engenharia de Features**: Criação de novas colunas a partir das existentes para capturar interações complexas e padrões não-lineares.
- **Seleção de Features**: Utilização de um método híbrido e robusto para classificar a importância de todas as features e selecionar apenas as "melhores" para cada alvo.

### 🔬 BLOCO 1.1: A Função Híbrida de Seleção de Features (`select_best_features_hybrid`)

**Objetivo**: Avaliar a importância de cada feature em relação a um target específico, utilizando uma combinação ponderada de múltiplas técnicas, para produzir um ranking robusto e confiável.

**Descrição Detalhada**: Esta função não confia em um único método. Em vez disso, ela agrega a "opinião" de seis abordagens diferentes:
1. **Correlação de Pearson (Linear)**: Mede a relação linear entre a feature e o alvo.
2. **Correlação de Spearman (Monotônica)**: Mede a relação monotônica (se uma variável aumenta, a outra também, mas não necessariamente a uma taxa constante). É menos sensível a outliers.
3. **Informação Mútua** (`mutual_info_regression`): Captura qualquer tipo de relação estatística, incluindo as não-lineares.
4. **Teste F** (`f_regression`): Um teste estatístico que avalia se a feature tem uma relação linear significativa com o alvo.
5. **Importância de Features (XGBoost)**: Treina um modelo XGBoost e extrai a importância que o modelo atribuiu a cada feature para fazer suas previsões.
6. **Importância de Features (LightGBM & RandomForest)**: O mesmo processo, mas usando outros dois modelos baseados em árvores, cada um com suas próprias nuances.

As pontuações de cada método são normalizadas (colocadas em uma escala de 0 a 1) e combinadas em uma pontuação final ponderada. As features com as maiores pontuações combinadas são selecionadas.

**Código Crítico**: A essência da função está na agregação ponderada das pontuações normalizadas. Os pesos (ex: `0.30 * normalize(corr_series)`) foram definidos para dar mais importância aos métodos de correlação e à robustez do XGBoost.

```python
# Normaliza uma série de pontuações para uma escala de 0 a 1
normalize = lambda s: (s - s.min()) / (s.max() - s.min() + 1e-10)

# Combina as pontuações normalizadas de 6 métodos com pesos diferentes
combined_score = (0.30 * normalize(corr_series) +      # Correlações (Pearson+Spearman)
                  0.15 * normalize(mi_series) +        # Informação Mútua
                  0.15 * normalize(f_series) +         # Teste F
                  0.20 * normalize(xgb_series) +       # Importância do XGBoost
                  0.10 * normalize(lgb_series) +       # Importância do LightGBM
                  0.10 * normalize(rf_series))         # Importância do RandomForest

# Seleciona as N features com a maior pontuação combinada
selected = combined_score.nlargest(n_features).index.tolist()
return selected
```

**Justificativa Técnica**: Confiar em um único método de seleção de features é arriscado. Um método pode ser enviesado para um certo tipo de relação (ex: correlação só pega relações lineares). Ao combinar múltiplas perspectivas, a seleção se torna muito mais robusta. Métodos baseados em modelos (como XGBoost) são particularmente poderosos porque avaliam a utilidade de uma feature no contexto de outras, capturando interações.

**Referência e Conceito**:
- **Ensemble Feature Selection**: O conceito de combinar múltiplos métodos de seleção para obter um resultado mais estável e confiável. Este princípio é análogo aos modelos de ensemble (como o Random Forest), que combinam múltiplos modelos fracos para criar um modelo forte. [Referência: Saeys, Y., Inza, I., & Larrañaga, P. (2007). A review of feature selection techniques in bioinformatics. Bioinformatics, 23(19), 2507-2517.]

### 🏗️ BLOCO 1.2: O Pipeline Principal (`feature_engineering_pipeline`)

Este pipeline organiza todo o processo em quatro etapas sequenciais.

#### Etapa 1: Carregamento e Preparação Inicial

**Objetivo**: Carregar os dados clusterizados e prepará-los para o processamento, separando features (X) e alvos (y).

**Descrição Detalhada**: Carrega o arquivo `jogadores_clusterizados_v2.csv` e remove as linhas onde qualquer um dos targets é nulo, garantindo que o pipeline trabalhe apenas com dados completos e utilizáveis para treinamento supervisionado.

#### Etapa 2: Limpeza Final e Estabilização Numérica

**Objetivo**: Garantir que o dataset esteja numericamente estável e livre de redundância.

**Descrição Detalhada**:
- **Imputação Final**: Preenche quaisquer valores nulos restantes com a mediana da coluna, e depois com 0, para garantir que não haja NaNs.
- **Tratamento de Infinitos**: Substitui valores `np.inf` (que podem surgir de divisões por zero) por 0.
- **Remoção de Variância Zero**: Remove colunas que não têm variação (todos os valores são iguais), pois não carregam informação preditiva.
- **Remoção de Alta Colinearidade**: Calcula a matriz de correlação entre todas as features. Se duas features tiverem uma correlação maior que 0.97, uma delas é removida.

**Justificativa Técnica**:
- **Colinearidade**: Manter duas features que são quase idênticas (alta correlação) é redundante. Isso pode desestabilizar alguns modelos (especialmente regressões lineares) e aumentar a complexidade do modelo sem adicionar nova informação. Remover uma delas simplifica o problema. A escolha de 0.97 é um limiar estrito, mas comum, para remover apenas a redundância mais óbvia.

**Referência e Conceito**:
- **Multicollinearity**: Um fenômeno no qual uma variável preditora em um modelo de regressão pode ser prevista linearmente a partir das outras com um grau substancial de precisão. [Referência: "A Beginner's Guide to Multicollinearity and VIF," da Statology.]

#### Etapa 3: Engenharia de Features Avançada

**Objetivo**: Criar novas features inteligentes para aumentar o poder preditivo do dataset.

**Descrição Detalhada**:
- **One-Hot Encoding do Cluster**: A coluna categórica Cluster é transformada em múltiplas colunas binárias (ex: `Cluster_0`, `Cluster_1`).
- **Interações com o Cluster**: Para cada feature numérica, uma nova feature é criada calculando a diferença entre o valor do jogador e a média do cluster ao qual ele pertence (ex: `T01_vs_cluster_mean`).
- **Agregações por Prefixo**: Para grupos de colunas com o mesmo prefixo (ex: `Q0...`, `P...`), são calculadas estatísticas agregadas como média, desvio padrão e range.
- **Transformações Não-Lineares**: Para as 10 features mais correlacionadas com cada target, são criadas novas features aplicando transformações de quadrado (`**2`) e raiz quadrada (`sqrt`).

**Justificativa Técnica**:
- **One-Hot Encoding**: É a maneira padrão de representar uma variável categórica para modelos de machine learning, permitindo que o modelo aprenda pesos específicos para cada categoria de cluster.
- **Interações com Cluster**: Esta é uma técnica poderosa. A feature `T01_vs_cluster_mean` não mede apenas o tempo T01, mas mede o quão anormal é o tempo daquele jogador em relação ao seu próprio grupo. Isso normaliza o comportamento e pode ser um preditor muito mais forte.
- **Transformações Não-Lineares**: Muitos modelos (especialmente os lineares) têm dificuldade em capturar relações não-lineares. Adicionar o quadrado de uma feature (`X**2`) permite que um modelo linear se ajuste a uma parábola, aumentando sua flexibilidade e poder de captura de padrões complexos.

#### Etapa 4: Seleção Híbrida de Features por Target

**Objetivo**: Aplicar a função de seleção híbrida para cada target, selecionando um número diferente de features conforme a estratégia definida.

**Descrição Detalhada**: O código itera sobre os três targets. Para Target1 e Target3, ele chama a função `select_best_features_hybrid` pedindo as 120 melhores features. Para Target2, ele adota uma estratégia diferente, pedindo as 180 melhores.

**Justificativa Técnica**: Esta é a implementação da estratégia adaptativa. A hipótese é que Target2 pode ser um fenômeno mais complexo ou sutil, beneficiando-se de um conjunto maior e mais diversificado de preditores. Em vez de usar um único conjunto de features para todos os modelos, o pipeline cria um "time de especialistas" de features para cada tarefa de previsão.

### 🚀 Execução e Saídas (`if __name__ == '__main__':`)

**Objetivo**: Orquestrar a execução do pipeline e salvar os resultados em arquivos para uso na próxima fase (modelagem).

**Descrição Detalhada**:
- Executa a função `feature_engineering_pipeline`.
- Salva o DataFrame completo com todas as features de engenharia em `X_engineered.csv`.
- Salva os targets em `y_targets.csv`.
- Salva as listas de features selecionadas para cada target em `selected_features.json`.
- Salva um relatório de todas

Referências Bibliográficas
Saeys, Y., Inza, I., Larraaga, P. (2007). A review of feature selection techniques in bioinformatics. Bioinformatics, 23(19), 2507-2517.
Bergstra, J., Bengio, Y. (2012). Random Search for Hyper-Parameter Optimization. Journal of Machine Learning Research, 13, 281-305.
Wolpert, D. H. (1992). Stacked generalization. Neural networks, 5(2), 241-259.
Documentação de Scikit-learn sobre Model Persistence.
Artigos e conceitos gerais sobre multicolinearidade, validação cruzada e otimização de hiperparâmetros (mencionados no conteúdo).
