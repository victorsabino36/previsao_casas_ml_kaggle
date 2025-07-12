# Previsão de Preços de Casas em Ames, Iowa (Kaggle)

## Visão Geral do Projeto

Este projeto aborda o desafio "House Prices - Advanced Regression Techniques" do Kaggle, que consiste em prever o preço final de venda de casas em Ames, Iowa. O conjunto de dados oferece 79 variáveis explicativas que descrevem quase todos os aspectos das propriedades residenciais, proporcionando uma excelente oportunidade para praticar técnicas de pré-processamento de dados e modelagem de regressão.

## Objetivo

O principal objetivo é construir um modelo de Machine Learning capaz de prever com precisão o `SalePrice` (Preço de Venda) de cada casa, minimizando o erro logarítmico quadrático médio da raiz (RMSLE).

## Conjunto de Dados

O conjunto de dados `Ames Housing` foi compilado por Dean De Cock e é uma alternativa moderna e expandida ao dataset Boston Housing. Ele inclui informações detalhadas sobre diversas características das casas, como área, qualidade geral, condição, número de quartos, características do porão e garagem, entre outras.

## Metodologia

O desenvolvimento deste projeto seguiu as seguintes etapas:

1.  **Análise Exploratória de Dados (EDA):** Carregamento e inspeção inicial dos dados (`df.info()`, `df.head()`).
2.  **Tratamento de Valores Ausentes:** Identificação e tratamento de valores `NaN`. Colunas com alta porcentagem de valores ausentes (`PoolQC`, `MiscFeature`, `Alley`, `Fence`) foram removidas. Outras colunas categóricas com `NaNs` foram preenchidas com 'None' (indicando ausência) e colunas numéricas com 0 ou a mediana/moda (calculada a partir dos dados de treino).
3.  **Engenharia de Features Categóricas:**
    * **Mapeamento Ordinal:** Colunas categóricas com ordem intrínseca (ex: `ExterQual`, `BsmtQual`, `FireplaceQu`) foram mapeadas para valores numéricos, refletindo sua hierarquia.
    * **One-Hot Encoding:** As demais colunas categóricas (nominais) foram convertidas usando `pd.get_dummies` para criar variáveis binárias.
4.  **Transformação da Variável Alvo (`SalePrice`):** A distribuição da variável `SalePrice` mostrou-se assimétrica. Para melhorar o desempenho do modelo de regressão e alinhar com a métrica de avaliação (RMSLE), foi aplicada uma transformação logarítmica (`np.log1p`).
5.  **Divisão dos Dados:** O conjunto de dados foi dividido em conjuntos de treino e teste (`X_train`, `X_test`, `y_train`, `y_test`) para treinamento e avaliação do modelo.
6.  **Modelagem:** Um modelo de Regressão **XGBoost (`XGBRegressor`)** foi utilizado para as previsões.

## Resultados

Após o treinamento do modelo e a geração das previsões no conjunto de teste da competição:

* **RMSE no Conjunto de Teste (Escala Original):** ~27229.16 (Este valor é do seu teste local, não o do Kaggle)
* **R-squared no Conjunto de Teste (Escala Original):** ~0.90 (Este valor é do seu teste local, não o do Kaggle)
* **Pontuação no Kaggle (RMSLE):** `0.14316`

A pontuação de 0.14316 no RMSLE é um resultado sólido e promissor para uma primeira submissão, indicando que o modelo capturou bem os padrões dos dados.

## Habilidades e Ferramentas

* **Linguagem de Programação:** Python
* **Bibliotecas:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`
* **Ambiente:** Jupyter Notebook (no ambiente Kaggle)
* **Habilidades:** Pré-processamento de Dados, Tratamento de Valores Ausentes, Engenharia de Features (Categóricas e Ordinais), Transformação de Variáveis, Modelagem de Regressão (XGBoost), Avaliação de Modelos.

## Como Reproduzir o Projeto

1.  Clone este repositório para sua máquina local.
2.  Certifique-se de ter Python e as bibliotecas listadas acima instaladas.
3.  Baixe os conjuntos de dados `train.csv` e `test.csv` da [competição House Prices no Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data).
4.  Coloque os arquivos `.csv` na pasta `/kaggle/input/house-prices-advanced-regression-techniques/` ou ajuste os caminhos no notebook.
5.  Abra o arquivo `.ipynb` (seu notebook Jupyter) e execute as células sequencialmente.

## Agradecimentos

* Ao Kaggle pela plataforma e pelo desafio.
* Ao Dean De Cock pela compilação do excelente conjunto de dados Ames Housing.

---
