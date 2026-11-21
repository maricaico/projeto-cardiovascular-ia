# Projeto Cardiovascular — DCT1401 Inteligência Artificial

**Disciplina:** DCT1401 — Inteligência Artificial  
**Instituição:** Universidade Federal do Rio Grande do Norte (UFRN)  
**Curso:** Bacharelado em Sistemas de Informação  
**Período:** 2026.2  
**Aluna:** Mariana dos Santos Batista Medeiros  
**Matrícula:** 20230083988

## Atividades do repositório

| Atividade                                                     | Conteúdo                                                                                                                       | Onde está                                                               |
| ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------- |
| **Atividade 01** — Análise Exploratória e Preparação de Dados | EDA, limpeza, pré-processamento e percepções para regressão e classificação                                                    | raiz do repositório (este README) — [relatório](docs/relatorio_final.pdf)                           |
| **Atividade 02** — Avaliação Comparativa de Modelos           | Sistema Especialista, Árvore de Decisão, Random Forest e Gradient Boosting; métricas, generalização e eficiência computacional | pasta [`atividade-02/projeto-cardiovascular-ia-02/`](atividade-02/projeto-cardiovascular-ia-02/) — [README](atividade-02/projeto-cardiovascular-ia-02/README.md) e [relatório](atividade-02/projeto-cardiovascular-ia-02/docs/relatorio_final.pdf) |
| **Atividade 03** — Redes Neurais Artificiais com PyTorch | Perceptrons Multicamadas (MLP) para classificação e regressão; estudo de hiperparâmetros, salvamento e inferência, generalização e eficiência computacional | pasta [`atividade-03/projeto-cardiovascular-ia-03/`](atividade-03/projeto-cardiovascular-ia-03/) — [README](atividade-03/projeto-cardiovascular-ia-03/README.md) e [relatório](atividade-03/projeto-cardiovascular-ia-03/docs/relatorio_final.pdf) |

Cada atividade possui seu próprio `requirements.txt` e suas próprias instruções de execução. As seções a seguir descrevem a **Atividade 01**; para as demais, consulte o [README da Atividade 02](atividade-02/projeto-cardiovascular-ia-02/README.md) e o [README da Atividade 03](atividade-03/projeto-cardiovascular-ia-03/README.md).

---

## Atividade 01 — Análise Exploratória e Preparação de Dados para Modelos de Aprendizado de Máquina

## Dataset

**Cardiovascular Disease Risk Prediction Dataset**

Fonte: Kaggle  
[https://www.kaggle.com/datasets/mobeenfatimah/cardiovascular-disease-risk-prediction-dataset](https://www.kaggle.com/datasets/mobeenfatimah/cardiovascular-disease-risk-prediction-dataset)

O arquivo utilizado no projeto deve permanecer em:

```text
data/raw/cardiovascular_disease_prediction.csv
```

## Descrição do projeto

A Atividade 01 contém a análise exploratória, a limpeza e o pré-processamento de um conjunto de dados relacionado a risco cardiovascular, além das percepções obtidas para dois problemas de aprendizado supervisionado:

- **Regressão:** utilização de `Risk_Score` como variável alvo;
- **Classificação:** utilização de `Cardiovascular_Disease` como variável alvo.

O trabalho foi desenvolvido como atividade prática avaliativa da disciplina DCT1401 — Inteligência Artificial. O objetivo é compreender o conjunto de dados, tratar problemas de qualidade, preparar as variáveis para uso em algoritmos de Machine Learning e identificar features com maior associação aos targets escolhidos.

Esta etapa do projeto não realiza o treinamento definitivo dos modelos. As análises produzidas servem para orientar etapas posteriores de modelagem.

## Estrutura do repositório

```text
.
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   └── cardiovascular_disease_prediction.csv
│   └── processed/
│       ├── dataset_clean.csv
│       ├── dataset_regression.csv
│       └── dataset_classification.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_regression_analysis.ipynb
│   └── 04_classification_analysis.ipynb
├── reports/
│   └── figures/
├── docs/
│   └── relatorio_final.pdf
├── atividade-02/
│   └── projeto-cardiovascular-ia-02/
│       ├── README.md
│       ├── requirements.txt
│       ├── notebooks/
│       ├── data/
│       ├── src/
│       ├── rules/
│       ├── models/
│       ├── logs/
│       ├── reports/
│       └── docs/
│           └── relatorio_final.pdf
└── atividade-03/
    └── projeto-cardiovascular-ia-03/
        ├── README.md
        ├── requirements.txt
        ├── notebooks/
        ├── data/
        ├── src/
        ├── models/
        ├── logs/
        ├── reports/
        └── docs/
            └── relatorio_final.pdf
```

## Etapas desenvolvidas

### 1. Análise Exploratória de Dados

O notebook `01_eda.ipynb` realiza a inspeção inicial do dataset, incluindo:

- visualização das primeiras e últimas linhas;
- identificação das dimensões do conjunto de dados;
- identificação dos tipos das variáveis;
- verificação de registros duplicados;
- identificação da quantidade e do percentual de valores ausentes;
- visualização dos valores ausentes;
- detecção de possíveis outliers pelo método IQR;
- visualização dos outliers com boxplots.

Seguindo a sequência metodológica adotada no projeto, as estatísticas descritivas, distribuições e correlações são calculadas após o tratamento dos dados.

### 2. Limpeza e Pré-processamento

O notebook `02_preprocessing.ipynb` realiza:

- reconstrução dos valores ausentes de `BMI` a partir de peso e altura;
- imputação pela mediana em `HDL`, `LDL`, `Triglycerides`, `Blood_Glucose` e `Sleep_Hours`;
- imputação pela moda em `Physical_Activity` e `Diet_Quality`;
- reavaliação dos outliers pelo método IQR;
- manutenção dos valores extremos quando não havia evidência suficiente de erro;
- cálculo das estatísticas descritivas após o tratamento;
- análise de correlação de Pearson;
- visualização da matriz de correlação por heatmap;
- codificação das variáveis categóricas binárias;
- codificação das variáveis ordinais preservando a ordem das categorias;
- One-Hot Encoding da variável nominal `Region`;
- padronização de 16 variáveis quantitativas com `StandardScaler`;
- remoção de `Patient_ID`;
- geração dos datasets destinados às análises posteriores.

Ao final são produzidos:

```text
data/processed/dataset_clean.csv
data/processed/dataset_regression.csv
data/processed/dataset_classification.csv
```

### 3. Percepções para Regressão

O notebook `03_regression_analysis.ipynb` utiliza `Risk_Score` como target de regressão.

As seis features com maior associação linear com o target foram:

- `Diabetes`;
- `Hypertension`;
- `ECG_Result`;
- `Blood_Glucose`;
- `Chest_Pain`;
- `Blood_Pressure_Systolic`.

As correlações individuais prioritárias ficaram aproximadamente entre 0,26 e 0,40. Como nenhuma delas apresenta isoladamente correlação linear muito elevada com o target, relações não lineares e combinações entre diferentes variáveis podem ser relevantes em etapas futuras.

Algoritmos sugeridos para avaliação posterior:

- Regressão Linear;
- Ridge / Lasso;
- Random Forest Regressor;
- Gradient Boosting Regressor.

Métricas sugeridas:

- MAE;
- RMSE;
- R².

### 4. Percepções para Classificação

O notebook `04_classification_analysis.ipynb` utiliza `Cardiovascular_Disease` como target de classificação.

A distribuição observada foi:

- **Classe 0:** 13.880 registros — 69,4%;
- **Classe 1:** 6.120 registros — 30,6%.

Foram utilizados:

- ANOVA para as features quantitativas;
- teste Qui-Quadrado para as features categóricas;
- boxplots e gráficos de distribuição por classe.

As features prioritárias foram:

- `Diabetes`;
- `Hypertension`;
- `Chest_Pain`;
- `ECG_Result`;
- `Blood_Glucose`;
- `Blood_Pressure_Systolic`.

Algoritmos sugeridos para avaliação posterior:

- Regressão Logística;
- Random Forest Classifier;
- Gradient Boosting Classifier;
- SVM.

Devido ao desbalanceamento entre as classes, recomenda-se que futuros modelos sejam avaliados utilizando, além de Accuracy:

- Precision;
- Recall;
- F1-Score;
- ROC-AUC.

## Principais resultados do pré-processamento

Após o tratamento:

- o conjunto permaneceu com 20.000 registros;
- não restaram valores ausentes nos datasets processados;
- não restaram colunas textuais nos arquivos destinados às análises;
- `Patient_ID` foi removido;
- as 16 variáveis quantitativas foram padronizadas com `StandardScaler`;
- `Risk_Score` e `Cardiovascular_Disease` foram preservados em suas representações originais.

Arquivos produzidos:

- `dataset_clean.csv` — 20.000 × 37;
- `dataset_regression.csv` — 20.000 × 36;
- `dataset_classification.csv` — 20.000 × 36.

## Visualizações geradas

Os gráficos são salvos em `reports/figures/`.

Arquivos gerados pelos notebooks:

```text
missing_values.png
outliers_boxplot.png
correlation_matrix.png
numeric_distributions.png
categorical_distributions.png
risk_score_distribution.png
correlation_with_risk_score.png
scatter_features_vs_risk.png
boxplot_risk_by_binary_features.png
class_distribution.png
boxplot_features_by_class.png
categorical_vs_disease.png
```

## Requisitos da Atividade 01

> As instruções desta seção e da seção seguinte valem **somente para a Atividade 01** (notebooks da pasta `notebooks/` na raiz). As **Atividades 02 e 03** são executadas no **Google Colab** e têm instruções próprias nos READMEs das respectivas pastas ([Atividade 02](atividade-02/projeto-cardiovascular-ia-02/README.md) e [Atividade 03](atividade-03/projeto-cardiovascular-ia-03/README.md)).

A Atividade 01 foi testada com **Python 3.11**. As versões utilizadas estão registradas no `requirements.txt` da raiz do repositório:

```text
pandas==3.0.5
numpy==2.4.6
matplotlib==3.11.1
seaborn==0.13.2
scipy==1.17.1
scikit-learn==1.9.1
ipykernel==7.3.0
```

## Como executar a Atividade 01

As instruções abaixo são compatíveis com Windows, Linux e macOS. A execução final da Atividade 01 foi validada em Windows com Python 3.11.

### 1. Clone o repositório

```bash
git clone https://github.com/maricaico/projeto-cardiovascular-ia.git
cd projeto-cardiovascular-ia
```

### 2. Crie um ambiente virtual

#### Windows

No PowerShell ou terminal do VS Code:

```powershell
py -m venv .venv
```

Ative o ambiente no PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Ou, no Prompt de Comando:

```cmd
.venv\Scripts\activate.bat
```

#### Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Em algumas distribuições Linux pode ser necessário instalar previamente o suporte a ambientes virtuais do Python.

#### macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Após a ativação, o terminal normalmente passa a exibir `(.venv)` no início da linha.

### 3. Instale as dependências

Com o ambiente virtual ativado:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure o ambiente de notebooks

Os notebooks podem ser executados em ambientes compatíveis com Jupyter.

No **Visual Studio Code**, recomenda-se instalar as extensões:

- Python — Microsoft;
- Jupyter — Microsoft.

Em seguida, abra cada notebook e selecione como kernel o interpretador Python do ambiente `.venv`.

Também é possível utilizar outras interfaces compatíveis com notebooks Jupyter, desde que o ambiente criado possua as dependências listadas em `requirements.txt`.

### 5. Execute os notebooks na ordem

A ordem de execução deve ser preservada:

1. `notebooks/01_eda.ipynb`
2. `notebooks/02_preprocessing.ipynb`
3. `notebooks/03_regression_analysis.ipynb`
4. `notebooks/04_classification_analysis.ipynb`

Fluxo do projeto:

```text
01_eda.ipynb
      ↓
02_preprocessing.ipynb
      ↓
03_regression_analysis.ipynb
      ↓
04_classification_analysis.ipynb
```

O Notebook 01 realiza a inspeção inicial do conjunto bruto.

O Notebook 02 realiza a limpeza e o pré-processamento e gera os três arquivos da pasta `data/processed/`.

O Notebook 03 utiliza `dataset_regression.csv`.

O Notebook 04 utiliza `dataset_classification.csv`.

### 6. Verifique os arquivos gerados

Após a execução completa, devem existir:

```text
data/processed/dataset_clean.csv
data/processed/dataset_regression.csv
data/processed/dataset_classification.csv
```

Também devem ser geradas as 12 figuras listadas anteriormente em:

```text
reports/figures/
```

## Relatório técnico

O relatório final da Atividade 01 está disponível em [`docs/relatorio_final.pdf`](docs/relatorio_final.pdf).

---

## Atividade 02 — Avaliação Comparativa de Modelos

A Atividade 02 está na pasta `atividade-02/projeto-cardiovascular-ia-02/` e possui estrutura, dependências e instruções de execução próprias:

- **README e instruções de execução (Google Colab):** [`atividade-02/projeto-cardiovascular-ia-02/README.md`](atividade-02/projeto-cardiovascular-ia-02/README.md)
- **Relatório técnico:** [`atividade-02/projeto-cardiovascular-ia-02/docs/relatorio_final.pdf`](atividade-02/projeto-cardiovascular-ia-02/docs/relatorio_final.pdf)

Nela, foram desenvolvidos e comparados modelos de Sistema Especialista, Árvore de Decisão, Random Forest e Gradient Boosting, dando continuidade à análise da Atividade 01.

## Atividade 03 — Redes Neurais Artificiais com PyTorch

A Atividade 03 está na pasta `atividade-03/projeto-cardiovascular-ia-03/` e possui estrutura, dependências e instruções de execução próprias:

- **README e instruções de execução (Google Colab):** [`atividade-03/projeto-cardiovascular-ia-03/README.md`](atividade-03/projeto-cardiovascular-ia-03/README.md)
- **Relatório técnico:** [`atividade-03/projeto-cardiovascular-ia-03/docs/relatorio_final.pdf`](atividade-03/projeto-cardiovascular-ia-03/docs/relatorio_final.pdf)

Nela, foram desenvolvidas redes neurais do tipo Perceptron Multicamadas (MLP), em PyTorch, para os mesmos problemas de classificação e regressão, com estudo de hiperparâmetros, salvamento e inferência dos modelos, análise de generalização e de eficiência computacional, mantendo a divisão dos dados da Atividade 02.

## Repositório

**GitHub:** [https://github.com/maricaico/projeto-cardiovascular-ia](https://github.com/maricaico/projeto-cardiovascular-ia)