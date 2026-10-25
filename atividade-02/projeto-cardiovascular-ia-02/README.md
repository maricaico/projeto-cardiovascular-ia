# Atividade 02 — Avaliação Comparativa de Modelos

**Sistema Especialista, Árvore de Decisão e Ensemble Learning**

**Disciplina:** DCT1401 — Inteligência Artificial  
**Instituição:** Universidade Federal do Rio Grande do Norte (UFRN) — CERES  
**Curso:** Bacharelado em Sistemas de Informação  
**Período:** 2026.2  
**Professor:** Prof. Dr. Thommas K. S. Flores  
**Aluna:** Mariana dos Santos Batista Medeiros  
**Matrícula:** 20230083988  

**Dataset:** [Cardiovascular Disease Risk Prediction Dataset (Kaggle)](https://www.kaggle.com/datasets/mobeenfatimah/cardiovascular-disease-risk-prediction-dataset)  
**Relatório técnico:** [`docs/relatorio_final.pdf`](docs/relatorio_final.pdf)

---

## Descrição

Esta atividade dá continuidade à [Atividade 01](../../README.md) e compara três famílias de modelos de Inteligência Artificial em dois problemas supervisionados:

- **Classificação:** `Cardiovascular_Disease` (0 = ausência, 1 = presença);
- **Regressão:** `Risk_Score`.

Foram construídos sete modelos:

| Família | Classificação | Regressão |
|---|:---:|:---:|
| Sistema Especialista (regras SE–ENTÃO) | ✔ | — ¹ |
| Árvore de Decisão | ✔ | ✔ |
| Random Forest | ✔ | ✔ |
| Gradient Boosting | ✔ | ✔ |

¹ Por orientação do professor, o Sistema Especialista foi desenvolvido somente para classificação.

Os modelos foram avaliados quanto a desempenho preditivo, generalização (gap, validação cruzada, viés e variância), interpretabilidade, esforço de construção e eficiência computacional (tempo, memória, armazenamento, energia e CO₂).

## Protocolo experimental

- Divisão treino/teste de 80%/20% (16.000/4.000 registros), com estratificação pela variável de classificação e `random_state=42`;
- Leitura do CSV original e divisão dos dados antes das transformações. Nos modelos de aprendizado de máquina, o pré-processamento (imputação e codificação) é realizado por meio de `Pipelines`, ajustados somente no treinamento. O Sistema Especialista possui preparação própria, também fundamentada apenas nos dados de treinamento, para evitar *data leakage*;
- Seis preditores comuns a todos os modelos: `Diabetes`, `Hypertension`, `Chest_Pain`, `ECG_Result`, `Blood_Glucose` e `Blood_Pressure_Systolic`;
- Seleção dos modelos de aprendizado de máquina por `GridSearchCV` com 5 folds (F1-score na classificação e RMSE na regressão); a base de regras do Sistema Especialista é definida separadamente;
- O conjunto de teste foi usado apenas na avaliação final.

## Estrutura do projeto

No GitHub, este projeto está em `atividade-02/projeto-cardiovascular-ia-02/`. A árvore abaixo parte dessa pasta.

```text
projeto-cardiovascular-ia-02/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   └── cardiovascular_disease_prediction.csv   # dataset original
│   └── processed/                                  # bases da Atividade 01 (não utilizadas como entrada dos modelos finais)
├── notebooks/
│   ├── 01_eda.ipynb                     # Atividade 01 (referência)
│   ├── 02_preprocessing.ipynb           # Atividade 01 (referência)
│   ├── 03_expert_system.ipynb           # regras, motor de inferência e medições
│   ├── 04_decision_tree.ipynb           # Árvores de Decisão
│   ├── 05_ensemble.ipynb                # Random Forest e Gradient Boosting
│   ├── 06_evaluation_metrics.ipynb      # métricas, ROC, PR e resíduos
│   ├── 07_generalization.ipynb          # gap, K-fold, viés e variância
│   └── 08_computational_efficiency.ipynb  # tempo, memória, energia, CO₂ e tabela-resumo
├── src/
│   ├── __init__.py
│   ├── pipeline_data.py      # carregamento, divisão e pré-processamento
│   ├── expert_system.py      # base de conhecimento e motor de inferência
│   ├── models.py             # construção e ajuste dos modelos
│   ├── metrics.py            # cálculo das métricas
│   └── profiling.py          # medição de tempo, memória e energia
├── rules/
│   └── regras_classificacao.txt   # 11 regras SE–ENTÃO
├── models/                   # seis Pipelines serializados (.joblib)
├── logs/
│   ├── cronometro_regras.csv # sessões de elaboração das regras
│   └── medicoes_*.csv        # medições de tempo e memória
├── reports/
│   ├── tables/               # tabelas CSV (inclui resumo_final_*.csv)
│   └── figures/              # gráficos
└── docs/
    └── relatorio_final.pdf
```

## Como executar (Google Colab)

Os notebooks foram desenvolvidos e executados no **Google Colab**. O código de configuração de cada notebook procura a pasta do projeto neste caminho:

```text
Meu Drive/projeto-cardiovascular-ia-02/
```

**Importante:** embora o projeto esteja dentro de `atividade-02/` no repositório GitHub, no Google Drive a pasta `projeto-cardiovascular-ia-02` precisa ficar **diretamente na raiz do Meu Drive**. Dessa forma, não é necessário modificar os caminhos dos notebooks.

### 1. Copiar o projeto para o Google Drive

**Opção A — pelo Colab.** Abra um notebook novo no [Google Colab](https://colab.research.google.com/) e execute:

```python
from google.colab import drive
drive.mount("/content/drive")

# Clona o repositório original em uma sessão nova do Colab.
!git clone --depth 1 https://github.com/maricaico/projeto-cardiovascular-ia.git /content/repositorio-cardiovascular

# Copia somente a pasta da Atividade 02 para a raiz do Meu Drive.
!cp -r "/content/repositorio-cardiovascular/atividade-02/projeto-cardiovascular-ia-02" "/content/drive/MyDrive/"

# Confere a estrutura copiada.
!ls "/content/drive/MyDrive/projeto-cardiovascular-ia-02"
```

> **Antes de copiar:** a pasta `Meu Drive/projeto-cardiovascular-ia-02/` não deve existir. Se já existir, renomeie-a ou mova-a, para não misturar arquivos de versões diferentes.

**Opção B — manualmente.** No GitHub, clique em **Code → Download ZIP** e descompacte o arquivo. Dentro da pasta `projeto-cardiovascular-ia-main/atividade-02/`, localize `projeto-cardiovascular-ia-02/` e envie somente essa pasta, mantendo o nome, para a raiz do *Meu Drive* (no Drive: **Novo → Upload de pasta**).

Em ambos os casos, o caminho resultante deve ser:

```text
Meu Drive/
└── projeto-cardiovascular-ia-02/
    ├── README.md
    ├── requirements.txt
    ├── notebooks/
    ├── data/
    ├── src/
    ├── models/
    ├── rules/
    ├── logs/
    ├── reports/
    └── docs/
```

### 2. Abrir e executar os notebooks

1. No Google Drive, abra `Meu Drive/projeto-cardiovascular-ia-02/notebooks/`.
2. Abra cada notebook com **Abrir com → Google Colaboratory**.
3. Execute os notebooks **na ordem 03 → 04 → 05 → 06 → 07 → 08**, usando **Ambiente de execução → Executar tudo**.
4. Autorize o acesso ao Google Drive quando solicitado pela primeira célula.

Os notebooks 04 e 05 treinam e salvam os modelos em `models/`, que são carregados pelos notebooks 06, 07 e 08. Os notebooks 01 e 02 pertencem à Atividade 01 e estão incluídos apenas como referência; não precisam ser executados.

Ao reexecutar os notebooks, os arquivos de `models/`, `reports/` e `logs/` são sobrescritos. As métricas devem se manter; os tempos e as estimativas de energia tendem a variar, conforme descrito em Reprodutibilidade.

### Dependências

O Google Colab já disponibiliza as principais bibliotecas utilizadas. As versões de referência estão registradas em `requirements.txt`.

Caso seja necessário instalar as dependências do projeto, primeiro conecte o Google Drive e, em seguida, execute:

```python
!pip install -r "/content/drive/MyDrive/projeto-cardiovascular-ia-02/requirements.txt"
```

Se solicitado, reinicie o ambiente de execução após a instalação. Alterações nas versões das bibliotecas podem produzir diferenças em relação à execução original.

### Execução local (opcional)

Para executar em Jupyter local, instale as dependências com `pip install -r requirements.txt` e, na primeira célula de código de cada notebook, remova as linhas de `google.colab`/`drive.mount` e altere `PROJETO` para o caminho local desta pasta.

## Reprodutibilidade

- O uso de sementes aleatórias fixas (`random_state=42`) favorece a reprodutibilidade da divisão dos dados e dos algoritmos que utilizam aleatoriedade. Entretanto, versões das bibliotecas e características do ambiente podem ocasionar variações nos resultados.
- Tempos, memória e estimativas de energia dependem do hardware e da carga do ambiente, e podem variar entre execuções. As medições finais foram feitas em 21/10/2026 no Google Colab (Intel Xeon 2,20 GHz, 2 CPUs lógicas, 12,67 GB de RAM, Python 3.13.16, scikit-learn 1.6.1).
- Os notebooks estão salvos com as saídas da execução final, que correspondem aos valores do relatório.

## Principais resultados

### Classificação — `Cardiovascular_Disease` (teste, 4.000 registros)

| Modelo | Accuracy | Recall | F1 | ROC-AUC | AP | F1 CV (média ± DP) |
|---|---:|---:|---:|---:|---:|---:|
| Sistema Especialista | 0,7728 | **0,6119** | **0,6224** | 0,7909 | 0,6213 | 0,6414 ± 0,0045 |
| Árvore de Decisão | 0,7795 | 0,5809 | 0,6172 | 0,7842 | 0,6121 | 0,6349 ± 0,0057 |
| Random Forest | 0,7845 | 0,5425 | 0,6064 | 0,8006 | 0,6921 | 0,6248 ± 0,0061 |
| Gradient Boosting | **0,7865** | 0,5523 | 0,6129 | **0,8009** | **0,6935** | 0,6239 ± 0,0033 |

### Regressão — `Risk_Score` (teste, 4.000 registros)

| Modelo | MAE | RMSE | R² | RMSE CV (média ± DP) |
|---|---:|---:|---:|---:|
| Árvore de Decisão | 2,3836 | 2,9707 | 0,4756 | 2,9834 ± 0,0398 |
| Random Forest | 2,3704 | 2,9529 | 0,4819 | 2,9703 ± 0,0408 |
| Gradient Boosting | **2,3701** | **2,9526** | **0,4820** | 2,9627 ± 0,0406 |

### Eficiência computacional (execução final)

| Tarefa | Modelo | Treino (s) | Inferência de 4.000 amostras (s) | Tamanho (KB) |
|---|---|---:|---:|---:|
| Classificação | Sistema Especialista | — ² | 0,054 | 3,89 |
| Classificação | Árvore de Decisão | 0,108 | 0,042 | 6,14 |
| Classificação | Random Forest | 1,358 | 0,051 | 487,60 |
| Classificação | Gradient Boosting | 0,801 | 0,027 | 80,61 |
| Regressão | Árvore de Decisão | 0,067 | 0,021 | 8,79 |
| Regressão | Random Forest | 2,629 | 0,114 | 888,17 |
| Regressão | Gradient Boosting | 1,312 | 0,044 | 88,37 |

² O Sistema Especialista não tem treinamento computacional. As 11 regras foram elaboradas manualmente em 3 sessões, com **60 minutos efetivos** (média de 20 minutos por sessão), registrados em `logs/cronometro_regras.csv`.

### Síntese

- **Classificação:** o Sistema Especialista obteve o maior F1 e o maior Recall no teste; o Gradient Boosting obteve as maiores Accuracy, ROC-AUC e Average Precision. As diferenças foram pequenas.
- **Regressão:** o Gradient Boosting obteve o menor RMSE, praticamente empatado com a Random Forest, com arquivo cerca de dez vezes menor.
- **Generalização:** os sete modelos finais apresentaram gaps pequenos e foram considerados bem ajustados; as árvores sem restrição de profundidade apresentaram sobreajuste.
- **Eficiência:** a Árvore de Decisão teve o treinamento mais rápido e o menor arquivo entre os modelos de aprendizado de máquina; a Random Forest teve o maior custo de treino e de armazenamento.

As tabelas comparativas completas estão em `reports/tables/resumo_final_classificacao.csv` e `reports/tables/resumo_final_regressao.csv`.

> Os resultados têm finalidade acadêmica e não constituem validação clínica.
