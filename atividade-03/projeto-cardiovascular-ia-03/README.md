# Atividade 03 — Redes Neurais Artificiais com PyTorch

**Classificação e Regressão com Perceptrons Multicamadas (MLP)**

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

Esta atividade dá continuidade à [Atividade 01](../../README.md) e à [Atividade 02](../../atividade-02/projeto-cardiovascular-ia-02/README.md) e aplica redes neurais do tipo Perceptron Multicamadas (MLP), implementadas em PyTorch, aos mesmos dois problemas supervisionados:

- **Classificação:** `Cardiovascular_Disease` (0 = ausência, 1 = presença);
- **Regressão:** `Risk_Score`.

O trabalho inclui:

- redes de referência (*baseline*) para os dois problemas;
- um estudo de hiperparâmetros com oito fatores (neurônios por camada, número de camadas, ativação, otimizador, taxa de aprendizado, tamanho do *batch*, *weight decay* e *dropout*), mapa de calor taxa × *batch*, busca aleatória e análise de sensibilidade;
- salvamento dos modelos finais em `.pt`, com o pré-processamento embutido, e um script de inferência independente;
- avaliação no teste, análise de generalização (gap, curvas de aprendizado, validação cruzada *K-fold*, subajuste e sobreajuste);
- eficiência computacional (tempo, memória, operações, tamanho em disco, energia e CO₂).

## Protocolo experimental

- **Dados:** leitura do CSV original. As bases pré-processadas da Atividade 01 (`dataset_regression.csv` e `dataset_classification.csv`) não são usadas como entrada, porque nelas a imputação e a padronização foram ajustadas com os 20.000 registros, antes da divisão; para as seis features usadas, essa troca não altera nenhum valor numérico (ver Notebook 03).
- **Divisão:** treino/teste de 80%/20% (16.000/4.000 registros), idêntica à da Atividade 02 (estratificada por `Cardiovascular_Disease`, `random_state=42`). Dentro do treino, ajuste (12.800) e validação (3.200), também estratificados. Os mesmos índices são usados nos dois problemas e estão salvos em `data/processed/indices_divisao.csv`.
- **Features:** `Diabetes`, `Hypertension`, `Chest_Pain`, `ECG_Result`, `Blood_Glucose` e `Blood_Pressure_Systolic`.
- **Pré-processamento:** codificação binária, imputação da glicose pela mediana e `StandardScaler` nas seis features, ajustados somente no subconjunto de ajuste. Na regressão, o alvo também é padronizado, e as previsões voltam à escala original antes das métricas.
- **Treinamento:** `TensorDataset` e `DataLoader`; até 200 épocas com *early stopping* (paciência 20) pela perda de validação e restauração do melhor estado; cinco sementes (0 a 4) por configuração, com média e desvio amostral.
- **Perdas:** `MSELoss` (regressão) e `BCEWithLogitsLoss` com um logit (classificação; a sigmoide é aplicada só na inferência).
- **Seleção:** RMSE de validação (regressão) e Average Precision de validação (classificação). O limiar de decisão da classificação é escolhido na validação (F1-score máximo).
- **Teste:** os modelos, as sementes e os limiares são congelados (com impressão digital dos pesos) antes de qualquer contato com o teste, usado na avaliação final. As análises adicionais no teste (variação entre sementes, curvas e comparação com referências) foram apenas descritivas, sem uso na seleção de modelos, hiperparâmetros, limiar ou semente.

## Modelos finais

A mesma configuração foi escolhida para os dois problemas:

| Hiperparâmetro | Valor |
|---|---|
| Arquitetura | 2 camadas ocultas × 32 neurônios, ReLU, saída linear (1.313 parâmetros) |
| Otimizador / taxa de aprendizado | Adam / 10⁻³ |
| *Batch* / *weight decay* / *dropout* | 32 / 10⁻³ / 0 |
| Épocas (máximo) / paciência | 200 / 20 |
| Limiar de decisão (classificação) | 0,404 (F1 máximo na validação) |

Ela corresponde à configuração de referência com regularização L2 leve. A diferença em relação à referência é da ordem da variação entre sementes e não é uma melhora demonstrada (ver Notebooks 05 e 07).

## Estrutura do projeto

No GitHub, este projeto está em `atividade-03/projeto-cardiovascular-ia-03/`. A árvore abaixo parte dessa pasta. A numeração dos notebooks começa em 03 para seguir a sequência das atividades anteriores.

```text
projeto-cardiovascular-ia-03/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   └── cardiovascular_disease_prediction.csv   # dataset original
│   ├── processed/
│   │   └── indices_divisao.csv                     # índices de ajuste, validação e teste
│   └── exemplo_amostras_inferencia.csv             # entradas de exemplo para o script de inferência
├── notebooks/
│   ├── 03_mlp_regressao.ipynb              # protocolo de dados e baseline de regressão
│   ├── 04_mlp_classificacao.ipynb          # baseline de classificação, pos_weight e limiar
│   ├── 05_hiperparametros.ipynb            # estudo de hiperparâmetros e escolha da configuração final
│   ├── 06_salvar_carregar_inferencia.ipynb # modelos finais, congelamento, .pt e inferência
│   ├── 07_avaliacao_generalizacao.ipynb    # teste, resíduos, curvas, K-fold, sub/sobreajuste
│   └── 08_eficiencia_computacional.ipynb   # tempo, memória, energia, CO₂ e tabela-resumo
├── src/
│   ├── __init__.py
│   ├── pipeline_data.py    # divisão dos dados e pré-processamento
│   ├── models.py           # classe MLP parametrizável
│   ├── train.py            # DataLoader, laço de treinamento, early stopping e log
│   ├── metrics.py          # métricas de regressão e classificação
│   ├── inference.py        # script de inferência a partir dos .pt
│   ├── profiling.py        # medição de tempo, memória e energia
│   └── visualization.py    # gráficos do estudo de hiperparâmetros
├── models/
│   ├── modelo_regressao.pt
│   ├── modelo_classificacao.pt
│   └── referencia/         # modelos da configuração de referência (baseline_*.pt)
├── logs/
│   ├── hiperparametros.csv                       # todas as execuções do estudo (546 treinamentos)
│   ├── hiperparametros_backup_20261113_100602.csv  # execuções-piloto feitas em outro processador ¹
│   ├── busca_aleatoria_configuracoes.csv
│   ├── configuracao_final.json
│   ├── congelamento_modelos.json                 # impressões digitais registradas antes do teste
│   ├── previsoes_teste.csv                       # previsões dos modelos carregados no teste
│   ├── curvas/                                   # curvas dos baselines (Notebooks 03 e 04)
│   └── curvas_estudo/                            # curvas de cada execução do estudo (Notebook 05)
├── reports/
│   ├── tables/             # tabelas CSV (inclui resumo_final.csv)
│   └── figures/            # gráficos
└── docs/
    └── relatorio_final.pdf
```

¹ As primeiras execuções-piloto foram feitas em um processador AMD EPYC. Para que todo o estudo fosse feito no mesmo hardware, elas foram refeitas no Intel Xeon; o registro original foi preservado nesse arquivo.

## Como executar (Google Colab)

Os notebooks foram desenvolvidos e executados no **Google Colab**, em CPU. O código de configuração de cada notebook procura a pasta do projeto neste caminho:

```text
Meu Drive/projeto-cardiovascular-ia-03/
```

**Importante:** embora o projeto esteja dentro de `atividade-03/` no repositório GitHub, no Google Drive a pasta `projeto-cardiovascular-ia-03` precisa ficar **diretamente na raiz do Meu Drive**. Dessa forma, não é necessário modificar os caminhos dos notebooks.

### 1. Copiar o projeto para o Google Drive

**Opção A — pelo Colab.** Abra um notebook novo no [Google Colab](https://colab.research.google.com/) e execute:

```python
from google.colab import drive
drive.mount("/content/drive")

# Clona o repositório original em uma sessão nova do Colab.
!git clone --depth 1 https://github.com/maricaico/projeto-cardiovascular-ia.git /content/repositorio-cardiovascular

# Copia somente a pasta da Atividade 03 para a raiz do Meu Drive.
!cp -r "/content/repositorio-cardiovascular/atividade-03/projeto-cardiovascular-ia-03" "/content/drive/MyDrive/"

# Confere a estrutura copiada.
!ls "/content/drive/MyDrive/projeto-cardiovascular-ia-03"
```

> **Antes de copiar:** a pasta `Meu Drive/projeto-cardiovascular-ia-03/` não deve existir. Se já existir, renomeie-a ou mova-a, para não misturar arquivos de versões diferentes.

**Opção B — manualmente.** No GitHub, clique em **Code → Download ZIP** e descompacte o arquivo. Dentro da pasta `projeto-cardiovascular-ia-main/atividade-03/`, localize `projeto-cardiovascular-ia-03/` e envie somente essa pasta, mantendo o nome, para a raiz do *Meu Drive* (no Drive: **Novo → Upload de pasta**).

Em ambos os casos, o caminho resultante deve ser:

```text
Meu Drive/
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
```

### 2. Abrir e executar os notebooks

1. No Google Drive, abra `Meu Drive/projeto-cardiovascular-ia-03/notebooks/`.
2. Abra cada notebook com **Abrir com → Google Colaboratory** e use um ambiente de execução em **CPU**.
3. Execute os notebooks **na ordem 03 → 04 → 05 → 06 → 07 → 08**, usando **Ambiente de execução → Reiniciar sessão e executar tudo**.
4. Autorize o acesso ao Google Drive quando solicitado pela primeira célula.

**Dependências entre os notebooks.** O 05 grava `logs/configuracao_final.json`, usado pelo 06; o 06 grava os modelos em `models/`, o congelamento e as previsões do teste em `logs/`, usados pelo 07; o 08 usa os modelos e as tabelas do 07.

**Arquivos já incluídos.** Os modelos (`models/`), os logs (`logs/`, incluindo `hiperparametros.csv`), as tabelas e as figuras (`reports/`) já estão no repositório. Por isso, a cópia inclui tudo: os notebooks podem ser consultados com as saídas salvas, e o script de inferência funciona sem executar nenhum notebook.

**Estudo de hiperparâmetros.** O Notebook 05 registra cada treinamento em `logs/hiperparametros.csv` e não repete execuções já registradas. Com o log incluído no repositório, ele apenas refaz as tabelas, os gráficos e uma conferência de reprodutibilidade (alguns minutos). Sem o log, o estudo completo leva várias horas.

**Tempo total aproximado** com o log presente: de 30 a 40 minutos, dos quais cerca de 15 no Notebook 07.

Ao reexecutar os notebooks, os arquivos de `models/`, `reports/` e `logs/` são sobrescritos. As métricas devem se manter no mesmo processador; os tempos e as estimativas de energia variam, conforme descrito em Reprodutibilidade.

### Dependências

O Google Colab já disponibiliza as bibliotecas utilizadas. As versões de referência estão registradas em `requirements.txt`. Caso seja necessário instalá-las, conecte o Google Drive e execute:

```python
!pip install -r "/content/drive/MyDrive/projeto-cardiovascular-ia-03/requirements.txt"
```

Se solicitado, reinicie o ambiente de execução após a instalação. Alterações nas versões das bibliotecas podem produzir diferenças em relação à execução original.

### Execução local (opcional)

Para executar em Jupyter local, instale as dependências com `pip install -r requirements.txt` e, na primeira célula de código de cada notebook, remova as linhas de `google.colab`/`drive.mount` e altere `PROJETO` para o caminho local desta pasta.

## Inferência com os modelos salvos

O script `src/inference.py` usa apenas os arquivos `.pt` e **não depende do Colab, do Google Drive nem dos notebooks**. Cada checkpoint contém os pesos, a arquitetura e todo o pré-processamento (codificação, medianas de imputação, média e desvio das features e, na regressão, do alvo; na classificação, o limiar). A partir desta pasta, com as dependências instaladas:

```bash
# Classificação: probabilidade da classe 1, classe prevista pelo limiar e nome da classe
python src/inference.py --modelo models/modelo_classificacao.pt --entrada data/exemplo_amostras_inferencia.csv

# Regressão: Risk_Score previsto na escala original, gravado em um CSV
python src/inference.py --modelo models/modelo_regressao.pt --entrada data/exemplo_amostras_inferencia.csv --saida previsoes.csv
```

A entrada é um CSV com as seis features em unidades originais:

```text
Diabetes,Hypertension,Chest_Pain,ECG_Result,Blood_Glucose,Blood_Pressure_Systolic
Yes,No,No,Normal,132,141
```

Valores numéricos ausentes são preenchidos com a mediana do subconjunto de ajuste; colunas ausentes e categorias desconhecidas geram erro. Os modelos são carregados com `torch.load(..., map_location="cpu", weights_only=True)` e colocados em modo de avaliação (`model.eval()`). As quatro amostras do arquivo de exemplo vêm do subconjunto de validação (sem os alvos) e servem apenas para demonstrar o uso.

## Reprodutibilidade

- A divisão dos dados usa sempre `random_state=42`; cada configuração da rede é treinada com as sementes 0 a 4, que controlam a inicialização dos pesos e a ordem dos lotes.
- Os resultados são reproduzidos **exatamente** no mesmo processador. Em outro processador, aparecem diferenças numéricas pequenas (da ordem de 10⁻⁴ no RMSE), bem menores que a variação entre sementes. Todo o estudo e a execução final foram feitos no Google Colab com **Intel Xeon @ 2,20 GHz**, 2 CPUs lógicas, 1 *thread* do PyTorch e 12,67 GB de RAM (Python 3.13.18, PyTorch 2.14.1+cpu, scikit-learn 1.6.1). O processador de cada execução está registrado em `logs/hiperparametros.csv`.
- Tempos, memória e estimativas de energia dependem do hardware e da carga do ambiente, e variam entre execuções.
- Os notebooks estão salvos com as saídas da execução final, que correspondem aos valores do relatório.

## Principais resultados

### Regressão — `Risk_Score` (teste, 4.000 registros)

| Modelo | MAE | RMSE | R² |
|---|---:|---:|---:|
| MLP final | **2,3854** | **2,9686** | **0,4763** |
| MLP de referência | 2,3918 | 2,9751 | 0,4740 |
| Preditor ingênuo (média do ajuste) | 3,2685 | 4,1028 | −0,0003 |

Validação cruzada (5 *folds*, configuração final): RMSE 2,9816 ± 0,0431; R² 0,4941 ± 0,0123.

### Classificação — `Cardiovascular_Disease` (teste, 4.000 registros)

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC | AP |
|---|---:|---:|---:|---:|---:|---:|
| MLP final (limiar 0,4038) | **0,7775** | **0,6486** | 0,5956 | **0,6210** | 0,7993 | **0,6900** |
| MLP de referência (limiar 0,362) | 0,7755 | 0,6427 | **0,5997** | 0,6205 | **0,7994** | **0,6900** |
| Regressão logística (referência linear) | — | — | — | — | 0,7968 | 0,6850 |

Validação cruzada (5 *folds*, configuração final): AP 0,6937 ± 0,0033; ROC-AUC 0,8165 ± 0,0049; F1 0,6394 ± 0,0054.

### Eficiência computacional (modelos finais, execução final)

| Tarefa | Parâmetros | MACs por amostra | Arquivo `.pt` (KB) | Treino (s) ³ | Acréscimo de RAM no treino (MB) | Inferência em lote (µs por amostra) ² | Uma amostra (ms) ² |
|---|---:|---:|---:|---:|---:|---:|---:|
| Regressão | 1.313 | 1.248 | 9,6 | 23,7 | 96,5 | 6,22 | 6,59 |
| Classificação | 1.313 | 1.248 | 9,9 | 26,6 | 76,9 | 3,35 | 6,23 |

² De ponta a ponta, com validação e pré-processamento (`inference.prever`).

³ Novo treinamento feito no Notebook 08 com a semente do modelo salvo. Nas cinco sementes do estudo de hiperparâmetros, a média foi de cerca de 41 s (regressão) e 31 s (classificação), porque o número de épocas até o *early stopping* varia entre sementes.

Energia estimada de um treinamento: cerca de 0,03 Wh e 1,2 a 1,4 mg de CO₂ (hipótese de 4 W por *thread*; fator de 45,8 gCO₂/kWh). O estudo de hiperparâmetros completo (546 treinamentos, cerca de 5,8 h de CPU) corresponde a cerca de 0,02 kWh e 1 g de CO₂.

### Síntese

- **Desempenho:** nos dois problemas, a rede final e a referência são equivalentes no teste; a diferença pareada em cinco sementes é menor que a variação entre inicializações. A rede supera com folga o preditor ingênuo na regressão e fica apenas um pouco acima da regressão logística na classificação.
- **Generalização:** gaps pequenos entre treino, validação e teste e curvas de aprendizado estáveis; nenhuma configuração extrema do estudo apresentou sobreajuste (o *early stopping* devolve o melhor estado), e os extremos problemáticos foram casos de subajuste ou de falha de otimização. O desempenho é limitado pela informação contida nas seis features.
- **Hiperparâmetros:** a taxa de aprendizado é o fator dominante; dentro da faixa utilizável, o desempenho forma um patamar estreito, e as diferenças aparecem nos extremos e no custo.
- **Eficiência:** a rede é pequena (cerca de 5 KB de pesos) e viável em dispositivos de borda; o tamanho do *batch* é o hiperparâmetro que mais afeta o tempo de treinamento. Pesos subnormais, surgidos com a regularização L2, tornaram a inferência em lote do modelo de regressão mais de dez vezes mais lenta; o *flush* de subnormais elimina o problema sem alterar as previsões.

As tabelas completas estão em `reports/tables/` (resumo em `reports/tables/resumo_final.csv`).

> Os resultados têm finalidade acadêmica e não constituem validação clínica.
