"""Atividade 02 (DCT1401): divisão e pré-processamento sem vazamento.

- A divisão estratificada ocorre ANTES de qualquer imputação.
- As seis variáveis definidas na Atividade 01 são preservadas.
- O pré-processamento é ajustado DENTRO de cada rodada de CV quando
  utilizado por meio de criar_pipeline_modelo().
- O sistema especialista usa os mesmos dados, convertidos sem padronização,
  para permitir regras interpretáveis em mg/dL e mmHg.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


# As seis variáveis selecionadas na Atividade 01 para classificação.
# O conjunto de regressão contém AS MESMAS seis variáveis.
FEATURES_CLASSIFICACAO = (
    "Diabetes",
    "Hypertension",
    "Chest_Pain",
    "ECG_Result",
    "Blood_Glucose",
    "Blood_Pressure_Systolic",
)
FEATURES_REGRESSAO = (
    "Diabetes",
    "Hypertension",
    "ECG_Result",
    "Blood_Glucose",
    "Chest_Pain",
    "Blood_Pressure_Systolic",
)

# Ordem canônica comum para criar o mesmo X nos dois problemas.
# A ordem das colunas não altera quais variáveis foram selecionadas.
FEATURES_MODELO = FEATURES_CLASSIFICACAO

COLUNAS_CATEGORICAS = (
    "Diabetes",
    "Hypertension",
    "Chest_Pain",
    "ECG_Result",
)
COLUNAS_NUMERICAS = ("Blood_Glucose", "Blood_Pressure_Systolic")


def carregar_e_dividir_dados_sem_leakage(
    caminho_raw="data/raw/cardiovascular_disease_prediction.csv",
    test_size=0.20,
    random_state=42,
):
    """Retorna os dados BRUTOS separados: Xtreino, Xteste e ambos os alvos.

    Atenção: X ainda possui categorias como 'Yes'/'No' e valores ausentes.
    Aplicar criar_pipeline_modelo() para modelos scikit-learn ou
    preparar_dados_sistema_especialista() para as regras.
    """
    df = pd.read_csv(caminho_raw)
    colunas_necessarias = set(FEATURES_MODELO) | {
        "Cardiovascular_Disease", "Risk_Score"
    }
    ausentes = sorted(colunas_necessarias - set(df.columns))
    if ausentes:
        raise ValueError(f"Colunas obrigatórias não encontradas: {ausentes}")
    if df[["Cardiovascular_Disease", "Risk_Score"]].isna().any().any():
        raise ValueError("Os targets contêm valores ausentes; revisar o dataset.")
    if not set(df["Cardiovascular_Disease"].unique()).issubset({0, 1}):
        raise ValueError("Target de classificação deve conter apenas 0 e 1.")

    # 1. Capturamos SOMENTE as seis features previamente selecionadas.
    X = df.loc[:, list(FEATURES_MODELO)].copy()
    y_classificacao = df["Cardiovascular_Disease"].copy()
    y_regressao = df["Risk_Score"].copy()

    # 2. Mesmos índices de treino/teste para os dois alvos.
    #    Não existe cálculo de estatísticas sobre o teste nesta etapa.
    indices_treino, indices_teste = train_test_split(
        np.arange(len(df)),
        test_size=test_size,
        random_state=random_state,
        stratify=y_classificacao,
    )

    X_treino = X.iloc[indices_treino].copy()
    X_teste = X.iloc[indices_teste].copy()
    y_treino_c = y_classificacao.iloc[indices_treino].copy()
    y_teste_c = y_classificacao.iloc[indices_teste].copy()
    y_treino_r = y_regressao.iloc[indices_treino].copy()
    y_teste_r = y_regressao.iloc[indices_teste].copy()

    assert len(set(indices_treino) & set(indices_teste)) == 0
    assert len(indices_treino) + len(indices_teste) == len(df)

    return (
        X_treino, X_teste, y_treino_c, y_teste_c,
        y_treino_r, y_teste_r, list(FEATURES_MODELO),
    )


def criar_preprocessador(padronizar=False):
    """Fábrica de transformações ajustáveis no treino ou em cada fold.

    Diabetes/Hypertension/Chest_Pain: No=0, Yes=1.
    ECG_Result: Normal=0, Abnormal=1.
    Blood_Glucose e Blood_Pressure_Systolic: mediana do conjunto usado em fit.
    padronizar=False por padrão: árvores e regras não precisam de StandardScaler.
    """
    numericas = [("imputacao", SimpleImputer(strategy="median"))]
    if padronizar:
        numericas.append(("escalonamento", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            (
                "categoricas",
                OrdinalEncoder(
                    categories=[
                        ["No", "Yes"],  # Diabetes
                        ["No", "Yes"],  # Hypertension
                        ["No", "Yes"],  # Chest_Pain
                        ["Normal", "Abnormal"],  # ECG_Result
                    ],
                    handle_unknown="error",
                ),
                list(COLUNAS_CATEGORICAS),
            ),
            (
                "numericas",
                Pipeline(numericas),
                list(COLUNAS_NUMERICAS),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def criar_pipeline_modelo(modelo, padronizar=False):
    """Pipeline que o GridSearchCV deve receber com X de treino BRUTO.

    Exemplo de parâmetros de GridSearchCV: 'modelo__max_depth'.
    O ajuste de imputação acontece novamente em CADA fold.
    """
    return Pipeline([
        ("preprocessamento", criar_preprocessador(padronizar=padronizar)),
        ("modelo", modelo),
    ])


def preparar_dados_sistema_especialista(X_treino, X_teste):
    """Converte treino e teste para DataFrames numéricos interpretáveis.

    Os valores de glicose permanecem em mg/dL e pressão sistólica em mmHg.
    O mesmo preprocessador é ajustado SÓ no treino e aplicado ao teste.
    Para o sistema especialista não há ajuste de modelo estatístico por CV.
    """
    preprocessador = criar_preprocessador(padronizar=False)
    dados_treino = preprocessador.fit_transform(X_treino)
    dados_teste = preprocessador.transform(X_teste)
    nomes = list(preprocessador.get_feature_names_out())

    treino = pd.DataFrame(dados_treino, columns=nomes, index=X_treino.index)
    teste = pd.DataFrame(dados_teste, columns=nomes, index=X_teste.index)
    return treino, teste, preprocessador


def salvar_indices_divisao(X_treino, X_teste, pasta="data/processed"):
    """Opcional: registra quais linhas foram destinadas a treino e teste.

    Os índices representam linhas do CSV original (a partir de zero), não
    identificadores de pacientes. Não salva variáveis já imputadas ou escaladas.
    """
    destino = Path(pasta)
    destino.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"indice_original": X_treino.index}).to_csv(
        destino / "indices_treino.csv", index=False
    )
    pd.DataFrame({"indice_original": X_teste.index}).to_csv(
        destino / "indices_teste.csv", index=False
    )
