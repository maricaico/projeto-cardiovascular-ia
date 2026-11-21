"""Atividade 03 (DCT1401): dados, divisão e pré-processamento sem vazamento.

Este módulo reproduz o protocolo de dados da Atividade 02 e acrescenta o que
uma rede neural (MLP) exige:

1. Divisão treino/teste IDÊNTICA à da Atividade 02 (função copiada sem
   alteração): 16.000 / 4.000 registros, estratificada, random_state=42.
2. Nova divisão, DENTRO dos 16.000 de treino, em ajuste (12.800) e
   validação (3.200), estratificada pela classe e com random_state=42.
   Os mesmos índices servem à classificação e à regressão.
3. Pré-processamento ajustado SOMENTE no subconjunto de ajuste:
   codificação das categorias (como na Atividade 02), imputação da glicose
   pela mediana e StandardScaler nas seis features.
4. Padronização do alvo de regressão (Risk_Score) com média e desvio do
   subconjunto de ajuste, e conversão de volta à escala original.

O conjunto de teste nunca é usado para calcular estatísticas.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


# ---------------------------------------------------------------------------
# Constantes (mesmas da Atividade 02)
# ---------------------------------------------------------------------------

RANDOM_STATE = 42

# Ordem canônica das seis features, igual à Atividade 02.
FEATURES_MODELO = (
    "Diabetes",
    "Hypertension",
    "Chest_Pain",
    "ECG_Result",
    "Blood_Glucose",
    "Blood_Pressure_Systolic",
)

COLUNAS_CATEGORICAS = ("Diabetes", "Hypertension", "Chest_Pain", "ECG_Result")
COLUNAS_NUMERICAS = ("Blood_Glucose", "Blood_Pressure_Systolic")

# Codificação da Atividade 02: a posição na lista é o código (0 ou 1).
CATEGORIAS = {
    "Diabetes": ["No", "Yes"],
    "Hypertension": ["No", "Yes"],
    "Chest_Pain": ["No", "Yes"],
    "ECG_Result": ["Normal", "Abnormal"],
}

ALVO_CLASSIFICACAO = "Cardiovascular_Disease"
ALVO_REGRESSAO = "Risk_Score"
NOMES_CLASSES = ["Ausência (0)", "Presença (1)"]

# Valores de referência observados nas saídas do Notebook 03 da Atividade 02
# e na simulação da divisão de validação. Servem como verificação automática:
# se algum valor mudar, a divisão deixou de ser a mesma e a execução para.
REFERENCIA_DIVISAO = {
    "n_treino": 16000,
    "n_teste": 4000,
    "positivos_treino": 4896,
    "positivos_teste": 1224,
    "ausentes_treino": 476,
    "ausentes_teste": 124,
    "mediana_glicose_treino": 105.0,
    "primeiros_indices_treino": [10656, 16130, 6870, 4331, 3982],
    "n_ajuste": 12800,
    "n_validacao": 3200,
    "positivos_ajuste": 3917,
    "positivos_validacao": 979,
}


# ---------------------------------------------------------------------------
# 1. Divisão treino/teste — CÓPIA LITERAL da Atividade 02
# ---------------------------------------------------------------------------

def carregar_e_dividir_dados_sem_leakage(
    caminho_raw="data/raw/cardiovascular_disease_prediction.csv",
    test_size=0.20,
    random_state=RANDOM_STATE,
):
    """Retorna os dados BRUTOS separados: Xtreino, Xteste e ambos os alvos.

    Mesma lógica de atividade-02/.../src/pipeline_data.py. X ainda possui
    categorias como 'Yes'/'No' e valores ausentes; os índices do DataFrame
    são as linhas do CSV original (a partir de zero).
    """
    df = pd.read_csv(caminho_raw)
    colunas_necessarias = set(FEATURES_MODELO) | {
        ALVO_CLASSIFICACAO, ALVO_REGRESSAO
    }
    ausentes = sorted(colunas_necessarias - set(df.columns))
    if ausentes:
        raise ValueError(f"Colunas obrigatórias não encontradas: {ausentes}")
    if df[[ALVO_CLASSIFICACAO, ALVO_REGRESSAO]].isna().any().any():
        raise ValueError("Os targets contêm valores ausentes; revisar o dataset.")
    if not set(df[ALVO_CLASSIFICACAO].unique()).issubset({0, 1}):
        raise ValueError("Target de classificação deve conter apenas 0 e 1.")

    X = df.loc[:, list(FEATURES_MODELO)].copy()
    y_classificacao = df[ALVO_CLASSIFICACAO].copy()
    y_regressao = df[ALVO_REGRESSAO].copy()

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


# ---------------------------------------------------------------------------
# 2. Divisão ajuste/validação — NOVA na Atividade 03
# ---------------------------------------------------------------------------

def dividir_ajuste_validacao(
    X_treino, y_treino_c, y_treino_r, val_size=0.20, random_state=RANDOM_STATE
):
    """Separa os 16.000 de treino em ajuste (12.800) e validação (3.200).

    - Estratificada pela classe, para preservar a proporção de positivos.
    - Os MESMOS índices são usados na regressão, para que os dois problemas
      sejam treinados e validados sobre os mesmos registros.
    - O teste não participa desta etapa.
    """
    idx_ajuste, idx_validacao = train_test_split(
        X_treino.index.values,
        test_size=val_size,
        random_state=random_state,
        stratify=y_treino_c,
    )
    assert len(set(idx_ajuste) & set(idx_validacao)) == 0

    return {
        "X_ajuste": X_treino.loc[idx_ajuste].copy(),
        "X_validacao": X_treino.loc[idx_validacao].copy(),
        "y_ajuste_c": y_treino_c.loc[idx_ajuste].copy(),
        "y_validacao_c": y_treino_c.loc[idx_validacao].copy(),
        "y_ajuste_r": y_treino_r.loc[idx_ajuste].copy(),
        "y_validacao_r": y_treino_r.loc[idx_validacao].copy(),
    }


def preparar_divisoes(caminho_raw="data/raw/cardiovascular_disease_prediction.csv"):
    """Atalho: executa as duas divisões, verifica as referências e devolve tudo."""
    (X_treino, X_teste, y_treino_c, y_teste_c,
     y_treino_r, y_teste_r, features) = carregar_e_dividir_dados_sem_leakage(caminho_raw)
    dados = dividir_ajuste_validacao(X_treino, y_treino_c, y_treino_r)
    dados.update({
        "X_treino": X_treino, "X_teste": X_teste,
        "y_treino_c": y_treino_c, "y_teste_c": y_teste_c,
        "y_treino_r": y_treino_r, "y_teste_r": y_teste_r,
        "features": features,
    })
    verificar_divisoes(dados)
    return dados


def verificar_divisoes(dados, referencia=REFERENCIA_DIVISAO):
    """Confere a divisão contra os valores de referência; levanta erro se divergir."""
    ref = referencia
    obtido = {
        "n_treino": len(dados["X_treino"]),
        "n_teste": len(dados["X_teste"]),
        "positivos_treino": int(dados["y_treino_c"].sum()),
        "positivos_teste": int(dados["y_teste_c"].sum()),
        "ausentes_treino": int(dados["X_treino"].isna().sum().sum()),
        "ausentes_teste": int(dados["X_teste"].isna().sum().sum()),
        "mediana_glicose_treino": float(dados["X_treino"]["Blood_Glucose"].median()),
        "primeiros_indices_treino": [int(i) for i in dados["X_treino"].index[:5]],
        "n_ajuste": len(dados["X_ajuste"]),
        "n_validacao": len(dados["X_validacao"]),
        "positivos_ajuste": int(dados["y_ajuste_c"].sum()),
        "positivos_validacao": int(dados["y_validacao_c"].sum()),
    }
    divergencias = {k: (obtido[k], ref[k]) for k in ref if obtido[k] != ref[k]}
    if divergencias:
        raise AssertionError(
            "A divisão difere da referência (obtido, esperado): " f"{divergencias}"
        )

    # Os três subconjuntos não podem compartilhar registros.
    a = set(dados["X_ajuste"].index)
    v = set(dados["X_validacao"].index)
    t = set(dados["X_teste"].index)
    assert not (a & v) and not (a & t) and not (v & t), "Subconjuntos sobrepostos."
    assert len(a | v | t) == 20000
    return obtido


def salvar_indices(dados, pasta="data/processed"):
    """Registra a qual subconjunto cada linha do CSV original pertence.

    Gera um único CSV com as colunas `indice_original` (linha do CSV bruto,
    a partir de zero) e `conjunto` (ajuste, validacao ou teste). Não salva
    nenhum valor imputado ou padronizado.
    """
    destino = Path(pasta)
    destino.mkdir(parents=True, exist_ok=True)
    partes = [
        pd.DataFrame({"indice_original": dados[f"X_{nome}"].index, "conjunto": nome})
        for nome in ("ajuste", "validacao", "teste")
    ]
    tabela = pd.concat(partes, ignore_index=True)
    caminho = destino / "indices_divisao.csv"
    tabela.to_csv(caminho, index=False)
    return caminho


# ---------------------------------------------------------------------------
# 3. Pré-processamento das features para a MLP
# ---------------------------------------------------------------------------

def criar_preprocessador_mlp():
    """Codificação + imputação (como na Atividade 02) e StandardScaler nas seis.

    Etapa 1 (ColumnTransformer):
      - categóricas: No/Normal = 0, Yes/Abnormal = 1 (categoria desconhecida
        gera erro, para não aceitar silenciosamente um valor inválido);
      - numéricas: mediana do conjunto usado no fit (só Blood_Glucose tem
        ausentes nas seis features).
    Etapa 2: StandardScaler nas seis colunas resultantes.

    A ordem de saída é a de FEATURES_MODELO (categóricas e depois numéricas).
    """
    codificacao_imputacao = ColumnTransformer(
        transformers=[
            (
                "categoricas",
                OrdinalEncoder(
                    categories=[CATEGORIAS[c] for c in COLUNAS_CATEGORICAS],
                    handle_unknown="error",
                ),
                list(COLUNAS_CATEGORICAS),
            ),
            (
                "numericas",
                SimpleImputer(strategy="median"),
                list(COLUNAS_NUMERICAS),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline([
        ("codificacao_imputacao", codificacao_imputacao),
        ("padronizacao", StandardScaler()),
    ])


def estatisticas_preprocessador(preprocessador):
    """Extrai, de um preprocessador JÁ AJUSTADO, tudo o que o checkpoint precisa.

    Devolve tipos simples (listas, floats), para que o checkpoint possa ser
    carregado com torch.load(..., weights_only=True).
    """
    ct = preprocessador.named_steps["codificacao_imputacao"]
    escala = preprocessador.named_steps["padronizacao"]
    imputador = ct.named_transformers_["numericas"]
    nomes = list(ct.get_feature_names_out())
    if nomes != list(FEATURES_MODELO):
        raise AssertionError(f"Ordem inesperada das features: {nomes}")
    return {
        "features": nomes,
        "categorias": {c: list(CATEGORIAS[c]) for c in COLUNAS_CATEGORICAS},
        "mediana_imputacao": {
            c: float(v) for c, v in zip(COLUNAS_NUMERICAS, imputador.statistics_)
        },
        "x_media": [float(v) for v in escala.mean_],
        "x_desvio": [float(v) for v in escala.scale_],
    }


# ---------------------------------------------------------------------------
# 4. Padronização do alvo de regressão
# ---------------------------------------------------------------------------

def ajustar_escala_alvo(y_ajuste):
    """Média e desvio do Risk_Score no subconjunto de ajuste.

    Usa desvio populacional (ddof=0), a mesma convenção do StandardScaler.
    """
    y = np.asarray(y_ajuste, dtype=np.float64)
    return {"y_media": float(y.mean()), "y_desvio": float(y.std(ddof=0))}


def padronizar_alvo(y, escala_alvo):
    y = np.asarray(y, dtype=np.float64)
    return (y - escala_alvo["y_media"]) / escala_alvo["y_desvio"]


def despadronizar_alvo(y_padronizado, escala_alvo):
    """Volta as previsões à escala original antes de calcular as métricas."""
    y = np.asarray(y_padronizado, dtype=np.float64)
    return y * escala_alvo["y_desvio"] + escala_alvo["y_media"]


# ---------------------------------------------------------------------------
# 5. Folds do K-fold final — mesmos da Atividade 02, em cada tarefa
# ---------------------------------------------------------------------------

def criar_cv(tarefa, n_splits=5, random_state=RANDOM_STATE):
    """Reproduz o protocolo de validação cruzada da Atividade 02.

    Classificação: StratifiedKFold; regressão: KFold. Ambos com shuffle=True
    e random_state=42, aplicados aos 16.000 de treino. Cada tarefa reproduz
    os folds dela na Atividade 02; os folds NÃO são iguais entre as tarefas.
    """
    if tarefa == "classificacao":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    if tarefa == "regressao":
        return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    raise ValueError("tarefa deve ser 'classificacao' ou 'regressao'.")
