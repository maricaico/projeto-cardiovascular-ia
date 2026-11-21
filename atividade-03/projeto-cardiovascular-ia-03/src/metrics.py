"""Atividade 03 (DCT1401): métricas de regressão e de classificação.

Mesma base do src/metrics.py da Atividade 02, com uma diferença deliberada:
o MAPE é devolvido em PERCENTUAL (proporção × 100), e a conversão fica explícita no nome da chave ("MAPE_%").

Atenção ao MAPE: Risk_Score tem valores ≤ 0 (inclusive um valor exatamente
igual a zero no treino). O scikit-learn divide por um épsilon nesses casos,
o que pode tornar o MAPE extremamente grande. Ele é reportado como
métrica obrigatória, mas não é usado para escolher modelos.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def calcular_metricas_regressao(y_true, y_pred):
    """MAE, MSE, RMSE, MAPE (%), R² e MedAE — sempre na escala ORIGINAL."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError("y_true e y_pred devem ter tamanhos iguais e não vazios.")
    if not np.all(np.isfinite(y_pred)):
        raise ValueError("Há previsões não finitas (NaN/inf): o treino divergiu.")
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "MAPE_%": float(100.0 * mean_absolute_percentage_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "MedAE": float(median_absolute_error(y_true, y_pred)),
    }


def calcular_metricas_classificacao(y_true, y_prob, limiar=0.5):
    """Métricas binárias (classe positiva = 1) a partir das probabilidades.

    y_prob : probabilidade da classe 1 (torch.sigmoid do logit).
    limiar : y_prob >= limiar -> classe 1. Na versão final, o limiar é
        escolhido na VALIDAÇÃO, nunca no teste.

    PR-AUC = área trapezoidal sob Precision × Recall;
    AP = Average Precision. São reportadas separadamente, como na
    Atividade 02, porque não são numericamente idênticas.
    Log_Loss e Brier avaliam as probabilidades (calibração), sem limiar.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()
    if len(y_true) == 0 or len(y_true) != len(y_prob):
        raise ValueError("y_true e y_prob devem ter tamanhos iguais e não vazios.")
    if not set(np.unique(y_true)).issubset({0, 1}) or np.unique(y_true).size != 2:
        raise ValueError("y_true deve conter as duas classes (0 e 1).")
    if not np.all(np.isfinite(y_prob)):
        raise ValueError("Há probabilidades não finitas (NaN/inf): o treino divergiu.")

    y_pred = (y_prob >= limiar).astype(int)
    precisao_curva, recall_curva, _ = precision_recall_curve(y_true, y_prob, pos_label=1)
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1-score": float(f1_score(y_true, y_pred, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_true, y_prob)),
        "PR-AUC": float(auc(recall_curva[::-1], precisao_curva[::-1])),
        "Average_Precision": float(average_precision_score(y_true, y_prob)),
        "Log_Loss": float(log_loss(y_true, np.clip(y_prob, 1e-15, 1 - 1e-15), labels=[0, 1])),
        "Brier": float(brier_score_loss(y_true, y_prob)),
        "Confusion_Matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]),
        "limiar": float(limiar),
    }


def calcular_gap(metrica_treino, metrica_outro):
    """Gap = treino − (validação ou teste), fórmula usada em todo o projeto.

    Métricas de qualidade (F1, R²...): gap positivo = pior fora do treino.
    Métricas de erro (RMSE, MAE...): gap NEGATIVO = erro maior fora do treino.
    O sinal é interpretado no relatório, como na Atividade 02.
    """
    return metrica_treino - metrica_outro
