"""Métricas da Atividade 02 (Aula 08) — classificação e regressão.

Na classificação, PR-AUC (área trapezoidal) e Average Precision (AP)
são indicadores relacionados, mas distintos. ``y_prob`` aceita escores
contínuos de classificação: eles não precisam ser probabilidades calibradas.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
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
    """Retorna MAE, MSE, RMSE, MAPE (proporção), R2 e MedAE."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "MedAE": median_absolute_error(y_true, y_pred),
    }


def calcular_metricas_classificacao(y_true, y_pred, y_prob=None):
    """Calcula métricas binárias com a classe positiva fixada em 1.

    ``y_prob`` é um escore contínuo para a classe 1; também pode ser uma
    frequência empírica estimada no treino, como no sistema especialista.
    ROC-AUC e PR-AUC avaliam discriminação, NÃO calibração probabilística.

    A área PR-AUC é calculada por integração trapezoidal de Precision x Recall;
    AP é a média das precisões ponderada pelo aumento do Recall.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if len(y_true) == 0 or len(y_pred) != len(y_true):
        raise ValueError("y_true e y_pred devem ter tamanhos iguais e não vazios.")
    if not set(np.unique(y_true)).issubset({0, 1}):
        raise ValueError("y_true deve conter somente classes 0 e 1.")
    if not set(np.unique(y_pred)).issubset({0, 1}):
        raise ValueError("y_pred deve conter somente classes 0 e 1.")

    resultado = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "Recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "Confusion_Matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]),
        "ROC-AUC": np.nan,
        "PR-AUC": np.nan,
        "Average_Precision": np.nan,
    }

    if y_prob is not None:
        scores = np.asarray(y_prob).ravel()
        if len(scores) != len(y_true) or not np.all(np.isfinite(scores)):
            raise ValueError("Os escores devem ser finitos e alinhados às classes.")
        if np.unique(y_true).size != 2:
            raise ValueError("ROC-AUC/PR-AUC requerem exemplos das duas classes.")
        precisao_curva, revocacao_curva, _ = precision_recall_curve(
            y_true, scores, pos_label=1
        )
        # recall retorna ordenado de 1 a 0; inverter para integrar de 0 a 1.
        resultado["PR-AUC"] = auc(
            revocacao_curva[::-1], precisao_curva[::-1]
        )
        resultado["ROC-AUC"] = roc_auc_score(y_true, scores)
        resultado["Average_Precision"] = average_precision_score(y_true, scores)

    return resultado


def calcular_gap_generalizacao(metric_train, metric_test, is_error_metric=False):
    """Gap de qualidade: treino - teste; erro: teste - treino."""
    if is_error_metric:
        return metric_test - metric_train
    return metric_train - metric_test
