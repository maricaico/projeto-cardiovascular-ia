"""
Módulo de Modelagem: Árvores de Decisão e Métodos de Ensemble.
Suporta treinamento, busca em grade/validação cruzada e extração de regras para
Classificação e Regressão.
"""

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold

def treinar_arvore_classificacao(X_train, y_train, cv=5, random_state=42):
    """Árvores de classificação com pré-processamento DENTRO da CV.

    Recebe X_train BRUTO (com categorias e valores ausentes), sem X_test.
    Compara Gini/Entropia e diversos controles de complexidade usando F1.
    Treina também uma árvore sem restrições para avaliar generalização.
    """
    from .pipeline_data import criar_pipeline_modelo

    arvore_irrestrita = criar_pipeline_modelo(
        DecisionTreeClassifier(random_state=random_state)
    )
    arvore_irrestrita.fit(X_train, y_train)

    parametros = {
        "modelo__criterion": ["gini", "entropy"],
        "modelo__max_depth": [3, 5, 7, 10, None],
        "modelo__min_samples_leaf": [5, 10, 20, 50],
        "modelo__ccp_alpha": [0.0, 0.001, 0.005],
    }
    cv_strat = StratifiedKFold(
        n_splits=cv, shuffle=True, random_state=random_state
    )
    grid = GridSearchCV(
        estimator=criar_pipeline_modelo(
            DecisionTreeClassifier(random_state=random_state)
        ),
        param_grid=parametros,
        cv=cv_strat,
        scoring="f1",
        n_jobs=2,
        return_train_score=True,
        error_score="raise",
    )
    grid.fit(X_train, y_train)

    indice = grid.best_index_
    return {
        "modelo_ajustado": grid.best_estimator_,  # pipeline completo, já fitted
        "melhores_parametros": grid.best_params_,
        "melhor_score_cv": float(grid.best_score_),
        "desvio_score_cv": float(grid.cv_results_["std_test_score"][indice]),
        "score_treino_medio_cv": float(grid.cv_results_["mean_train_score"][indice]),
        "modelo_irrestrito": arvore_irrestrita,  # pipeline completo
        "grid_cv": grid,
    }


def treinar_arvore_regressao(X_train, y_train, cv=5, random_state=42):
    """Árvores de regressão com pré-processamento DENTRO da CV.

    Recebe X_train BRUTO, usa a mesma divisão do problema de classificação.
    Compara squared_error (MSE) e friedman_mse, otimizando RMSE médio.
    """
    from .pipeline_data import criar_pipeline_modelo

    arvore_irrestrita = criar_pipeline_modelo(
        DecisionTreeRegressor(random_state=random_state)
    )
    arvore_irrestrita.fit(X_train, y_train)

    parametros = {
        "modelo__criterion": ["squared_error", "friedman_mse"],
        "modelo__max_depth": [5, 10, 15, None],
        "modelo__min_samples_leaf": [5, 10, 20],
        "modelo__ccp_alpha": [0.0, 0.001],
    }
    cv_kfold = KFold(
        n_splits=cv, shuffle=True, random_state=random_state
    )
    grid = GridSearchCV(
        estimator=criar_pipeline_modelo(
            DecisionTreeRegressor(random_state=random_state)
        ),
        param_grid=parametros,
        cv=cv_kfold,
        scoring="neg_root_mean_squared_error",
        n_jobs=2,
        return_train_score=True,
        error_score="raise",
    )
    grid.fit(X_train, y_train)

    indice = grid.best_index_
    return {
        "modelo_ajustado": grid.best_estimator_,  # pipeline completo, já fitted
        "melhores_parametros": grid.best_params_,
        "melhor_score_cv": float(-grid.best_score_),  # RMSE, menor é melhor
        "desvio_score_cv": float(grid.cv_results_["std_test_score"][indice]),
        "score_treino_medio_cv": float(-grid.cv_results_["mean_train_score"][indice]),
        "modelo_irrestrito": arvore_irrestrita,  # pipeline completo
        "grid_cv": grid,
    }


# =============================================================
# Métodos Ensemble — Atividade 02
# Os quatro treinamentos recebem X_train BRUTO e nunca usam X_test.
# O pré-processamento é ajustado DENTRO de cada fold de CV.
# =============================================================


def _ajustar_ensemble(
    X_train, y_train, modelo, parametros, tarefa, cv=5, random_state=42
):
    """Busca de hiperparâmetros em Pipeline; protocolo comum aos ensembles.

    tarefa='classificacao' -> F1 com StratifiedKFold.
    tarefa='regressao' -> RMSE com KFold.
    No retorno, RMSE é positivo (menor é melhor), como nas árvores.
    """
    from .pipeline_data import criar_pipeline_modelo

    if tarefa == "classificacao":
        divisao_cv = StratifiedKFold(
            n_splits=cv, shuffle=True, random_state=random_state
        )
        pontuacao = "f1"
    elif tarefa == "regressao":
        divisao_cv = KFold(
            n_splits=cv, shuffle=True, random_state=random_state
        )
        pontuacao = "neg_root_mean_squared_error"
    else:
        raise ValueError("tarefa deve ser 'classificacao' ou 'regressao'.")

    grid = GridSearchCV(
        estimator=criar_pipeline_modelo(modelo),
        param_grid=parametros,
        cv=divisao_cv,
        scoring=pontuacao,
        n_jobs=2,
        return_train_score=True,
        error_score="raise",
        refit=True,
    )
    grid.fit(X_train, y_train)

    indice = grid.best_index_
    converter = (lambda v: -float(v)) if tarefa == "regressao" else float

    return {
        "modelo_ajustado": grid.best_estimator_,  # Pipeline completo, treinado
        "melhores_parametros": grid.best_params_,
        "melhor_score_cv": converter(grid.best_score_),
        "desvio_score_cv": float(grid.cv_results_["std_test_score"][indice]),
        "score_treino_medio_cv": converter(
            grid.cv_results_["mean_train_score"][indice]
        ),
        "grid_cv": grid,
    }


def treinar_random_forest_classificacao(X_train, y_train, cv=5, random_state=42):
    """Random Forest para classificação, com F1 e validação estratificada.

    Busca 16 combinações. Com 6 features, usar 'sqrt' e 'log2'
    produziria o mesmo número de features em cada divisão; por isso
    comparamos 'sqrt' com todas as features (1.0).
    """
    parametros = {
        "modelo__n_estimators": [50, 100],
        "modelo__max_depth": [6, None],
        "modelo__max_features": ["sqrt", 1.0],
        "modelo__min_samples_leaf": [4, 10],
    }
    return _ajustar_ensemble(
        X_train, y_train,
        modelo=RandomForestClassifier(random_state=random_state, n_jobs=1),
        parametros=parametros,
        tarefa="classificacao", cv=cv, random_state=random_state,
    )


def treinar_random_forest_regressao(X_train, y_train, cv=5, random_state=42):
    """Random Forest para regressão, com RMSE e KFold."""
    parametros = {
        "modelo__n_estimators": [50, 100],
        "modelo__max_depth": [6, None],
        "modelo__max_features": ["sqrt", 1.0],
        "modelo__min_samples_leaf": [4, 10],
    }
    return _ajustar_ensemble(
        X_train, y_train,
        modelo=RandomForestRegressor(random_state=random_state, n_jobs=1),
        parametros=parametros,
        tarefa="regressao", cv=cv, random_state=random_state,
    )


def treinar_gradient_boosting_classificacao(
    X_train, y_train, cv=5, random_state=42
):
    """Gradient Boosting para classificação, otimizando F1 com StratifiedKFold.

    A grade compara quantidade de árvores, taxa de aprendizado,
    profundidade das árvores base e fração amostrada por etapa.
    """
    parametros = {
        "modelo__n_estimators": [50, 100],
        "modelo__learning_rate": [0.05, 0.1],
        "modelo__max_depth": [2, 3],
        "modelo__subsample": [0.8, 1.0],
    }
    return _ajustar_ensemble(
        X_train, y_train,
        modelo=GradientBoostingClassifier(random_state=random_state),
        parametros=parametros,
        tarefa="classificacao", cv=cv, random_state=random_state,
    )


def treinar_gradient_boosting_regressao(
    X_train, y_train, cv=5, random_state=42
):
    """Gradient Boosting para regressão, otimizando RMSE com KFold."""
    parametros = {
        "modelo__n_estimators": [50, 100],
        "modelo__learning_rate": [0.05, 0.1],
        "modelo__max_depth": [2, 3],
        "modelo__subsample": [0.8, 1.0],
    }
    return _ajustar_ensemble(
        X_train, y_train,
        modelo=GradientBoostingRegressor(random_state=random_state),
        parametros=parametros,
        tarefa="regressao", cv=cv, random_state=random_state,
    )
