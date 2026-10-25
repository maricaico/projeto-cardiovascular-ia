"""Sistema especialista acadêmico para Cardiovascular_Disease (classificação).

Fonte única das regras: rules/regras_classificacao.txt.
As regras são decisões humanas explícitas; não se treinam árvores/modelos aqui.
As frequências utilizadas como ESCORES são medidas somente no treinamento,
não são probabilidades clínicas calibradas.
"""
from __future__ import annotations

import operator
import re
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES_PERMITIDAS = frozenset({
    "Diabetes", "Hypertension", "Chest_Pain", "ECG_Result",
    "Blood_Glucose", "Blood_Pressure_Systolic",
})
OPERADORES = {"==": operator.eq, ">": operator.gt, "<": operator.lt,
              ">=": operator.ge, "<=": operator.le}
TIPOS_PERMITIDOS = {"Relação", "Recomendação", "Diretriz", "Estratégia", "Heurística"}
_LINHA = re.compile(
    r"^(RC\d{2})\s*\|\s*([^|]+)\s*\|\s*SE\s+(.+?)\s+ENTÃO\s+"
    r"Cardiovascular_Disease\s*=\s*([01])\s*$"
)
_CONDICAO = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*(==|>=|<=|>|<)\s*(-?\d+(?:\.\d+)?)$")


class BaseConhecimentoClassificacao:
    """Carrega regras SE–ENTÃO e calcula evidências exclusivamente no treino.

    A ordem do arquivo constitui a prioridade. Regras com conclusão conflitante
    não são combinadas: a primeira correspondente prevalece.
    """

    def __init__(self, arquivo_regras):
        self.arquivo_regras = Path(arquivo_regras)
        self.regras = self._carregar_regras()
        self.estatisticas = None
        self.classe_padrao = None
        self.escore_padrao = None

    def _carregar_regras(self):
        regras = []
        ids = set()
        for numero, texto in enumerate(self.arquivo_regras.read_text(encoding="utf-8").splitlines(), 1):
            texto = texto.strip()
            if not texto or texto.startswith("#"):
                continue
            resultado = _LINHA.fullmatch(texto)
            if resultado is None:
                raise ValueError(f"Regra malformada na linha {numero}: {texto}")
            codigo, tipo, antecedente, classe = resultado.groups()
            tipo = tipo.strip()
            if codigo in ids:
                raise ValueError(f"Regra duplicada: {codigo}")
            if tipo not in TIPOS_PERMITIDOS:
                raise ValueError(f"Tipo não reconhecido em {codigo}: {tipo}")
            if re.search(r"\bOU\b", antecedente):
                raise ValueError(f"OU não é permitido neste conjunto de regras: {codigo}")
            condicoes = []
            for item in re.split(r"\s+E\s+", antecedente):
                m = _CONDICAO.fullmatch(item.strip())
                if m is None:
                    raise ValueError(f"Condição inválida em {codigo}: {item}")
                coluna, simbolo, valor = m.groups()
                if coluna not in FEATURES_PERMITIDAS:
                    raise ValueError(f"Feature fora das seis selecionadas: {coluna}")
                condicoes.append((coluna, simbolo, float(valor)))
            regras.append({"id": codigo, "tipo": tipo, "condicoes": condicoes,
                           "conclusao": int(classe), "ordem": len(regras) + 1})
            ids.add(codigo)
        if not regras:
            raise ValueError("Nenhuma regra válida foi encontrada.")
        return regras

    @staticmethod
    def mascara(X, regra):
        resultado = pd.Series(True, index=X.index)
        for coluna, simbolo, limite in regra["condicoes"]:
            if coluna not in X.columns:
                raise ValueError(f"Coluna ausente: {coluna}")
            resultado &= OPERADORES[simbolo](X[coluna], limite)
        return resultado

    def analisar_no_treinamento(self, X_treino, y_treino):
        """Apura suporte bruto e efetivo, frequências e fallback (apenas treino).

        Suporte bruto: registros que satisfazem a regra, mesmo que outra vença.
        Suporte efetivo: registros em que a regra é a PRIMEIRA correspondente.
        """
        if not X_treino.index.is_unique:
            raise ValueError("Os índices de treinamento precisam ser únicos.")
        y = pd.Series(y_treino).reindex(X_treino.index)
        if y.isna().any() or not set(y.unique()).issubset({0, 1}):
            raise ValueError("O target de treinamento deve estar alinhado e conter 0/1.")
        disponiveis = pd.Series(True, index=X_treino.index)
        resultados = []
        for regra in self.regras:
            mascara_bruta = self.mascara(X_treino, regra)
            mascara_efetiva = mascara_bruta & disponiveis
            n_bruto = int(mascara_bruta.sum())
            n_efetivo = int(mascara_efetiva.sum())
            y_bruto = y.loc[mascara_bruta]
            y_efetivo = y.loc[mascara_efetiva]
            positivos_brutos = int(y_bruto.sum())
            positivos_efetivos = int(y_efetivo.sum())
            # Escores para ROC/PR: P(classe 1 | a regra FOI ESCOLHIDA), no treino.
            # Não interpretar como probabilidade clínica calibrada.
            escore = (positivos_efetivos / n_efetivo) if n_efetivo else float(y.mean())
            acertos = (positivos_efetivos if regra["conclusao"] == 1
                       else n_efetivo - positivos_efetivos)
            resultados.append({
                "Regra": regra["id"], "Tipo": regra["tipo"],
                "Classe": regra["conclusao"], "Ordem": regra["ordem"],
                "Cobertura_bruta": n_bruto, "Positivos_brutos": positivos_brutos,
                "Cobertura_efetiva": n_efetivo, "Acertos_efetivos": acertos,
                "Frequencia_acerto_efetivo": (acertos / n_efetivo) if n_efetivo else np.nan,
                "Escore_positivo": escore,
            })
            disponiveis &= ~mascara_bruta
        y_restante = y.loc[disponiveis]
        if len(y_restante):
            self.escore_padrao = float(y_restante.mean())
            self.classe_padrao = int(self.escore_padrao >= 0.5)
        else:
            self.escore_padrao = float(y.mean())
            self.classe_padrao = int(self.escore_padrao >= 0.5)
        self.estatisticas = pd.DataFrame(resultados)
        self.quantidade_sem_regra = int(disponiveis.sum())
        return self.estatisticas.copy()


class MotorInferenciaClassificacao:
    """Aplica regras em ordem de precedência, sem acessar rótulos de teste."""

    def __init__(self, base_conhecimento):
        if base_conhecimento.estatisticas is None:
            raise ValueError("Analise a base SOMENTE no treino antes de criar o motor.")
        self.base = base_conhecimento

    def inferir(self, X):
        if not X.index.is_unique:
            raise ValueError("Os índices precisam ser únicos.")
        n = len(X)
        preds = np.full(n, self.base.classe_padrao, dtype=int)
        scores = np.full(n, self.base.escore_padrao, dtype=float)
        disparadas = np.full(n, "FALLBACK", dtype=object)
        livres = np.ones(n, dtype=bool)
        for regra, (_, info) in zip(self.base.regras, self.base.estatisticas.iterrows()):
            ativou = self.base.mascara(X, regra).to_numpy() & livres
            preds[ativou] = regra["conclusao"]
            scores[ativou] = float(info["Escore_positivo"])
            disparadas[ativou] = regra["id"]
            livres[ativou] = False
        return preds, scores, disparadas

    def predict(self, X):
        return self.inferir(X)[0]

    def predict_proba(self, X):
        scores = self.inferir(X)[1]
        return np.column_stack((1 - scores, scores))

    def regras_disparadas(self, X):
        return self.inferir(X)[2]
