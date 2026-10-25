"""Medidas de eficiência computacional do projeto cardiovascular (Atividade 02).

- Medição padronizada de inferência, reutilizável por todas as famílias de modelos.
- tracemalloc acompanha alocações de memória geridas pelo Python; NÃO mede toda a RAM.
- Energia/CO2 são ESTIMATIVAS derivadas de potência assumida, não leituras do hardware.
"""
from __future__ import annotations

import os
import time
import tracemalloc

import numpy as np

FATOR_EMISSAO_BRASIL_GCO2_KWH = 45.8  # g CO2 / kWh, valor fornecido na atividade
POTENCIA_PADRAO_WATTS = 45.0  # HIPÓTESE; não é medição elétrica


def medir_inferencia(funcao_predicao, X, n_rep=30, n_aquecimento=1, retornar_detalhes=False):
    """Mede exclusivamente a inferência, com o mesmo protocolo em todos os modelos.

    Parameters
    ----------
    funcao_predicao : callable
        Recebe X e gera previsões. Use, por exemplo, motor.predict.
    X : array-like
        Conjunto já preparado para o modelo. Não há ajuste de regras aqui.
    n_rep : int
        Número de repetições cronometradas (30 para a atividade).
    n_aquecimento : int
        Execuções iniciais não contabilizadas, úteis para inicializar bibliotecas.
    retornar_detalhes : bool
        Se True, inclui os resultados individuais em 'detalhes', para exportar CSV.

    Notas
    -----
    A medição de memória captura o pico de alocações Python em cada repetição.
    O rastreamento é ativo durante a cronometragem, afetando o tempo medido;
    por isso, aplique este mesmo procedimento a TODOS os modelos comparados.
    """
    if not callable(funcao_predicao):
        raise TypeError("funcao_predicao deve ser uma função ou método chamável.")
    if not isinstance(n_rep, int) or n_rep < 2:
        raise ValueError("n_rep deve ser inteiro >= 2, para calcular desvio padrão.")
    if not isinstance(n_aquecimento, int) or n_aquecimento < 0:
        raise ValueError("n_aquecimento deve ser inteiro >= 0.")
    n_amostras = len(X)
    if n_amostras == 0:
        raise ValueError("Não é possível medir inferência com X vazio.")
    if tracemalloc.is_tracing():
        raise RuntimeError("Há outro rastreamento tracemalloc ativo; encerre-o antes de medir.")

    for _ in range(n_aquecimento):
        funcao_predicao(X)

    tempos_s = []
    picos_kb = []
    for _ in range(n_rep):
        tracemalloc.start()
        try:
            inicio = time.perf_counter()
            funcao_predicao(X)
            duracao = time.perf_counter() - inicio
            _, pico_bytes = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        tempos_s.append(float(duracao))
        picos_kb.append(float(pico_bytes / 1024.0))

    media_tempo = float(np.mean(tempos_s))
    vazoes = [n_amostras / t if t > 0 else float("nan") for t in tempos_s]
    resultado = {
        "n_repeticoes": n_rep,
        "n_amostras": n_amostras,
        "tempo_medio_s": media_tempo,
        "tempo_desvio_s": float(np.std(tempos_s, ddof=1)),
        "tempo_por_amostra_ms": 1000.0 * media_tempo / n_amostras,
        "vazao_amostras_s": n_amostras / media_tempo if media_tempo > 0 else float("nan"),
        "vazao_desvio_amostras_s": float(np.nanstd(vazoes, ddof=1)),
        "pico_memoria_kb": float(np.max(picos_kb)),
        "pico_memoria_medio_kb": float(np.mean(picos_kb)),
        "pico_memoria_desvio_kb": float(np.std(picos_kb, ddof=1)),
    }
    if retornar_detalhes:
        resultado["detalhes"] = [
            {
                "repeticao": i + 1,
                "tempo_s": tempos_s[i],
                "pico_memoria_python_kb": picos_kb[i],
                "vazao_amostras_s": vazoes[i],
            }
            for i in range(n_rep)
        ]
    return resultado


def medir_treinamento(funcao_fit, *args, **kwargs):
    """Mede fit computacional; não inclui seleção CV externa à função recebida."""
    tracemalloc.start()
    try:
        inicio = time.perf_counter()
        modelo = funcao_fit(*args, **kwargs)
        tempo_total = time.perf_counter() - inicio
        _, pico = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return modelo, {
        "tempo_treino_s": float(tempo_total),
        "pico_memoria_treino_kb": pico / 1024.0,
    }


def estimar_energia_carbono(tempo_segundos, potencia_watts=POTENCIA_PADRAO_WATTS):
    """Estima kWh e g CO2 para uma execução de duração informada."""
    if tempo_segundos < 0 or potencia_watts < 0:
        raise ValueError("Tempo e potência devem ser não negativos.")
    energia_kwh = potencia_watts * tempo_segundos / 3_600_000.0
    return {
        "energia_kwh": energia_kwh,
        "emissao_co2_g": energia_kwh * FATOR_EMISSAO_BRASIL_GCO2_KWH,
    }


def obter_tamanho_modelo_disco(objeto, caminho_temporario="temp_model.joblib"):
    """Compatibilidade com o fluxo atual: mede tamanho serializado em KB."""
    import joblib
    try:
        joblib.dump(objeto, caminho_temporario)
        tamanho_bytes = os.path.getsize(caminho_temporario)
    finally:
        if os.path.exists(caminho_temporario):
            os.remove(caminho_temporario)
    return tamanho_bytes / 1024.0
