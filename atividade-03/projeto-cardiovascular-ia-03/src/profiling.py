"""Atividade 03 (DCT1401): medição de tempo, memória, complexidade e energia.

Funções usadas no Notebook 08 para medir o custo computacional dos modelos.

- Tempo: time.perf_counter, com repetições de aquecimento (warm-up) antes de
  medir e torch.cuda.synchronize() quando o modelo está em GPU.
- Memória: psutil (memória residente, RSS). O tracemalloc não é usado porque
  não contabiliza a memória alocada pelo código nativo do PyTorch.
- Energia: potência média × tempo, convertida em emissão de CO2 com o fator
  médio de emissão da eletricidade no Brasil adotado no projeto.
"""

import io
import threading
import time

import numpy as np
import psutil
import torch
import torch.nn as nn

FATOR_EMISSAO_BRASIL_G_KWH = 45.8   # gCO2/kWh (fator médio adotado para o Brasil)


class MonitorMemoria:
    """Mede o pico de memória residente (RSS) do processo durante um bloco.

    Uma thread lê o RSS a cada `intervalo_s` segundos. O RSS inclui tudo o que
    o processo ocupa (Python, bibliotecas, dados e tensores); por isso, o valor
    mais informativo é o acréscimo durante o bloco (pico − valor inicial).

    Uso:
        with MonitorMemoria() as mem:
            ...código medido...
        mem.inicial_mb, mem.pico_mb, mem.acrescimo_mb
    """

    def __init__(self, intervalo_s=0.002):
        self.intervalo_s = intervalo_s
        self._proc = psutil.Process()
        self._parar = threading.Event()

    def _amostrar(self):
        while not self._parar.is_set():
            self._pico = max(self._pico, self._proc.memory_info().rss)
            time.sleep(self.intervalo_s)

    def __enter__(self):
        self._inicial = self._proc.memory_info().rss
        self._pico = self._inicial
        self._thread = threading.Thread(target=self._amostrar, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._parar.set()
        self._thread.join()
        self._pico = max(self._pico, self._proc.memory_info().rss)
        self.inicial_mb = self._inicial / 2**20
        self.pico_mb = self._pico / 2**20
        self.acrescimo_mb = (self._pico - self._inicial) / 2**20
        return False


def cronometrar(funcao, n_rep=30, n_aquecimento=5, sincronizar_gpu=False):
    """Executa `funcao()` n_aquecimento vezes sem medir e n_rep vezes medindo.

    Retorna os tempos (s) de cada repetição.
    """
    for _ in range(n_aquecimento):
        funcao()
    if sincronizar_gpu:
        torch.cuda.synchronize()
    tempos = []
    for _ in range(n_rep):
        inicio = time.perf_counter()
        funcao()
        if sincronizar_gpu:
            torch.cuda.synchronize()   # a GPU é assíncrona: espera terminar
        tempos.append(time.perf_counter() - inicio)
    return np.asarray(tempos)


def resumir_tempos(tempos, n_amostras):
    """Média e desvio (ddof=1) do tempo, tempo por amostra e vazão."""
    media = float(np.mean(tempos))
    return {
        "tempo_medio_ms": 1000 * media,
        "tempo_desvio_ms": 1000 * float(np.std(tempos, ddof=1)),
        "tempo_por_amostra_us": 1e6 * media / n_amostras,
        "vazao_amostras_s": n_amostras / media,
    }


def medir_inferencia(modelo, X, n_rep=30, n_aquecimento=5, individual=False):
    """Mede tempo e memória apenas da etapa de inferência do modelo.

    - individual=False: o conjunto X inteiro em uma única chamada (lote);
    - individual=True: uma amostra por chamada, percorrendo todo o X em cada
      repetição.
    """
    modelo.eval()
    em_gpu = any(p.is_cuda for p in modelo.parameters())   # modelos quantizados podem não ter parâmetros
    if individual:
        linhas = [X[i:i + 1] for i in range(len(X))]

        def funcao():
            for x in linhas:
                modelo(x)
    else:
        def funcao():
            modelo(X)

    with torch.inference_mode(), MonitorMemoria() as mem:
        tempos = cronometrar(funcao, n_rep, n_aquecimento, sincronizar_gpu=em_gpu)
    return {**resumir_tempos(tempos, len(X)), "n_amostras": len(X),
            "memoria_inicial_mb": mem.inicial_mb, "acrescimo_memoria_mb": mem.acrescimo_mb}


def macs_por_amostra(modelo):
    """Multiplicações-acumulações (MACs) das camadas lineares por amostra.

    Cada nn.Linear(entrada, saída) faz entrada × saída multiplicações e somas;
    as ativações e os vieses acrescentam operações de ordem bem menor.
    """
    return sum(m.in_features * m.out_features for m in modelo.modules() if isinstance(m, nn.Linear))


def resumo_complexidade(modelo):
    lineares = [m for m in modelo.modules() if isinstance(m, nn.Linear)]
    return {
        "parametros_treinaveis": sum(p.numel() for p in modelo.parameters() if p.requires_grad),
        "camadas_lineares": len(lineares),
        "camadas_ocultas": len(lineares) - 1,
        "macs_por_amostra": macs_por_amostra(modelo),
    }


def tamanho_serializado_bytes(objeto):
    """Tamanho, em bytes, de `torch.save(objeto)` (sem gravar em disco)."""
    buffer = io.BytesIO()
    torch.save(objeto, buffer)
    return buffer.getbuffer().nbytes


def energia_kwh(potencia_w, tempo_s):
    """Energia (kWh) = potência média (W) × tempo (s) / 3,6 × 10^6."""
    return potencia_w * tempo_s / 3.6e6


def emissao_co2_g(energia, fator_g_kwh=FATOR_EMISSAO_BRASIL_G_KWH):
    """Emissão (gCO2) = energia (kWh) × fator de emissão (gCO2/kWh)."""
    return energia * fator_g_kwh
