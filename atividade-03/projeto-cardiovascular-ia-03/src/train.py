"""Atividade 03 (DCT1401): sementes, tensores, DataLoader e laço de treinamento.

O laço segue o padrão: zero_grad -> forward -> perda ->
backward -> step, avaliação na validação a cada época e early stopping com
restauração do melhor estado. Também registra:
- registro da época do melhor estado, das épocas executadas e do tempo;
- detecção de divergência (perda NaN/inf), necessária nos testes com taxa
  de aprendizado alta;
- uma função de alto nível que executa UMA configuração com UMA semente,
  que é a unidade de todos os experimentos.
"""

import copy
import random
import time

import numpy as np
import torch
from scipy.special import expit
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from . import metrics as metricas
from . import pipeline_data as dados_mod
from .models import MLP, contar_parametros


# Configuração de referência (baseline).
CONFIG_BASELINE = {
    "hidden": (32, 32),        # 2 camadas ocultas de 32 neurônios
    "activation": "ReLU",
    "dropout": 0.0,
    "otimizador": "Adam",
    "lr": 1e-3,
    "batch_size": 32,
    "weight_decay": 0.0,
    "n_epocas": 200,
    "paciencia": 20,
    "pos_weight": False,       # só tem efeito na classificação
}

# Cinco sementes. Controlam a inicialização dos pesos
# e a ordem dos lotes; a divisão dos dados usa sempre random_state=42.
SEMENTES = (0, 1, 2, 3, 4)

TAMANHO_LOTE_AVALIACAO = 4096  # só para avaliar; não altera o resultado


# ---------------------------------------------------------------------------
# Reprodutibilidade
# ---------------------------------------------------------------------------

def fixar_sementes(semente):
    """Fixa as sementes de random, numpy e torch."""
    random.seed(semente)
    np.random.seed(semente)
    torch.manual_seed(semente)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semente)


# ---------------------------------------------------------------------------
# Dados -> arrays -> tensores
# ---------------------------------------------------------------------------

def preparar_dados_tarefa(dados, tarefa, incluir_teste=False):
    """Ajusta o pré-processamento no AJUSTE e transforma os demais conjuntos.

    tarefa : "regressao" ou "classificacao".
    incluir_teste : False durante todo o estudo; True só na avaliação final.
        Transformar o teste não vaza informação (nada é ajustado nele), mas
        deixá-lo de fora torna explícito que ele não participa das escolhas.
    """
    if tarefa not in ("regressao", "classificacao"):
        raise ValueError("tarefa deve ser 'regressao' ou 'classificacao'.")
    conjuntos = ["ajuste", "validacao"] + (["teste"] if incluir_teste else [])

    pre = dados_mod.criar_preprocessador_mlp().fit(dados["X_ajuste"])
    X = {c: pre.transform(dados[f"X_{c}"]).astype(np.float32) for c in conjuntos}

    if tarefa == "regressao":
        escala = dados_mod.ajustar_escala_alvo(dados["y_ajuste_r"])
        y_original = {c: dados[f"y_{c}_r"].to_numpy(np.float64) for c in conjuntos}
        y = {c: dados_mod.padronizar_alvo(y_original[c], escala).astype(np.float32)
             for c in conjuntos}
    else:
        escala = None
        y_original = {c: dados[f"y_{c}_c"].to_numpy(np.int64) for c in conjuntos}
        y = {c: y_original[c].astype(np.float32) for c in conjuntos}

    return {
        "tarefa": tarefa,
        "X": X,
        "y": y,                    # alvo usado na perda (padronizado na regressão)
        "y_original": y_original,  # alvo na escala original, para as métricas
        "preprocessador": pre,
        "estatisticas": dados_mod.estatisticas_preprocessador(pre),
        "escala_alvo": escala,
        "n_positivos_ajuste": int(dados["y_ajuste_c"].sum()),
        "n_negativos_ajuste": int(len(dados["y_ajuste_c"]) - dados["y_ajuste_c"].sum()),
    }


def criar_dataloader(X, y, batch_size, embaralhar, semente=None):
    """TensorDataset + DataLoader com float32 (BCEWithLogitsLoss e MSELoss).

    O alvo vira uma coluna (n, 1) para ter o mesmo formato da saída da rede.
    O embaralhamento usa um gerador com semente própria: a ordem dos lotes
    é reproduzível e depende só da semente do experimento.
    """
    dataset = TensorDataset(
        torch.as_tensor(X, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.float32).reshape(-1, 1),
    )
    gerador = torch.Generator().manual_seed(int(semente)) if embaralhar else None
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=embaralhar,
                      generator=gerador, num_workers=0)


# ---------------------------------------------------------------------------
# Perda e otimizador
# ---------------------------------------------------------------------------

def criar_perda(tarefa, usar_pos_weight=False, n_positivos=None, n_negativos=None, device="cpu"):
    """MSELoss na regressão; BCEWithLogitsLoss na classificação.

    pos_weight = negativos / positivos, calculado no subconjunto de AJUSTE.
    """
    if tarefa == "regressao":
        return nn.MSELoss()
    if usar_pos_weight:
        peso = torch.tensor([n_negativos / n_positivos], dtype=torch.float32, device=device)
        return nn.BCEWithLogitsLoss(pos_weight=peso)
    return nn.BCEWithLogitsLoss()


def criar_otimizador(nome, parametros, lr, weight_decay=0.0):
    """Otimizadores comparados no estudo de hiperparâmetros."""
    if nome == "SGD":
        return torch.optim.SGD(parametros, lr=lr, weight_decay=weight_decay)
    if nome == "SGD_momentum":
        return torch.optim.SGD(parametros, lr=lr, momentum=0.9, weight_decay=weight_decay)
    if nome == "RMSprop":
        return torch.optim.RMSprop(parametros, lr=lr, weight_decay=weight_decay)
    if nome == "Adam":
        return torch.optim.Adam(parametros, lr=lr, weight_decay=weight_decay)
    if nome == "AdamW":
        return torch.optim.AdamW(parametros, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Otimizador desconhecido: {nome}")


# ---------------------------------------------------------------------------
# Laço de treinamento com early stopping
# ---------------------------------------------------------------------------

def avaliar_perda(modelo, dl, loss_fn, device="cpu"):
    """Perda média por amostra, em modo de avaliação e sem gradiente."""
    modelo.eval()
    soma = 0.0
    with torch.no_grad():
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            soma += loss_fn(modelo(xb), yb).item() * len(xb)
    return soma / len(dl.dataset)


def treinar(modelo, dl_tr, dl_val, loss_fn, otimizador,
            n_epocas=200, paciencia=20, min_delta=1e-6, device="cpu"):
    """Treina e restaura o estado com MENOR perda de validação.

    Retorna o histórico e: melhor perda de validação, época do melhor estado,
    épocas executadas (= época de parada), tempo de parede e se divergiu.
    A perda de treino de cada época é a média dos lotes durante a época.
    """
    modelo.to(device)
    hist = {"treino": [], "val": []}
    melhor, espera, melhor_estado, epoca_melhor = float("inf"), 0, None, 0
    divergiu = False

    if device != "cpu":
        torch.cuda.synchronize()
    inicio = time.perf_counter()

    for ep in range(n_epocas):
        modelo.train()
        soma = 0.0
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            otimizador.zero_grad()
            perda = loss_fn(modelo(xb), yb)
            perda.backward()
            otimizador.step()
            soma += perda.item() * len(xb)
        hist["treino"].append(soma / len(dl_tr.dataset))

        val = avaliar_perda(modelo, dl_val, loss_fn, device)
        hist["val"].append(val)

        pesos_finitos = all(torch.isfinite(p).all() for p in modelo.parameters())
        if not (np.isfinite(val) and np.isfinite(hist["treino"][-1]) and pesos_finitos):
            divergiu = True   # perda ou pesos não finitos: interrompe e registra
            break
        if val < melhor - min_delta:
            melhor, espera, epoca_melhor = val, 0, ep + 1
            melhor_estado = copy.deepcopy(modelo.state_dict())
        else:
            espera += 1
            if espera >= paciencia:
                break

    if device != "cpu":
        torch.cuda.synchronize()
    tempo = time.perf_counter() - inicio

    if melhor_estado is not None:
        modelo.load_state_dict(melhor_estado)  # restaura o melhor ponto
    return {
        "hist": hist,
        "melhor_val": melhor,
        "epoca_melhor": epoca_melhor,
        "epocas_executadas": len(hist["val"]),
        "tempo_s": tempo,
        "divergiu": divergiu,
    }


def prever(modelo, X, device="cpu"):
    """Saída bruta da rede (valor padronizado ou logit), em modo de avaliação."""
    modelo.eval()
    X_t = torch.as_tensor(X, dtype=torch.float32)
    saidas = []
    with torch.inference_mode():
        for i in range(0, len(X_t), TAMANHO_LOTE_AVALIACAO):
            saidas.append(modelo(X_t[i:i + TAMANHO_LOTE_AVALIACAO].to(device)).cpu())
    return torch.cat(saidas).squeeze(1).numpy()


# ---------------------------------------------------------------------------
# Uma configuração × uma semente (unidade dos experimentos)
# ---------------------------------------------------------------------------

def metricas_conjunto(modelo, dt, conjunto, device="cpu", limiar=0.5):
    """Métricas de um conjunto na escala original (regressão) ou com limiar."""
    saida = prever(modelo, dt["X"][conjunto], device)
    if dt["tarefa"] == "regressao":
        y_pred = dados_mod.despadronizar_alvo(saida, dt["escala_alvo"])
        return metricas.calcular_metricas_regressao(dt["y_original"][conjunto], y_pred)
    prob = expit(saida.astype(np.float64))  # sigmoide numericamente estável
    return metricas.calcular_metricas_classificacao(dt["y_original"][conjunto], prob, limiar)


def executar_configuracao(dt, config, semente, device="cpu"):
    """Treina UMA configuração com UMA semente e devolve modelo e registros.

    dt : saída de preparar_dados_tarefa().
    config : dicionário no formato de CONFIG_BASELINE.
    """
    cfg = {**CONFIG_BASELINE, **config}
    fixar_sementes(semente)

    modelo = MLP(in_dim=dt["X"]["ajuste"].shape[1], out_dim=1,
                 hidden=tuple(cfg["hidden"]), activation=cfg["activation"],
                 dropout=cfg["dropout"]).to(device)  # antes de criar o otimizador
    dl_tr = criar_dataloader(dt["X"]["ajuste"], dt["y"]["ajuste"],
                             cfg["batch_size"], embaralhar=True, semente=semente)
    dl_val = criar_dataloader(dt["X"]["validacao"], dt["y"]["validacao"],
                              TAMANHO_LOTE_AVALIACAO, embaralhar=False)
    perda = criar_perda(dt["tarefa"], cfg["pos_weight"],
                        dt["n_positivos_ajuste"], dt["n_negativos_ajuste"], device)
    otimizador = criar_otimizador(cfg["otimizador"], modelo.parameters(),
                                  cfg["lr"], cfg["weight_decay"])

    resultado = treinar(modelo, dl_tr, dl_val, perda, otimizador,
                        n_epocas=cfg["n_epocas"], paciencia=cfg["paciencia"],
                        device=device)

    registro = {
        "semente": semente,
        "n_parametros": contar_parametros(modelo),
        "epoca_melhor": resultado["epoca_melhor"],
        "epocas_executadas": resultado["epocas_executadas"],
        "tempo_s": resultado["tempo_s"],
        "divergiu": resultado["divergiu"],
        "melhor_perda_val": resultado["melhor_val"],
    }
    # Se a perda divergiu depois de alguma melhora, o melhor estado anterior
    # já foi restaurado e pode ser avaliado; a divergência fica registrada.
    # Se divergiu antes de qualquer melhora, não há modelo válido a avaliar.
    if resultado["epoca_melhor"] > 0:
        try:
            registro["metricas_ajuste"] = metricas_conjunto(modelo, dt, "ajuste", device)
            registro["metricas_validacao"] = metricas_conjunto(modelo, dt, "validacao", device)
        except ValueError:
            # Previsões não finitas mesmo no melhor estado: tratado como divergência
            registro["divergiu"] = True
            registro.pop("metricas_ajuste", None)
    return modelo, resultado, registro


# ---------------------------------------------------------------------------
# Organização dos resultados de várias sementes
# ---------------------------------------------------------------------------

def tabela_registros(registros):
    """Uma linha por semente, com as métricas em colunas planas.

    Ex.: metricas_ajuste["RMSE"] -> coluna "RMSE_ajuste". A matriz de
    confusão e o limiar (classificação) não entram nesta tabela.
    """
    import pandas as pd

    linhas = []
    for r in registros:
        linha = {k: v for k, v in r.items() if not k.startswith("metricas_")}
        for conjunto in ("ajuste", "validacao"):
            for nome, valor in r.get(f"metricas_{conjunto}", {}).items():
                if nome not in ("Confusion_Matrix", "limiar"):
                    linha[f"{nome}_{conjunto}"] = valor
        linhas.append(linha)
    return pd.DataFrame(linhas)


def resumir_sementes(tabela, colunas):
    """Média, desvio padrão AMOSTRAL (ddof=1), mínimo e máximo entre sementes.

    ddof=1 porque as sementes são uma amostra das inicializações possíveis;
    com 5 sementes, o desvio amostral é cerca de 12% maior que o populacional
    (ddof=0, usado nos folds da Atividade 02). A convenção é registrada no
    relatório.
    """
    import pandas as pd

    return pd.DataFrame({
        "média": tabela[colunas].mean(),
        "desvio (ddof=1)": tabela[colunas].std(ddof=1),
        "mínimo": tabela[colunas].min(),
        "máximo": tabela[colunas].max(),
    })


# ---------------------------------------------------------------------------
# Estudo de hiperparâmetros: execução com registro em log e retomada
# ---------------------------------------------------------------------------

COLUNAS_CONFIG = ("hidden", "activation", "dropout", "otimizador", "lr", "batch_size",
                  "weight_decay", "n_epocas", "paciencia", "pos_weight")


METRICAS_REGRESSAO = ("MAE", "MSE", "RMSE", "MAPE_%", "R2", "MedAE")
METRICAS_CLASSIFICACAO = ("Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC", "PR-AUC",
                          "Average_Precision", "Log_Loss", "Brier")

# Ordem FIXA das colunas do log: as mesmas para os dois problemas e para
# execuções divergentes (métricas ausentes ficam vazias). Sem isso, linhas
# com colunas diferentes desalinhariam o CSV ao serem acrescentadas.
COLUNAS_LOG = (
    ["problema", "config_id", "semente", "estudo_origem", "hidden", "n_camadas", "neuronios"]
    + [c for c in COLUNAS_CONFIG if c != "hidden"]
    + ["n_parametros", "epoca_melhor", "epocas_executadas", "tempo_s", "tempo_por_epoca_s",
       "divergiu", "melhor_perda_val"]
    + [f"{m}_{c}" for c in ("ajuste", "validacao") for m in METRICAS_REGRESSAO + METRICAS_CLASSIFICACAO]
    + ["dispositivo", "torch_versao", "data_hora"]
)


def config_completa(config_extra):
    """Baseline com as alterações indicadas (variação de um fator por vez).

    Os tipos são padronizados (ex.: lr=1 e lr=1.0 viram o mesmo float) para
    que a mesma configuração sempre receba o mesmo identificador.
    """
    cfg = {**CONFIG_BASELINE, **config_extra}
    cfg["hidden"] = tuple(int(h) for h in cfg["hidden"])
    for k in ("dropout", "lr", "weight_decay"):
        cfg[k] = float(cfg[k])
    for k in ("batch_size", "n_epocas", "paciencia"):
        cfg[k] = int(cfg[k])
    cfg["activation"], cfg["otimizador"] = str(cfg["activation"]), str(cfg["otimizador"])
    cfg["pos_weight"] = bool(cfg["pos_weight"])
    return cfg


def id_configuracao(cfg):
    """Identificador determinístico de uma configuração completa.

    Depende só dos valores dos hiperparâmetros: a mesma configuração recebe
    o mesmo id em qualquer estudo (por exemplo, o baseline aparece em todos
    os estudos e no mapa de calor, mas é treinado uma única vez por semente).
    """
    import hashlib
    import json

    canonica = {k: (list(cfg[k]) if k == "hidden" else cfg[k]) for k in COLUNAS_CONFIG}
    texto = json.dumps(canonica, sort_keys=True, default=float)
    return "cfg_" + hashlib.sha1(texto.encode("utf-8")).hexdigest()[:10]


def carregar_log(caminho_log):
    """Lê o log de execuções (vazio se ainda não existir)."""
    import pandas as pd
    from pathlib import Path

    caminho_log = Path(caminho_log)
    if not caminho_log.is_file():
        return pd.DataFrame()
    log = pd.read_csv(caminho_log)
    # Linha incompleta (queda do ambiente durante a gravação): a última coluna
    # fica vazia. Essas linhas são descartadas e o treino é refeito.
    log = log[log["data_hora"].notna()]
    # Proteção contra linha duplicada: mantém a última
    return log.drop_duplicates(subset=["problema", "config_id", "semente"], keep="last")


def modelo_cpu():
    """Modelo do processador (Linux/Colab: /proc/cpuinfo).

    Registrado em cada execução: sessões diferentes do Colab podem alocar
    processadores diferentes, o que altera tempos e pode produzir pequenas
    diferenças numéricas entre sessões.
    """
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for linha in f:
                if linha.lower().startswith("model name"):
                    return linha.split(":", 1)[1].strip()
    except OSError:
        pass
    import platform
    return platform.processor() or "não identificado"


def _linha_log(problema, estudo, cfg, cid, registro, device):
    import datetime

    linha = {
        "problema": problema,
        "config_id": cid,
        "semente": registro["semente"],
        "estudo_origem": estudo,
        "hidden": "-".join(str(h) for h in cfg["hidden"]),
        "n_camadas": len(cfg["hidden"]),
        "neuronios": cfg["hidden"][0] if len(set(cfg["hidden"])) == 1 else None,
    }
    linha.update({k: cfg[k] for k in COLUNAS_CONFIG if k != "hidden"})
    linha.update({
        "n_parametros": registro["n_parametros"],
        "epoca_melhor": registro["epoca_melhor"],
        "epocas_executadas": registro["epocas_executadas"],
        "tempo_s": registro["tempo_s"],
        "tempo_por_epoca_s": registro["tempo_s"] / max(registro["epocas_executadas"], 1),
        "divergiu": bool(registro["divergiu"]),
        "melhor_perda_val": registro["melhor_perda_val"],
    })
    for conjunto in ("ajuste", "validacao"):
        for nome, valor in registro.get(f"metricas_{conjunto}", {}).items():
            if nome not in ("Confusion_Matrix", "limiar"):
                linha[f"{nome}_{conjunto}"] = valor
    linha.update({
        "dispositivo": f"{device} ({modelo_cpu()})" if device == "cpu" else device,
        "torch_versao": torch.__version__,
        "data_hora": datetime.datetime.now().isoformat(timespec="seconds"),
    })
    return linha


def executar_com_registro(dt, problema, config_extra, sementes, caminho_log, pasta_curvas,
                          estudo, device="cpu", verbose=True):
    """Treina UMA configuração com as sementes indicadas, retomando do log.

    - Pula as sementes já registradas para (problema, config_id, semente).
    - Grava cada semente no log (CSV) e a curva de perda (JSON) assim que
      termina; uma queda do Colab perde no máximo o treino em andamento.
    - Execuções divergentes também são gravadas (coluna `divergiu`), nunca
      puladas.
    Retorna as linhas do log desta configuração, para as sementes pedidas.
    """
    import json
    from pathlib import Path

    import pandas as pd

    caminho_log, pasta_curvas = Path(caminho_log), Path(pasta_curvas)
    pasta_curvas.mkdir(parents=True, exist_ok=True)
    cfg = config_completa(config_extra)
    cid = id_configuracao(cfg)

    log = carregar_log(caminho_log)
    feitas = set()
    if len(log):
        feitas = set(log.loc[(log["problema"] == problema) & (log["config_id"] == cid), "semente"])

    for semente in sementes:
        if semente in feitas:
            continue
        _, resultado, registro = executar_configuracao(dt, cfg, semente, device=device)
        # 1º a curva, 2º a linha do log: a linha funciona como confirmação de
        # que a execução foi salva por completo (ver carregar_log).
        with open(pasta_curvas / f"{problema}_{cid}_s{semente}.json", "w", encoding="utf-8") as f:
            json.dump({"problema": problema, "config_id": cid, "semente": semente,
                       "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
                       "epoca_melhor": registro["epoca_melhor"], "divergiu": bool(registro["divergiu"]),
                       "hist": resultado["hist"]}, f)
        linha = _linha_log(problema, estudo, cfg, cid, registro, device)
        if caminho_log.is_file():
            cabecalho = list(pd.read_csv(caminho_log, nrows=0).columns)
            if cabecalho != list(COLUNAS_LOG):
                raise RuntimeError("O cabeçalho do log existente difere de COLUNAS_LOG; "
                                   "não acrescentar linhas para não desalinhar o arquivo.")
        if caminho_log.is_file() and caminho_log.stat().st_size > 0:
            # Se uma gravação anterior foi interrompida sem a quebra de linha
            # final, a completa antes de acrescentar (evita colar duas linhas).
            with open(caminho_log, "rb+") as f:
                f.seek(-1, 2)
                if f.read(1) != b"\n":
                    f.write(b"\n")
        pd.DataFrame([linha]).reindex(columns=list(COLUNAS_LOG)).to_csv(
            caminho_log, mode="a", index=False, header=not caminho_log.is_file())
        if verbose:
            aviso = " | DIVERGIU" if registro["divergiu"] else ""
            print(f"  [{problema}] {estudo} {cid} semente {semente}: "
                  f"melhor época {registro['epoca_melhor']}, parada {registro['epocas_executadas']}, "
                  f"{registro['tempo_s']:.1f} s{aviso}")

    log = carregar_log(caminho_log)
    return log[(log["problema"] == problema) & (log["config_id"] == cid)
               & (log["semente"].isin(list(sementes)))]


def carregar_curva(pasta_curvas, problema, config_id, semente):
    """Lê o histórico de perdas salvo para uma execução."""
    import json
    from pathlib import Path

    with open(Path(pasta_curvas) / f"{problema}_{config_id}_s{semente}.json", encoding="utf-8") as f:
        return json.load(f)


def resumir_configuracoes(linhas, colunas):
    """Média e desvio (ddof=1) entre sementes, por configuração, + contagens.

    `linhas` são linhas do log; o resultado tem uma linha por config_id.
    Sementes divergentes sem métricas aparecem como NaN e não entram na média;
    a coluna `sementes_divergentes` deixa isso explícito.
    """
    agrupado = linhas.groupby("config_id", sort=False)
    resumo = agrupado[colunas].agg(["mean", lambda x: x.std(ddof=1)])
    resumo.columns = [f"{c}_{'media' if e == 'mean' else 'desvio'}" for c, e in resumo.columns]
    resumo["n_sementes"] = agrupado["semente"].nunique()
    resumo["sementes_divergentes"] = agrupado["divergiu"].sum().astype(int)
    resumo["n_parametros"] = agrupado["n_parametros"].first()
    return resumo
