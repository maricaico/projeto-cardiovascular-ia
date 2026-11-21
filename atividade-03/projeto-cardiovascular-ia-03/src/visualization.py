"""Atividade 03 (DCT1401): funções de gráfico do estudo de hiperparâmetros.

Concentra os gráficos repetidos para cada hiperparâmetro, para que o
notebook mostre a análise sem centenas de linhas iguais:

- métrica final (ajuste e validação) em função do valor testado, com barras
  de desvio padrão entre sementes;
- curvas de perda (ajuste e validação) por época, uma por valor testado;
- mapa de calor para a interação entre dois hiperparâmetros.

Todas as funções salvam a figura (dpi=150) quando `arquivo` é informado.
"""

import matplotlib.pyplot as plt
import numpy as np


def _salvar(fig, arquivo):
    fig.tight_layout()
    if arquivo is not None:
        fig.savefig(arquivo, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def grafico_metrica_por_valor(paineis, rotulos_valores, titulo, nome_eixo_x,
                              indice_baseline=None, arquivo=None):
    """Métrica de ajuste e de validação (média ± desvio) para cada valor testado.

    paineis : lista de dicionários, um por problema, com as chaves
        "titulo", "rotulo_metrica", "media_ajuste", "desvio_ajuste",
        "media_validacao", "desvio_validacao" e, opcionalmente,
        "divergentes" (número de sementes divergentes por valor) e
        "referencia" (valor de uma referência ingênua, linha horizontal).
    rotulos_valores : textos do eixo x, na ordem dos valores.
    indice_baseline : posição do valor do baseline (destacado no eixo x).

    Os valores são posicionados em espaçamento uniforme (categorias), o que
    serve tanto a hiperparâmetros numéricos (em escala log ou não) quanto a
    categóricos.
    """
    x = np.arange(len(rotulos_valores))
    fig, eixos = plt.subplots(1, len(paineis), figsize=(6.5 * len(paineis), 4.6), squeeze=False)
    for eixo, p in zip(eixos[0], paineis):
        eixo.errorbar(x - 0.06, p["media_ajuste"], yerr=p["desvio_ajuste"], fmt="o-",
                      capsize=4, linewidth=1.5, label="Ajuste (treino)")
        eixo.errorbar(x + 0.06, p["media_validacao"], yerr=p["desvio_validacao"], fmt="s--",
                      capsize=4, linewidth=1.5, label="Validação")
        if p.get("referencia") is not None:
            eixo.axhline(p["referencia"], color="gray", linestyle=":", linewidth=1.2,
                         label="Referência ingênua (validação)")
        for i, n_div in enumerate(p.get("divergentes", [0] * len(x))):
            if n_div:
                eixo.annotate(f"{n_div} div.", (x[i], np.nanmax([p["media_validacao"][i], 0])),
                              textcoords="offset points", xytext=(0, 10), ha="center",
                              fontsize=8, color="red")
        eixo.set_xticks(x)
        eixo.set_xticklabels(rotulos_valores)
        if indice_baseline is not None:
            eixo.get_xticklabels()[indice_baseline].set_fontweight("bold")
        eixo.set_xlabel(nome_eixo_x + (" (baseline em negrito)" if indice_baseline is not None else ""))
        eixo.set_ylabel(p["rotulo_metrica"])
        eixo.set_title(p["titulo"])
        eixo.grid(alpha=0.25)
        eixo.legend(fontsize=8)
    fig.suptitle(titulo)
    _salvar(fig, arquivo)


def grafico_curvas_por_valor(paineis, titulo, arquivo=None, max_epocas=None):
    """Curvas de perda por época (ajuste: linha cheia; validação: tracejada).

    paineis : lista de dicionários, um por problema, com as chaves
        "titulo", "rotulo_perda", "ylim" (opcional) e "curvas" — lista de tuplas
        (rotulo_valor, historico, epoca_melhor), em que historico tem as
        chaves "treino" e "val".
    As curvas mostradas são de uma única semente (indicada no título pelo
    notebook), porque sobrepor várias sementes de vários valores tornaria o
    gráfico ilegível; a variação entre sementes aparece no gráfico de métrica.
    """
    fig, eixos = plt.subplots(1, len(paineis), figsize=(6.5 * len(paineis), 4.6), squeeze=False)
    cores = plt.cm.viridis(np.linspace(0, 0.9, max(len(p["curvas"]) for p in paineis)))
    for eixo, p in zip(eixos[0], paineis):
        for cor, (rotulo, hist, epoca_melhor) in zip(cores, p["curvas"]):
            ep = np.arange(1, len(hist["val"]) + 1)
            eixo.plot(ep, hist["treino"], color=cor, linewidth=1.1, alpha=0.55)
            eixo.plot(ep, hist["val"], color=cor, linewidth=1.8, linestyle="--", label=str(rotulo))
            if epoca_melhor and epoca_melhor <= len(hist["val"]):
                eixo.scatter(epoca_melhor, hist["val"][epoca_melhor - 1], color=cor, s=30,
                             zorder=3, edgecolor="black")
        if max_epocas:
            eixo.set_xlim(0, max_epocas)
        if p.get("ylim") is not None:
            eixo.set_ylim(*p["ylim"])   # recorte vertical: curvas muito altas no início ficam fora
        eixo.set_xlabel("Época")
        eixo.set_ylabel(p["rotulo_perda"])
        eixo.set_title(p["titulo"])
        eixo.grid(alpha=0.25)
        eixo.legend(title="Valor (tracejada: validação)", fontsize=8)
    fig.suptitle(titulo)
    _salvar(fig, arquivo)


def mapa_calor(paineis, rotulos_linhas, rotulos_colunas, titulo, nome_linhas, nome_colunas,
               arquivo=None):
    """Mapa de calor da métrica de validação (média entre sementes).

    paineis : lista de dicionários, um por problema, com as chaves
        "titulo", "matriz" (linhas × colunas, média), "desvio" (mesma forma),
        "rotulo_metrica" e "maior_melhor" (True para AP; False para RMSE).
    Cada célula mostra a média e, abaixo, o desvio entre sementes. Células sem
    resultado válido (por exemplo, divergência em todas as sementes) ficam em
    branco com a marcação "div.".
    """
    fig, eixos = plt.subplots(1, len(paineis), figsize=(6.5 * len(paineis), 5.2), squeeze=False)
    for eixo, p in zip(eixos[0], paineis):
        matriz = np.asarray(p["matriz"], dtype=float)
        desvio = np.asarray(p["desvio"], dtype=float)
        mapa = "viridis" if p["maior_melhor"] else "viridis_r"
        img = eixo.imshow(np.ma.masked_invalid(matriz), cmap=mapa, aspect="auto")
        fig.colorbar(img, ax=eixo, label=p["rotulo_metrica"] + " (validação)")
        validos = matriz[np.isfinite(matriz)]
        meio = (validos.min() + validos.max()) / 2 if validos.size else 0
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                if np.isfinite(matriz[i, j]):
                    claro = (matriz[i, j] < meio) if p["maior_melhor"] else (matriz[i, j] > meio)
                    eixo.text(j, i, f"{matriz[i, j]:.4f}\n±{desvio[i, j]:.4f}", ha="center",
                              va="center", fontsize=8, color="white" if claro else "black")
                else:
                    eixo.text(j, i, "div.", ha="center", va="center", fontsize=9, color="red")
        eixo.set_xticks(range(len(rotulos_colunas)))
        eixo.set_xticklabels(rotulos_colunas)
        eixo.set_yticks(range(len(rotulos_linhas)))
        eixo.set_yticklabels(rotulos_linhas)
        eixo.set_xlabel(nome_colunas)
        eixo.set_ylabel(nome_linhas)
        eixo.set_title(p["titulo"])
    fig.suptitle(titulo)
    _salvar(fig, arquivo)
