"""Atividade 03 (DCT1401): inferência com os modelos salvos (.pt).

Script independente: carrega um checkpoint, aplica o MESMO pré-processamento
usado no treinamento (com as estatísticas guardadas no próprio checkpoint) e
devolve as previsões. Não depende de nenhuma variável do treinamento nem do
scikit-learn (usa apenas NumPy, pandas, SciPy e PyTorch).

Uso pela linha de comando (a partir da pasta do projeto):

    python src/inference.py --modelo models/modelo_classificacao.pt \
                            --entrada data/exemplo_amostras_inferencia.csv

    python src/inference.py --modelo models/modelo_regressao.pt \
                            --entrada data/exemplo_amostras_inferencia.csv \
                            --saida previsoes.csv

A entrada é um CSV com as seis features em unidades originais, por exemplo:

    Diabetes,Hypertension,Chest_Pain,ECG_Result,Blood_Glucose,Blood_Pressure_Systolic
    Yes,No,No,Normal,132,141

Valores numéricos ausentes são preenchidos com a mediana do subconjunto de
ajuste; categorias desconhecidas ou ausentes geram erro.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import expit

try:                                  # importado como pacote: src.inference
    from .models import MLP
except ImportError:                   # executado como script: python src/inference.py
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from models import MLP


def carregar_modelo(caminho):
    """Carrega o checkpoint de forma segura e recria a rede em modo de avaliação.

    - map_location="cpu": o modelo é carregado na CPU mesmo que tenha sido
      treinado em GPU;
    - weights_only=True: o arquivo só pode conter tensores e tipos simples;
      objetos Python arbitrários (pickle) são recusados.
    """
    ckpt = torch.load(caminho, map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    modelo = MLP(cfg["in_dim"], cfg["out_dim"], tuple(cfg["hidden"]),
                 cfg["activation"], cfg["dropout"])
    modelo.load_state_dict(ckpt["state_dict"])
    modelo.eval()                     # desativa dropout (e fixaria BatchNorm)
    return modelo, ckpt


def preprocessar(amostras, ckpt):
    """Aplica codificação, imputação e padronização guardadas no checkpoint.

    Retorna um array float32 com as features na ordem do treinamento.
    """
    faltantes = [f for f in ckpt["features"] if f not in amostras.columns]
    if faltantes:
        raise ValueError(f"Colunas ausentes na entrada: {faltantes}")

    X = pd.DataFrame(index=amostras.index)
    for f in ckpt["features"]:
        coluna = amostras[f]
        if f in ckpt["categorias"]:
            codigos = {categoria: i for i, categoria in enumerate(ckpt["categorias"][f])}
            invalidos = sorted(set(coluna.astype(str)) - set(codigos))
            if coluna.isna().any() or invalidos:
                raise ValueError(
                    f"Valores inválidos em '{f}': {invalidos or ['ausente']}. "
                    f"Valores aceitos: {list(codigos)}")
            X[f] = coluna.astype(str).map(codigos).astype(np.float64)
        else:
            X[f] = pd.to_numeric(coluna, errors="raise").astype(np.float64)
            X[f] = X[f].fillna(ckpt["mediana_imputacao"][f])

    media = np.asarray(ckpt["x_media"], dtype=np.float64)
    desvio = np.asarray(ckpt["x_desvio"], dtype=np.float64)
    return ((X[ckpt["features"]].to_numpy(np.float64) - media) / desvio).astype(np.float32)


def prever(amostras, modelo, ckpt):
    """Previsões em um DataFrame, na mesma ordem das amostras de entrada.

    Regressão: Risk_Score previsto na escala original.
    Classificação: probabilidade da classe 1 e classe prevista pelo limiar.
    """
    X = torch.as_tensor(preprocessar(amostras, ckpt))
    with torch.inference_mode():
        saida = modelo(X).squeeze(1).double().numpy()

    if ckpt["tarefa"] == "regressao":
        previsto = saida * ckpt["y_desvio"] + ckpt["y_media"]
        return pd.DataFrame({"Risk_Score_previsto": previsto}, index=amostras.index)

    probabilidade = expit(saida)      # sigmoide estável, a mesma do treinamento
    classe = (probabilidade >= ckpt["limiar"]).astype(int)
    return pd.DataFrame({
        "probabilidade_classe_1": probabilidade,
        "classe_prevista": classe,
        "classe_nome": [ckpt["classes"][c] for c in classe],
    }, index=amostras.index)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Inferência com um modelo MLP salvo (.pt).")
    parser.add_argument("--modelo", required=True, help="caminho do arquivo .pt")
    parser.add_argument("--entrada", required=True, help="CSV com as features em unidades originais")
    parser.add_argument("--saida", help="CSV de saída (opcional; sem ele, as previsões são exibidas)")
    args = parser.parse_args(argv)

    modelo, ckpt = carregar_modelo(args.modelo)
    amostras = pd.read_csv(args.entrada, float_precision="round_trip")
    previsoes = prever(amostras, modelo, ckpt)

    if args.saida:
        previsoes.to_csv(args.saida, index=False, float_format="%.17g")
        print(f"{len(previsoes)} previsões gravadas em {args.saida}")
    else:
        print(f"Modelo: {args.modelo} | tarefa: {ckpt['tarefa']}")
        if ckpt["tarefa"] == "classificacao":
            print(f"Limiar de decisão: {ckpt['limiar']:.4f}")
        print(pd.concat([amostras[ckpt["features"]], previsoes], axis=1).to_string(index=False))


if __name__ == "__main__":
    main()
