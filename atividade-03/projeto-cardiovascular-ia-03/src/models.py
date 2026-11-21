"""Atividade 03 (DCT1401): rede MLP parametrizável em PyTorch.

Os hiperparâmetros de arquitetura são passados
como argumentos, para que o estudo possa variá-los sem reescrever o código.

Saída SEMPRE linear (sem ativação):
- regressão: 1 neurônio, perda nn.MSELoss;
- classificação binária: 1 logit, perda nn.BCEWithLogitsLoss
  (a sigmoide é aplicada só na inferência e nas curvas ROC/PR).
"""

import torch.nn as nn


# Fábricas das funções de ativação estudadas.
# LeakyReLU usa inclinação 0,1 (o padrão do PyTorch seria 0,01); a escolha
# é registrada no relatório.
ATIVACOES = {
    "ReLU": lambda: nn.ReLU(),
    "LeakyReLU": lambda: nn.LeakyReLU(negative_slope=0.1),
    "Tanh": lambda: nn.Tanh(),
    "Sigmoid": lambda: nn.Sigmoid(),
    "ELU": lambda: nn.ELU(),
    "GELU": lambda: nn.GELU(),
}


class MLP(nn.Module):
    """Perceptron multicamadas: [Linear -> ativação (-> Dropout)] x L -> Linear.

    Parâmetros
    ----------
    in_dim : número de features de entrada (6 neste projeto).
    out_dim : número de saídas (1 para regressão e para classificação binária).
    hidden : tupla com o número de neurônios de cada camada oculta,
        por exemplo (32, 32) = 2 camadas de 32 neurônios.
    activation : nome de uma chave de ATIVACOES.
    dropout : probabilidade de zerar ativações após cada camada oculta
        (0.0 = sem dropout).
    """

    def __init__(self, in_dim, out_dim, hidden=(32, 32), activation="ReLU", dropout=0.0):
        super().__init__()
        if activation not in ATIVACOES:
            raise ValueError(f"Ativação desconhecida: {activation}. Opções: {list(ATIVACOES)}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout deve estar em [0, 1).")

        camadas, d = [], in_dim
        for h in hidden:
            camadas += [nn.Linear(d, h), ATIVACOES[activation]()]
            if dropout > 0:
                camadas.append(nn.Dropout(dropout))
            d = h
        camadas.append(nn.Linear(d, out_dim))  # saída linear (logit ou valor)
        self.net = nn.Sequential(*camadas)

        # Configuração em tipos simples: vai para o checkpoint e permite
        # recriar exatamente a mesma arquitetura no script de inferência.
        self.config = {
            "in_dim": int(in_dim),
            "out_dim": int(out_dim),
            "hidden": [int(h) for h in hidden],
            "activation": str(activation),
            "dropout": float(dropout),
        }

    def forward(self, x):
        return self.net(x)


def contar_parametros(modelo):
    """Número de parâmetros treináveis (pesos e vieses)."""
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)


def parametros_teoricos(d, n, L, saida=1):
    """Número de parâmetros de uma MLP com L camadas ocultas de n neurônios.

    P = (d·n + n) + (L − 1)·(n² + n) + (n·saida + saida)
    Exemplo (baseline): d=6, n=32, L=2 -> 224 + 1056 + 33 = 1313.
    """
    return (d * n + n) + (L - 1) * (n * n + n) + (n * saida + saida)
