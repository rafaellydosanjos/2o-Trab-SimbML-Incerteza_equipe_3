"""
utils.py - funcoes auxiliares da equipe (carregar, limpar, discretizar, separar treino/teste).

Quem criou: Membro 1. Os outros membros podem importar daqui:
    from utils import SEED, carregar_csv, separar_treino_teste
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42  # semente fixa: garante que todo mundo obtem os mesmos resultados
np.random.seed(SEED)


def carregar_csv(caminho, sep=",", encoding="utf-8", amostra=None):
    """Le um CSV. Se o arquivo for gigante, use amostra=20000 para pegar so 20 mil linhas."""
    df = pd.read_csv(caminho, sep=sep, encoding=encoding, low_memory=False)
    if amostra is not None and len(df) > amostra:
        df = df.sample(n=amostra, random_state=SEED).reset_index(drop=True)
    return df


def limpar(df, colunas):
    """Fica so com as colunas escolhidas e joga fora linhas com valor faltando."""
    df = df[colunas].copy()
    antes = len(df)
    df = df.dropna().reset_index(drop=True)
    print(f"Linhas: {antes} -> {len(df)} (removidas {antes - len(df)} com valor faltando)")
    return df


def discretizar(df, faixas):
    """
    Transforma colunas numericas em faixas (categorias), porque a nossa arvore
    trabalha com categorias.
    faixas = {"idade": ([0, 18, 40, 60, 120], ["0-17", "18-39", "40-59", "60+"])}
    """
    df = df.copy()
    for coluna, (limites, rotulos) in faixas.items():
        df[coluna] = pd.cut(df[coluna], bins=limites, labels=rotulos, include_lowest=True).astype(str)
    return df


def separar_treino_teste(df, alvo, teste=0.2):
    """Separa 80% para treinar e 20% para testar (mantendo a proporcao das classes)."""
    treino, teste_df = train_test_split(
        df, test_size=teste, random_state=SEED, stratify=df[alvo]
    )
    return treino.reset_index(drop=True), teste_df.reset_index(drop=True)


def gerar_exemplo(n=1000, seed=SEED):
    """
    Dados FALSOS so para voce testar o notebook antes de ter os dados oficiais.
    Pergunta de exemplo: "o aluno vai evadir?"
    """
    rng = np.random.default_rng(seed)
    faixa_etaria = rng.choice(["ate_14", "15_17", "18_mais"], size=n, p=[0.3, 0.4, 0.3])
    renda = rng.choice(["baixa", "media", "alta"], size=n, p=[0.45, 0.4, 0.15])
    zona = rng.choice(["urbana", "rural"], size=n, p=[0.75, 0.25])
    faltas = rng.choice(["poucas", "muitas"], size=n, p=[0.6, 0.4])
    turno = rng.choice(["manha", "tarde", "noite"], size=n, p=[0.4, 0.35, 0.25])

    # probabilidade de evadir depende (com ruido) dos atributos
    p = np.full(n, 0.08)
    p += np.where(faltas == "muitas", 0.35, 0.0)
    p += np.where(renda == "baixa", 0.12, 0.0)
    p += np.where(faixa_etaria == "18_mais", 0.10, 0.0)
    p += np.where(zona == "rural", 0.05, 0.0)
    p += np.where(turno == "noite", 0.08, 0.0)
    p = np.clip(p, 0.02, 0.95)
    evadiu = np.where(rng.random(n) < p, "sim", "nao")

    return pd.DataFrame({
        "faixa_etaria": faixa_etaria, "renda": renda, "zona": zona,
        "faltas": faltas, "turno": turno, "evadiu": evadiu,
    })
