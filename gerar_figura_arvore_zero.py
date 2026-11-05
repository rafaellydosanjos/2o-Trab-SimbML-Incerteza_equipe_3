"""
gerar_figura_arvore_zero.py - desenha a árvore de decisão FEITA DO ZERO (um ramo para cada valor
do atributo) nos 3 primeiros níveis, com os mesmos números das contas de entropia e ganho do relatório.

Como usar (na pasta principal do trabalho, depois de rodar etapa1_p4.py):
    python gerar_figura_arvore_zero.py
Cria: arvore_do_zero_etapa1.png
"""
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

ATRIBUTOS = ["BIOMA", "TP_PESSOA_INFRATOR", "TIPO_ACAO", "CHUVA_FAIXA", "TEMP_FAIXA", "UMID_FAIXA"]
ALVO = "Multa_Alta"
NIVEIS = 3   # quantos níveis desenhar

t = pd.read_csv("dados/tabela_p4.csv", encoding="utf-8-sig")
df = t[ATRIBUTOS + [ALVO]].astype(str)       # tudo como texto; autos sem bioma ficam de fora dos grupos


def entropia(y):
    n = len(y)
    if n == 0:
        return 0.0
    return -sum((c / n) * np.log2(c / n) for c in Counter(y).values()) + 0.0   # + 0.0 evita mostrar -0,000


def ganho(d, atributo):
    n = len(d)
    return entropia(d[ALVO]) - sum(len(s) / n * entropia(s[ALVO]) for _, s in d.groupby(atributo))


def construir(d, atributos, nivel, rotulo):
    c = Counter(d[ALVO])
    no = {
        "rotulo": rotulo, "n": len(d), "baixa": c.get("0", 0), "alta": c.get("1", 0),
        "H": entropia(d[ALVO]), "prev": c.most_common(1)[0][0], "pergunta": None, "filhos": [], "nivel": nivel,
    }
    puro = len(c) == 1
    if puro or not atributos:
        return no
    ganhos = {a: ganho(d, a) for a in atributos}
    melhor = max(ganhos, key=ganhos.get)
    no["pergunta"] = melhor
    if nivel < NIVEIS:
        restantes = [a for a in atributos if a != melhor]
        for valor, sub in d.groupby(melhor):
            no["filhos"].append(construir(sub, restantes, nivel + 1, str(valor)))
    return no


raiz = construir(df, ATRIBUTOS, 1, "todos os autos")


# ------------------------------------------------------------------ posições (esquerda -> direita)
def atribuir_y(no, proximo):
    if not no["filhos"]:
        no["y"] = proximo[0]
        proximo[0] += 1
    else:
        for f in no["filhos"]:
            atribuir_y(f, proximo)
        no["y"] = np.mean([f["y"] for f in no["filhos"]])


atribuir_y(raiz, [0])


def todos(no):
    yield no
    for f in no["filhos"]:
        yield from todos(f)


nos = list(todos(raiz))
linhas = max(n["y"] for n in nos) + 1
LARG, ALT = 3.5, 0.82            # tamanho das caixas
ESPACO_X = 5.6                   # distância entre colunas
fig_alt = max(6, linhas * 0.95 + 1.5)
fig, ax = plt.subplots(figsize=(15, fig_alt))
ax.set_xlim(-0.4, (NIVEIS - 1) * ESPACO_X + LARG + 0.4)
ax.set_ylim(linhas + 0.1, -1.2)
ax.axis("off")


def fmt(x, casas=3):
    return f"{x:.{casas}f}".replace(".", ",")


def cor(no):
    total = no["baixa"] + no["alta"]
    pureza = abs(no["alta"] - no["baixa"]) / total if total else 0
    forca = 0.25 + 0.6 * pureza
    base = (0.93, 0.55, 0.25) if no["prev"] == "0" else (0.25, 0.55, 0.85)   # laranja = baixa, azul = alta
    return tuple(1 - forca * (1 - c) for c in base)


def desenhar(no):
    x = (no["nivel"] - 1) * ESPACO_X
    y = no["y"]
    if no["pergunta"] and no["filhos"]:
        titulo = f"{no['pergunta']}?"
    elif no["pergunta"]:
        titulo = f"prevê: multa {'alta' if no['prev'] == '1' else 'baixa'}   (próx.: {no['pergunta']})"
    else:
        titulo = f"folha: multa {'alta' if no['prev'] == '1' else 'baixa'}"
    texto = (f"{titulo}\nn = {no['n']}   H = {fmt(no['H'])}\nbaixa {no['baixa']} · alta {no['alta']}")
    caixa = FancyBboxPatch((x, y - ALT / 2), LARG, ALT, boxstyle="round,pad=0.02,rounding_size=0.08",
                           fc=cor(no), ec="#333333", lw=1)
    ax.add_patch(caixa)
    ax.text(x + LARG / 2, y, texto, ha="center", va="center", fontsize=8.3, linespacing=1.35)
    for f in no["filhos"]:
        x2 = (f["nivel"] - 1) * ESPACO_X
        ax.annotate("", xy=(x2, f["y"]), xytext=(x + LARG, y),
                    arrowprops=dict(arrowstyle="-", color="#555555", lw=0.9))
        xm = (x + LARG + x2) / 2
        ym = (y + f["y"]) / 2 * 0.5 + f["y"] * 0.5
        ax.text(xm + 0.15, ym, f["rotulo"], ha="center", va="center", fontsize=8, style="italic",
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9))
    for f in no["filhos"]:
        desenhar(f)


desenhar(raiz)

for k, nome in enumerate(["Nível 1", "Nível 2", "Nível 3"][:NIVEIS]):
    ax.text(k * ESPACO_X + LARG / 2, -0.95, nome, ha="center", fontsize=11, fontweight="bold", color="#1F3864")

sem_bioma = int((t["BIOMA"].isna()).sum())
ax.text(0, linhas + 0.0,
        "H = entropia · laranja = maioria multa baixa, azul = maioria multa alta (cor mais forte = grupo mais puro) · próx. = próxima pergunta (nível seguinte, não desenhado).\n"
        f"{sem_bioma} autos sem bioma informado não entram em nenhum ramo de BIOMA. Árvore feita do zero, treinada com todos os autos da tabela.",
        fontsize=8, va="top", color="#444444")
plt.title("Árvore de decisão feita do zero — autos de infração por desmatamento (Ibama, 2021)", fontsize=13, pad=14)
plt.tight_layout()
plt.savefig("arvore_do_zero_etapa1.png", dpi=200, bbox_inches="tight")
print(f"Figura salva em arvore_do_zero_etapa1.png ({len(nos)} caixas, {linhas:.0f} linhas)")
