import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

from preparar_inmet import em_faixas, juntar_com_autos, processar_inmet

np.random.seed(42)

# ------------------------------------------------------------------ configuração
CAMINHO_AUTOS = sys.argv[1] if len(sys.argv) > 1 else "dados/auto_infracao_2021.csv"
PASTA_INMET = sys.argv[2] if len(sys.argv) > 2 else "dados/inmet"
CACHE_INMET = "dados/inmet_uf_mes.csv" 

ATRIBUTOS = ["BIOMA", "TP_PESSOA_INFRATOR", "TIPO_ACAO", "CHUVA_FAIXA", "TEMP_FAIXA", "UMID_FAIXA"]


def texto(serie):
    """Minúsculas e sem acento, para comparar textos sem erro."""
    return (serie.astype(str).str.normalize("NFKD").str.encode("ascii", "ignore")
            .str.decode("ascii").str.lower().str.strip())


def para_numero(serie):
    """Converte para número sem estragar colunas que o pandas já leu como número."""
    if pd.api.types.is_numeric_dtype(serie):
        return serie.astype(float)
    t = serie.astype(str).str.strip().str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(t, errors="coerce")


# ------------------------------------------------------------------ 1. Ibama
try:
    df = pd.read_csv(CAMINHO_AUTOS, sep=";", encoding="utf-8", low_memory=False)
except UnicodeDecodeError:
    df = pd.read_csv(CAMINHO_AUTOS, sep=";", encoding="latin1", low_memory=False)

print("=== FILTROS (quantos autos sobram em cada passo) ===")
print(f"Todos os autos do arquivo: {len(df)}")

df = df[texto(df["TIPO_INFRACAO"]) == "flora"]
print(f"Só infrações de Flora: {len(df)}")

df = df[texto(df["INFRACAO_AREA"]).str.contains("desmat", na=False)]
print(f"Só os marcados como Desmatamento: {len(df)}")

df = df[df["SIT_CANCELADO"].astype(str).str.strip().str.upper() == "N"]
df = df[~texto(df["DES_STATUS_FORMULARIO"]).isin(["cancelado", "excluido", "substituido por outro"])]
print(f"Sem autos cancelados/excluídos: {len(df)}")

df = df[texto(df["TIPO_AUTO"]) == "multa simples"]
print(f"Só 'Multa simples' (sem advertência): {len(df)}")

df = df.copy()
df["VAL_AUTO_INFRACAO"] = para_numero(df["VAL_AUTO_INFRACAO"])
df = df[df["VAL_AUTO_INFRACAO"] > 0].copy()
print(f"Com valor de multa válido: {len(df)}")

print("\nResumo do valor da multa (R$):")
print(df["VAL_AUTO_INFRACAO"].describe().round(0).to_string())

# ------------------------------------------------------------------ 2. Atributos do Ibama
# Bioma: alguns vêm repetidos ("Amazonia, Amazonia"); ficamos com o primeiro
df["BIOMA"] = df["DS_BIOMAS_ATINGIDOS"].astype(str).str.split(",").str[0].str.strip()
df["BIOMA"] = df["BIOMA"].replace({"nan": "Nao_informado", "": "Nao_informado"})

df["TP_PESSOA_INFRATOR"] = df["TP_PESSOA_INFRATOR"].fillna("Outros").astype(str).str.strip()

# Tipo de ação: guardamos as 2 mais comuns e juntamos o resto em "Outros"
df["TIPO_ACAO"] = df["TIPO_ACAO"].fillna("Outros").astype(str).str.strip()
df.loc[~df["TIPO_ACAO"].isin(["Operação", "Rotina"]), "TIPO_ACAO"] = "Outros"

# ------------------------------------------------------------------ 3. INMET (clima por UF e mês)
if os.path.exists(CACHE_INMET):
    inmet = pd.read_csv(CACHE_INMET)
    print(f"\nUsando o INMET já processado ({CACHE_INMET}). Apague esse arquivo para refazer.")
else:
    inmet = processar_inmet(PASTA_INMET)
    os.makedirs("dados", exist_ok=True)
    inmet.to_csv(CACHE_INMET, index=False)

df = juntar_com_autos(df, inmet, coluna_data="DAT_HORA_AUTO_INFRACAO")
df["CHUVA_FAIXA"] = em_faixas(df["precip_mm"])
df["TEMP_FAIXA"] = em_faixas(df["temp_media"])
df["UMID_FAIXA"] = em_faixas(df["umid_media"])

# ------------------------------------------------------------------ 4. Variável-alvo
mediana = df["VAL_AUTO_INFRACAO"].median()
df["Multa_Alta"] = (df["VAL_AUTO_INFRACAO"] > mediana).astype(int)
print(f"\nMediana da multa (corte): R$ {mediana:,.2f}")
print("Proporção das classes (0 = multa baixa, 1 = multa alta):")
print(df["Multa_Alta"].value_counts(normalize=True).round(3).to_string())

# ------------------------------------------------------------------ 5. Salva a tabela pronta
os.makedirs("dados", exist_ok=True)
COLUNAS_TABELA = ["UF", "MUNICIPIO", "DAT_HORA_AUTO_INFRACAO", "ano", "mes", "VAL_AUTO_INFRACAO",
                  "BIOMA", "TP_PESSOA_INFRATOR", "TIPO_ACAO",
                  "precip_mm", "temp_media", "umid_media", "n_estacoes",
                  "CHUVA_FAIXA", "TEMP_FAIXA", "UMID_FAIXA", "Multa_Alta",
                  "EFEITO_MEIO_AMBIENTE", "NUM_PROCESSO"]
# (de propósito, NÃO salvamos nome nem CPF/CNPJ de quem foi multado)
df[COLUNAS_TABELA].to_csv("dados/tabela_p4.csv", index=False, encoding="utf-8-sig")
print(f"\nTabela salva em dados/tabela_p4.csv ({df.shape[0]} linhas, {len(COLUNAS_TABELA)} colunas)")

# ------------------------------------------------------------------ 6. Árvore (scikit-learn) + validação cruzada
X = pd.get_dummies(df[ATRIBUTOS], drop_first=False)
y = df["Multa_Alta"]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
resultados = {}
print("\nprofundidade | acurácia treino | acurácia teste (média 5-fold)")
for prof in [1, 2, 3, 4, 5, 6, None]:
    modelo = DecisionTreeClassifier(max_depth=prof, criterion="entropy", random_state=42)
    r = cross_validate(modelo, X, y, cv=cv, scoring="accuracy", return_train_score=True)
    resultados[prof] = r["test_score"].mean()
    nome = "sem poda" if prof is None else prof
    print(f"{str(nome):12s} | {r['train_score'].mean():.3f}           | "
          f"{r['test_score'].mean():.3f} (+/- {r['test_score'].std():.3f})")

melhor = max([p for p in resultados if p is not None], key=lambda p: resultados[p])
print(f"\nProfundidade ótima (entre 1 e 6): {melhor}")

modelo_final = DecisionTreeClassifier(max_depth=melhor, criterion="entropy", random_state=42)
y_pred_cv = cross_val_predict(modelo_final, X, y, cv=cv)
print(classification_report(y, y_pred_cv, target_names=["Multa Baixa", "Multa Alta"]))

# ------------------------------------------------------------------ 7. Desenho da árvore
# (treino com todos os dados só para desenhar; a acurácia do relatório é a da validação cruzada acima)
modelo_final.fit(X, y)

plt.figure(figsize=(18, 8))
plot_tree(modelo_final, feature_names=X.columns.tolist(), class_names=["Multa Baixa", "Multa Alta"],
          max_depth=3, filled=True, rounded=True, fontsize=10)
plt.title("Árvore de Decisão - Autos de infração por desmatamento (Ibama, 2021)", fontsize=14)
plt.tight_layout()
plt.savefig("arvore_etapa1.png", dpi=200, bbox_inches="tight")
print("\nFigura salva em arvore_etapa1.png")
plt.show()

print(export_text(modelo_final, feature_names=X.columns.tolist(), max_depth=3))
