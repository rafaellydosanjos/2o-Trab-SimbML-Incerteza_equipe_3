"""
preparar_inmet.py - le os dados historicos do INMET (ZIPs anuais), resume por UF e mes
e junta com os autos de infracao do Ibama.

Uso rapido (no notebook):
    from preparar_inmet import processar_inmet, juntar_com_autos, em_faixas
    inmet = processar_inmet('dados/inmet')              # pasta com os .zip baixados
    df = juntar_com_autos(df_autos, inmet, coluna_data='DAT_HORA_AUTO_INFRACAO')
"""
import io
import re
import unicodedata
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

MIN_HORAS_VALIDAS = 360   # estacao-mes so vale se tiver pelo menos ~15 dias de dados (15 x 24 h)


# ----------------------------------------------------------------------------
# Funcoes pequenas de apoio
# ----------------------------------------------------------------------------
def _sem_acento(texto):
    texto = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()
    return texto.upper().strip()


def _decodificar(dados):
    """Os arquivos do INMET misturam codificacoes: tenta UTF-8 e, se falhar, latin1."""
    try:
        return dados.decode("utf-8")
    except UnicodeDecodeError:
        return dados.decode("latin1")


def _para_numero(serie):
    """Texto com virgula ou ponto decimal -> numero. Valores como -9999 (sem medicao) viram NaN."""
    s = pd.to_numeric(serie.astype(str).str.strip().str.replace(",", ".", regex=False), errors="coerce")
    return s.where(s > -9000)


def extrair_ano_mes(serie):
    """
    Entende datas como '2021/03/05', '2021-03-05' e '05/03/2021' (com ou sem hora depois).
    Devolve (ano, mes) como numeros.
    """
    d = serie.astype(str).str.strip().str.replace("/", "-", regex=False).str[:10]
    iso = d.str.match(r"^\d{4}-\d{2}-\d{2}$")
    data_iso = pd.to_datetime(d.where(iso), format="%Y-%m-%d", errors="coerce")
    data_br = pd.to_datetime(d.where(~iso), format="%d-%m-%Y", errors="coerce")
    data = data_iso.fillna(data_br)
    return data.dt.year, data.dt.month


# ----------------------------------------------------------------------------
# Leitura de UMA estacao
# ----------------------------------------------------------------------------
def _resumo_mensal_estacao(texto, nome_arquivo):
    linhas = texto.splitlines()

    # UF: vem numa linha do cabecalho ("UF:;AM"); se nao achar, tenta pelo nome do arquivo
    uf = None
    for linha in linhas[:15]:
        if linha.upper().startswith("UF:"):
            uf = linha.split(";")[1].strip().upper()
            break
    if not uf:
        partes = Path(nome_arquivo).name.split("_")        # INMET_N_AM_A101_MANAUS_...
        uf = partes[2].upper() if len(partes) > 2 else None
    if not uf or len(uf) != 2:
        raise ValueError("nao achei a UF")

    # cabecalho dos dados = primeira linha (nas 30 primeiras) com varios ';'
    idx = next(i for i, l in enumerate(linhas[:30]) if l.count(";") >= 5)
    df = pd.read_csv(io.StringIO("\n".join(linhas[idx:])), sep=";", dtype=str)

    # acha as colunas pelo nome (sem depender de acento nem de caixa alta/baixa)
    nomes = {c: _sem_acento(c) for c in df.columns}
    col_data = next(c for c, n in nomes.items() if n.startswith("DATA"))
    col_chuva = next(c for c, n in nomes.items() if "PRECIPITA" in n)
    col_temp = next(c for c, n in nomes.items() if "TEMPERATURA DO AR" in n and "BULBO SECO" in n)
    col_umid = next(c for c, n in nomes.items()
                    if "UMIDADE REL" in n and "MAX" not in n and "MIN" not in n)

    ano, mes = extrair_ano_mes(df[col_data])
    tabela = pd.DataFrame({
        "ano": ano, "mes": mes,
        "chuva": _para_numero(df[col_chuva]),
        "temp": _para_numero(df[col_temp]),
        "umid": _para_numero(df[col_umid]),
    }).dropna(subset=["ano", "mes"])

    resumo = tabela.groupby(["ano", "mes"]).agg(
        precip_mm=("chuva", lambda s: s.sum(min_count=1)),   # chuva do mes = soma das horas
        temp_media=("temp", "mean"),
        umid_media=("umid", "mean"),
        horas_validas=("temp", "count"),
    ).reset_index()
    resumo = resumo[resumo["horas_validas"] >= MIN_HORAS_VALIDAS].drop(columns="horas_validas")
    resumo["UF"] = uf
    return resumo


# ----------------------------------------------------------------------------
# Leitura de TODOS os arquivos da pasta
# ----------------------------------------------------------------------------
def processar_inmet(pasta):
    """Le todos os .zip (e .csv soltos) da pasta e devolve uma linha por UF, ano e mes."""
    pasta = Path(pasta)
    partes, ok, falhas = [], 0, []

    def tentar(texto, nome):
        nonlocal ok
        try:
            partes.append(_resumo_mensal_estacao(texto, nome))
            ok += 1
        except Exception as erro:                       # nao para tudo por causa de 1 arquivo ruim
            falhas.append((Path(nome).name, repr(erro)[:90]))

    for caminho in sorted(pasta.rglob("*")):
        if caminho.suffix.lower() == ".zip":
            with zipfile.ZipFile(caminho) as z:
                for nome in z.namelist():
                    if nome.lower().endswith(".csv"):
                        tentar(_decodificar(z.read(nome)), nome)
        elif caminho.suffix.lower() == ".csv":
            tentar(_decodificar(caminho.read_bytes()), str(caminho))

    print(f"Estacoes lidas com sucesso: {ok} | com problema: {len(falhas)}")
    for nome, erro in falhas[:5]:
        print("  falhou:", nome, "->", erro)
    if not partes:
        raise RuntimeError("Nenhum arquivo foi lido. Confira a pasta e me mande o erro.")

    por_estacao = pd.concat(partes, ignore_index=True)
    por_uf = por_estacao.groupby(["UF", "ano", "mes"]).agg(
        precip_mm=("precip_mm", "mean"),      # media das estacoes da UF
        temp_media=("temp_media", "mean"),
        umid_media=("umid_media", "mean"),
        n_estacoes=("temp_media", "count"),
    ).reset_index()
    por_uf[["ano", "mes"]] = por_uf[["ano", "mes"]].astype(int)
    return por_uf


# ----------------------------------------------------------------------------
# Juntar com os autos do Ibama e transformar em faixas
# ----------------------------------------------------------------------------
def juntar_com_autos(df_autos, inmet_uf_mes, coluna_data="DAT_HORA_AUTO_INFRACAO", coluna_uf="UF"):
    """Para cada auto de infracao, pega o clima da mesma UF e do mesmo mes."""
    df = df_autos.copy()
    if coluna_data not in df.columns:
        candidatas = [c for c in df.columns if "DAT" in c.upper()]
        raise KeyError(f"Coluna '{coluna_data}' nao existe. Colunas com 'DAT': {candidatas}")
    df["ano"], df["mes"] = extrair_ano_mes(df[coluna_data])
    df[coluna_uf] = df[coluna_uf].astype(str).str.strip().str.upper()
    df = df.merge(inmet_uf_mes.rename(columns={"UF": coluna_uf}), on=[coluna_uf, "ano", "mes"], how="left")
    sem_clima = df["precip_mm"].isna().mean()
    print(f"Autos sem clima do INMET (UF/mes sem estacao): {sem_clima:.1%}")
    return df


def em_faixas(serie, rotulos=("baixa", "media", "alta")):
    """Divide em 3 grupos do mesmo tamanho (tercis). Valor faltando vira 'sem_dado'."""
    posicao = serie.rank(method="first")
    faixa = pd.qcut(posicao, q=len(rotulos), labels=list(rotulos))
    return faixa.astype(object).where(serie.notna(), "sem_dado")
