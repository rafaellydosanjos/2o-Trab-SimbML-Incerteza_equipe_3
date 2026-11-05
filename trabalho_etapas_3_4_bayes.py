# ==================================================================
# ETAPAS 3 E 4 (PARTE 1) - PROBABILIDADE E NAIVE BAYES
# Base: dados/tabela_p4.csv (a mesma tabela das Etapas 1, 5 e 6).
# - Escopo: MLE, MAP e Naive Bayes implementado do zero.
# - Rótulo igual ao da Etapa 1: Multa_Alta = 1 quando a multa é maior
#   que a mediana de TODOS os autos da tabela (R$ 100.000,00).
# - Treino 70% e teste 30% (estratificado, semente 42), como na Etapa 1.
# - Validação cruzada de 5 partes apenas no treino.
# - Sem algoritmo EM: a Etapa 4 (parte 2) está no notebook do EM.
# ==================================================================

import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    balanced_accuracy_score, confusion_matrix
)

# ==================================================================
# 1. INICIALIZAÇÃO E DETERMINISMO
# ==================================================================
SEED = 42
np.random.seed(SEED)

PASTA_PROJETO = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()

# Se a tabela não estiver em nenhuma pasta, ela é baixada do GitHub do grupo.
URL_DADOS = ("https://raw.githubusercontent.com/rafaellydosanjos/"
             "2o-Trab-SimbML-Incerteza_equipe_3/main/dados/")


def obter_tabela(nome='tabela_p4.csv'):
    """Procura a tabela em algumas pastas. Se não achar, baixa do GitHub."""
    pastas = [Path.cwd(), Path.cwd() / 'dados', Path.cwd().parent / 'dados',
              PASTA_PROJETO, PASTA_PROJETO / 'dados', PASTA_PROJETO.parent / 'dados',
              Path('/content'), Path('/content/dados')]
    for pasta in pastas:
        if (pasta / nome).is_file():
            return str(pasta / nome)
    os.makedirs('dados', exist_ok=True)
    destino = os.path.join('dados', nome)
    urllib.request.urlretrieve(URL_DADOS + nome, destino)
    print('Baixado do GitHub:', nome)
    return destino


CAMINHO_CSV = obter_tabela()

PERGUNTA_CLASSIFICACAO = (
    'Um auto de infração do IBAMA por desmatamento terá multa de valor elevado?'
)
ALVO_ATUAL = 'Multa_Alta'  # 1 = sim; 0 = não

# Atributos da Etapa 1 (todos categóricos).
ATRIBUTOS_CATEGORICOS = ['BIOMA', 'TP_PESSOA_INFRATOR', 'TIPO_ACAO',
                         'CHUVA_FAIXA', 'TEMP_FAIXA', 'UMID_FAIXA']

# Limiar fixo da Etapa 1 (a mediana de todos os autos da tabela).
# É preenchido logo depois que a tabela é lida.
LIMIAR_ETAPA1_CONFIRMADO = None


def formatar_reais(valor):
    """Usa a convenção brasileira de separadores em valores monetários."""
    return 'R$ ' + f'{valor:,.2f}'.replace(',', '_').replace('.', ',').replace('_', '.')


# ==================================================================
# 2. PIPELINE: PRÉ-PROCESSAMENTO AJUSTADO EXCLUSIVAMENTE NO TREINO
# ==================================================================
def carregar_dados_brutos(caminho=CAMINHO_CSV):
    """Lê a tabela_p4.csv (já filtrada na Etapa 1) e devolve os autos."""
    tabela = pd.read_csv(caminho, encoding='utf-8-sig')
    tabela['VALOR_NUMERICO'] = pd.to_numeric(tabela['VAL_AUTO_INFRACAO'], errors='coerce')
    tabela = tabela[tabela['VALOR_NUMERICO'].notna()].reset_index(drop=True)
    return tabela


def preparar_particoes_ml(X_tr_raw, X_te_raw, y_tr_cont, y_te_cont):
    """Transforma as colunas em números 0/1 usando SOMENTE o treino recebido.

    Usada no holdout e em cada fold da validação cruzada. As colunas do
    one-hot são definidas pelo treino; categoria nova no teste fica toda 0.
    """
    limiar = float(LIMIAR_ETAPA1_CONFIRMADO)
    if not np.isfinite(limiar) or limiar < 0:
        raise ValueError('O limiar da Etapa 1 deve ser um número não negativo e finito.')
    y_tr = (y_tr_cont > limiar).astype(int)
    y_te = (y_te_cont > limiar).astype(int)

    def limpar_categoricas(df_cat):
        return df_cat[ATRIBUTOS_CATEGORICOS].fillna('Nao informado').astype(str)

    cat_tr = limpar_categoricas(X_tr_raw)
    cat_te = limpar_categoricas(X_te_raw)

    X_tr_dum = pd.get_dummies(cat_tr).astype(int)
    X_te_dum = pd.get_dummies(cat_te).astype(int)
    X_te_dum = X_te_dum.reindex(columns=X_tr_dum.columns, fill_value=0)

    # Tabela do treino com o alvo, usada na Etapa 3 (probabilidades conjuntas)
    df_treino_completo = X_tr_dum.copy()
    df_treino_completo['Multa_Alta'] = y_tr

    return (X_tr_dum.to_numpy(), X_te_dum.to_numpy(),
            y_tr.to_numpy(), y_te.to_numpy(),
            X_tr_dum.columns, df_treino_completo, limiar)


def dividir_treino_teste(df_bruto, y_cont=None):
    """Separa 70% para treino e 30% para teste, estratificado pela classe."""
    y = (df_bruto['VALOR_NUMERICO'] > LIMIAR_ETAPA1_CONFIRMADO).astype(int)
    indices = np.arange(len(df_bruto))
    idx_tr, idx_te = train_test_split(indices, test_size=0.30, random_state=SEED, stratify=y)
    return np.sort(idx_tr), np.sort(idx_te)


def construir_pipeline_ml(df_bruto, idx_tr, idx_te):
    """Monta as matrizes do treino e do teste; o teste só é usado no final."""
    if df_bruto.empty:
        raise ValueError('A tabela está vazia.')
    X_raw = df_bruto[ATRIBUTOS_CATEGORICOS]
    y_raw_cont = df_bruto['VALOR_NUMERICO']
    resultado = preparar_particoes_ml(
        X_raw.iloc[idx_tr], X_raw.iloc[idx_te],
        y_raw_cont.iloc[idx_tr], y_raw_cont.iloc[idx_te])
    treino_bruto = df_bruto.iloc[idx_tr].copy()
    return (*resultado, treino_bruto)


# Execução do pipeline externo (holdout)
df_bruto = carregar_dados_brutos()
LIMIAR_ETAPA1_CONFIRMADO = float(df_bruto['VALOR_NUMERICO'].median())
# Confere que o rótulo é o mesmo da coluna Multa_Alta da tabela da Etapa 1.
assert ((df_bruto['VALOR_NUMERICO'] > LIMIAR_ETAPA1_CONFIRMADO).astype(int)
        == df_bruto['Multa_Alta']).all(), 'Rótulo diferente do da Etapa 1.'

IDX_TR, IDX_TE = dividir_treino_teste(df_bruto)
df_teste_bruto = df_bruto.iloc[IDX_TE].copy()
(X_tr, X_te, y_tr, y_te, ATRIBUTOS, df_treino,
 LIMIAR_MULTA, df_treino_bruto) = construir_pipeline_ml(df_bruto, IDX_TR, IDX_TE)


# ==================================================================
# 3. ETAPA 3: AXIOMAS DA PROBABILIDADE
# ==================================================================
def distribuicao_conjunta(df, var1, var2):
    """Retorna P(var1, var2) para duas variáveis binárias codificadas em 0/1."""
    if df.empty:
        raise ValueError("Não é possível calcular a distribuição conjunta de um conjunto vazio.")

    valores_validos = {0, 1}
    if not set(df[var1].dropna().unique()).issubset(valores_validos):
        raise ValueError(f"A variável {var1} deve estar codificada apenas como 0 e 1.")
    if not set(df[var2].dropna().unique()).issubset(valores_validos):
        raise ValueError(f"A variável {var2} deve estar codificada apenas como 0 e 1.")
    if df[[var1, var2]].isna().any().any():
        raise ValueError("As variáveis da distribuição conjunta não podem conter valores ausentes.")

    contagens = pd.crosstab(df[var1], df[var2])
    contagens = contagens.reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    conjunta = contagens.to_numpy(dtype=float) / len(df)
    if not np.isclose(conjunta.sum(), 1.0):
        raise ArithmeticError("A distribuição conjunta calculada não soma 1.")
    return conjunta


def probabilidade_marginal(dist_conjunta, var_alvo):
    """Marginal de var_alvo: 1 = variável das linhas; 2 = variável das colunas."""
    conjunta = np.asarray(dist_conjunta, dtype=float)
    if conjunta.shape != (2, 2):
        raise ValueError("A distribuição conjunta deve ser uma matriz 2x2.")
    if var_alvo == 1:
        return conjunta.sum(axis=1)
    if var_alvo == 2:
        return conjunta.sum(axis=0)
    raise ValueError("var_alvo deve ser 1 (linhas) ou 2 (colunas).")


def probabilidade_condicional(dist_conjunta, prob_marginal_condicao, condicao=2):
    """
    Calcula P(linhas | colunas), se condicao=2; ou P(colunas | linhas),
    se condicao=1. Retorna NaN quando a condição tem probabilidade zero.
    """
    conjunta = np.asarray(dist_conjunta, dtype=float)
    marginal = np.asarray(prob_marginal_condicao, dtype=float)
    if conjunta.shape != (2, 2) or marginal.shape != (2,):
        raise ValueError("Esperada uma matriz 2x2 e uma marginal com dois elementos.")
    if condicao == 2:
        divisor = marginal.reshape(1, 2)
    elif condicao == 1:
        divisor = marginal.reshape(2, 1)
    else:
        raise ValueError("condicao deve ser 1 (linhas) ou 2 (colunas).")

    return np.divide(
        conjunta,
        divisor,
        out=np.full(conjunta.shape, np.nan, dtype=float),
        where=divisor != 0
    )


def regra_da_cadeia(df, a, b, c, va=1, vb=1, vc=1):
    """Confere P(A,B,C) = P(A) P(B|A) P(C|A,B)."""
    if df.empty:
        raise ValueError("Não é possível aplicar a regra da cadeia a um conjunto vazio.")

    p_a = float((df[a] == va).mean())
    sub_a = df[df[a] == va]
    if sub_a.empty:
        raise ValueError("P(B|A) é indefinida porque P(A) = 0.")

    p_b_dado_a = float((sub_a[b] == vb).mean())
    sub_ab = sub_a[sub_a[b] == vb]
    if sub_ab.empty:
        raise ValueError("P(C|A,B) é indefinida porque P(A,B) = 0.")

    p_c_dado_ab = float((sub_ab[c] == vc).mean())
    pela_cadeia = p_a * p_b_dado_a * p_c_dado_ab
    direto = float(((df[a] == va) & (df[b] == vb) & (df[c] == vc)).mean())
    return float(pela_cadeia), direto


def teorema_bayes(p_e_dado_h, p_h, p_e):
    """Calcula P(H|E), validando probabilidades e o denominador."""
    probabilidades = (p_e_dado_h, p_h, p_e)
    if any(not 0 <= p <= 1 for p in probabilidades):
        raise ValueError("Todas as entradas devem ser probabilidades entre 0 e 1.")
    if p_e == 0:
        raise ValueError("P(E) = 0; P(H|E) é indefinida.")
    resultado = p_e_dado_h * p_h / p_e
    if resultado > 1 + 1e-12:
        raise ValueError("As probabilidades informadas são incompatíveis com o Teorema de Bayes.")
    return float(min(resultado, 1.0))


def independencia(df, a, b, tolerancia_numerica=1e-12):
    """Verifica fatoração da distribuição empírica P(A,B) = P(A)P(B).

    Compara os quatro eventos atômicos de duas variáveis binárias.
    A tolerância trata somente arredondamento de ponto flutuante; não é
    um teste estatístico nem prova independência na população.
    """
    conjunta = distribuicao_conjunta(df, a, b)
    marginal_a = probabilidade_marginal(conjunta, 1)
    marginal_b = probabilidade_marginal(conjunta, 2)
    produto = np.outer(marginal_a, marginal_b)
    diferenca = np.abs(conjunta - produto)
    return {
        'p_11': float(conjunta[1, 1]),
        'p_a1': float(marginal_a[1]),
        'p_b1': float(marginal_b[1]),
        'produto_11': float(produto[1, 1]),
        'diferenca_maxima': float(diferenca.max()),
        'fatoracao_empirica': bool(np.allclose(
            conjunta, produto, atol=tolerancia_numerica, rtol=0.0
        )),
    }


def independencia_condicional(df, a, b, y):
    """Compara P(A,B|C=c) com P(A|C=c)P(B|C=c) para c em {0,1}."""
    resultado = {}
    for classe in (0, 1):
        subconjunto = df.loc[df[y] == classe]
        resultado[classe] = (
            None if subconjunto.empty else independencia(subconjunto, a, b)
        )
    return resultado


def imprimir_independencia_condicional(resultados, detalhado=False):
    """Exibe a verificação matemática exigida na Etapa 3."""
    for classe, resultado in resultados.items():
        if resultado is None:
            print(f'C={classe}: sem observações; probabilidades condicionais indefinidas.')
            continue
        print(
            f'C={classe}: P(A=1,B=1|C)={resultado["p_11"]:.6f}; '
            f'P(A=1|C)·P(B=1|C)={resultado["produto_11"]:.6f}'
        )
        if detalhado:
            print(
                f'  P(A=1|C)={resultado["p_a1"]:.6f}; '
                f'P(B=1|C)={resultado["p_b1"]:.6f}'
            )
        print(
            f'  Maior diferença nos quatro eventos={resultado["diferenca_maxima"]:.6f}; '
            f'fatoração empírica={"satisfeita" if resultado["fatoracao_empirica"] else "não satisfeita"}'
        )


# ==================================================================
# 4. ETAPA 4 (Parte 1): APRENDIZADO PROBABILÍSTICO (NAIVE BAYES)
# ==================================================================
def estimador_mle(k, N):
    """Estimador de máxima verossimilhança para Bernoulli: theta_hat = k/N."""
    if N <= 0:
        raise ValueError("N deve ser maior que zero.")
    if not 0 <= k <= N:
        raise ValueError("k deve estar no intervalo [0, N].")
    return k / N


def estimador_map(k, N, alpha=2, beta=2):
    """Estimador MAP para Bernoulli com prior Beta(alpha, beta)."""
    k_array = np.asarray(k)
    if N < 0 or np.any(k_array < 0) or np.any(k_array > N):
        raise ValueError("É necessário N >= 0 e 0 <= k <= N.")
    if alpha <= 0 or beta <= 0:
        raise ValueError("Os parâmetros alpha e beta devem ser positivos.")
    denominador = N + alpha + beta - 2
    if denominador <= 0:
        raise ValueError("A moda posterior não está definida para esses parâmetros.")
    return (k_array + alpha - 1) / denominador


class NaiveBayesRaiz:
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n_exemplos, n_atributos = X.shape

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        self.log_prior_ = np.zeros(n_classes)
        self.p_atributo_ = np.zeros((n_classes, n_atributos))

        for j, c in enumerate(self.classes_):
            X_c = X[y == c]
            N_c = len(X_c)
            # Suavização de Laplace no prior: impede np.log(0) caso uma classe seja muito rara
            self.log_prior_[j] = np.log((N_c + 1) / (n_exemplos + n_classes))
            self.p_atributo_[j] = estimador_map(X_c.sum(axis=0), N_c)
        return self

    def _log_scores(self, X):
        X = np.asarray(X)
        p = self.p_atributo_
        # Produto matricial, sem laços: soma dos logs das probabilidades dos atributos.
        return X @ (np.log(p) - np.log(1 - p)).T + np.log(1 - p).sum(axis=1) + self.log_prior_

    def predict(self, X):
        return self.classes_[np.argmax(self._log_scores(X), axis=1)]

    def predict_proba(self, X):
        scores = self._log_scores(X)
        scores = scores - scores.max(axis=1, keepdims=True)  # previne underflow
        exp = np.exp(scores)
        return exp / exp.sum(axis=1, keepdims=True)


def acuracia(y_real, y_previsto):
    return np.mean(y_real == y_previsto)


def validacao_cruzada(df_bruto_treino, k=5, modelo=NaiveBayesRaiz):
    """Validação sem vazamento: as partes são feitas só dentro do treino.

    A cada parte, o one-hot é aprendido do zero só com o treino da parte.
    O teste externo de 30% nunca participa da validação cruzada.

    Retorna as acurácias de cada parte e os baselines (classe mais comum
    do treino de cada parte, nunca escolhida pela validação).
    """
    if len(df_bruto_treino) < k or k < 2:
        raise ValueError("k deve estar entre 2 e a quantidade de amostras do treino.")
    X_raw = df_bruto_treino[ATRIBUTOS_CATEGORICOS]
    y_cont = df_bruto_treino['VALOR_NUMERICO']
    y_classe = (y_cont > LIMIAR_ETAPA1_CONFIRMADO).astype(int)

    acuracias, baselines = [], []
    divisao = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
    for idx_tr, idx_val in divisao.split(X_raw, y_classe):
        X_fit_num, X_val_num, y_fit, y_val, _, _, _ = preparar_particoes_ml(
            X_raw.iloc[idx_tr], X_raw.iloc[idx_val],
            y_cont.iloc[idx_tr], y_cont.iloc[idx_val]
        )
        m = modelo().fit(X_fit_num, y_fit)
        acuracias.append(float(acuracia(y_val, m.predict(X_val_num))))
        classes, contagens = np.unique(y_fit, return_counts=True)
        classe_majoritaria = classes[np.argmax(contagens)]
        baselines.append(float(np.mean(y_val == classe_majoritaria)))
    return acuracias, baselines


def imprimir_metricas_teste(nome, modelo, X_teste, y_teste):
    """Resumo do teste externo, que não foi empregado na validação cruzada."""
    y_pred = modelo.predict(X_teste)
    print(f"{nome} | acurácia: {acuracia(y_teste, y_pred):.4f} | "
          f"acurácia balanceada: {balanced_accuracy_score(y_teste, y_pred):.4f} | "
          f"precisão (classe 1): {precision_score(y_teste, y_pred, zero_division=0):.4f} | "
          f"recall (classe 1): {recall_score(y_teste, y_pred, zero_division=0):.4f} | "
          f"F1 (classe 1): {f1_score(y_teste, y_pred, zero_division=0):.4f}")
    cm = confusion_matrix(y_teste, y_pred, labels=[0, 1])
    print("Matriz de confusão (linhas=real, colunas=previsto; ordem 0, 1):")
    print(cm)
    return y_pred


# ==================================================================
# 5. ETAPA 3: APLICAÇÃO PRÁTICA DOS FUNDAMENTOS DE PROBABILIDADE
# ==================================================================
VAR_A = 'BIOMA_Amazonia'
VAR_B = 'TP_PESSOA_INFRATOR_PJ'
VAR_C = 'Multa_Alta'


def executar_etapa3(df):
    """
    Demonstra os requisitos da Etapa 3 (versão longa).

    A = ocorrência no bioma Amazônia
    B = infrator pessoa jurídica (PJ)
    C = multa alta (valor acima da mediana da tabela)
    """
    A, B, C = VAR_A, VAR_B, VAR_C

    ausentes = [col for col in (A, B, C) if col not in df.columns]
    if ausentes:
        raise KeyError(f"Variáveis necessárias à Etapa 3 não encontradas: {ausentes}")

    dados = df[[A, B, C]].dropna().copy()
    if dados.empty:
        raise ValueError("Não há observações válidas para executar a Etapa 3.")

    print("\n=== ETAPA 3: FUNDAMENTOS PRÁTICOS DE PROBABILIDADE ===")
    print("Variáveis utilizadas:")
    print("A = auto associado ao bioma Amazônia (1 = sim; 0 = não)")
    print("B = infrator pessoa jurídica (1 = sim; 0 = não)")
    print("C = multa alta (1 = acima da mediana da tabela; 0 = caso contrário)")
    print(f"Amostras válidas: {len(dados)}")

    conjunta = distribuicao_conjunta(dados, A, B)
    print("\n1. Distribuição conjunta P(A, B) - quatro eventos atômicos:")
    print(pd.DataFrame(conjunta, index=['A=0', 'A=1'], columns=['B=0', 'B=1'])
          .to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"Soma das probabilidades: {conjunta.sum():.6f}")
    print(f"Não negatividade (todas >= 0): {bool(np.all(conjunta >= 0))}")
    print(f"Normalização P(Ω)=1: {np.isclose(conjunta.sum(), 1.0)}")
    print("Evento impossível: P(∅)=0 (nenhuma observação pertence ao evento vazio).")
    print("Aditividade para eventos disjuntos: "
          f"P(A=0) = P(A=0,B=0) + P(A=0,B=1) = "
          f"{conjunta[0, 0]:.4f} + {conjunta[0, 1]:.4f} = {conjunta[0, :].sum():.4f}")

    marginal_A = probabilidade_marginal(conjunta, 1)
    marginal_B = probabilidade_marginal(conjunta, 2)
    cond_A_dado_B = probabilidade_condicional(conjunta, marginal_B, condicao=2)
    cond_B_dado_A = probabilidade_condicional(conjunta, marginal_A, condicao=1)

    print("\n2. Probabilidades marginais:")
    print(f"P(A=0)={marginal_A[0]:.4f} | P(A=1)={marginal_A[1]:.4f}")
    print(f"P(B=0)={marginal_B[0]:.4f} | P(B=1)={marginal_B[1]:.4f}")
    print("\nProbabilidades condicionais:")
    print("P(A | B), linhas=A e colunas=B:")
    print(pd.DataFrame(cond_A_dado_B, index=['A=0', 'A=1'], columns=['B=0', 'B=1'])
          .to_string(float_format=lambda x: f"{x:.4f}"))
    print("P(B | A), linhas=B e colunas=A:")
    print(pd.DataFrame(cond_B_dado_A.T, index=['B=0', 'B=1'], columns=['A=0', 'A=1'])
          .to_string(float_format=lambda x: f"{x:.4f}"))

    p_cadeia, p_direta = regra_da_cadeia(dados, A, B, C, 1, 1, 1)
    print("\n3. Regra da cadeia:")
    print("P(A=1, B=1, C=1) = P(A=1) * P(B=1|A=1) * P(C=1|A=1,B=1)")
    print(f"Produto pela regra da cadeia: {p_cadeia:.6f}")
    print(f"Frequência conjunta direta:   {p_direta:.6f}")
    print(f"Valores coincidem: {np.isclose(p_cadeia, p_direta, atol=1e-12)}")

    p_A_dado_C = float(dados.loc[dados[C] == 1, A].mean()) if (dados[C] == 1).any() else np.nan
    p_C = float(dados[C].mean())
    p_A = float(dados[A].mean())
    if np.isfinite(p_A_dado_C) and p_A > 0:
        p_C_dado_A_bayes = teorema_bayes(p_A_dado_C, p_C, p_A)
        p_C_dado_A_direto = float(dados.loc[dados[A] == 1, C].mean())
        print("\n4. Teorema de Bayes:")
        print("H = multa alta; E = auto associado ao bioma Amazônia.")
        print("P(H|E) = P(E|H) * P(H) / P(E)")
        print(f"P(E|H)={p_A_dado_C:.4f}; P(H)={p_C:.4f}; P(E)={p_A:.4f}")
        print(f"P(Amazônia | multa alta): {p_A_dado_C:.4f}")
        print(f"P(multa alta | Amazônia), por Bayes: {p_C_dado_A_bayes:.4f}")
        print(f"P(multa alta | Amazônia), frequência direta: {p_C_dado_A_direto:.4f}")
    else:
        print("\n4. Teorema de Bayes: não foi possível calcular, pois o evento observado tem probabilidade zero.")

    print("\n5. Independência condicional entre bioma Amazônia e tipo PJ, dada a classe Multa_Alta:")
    resultados = independencia_condicional(dados, A, B, C)
    imprimir_independencia_condicional(resultados, detalhado=True)
    print('Interpretação: a verificação é sobre as frequências observadas,')
    print('não representa uma prova de independência na população.')


def executar_etapa3_resumo(df):
    """Resumo legível dos cálculos; a versão longa é executar_etapa3."""
    A, B, C = VAR_A, VAR_B, VAR_C
    dados = df[[A, B, C]].dropna()
    conjunta = distribuicao_conjunta(dados, A, B)
    marginal_A = probabilidade_marginal(conjunta, 1)
    marginal_B = probabilidade_marginal(conjunta, 2)
    cond_A_B = probabilidade_condicional(conjunta, marginal_B, condicao=2)
    produto, direta = regra_da_cadeia(dados, A, B, C, 1, 1, 1)
    p_C = float(dados[C].mean())
    p_A = float(dados[A].mean())
    p_A_C = float(dados.loc[dados[C] == 1, A].mean()) if (dados[C] == 1).any() else float('nan')
    p_C_A = teorema_bayes(p_A_C, p_C, p_A) if np.isfinite(p_A_C) and p_A > 0 else float('nan')
    indep = independencia_condicional(dados, A, B, C)

    print('\nETAPA 3 | Probabilidade (dados de treino)')
    print(f'A=bioma Amazônia; B=pessoa jurídica; C=multa alta | n={len(dados)}')
    print('Distribuição conjunta P(A,B) [linhas A=0,1; colunas B=0,1]:')
    print(np.array2string(conjunta, formatter={'float_kind': lambda v: f'{v:.4f}'}))
    print(f'Axiomas: soma={conjunta.sum():.4f}; não negativas={bool(np.all(conjunta>=0))}')
    print(f'Marginais: P(A=1)={marginal_A[1]:.4f}; P(B=1)={marginal_B[1]:.4f}')
    print(f'Condicional: P(A=1|B=1)={cond_A_B[1,1]:.4f}')
    print(f'Regra da cadeia: {produto:.5f} (frequência direta: {direta:.5f})')
    print(f'Bayes: P(Amazônia|multa alta)={p_A_C:.4f}')
    print(f'Bayes: P(multa alta|Amazônia)={p_C_A:.4f}')
    print('Independência condicional - fatoração P(A,B|C)=P(A|C)P(B|C):')
    imprimir_independencia_condicional(indep)
    print('Nota: a comparação descreve a amostra, sem inferir causalidade.')

    print('Interpretação de Bayes (causal vs. diagnóstica):')
    print('  P(multa alta|Amazônia) segue a direção causal plausível')
    print('  (características do auto influenciam o valor da multa).')
    print('  P(Amazônia|multa alta) é diagnóstica: parte do efeito para a característica.')
    print('  O NB usa a forma generativa P(atributos|classe) e a inverte por Bayes;')
    print('  isso é modelagem, não prova de causalidade.')
    print('Limite da LPO: predicados clássicos são verdadeiros ou falsos;')
    print('  uma afirmação como "70% de chance" exige probabilidades, não só LPO.')


# ==================================================================
# 6. DEMONSTRAÇÃO E INTEGRAÇÃO
# ==================================================================
def demonstrar_mle_map(df):
    """Mostra MLE e MAP/Beta(2,2) com a mesma variável usada pelo NB.

    A amostra usa APENAS o treino. k conta autos do bioma Amazônia na
    classe Multa_Alta=1; N é o número de autos dessa classe no treino.
    """
    variavel = VAR_A
    classe = 'Multa_Alta'
    if variavel not in df.columns or classe not in df.columns:
        raise KeyError('Faltam as colunas necessárias para a demonstração MLE/MAP.')

    amostra = df.loc[df[classe] == 1, variavel]
    if amostra.empty:
        raise ValueError('Não há exemplos da classe Multa_Alta=1 para estimar Bernoulli.')
    N = len(amostra)
    k = int(amostra.sum())
    mle = float(estimador_mle(k, N))
    map_beta22 = float(estimador_map(k, N, alpha=2, beta=2))

    print('\nETAPA 4 (PARTE 1) | Estimativas MLE e MAP')
    print('Exemplo: P(bioma Amazônia=1 | multa alta=1), usando só o treino.')
    print(f'Bernoulli: sucessos k={k}; amostras N={N}')
    print(f'MLE: theta = k/N = {k}/{N} = {mle:.6f}')
    print(f'MAP com prior Beta(2,2): theta = (k+1)/(N+2) = {map_beta22:.6f}')
    print('O MAP incorpora o prior e é usado na estimação dos atributos do NB.')
    return {'k': k, 'N': N, 'mle': mle, 'map': map_beta22}


def verificar_alinhamento_etapa1():
    """Mostra que a Etapa 4 usa a mesma tabela e o mesmo rótulo da Etapa 1."""
    print('\nINTEGRAÇÃO COM A ETAPA 1 | Definição do problema')
    print(f'Pergunta: {PERGUNTA_CLASSIFICACAO}')
    print(f'Tabela: {CAMINHO_CSV} ({len(df_bruto)} autos)')
    print(f'Alvo binário: {ALVO_ATUAL} (1=valor elevado; 0=valor não elevado)')
    print(f'Regra: valor da multa > {formatar_reais(LIMIAR_MULTA)} (mediana de todos os autos da tabela).')
    print('O rótulo calculado aqui é igual à coluna Multa_Alta da tabela da Etapa 1 (conferido por assert).')
    print(f'Treino: {len(df_treino_bruto)} autos | teste externo: {len(df_teste_bruto)} autos (30%, estratificado).')


if __name__ == "__main__":
    print('=== ETAPA 3 E ETAPA 4 (PARTE 1) | RESULTADOS ===')
    print(f'Dados: treino={len(X_tr)} | teste={len(X_te)} | atributos={len(ATRIBUTOS)}')
    print(f'Limiar de multa alta: {formatar_reais(LIMIAR_MULTA)}')

    executar_etapa3_resumo(df_treino)
    demonstrar_mle_map(df_treino)
    verificar_alinhamento_etapa1()

    print('\nETAPA 4 (PARTE 1) | Naive Bayes')
    nb = NaiveBayesRaiz().fit(X_tr, y_tr)
    acs, bases = validacao_cruzada(df_treino_bruto, k=5)
    print(f'Validação cruzada (5 partes, apenas treino): {np.mean(acs):.2%} '
          f'± {np.std(acs, ddof=1):.2%} | baseline={np.mean(bases):.2%}')
    print('Teste externo (30%):')
    previsto = nb.predict(X_te)
    print(f'  NB puro | acurácia={acuracia(y_te, previsto):.2%} | '
          f'precisão={precision_score(y_te, previsto, zero_division=0):.2%} | '
          f'recall={recall_score(y_te, previsto, zero_division=0):.2%} | '
          f'F1={f1_score(y_te, previsto, zero_division=0):.2%}')
