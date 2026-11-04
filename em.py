
import numpy as np

try:
    from utils import SEED          # semente fixa da equipe (42)
except ImportError:                 # se rodar fora do projeto
    SEED = 42

np.random.seed(SEED)


# ---------------------------------------------------------------- funcoes de apoio
def logsumexp(a, eixo=1):
    """log(soma(exp(a))) calculado de forma estavel (evita overflow/underflow)."""
    m = np.max(a, axis=eixo, keepdims=True)
    return (m + np.log(np.sum(np.exp(a - m), axis=eixo, keepdims=True))).squeeze(eixo)


def log_gaussiana(X, mu, sigma):
    """log N(x | mu, sigma) para todas as linhas de X (gaussiana multivariada)."""
    N, d = X.shape
    L = np.linalg.cholesky(sigma)                       # sigma = L @ L.T
    diff = np.linalg.solve(L, (X - mu).T)               # L^-1 (x - mu)
    mahalanobis = np.sum(diff ** 2, axis=0)             # (x-mu)^T sigma^-1 (x-mu)
    log_det = 2.0 * np.sum(np.log(np.diag(L)))          # log |sigma|
    return -0.5 * (d * np.log(2 * np.pi) + log_det + mahalanobis)


# ---------------------------------------------------------------- o modelo
class MisturaGaussianaEM:
    """
    Mistura de K gaussianas treinada com EM.

    Parametros
    ----------
    K          : numero de grupos escondidos (2 ou 3 no enunciado)
    max_iter   : maximo de iteracoes
    tol        : para quando a melhora da log-verossimilhanca MEDIA for menor que isso
    reg_covar  : valor pequeno somado na diagonal da covariancia (evita variancia ~ 0)
    n_init     : numero de inicializacoes aleatorias; fica com a de maior verossimilhanca
    seed       : semente (42)
    """

    def __init__(self, K=2, max_iter=200, tol=1e-6, reg_covar=1e-6, n_init=5, seed=SEED):
        self.K, self.max_iter, self.tol = K, max_iter, tol
        self.reg_covar, self.n_init, self.seed = reg_covar, n_init, seed

    # ---------- passo E
    def _log_prob_ponderada(self, X, theta, mu, sigma):
        """matriz N x K com log( theta_k * N(x_i | mu_k, sigma_k) )."""
        return np.column_stack([
            np.log(theta[k]) + log_gaussiana(X, mu[k], sigma[k]) for k in range(self.K)
        ])

    def _passo_E(self, X, theta, mu, sigma):
        lp = self._log_prob_ponderada(X, theta, mu, sigma)
        ll_por_linha = logsumexp(lp)                    # log P(x_i)
        r = np.exp(lp - ll_por_linha[:, None])          # responsabilidades (cada linha soma 1)
        return r, ll_por_linha.sum()

    # ---------- passo M
    def _passo_M(self, X, r):
        N, d = X.shape
        Nk = r.sum(axis=0) + 1e-12                      # "quantos pontos" cada grupo explica
        theta = Nk / N
        mu = (r.T @ X) / Nk[:, None]
        sigma = np.empty((self.K, d, d))
        for k in range(self.K):
            diff = X - mu[k]
            sigma[k] = (r[:, k, None] * diff).T @ diff / Nk[k]
            sigma[k] += self.reg_covar * np.eye(d)      # evita covariancia singular
        return theta, mu, sigma

    # ---------- uma rodada completa do EM
    def _rodar_uma_vez(self, X, rng):
        N, d = X.shape
        # inicializacao: K pontos aleatorios como medias, covariancia global, pesos iguais
        idx = rng.choice(N, size=self.K, replace=False)
        mu = X[idx].copy()
        cov_global = np.cov(X.T).reshape(d, d) + self.reg_covar * np.eye(d)
        sigma = np.array([cov_global.copy() for _ in range(self.K)])
        theta = np.full(self.K, 1.0 / self.K)

        historico = []
        for it in range(self.max_iter):
            r, ll = self._passo_E(X, theta, mu, sigma)          # E
            historico.append(ll)
            theta, mu, sigma = self._passo_M(X, r)              # M
            if it > 0 and abs(historico[-1] - historico[-2]) / N < self.tol:
                break
        # log-verossimilhanca final, ja com os parametros atualizados
        _, ll = self._passo_E(X, theta, mu, sigma)
        historico.append(ll)
        return theta, mu, sigma, historico

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.seed)
        melhor = None
        for _ in range(self.n_init):
            try:
                resultado = self._rodar_uma_vez(X, rng)
            except np.linalg.LinAlgError:               # covariancia ruim nessa inicializacao
                continue
            if melhor is None or resultado[3][-1] > melhor[3][-1]:
                melhor = resultado
        if melhor is None:
            raise RuntimeError("EM nao convergiu em nenhuma inicializacao.")
        self.theta_, self.mu_, self.sigma_, self.loglik_ = melhor
        self.n_iter_ = len(self.loglik_) - 1
        return self

    # ---------- uso do modelo treinado
    def predict_proba(self, X):
        """Responsabilidades P(Z=k | x) para dados novos."""
        r, _ = self._passo_E(np.asarray(X, dtype=float), self.theta_, self.mu_, self.sigma_)
        return r

    def predict(self, X):
        """Grupo mais provavel (0, 1, ..., K-1) de cada linha."""
        return self.predict_proba(X).argmax(axis=1)

    def score(self, X):
        """Log-verossimilhanca media por linha."""
        _, ll = self._passo_E(np.asarray(X, dtype=float), self.theta_, self.mu_, self.sigma_)
        return ll / len(X)


# ---------------------------------------------------------------- integracao com o projeto
def adicionar_Z(df_treino, df_teste, colunas, K=2, nome_z="Z"):
    """
    Ajusta o EM SO no treino e cria a coluna Z (texto) em treino e teste.
    Linhas com dado faltando nas `colunas` recebem Z = "sem_dado".
    (Ajustar so no treino evita vazamento de informacao na validacao cruzada.)
    """
    df_treino, df_teste = df_treino.copy(), df_teste.copy()
    ok_tr = df_treino[colunas].notna().all(axis=1).values
    ok_te = df_teste[colunas].notna().all(axis=1).values

    Xtr = df_treino.loc[ok_tr, colunas].values.astype(float)
    media, desvio = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-12      # padroniza com stats do treino
    modelo = MisturaGaussianaEM(K=K).fit((Xtr - media) / desvio)

    for df, ok in ((df_treino, ok_tr), (df_teste, ok_te)):
        z = np.full(len(df), "sem_dado", dtype=object)
        if ok.any():
            X = (df.loc[ok, colunas].values.astype(float) - media) / desvio
            z[ok] = ["Z" + str(g) for g in modelo.predict(X)]
        df[nome_z] = z
    return df_treino, df_teste, modelo


def z_para_matriz(z, K):
    """
    Transforma a coluna Z (texto) em colunas 0/1: Z0, Z1, ..., Z{K-1} e "sem_dado".
    Sempre devolve K+1 colunas na mesma ordem (serve para Naive Bayes com atributos 0/1).
    """
    z = np.asarray(z, dtype=object)
    categorias = ["Z" + str(k) for k in range(K)] + ["sem_dado"]
    return np.column_stack([(z == c).astype(int) for c in categorias])


def plotar_loglik(loglik, titulo="EM - log-verossimilhanca por iteracao", salvar=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 4))
    plt.plot(range(len(loglik)), loglik, marker="o", ms=3)
    plt.xlabel("Iteracao")
    plt.ylabel("Log-verossimilhanca")
    plt.title(titulo)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if salvar:
        plt.savefig(salvar, dpi=150)
    plt.show()


# ---------------------------------------------------------------- testes
def teste_sintetico():
    """Gera 2 grupos conhecidos e confere se o EM os recupera."""
    rng = np.random.default_rng(SEED)
    A = rng.multivariate_normal([0, 0], [[1, 0.3], [0.3, 1]], size=300)
    B = rng.multivariate_normal([5, 4], [[1, -0.2], [-0.2, 0.8]], size=200)
    X = np.vstack([A, B])
    m = MisturaGaussianaEM(K=2).fit(X)
    ordem = np.argsort(m.mu_[:, 0])
    print("=== TESTE SINTETICO ===")
    print("medias reais   : [0, 0] e [5, 4]   | pesos reais: 0.60 e 0.40")
    print("medias achadas :", np.round(m.mu_[ordem], 2).tolist(), "| pesos:", np.round(m.theta_[ordem], 2))
    dif = np.diff(m.loglik_)
    print(f"iteracoes: {m.n_iter_} | loglik so sobe? {bool(np.all(dif > -1e-8))}")
    return m


def teste_dados_reais(caminho="dados/tabela_p4.csv", K=2):
    import pandas as pd
    df = pd.read_csv(caminho, encoding="utf-8-sig")
    cols = ["precip_mm", "temp_media", "umid_media"]
    d = df[cols].dropna()
    X = (d.values - d.values.mean(axis=0)) / d.values.std(axis=0)
    m = MisturaGaussianaEM(K=K).fit(X)
    print(f"\n=== DADOS REAIS (K={K}, {len(d)} linhas com clima completo) ===")
    print("pesos (theta):", np.round(m.theta_, 3))
    media_real = m.mu_ * d.values.std(axis=0) + d.values.mean(axis=0)
    for k in range(K):
        print(f"  Z{k}: chuva={media_real[k, 0]:.0f} mm | temp={media_real[k, 1]:.1f} C | umid={media_real[k, 2]:.0f}%")
    print(f"iteracoes: {m.n_iter_} | loglik so sobe? {bool(np.all(np.diff(m.loglik_) > -1e-8))}")
    return m


if __name__ == "__main__":
    teste_sintetico()
    try:
        m = teste_dados_reais()
        plotar_loglik(m.loglik_, salvar="loglik_em.png")
    except FileNotFoundError:
        print("\n(dados/tabela_p4.csv nao encontrado - rode na pasta principal do projeto)")
