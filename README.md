# Do Simbólico à Incerteza — Pipeline de Aprendizado (P4: Meio Ambiente)

Trabalho Integrado de Fundamentos de IA (Prof. Edjard Mota).

**Pergunta do trabalho:** um auto de infração do Ibama por desmatamento terá multa de valor elevado?

**Domínio:** P4 — Meio Ambiente (desmatamento + INMET)

**Equipe:**
* Adrya Vieira
* Hagata Rodrigues
* Luís Abdalla
* Maria Penha
* Miguel Bezerra
* Rafaelly Dos Anjos

---

## Visão geral das etapas

Todas as etapas usam os mesmos dados e respondem à mesma pergunta. A Etapa 1 prepara a tabela; as seguintes partem dela.

| Etapa | Tema | O que faz |
|---|---|---|
| 1 | Árvore de decisão | Prepara os dados (Ibama + INMET), cria o alvo `Multa_Alta` e treina a árvore (do zero e com scikit-learn) |
| 2 | Aprendizado linear | Regressão linear, gradiente descendente, regularização e regressão logística |
| 3 | Probabilidade | Conjunta, condicional, regra da cadeia, Bayes e independência |
| 4 | Aprendizado probabilístico | MLE, MAP, Naive Bayes (parte 1) e EM com variável oculta (parte 2) |
| 5 | Redes Bayesianas | Grafo de dependências, tabelas de probabilidade e inferência |
| 6 | Lógica fuzzy | Sistema Mamdani que usa as probabilidades da Etapa 5 e devolve a decisão final |

```
Dados (Ibama + INMET) → Etapa 1 → dados/tabela_p4.csv → Etapas 2, 3, 4, 5 → dados/probs_rede.csv → Etapa 6 → decisão final
```

---

## Dados

| Fonte | O que usamos | Ano | Data de acesso |
|---|---|---|---|
| Ibama — Portal de Dados Abertos (https://dadosabertos.ibama.gov.br) | Autos de Infração (Fiscalização) | 2021 | 29/10/2026 |
| INMET — Portal do INMET (https://portal.inmet.gov.br/dadoshistoricos) | Dados Históricos Anuais, Estações Automáticas | 2021 | 01/11/2026 |

- **Ibama** = o que aconteceu: cada linha é um auto de infração (valor da multa, bioma, tipo de pessoa, tipo de ação, data e estado).
- **INMET** = o tempo que fazia: chuva, temperatura e umidade do mesmo estado e mês do auto.
- Ficamos com os **2.639 autos de desmatamento** (infrações de flora marcadas como desmatamento, sem autos cancelados nem advertências).
- **"Multa elevada"** = acima da **mediana** (R$ 100.000). Isso deixa as duas classes quase meio a meio.
- A tabela pronta que todas as etapas usam é `dados/tabela_p4.csv`.

> Os dados originais do Ibama trazem nome e CPF/CNPJ de quem foi multado. Eles **não** estão neste
> repositório (baixe nos links acima). A `tabela_p4.csv` já vem sem esses dados pessoais.

### Colunas da `tabela_p4.csv`

| Coluna | O que é |
|---|---|
| `UF`, `MUNICIPIO` | Estado e município do auto |
| `DAT_HORA_AUTO_INFRACAO`, `ano`, `mes` | Data do auto |
| `VAL_AUTO_INFRACAO` | Valor da multa (R$) |
| `Multa_Alta` | **Alvo**: 1 = multa acima da mediana (R$ 100.000), 0 = abaixo |
| `BIOMA`, `TP_PESSOA_INFRATOR`, `TIPO_ACAO` | Atributos do auto (bioma, PF/PJ, tipo de ação) |
| `precip_mm`, `temp_media`, `umid_media`, `n_estacoes` | Clima do estado e mês (INMET) |
| `CHUVA_FAIXA`, `TEMP_FAIXA`, `UMID_FAIXA` | Clima em 3 faixas (baixa, media, alta) ou `sem_dado` |
| `EFEITO_MEIO_AMBIENTE` | Efeito da infração no meio ambiente, como informado pelo Ibama |
| `NUM_PROCESSO` | Número do processo (quando presente na tabela). **Ler como texto** (veja abaixo) |

Para ler a tabela em qualquer etapa:

```python
import pandas as pd
tabela = pd.read_csv("dados/tabela_p4.csv", encoding="utf-8-sig", dtype={"NUM_PROCESSO": str})
```

O `dtype={"NUM_PROCESSO": str}` é importante: o número tem 17 dígitos com zero na frente, e lido como número o pandas perde o zero e estraga os últimos dígitos. Se a tabela não tiver essa coluna, o parâmetro é simplesmente ignorado.

---

## Como reproduzir

### 1. Preparar o ambiente

Precisa de Python 3 (usamos 3.12 e 3.13). No Windows, se `python` não funcionar, use `py`.

```bash
git clone https://github.com/rafaellydosanjos/2o-Trab-SimbML-Incerteza_equipe_3.git
cd 2o-Trab-SimbML-Incerteza_equipe_3
pip install -r requirements.txt
```

### 2. Baixar os dados (só para refazer a Etapa 1)

As etapas seguintes partem da `dados/tabela_p4.csv`. Os dados originais só são necessários para regenerar essa tabela.

```
dados/auto_infracao_2021.csv     (Ibama)
dados/inmet/2021.zip             (INMET, sem descompactar)
```

### 3. Executar as etapas

Rode na **ordem** abaixo. Os scripts `.py` rodam a partir da **pasta principal** do repositório (a que tem o `utils.py`). Os notebooks ficam em `notebooks/`: abra o Jupyter a partir da pasta principal (`jupyter notebook`) ou abra a pasta no VS Code, escolha o kernel do Python onde instalou as bibliotecas e use **Run All**.

| Etapa | Arquivo | Como rodar | O que precisa | O que gera |
|---|---|---|---|---|
| 1 | `etapa1_p4.py` | `python etapa1_p4.py` | Os dois arquivos de `dados/` | `dados/tabela_p4.csv`, `arvore_etapa1.png` |
| 1 | `notebooks/etapa1.ipynb` | Abrir e rodar as células | `dados/tabela_p4.csv` | Árvore do zero, comparação com o scikit-learn e validação cruzada |
| 2 | `notebooks/etapa2.ipynb` | Abrir e rodar as células | `dados/tabela_p4.csv` | Regressões, curvas e tabela de resultados |
| 3 | `notebooks/etapa3_probabilidade.ipynb` | Abrir e rodar as células | `dados/tabela_p4.csv` | Tabelas de probabilidade e cálculos de Bayes |
| 4 (parte 1) | `notebooks/etapa4_parte1_nb.ipynb` | Abrir e rodar as células | `dados/tabela_p4.csv` | MLE, MAP e Naive Bayes |
| 4 (parte 2) | `notebooks/etapa4_parte2_em.ipynb` (usa `em.py`) | Abrir e rodar as células | `dados/tabela_p4.csv` | Curva de log-verossimilhança e Naive Bayes com a variável oculta |
| 5 | `notebooks/etapa5.ipynb` | Abrir e rodar as células | `dados/tabela_p4.csv` | Rede Bayesiana e `dados/probs_rede.csv` |
| 6 | `notebooks/etapa6.ipynb` (usa `notebooks/fuzzy.py`) | Abrir e rodar as células | `dados/probs_rede.csv` | Decisão fuzzy comparada com a decisão "rígida" (corte em 0,5) |

#### Etapa 1 em detalhe

1. Rode `python etapa1_p4.py`. Na primeira vez leva cerca de 1 minuto, porque lê 588 estações do INMET. Nas próximas ele reaproveita o clima já processado em `dados/inmet_uf_mes.csv` (apague esse arquivo para refazer).
2. **Feche a janela da figura** que abre no final, senão o script não termina de imprimir.
3. Confira na tela: `Todos os autos do arquivo: 9663`, `Com valor de multa válido: 2639`, `Mediana da multa (corte): R$ 100,000.00` e `Tabela salva em dados/tabela_p4.csv`.
4. O resultado esperado é cerca de **73,6% de acerto médio no teste** (5-fold), com profundidade ótima 3, dentro da faixa de 72–80% da atividade. Os valores podem variar um pouco entre computadores.
5. No `notebooks/etapa1.ipynb`, troque `USAR_EXEMPLO = True` por `False` na célula do Passo 1 e rode as células de cima para baixo.
6. Opcional: `python gerar_figura_arvore_zero.py` desenha a árvore feita do zero (3 níveis) em `arvore_do_zero_etapa1.png`.

#### Cuidados ao rodar as outras etapas

- O `notebooks/etapa1.ipynb` sobe uma pasta sozinho para achar o `utils.py` e a pasta `dados/`. Se outro notebook der erro de arquivo não encontrado, cole estas linhas no começo dele:

  ```python
  import os
  if not os.path.exists("utils.py") and os.path.exists("../utils.py"):
      os.chdir("..")
  ```
- Todas as etapas usam a mesma `dados/tabela_p4.csv`. **Não** regenere a tabela no meio do trabalho sem avisar o grupo: os resultados das outras etapas dependem da ordem das linhas.
- A Etapa 6 usa as probabilidades da Etapa 5 linha a linha, então a tabela e o `dados/probs_rede.csv` precisam ter as mesmas 2.639 linhas, na mesma ordem.

---

## Arquivos do repositório

```
.
├── README.md
├── requirements.txt
├── utils.py                      funções de apoio
├── preparar_inmet.py             clima do INMET por estado e mês
├── etapa1_p4.py                  prepara os dados e gera a tabela (Etapa 1)
├── em.py                         algoritmo EM (Etapa 4)
├── gerar_figura_arvore_zero.py   figura da árvore do zero (Etapa 1)
├── dados/
│   ├── tabela_p4.csv             tabela pronta, sem nome nem CPF/CNPJ
│   └── probs_rede.csv            probabilidades da rede (saída da Etapa 5, entrada da Etapa 6)
├── notebooks/
│   ├── etapa1.ipynb
│   ├── etapa2.ipynb
│   ├── etapa3_probabilidade.ipynb
│   ├── etapa4_parte1_nb.ipynb
│   ├── etapa4_parte2_em.ipynb
│   ├── etapa5.ipynb
│   ├── etapa6.ipynb
│   └── fuzzy.py                  sistema fuzzy (Etapa 6)
└── relatorio/                    relatório em LaTeX (veja a seção Relatório)
```

Ficam **fora do repositório**, por serem grandes ou conterem dados pessoais: `dados/auto_infracao_2021.csv`, `dados/inmet/2021.zip` e o cache `dados/inmet_uf_mes.csv`, que o `etapa1_p4.py` cria sozinho.

---

## Relatório

O relatório (formato artigo) fica na pasta `relatorio/`:

```
relatorio/
├── main.tex                                 texto do relatório (LaTeX)
├── referencias.bib                          referências
├── TRABALHOFIA_Pipeline_MeioAmbiente.pdf    relatório compilado
└── figuras/                                 figuras usadas no texto
```

As figuras em `relatorio/figuras/` vêm dos notebooks e scripts de cada etapa:

| Figura | Etapa | O que mostra |
|---|---|---|
| `arvore_etapa1.png` | 1 | Árvore de decisão do scikit-learn (divisões binárias) |
| `arvore_do_zero.png` | 1 | Árvore feita do zero (um ramo por valor, 3 níveis) |
| `curva_custo.png` | 2 | Curva de custo do gradiente descendente |
| `regressao_simples.png` | 2 | Regressão linear simples |
| `ridge_coeficientes.png` | 2 | Coeficientes do Ridge em função de λ |
| `loglik_em.png` | 4 | Log-verossimilhança do EM a cada iteração |
| `pertinencias.png` | 6 | Funções de pertinência do sistema fuzzy |
| `superficie.png` | _(confirmar etapa)_ | Superfície do modelo _(confirmar)_ |

**Para recompilar o PDF:** no Overleaf, envie a pasta `relatorio/` e compile o `main.tex`. Pela linha de comando, com TeX Live ou MiKTeX instalado:

```bash
cd relatorio
latexmk -pdf main.tex
```

Se uma figura mudar (por exemplo, depois de rodar uma etapa de novo), copie a nova imagem para `relatorio/figuras/` com o mesmo nome e compile outra vez.

---

## Solução de problemas

| Problema | O que fazer |
|---|---|
| `python` não é reconhecido | Use `py` no lugar de `python` |
| `ModuleNotFoundError` (pandas, sklearn...) | Rode `pip install -r requirements.txt` de novo. No notebook, confira se o kernel é o mesmo Python |
| `No module named 'utils'` ou `'preparar_inmet'` | Você está fora da pasta principal. Entre nela antes de rodar o script. Em um notebook, use as linhas de `os.chdir("..")` mostradas acima |
| `FileNotFoundError` em `dados/...` | Confira o nome e o lugar dos arquivos de `dados/` |
| O script da Etapa 1 parece travado no fim | Feche a janela da figura |
| `NUM_PROCESSO` com números estranhos (terminando em `.0` ou sem o zero) | Leia a tabela com `dtype={"NUM_PROCESSO": str}` |

---

## Limitações que declaramos

- O portal do Ibama traz **registros de fiscalização** (autos e multas), e não a área desmatada medida por satélite. Por isso "desmatamento" aqui quer dizer **autos de infração por desmatamento**.
- **21 autos** não têm bioma informado.
- Cerca de **6,7% dos autos** ficam sem dado de clima, principalmente em Roraima, cuja estação teve quase todos os dados faltando em 2021.
- A coluna de área (`QT_AREA`) vem quase toda vazia, então não foi usada.

## Referências

- Russell, S., & Norvig, P. *Artificial Intelligence: A Modern Approach*. 4ª ed., Pearson, 2020.
- Zadeh, L. A. Fuzzy sets. *Information and Control*, 8(3), 338–353, 1965.
- Hüllermeier, E. Fuzzy machine learning and data mining. *WIREs DMKD*, 1(4), 269–283, 2011.
- Portais de dados: https://dadosabertos.ibama.gov.br e https://portal.inmet.gov.br/dadoshistoricos.
- Bibliotecas: scikit-learn, scikit-fuzzy, pgmpy, numpy, pandas.