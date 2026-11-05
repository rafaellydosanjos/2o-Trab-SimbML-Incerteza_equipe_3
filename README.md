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

## As 6 etapas em palavras simples

### Etapa 1 — Árvore de decisão (aprendizado simbólico)
**A ideia:** uma sequência de perguntas, como no jogo dos 20 perguntas ("o auto foi na Amazônia? o infrator é pessoa física?"). O computador aprende sozinho quais perguntas fazer e em que ordem, usando a **entropia** (uma medida de "bagunça") e o **ganho de informação** (quanto uma pergunta organiza a bagunça).
**No nosso trabalho:** a árvore prevê se a multa será alta, com 6 atributos: bioma, tipo de pessoa, tipo de ação, chuva, temperatura e umidade. Fizemos a árvore do zero, comparamos com a do scikit-learn e validamos com 5-fold (cerca de 73% de acerto no teste).
**Também mostramos uma limitação da lógica:** um exemplo novo pode derrubar uma regra que parecia certa (monotonicidade).

### Etapa 2 — Aprendizado linear
**A ideia:** encontrar a reta (ou o plano) que melhor liga os atributos a um número. É a **regressão linear**, ajustada pelo **gradiente descendente** (dar passinhos para errar cada vez menos). A **regularização** (Ridge/L2) evita exageros. A **regressão logística** transforma o resultado em uma probabilidade de "sim" ou "não".
**No nosso trabalho:** relacionar os atributos ao valor da multa e classificar multa alta ou baixa.

### Etapa 3 — Probabilidade
**A ideia:** a base de tudo que vem depois. Probabilidade **conjunta** (duas coisas juntas), **marginal**, **condicional** ("a chance de A, sabendo B"), **regra da cadeia** e o **Teorema de Bayes**, que permite inverter uma condicional. Também discute por que a lógica clássica não consegue dizer "70% de chance".
**No nosso trabalho:** cálculos com variáveis do próprio domínio, como a chance de multa alta sabendo o bioma.

### Etapa 4 — Aprendizado probabilístico e EM
**A ideia:** aprender probabilidades a partir dos dados. **MLE** escolhe o valor que deixa os dados mais prováveis; **MAP** faz o mesmo partindo de um palpite inicial (o *prior*). O **Naive Bayes** classifica multiplicando probabilidades, supondo que os atributos são independentes dada a classe. O algoritmo **EM** lida com uma **variável oculta**, que não vemos nos dados: ele alterna entre estimar e atualizar até estabilizar.
**No nosso trabalho:** Naive Bayes para prever multa alta e comparação com e sem variável oculta (`em.py`).

### Etapa 5 — Redes Bayesianas
**A ideia:** um desenho (grafo) em que cada seta diz "esta variável influencia aquela", com tabelas de probabilidade (CPT). Permite perguntar: *"sabendo o que já sei, qual a probabilidade de multa alta?"*. Também mostra quando duas variáveis são independentes (**d-separação**) e economiza números em relação à tabela completa.
**No nosso trabalho:** uma rede com 4 a 6 variáveis do domínio (por exemplo bioma, tipo de ação e multa), com inferência feita do zero e comparada com a biblioteca `pgmpy`.

### Etapa 6 — Lógica fuzzy
**A ideia:** em vez de "sim ou não", trabalha com **graus** de 0 a 1 e com palavras ("risco baixo", "risco alto"). As regras ficam legíveis, como "SE o risco é alto ENTÃO a urgência é alta" (sistema **Mamdani**).
**No nosso trabalho:** as probabilidades da Etapa 5 entram no sistema fuzzy, que devolve uma **decisão suave e explicável**, comparada com a decisão "rígida" (corte em 0,5) (`fuzzy.py`).

## Como as etapas se conectam

```
Dados oficiais (Ibama + INMET)
        |
   Etapa 1: Árvore  ->  Etapa 2: Regressão  ->  Etapa 3: Probabilidade
                                                        |
                              Etapa 4: Naive Bayes + EM  <-
                                                        |
                              Etapa 5: Rede Bayesiana    <-
                                                        |
                              Etapa 6: Sistema Fuzzy     <-
                                                        |
                       Decisão final interpretável
```
As decisões da Etapa 1 e as probabilidades da Etapa 4 são reaproveitadas nas Etapas 5 e 6.

## Quem fez o quê

| Membro | Etapa principal | Papel extra | Nome |
|---|---|---|---|
| 1 | Etapa 1 — Árvore de decisão | Dados e GitHub | _(preencher)_ |
| 2 | Etapa 2 — Regressão e logística | Slides da apresentação | _(preencher)_ |
| 3 | Etapa 3 — Probabilidade | Relatório em LaTeX | _(preencher)_ |
| 4 | Etapa 4 — Naive Bayes e EM | — | _(preencher)_ |
| 5 | Etapa 5 — Rede Bayesiana | — | _(preencher)_ |
| 6 | Etapa 6 — Fuzzy | Integração final do pipeline | _(preencher)_ |

## Como reproduzir

1. Instale as bibliotecas:
   ```
   pip install -r requirements.txt
   ```
2. Baixe os dados e coloque assim:
   ```
   dados/auto_infracao_2021.csv     (Ibama)
   dados/inmet/2021.zip             (INMET, sem descompactar)
   ```
3. Gere a tabela e a árvore da Etapa 1:
   ```
   python etapa1_p4.py
   ```
   Isso cria `dados/tabela_p4.csv` (usada por todas as etapas) e `arvore_etapa1.png`.
4. Abra os notebooks na ordem (`jupyter notebook`): `etapa1.ipynb` → … → `etapa6.ipynb`.

Todos os códigos usam semente fixa (`np.random.seed(42)`), então os resultados se repetem.

## Arquivos do repositório

| Arquivo | O que é |
|---|---|
| `etapa1.ipynb` … `etapa6.ipynb` | Um notebook por etapa do pipeline |
| `etapa1_p4.py` | Prepara os dados (Ibama + INMET), cria o alvo `Multa_Alta` e treina a árvore |
| `preparar_inmet.py` | Lê o ZIP do INMET e resume o clima por estado e mês |
| `utils.py` | Funções de apoio compartilhadas |
| `em.py` | Algoritmo EM (Etapa 4) |
| `fuzzy.py` | Sistema fuzzy (Etapa 6) |
| `trabalho_etapas_3_4_bayes.py` | Código das Etapas 3 e 4 (probabilidade e Naive Bayes) |
| `dados/tabela_p4.csv` | Tabela pronta, sem dados pessoais |
| `arvore_etapa1.png` | Figura da árvore da Etapa 1 |
| `relatorio/` | Relatório em LaTeX |

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
