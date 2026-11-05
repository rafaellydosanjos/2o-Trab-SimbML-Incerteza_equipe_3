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

---

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
