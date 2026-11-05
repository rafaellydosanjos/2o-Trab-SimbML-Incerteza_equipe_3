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

## Fontes dos dados

| Fonte | Conjunto usado | Ano | Data de acesso |
|---|---|---|---|
| Ibama — Portal de Dados Abertos (https://dadosabertos.ibama.gov.br) | Autos de Infração (Fiscalização) | 2021 | 29/10/2026 |
| INMET — Portal do INMET (https://portal.inmet.gov.br/dadoshistoricos) | Dados Históricos Anuais, Estações Automáticas | 2021 | 01/11/2026 |

> Os dados originais do Ibama trazem nome e CPF/CNPJ de quem foi multado. **Não** estão neste repositório;
> baixe-os nos links acima. A tabela `dados/tabela_p4.csv` já vem sem esses dados pessoais.

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
   Isso cria `dados/tabela_p4.csv` (usada por todas as etapas) e `arvore_etapa1.png` (figura da árvore).
4. Abra os notebooks na ordem (`jupyter notebook`): `etapa1.ipynb` → `etapa2.ipynb` → … → `etapa6.ipynb`.

Todos os códigos usam `np.random.seed(42)` (definido em `utils.py`), então os resultados são reprodutíveis.

## Arquivos

| Arquivo | O que é |
|---|---|
| `etapa1_p4.py` | Prepara os dados (Ibama + INMET), cria o alvo `Multa_Alta` e treina a árvore do scikit-learn |
| `preparar_inmet.py` | Lê o ZIP do INMET e resume o clima por UF e mês |
| `utils.py` | Funções de apoio (semente, treino/teste, faixas) |
| `etapa1.ipynb` … `etapa6.ipynb` | Um notebook por etapa do pipeline |
| `em.py`, `fuzzy.py` | Algoritmo EM (Etapa 4) e sistema fuzzy (Etapa 6) |
| `relatorio/` | Relatório em LaTeX |
