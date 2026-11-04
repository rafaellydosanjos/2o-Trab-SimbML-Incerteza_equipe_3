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

# Etapas 3 e 4 (Parte 1) — Probabilidade e Naive Bayes

**Pergunta de classificação:** “Um auto de infração do IBAMA por desmatamento terá multa de valor elevado?”

## Escopo individual

- **Etapa 3:** distribuição conjunta (quatro eventos), axiomas, marginais e condicionais, regra da cadeia, Teorema de Bayes e verificação matemática de independência condicional.
- **Etapa 4, Parte 1:** MLE (Bernoulli), MAP (Beta(2,2)), Naive Bayes implementado do zero, validação cruzada com cinco folds e teste externo.
- O **EM e seu gráfico** são responsabilidade da Etapa 4, Parte 2 (outro integrante) e **não** estão incluídos neste código.

## Arquivos

- `trabalho_etapas_3_4_v6_1_bayes_corrigido.py`: implementação central, utilizada pelos notebooks.
- `etapa3_probabilidade_v6_1.ipynb`: demonstração da Etapa 3.
- `etapa4_parte1_nb_v6_1.ipynb`: demonstração da Etapa 4 (Parte 1).
- `auto_infracao_2021.csv`: arquivo de dados do IBAMA utilizado no trabalho (**disponibilizado separadamente pela equipe**, caso não esteja versionado no repositório).

## Dependências

Python 3.10+ com `numpy`, `pandas` e `scikit-learn`.

```bash
python -m pip install numpy pandas scikit-learn
```

## Como reproduzir

1. Coloque o CSV `auto_infracao_2021.csv` **na mesma pasta** do arquivo Python.
2. Execute no terminal:

   ```bash
   python trabalho_etapas_3_4_v6_1_bayes_corrigido.py
   ```

   Ou, no Google Colab, envie os dois arquivos para `/content` e execute:

   ```python
   %run /content/trabalho_etapas_3_4_v6_1_bayes_corrigido.py
   ```

3. Para o formato de notebooks exigido pelo trabalho, execute os notebooks fornecidos, mantendo o arquivo `.py` e o CSV na mesma pasta.

A saída principal apresenta apenas os cálculos essenciais da Etapa 3, as estimativas MLE/MAP e as métricas do Naive Bayes. Não há geração de gráfico nesta parte individual.

## Preparação e avaliação

- Os autos são selecionados por `TIPO_INFRACAO=Flora` e por expressões da descrição compatíveis com desmatamento; **esse filtro textual é heurístico**.
- Por padrão, `Multa_Alta` assume 1 se a multa ultrapassar a **mediana do respectivo conjunto de treino** e 0 caso contrário.
- Se a Etapa 1 fixou um limiar em reais, edite `LIMIAR_ETAPA1_CONFIRMADO` no script, substituindo `None` pelo valor adotado; o limite fixo passa a ser usado no teste externo e nos folds.
- A separação treino/teste é feita por processo (`NUM_PROCESSO`, quando disponível) para reduzir vazamento de informações. O conjunto externo de teste não participa da validação cruzada.
- A comparação de independência condicional usa a **fatoração matemática das probabilidades observadas**; não emprega o teste exato de Fisher nem conclui causalidade.

## Resultados de referência (configuração padrão, CSV fornecido)

- Treino: **1.529** registros; teste: **660**; **41** atributos binários.
- Mediana do treino: **R$ 145.000,00**.
- Naive Bayes: **85,91%** de acurácia no teste, **85,84%** de F1.
- Validação cruzada (5 folds): **83,71%** de acurácia média (desvio-padrão **2,29 p.p.**).

## Integração com a equipe — verificar antes do relatório final

Confirmar com o responsável pela Etapa 1 que a **definição numérica de “multa elevada”** e os **critérios de seleção de autos de desmatamento** são os mesmos usados no restante do projeto. A pergunta, por si só, não especifica esse limiar. O projeto completo também contempla INMET e as demais etapas, que não fazem parte deste módulo individual.

## Interpretação causal versus diagnóstica (correção V6.1)

No contexto dos autos, **P(multa alta | Amazônia)** acompanha uma direção causal **plausível** (características do auto podem influenciar o valor). **P(Amazônia | multa alta)** percorre a direção diagnóstica. O Naive Bayes adota uma **fatoração generativa** P(atributos | classe) para calcular P(classe | atributos); isso **não** atribui causalidade à classe. Por serem dados observacionais, as probabilidades mostram associação, não demonstram que o bioma cause multas maiores.

Na execução de referência, **P(Amazônia | multa alta) = 0,8581** e **P(multa alta | Amazônia) = 0,6838**, calculadas apenas nos dados de treinamento. Consulte `texto_relatorio_bayes_v6_1.md` para o parágrafo sugerido ao relatório.

**Rastreabilidade:** a V6.1 modifica apenas a apresentação e interpretação de Bayes, sem alterar o treinamento, a base, as métricas ou as outras etapas. Diferenças em `alteracoes_v6_para_v6_1_bayes.patch`.
