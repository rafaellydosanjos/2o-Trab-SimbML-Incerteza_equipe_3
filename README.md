# Documentação de Variáveis - Etapa 1 (IBAMA 2021)

## 1. Variável-Alvo Binária (y)

* **`Multa_Alta`**: Variável categórica binária que representa a gravidade financeira da infração ambiental.
  * **`1` (Multa Alta):** Valor da autuação superior à mediana exclusiva dos crimes de desmatamento/flora (> R$ 141.000,00).
  * **`0` (Multa Baixa):** Valor da autuação igual ou inferior à mediana (<= R$ 141.000,00).

---

## 2. Atributos Explicativos e Interpretáveis (X)

| Atributo | Tipo de Dado | Descrição | Exemplo de Valores |
| :--- | :--- | :--- | :--- |
| **`UF`** | Categórico | Estado (Unidade Federativa) onde ocorreu a infração ambiental. | `AM`, `PA`, `GO`, `PR` |
| **`DS_BIOMAS_ATINGIDOS`** | Categórico | Bioma diretamente afetado pela ação infracional. | `Amazônia`, `Cerrado`, `Mata Atlântica` |
| **`TP_PESSOA_INFRATOR`** | Categórico | Tipo de cadastro do autuado perante a fiscalização. | `PF` (Pessoa Física), `PJ` (Pessoa Jurídica) |
| **`QT_AREA`** | Numérico (`float`) | Extensão da área desmatada, mensurada em hectares (ha). | `0.0`, `19.0`, `26.213` |

---

## 3. Resumo do Pré-processamento e Validação
* **Base de Dados:** IBAMA - Autos de Infração (2021) filtrados por `TIPO_INFRACAO = 'Flora'`.
* **Tratamento de Dados Ausentes:** Valores nulos em `QT_AREA` foram imputados com 0; textos nulos preenchidos com categorias padrão.
* **Codificação:** Transformação das variáveis categóricas via *One-Hot Encoding* (`pd.get_dummies`).
* **Método de Validação:** Validação Cruzada Estratificada (*Stratified 5-Fold Cross-Validation*) para garantir robustez e evitar sobreajuste.

---

## Etapa 1: Árvore de Decisão Interpretável
* **Arquivo do código:** `notebooks/etapa1.ipynb`
* **Base de Dados:** `dados/auto_infracao_2021.csv`
* **Pergunta Prática:** Um auto de infração do IBAMA por desmatamento terá multa de valor elevado?
* **Semente Fixa:** `42`