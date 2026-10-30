# Trabalho 2: Do Simbólico à Incerteza: Pipeline

**Disciplina:** Fundamentos de Inteligência Artificial    
**Professor:** Edjard Mota    
**Repositório Oficial:** 2o-Trab-SimbML-Incerteza_equipe_3  

## Registo Acadêmico
- Adrya Vieira
- Hagata Rodrigues
- Luís Abdalla
- Maria Penha
- Miguel Bezerra
- Rafaelly Dos Anjos

---

# Documentação de Variáveis - Etapa 1 (IBAMA 2021)

## 1. Variável-Alvo Binária ($y$)

- **`Multa_Alta`**: Variável categórica binária que representa a gravidade financeira da infração ambiental.
  * **`1` (Multa Alta):** Valor da autuação superior à mediana do conjunto de dados ($\text{> R\$ 96.500,00}$).
  * **`0` (Multa Baixa):** Valor da autuação igual ou inferior à mediana ($\text{\le R\$ 96.500,00}$).

---

## 2. Atributos Explicativos e Interpretáveis ($X$)

| Atributo | Tipo de Dado | Descrição | Exemplo de Valores |
| :--- | :--- | :--- | :--- |
| **`UF`** | Categórico | Estado (Unidade Federativa) onde ocorreu a infração ambiental. | `AM`, `PA`, `GO`, `PR` |
| **`DS_BIOMAS_ATINGIDOS`** | Categórico | Bioma diretamente afetado pela ação infracional. | `Amazônia`, `Cerrado`, `Mata Atlântica` |
| **`TP_PESSOA_INFRATOR`** | Categórico | Tipo de cadastro do autuado perante a fiscalização. | `PF` (Pessoa Física), `PJ` (Pessoa Jurídica) |
| **`QT_AREA`** | Numérico (`float`) | Extensão da área afetada ou desmatada, mensurada em hectares ($\text{ha}$). | `0.0`, `19.0`, `26.213` |

---

## 3. Resumo do Pré-processamento
- **Base de Dados:** IBAMA - Autos de Infração (2021)
- **Total de Registros Limpos:** $9.620$ amostras
- **Divisão de Dados:** $70\%$ Treinamento / $30\%$ Teste ($\text{random\_state} = 42$)
- **Tratamento de Dados Ausentes:** Valores nulos em `QT_AREA` foram imputados com $0$; textos nulos foram preenchidos com categorias padrão.
- **Codificação:** Transformação das variáveis categóricas via *One-Hot Encoding* (`pd.get_dummies`).
---

## Etapa 1: Árvore de Decisão Interpretável
- **Arquivo do código:** `notebooks/etapa1.ipynb`
- **Base de Dados:** `dados/auto_infracao_2021.csv`
- **Pergunta Prática:** Esta infração ambiental resultará em uma multa de valor elevado?
- **Semente Fixa:** `42`
- **Desempenho:** Acurácia de 66,56% e Recall de 78% para multas graves.
