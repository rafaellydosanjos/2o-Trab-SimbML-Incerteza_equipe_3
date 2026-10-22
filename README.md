# Pipeline Completo de IA: Do Simbólico à Incerteza

## Visão Geral
Este repositório contém a implementação do pipeline integrado de aprendizado desenvolvido para a disciplina de Fundamentos de Inteligência Artificial da Universidade Federal do Amazonas (UFAM). O sistema percorre as seis etapas do pipeline de IA, desde o aprendizado simbólico até à lógica fuzzy, aplicado a microdados oficiais[cite: 3, 9].

## Estrutura de Ficheiros
O projeto está estruturado de forma modular para garantir a reprodutibilidade, conforme exigido nas diretrizes do trabalho[cite: 4]:
- `etapa1.ipynb` a `etapa6.ipynb`: Notebooks com a execução isolada de cada etapa[cite: 4].
- `utils.py`, `em.py`, `fuzzy.py`: Scripts auxiliares contendo as classes e funções base[cite: 4].
- `relatorio_final.pdf`: Relatório do projeto em formato de artigo (escrito em LaTeX) detalhando as decisões e resultados[cite: 4].

## Pré-requisitos
Para executar o ambiente de forma adequada, é necessário ter o Python instalado com as seguintes bibliotecas[cite: 10]:
- `numpy`
- `pandas`
- `scikit-learn`
- `scikit-fuzzy`
- `pgmpy`

## Instruções de Reprodução
1. Clone este repositório localmente: `git clone https://github.com/rafaellydosanjos/FIA_RACIOCINIO_PROB`
2. Instale as dependências listadas na secção anterior.
3. Execute os notebooks sequencialmente, do 1 ao 6.
> **Nota de Reprodutibilidade:** Todos os algoritmos que envolvem parâmetros aleatórios utilizam uma seed fixa (`np.random.seed(42)`) para garantir que os resultados sejam rigorosamente iguais em qualquer execução[cite: 9].

## Dados Utilizados
Os dados empregues neste estudo pertencem ao domínio de **[Inserir o domínio escolhido: Saúde Pública / Educação / Segurança Pública / Meio Ambiente]**[cite: 3].
- **Fonte de Origem:** **[Inserir link do portal oficial, ex: dadosabertos.saude.gov.br]**[cite: 10].
- **Data de Acesso:** **[Inserir a data da recolha e consulta dos dados]**[cite: 10].

## Equipa
- Rafaelly Barreto dos Anjos
- [Nome do segundo integrante, se aplicável]
- [Nome do terceiro integrante, se aplicável]
