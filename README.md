# Estudos de Ciência de Dados para Análise Técnica de Ativos da Bolsa
`Em construção`

Este projeto tem como objetivo explorar técnicas de ciência de dados e machine learning aplicadas à análise técnica de ativos negociados na bolsa de valores. Utiliza bibliotecas populares para importar, processar, visualizar e prever cotações de ativos financeiros.

## Objetivos

- Importar dados históricos de ativos financeiros.
- Visualizar séries temporais e gráficos de candlestick.
- Aplicar modelos de machine learning para prever cotações futuras.
- Comparar previsões com valores reais para avaliar desempenho.

## Entrada de Dados
1. Os Tickers são importados de fontes mistas, parte utilizando a biblioteca Request e parte baixando o csv da B3.
2. Após, os dados históricos do ativo selecionado são importados diretamente do Yahoo Finance e processados em formato de tabela utilizando bibliotecas como `pandas` e `yfinance`.

## Visualização dos Dados

Os dados são plotados em gráficos de candlestick para facilitar a análise visual dos movimentos de preço. Para isso, são utilizadas bibliotecas como `streamlit` e `plotly`.

## Previsão e Avaliação

O projeto tenta prever a cotação futura baseada em dados passados, utilizando modelos de machine learning como regressão linear, redes neurais ou outros algoritmos disponíveis em `scikit-learn`. As previsões são comparadas com as cotações reais para avaliar a acurácia dos modelos.

## Estrutura do Projeto


Workspace
(rerun without)
Collecting workspace information
```
├── Análise_técnica_de_ações.ipynb # Notebook de origem com análises e experimentos 
├── pyproject.toml # Configuração do Poetry e dependências 
├── poetry.lock # Lockfile de dependências 
├── .python-version # Versão do Python utilizada 
└── README.md # Este arquivo
```

## Instalação de Dependências

Clone o repositório:

```bash
git clone git@github.com:Mateus-cpa/analise-tecnica-bolsa.git
cd analise-tecnica-bolsa
```

Configure o ambiente virtual com as dependências necessárias:

```bash
pyenv install 3.10.11             # Instale o Python 3.10.11 se ainda não tiver
pyenv local 3.10.11               # Defina a versão local do projeto
python -m venv .venv              # Crie o ambiente virtual
source .venv/Scripts/activate     # No Windows
poetry init
poetry shell
```

Iniciar aplicativo
```bash
poetry install
poetry run streamlit run src/main.py
```


## Principais Bibliotecas Utilizadas
- pandas
- yfinance
- matplotlib
- mplfinance
- scikit-learn

### Contribuição

Sinta-se à vontade para abrir issues ou enviar pull requests com melhorias, correções ou novas análises.

## Licença
Este projeto está sob a licença MIT.


# Inspirações
- https://www.infomoney.com.br/guias/analise-tecnica/
- Video previsão 10 dias: # https://www.youtube.com/watch?v=CvfAx3_nGME&list=PL1woXE9p74ASlH4i2QQytmASjle8Bt8An&index=7